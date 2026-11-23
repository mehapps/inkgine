"""CPU, tokenizer only (~1 min): the OpenAI API features on the engine side, with a stub engine (no weights).
1. logprobs: entries == log_softmax of the processed logits, independent of temperature; top_logprobs sorted; entries
   ride on content deltas and their bytes rebuild the content.
2. logprobs with speculation: greedy prompt-lookup spec through Engine.decode_step (verify rows + commit) gives the
   same tokens and the same logprob entries as plain decoding, with drafts actually accepted; also with logit
   processors (verify rows processed per draft prefix) and min_p (Spec.settle).
3. logit processors: logit_bias, presence / frequency / repetition penalties, min_tokens, min_p, stop_token_ids.
4. forced tool_choice (grammar.ToolCall): random logits always yield a valid call of an allowed tool (thinking on/off,
   required / named, parallel or not); spec_runner's draft masks + rollback across the name -> json switch.
5. prefix cache across an agent loop: the reply is re-sent as canonical tool-call JSON (differs from the generated
   tokens); turn 2 still restores turn 1's whole prompt (GPU-parked and from disk), nothing is prefilled cold.
   5c: mid-prompt checkpoints (INK_CKPT_EVERY): an edit inside a long prompt restores from the deepest checkpoint
   before it, bit-identical to cold (also with images, pages from disk, failed writes).
6. server: n > 1 (choices, seeds, usage), logprobs fields, param validation, streaming indexes.
Run: cd engine && PATH=$PWD/.venv/bin:$PATH python tests/test_api.py"""
import json
import math
import os
import shutil
import sys
import tempfile
from types import SimpleNamespace

import torch
from tokenizers import Tokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import chat
import grammar
import kvpool
import spec_runner
from engine import MODEL_DIR, Engine, Request
from spec import Lookup

tok = Tokenizer.from_file(os.path.join(MODEL_DIR, 'tokenizer.json'))
enc = lambda s: tok.encode(s, add_special_tokens=False).ids
V = grammar.VOCAB
CPU = torch.device('cpu')


def stub(pool=None):
    E = Engine.__new__(Engine)
    E.tok, E.max_ctx, E.vision, E.spec, E.parked, E.pool, E.spec_live = tok, 1 << 16, None, 'off', [], pool, set()
    E.model = SimpleNamespace(dev=CPU, new_seq=lambda tbl=None: SimpleNamespace(
        kv=tbl if tbl is not None else pool.new_table(), conv=[], ring=[], pos=0))
    return E


def mkreq(E, ids=(1, 2, 3), **kw):
    out = []
    return Request(E, list(ids), [], emit=lambda k, v: out.append(v), **kw), out


def noise(seed, scale=3.0):
    lg = torch.randn(V, generator=torch.Generator().manual_seed(seed)) * scale
    lg[chat.USER:] = float('-inf')  # (plain text only, so every token is a content token)
    return lg


E = stub()

# 1. logprobs ------------------------------------------------------------------------------------------------------
for T in (0.0, 0.7):
    r, out = mkreq(E, max_tokens=50, temperature=T, top_p=0.9, prefill_text=True, logprobs=3, seed=1)
    lgs = []
    for s in range(12):
        lg = noise(s)[None].clone()
        lgs.append(lg[0].clone())
        E._take([r], lg)
    ents = [e for d in out if len(d) > 2 for e in d[2]]
    assert len(ents) == len(r.out) == 12, (len(ents), r.out)
    for lg, t, e in zip(lgs, r.out, ents):
        ref = torch.log_softmax(lg, -1)
        assert abs(e['logprob'] - ref[t].item()) < 1e-5, (e, ref[t])
        tops = [x['logprob'] for x in e['top_logprobs']]
        assert len(tops) == 3 and tops == sorted(tops, reverse=True) and abs(tops[0] - ref.max().item()) < 1e-5
    assert bytes(b for e in ents for b in e['bytes']).decode('utf-8', 'replace') == ''.join(d[1] for d in out)
print('1 logprobs == log_softmax of the logits (temperature 0 and 0.7), top 3 sorted, bytes rebuild the text')

# 2. logprobs with speculation (prompt lookup, greedy, forced drafting) == plain ------------------------------------
C = enc(' one two three four five six seven eight nine ten')


def cyc_logits(h):
    lg = noise(hash(tuple(h[-3:])) % (2 ** 31), 1.0)
    if h[-1] in C and hash(tuple(h[-2:])) % 4:  # mostly continue the cycle (lookup drafts get accepted), sometimes not
        lg[C[(C.index(h[-1]) + 1) % len(C)]] += 8
    elif h[-1] not in C:  # back into the cycle after a detour
        lg[C[0]] += 8
    return lg


class FakeBD:
    def __init__(self, hist):
        self.hist, self.pend = list(hist), None

    def step(self, rows):
        toks = [t for _, _, t in rows]
        lg = torch.stack([cyc_logits(self.hist + toks[:i + 1]) for i in range(len(toks))])
        if len(rows) == 1:
            self.hist.append(toks[0])
        else:
            self.pend = toks
        return lg

    def commit(self, acc):
        self.hist += self.pend[:acc[0]]
        self.pend = None


def run_spec(on, n=60, kmax=7, **kw):
    prompt = C * 3
    E.bd = FakeBD(prompt)
    kw = dict(dict(temperature=0, logprobs=2), **kw)
    r, out = mkreq(E, prompt, max_tokens=n, prefill_text=True, **kw)
    spec_runner.VERIFY_MS = (1.0,) * 8  # always draft
    if on:  # (as Engine._spec builds it)
        r.sp = spec_runner.Spec(None, E.bd, None, lookup=Lookup(r.prompt), temperature=r.temperature, top_p=r.top_p,
                                top_k=r.top_k, gen=r.gen, stop=(chat.EOS,), min_p=r.min_p, seed=r.seed, kmax=kmax)
        r.sp.attach(0, len(prompt))
    r.slot, r.pos = 0, len(prompt)
    E._take([r], cyc_logits(prompt)[None])
    while not r.done:
        E.decode_step([r])
    return r, [e for d in out if len(d) > 2 for e in d[2]]


(rp, ep), (rs, es) = run_spec(False), run_spec(True)
assert rp.out == rs.out and len(es) == len(rs.out) == 60, (rp.out, rs.out)
assert all(a['token'] == b['token'] and abs(a['logprob'] - b['logprob']) < 1e-5 and
           [x['token'] for x in a['top_logprobs']] == [x['token'] for x in b['top_logprobs']] for a, b in zip(ep, es))
assert rs.sp.stats['accepted'] > 10, rs.sp.stats
print(f'2 spec logprobs == plain ({rs.sp.stats["accepted"]} drafts accepted in {rs.sp.stats["rounds"]} rounds, '
      f'{len(es)} entries)')

# 2b. speculation with logit processors (penalties, logit_bias, min_tokens): verify row i is processed for the history
# out + d_1..d_i; greedy spec == plain (tokens + logprobs), with drafts accepted
PROC = dict(presence_penalty=0.4, frequency_penalty=0.3, repetition_penalty=1.2, logit_bias={C[3]: -2.0, C[5]: 1.5},
            min_tokens=20)
(rp, ep), (rs, es) = run_spec(False, **PROC), run_spec(True, **PROC)
assert rp.out == rs.out and len(es) == len(ep), (rp.out, rs.out)
assert all(a['token'] == b['token'] and abs(a['logprob'] - b['logprob']) < 1e-4 for a, b in zip(ep, es))
assert rs.sp.stats['accepted'] > 5 and rp.out != run_spec(False)[0].out, rs.sp.stats  # (the processors changed it)
# _process_rows: row i == _process on a copy whose output is out + d[:i]; the request's counts untouched
rq, _ = mkreq(E, max_tokens=99, **PROC)
rq.out = [C[0], C[1], C[1]]
E._process(rq, torch.zeros(V))                 # counts built for 3 tokens
cnt0, n0 = rq.cnt.clone(), rq.ncnt
d = [C[1], C[2], C[2], 7]
rows = torch.stack([noise(40 + i) for i in range(len(d) + 1)])
got = rows.clone()
E._process_rows(rq, got, d)
assert rq.out == [C[0], C[1], C[1]] and torch.equal(rq.cnt, cnt0) and rq.ncnt == n0
for i in range(len(d) + 1):
    ref_r, _ = mkreq(E, max_tokens=99, **PROC)
    ref_r.out = rq.out + d[:i]
    x = rows[i].clone()
    E._process(ref_r, x)
    assert torch.equal(x, got[i]), i
# sampled + min_p: settle filters the target rows (l >= l_max + T ln(min_p)) before accept: nothing below it is drawn
sp = spec_runner.Spec(None, None, 0, lookup=Lookup([]), temperature=1.0, gen=torch.Generator().manual_seed(4), min_p=0.5)
sp.bd = SimpleNamespace(commit=lambda acc: None)
lg = torch.full((1, V), -30.0)
lg[0, [10, 11, 12]] = torch.tensor([2.0, 2.0 + math.log(0.6), 2.0 + math.log(0.4)])
seen_t = {sp.settle(lg, 0, [[], None, None, None])[0] for _ in range(300)}
assert seen_t == {10, 11}, seen_t
# sampled spec with processors + min_p runs through decode_step (drafts accepted, min_tokens honoured)
r07, _ = run_spec(True, temperature=0.8, seed=3, logprobs=None, min_p=0.02, **PROC)
assert len(r07.out) == 60 and chat.EOS not in r07.out[:20] and r07.sp.stats['accepted'] > 0, r07.sp.stats
print(f'2b spec with penalties / logit_bias / min_tokens == plain ({rs.sp.stats["accepted"]} drafts accepted), '
      f'verify rows processed per draft prefix, min_p in settle')

# 2c. a drafter failure (window g 11-23: graph capture assert -> 500) never fails the request: spec off from there on,
# the request finishes plain with the same greedy output
calls = [0]
real_propose = spec_runner.Spec.propose


def flaky_propose(self, *a, **k):
    calls[0] += 1
    if calls[0] == 3:
        raise RuntimeError('it->second->use_count > 0 INTERNAL ASSERT FAILED (injected)')
    return real_propose(self, *a, **k)


spec_runner.Spec.propose = flaky_propose
try:
    E.spec_fails = 0
    (rp, ep), (rs, es) = run_spec(False), run_spec(True)
finally:
    spec_runner.Spec.propose = real_propose
assert rs.out == rp.out and rs.sp is None and E.spec_fails == 1 and calls[0] == 3, (E.spec_fails, calls)
print('2c drafter failure mid-request -> spec off for it, output == plain, no error')

# 2d. an error finishing one request's part of a step (settle, processors, feed) fails that request only (11-23 review
# 5): r.err set, decode_step doesn't raise, the decoder has no pending verify rows; co-batched plain rows go on
real_settle = spec_runner.Spec.settle
calls[0] = 0


def bad_settle(self, *a, **k):
    calls[0] += 1
    if calls[0] == 2:
        raise RuntimeError('grammar rejected token 7 (injected)')
    return real_settle(self, *a, **k)


spec_runner.Spec.settle = bad_settle
try:
    r_bad, _ = run_spec(True, n=40)
finally:
    spec_runner.Spec.settle = real_settle
assert isinstance(r_bad.err, RuntimeError) and r_bad.done, r_bad.err
ra, _ = mkreq(E, max_tokens=5, temperature=0)
rb, _ = mkreq(E, max_tokens=5, temperature=0)
real_feed = E._feed
E._feed = lambda r, t, li=None, last=True: (_ for _ in ()).throw(ValueError('parser (injected)')) if r is ra \
    else real_feed(r, t, li, last)
try:
    E._take([ra, rb], torch.stack([noise(1), noise(2)]))
finally:
    E._feed = real_feed
assert isinstance(ra.err, ValueError) and ra.done and rb.err is None and len(rb.out) == 1, (ra.err, rb.err, rb.out)
print('2d an error in one request\'s settle / feed fails only that request (r.err), the step and its batch go on')

# 2e. seeded sampling is reproducible whatever the decode schedule (window g2 11-23: "seed 42 twice" differed with spec
# on: the drafter's noise, accept_dev's uniforms and plain steps all drew from one generator, and how many draws a round
# took depended on its width, which the controller picks from measured timings). Each sampled token now uses the noise
# of its position: plain, spec with any width schedule (kmax 1 / 3 / 7, top-p, top-k, min_p, processors) == the same
# tokens for one seed; another seed differs; drafts still get accepted.
SAMP = dict(temperature=0.3, logprobs=None)  # (the fake target mostly continues its cycle at this T)
for extra in (dict(top_p=0.95), dict(top_k=40, min_p=0.02), dict(top_p=0.9, **PROC)):
    ref = run_spec(False, seed=42, **SAMP, **extra)[0].out
    runs = [run_spec(True, kmax=k, seed=42, **SAMP, **extra)[0] for k in (1, 3, 7)]
    assert all(x.out == ref for x in runs), (extra, ref, [x.out for x in runs])
    assert sum(x.sp.stats['accepted'] for x in runs) > 10, [x.sp.stats for x in runs]
    assert run_spec(True, seed=43, **SAMP, **extra)[0].out != ref
print('2e seeded sampling: plain == spec at widths 1/3/7 (top-p, top-k + min_p, processors), seed 43 differs')

# 3. logit processors -----------------------------------------------------------------------------------------------
r, _ = mkreq(E, max_tokens=50, temperature=0, logit_bias={42: 100.0})
E._take([r], noise(0)[None].clone())
assert r.out == [42]
r, _ = mkreq(E, max_tokens=50, presence_penalty=0.5, frequency_penalty=1.0)
r.out = [5, 5, 7]
row = torch.zeros(V)
row[9] = 2.0
E._process(r, row)
assert row[5] == -2.5 and row[7] == -1.5 and row[9] == 2.0 and row[11] == 0, row[[5, 7, 9]]
r, _ = mkreq(E, max_tokens=50, repetition_penalty=2.0)
r.out = [5, 7]
row = torch.zeros(V)
row[5], row[7], row[9] = 4.0, -4.0, 4.0
E._process(r, row)
assert (row[5], row[7], row[9]) == (2.0, -8.0, 4.0)
r, _ = mkreq(E, max_tokens=50, temperature=0, min_tokens=3, prefill_text=True)
for s in range(4):
    lg = noise(s)[None].clone()
    lg[0, chat.EOS] = lg[0, chat.END] = 50.0
    E._take([r], lg)
assert r.out[:3] != [chat.EOS] * 3 and chat.EOS not in r.out[:3] and r.out[3] in (chat.EOS, chat.END) and r.done
r, _ = mkreq(E, max_tokens=500, temperature=1.0, min_p=0.5, seed=3)
lg = torch.full((V,), -30.0)
lg[[10, 11, 12]] = torch.tensor([2.0, 2.0 + math.log(0.6), 2.0 + math.log(0.4)])  # p/pmax 1, 0.6, 0.4
for _ in range(300):
    E._take([r], lg[None].clone())
assert set(r.out) == {10, 11}, set(r.out)
r, out = mkreq(E, max_tokens=50, temperature=0, prefill_text=True, stop_ids=[77])
lg = noise(1)[None].clone()
lg[0, 77] = 99
E._take([r], lg)
assert r.done and r.stats['finish_reason'] == 'stop' and out == []
assert E._spec(mkreq(E, presence_penalty=1.0)[0]) is None
print('3 logit_bias, presence/frequency/repetition penalties, min_tokens, min_p, stop_token_ids ok')

# 4. forced tool_choice ---------------------------------------------------------------------------------------------
TOOLS = [{'type': 'function', 'function': {'name': 'get_weather', 'parameters': {
    'type': 'object', 'properties': {'city': {'type': 'string'}, 'days': {'type': 'integer'}}, 'required': ['city'],
    'additionalProperties': False}}},
    {'type': 'function', 'function': {'name': 'get_time', 'parameters': {}}}]


def favor(lg, ids, think):
    for s in ('"', '}', '",', '1'):
        lg[enc(s)[0]] += 6
    if think and len(ids) == 1:
        lg[chat.THINK] += 100
    if think and len(ids) % 20 == 0:
        lg[chat.END] += 100
    lg[chat.TEXT] += 50  # the model "wants" to answer in text: must be overridden
    lg[chat.EOS] += 20


def tool_run(choice, think, parallel, seed):
    c = grammar.ToolCall(grammar.compile_tools(tok, TOOLS, choice, think, parallel))
    p, ids, g = chat.Parser(tok), [], torch.Generator().manual_seed(seed)
    for _ in range(400):
        lg = torch.randn(V, generator=g) * 3
        favor(lg, ids, think)
        c.mask(lg)
        t = int(torch.softmax(lg, -1).multinomial(1, generator=g))
        c.accept(t)
        ids.append(t)
        p.feed(t)
        if c.done or c.state == 'free':
            return c, p, ids
    raise AssertionError('no call in 400 tokens')


for seed in range(12):
    choice = 'required' if seed % 2 else {'type': 'function', 'function': {'name': 'get_weather'}}
    think, par = seed % 3 == 0, seed % 4 < 2
    c, p, ids = tool_run(choice, think, par, seed)
    assert len(p.tool_calls) == 1 and c.done == (not par), (p.tool_calls, c.state)
    call = p.tool_calls[0]
    args = json.loads(call['arguments'])
    assert call['name'] in ('get_weather', 'get_time') and (seed % 2 or call['name'] == 'get_weather'), call
    if call['name'] == 'get_weather':
        assert isinstance(args['city'], str) and set(args) <= {'city', 'days'}, args
    assert (chat.THINK in ids) == think and chat.EOS not in ids and all(  # no text message, only reasoning + the call
        b != chat.TEXT for a, b in zip(ids, ids[1:]) if a == chat.MODEL), ids[:5]
for bad in ('bogus', {'type': 'function', 'function': {'name': 'nope'}}):
    try:
        grammar.compile_tools(tok, TOOLS, bad, True)
        raise AssertionError(bad)
    except ValueError:
        pass
assert grammar.compile_tools(tok, TOOLS, 'auto', True).lazy and grammar.compile_tools(tok, TOOLS, 'none', True) is None
assert grammar.compile_tools(tok, [], 'auto', True) is None and grammar.compile_tools(tok, None, None, True) is None

# spec draft masks across the header -> name -> json switch, then rollback: settle-side accepts still work
tc = grammar.ToolCall(grammar.compile_tools(tok, TOOLS, 'required', False))
sp = spec_runner.Spec(None, None, 0, lookup=Lookup([]), cons=tc)
call = enc('get_time') + [chat.INVOKE_JSON] + enc('{"name":"get_time","args":{}}') + [chat.END]
d = [chat.MODEL] + call[:5]
kept, bad, masks = sp._grammar(d)
assert kept == d and bad is None and tc.state == 'start' and not tc.done
assert all(torch.isfinite(masks[i, t]) for i, t in enumerate(d))
kept, bad, _ = sp._grammar([chat.MODEL, chat.TEXT])
assert kept == [chat.MODEL] and bad == chat.TEXT and tc.state == 'start'
for t in [chat.MODEL] + call:
    row = torch.zeros(V)
    tc.mask(row)
    assert torch.isfinite(row[t]), t
    tc.accept(t)
assert tc.state == 'free'
print('4 forced tool_choice: 12/12 valid calls (required / named, thinking on/off, parallel on/off); spec draft '
      'masks + rollback ok')

# 4c. forced call with thinking on (window g2 11-23: "think=True required: []"): a model that wants a text answer
# (TEXT masked) keeps choosing THINK after every reasoning block, and never ends reasoning on its own. One reasoning
# block only, capped at max_tokens // 2 (then END is forced): the call comes, valid, within max_tokens. The 'think'
# state (with its count) rolls back with spec drafts like the others.
for choice in ('required', {'type': 'function', 'function': {'name': 'get_time'}}):
    tf = grammar.compile_tools(tok, TOOLS, choice, True)
    r, _ = mkreq(E, max_tokens=120, temperature=0, tool_force=tf)
    g = torch.Generator().manual_seed(5)
    while not r.done:
        lg = torch.randn(V, generator=g)
        lg[chat.THINK] += 60
        lg[chat.TEXT] += 80
        lg[chat.EOS] += 70
        lg[chat.END] -= 30 if r.cons.state != 'json' else 0
        for s_ in ('"', '}', '1'):
            lg[enc(s_)[0]] += 6
        E._take([r], lg[None])
    calls = r.parser.tool_calls
    assert len(calls) >= 1 and r.out.count(chat.THINK) == 1, (calls, r.out.count(chat.THINK), len(r.out))
    assert (choice == 'required' or calls[0]['name'] == 'get_time') and isinstance(json.loads(calls[0]['arguments']), dict)
    assert r.out.index(chat.END) <= 1 + 1 + 60 + 1, r.out.index(chat.END)  # MODEL THINK, <= 60 reasoning, END
tc = grammar.ToolCall(grammar.compile_tools(tok, TOOLS, 'required', True), think_max=3)
sp = spec_runner.Spec(None, None, 0, lookup=Lookup([]), cons=tc)
for t in (chat.MODEL, chat.THINK, 11):
    tc.accept(t)
kept, bad, _ = sp._grammar([12, 13, chat.MODEL])   # budget 3: after 3 reasoning tokens only END
assert kept == [12, 13] and bad == chat.MODEL and tc.state == ('think', 1), (kept, bad, tc.state)
kept, bad, _ = sp._grammar([12, chat.END, chat.MODEL, chat.THINK])  # after the block: no second one
assert kept == [12, chat.END, chat.MODEL] and bad == chat.THINK and tc.state == ('think', 1), (kept, bad, tc.state)
print('4c forced call with thinking on: one bounded reasoning block, then a valid call (required / named); spec '
      'draft masks honour the budget and roll back')

# 4b. json grammar + speculation through Engine.decode_step (window g 11-23: replies cut mid-object): settle advances the
# grammar past every emitted token, so _feed must treat cons.done as true only at the round's last token. Oracle drafts
# (the plain continuation with errors) on a fake target, thinking on (json_object) and off (json_schema): == plain.
from test_dspark import FakeBD as DFakeBD, Oracle  # noqa: E402

SCH = {'type': 'json_schema', 'json_schema': {'schema': {
    'type': 'object', 'properties': {'name': {'type': 'string'}, 'size': {'type': 'string'}}, 'required': ['name', 'size']}}}


def json_run(think, fmt, ref=None, err=0.0, n=200):
    P = 12
    hist = list(range(1000, 1000 + P))
    bd = DFakeBD(V, P, hist, scale=1.0)
    fav = [enc(x)[0] for x in ('"', 'a', 'name', ' "', '":', ': ', ', "', 'x', 'size')]
    close = [enc(x)[0] for x in ('}', '"}', ' }', '{}', ' {}')]
    base = bd.logits

    def lgf(h):
        lg = base(h)
        lg[fav] += 4
        lg[close] += -20 if len(h) - P < 60 else 6
        if think:
            if chat.END in h[P:]:
                lg[chat.TEXT] += 30
            else:
                lg[chat.THINK] += 20
                lg[chat.END] += 30 * (len(h) - P > 25)
        return lg
    bd.logits = lgf
    E.bd = bd
    r, out = mkreq(E, hist, max_tokens=n, temperature=0, prefill_text=not think,
                   response_format=grammar.compile_format(tok, fmt))
    if ref is not None:
        r.sp = spec_runner.Spec(Oracle(ref, V, err, 1), bd, None, start=P, temperature=0, cons=r.cons,
                                stop=(chat.EOS,) if think else (chat.EOS, chat.END), times=(1.0,) * 8)
        r.sp.attach(0, P)
    r.slot, r.pos = 0, P
    E._take([r], bd.logits(hist + [0])[None].clone())
    while not r.done:
        E.decode_step([r])
    return r, ''.join(d[1] for d in out if d[0] == 'content')


for think, fmt in ((True, {'type': 'json_object'}), (False, SCH), (True, SCH)):
    rp, cp = json_run(think, fmt)
    assert rp.cons.done and len(rp.out) > 15, (rp.out, cp)
    for err in (0.0, 0.5):
        rs, cs = json_run(think, fmt, list(range(1000, 1012)) + rp.out, err)
        assert rs.out == rp.out and cs == cp and rs.stats['finish_reason'] == 'stop', (think, err, cp, cs)
        assert rs.sp.stats['accepted'] > 5, rs.sp.stats
print('4b json_object / json_schema + spec through decode_step (thinking on/off, oracle drafts) == plain, whole object')

# 4d. tool_choice auto (lazy grammar.ToolCall) + spec through decode_step: the model writes free text, then starts a
# call; the call body is held to the schema (a fake target that "wants" an extra '}' / a top-level key can't), drafts
# (host ids and device ids, i.e. the dev_tok path that host() masks only in settle) cross the call start and the
# output == plain, one valid call.
SCRIPT = [chat.MODEL, chat.TEXT] + enc(' Checking the weather.') + [chat.END, chat.MODEL] + enc('get_weather') + \
    [chat.INVOKE_JSON]


def auto_run(ref=None, err=0.0, dev_tok=False, n=200):
    P = 12
    hist = list(range(1000, 1000 + P))
    bd = DFakeBD(V, P, hist, scale=1.0, dev_tok=dev_tok)
    fav = [enc(x)[0] for x in ('"', '}', '",', '1')]
    want_bad = [enc(x)[0] for x in ('}}', ',"', ',"timeout', '"}}')]
    base = bd.logits

    def lgf(h):
        lg, g = base(h), h[P:]
        if len(g) < len(SCRIPT):
            lg[SCRIPT[len(g)]] += 50
        elif chat.END in g[len(SCRIPT):]:
            lg[chat.EOS] += 50
        else:
            lg[fav] += 6
            lg[want_bad] += 9
        return lg
    bd.logits = lgf
    E.bd = bd
    r, out = mkreq(E, hist, max_tokens=n, temperature=0, tool_force=grammar.compile_tools(tok, TOOLS, 'auto', True))
    assert r.cons.state == 'free'
    if ref is not None:
        r.sp = spec_runner.Spec(Oracle(ref, V, err, 1), bd, None, start=P, temperature=0, cons=r.cons,
                                stop=(chat.EOS,), times=(1.0,) * 8)
        r.sp.attach(0, P)
        assert r.sp.dev_tok == dev_tok
    r.slot, r.pos = 0, P
    E._take([r], lgf(hist)[None].clone())
    while not r.done:
        E.decode_step([r])
    return r, ''.join(d[1] for d in out if d[0] == 'content'), bd


rp, cp, _ = auto_run()
assert rp.out[:len(SCRIPT)] == SCRIPT and rp.out[-1] == chat.EOS and cp == ' Checking the weather.', (rp.out, cp)
calls = rp.stats['tool_calls']
assert len(calls) == 1 and calls[0]['name'] == 'get_weather', calls
args = json.loads(calls[0]['arguments'])
assert isinstance(args['city'], str) and set(args) <= {'city', 'days'}, args
for dt in (False, True):
    for err in (0.0, 0.5):
        rs, cs, bd = auto_run(list(range(1000, 1012)) + rp.out, err, dev_tok=dt)
        assert rs.out == rp.out and cs == cp and rs.stats['tool_calls'] == calls, (dt, err, rs.out, rp.out)
        assert rs.sp.stats['accepted'] > 5 and (bd.devrows > 0) == dt, (rs.sp.stats, bd.devrows)
print(f'4d tool_choice auto: free text, then a schema-held call {calls[0]["arguments"]}; spec (host / device draft '
      f'ids, oracle err 0 / 0.5) == plain')

# 5. prefix cache across an agent loop (canonical re-render of the generated tool call) ----------------------------
kvpool.PAGE = 16
D = tempfile.mkdtemp(prefix='inkcache-api-')
try:
    pool = kvpool.KVPool(4096, 'cpu', 'w-api', cache_dir=D, min_free_gb=0)
    E5 = stub(pool)
    u1 = [{'role': 'user', 'content': 'What is the weather in Oslo? Use the tool.'}]
    r1 = E5.request(u1, tools=TOOLS, max_tokens=64)
    # the model wrote the call with "name" first and spaces: not the canonical render
    gen = [chat.MODEL] + enc('get_weather') + [chat.INVOKE_JSON] + enc('{"name": "get_weather", "args": {"city": "Oslo"}}') \
        + [chat.END, chat.EOS]
    p = chat.Parser(tok)
    for t in gen:
        p.feed(t)
    tc_ = p.tool_calls[0]
    u2 = u1 + [{'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'call_x', 'type': 'function', 'function': tc_}]},
               {'role': 'tool', 'tool_call_id': 'call_x', 'content': '-3 C, snow'}]
    r2 = E5.request(u2, tools=TOOLS, max_tokens=64)
    P1 = len(r1.ids)
    assert r2.ids[:P1] == r1.ids and r2.ids[P1:P1 + len(gen)] != gen, 'canonical re-render should differ from gen'
    k1, k2 = E5.keys(r1.ids, []), E5.keys(r2.ids, [])
    tbl = pool.new_table()
    assert pool.reserve(tbl, P1)
    st = {'f': torch.zeros(1, 4)}
    E5.park(k1, SimpleNamespace(kv=tbl), None, True, st=dict(st))  # end-of-prompt state of turn 1
    pool.release(tbl)
    seq, P, f = E5.restore(k2)
    assert P == P1, (P, P1)
    pool.release(seq.kv)
    while E5.parked:  # pushed off the GPU (kept: an agent chat with tools) -> disk
        E5.unpark(E5.parked.pop(0))
    seq, P, f = E5.restore(k2)
    assert P == P1, ('disk', P, P1)
    pool.release(seq.kv)
    # 11-23 review 9: a request cancelled while waiting after a preemption: its parked progress is dropped (no disk)
    rq = E5.request(u2, tools=TOOLS, max_tokens=64)
    tbl = pool.new_table()
    assert pool.reserve(tbl, len(rq.ids))
    kq = E5.keys(rq.ids, [])
    E5.park(kq[:40], SimpleNamespace(kv=tbl), None, True, st=dict(st))   # a preempted prefill's progress at 40
    pool.release(tbl)
    rq.keys, rq.P, rq.a = kq, 0, 40
    n0, files0 = len(E5.parked), len(pool.files)
    E5.drop_parked(rq)
    assert len(E5.parked) == n0 - 1 and len(pool.files) == files0, (len(E5.parked), n0)
    print(f'5 agent loop: turn 2 ({len(r2.ids)} tokens) restores turn 1\'s whole prompt ({P1}) from the GPU and from '
          f'disk (saved state after unpark); the re-rendered call starts right after it (decoded tokens are never cached, so no divergence point)')
finally:
    shutil.rmtree(D, ignore_errors=True)

# 5b. an image in a cached multi-turn chat (stub vision + model, real prepare/keys/start/_chunk/park/restore): turn 2
# restores turn 1 incl. the image tiles and skips vision (GPU-parked and from disk); different pixels at the same
# position miss (no stale reuse). GPU counterpart: gpu_features.py imgcache.
import base64, io  # noqa: E401,E402
import engine as engmod  # noqa: E402
from PIL import Image  # noqa: E402


def png(px=None):
    im = Image.new('RGB', (80, 120), (200, 30, 30))
    if px:
        im.putpixel(px, (0, 0, 0))
    b = io.BytesIO(); im.save(b, 'PNG')
    return 'data:image/png;base64,' + base64.b64encode(b.getvalue()).decode()


D = tempfile.mkdtemp(prefix='inkcache-img-')
snap, dmin = engmod.SNAP_MIN, engmod.DISK_MIN
engmod.SNAP_MIN = engmod.DISK_MIN = 0  # short test prompts: park them, and keep continued chats on disk
try:
    pool = kvpool.KVPool(4096, 'cpu', 'w-img', cache_dir=D, min_free_gb=0)
    E6 = stub(pool)
    calls = []
    E6.vision = lambda tiles, r, c: calls.append(r * c) or torch.ones(r * c, 4)
    E6.vision_images, E6.chunk, E6.mem_free = 0, 1 << 16, None
    E6.model.embed = lambda ids: torch.zeros(len(ids), 4)
    E6.model.forward = lambda x, seq: x

    def turns(img):
        u1 = [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': img}},
                                           {'type': 'text', 'text': 'What colour is this image?'}]}]
        return u1, u1 + [{'role': 'assistant', 'content': 'Red.'}, {'role': 'user', 'content': 'Any other colours?'}]

    def go(msgs):
        r = E6.request(msgs, max_tokens=8, stats={})
        E6.start(r)
        while not E6.prefill_chunk(r):
            pass
        E6.finish(r)
        return r
    u1, u2 = turns(png())
    r1 = go(u1)
    assert r1.P == 0 and calls == [6] and E6.vision_images == 1, (r1.P, calls)
    img_at = r1.ids.index(chat.IMG_TOK)
    r2 = go(u2)
    assert r2.P == len(r1.ids) > img_at and r2.stats['cached_tokens'] == r2.P, (r2.P, len(r1.ids))
    assert calls == [6] and E6.vision_images == 1, 'vision re-ran for a restored image'
    while E6.parked:  # turn 2's own state (a continued chat: keep) -> disk
        E6.unpark(E6.parked.pop(0))
    assert not E6.parked and pool.states
    r3 = go(u2)
    assert r3.P == len(r3.ids) and calls == [6], ('disk', r3.P, len(r3.ids), calls)
    _, u2b = turns(png((79, 119)))  # one pixel differs: same token ids, different tile keys
    k2, k2b = E6.keys(r2.ids, r2.pre), E6.keys(r3.ids, E6.prepare(u2b, effort=0.9)[1])
    diff = (k2 != k2b).nonzero().flatten()
    assert len(diff) == 6 and int(diff[0]) == img_at, diff
    r4 = go(u2b)
    assert r4.P <= img_at and calls == [6, 6] and E6.vision_images == 2, ('stale reuse', r4.P, img_at, calls)
    print(f'5b image in a cached chat: turn 2 restores {r2.P} tokens incl. the 6 image tiles at {img_at} (GPU-parked) '
          f'and all {r3.P} from disk without re-running vision; a 1-pixel-different image misses (P {r4.P}, vision ran)')
finally:
    engmod.SNAP_MIN, engmod.DISK_MIN = snap, dmin
    shutil.rmtree(D, ignore_errors=True)

# 5c. mid-prompt checkpoints (INK_CKPT_EVERY; stub model whose state depends on every earlier token: an in-place conv
# recurrence + a replaced ring, like the real Seq): a cold agent prompt writes a checkpoint every CKPT tokens; the same
# prompt edited in the middle restores from the deepest checkpoint before the edit (not 0, not past it) and ends in a
# state bit-identical to a cold run with no cache; image tiles: an edit after the image, a checkpoint inside the image,
# and a different image all restore from the right place (vision re-runs only when needed). GPU: gpu_features.py ckpt.
D = tempfile.mkdtemp(prefix='inkcache-ckpt-')
saved = engmod.SNAP_MIN, engmod.DISK_MIN, engmod.CKPT
engmod.SNAP_MIN = engmod.DISK_MIN = 0


def ck_stub(pool, ck_calls):
    E = stub(pool)
    E.vision_images, E.chunk, E.mem_free = 0, 40, None  # 40: unaligned with CKPT, so chunks get clipped at boundaries
    E.vision = lambda tiles, r, c: ck_calls.append(r * c) or torch.full((r * c, 4), float(tiles.double().sum()), dtype=torch.float64)

    def new_seq(tbl=None):
        z = torch.zeros(3, 4, dtype=torch.float64)
        return SimpleNamespace(kv=tbl if tbl is not None else pool.new_table(), conv=[{'k': z}],
                               ring=[(z[:0], z[:0])], pos=0)

    def forward(x, seq):
        c, out = seq.conv[0]['k'], []
        rk, _ = seq.ring[0]
        for row in x:
            c[0] = c[0] * 0.75 + row + c[1] * 0.125  # in place, like sconv: the snapshot must clone it
            c[1], c[2] = c[0] * 0.5, c[1]
            out.append(c[0] + rk.sum(0))
            rk = torch.cat((rk, row[None]))[-5:]
        seq.ring[0] = (rk, rk * 2)
        seq.pos += len(x)
        return torch.stack(out)
    E.model.new_seq, E.model.forward = new_seq, forward
    E.model.embed = lambda ids: torch.tensor(ids, dtype=torch.float64)[:, None].repeat(1, 4) * 1e-3 + 1
    return E


def ck_go(E, msgs):
    r = E.request(msgs, tools=TOOLS, max_tokens=8, stats={})
    E.start(r)
    while not E.prefill_chunk(r):
        pass
    st = E.state_of(r.seq, r.f)
    E.finish(r)
    return r, st


def ck_cold(msgs):  # reference: no cache dir at all (no restore, no clipping: other chunk boundaries)
    E = ck_stub(kvpool.KVPool(8192, 'cpu', 'w-ckpt', cache_dir=None), [])
    return ck_go(E, msgs)


def ck_same(a, b, what):
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a), what


try:
    pool = kvpool.KVPool(8192, 'cpu', 'w-ckpt', cache_dir=D, min_free_gb=0)
    calls = []
    E7 = ck_stub(pool, calls)
    engmod.CKPT = CK = 64
    words = ' '.join(f'item{i} value{i * 7 % 13}' for i in range(150))
    m1 = [{'role': 'system', 'content': 'You are a careful agent.'}, {'role': 'user', 'content': words + ' Done?'}]
    m2 = [m1[0], {'role': 'user', 'content': words.replace('item80 ', 'itemX ') + ' Done?'}]
    r1, s1 = ck_go(E7, m1)
    L = len(r1.ids)
    nck = (L - 1) // CK
    assert r1.P == 0 and r1.stats.get('ckpts') == nck and L > 4 * CK, (r1.P, r1.stats, L)
    pool.flush()
    cfiles = sorted(f for f in os.listdir(pool.dir) if f.startswith('c-'))
    assert len(cfiles) == nck and sorted(int(f.split('-')[1]) for f in cfiles) == [CK * i for i in range(1, nck + 1)], cfiles
    ck_same(s1, ck_cold(m1)[1], 'cold with checkpoints (clipped chunks) == cold without')
    k1, k2 = E7.keys(r1.ids, []), E7.keys(E7.prepare(m2, TOOLS)[0], [])
    d = int((k1[:len(k2)] != k2[:len(k1)]).nonzero()[0])
    r2, s2 = ck_go(E7, m2)
    assert r2.P == d // CK * CK and 0 < r2.P <= d, (r2.P, d)  # the deepest checkpoint before the edit
    ck_same(s2, ck_cold(m2)[1], 'edited prompt restored from a checkpoint == cold')
    # GPU park gone + every page evicted: the checkpoint's pages come back from disk (published keep at the checkpoint)
    while E7.parked:
        E7.unpark(E7.parked.pop(0), disk=False)
    t = pool.new_table(); assert pool.reserve(t, 8192); pool.release(t)
    assert not pool.nodes
    r2b, s2b = ck_go(E7, m2)
    assert r2b.P >= r2.P, (r2b.P, r2.P)  # (m2's own checkpoints past the edit exist now)
    ck_same(s2b, s2, 'checkpoint restore with pages from disk == cold')
    m3 = [m1[0], {'role': 'user', 'content': words.replace('item10 ', 'itemY ') + ' Done?'}]
    d3 = int((k1 != E7.keys(E7.prepare(m3, TOOLS)[0], [])[:len(k1)]).nonzero()[0])
    r3, s3 = ck_go(E7, m3)
    assert r3.P == d3 // CK * CK, (r3.P, d3)
    ck_same(s3, ck_cold(m3)[1], 'early edit')
    print(f'5c checkpoints: {L}-token agent prompt -> {nck} checkpoints; edit at {d} restores from {r2.P} (== cold, also '
          f'with pages from disk), edit at {d3} from {r3.P}')

    # images: 320x320 = 64 tiles, checkpoints every 32 tokens, so some fall inside the image
    engmod.CKPT = CK = 32

    def big_png(px=None):
        img = Image.new('RGB', (320, 320), (20, 120, 200))
        if px:
            img.putpixel(px, (0, 0, 0))
        b = io.BytesIO(); img.save(b, 'PNG')
        return 'data:image/png;base64,' + base64.b64encode(b.getvalue()).decode()

    def imsg(img, text):
        return [{'role': 'user', 'content': [{'type': 'text', 'text': 'Look at this. ' * 9},
                                             {'type': 'image_url', 'image_url': {'url': img}},
                                             {'type': 'text', 'text': text}]}]
    tail = ' '.join(f'note{i}' for i in range(60))
    a = imsg(big_png(), tail)
    ri, si = ck_go(E7, a)
    img_at = ri.ids.index(chat.IMG_TOK)
    img_end = img_at + 64
    assert calls[-1] == 64 and any(img_at < CK * i < img_end for i in range(1, 99)), img_at
    n0 = len(calls)
    # (1) edit well after the image: restore past the image, vision not re-run
    b = imsg(big_png(), tail.replace('note40', 'noteZ'))
    rb, sb = ck_go(E7, b)
    assert rb.P > img_end and len(calls) == n0, (rb.P, img_end, calls)
    ck_same(sb, ck_cold(b)[1], 'edit after the image')
    # (2) edit at the first text token after the image: the deepest checkpoint is inside the image; vision re-runs
    # for the tiles at or past it, the restored rows before it are reused
    c_ = imsg(big_png(), 'Different ' + tail)
    rc, sc = ck_go(E7, c_)
    ka, kc = E7.keys(ri.ids, ri.pre), E7.keys(rc.ids, rc.pre)
    n_ = min(len(ka), len(kc))
    dc = int((ka[:n_] != kc[:n_]).nonzero()[0])
    assert dc >= img_end and img_at < rc.P < img_end and rc.P == dc // CK * CK and len(calls) == n0 + 1, (rc.P, dc, img_at)
    ck_same(sc, ck_cold(c_)[1], 'checkpoint inside the image')
    # (3) one pixel differs: same token ids, other tile keys -> restore from before the image, never a stale tile
    e_ = imsg(big_png((5, 5)), tail)
    re_, se = ck_go(E7, e_)
    assert re_.P == img_at // CK * CK and len(calls) == n0 + 2, (re_.P, img_at)
    ck_same(se, ck_cold(e_)[1], 'different image')
    print(f'5c checkpoints + image (tiles {img_at}..{img_end}): edit after it restores {rb.P} (no vision), checkpoint '
          f'inside it {rc.P} (vision re-run), other pixels {re_.P} (before the image); all == cold')

    # disk full: the background write fails -> no file, no index entry, no tmp; restore falls back, still == cold
    D2 = tempfile.mkdtemp(prefix='inkcache-ckpt-full-')
    pool2 = kvpool.KVPool(8192, 'cpu', 'w-ckpt', cache_dir=D2, min_free_gb=0)
    E8 = ck_stub(pool2, [])
    engmod.CKPT = 64
    real_w = kvpool._write

    def full(path, meta, tensors, pause=None):
        if os.path.basename(path).startswith('c-'):
            open(f'{path}.{os.getpid()}.tmp', 'wb').write(b'partial')
            raise OSError(28, 'No space left on device (injected)')
        return real_w(path, meta, tensors)
    kvpool._write = full
    try:
        rf, sf = ck_go(E8, m1)
    finally:
        kvpool._write = real_w
    pool2.flush()
    assert not [f for f in os.listdir(pool2.dir) if f.startswith('c-') or f.endswith('.tmp')] and not pool2.pending
    assert not [p for p in pool2.files if os.path.basename(p).startswith('c-')] and not any(pool2.states.values()), pool2.states
    while E8.parked:
        E8.unpark(E8.parked.pop(0), disk=False)
    rg, sg = ck_go(E8, m2)
    assert rg.P == 0
    ck_same(sg, s2, 'failed checkpoint writes: cold fallback')
    shutil.rmtree(D2, ignore_errors=True)
    print('5c checkpoint write failure (ENOSPC): skipped, nothing left behind, request served cold')
finally:
    engmod.SNAP_MIN, engmod.DISK_MIN, engmod.CKPT = saved
    shutil.rmtree(D, ignore_errors=True)

# 5d. VRAM: a prefill OOM where only drafter state and a paused prefill can be freed (window g2 11-23: "prefill OOM,
# freeing 0 idle slot decoders + 0 parked states" -> 503 with 4 spec decodes running). The OOM restart frees the parked
# (paused, keep -> disk) prefill, then turns speculative decoding off for the running requests (States + graphs +
# drafter weights); the request completes, the running ones keep decoding plain. Without the OOM, the chunk's margin
# check does the same first (spec=True), so no OOM happens at all.
D = tempfile.mkdtemp(prefix='inkcache-oom-')
saved = engmod.SNAP_MIN, engmod.DISK_MIN
engmod.SNAP_MIN = engmod.DISK_MIN = 0
try:
    for preventive in (False, True):
        pool = kvpool.KVPool(8192, 'cpu', 'w-oom', cache_dir=D, min_free_gb=0)
        E8 = ck_stub(pool, [])
        E8.bd = SimpleNamespace(drop=lambda keep=0: 0, idle=lambda: 0, busy=lambda: True, dec=False)
        E8.ooms, E8.ds_free = 0, []
        E8.ds = SimpleNamespace(live=2)
        run = [type('Run', (), dict(sp=SimpleNamespace(st=SimpleNamespace(graphs={'k': 1}))))() for _ in range(2)]
        E8.spec_live = set(run)                                               # two running spec decodes
        paused = E8.request([{'role': 'user', 'content': 'x ' * 300}], tools=TOOLS, max_tokens=8, stats={})
        E8.start(paused); E8.prefill_chunk(paused); E8.preempt_prefill(paused)   # a paused prefill, parked (keep)
        assert len(E8.parked) == 1 and paused.a > 0
        held = lambda: 200 * len(E8.parked) + (800 if E8.ds is not None else 0) + 160 * sum(
            x.sp is not None for x in run)                                     # MiB "held" by freeable things
        fwd, oom = E8.model.forward, [0]

        def forward(x, seq, fwd=fwd):
            if held() > 0:
                oom[0] += 1
                raise torch.OutOfMemoryError('CUDA out of memory (injected)')
            return fwd(x, seq)
        E8.model.forward = forward
        if preventive:   # the margin check sees the shortfall: frees the same things before the chunk, no OOM
            E8.mem_free = lambda: (4000 - held()) << 20
            saved_m, engmod.MARGIN = engmod.MARGIN, 3999 << 20
        r8 = E8.request([{'role': 'user', 'content': 'y ' * 200}], tools=TOOLS, max_tokens=8, stats={})
        try:
            E8.start(r8)
            while not E8.prefill_chunk(r8):
                pass
        finally:
            if preventive:
                engmod.MARGIN = saved_m
        assert r8.a == len(r8.ids) and E8.ooms == (0 if preventive else 1) and oom[0] == (0 if preventive else 1), \
            (r8.a, E8.ooms, oom)
        assert all(x.sp is None for x in run) and E8.ds is None and not E8.spec_live and len(E8.parked) == 1, E8.parked
        assert E8.parked[0][0].shape[0] == len(r8.ids)                        # (only r8's own end state is parked now)
        pool.flush()
        assert any(f.startswith('s-') for f in os.listdir(pool.dir)), 'the paused prefill went to disk'
        E8.finish(r8)
    print('5d prefill OOM with only drafter States/weights + a paused prefill freeable: restart frees them (spec off for '
          'running requests, paused prefill to disk), request completes; the margin check does it without an OOM')
finally:
    engmod.SNAP_MIN, engmod.DISK_MIN = saved
    shutil.rmtree(D, ignore_errors=True)

# 6. server ---------------------------------------------------------------------------------------------------------
from fastapi.testclient import TestClient  # noqa: E402

import server  # noqa: E402
from test_batch_sched import FakeEngine  # noqa: E402

seen = []


class Fake(FakeEngine):
    def request(self, messages, **kw):
        seen.append({k: v for k, v in kw.items() if k not in ('stats', 'emit', 'prio', 'cancel')})
        return super().request([{'content': 'ok:1:2'}], **{k: kw[k] for k in ('stats', 'emit', 'prio', 'cancel')})


server.eng = Fake(chunk_s=0, step_s=0)
server.eng.tok = tok
server.sched = server.Scheduler(server.eng)
c = TestClient(server.app)
base = dict(messages=[{'role': 'user', 'content': 'hi'}])
post = lambda **kw: c.post('/v1/chat/completions', json=dict(base, **kw))
r = post(n=3, seed=7, logprobs=True, top_logprobs=2).json()
assert [ch['index'] for ch in r['choices']] == [0, 1, 2] and all(ch['logprobs'] == {'content': []} for ch in r['choices'])
assert r['usage']['completion_tokens'] == 6 and r['usage']['prompt_tokens'] == 1, r['usage']
assert [s['seed'] for s in seen[-3:]] == [7, 8, 9] and seen[-1]['logprobs'] == 2
assert post().json()['choices'][0]['logprobs'] is None
s = post(n=2, stream=True, stream_options={'include_usage': True}).text
chunks = [json.loads(x[6:]) for x in s.split('\n\n') if x.startswith('data: {')]
assert {ch['choices'][0]['index'] for ch in chunks if ch['choices']} == {0, 1}
assert sum(1 for ch in chunks if ch['choices'] and ch['choices'][0]['finish_reason']) == 2
assert chunks[-1]['usage']['completion_tokens'] == 4 and s.endswith('data: [DONE]\n\n')
r = post(temperature=None, top_p=None, logit_bias={'42': 5}, presence_penalty=0.5, frequency_penalty=0.2,
         repetition_penalty=1.1, min_p=0.05, top_k=-1, min_tokens=2, user='u', store=False, metadata={'a': 1},
         parallel_tool_calls=False, tools=TOOLS, tool_choice={'type': 'function', 'function': {'name': 'get_time'}},
         max_completion_tokens=99, stop_token_ids=None)
assert r.status_code == 400  # min_tokens + forced tool
r = post(temperature=None, top_p=None, logit_bias={'42': 5}, presence_penalty=0.5, frequency_penalty=0.2,
         repetition_penalty=1.1, min_p=0.05, top_k=-1, user='u', store=False, metadata={'a': 1},
         parallel_tool_calls=False, tools=TOOLS, tool_choice={'type': 'function', 'function': {'name': 'get_time'}},
         max_completion_tokens=99)
assert r.status_code == 200, r.text
k = seen[-1]
assert k['logit_bias'] == {42: 5.0} and k['temperature'] == 1.0 and k['top_k'] == 0 and k['max_tokens'] == 99
assert k['one_call'] and isinstance(k['tool_force'], grammar.Tools) and k['tool_force'].think and k['min_p'] == 0.05
assert isinstance(post(tools=TOOLS, tool_choice='required').status_code, int) and \
    isinstance(seen[-1]['tool_force'], grammar.Tools) and post(tools=TOOLS).status_code == 200 and \
    seen[-1]['tool_force'].lazy  # tool_choice auto (default): lazy constraint
for kw in (dict(tool_choice='none'), dict(response_format={'type': 'json_object'}), dict(min_tokens=3)):
    assert post(tools=TOOLS, **kw).status_code == 200 and seen[-1]['tool_force'] is None, kw  # auto: unconstrained
for bad in (dict(n=0), dict(n=9), dict(top_logprobs=2), dict(logprobs=True, top_logprobs=21), dict(ignore_eos=True),
            dict(tool_choice='required'), dict(tools=TOOLS, tool_choice={'type': 'function', 'function': {'name': 'x'}}),
            dict(presence_penalty=3), dict(logit_bias={'999999': 1}), dict(modalities=['text', 'audio']),
            dict(tools=TOOLS, tool_choice='required', response_format={'type': 'json_object'})):
    assert post(**bad).status_code == 400, bad
print('6 server: n (choices, seed + i, usage once/summed, streamed indexes), logprobs fields, params, 400s')
print('ALL OK')
