"""CPU: random logits + grammar mask -> always valid JSON matching the schema, message ends with END.
Run: cd engine && PATH=$PWD/.venv/bin:$PATH python tests/test_grammar.py"""
import json
import os
import sys

import torch
from tokenizers import Tokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import chat
import grammar
from decode import sample
from engine import MODEL_DIR

tok = Tokenizer.from_file(os.path.join(MODEL_DIR, 'tokenizer.json'))
SCHEMA = {'type': 'object', 'properties': {'city': {'type': 'string'}, 'population_millions': {'type': 'number'},
                                           'tags': {'type': 'array', 'items': {'enum': ['a', 'b']}}},
          'required': ['city', 'population_millions'], 'additionalProperties': False}


def run(rf, in_text, seed, bias=None, max_steps=400):
    g = torch.Generator().manual_seed(seed)
    c = grammar.Json(grammar.compile_format(tok, rf), in_text)
    p = chat.Parser(tok)
    if in_text:
        p.kind = 'content'
    out, ids = [], []
    for _ in range(max_steps):
        lg = torch.randn(grammar.VOCAB, generator=g) * 3
        if bias:
            bias(lg, ids)
        c.mask(lg)
        t = int(sample(lg, 1.0, 1.0, 0, g))
        c.accept(t)
        ids.append(t)
        out += [d[1] for d in p.feed(t) if d[0] == 'content']
        if c.done:
            return ''.join(out), ids
    raise AssertionError(f'grammar did not finish in {max_steps} steps: {"".join(out)[:200]!r}')


def favor_close(lg, ids):  # random logits rarely close strings/arrays; nudge so outputs stay short
    for s in ('"', '}', ']', '",', '1', '.'):
        lg[tok.encode(s, add_special_tokens=False).ids[0]] += 6


ok = 0
for seed in range(20):
    s, ids = run({'type': 'json_schema', 'json_schema': {'name': 'c', 'schema': SCHEMA, 'strict': True}}, True, seed,
                 favor_close)
    d = json.loads(s)
    assert set(d) <= {'city', 'population_millions', 'tags'} and isinstance(d['city'], str), d
    assert isinstance(d['population_millions'], (int, float)) and not isinstance(d['population_millions'], bool), d
    assert all(x in ('a', 'b') for x in d.get('tags', [])), d
    assert ids[-1] == chat.END and all(i < chat.USER for i in ids[:-1]), ids
    ok += 1
print(f'json_schema, thinking off: {ok}/20 valid; e.g. {s[:100]!r}')

for seed in range(5):
    s, _ = run({'type': 'json_object'}, True, seed, favor_close)
    assert isinstance(json.loads(s), dict)
print('json_object: 5/5 valid dict; e.g.', repr(s[:80]))


def think_then(lg, ids):  # thinking on: the model "wants" to reason for a while, then try EOS / tool calls
    favor_close(lg, ids)
    if len(ids) == 1:
        lg[chat.THINK] += 100  # wants a reasoning message
    elif ids and ids[-1] == chat.END:
        lg[chat.EOS] += 100  # wants to stop right after reasoning (must be overridden)
    if len(ids) == 30:
        lg[chat.END] += 100  # end reasoning


s, ids = run({'type': 'json_schema', 'json_schema': {'schema': SCHEMA}}, False, 7, think_then)
assert ids[0] == chat.MODEL and ids[1] == chat.THINK and chat.EOS not in ids, ids[:3]
i = ids.index(chat.TEXT)
assert ids[i - 1] == chat.MODEL and ids[-1] == chat.END
json.loads(s)
print(f'thinking on: reasoning {i - 3} tokens free, then MODEL TEXT + JSON: {s[:80]!r}')

# tool_choice auto: lazy grammar.ToolCall --------------------------------------------------------------------------
enc = lambda x: tok.encode(x, add_special_tokens=False).ids
BASH = {'type': 'function', 'function': {'name': 'Bash', 'parameters': {
    'type': 'object', 'properties': {'command': {'type': 'string'}, 'description': {'type': 'string'},
                                     'timeout': {'type': 'number'}}, 'required': ['command'], 'additionalProperties': False}}}
READ = {'type': 'function', 'function': {'name': 'Read', 'parameters': {
    'type': 'object', 'properties': {'file_path': {'type': 'string'}}, 'required': ['file_path']}}}
AUTO = [BASH, READ]


def allowed(c, t):
    row = torch.zeros(grammar.VOCAB)
    c.mask(row)
    return bool(torch.isfinite(row[t]))


def feed(c, ids, p=None):
    for t in ids:
        assert allowed(c, t), (tok.decode([t]), c.state)
        c.accept(t)
        if p is not None:
            p.feed(t)


assert grammar.compile_tools(tok, AUTO, 'none', True) is None and grammar.compile_tools(tok, None, 'auto', True) is None
c = grammar.ToolCall(grammar.compile_tools(tok, AUTO, 'auto', True))
assert c.state == 'free' and not c.done
# free text / reasoning / EOS: untouched (mask is a no-op outside a call)
g = torch.Generator().manual_seed(0)
for ids in ([chat.MODEL, chat.THINK] + enc('Let me look.') + [chat.END, chat.MODEL, chat.TEXT] + enc('Sure: done.') +
            [chat.END],):
    for t in ids:
        lg = torch.randn(grammar.VOCAB, generator=g)
        if c.state != 'lhead':
            assert torch.equal(c.mask(lg.clone()), lg), c.state
        c.accept(t)
assert c.state == 'free' and allowed(c, chat.EOS) and not c.done
# header: block types and the declared names' first tokens only (no nameless call, no other name)
c.accept(chat.MODEL)
assert c.state == 'lhead' and all(allowed(c, t) for t in (chat.THINK, chat.TEXT, chat.EOS, enc('Bash')[0], enc('Read')[0]))
assert not allowed(c, chat.INVOKE_JSON) and not allowed(c, enc('Grep')[0]) and not allowed(c, enc('hello')[0])
c.accept(chat.TEXT)
assert c.state == 'free'
# the live failure: after MODEL Bash INVOKE_JSON neither the out-of-schema top-level key nor the extra '}' can come
c = grammar.ToolCall(grammar.compile_tools(tok, AUTO, 'auto', True))
feed(c, [chat.MODEL] + enc('Bash') + [chat.INVOKE_JSON])
live = '{"name":"Bash","args":{"command":"ls ...","description":"..."},"timeout":30000}}'
ok = live[:live.index(',"timeout"')]  # canonical up to the end of args (name first, as the chat template writes it)
feed(c, enc(ok))
assert not allowed(c, enc(',')[0]) and not allowed(c, enc(',"timeout"')[0])  # no top-level key after args
c.accept(enc('}')[0])
assert not allowed(c, enc('}')[0]) and allowed(c, chat.END)  # no extra '}': only END
c.accept(chat.END)
assert c.state == 'free' and not c.done
# random logits that "want" a call: always a valid call; parallel -> several calls, parallel off -> done after one


def auto_run(parallel, seed, max_steps=600):
    c = grammar.ToolCall(grammar.compile_tools(tok, AUTO, 'auto', True, parallel))
    p, ids, g = chat.Parser(tok), [], torch.Generator().manual_seed(seed)
    for _ in range(max_steps):
        lg = torch.randn(grammar.VOCAB, generator=g) * 3
        favor_close(lg, ids)
        lg[[enc('}}')[0], enc(',"')[0], enc('"}}')[0]]] += 9   # the failure shapes, everywhere
        lg[chat.MODEL] += 30 if c.state == 'free' and len(p.tool_calls) < 3 else 0
        lg[chat.EOS] += 30 if len(p.tool_calls) >= 3 else -30
        c.mask(lg)
        t = int(sample(lg, 1.0, 1.0, 0, g))
        c.accept(t)
        ids.append(t)
        p.feed(t)
        if c.done or t == chat.EOS:
            return c, p, ids
    raise AssertionError('no end')


for seed in range(6):
    par = seed % 2 == 0
    c, p, ids = auto_run(par, seed)
    assert len(p.tool_calls) == (3 if par else 1) and c.done == (not par), (p.tool_calls, c.state)
    for call in p.tool_calls:
        a = json.loads(call['arguments'])
        assert call['name'] in ('Bash', 'Read') and isinstance(a, dict), call
        assert (set(a) <= {'command', 'description', 'timeout'} and isinstance(a['command'], str)) \
            if call['name'] == 'Bash' else isinstance(a['file_path'], str), call
print(f'tool_choice auto: free text untouched, header limited to blocks + declared names, live failure body '
      f'rejected, 6/6 random runs valid (parallel: 3 calls; off: done after 1), e.g. {p.tool_calls[0]}')

# spec draft masks + rollback across the call start, and across a call's end into the next call (own matcher)
import spec_runner  # noqa: E402
from spec import Lookup  # noqa: E402
c = grammar.ToolCall(grammar.compile_tools(tok, AUTO, 'auto', True))
sp = spec_runner.Spec(None, None, 0, lookup=Lookup([]), cons=c)
txt = enc(' ok') + [chat.END]
kept, bad, masks = sp._grammar(txt)
assert kept == txt and bad is None and masks is None  # outside a call: no masks at all
call = [chat.MODEL] + enc('Read') + [chat.INVOKE_JSON] + enc('{"name":"Read","args":{"file_path":"/a"}}') + [chat.END]
d = txt + call[:6]
kept, bad, masks = sp._grammar(d)
assert kept == d and bad is None and c.state == 'free' and c.m is None and all(torch.isfinite(masks[i, t])
                                                                              for i, t in enumerate(d))
kept, bad, _ = sp._grammar(txt + [chat.MODEL, chat.INVOKE_JSON])
assert kept == txt + [chat.MODEL] and bad == chat.INVOKE_JSON and c.state == 'free'
kept, bad, _ = sp._grammar([chat.MODEL] + enc('Bash') + [chat.INVOKE_JSON] + enc('{"args"'))
assert bad is not None and bad in enc('{"args"') and c.state == 'free'  # args before name: off-template
# device draft ids (dev_tok): host() reads + cuts + masks them in settle
ctx = [torch.tensor(txt + [chat.MODEL, chat.INVOKE_JSON]), None, None, None]
sp.host(ctx)
assert ctx[0] == txt + [chat.MODEL] and ctx[3] == chat.INVOKE_JSON and ctx[2] is not None
feed(c, call[:-3])
m0, s0 = c.m, c.state
two = call[-3:] + call[:8]  # end of call 1, then the start of call 2 (a fresh matcher); <= RMAX rows
kept, bad, _ = sp._grammar(two)
assert kept == two and bad is None and c.m is m0 and c.state == s0
feed(c, two + call[8:])  # m0 rolled back correctly: the same tokens are accepted for real
assert c.state == 'free' and not c.done
print('tool_choice auto + spec drafts: no masks outside calls, masks + rollback across the call start and into a '
      'second call, device draft ids cut in host()')

# chat.Parser: name from the header, args from "args" / "arguments" / the whole body; never a nameless call


def parse(body, name='Bash'):
    p, out = chat.Parser(tok), []
    for t in [chat.MODEL] + enc(name) + [chat.INVOKE_JSON] + enc(body) + [chat.END]:
        out += p.feed(t)
    return p.tool_calls, out


want = [{'name': 'Bash', 'arguments': '{"command": "ls"}'}]
for b in ('{"command":"ls"}', '{"name":"Bash","arguments":{"command":"ls"}}', '{"name":"Bash","arguments":"{\\"command\\":\\"ls\\"}"}',
          '{"args":{"command":"ls"},"name":"Bash"}', '{"args":{"command":"ls"},"name":"Other"}'):
    assert parse(b) == (want, []), (b, parse(b))
assert parse('{"args":{"command":"ls"},"name":"Bash"}', name='') == (want, [])  # body name as the fallback
for b in ('{"command":"ls"}', '{"name":"","args":{}}'):
    assert parse(b, name='') == ([], [('content', b)]), b  # no name anywhere: raw text, not an empty call
assert parse('{"args":[1]}') == ([], [('content', '{"args":[1]}')]) and parse('{oops') == ([], [('content', '{oops')])
print('chat.Parser tool calls: header name, args / arguments / bare body, no nameless calls')

try:
    grammar.compile_format(tok, {'type': 'xml'}); raise SystemExit('no error for bad type')
except ValueError as e:
    print('bad type ->', e)
print('ALL OK')
