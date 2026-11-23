"""CPU, real weights (slow: several minutes). Prefix reuse == cold prefill (bit-exact when chunk boundaries match) for
GPU-parked, disk-restored and request-in-between cases; a preempted prompt resumes where it stopped (also from disk);
Decoder slots (eager on CPU) == plain decode in generate; 2 concurrent requests + a VIP that parks a decode
(server.Scheduler) == each run alone; speculative decoding through the engine (SeqBatch: verify rows = single-row
forwards; drafts forced): greedy DSpark / lookup spec == plain incl. max_tokens and stop, prefix restore with spec
(drafter starts at P) == cold plain, the scheduler case with a spec: false slot next to spec slots and the VIP parking
a spec decode (resumes with a fresh drafter) == solo, greedy spec with logit processors == plain; then (--json) a greedy json_schema generation (+ with spec).
Mid-prompt checkpoints (INK_CKPT_EVERY): an edited prompt restored from a checkpoint == cold.
Run: cd engine && PATH=$PWD/.venv/bin:$PATH python tests/test_engine_cpu.py [--json] [--rest: skip the prefix + checkpoint cases] [--ckpt: only the checkpoint case]"""
import json
import os
import shutil
import sys
import tempfile
import time

CACHE = tempfile.mkdtemp(prefix='inkcache-cpu-')
os.environ.update(INK_CACHE_DIR=CACHE, INK_CACHE_MIN_FREE_GB='0', INK_SNAP_MIN='0', INK_DISK_MIN='0')
memo = os.path.expanduser('~/llm/inkcache/weights_id.json')  # skip re-hashing 19 GB if the server already did
if os.path.exists(memo):
    shutil.copy(memo, CACHE)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch

import chat
import engine
import grammar
import kvpool
from decode import Decoder
from engine import Engine, Preempted
from loopbatch import LoopBatch, SoloFused
import spec_runner
from spec_runner import SeqBatch

kvpool.PAGE = 16  # shortcut: tiny pages so these short prompts span several full pages (test-only; FORMAT unchanged)
eng = Engine('cpu', max_ctx=2048, vision=False)
m, pool = eng.model, eng.pool
turn1 = [{'role': 'user', 'content': 'My name is Ada and I keep three cats. Remember that.'}]
turn2 = turn1 + [{'role': 'assistant', 'content': 'Got it, Ada: three cats.'},
                 {'role': 'user', 'content': 'How many cats do I have?'}]
A, _ = eng.prepare(turn1, effort=0)
B, _ = eng.prepare(turn2, effort=0)
assert B[:len(A)] == A
other, _ = eng.prepare([{'role': 'user', 'content': 'Something unrelated entirely, about the weather in Oslo.'}], effort=0)
eng.chunk = len(A)  # cold B then splits at len(A) too, so warm and cold run the same ops: expect bit-exact
print(f'A {len(A)} tokens, B {len(B)} tokens, chunk {eng.chunk}, cache {CACHE}', flush=True)


def run(ids, label, preempt=None):
    t0, st = time.time(), {}
    seq, f = eng.prefill(ids, [], st, preempt=preempt, max_tokens=4, keep=True)
    t = int(m.logits(f)[0].argmax())
    f2 = m.forward(m.embed([t]), seq)  # one decode step after the prompt (exercises the restored rings/conv)
    pool.release(seq.kv)
    print(f'{label}: cached {st["cached_tokens"]}/{st["prompt_tokens"]}, {time.time() - t0:.0f} s, next {t}', flush=True)
    return st['cached_tokens'], f, f2


def forget_gpu(disk, wipe=False):
    """Drop all parked states (to disk if `disk`) and evict every cached page (spilled if kept); wipe: also delete the
    disk cache (for true cold references)."""
    while eng.parked:
        eng.unpark(eng.parked.pop(0), disk)
    t = pool.new_table()
    assert pool.reserve(t, eng.max_ctx)
    pool.release(t)
    assert not pool.nodes
    for f in list(pool.files) if wipe else []:
        pool._drop_file(f)


def same(a, b, what):
    assert torch.equal(a[1], b[1]) and torch.equal(a[2], b[2]), (what, (a[1].float() - b[1].float()).abs().max())
    print('PASS', what)


@torch.no_grad()
def prefix_cases():
    """GPU-parked, disk-restored and request-in-between warm runs == cold (bit-exact)."""
    run(A, 'turn 1 cold')
    warm = run(B, 'turn 2 warm (GPU-parked)')
    assert warm[0] == len(A)
    forget_gpu(disk=False)
    cold = run(B, 'turn 2 cold')
    assert cold[0] == 0
    same(warm, cold, 'GPU-parked warm == cold (prompt end + one decode step, bit-exact)')

    forget_gpu(disk=False)
    run(A, 'turn 1 cold again')
    forget_gpu(disk=True)
    assert any(f.endswith('.page') for f in os.listdir(pool.dir)) and pool.states
    disk = run(B, 'turn 2 warm (from disk)')
    assert disk[0] == len(A)
    same(disk, cold, 'disk-restored warm == cold')

    forget_gpu(disk=False)
    run(A, 'turn 1 again (its state is on disk now)')
    run(other, 'unrelated request in between')
    mid = run(B, 'turn 2 warm after another request')
    assert mid[0] == len(A)
    same(mid, cold, 'warm after a request in between == cold')



@torch.no_grad()
def ckpt_case():
    """INK_CKPT_EVERY: B edited in the middle restores from the deepest mid-prompt checkpoint of B (state and pages
    from disk) and == a cold run of the edited prompt (bit-exact: the same chunk boundaries)."""
    ck0, chunk0 = engine.CKPT, eng.chunk
    engine.CKPT, eng.chunk = 2 * kvpool.PAGE, kvpool.PAGE
    try:
        i = len(B) * 2 // 3
        B2 = B[:i] + [B[i] + 1] + B[i + 1:]
        forget_gpu(disk=False, wipe=True)
        cold2 = run(B2, 'B edited cold')
        forget_gpu(disk=False, wipe=True)
        run(B, 'B cold, writes checkpoints')
        pool.flush()
        n = sum(os.path.basename(f).startswith('c-') for f in pool.files)
        assert n == (len(B) - 1) // engine.CKPT, (n, len(B))
        forget_gpu(disk=False)  # parked end state dropped, pages evicted (spilled: published keep at each checkpoint)
        warm = run(B2, 'B edited warm')
        assert warm[0] == i // engine.CKPT * engine.CKPT > 0, (warm[0], i)
        same(warm, cold2, f'edited prompt restored from the checkpoint at {warm[0]} (disk) == cold (edit at {i})')
    finally:
        engine.CKPT, eng.chunk = ck0, chunk0


try:
    if '--ckpt' in sys.argv:  # only the checkpoint case
        ckpt_case()
        print('ALL OK')
        raise SystemExit(0)
    if '--rest' not in sys.argv:
        prefix_cases()
        ckpt_case()
    with torch.no_grad():
        # preempted prompt: stops after one chunk, parks, goes to disk, resumes at the next chunk (no restart)
        eng.chunk = 16
        forget_gpu(disk=False, wipe=True)
        ref = run(B, 'B cold, chunk 16')
        assert ref[0] == 0
        forget_gpu(disk=False, wipe=True)
        calls = []
        try:
            run(B, 'B preempted', preempt=lambda: calls.append(1) or len(calls) == 2)
            raise AssertionError('not preempted')
        except Preempted:
            pass
        assert [len(e[0]) for e in eng.parked] == [16] and pool.free_pages() + len(pool._evictable()) == pool.n - 2
        forget_gpu(disk=True)
        res = run(B, 'B resumed (from disk)')
        assert res[0] == 16
        same(res, ref, 'preempted + resumed == cold')

        # VRAM pressure (fake mem_get_info / injected OOM): parked states are pushed out, warm == cold still holds
        eng.chunk = len(A)
        forget_gpu(disk=False, wipe=True)
        cold = run(B, 'B cold, chunk len(A)')
        forget_gpu(disk=False, wipe=True)
        unparks, real_unpark = [], eng.unpark
        eng.unpark = lambda e, disk=True: unparks.append(len(e[0])) or real_unpark(e, disk)
        run(A, 'A (parked)')
        eng.mem_free = lambda: 0  # always under the margin
        warm = run(B, 'B warm, margin unmet -> A unparked before the first chunk')
        eng.mem_free = None
        assert warm[0] == len(A) and len(A) in unparks and [len(e[0]) for e in eng.parked] == [len(B)], (warm[0], unparks)
        same(warm, cold, 'warm == cold after a margin-forced unpark')

        forget_gpu(disk=False, wipe=True)
        run(A, 'A (parked, keep)')
        real_fwd, n_oom = m.forward, [1]

        def flaky(x, seq):  # first call OOMs (before any write), then the real forward
            if n_oom[0]:
                n_oom[0] -= 1; raise torch.OutOfMemoryError('CUDA out of memory (injected)')
            return real_fwd(x, seq)
        m.forward, ooms0 = flaky, eng.ooms
        retry = run(B, 'B, one injected OOM -> unpark all + restart (A from disk)')
        m.forward = real_fwd
        assert eng.ooms == ooms0 + 1 and retry[0] == len(A), retry[0]
        same(retry, cold, 'OOM restart == cold')

        from engine import OutOfMemory
        forget_gpu(disk=False, wipe=True)
        run(A, 'A (parked)')

        def oom(*a):
            raise torch.OutOfMemoryError('CUDA out of memory (injected)')
        m.forward = oom
        try:
            eng.prefill(B, [], {}, max_tokens=4)
            raise AssertionError('no OutOfMemory')
        except OutOfMemory:
            pass
        m.forward = real_fwd
        assert not eng.parked and pool.free_pages() + len(pool._evictable()) == pool.n - 1, 'pages leaked'
        print('PASS second OOM -> OutOfMemory, parked states dropped, pages released')
        real_logits, m.logits = m.logits, oom
        try:
            list(eng.generate(turn1, max_tokens=3, temperature=0, effort=0))
            raise AssertionError('no OutOfMemory')
        except OutOfMemory:
            pass
        m.logits = real_logits
        assert pool.free_pages() + len(pool._evictable()) == pool.n - 1 - sum(len(e[1].pages) for e in eng.parked)
        print('PASS decode OOM -> OutOfMemory, pages released')
        eng.unpark = real_unpark

        # batch decoder in generate: plain Seq slots vs per-slot INK_GRAPH Decoders (eager on CPU) give the same output
        eng.chunk = len(A)
        outs = []
        for bd in (LoopBatch(m, 2), LoopBatch(m, 2, dec=True)):
            eng.bd, st = bd, {}
            outs.append(''.join(d[1] for d in eng.generate(turn1, max_tokens=3, temperature=0, effort=0, stats=st)))
            print(f'generate ({"Decoder" if bd.dec else "plain"} slots): {outs[-1]!r}, cached {st["cached_tokens"]}',
                  flush=True)
        # stop sequence (thinking off, so the reply is content): a string from inside the greedy reply -> output cut
        # just before it, finish_reason stop; Decoder slots, then plain
        ref = ''.join(d[1] for d in eng.generate(turn1, max_tokens=5, temperature=0, effort=0, prefill_text=True))
        cut = ref.strip()[1:3]
        assert len(cut) == 2, ref
        for bd in (LoopBatch(m, 2, dec=True), LoopBatch(m, 2)):
            eng.bd, st = bd, {}
            got = ''.join(d[1] for d in eng.generate(turn1, max_tokens=5, temperature=0, effort=0, prefill_text=True,
                                                     stats=st, stop=['never seen', cut]))
            assert got == ref[:ref.index(cut)] and st['finish_reason'] == 'stop', (got, ref, cut, st)
        print(f'PASS stop {cut!r}: {ref!r} -> {got!r}')
        # long uncached prompt (DEC_DROP, counted on the uncached suffix since S1): solo, the idle slots' Decoders
        # dropped before prefill, a new one built for decode; == the same prompt run again warm (no solo, no drop)
        fresh = [{'role': 'user', 'content': 'Solo check: a prompt nothing has cached yet.'}]
        eng.bd = LoopBatch(m, 2, dec=True)
        eng.bd.decs[0] = first = Decoder(m, graph=False)
        old_drop, engine.DEC_DROP = engine.DEC_DROP, 1
        a1 = ''.join(d[1] for d in eng.generate(fresh, max_tokens=3, temperature=0, effort=0))
        assert eng.bd.decs[0] is not None and eng.bd.decs[0] is not first, 'decoder not dropped + rebuilt'
        first = eng.bd.decs[0]
        st = {}
        a2 = ''.join(d[1] for d in eng.generate(fresh, max_tokens=3, temperature=0, effort=0, stats=st))
        assert st['cached_tokens'] > 0 and eng.bd.decs[0] is first, 'warm turn went solo (S1)'
        engine.DEC_DROP = old_drop
        assert outs[0] == outs[1] and a1 == a2, (outs, a1, a2)
        print('PASS Decoder slots in generate == plain decode (also dropped + rebuilt for a long prompt)')

        # continuous batching (server.Scheduler, 2 slots, Decoder slots): two concurrent greedy requests, then a VIP
        # that parks the newer decode at its position; every output == the same request run alone, and the parked
        # one resumed from its parked state (no recompute)
        import threading
        import server
        lows = [turn1, [{'role': 'user', 'content': 'Name three colors.'}]]
        vipm = [{'role': 'user', 'content': 'Say hi.'}]
        eng.bd = LoopBatch(m, 2, dec=True)
        solo = [''.join(d[1] for d in eng.generate(p, max_tokens=n, temperature=0, effort=0))
                for p, n in ((lows[0], 8), (lows[1], 8), (vipm, 3))]
        print('solo:', solo, flush=True)
        # SoloFused (INK_SOLO_FUSED): the lone decode on a Decoder, moved to the batch side (plain Seq slots here) when
        # a second one loads and back when it is alone again (state(slot) -> load): same outputs
        # spec: drafts every round at k <= 2 (+ deeper probes) so a CPU verify stays short
        spec_runner.step_ms = lambda sampled=False: (1.0, 1.0, 1.0) + (100.0,) * 5
        spec_runner.VERIFY_MS = (1.0, 1.0, 1.0) + (100.0,) * 5
        for label, bd in (('Decoder slots', LoopBatch(m, 2, dec=True)), ('SoloFused', SoloFused(LoopBatch(m, 2), m)),
                          ('spec', SeqBatch(m, 2))):
            eng.bd, moves = bd, []
            eng.spec = 'dspark' if label == 'spec' else 'off'
            if label == 'SoloFused':
                real_move = bd._move
                bd._move = lambda *a: (moves.append(a[0] is bd.one), real_move(*a))
            sched, res = server.Scheduler(eng), {}

            def submit(name, msgs, n, prio):
                out, ev = [], threading.Event()

                def emit(kind, v):
                    if kind == 'delta':
                        out.append(v[1])
                    else:
                        res[name] = (kind, v, ''.join(out)); ev.set()
                # spec label: low0 opts out (spec: false), low1 (parked by the VIP) and the VIP draft
                req = eng.request(msgs, max_tokens=n, temperature=0, effort=0, emit=emit, prio=prio,
                                  spec=False if name == 'low0' else None)
                sched.submit(req)
                return req, ev
            t0, rows0 = time.time(), server.M['decode_rows']
            jobs = [submit('low0', lows[0], 8, 0), submit('low1', lows[1], 8, 0)]
            while not (len(sched.decoding) == 2 and len(jobs[1][0].out) >= 2) and time.time() - t0 < 3600:
                time.sleep(0.2)
            assert len(sched.decoding) == 2, 'both lows never decoded together'
            p0 = server.M['preemptions']
            jobs.append(submit('vip', vipm, 3, 10))
            for _, ev in jobs:
                assert ev.wait(3600)
            got = [res[k][2] for k in ('low0', 'low1', 'vip')]
            print(f'batched ({time.time() - t0:.0f} s, {server.M["decode_rows"]} rows in {server.M["decode_steps"]} steps):', got,
                  flush=True)
            assert all(res[k][0] == 'done' for k in res) and got == solo, (res, solo)
            low1 = jobs[1][0]
            assert server.M['preemptions'] == p0 + 1 and low1.P == len(low1.ids) > len(low1.prompt), \
                (server.M['preemptions'], low1.P, len(low1.ids), len(low1.prompt))
            assert server.M['decode_rows'] > server.M['decode_steps'], 'never more than one row per step'
            forget_gpu(disk=False)
            assert pool.free_pages() == pool.n - 1, 'pages leaked'
            assert label != 'SoloFused' or (True in moves and False in moves and bd.at is None), moves
            if label == 'spec':
                assert bd.fw > server.M['decode_rows'] - rows0, 'no verify rows ran'
                assert eng.ds is not None and eng.ds.live == 0 and len(eng.ds_free) <= 1, 'drafter States leaked'
            print(f'PASS [{label}] 2 concurrent + VIP-preempted decode == solo (resumed at {low1.P} = prompt {len(low1.prompt)} + '
                  f'{low1.P - len(low1.prompt)} fed, no recompute), no page leak; moves (from the Decoder?) {moves}')
        eng.spec = 'off'

        # spec through generate: greedy == plain (max_tokens, stop string), DSpark and prompt lookup; then prefix
        # restore: turn 2 warm from the parked turn 1 with spec (drafter context starts at P) == turn 2 cold plain
        spec_runner.step_ms = lambda sampled=False: (1.0,) * 8   # always draft, up to k = 7
        spec_runner.VERIFY_MS = (1.0,) * 8
        eng.chunk = len(A)
        code = [{'role': 'user', 'content': 'Write a Python function is_prime(n).'}]

        def gen(msgs, mode, n=10, **kw):
            eng.bd, eng.spec, st = (LoopBatch(m, 2) if mode == 'off' else SeqBatch(m, 2)), mode, {}
            out = ''.join(d[1] for d in eng.generate(msgs, max_tokens=n, temperature=0, effort=0, prefill_text=True,
                                                     stats=st, **kw))
            return out, st
        forget_gpu(disk=False, wipe=True)
        ref, rst = gen(code, 'off')
        for mode in ('dspark', 'lookup'):
            got, st = gen(code, mode)
            print(f'spec {mode}: {got!r} ({st["finish_reason"]}, {eng.bd.fw} forwards for {st["completion_tokens"]} '
                  f'tokens) vs plain {ref!r}', flush=True)
            assert (got, st['finish_reason'], st['completion_tokens']) == (ref, rst['finish_reason'],
                                                                           rst['completion_tokens']), (got, ref)
        cut = ref.strip()[4:6]
        sref, _ = gen(code, 'off', stop=[cut])
        sgot, st = gen(code, 'dspark', stop=[cut])
        assert sgot == sref == ref[:ref.index(cut)] and st['finish_reason'] == 'stop', (sgot, sref, cut)
        print(f'PASS greedy spec (dspark, lookup) == plain through the engine, max_tokens + stop {cut!r}')
        # logit processors with spec (verify rows processed per draft prefix): greedy == plain
        procs = dict(frequency_penalty=0.8, presence_penalty=0.3, logit_bias={eng.tok.encode(':', add_special_tokens=False).ids[0]: -3.0},
                     min_tokens=4)
        pref, pst = gen(code, 'off', **procs)
        pgot, pgs = gen(code, 'dspark', **procs)
        assert (pgot, pgs['completion_tokens']) == (pref, pst['completion_tokens']), (pgot, pref)
        print(f'PASS greedy spec with penalties / logit_bias / min_tokens == plain: {pgot!r} ({eng.bd.fw} forwards)')
        forget_gpu(disk=False, wipe=True)
        cold2, _ = gen(turn2, 'off', n=6)
        forget_gpu(disk=False, wipe=True)
        gen(turn1, 'dspark', n=2)                                 # parks turn 1's prompt
        warm2, st = gen(turn2, 'dspark', n=6)
        assert st['cached_tokens'] >= len(A) and warm2 == cold2, (st['cached_tokens'], warm2, cold2)  # (A + MODEL TEXT)
        assert eng.ds.live == 0, 'drafter State not returned'
        print(f'PASS prefix restore with spec (cached {st["cached_tokens"]}, drafter from P) == cold plain: {warm2!r}')
        forget_gpu(disk=False)
        eng.spec = 'off'

        # startup warm-up (server): runs prefill/park/restore/2-slot decode/json/decode park, then leaves nothing behind
        eng.bd, eng.chunk = LoopBatch(m, 2, dec=True), 16
        pool.flush()                                              # async writes: earlier spills still landing as *.tmp
        files0 = sorted(os.path.join(d, f) for d, _, fs in os.walk(CACHE) for f in fs)
        assert not eng.parked and not pool.nodes and pool.free_pages() == pool.n - 1
        secs = eng.warmup(mix=8)
        pool.flush()
        files1 = sorted(os.path.join(d, f) for d, _, fs in os.walk(CACHE) for f in fs)
        assert not eng.parked and not pool.nodes and pool.free_pages() == pool.n - 1, 'warm-up left cache/pages'
        assert files1 == files0 and pool.dir and not eng.bd.busy(), (files0, files1)
        assert sum(d is not None for d in eng.bd.decs) <= 1, 'more than one warm Decoder kept'
        print(f'PASS warm-up ({secs:.0f} s): no pages, parked entries, prefix-tree nodes or disk files left')

        k1 = eng.keys([5, chat.IMG_TOK, chat.IMG_TOK, 6], [(None, 1, 2, 111)])
        k2 = eng.keys([5, chat.IMG_TOK, chat.IMG_TOK, 6], [(None, 1, 2, 222)])
        assert k1[0] == k2[0] and not torch.equal(k1, k2) and (k1[1:3] < 0).all()
        print('PASS image hash in prefix key')

    if '--json' in sys.argv:
        schema = {'type': 'object', 'properties': {'city': {'type': 'string'}, 'population_millions': {'type': 'number'}},
                  'required': ['city', 'population_millions'], 'additionalProperties': False}
        rf = grammar.compile_format(eng.tok, {'type': 'json_schema', 'json_schema': {'name': 'c', 'schema': schema}})
        t0, st = time.time(), {}
        out = ''.join(d[1] for d in eng.generate(
            [{'role': 'user', 'content': 'Give the capital of France and its rough metro population in millions.'}],
            max_tokens=40, temperature=0, effort=0, prefill_text=True, response_format=rf, stats=st))
        print(f'json_schema output ({time.time() - t0:.0f} s, {st["completion_tokens"]} tokens, {st["finish_reason"]}): {out!r}')
        d = json.loads(out)
        assert d['city'].lower() == 'paris' and isinstance(d['population_millions'], (int, float))
        print('PASS json_schema real weights')
        eng.bd, eng.spec, st2 = SeqBatch(m, 2), 'dspark', {}  # json + spec: drafts cut at the first rejected token
        out2 = ''.join(d[1] for d in eng.generate(
            [{'role': 'user', 'content': 'Give the capital of France and its rough metro population in millions.'}],
            max_tokens=40, temperature=0, effort=0, prefill_text=True, response_format=rf, stats=st2))
        assert out2 == out, (out2, out)
        print(f'PASS json_schema with spec == without ({eng.bd.fw} forwards)')
    print('ALL OK')
finally:
    shutil.rmtree(CACHE, ignore_errors=True)
