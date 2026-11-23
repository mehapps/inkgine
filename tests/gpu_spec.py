"""GPU window only: speculative decoding with the DSpark drafter (spec_runner.py) on the BatchDecoder, one sequence.
Per prompt (code / json (json_schema grammar) / chat / prose; thinking off, reply as a text message):
  1. greedy: plain BatchDecoder decoding (1 row per step) vs spec; gate: identical tokens (verify rows are
     batch-invariant, so this is exact; exit 1 otherwise). tok/s both ways, acceptance per draft position, mean
     accepted per round, drafter ms per round, chosen widths; + spec with drafting off (kmax=0) = the cost of k = 0
     rounds (now plain steps + one drafter catch-up pass per 8 positions from bd.tap_hist; target x1.00 of plain;
     same tokens gated too).
  2. temperature 0.7 and 1.0: plain vs spec tok/s and acceptance (outputs differ by design; both are exact samples).
     Sampled drafter graphs are captured in the warm-up (before 11-23 the first sampled run captured them inside its
     timing: the "6.5 ms" sampled drafter rounds of the 11-22 window were mostly that).
  3. drafter alone: ms per round (graph replay) for k = 7 greedy / sampled, k = 3, at the current context, eager vs
     graph; context-only flush of 8 buffered rows; k = 7 at 8k / 30k context (random taps: timing only).
  4. round-cost breakdown per run: wall ms per drafting round = drafter + verify/accept (settle waits for the verify)
     + the rest (host); verify step ms at T = 1..8 rows alone.
Flags of the new round-cost work are read from the environment (dspark.py / spec_runner.py: INK_DSPARK_SHORT,
INK_DSPARK_CTXB, INK_SPEC_GATE, INK_SPEC_ACCEPT_DEV, INK_SPEC_HYBRID); tests/window_spec2.sh / window_spec4.sh run this
once per combination. --agent adds agent-loop cases on a ~3k-token source file (spec.py's head): `tool` (an edit_file
call whose old_string copies from a read_file result), `edit` (a rewritten function copying the provided file),
`reason` (reasoning text about the file, thinking on). --temps 0,0.7 picks the temperatures (default 0,0.7,1.0).
Prefill taps come from spec_runner.forward_taps (Model.layer wrapped for the call). Exit 2 on an exception.
Usage (Qwen down, nothing else on the GPU):
  ~/llm/engine/swap_test.sh 25 env INK_MOE=triton INK_DEC_FAR_DOT=1 INK_GRAPH=1 python tests/gpu_spec.py [--n 256]
"""
import os
import statistics
import sys
import time
import traceback

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import torch  # noqa: E402

import chat  # noqa: E402
import decode_batch as DB  # noqa: E402
import dspark as D  # noqa: E402
import grammar  # noqa: E402
import model as M  # noqa: E402
import spec_runner as S  # noqa: E402
from decode import sample  # noqa: E402
from engine import Engine  # noqa: E402
from gpu_check import PROMPTS  # noqa: E402

N = int(sys.argv[sys.argv.index('--n') + 1]) if '--n' in sys.argv else 256
CODE = ('Write a Python module with a class LRUCache (get, put, capacity eviction, type hints, docstrings) and '
        'pytest unit tests for it.')
JSON = ('Give me a JSON record for a fictional bookstore with name, address, opening hours per weekday and a list of '
        'ten books (title, author, year, price, in_stock).')
SCHEMA = {'type': 'object', 'required': ['name', 'address', 'hours', 'books'],
          'properties': {'name': {'type': 'string'}, 'address': {'type': 'string'},
                         'hours': {'type': 'object', 'additionalProperties': {'type': 'string'}},
                         'books': {'type': 'array', 'items': {
                             'type': 'object', 'required': ['title', 'author', 'year', 'price', 'in_stock'],
                             'properties': {'title': {'type': 'string'}, 'author': {'type': 'string'},
                                            'year': {'type': 'integer'}, 'price': {'type': 'number'},
                                            'in_stock': {'type': 'boolean'}}}}}}
TEMPS = tuple(float(x) for x in sys.argv[sys.argv.index('--temps') + 1].split(',')) if '--temps' in sys.argv else \
    (0.0, 0.7, 1.0)
text_case = lambda text: lambda tok: chat.render(tok, [{'role': 'user', 'content': text}], effort=0) + [chat.MODEL,
                                                                                                       chat.TEXT]
CASES = [('code', text_case(CODE), None), ('json', text_case(JSON), SCHEMA), ('chat', text_case(PROMPTS[1]), None),
         ('prose', text_case(PROMPTS[0]), None)]
STOP = (chat.EOS, chat.END)
SRC = open(os.path.join(os.path.dirname(HERE), 'spec.py')).read()[:12000]      # ~3k tokens of our own source
TOOLS = [{'type': 'function', 'function': {
    'name': 'edit_file', 'description': 'Replace the exact text old_string in a file with new_string.',
    'parameters': {'type': 'object', 'required': ['path', 'old_string', 'new_string'], 'properties': {
        'path': {'type': 'string'}, 'old_string': {'type': 'string'}, 'new_string': {'type': 'string'}}}}},
    {'type': 'function', 'function': {
        'name': 'read_file', 'description': 'Read a file.',
        'parameters': {'type': 'object', 'required': ['path'], 'properties': {'path': {'type': 'string'}}}}}]
SYS = 'You are a coding agent working in a Python repository. Use the tools to make the requested change.'


def agent_tool(tok):
    msgs = [{'role': 'system', 'content': SYS},
            {'role': 'user', 'content': 'In engine/spec.py, make Lookup.draft return an empty list when k is negative '
                                        'or zero before computing anything else, and add a docstring line saying so. '
                                        'Use one edit_file call that replaces the whole draft method.'},
            {'role': 'assistant', 'content': '', 'tool_calls': [
                {'id': 'c1', 'function': {'name': 'read_file', 'arguments': '{"path": "engine/spec.py"}'}}]},
            {'role': 'tool', 'tool_call_id': 'c1', 'content': SRC}]
    return chat.render(tok, msgs, tools=TOOLS, effort=0) + [chat.MODEL]


def agent_edit(tok):
    q = ('Here is a file:\n```python\n' + SRC + '\n```\nRewrite the method Lookup.find so that it also returns the '
         'number of candidate occurrences. Reply with the complete updated class Lookup, unchanged lines included.')
    return chat.render(tok, [{'role': 'user', 'content': q}], effort=0) + [chat.MODEL, chat.TEXT]


def agent_reason(tok):
    q = ('```python\n' + SRC + '\n```\nCan Width.width() ever return a width larger than kmax? Think it through '
         'carefully, then answer.')
    return chat.render(tok, [{'role': 'user', 'content': q}], effort=0.7) + [chat.MODEL]


if '--agent' in sys.argv:
    CASES += [('tool', agent_tool, None), ('edit', agent_edit, None), ('reason', agent_reason, None)]


def fork(m, seq, extra):
    s = M.Seq(m.dev, m.pool.fork(seq.kv, seq.pos))
    s.pos, s.ring = seq.pos, list(seq.ring)
    s.conv = [{k: v.clone() for k, v in d.items()} for d in seq.conv]
    assert m.pool.reserve(s.kv, seq.pos + extra), 'pool full'
    return s


def prefill(m, ids, chunk):
    seq, tps = m.new_seq(), []
    for a in range(0, len(ids), chunk):
        f, tp = S.forward_taps(m, m.embed(ids[a:a + chunk]), seq)
        tps.append(tp)
    return seq, m.logits(f[-1:])[0], torch.cat(tps, 1), list(ids)


def run(eng, bd, ds, st, pre, schema, temperature, spec, seed=0, kmax=7):
    """-> (tokens, seconds, Spec or None). The first token is sampled from the prompt's logits outside the timing."""
    m = eng.model
    seq, lg0, taps, ids = pre
    cons = None
    if schema is not None:
        cons = grammar.Json(grammar.compile_format(eng.tok, {'type': 'json_schema', 'json_schema': {'schema': schema}}),
                            True)
    gen = torch.Generator(device=m.dev).manual_seed(seed)
    lg = lg0.clone()
    if cons is not None:
        cons.mask(lg)
    first = int(sample(lg, temperature, generator=gen))
    s = fork(m, seq, N + 16)
    bd.load(0, s)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out, sp = S.generate(bd, 0, s.pos, first, N, ds=ds if spec else None, taps=taps if spec else None,
                         temperature=temperature, gen=gen, cons=cons, stop=STOP, st=st, kmax=kmax, prompt=ids)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    bd.free(0)
    m.pool.release(s.kv)
    return out, dt, sp


def fmt_acc(sp):
    s = sp.stats
    nt = max([i + 1 for i, b in enumerate(s['tried']) if b] or [7])
    per = ' '.join(f'{a / b:.2f}' if b else '  - ' for a, b in zip(s['kept'][:nt], s['tried'][:nt]))
    r = max(1, s['rounds'])
    return (f'rounds {s["rounds"]}, drafted {s["drafted"]}, accepted {s["accepted"]} '
            f'({s["accepted"] / max(1, s["drafted"]):.2f}), tokens/round {s["emitted"] / max(1, s["rounds"]):.2f}, '
            f'drafter {1e3 * s["draft_s"] / max(1, s["rounds"]):.2f} ms/round, verify {1e3 * s["step_s"] / max(1, s["rounds"]):.2f} '
            f'ms/round, settle (verify wait + accept) {1e3 * s["settle_s"] / r:.2f} ms/round | per position {per} | '
            f'widths k=0..7 {s["widths"]}' + (f' | drafter ms (measured EMA) {sp.dms:.2f}' if sp.dms else '')
            + (f' | copy rounds {s.get("copy_rounds", 0)}, DSpark+copy rounds {s.get("ext_rounds", 0)}, copy c '
               + ' '.join(f'{c:.2f}' for c in sp.cpr.c) if sp.hy is not None else ''))


def verify_ms(m, bd, pre, n=20):
    """Verify step alone: ms of bd.step with T = 1..16 rows of one slot (synced), committing 1 row each time."""
    seq = pre[0]
    s = fork(m, seq, 16 * (n + 3) + 32)
    bd.load(0, s)
    out, pos = [], s.pos
    for T in range(1, 17):
        ts = []
        for i in range(n + 3):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            bd.step([(0, pos + j, 1000 + j) for j in range(T)])
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
            if T > 1:
                bd.commit({0: 1})
            pos += 1
        out.append(1e3 * statistics.median(ts[3:]))
    bd.free(0)
    m.pool.release(s.kv)
    return out


def drafter_ms(ds, st, k, temperature, graph, n=30):
    ds.graph = graph
    for _ in range(3):
        ds.round(st, 1000, k, temperature)                    # (first use captures)
    ts = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ds.round(st, 1000, k, temperature)
        ts.append(time.perf_counter() - t0)
    return 1e3 * statistics.median(ts)


def flush_ms(ds, st, taps, n=30):
    """Context-only drafter pass for 8 buffered rows (what k = 0 rounds cost the drafter, once per 8 positions)."""
    ts = []
    for i in range(n + 3):
        ds.stage(st, taps[:, -8:])
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ds.flush(st)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return 1e3 * statistics.median(ts[3:])


def main():
    eng = Engine('cuda', max_ctx=32768, vision=False)
    m, ok = eng.model, True
    M.MOE_IMPL = 'triton'
    u8 = m.d8.un if getattr(m, 'd8', None) is not None and m.d8.dec and m.d8.un is not None else None
    ds = D.DSpark(embed=m.W.embed, unembed=None if u8 is not None else m.W['unembed.weight'][:M.VOCAB],
                  embed_norm=m.W['embed_norm.weight'], device='cuda', graph=True, unembed8=u8)
    print(f'drafter unembed: {"FP8 (INK_FP8_DECODE), shortlist" if u8 is not None else "bf16"}, shortlist {ds.short}')
    st = ds.new_state()
    print(f'loaded target + drafter: {torch.cuda.memory_allocated() / 2**30:.2f} GiB (drafter ctx cap {D.CTX})',
          flush=True)
    bd = DB.BatchDecoder(m, 2, graph=True)
    rows = []
    for name, build, schema in CASES:
        ids = build(eng.tok)
        pre = prefill(m, ids, eng.chunk)
        print(f'\n== {name}: prompt {len(ids)} tokens', flush=True)
        run(eng, bd, ds, st, pre, schema, 0.0, False)          # warm-up: graph buckets 1/2/4/8 captured
        run(eng, bd, ds, st, pre, schema, 0.0, True)           # (drafter graphs per k, verify buckets)
        run(eng, bd, ds, st, pre, schema, 0.7, True, seed=9)   # (sampled drafter graphs)
        run(eng, bd, ds, st, pre, schema, 0.0, True, kmax=0)   # (context-only flush graph)
        for temp in TEMPS:
            p_out, p_dt, _ = run(eng, bd, ds, st, pre, schema, temp, False, seed=1)
            s_out, s_dt, sp = run(eng, bd, ds, st, pre, schema, temp, True, seed=1)
            same = p_out == s_out
            if temp == 0:
                ok &= same
            nd = next((i for i, (a, b) in enumerate(zip(p_out, s_out)) if a != b), min(len(p_out), len(s_out)))
            ptps, stps = len(p_out) / p_dt, len(s_out) / s_dt
            rows.append((name, temp, ptps, stps, sp, s_dt))
            print(f'T={temp}: plain {len(p_out)} tok {ptps:6.1f} tok/s | spec {len(s_out)} tok {stps:6.1f} tok/s '
                  f'(x{stps / ptps:.2f}) | outputs ' + ('IDENTICAL' if same else f'differ from token {nd}')
                  + ('' if temp or same else '  <-- GATE FAIL'), flush=True)
            print(f'   {fmt_acc(sp)}', flush=True)
            if temp == 0:
                print(f'   {eng.tok.decode([t for t in s_out if t < chat.USER])[:160]!r}', flush=True)
                z_out, z_dt, _ = run(eng, bd, ds, st, pre, schema, 0.0, True, seed=1, kmax=0)
                ok &= z_out == p_out
                print(f'   drafting off (kmax=0: k=0 rounds only, taps buffered): {len(z_out) / z_dt:6.1f} tok/s '
                      f'(x{len(z_out) / z_dt / ptps:.3f} of plain) | outputs '
                      + ('IDENTICAL' if z_out == p_out else 'DIFFER  <-- GATE FAIL'), flush=True)
        if name == 'code':                                     # drafter alone at this context
            seq, _, taps, _ = pre
            ds.reset(st)
            ds.extend(st, taps)
            for k, temp in ((7, 0.0), (7, 0.7), (3, 0.0)):
                e, g = drafter_ms(ds, st, k, temp, False), drafter_ms(ds, st, k, temp, True)
                print(f'drafter round k={k} T={temp}: eager {e:.2f} ms, graph {g:.2f} ms', flush=True)
            print(f'drafter context-only flush of 8 rows (graph): {flush_ms(ds, st, taps):.3f} ms', flush=True)
            for n in (8192, 30000):                            # longer contexts (random taps: timing only)
                ds.reset(st)
                ds.extend(st, torch.randn(4, n, M.H, device='cuda', dtype=torch.bfloat16))
                print(f'drafter round k=7 at context {n}: greedy {drafter_ms(ds, st, 7, 0.0, True):.2f} ms, '
                      f'T=0.7 {drafter_ms(ds, st, 7, 0.7, True):.2f} ms', flush=True)
            ds.graph = True
            vm = verify_ms(m, bd, pre)
            print('verify step alone, T=1..16 rows: ' + ' '.join(f'{x:.2f}' for x in vm) + ' ms (model: '
                  + ' '.join(f'{x:.2f}' for x in S.verify16(S.VERIFY_MS)) + ')', flush=True)
    flags = {k: v for k, v in os.environ.items() if k.startswith(('INK_DSPARK', 'INK_SPEC'))}
    print(f'\n== summary (tok/s, one sequence, BatchDecoder) flags {flags}')
    for name, temp, a, b, sp, wall in rows:
        s = sp.stats
        r = max(1, s['rounds'])
        dr, se = 1e3 * s['draft_s'] / r, 1e3 * s['settle_s'] / r
        print(f'{name:5s} T={temp}: plain {a:6.1f}  spec {b:6.1f}  x{b / a:.2f}  accepted/round '
              f'{s["accepted"] / r:.2f}  tokens/round {s["emitted"] / r:.2f}  drafting rounds {s["rounds"]} / '
              f'{s["rounds"] + s["widths"][0]} | round ms: drafter {dr:.2f} + verify wait/accept {se:.2f}'
              + (f' | copy rounds {s.get("copy_rounds", 0)} + mixed {s.get("ext_rounds", 0)}' if sp.hy is not None else ''))
    print(f'VRAM peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB')
    print('PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 2
    sys.exit(rc)
