"""GPU window only: the batched decoder (decode_batch.BatchDecoder) vs the tuned single-row Fused path, plus speed.
Checks (exit 1 on a gate failure, 2 on an exception):
  1. correctness, B = 1, 2, 4 distinct prompts (3 natural chats + a ~1k prose prompt crossing position 1024):
     reference = fused-graph Decoder per prompt (greedy, N steps); the batched graph decoder runs all B rows
     teacher-forced on those tokens. Gate per B: mean KL(fused || batch) <= 2 x the KL of a second valid path (eager
     unfused Decoder vs fused-graph: the known noise between correct implementations) + 1e-3, top-1 >= 95%.
  2. batch invariance on the GPU: prompt 0's first 8 steps give bit-identical logits at B=1, 2 and 4 (gate), and
     the graph == eager batched decoder (lockstep from the same state each step, gpu_decode.lock_gate; the batched
     kernels have no atomics, so expect exact).
  3. speed: aggregate and per-request tok/s at B = 1, 2, 4, 8 with ~1k contexts and with a 40k context (B forks of
     one prefill), host argmax per step as in real decoding; fused-graph single stream for comparison.
  4. verify step cost: one slot, T = 1, 2, 4, 8 rows (+ commit) at ~1k and 40k context, ms and x vs T=1.
  5. VRAM: static bytes per slot / shared, graph pool growth per R bucket.
Usage (Qwen down, nothing else on the GPU):
  ~/llm/engine/swap_test.sh 30 env INK_MOE=triton INK_DEC_FAR_DOT=1 python tests/gpu_batch.py [--long]
  --long adds 130k context to the speed tables (larger pool)."""
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
import decode as D  # noqa: E402
import decode_batch as DB  # noqa: E402
import model as M  # noqa: E402
from engine import Engine  # noqa: E402
from gpu_check import PROMPTS  # noqa: E402
from gpu_check import ulp_nudge  # noqa: E402
from gpu_decode import clone, lock_gate, q  # noqa: E402
from oracle import prose  # noqa: E402

LONG = '--long' in sys.argv
N = 48            # correctness steps per prompt
SPD = 24          # timed steps per speed point


def fork(m, seq, extra=SPD * 9 + 64):
    """Independent copy of a prefilled Seq with its own pages past seq.pos (full pages shared)."""
    s = M.Seq(m.dev, m.pool.fork(seq.kv, seq.pos))
    s.pos, s.ring = seq.pos, list(seq.ring)
    s.conv = [{k: v.clone() for k, v in d.items()} for d in seq.conv]
    assert m.pool.reserve(s.kv, seq.pos + extra), 'pool full'
    return s


def prefill(m, ids, chunk):
    seq = m.new_seq()
    for a in range(0, len(ids), chunk):
        f = m.forward(m.embed(ids[a:a + chunk]), seq)[-1:]
    return seq, m.logits(f)[0]


def kl(p_lp, q_lp):
    return (p_lp.exp() * (p_lp - q_lp)).sum(-1)


def teacher(dec, seq, first, n, toks=None):
    """Single-row Decoder: greedy (toks None) or teacher-forced; -> (tokens fed, log-probs [n, V])."""
    dec.load(clone(seq))
    t = int(first.argmax()) if toks is None else toks[0]
    fed, lps = [], []
    for i in range(n):
        fed.append(t)
        lg = dec.step(t)[0]
        lps.append(torch.log_softmax(lg.float(), -1))
        t = int(lg.argmax()) if toks is None else (toks[i + 1] if i + 1 < n else 0)
    dec.seq = None
    return fed, torch.stack(lps)


def bd_teacher(bd, seqs, toks, n):
    """All rows of the batch, row i teacher-forced on toks[i]. -> log-probs [B, n, V]."""
    for i, s in enumerate(seqs):
        bd.load(i, s)
    out = []
    for j in range(n):
        lg = bd.step([(i, bd.seqs[i].pos, toks[i][j]) for i in range(len(seqs))])
        out.append(torch.log_softmax(lg.float(), -1))
    for i in range(len(seqs)):
        bd.free(i)
    return torch.stack(out, 1)


def sync(dst, src):
    """dst's first src.n slots := src's state (in place: graphs hold these addresses)."""
    n = src.n
    dst.b.ring[:, :, :n].copy_(src.b.ring), dst.b.rpos[:n].copy_(src.b.rpos), dst.b.conv[:, :n].copy_(src.b.conv)


def speed(bd, seqs, n=SPD):
    """Greedy batched decode of all slots; -> median step seconds."""
    for i, s in enumerate(seqs):
        bd.load(i, s)
    t = [1000 + 7919 * i for i in range(len(seqs))]   # distinct per row: forks of one prompt must not run identical rows
    times = []
    for j in range(n + 2):
        torch.cuda.synchronize()
        t0 = time.time()
        lg = bd.step([(i, bd.seqs[i].pos, t[i]) for i in range(len(seqs))])
        t = lg.argmax(-1).tolist()
        times.append(time.time() - t0)
    for i in range(len(seqs)):
        bd.free(i)
    return statistics.median(times[2:])


def verify_cost(bd, seq, T, n=12):
    bd.load(0, seq)
    times = []
    for j in range(n + 2):
        p = bd.seqs[0].pos
        torch.cuda.synchronize()
        t0 = time.time()
        lg = bd.step([(0, p + i, 1000 + i) for i in range(T)])
        bd.commit({0: 1} if T > 1 else {})
        int(lg[0].argmax())
        times.append(time.time() - t0)
    bd.free(0)
    return statistics.median(times[2:])


def main():
    eng = Engine('cuda', max_ctx=150_000 if LONG else 60_000, vision=False)
    m, ok = eng.model, True
    M.MOE_IMPL = 'triton'
    print(f'loaded: {torch.cuda.memory_allocated() / 2**30:.2f} GiB', flush=True)
    ids = [chat.render(eng.tok, [{'role': 'user', 'content': p}], effort=0.9) for p in PROMPTS[:3]]
    ids.append(chat.render(eng.tok, [{'role': 'user', 'content': prose(1100, eng.tok, 5)}], effort=0.9)[:1000])
    pre = [prefill(m, x, eng.chunk) for x in ids]
    print('prompt tokens', [len(x) for x in ids], flush=True)

    # ---- 1. correctness vs fused-graph
    fg, ed = D.Decoder(m, graph=True, fuse=True), D.Decoder(m, graph=False, fuse=False)
    refs = [teacher(fg, s, f, N) for s, f in pre]
    # noise per prompt: per-step KL vs the reference of 3 other valid paths (fused-graph rerun: MoE atomics; eager
    # unfused; fused after a +-1-ulp nudged prefill). Per-step KL is heavy-tailed (one near-tie step reached 0.3-2.9
    # on prompt 3 for every path, 2026-11-22), so the gate compares median and p95, not the mean.
    base = []
    for r, (s, f), x in zip(refs, pre, ids):
        sq = m.new_seq()
        for a in range(0, len(x), eng.chunk):
            fn = m.forward(ulp_nudge(m.embed(x[a:a + eng.chunk])), sq)[-1:]
        ks = [kl(r[1], teacher(fg, s, f, N, r[0])[1]), kl(r[1], teacher(ed, s, f, N, r[0])[1]),
              kl(r[1], teacher(fg, sq, m.logits(fn)[0], N, r[0])[1])]
        base.append((max(q(k.tolist(), .5) for k in ks), max(q(k.tolist(), .95) for k in ks)))
    print('noise (max over fused rerun / eager unfused / nudged prefill) KL median/p95 per prompt: '
          + ' '.join(f'{a:.1e}/{b:.1e}' for a, b in base))
    m0 = torch.cuda.memory_allocated()
    bd = DB.BatchDecoder(m, 8, graph=True)
    for B in (1, 2, 4):
        lp = bd_teacher(bd, [fork(m, pre[i][0]) for i in range(B)], [refs[i][0] for i in range(B)], N)
        for i in range(B):
            ks = kl(refs[i][1], lp[i]).tolist()
            km, kp = q(ks, .5), q(ks, .95)
            top = (lp[i].argmax(-1) == refs[i][1].argmax(-1)).float().mean().item()
            lm, lpp = 2 * base[i][0] + 1e-4, 2 * base[i][1] + 1e-3
            good = km <= lm and kp <= lpp and top >= 0.95 and bool(torch.isfinite(lp[i]).all())
            ok &= good
            print(f'B={B} row {i}: KL(fused || batch) median {km:.1e} / p95 {kp:.1e} / mean {statistics.mean(ks):.1e} '
                  f'(limits {lm:.1e} / {lpp:.1e}), top-1 {top:.3f}  {"OK" if good else "TOO BIG"}', flush=True)

    # ---- 2. invariance across B and graph == eager
    rows = {}
    for B in (1, 2, 4):
        lp = bd_teacher(bd, [fork(m, pre[i][0]) for i in range(B)], [refs[i][0] for i in range(B)], 8)
        rows[B] = lp[0]
    inv = all(torch.equal(rows[1], rows[B]) for B in (2, 4))
    ok &= inv
    print(f'batch invariance (prompt 0, 8 steps at B=1/2/4 bit-identical): {inv}'
          + ('' if inv else f' max diff {max((rows[1] - rows[B]).abs().max().item() for B in (2, 4)):.2e}'), flush=True)
    be, be2 = DB.BatchDecoder(m, 4, graph=False), DB.BatchDecoder(m, 4, graph=False)
    lrows = []
    for d in (bd, be, be2):
        for i in range(4):
            d.load(i, fork(m, pre[i][0]))
    for j in range(16):
        sync(bd, be), sync(be2, be)
        r = [(i, be.seqs[i].pos, refs[i][0][j]) for i in range(4)]
        g, e2, e = bd.step(r).clone(), be2.step(r).clone(), be.step(r).clone()
        rel = lambda x: ((x - e).abs().max() / e.abs().max()).item()
        lrows.append((j, r[0][1], 4, rel(g), rel(e2), bool((g.argmax(-1) == e.argmax(-1)).all())))
    for d in (bd, be, be2):
        for i in range(4):
            d.free(i)
    ok &= lock_gate('batched graph vs eager (B=4)', lrows)
    print(f'  graph == eager bit-exact on {sum(r[3] == 0 for r in lrows)}/{len(lrows)} steps')
    be.drop(), be2.drop()
    del be, be2

    # ---- 3. speed
    table = []
    teacher(fg, pre[0][0], pre[0][1], SPD)                     # warm-up (graph capture)
    t1 = time.time()
    teacher(fg, pre[0][0], pre[0][1], SPD)
    table.append(('~1k', 'fused-graph', 1, SPD / (time.time() - t1)))
    long = [('40k', chat.render(eng.tok, [{'role': 'user', 'content': prose(40000, eng.tok, 2)}], effort=0.9))]
    if LONG:
        long.append(('130k', chat.render(eng.tok, [{'role': 'user', 'content': prose(130000, eng.tok, 3)}], effort=0.9)))
    ctxs = [('~1k', [s for s, _ in pre])]
    for name, x in long:
        t0 = time.time()
        s, _ = prefill(m, x, eng.chunk)
        print(f'{name}: {len(x)} tokens prefilled in {time.time() - t0:.0f} s', flush=True)
        ctxs.append((name, [s]))
    for name, srcs in ctxs:
        for B in (1, 2, 4, 8):
            dt = speed(bd, [fork(m, srcs[i % len(srcs)]) for i in range(B)])
            table.append((name, 'batch', B, B / dt))
            print(f'{name:5s} B={B}: step {dt * 1e3:6.2f} ms, aggregate {B / dt:6.1f} tok/s, per request {1 / dt:6.1f}',
                  flush=True)
        t_1 = None
        for T in (1, 2, 4, 8):
            dt = verify_cost(bd, fork(m, srcs[0]), T)
            t_1 = t_1 or dt
            print(f'{name:5s} verify T={T}: {dt * 1e3:6.2f} ms (x{dt / t_1:.2f} of T=1)', flush=True)

    # ---- 5. memory
    per, shared = bd.nbytes()
    print(f'VRAM: {per / 2**20:.0f} MiB static per slot, {shared / 2**20:.0f} MiB shared scratch (8 slots: '
          f'{(per * 8 + shared) / 2**30:.2f} GiB); batched decoder + graphs {sorted(bd.graphs)} allocated '
          f'{(torch.cuda.memory_allocated() - m0) / 2**30:.2f} GiB, reserved {torch.cuda.memory_reserved() / 2**30:.2f} GiB',
          flush=True)
    print('\n== summary (tok/s)')
    for name, path, B, tps in table:
        print(f'{name:5s} {path:12s} B={B}  {tps:7.1f} aggregate  {tps / B:7.1f} per request')
    print('PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 2
    sys.exit(rc)
