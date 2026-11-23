"""GPU window only: decode_batch.BatchDecoder tuning + KL noise study.
  --kl     prompt 3 of gpu_batch (1000-token prose, crosses 1024): fused-graph reference vs several noise paths
           (fused rerun, fused after a +-1-ulp nudged prefill x3 seeds, eager unfused) and vs the batched decoder
           (B=1 and B=4 row 3); per-step KL median / p95 / max / mean, top-1, KL by step quarter (drift check).
  default  per-kernel micro-benchmark of one batched step (tests/tune_decode.py pattern: record the step's Triton
           launches, time each kernel's launches as a CUDA graph) at R = 1, 2, 4, 8, 16 rows (~1.1k context, distinct
           tokens per row so the expert unions are realistic); whole step (graph) ms per R.
  --sweep  BB/NWB variants of the GEMM kernels (proj, wo, w13u, w2u) x num_warps x num_stages, timed per R, each
           checked against the current variant's outputs on identical inputs (max rel diff; 0 = bit-exact).
Usage: INK_MOE=triton INK_DEC_FAR_DOT=1 python tests/tune_batch.py [--kl] [--sweep] [--only proj,wo]"""
import itertools
import os
import statistics
import sys

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('TRITON_DEFAULT_FP_FUSION', '1')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import torch  # noqa: E402

import chat  # noqa: E402
import decode as D  # noqa: E402
import decode_batch as DB  # noqa: E402
import model as M  # noqa: E402
from gpu_batch import bd_teacher, fork, kl, prefill, teacher  # noqa: E402
from gpu_check import ulp_nudge  # noqa: E402
from gpu_decode import q  # noqa: E402
from oracle import prose  # noqa: E402
from tune_decode import outputs, rel, state_args, time_graph  # noqa: E402

RS = (1, 2, 4, 8, 16)
H, NI = M.H, DB.NI
# kernel -> (constexpr candidates, BB key per constexpr, grid(recorded grid, constexprs))
TUNE = {
    'bproj': (dict(BN=[16, 32, 64]), dict(BN='proj'), lambda g, a, c: (triton.cdiv(a[3], c['BN']),)),
    'bwo': (dict(BN=[16, 32, 64]), dict(BN='wo'), lambda g, a, c: (H // c['BN'],)),
    'w13u': (dict(BN=[16, 32, 64], BK=[64, 128, 256]), dict(BN='w13', BK='kbk'), lambda g, a, c: (NI // c['BN'], g[1])),
    'w2u': (dict(BN=[16, 32, 64], BK=[64, 128, 256]), dict(BN='w2', BK='kbk2'), lambda g, a, c: (H // c['BN'], g[1])),
}
META = dict(num_warps=[2, 4, 8], num_stages=[1, 2, 3])
import triton  # noqa: E402


def kl_study(eng):
    m = eng.model
    ids = chat.render(eng.tok, [{'role': 'user', 'content': prose(1100, eng.tok, 5)}], effort=0.9)[:1000]
    seq, first = prefill(m, ids, eng.chunk)
    N = 48
    fg, ed = D.Decoder(m, graph=True, fuse=True), D.Decoder(m, graph=False, fuse=False)
    toks, ref = teacher(fg, seq, first, N)
    paths = {'fused rerun': teacher(fg, seq, first, N, toks)[1], 'eager unfused': teacher(ed, seq, first, N, toks)[1]}
    for s in (1, 2, 3):                                          # same math, +-1 ulp on the prompt embeddings
        sq = m.new_seq()
        for a in range(0, len(ids), eng.chunk):
            f = m.forward(ulp_nudge(m.embed(ids[a:a + eng.chunk]), s), sq)[-1:]
        paths[f'fused nudged prefill s{s}'] = teacher(fg, sq, m.logits(f)[0], N, toks)[1]
    bd = DB.BatchDecoder(m, 4, graph=True)
    paths['batch B=1'] = bd_teacher(bd, [fork(m, seq)], [toks], N)[0]
    others = [chat.render(eng.tok, [{'role': 'user', 'content': p}], effort=0.9) for p in ('Hi', 'What is 2+2?', 'Name a color.')]
    pre = [prefill(m, x, eng.chunk) for x in others]
    otoks = [teacher(fg, s, f, N)[0] for s, f in pre]
    paths['batch B=4 row 3'] = bd_teacher(bd, [fork(m, s) for s, _ in pre] + [fork(m, seq)], otoks + [toks], N)[3]
    print(f'prompt 3 ({len(ids)} tokens, steps at pos {seq.pos}..{seq.pos + N - 1}): KL(fused-graph || path) per step')
    per = {}
    for name, lp in paths.items():
        k = kl(ref, lp).tolist()
        per[name] = k
        top = (lp.argmax(-1) == ref.argmax(-1)).float().mean().item()
        qs = [statistics.mean(k[i:i + N // 4]) for i in range(0, N, N // 4)]
        print(f'  {name:26s} mean {statistics.mean(k):.2e}  median {q(k, .5):.2e}  p95 {q(k, .95):.2e}  max '
              f'{max(k):.2e} (step {k.index(max(k))})  top-1 {top:.3f}  by quarter ' + ' '.join(f'{x:.1e}' for x in qs),
              flush=True)
    print('  per step (fused rerun / nudged s1 / eager unfused / batch B=4 row 3):')
    for i in range(N):
        print(f'   {i:2d} ' + ' '.join(f'{per[n][i]:.1e}' for n in ('fused rerun', 'fused nudged prefill s1',
                                                               'eager unfused', 'batch B=4 row 3')))


def record(bd, rows):
    """One eager batched step, recording every decode_batch launch (name, kernel, grid, args, kwargs)."""
    recs, saved = [], []
    for n, k in list(vars(DB).items()):
        if n.endswith('_kernel') and hasattr(k, 'run') and (n.startswith('_b') or n in ('_plan_kernel', '_w13u_kernel',
                                                                                        '_w2u_kernel', '_commit_kernel')):
            class L:
                def __init__(self, k, n):
                    self.k, self.n = k, n

                def __getitem__(self, grid):
                    def run(*a, **kw):
                        recs.append((self.n.strip('_').replace('_kernel', ''), self.k, grid, a, kw))
                        self.k[grid](*a, **kw)
                    return run
            saved.append((n, k))
            setattr(DB, n, L(k, n))
    try:
        bd.step(rows)
        torch.cuda.synchronize()
    finally:
        for n, k in saved:
            setattr(DB, n, k)
    return recs


def step_ms(bd, rows, n=30):
    ts = []
    for _ in range(n + 3):
        for s, p, _ in rows:          # rewind: the same step again (single rows commit inside step)
            bd.seqs[s].pos = p
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        bd.step(rows)
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    for s, p, _ in rows:
        bd.seqs[s].pos = p
    return statistics.median(ts[3:])


def bench(eng, sweep, only):
    m = eng.model
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(1000, 100000, (1100,), generator=g).tolist()
    seq, _ = prefill(m, ids, eng.chunk)
    bd = DB.BatchDecoder(m, 16, graph=True)
    for i in range(16):
        bd.load(i, fork(m, seq))
    P = seq.pos
    tok = torch.randint(1000, 100000, (16,), generator=g).tolist()
    recs = {}
    for R in RS:
        rows = [(i, P, tok[i]) for i in range(R)]
        bd.graph = False
        bd.step(rows)                                            # compile
        for s, _, _ in rows:
            bd.seqs[s].pos = P
        recs[R] = record(bd, rows)
        for s, _, _ in rows:
            bd.seqs[s].pos = P
        bd.graph = True
        print(f'R={R:2d}: whole step (graph) {step_ms(bd, rows):6.3f} ms, union experts/visit (last) '
              f'{int(bd.b.ucnt.item())}', flush=True)
    names = sorted({r[0] for r in recs[1]}, key=lambda x: [r[0] for r in recs[1]].index(x))
    print(f'{"kernel":11s} ' + ' '.join(f'R={R:<5d}' for R in RS) + '  (ms per step, CUDA-graph replay of its launches)')
    for nm in names:
        print(f'{nm:11s} ' + ' '.join(f'{time_graph([r for r in recs[R] if r[0] == nm]):7.3f}' for R in RS), flush=True)
    print(f'{"sum":11s} ' + ' '.join(f'{time_graph(recs[R]):7.3f}' for R in RS) + '  (all Triton launches, one graph)')
    if not sweep:
        return
    for nm, (space, bbk, gridf) in TUNE.items():
        if only and nm not in only:
            continue
        rsets = {R: [r for r in recs[R] if r[0] == nm] for R in RS}
        refs = {}
        for R in (1, 16):
            ts = state_args(rsets[R])
            snap = [t.clone() for t in ts]
            refs[R] = (ts, snap, outputs(rsets[R], ts, snap))
        cur = {c: DB.BB[k] for c, k in bbk.items()}
        cur.update(num_warps=DB.NWB[bbk['BN']], num_stages=None)
        res = []
        keys = list(space) + list(META)
        for vals in itertools.product(*(list(space.values()) + list(META.values()))):
            over = dict(zip(keys, vals))
            try:
                ms = [time_graph(rsets[R], over, gridf) for R in RS]
                err = max(rel(outputs(rsets[R], ts, snap, over, gridf), ref) for R, (ts, snap, ref) in refs.items())
            except Exception as e:  # noqa: BLE001
                print(f'  {nm} {over}: skipped ({str(e).splitlines()[0][:80] if str(e) else type(e).__name__})')
                torch.cuda.synchronize()
                continue
            res.append((sum(ms), ms, over, err))
        res.sort(key=lambda x: x[0])
        print(f'\n{nm}: current {cur}: ' + ' '.join(f'{time_graph(rsets[R]):.3f}' for R in RS) + ' ms at R=' +
              '/'.join(map(str, RS)) + '; best (by sum over R):', flush=True)
        for s, ms, over, err in res[:8]:
            print('  ' + ' '.join(f'{x:6.3f}' for x in ms) + f'  {over}  max rel vs current {err:.1e}', flush=True)
        best1 = min(res, key=lambda x: x[1][0])
        print(f'  best at R=1: {best1[1][0]:.3f} ms {best1[2]} rel {best1[3]:.1e}', flush=True)


def main():
    from engine import Engine
    eng = Engine('cuda', max_ctx=60_000, vision=False)
    M.MOE_IMPL = 'triton'
    if '--kl' in sys.argv:
        return kl_study(eng)
    only = set(sys.argv[sys.argv.index('--only') + 1].split(',')) if '--only' in sys.argv else None
    bench(eng, '--sweep' in sys.argv, only)


if __name__ == '__main__':
    main()
