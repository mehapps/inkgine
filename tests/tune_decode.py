"""GPU window only: micro-benchmark of the fused decode kernels (decode_kernels.py) on real weights and shapes.
Records every launch of one fused step (prompt of N tokens, default 1100: ring full, far active), then per kernel
replays that step's launches (all visits) inside a CUDA graph and times the graph with CUDA events (median of
REPS replays). Reports ms per step, bytes moved per step (weights/ring reads; small vectors ignored), GB/s and % of
PEAK. --sweep: for the tunable kernels, every block size / num_warps / num_stages variant (grid recomputed), each
checked against the default variant's outputs on the same inputs (state args snapshot + restored; max rel diff
printed; out-of-resources / shared-memory > 101376 B variants are skipped). --only k1,k2 limits kernels.
--ab: the tuned/split kernels vs the previous ones (old block sizes, the one-program-per-head near kernel, the
unsplit w2) on identical inputs, per visit.
Usage: INK_MOE=triton python tests/tune_decode.py [N] [--ab] [--sweep] [--only w2,w13]"""
import itertools
import os
import statistics
import sys

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('TRITON_DEFAULT_FP_FUSION', '1')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

import decode as D  # noqa: E402
import decode_kernels as K  # noqa: E402
import model as M  # noqa: E402

PEAK = 1792e9        # RTX 5090 DRAM bandwidth, B/s
REPS = 30
H, NI = 2304, K.NI
SMEM = 101376


def wbytes(name, a, grid):
    """Bytes a launch must read from DRAM (weights, ring); vectors ignored."""
    nb = lambda t: t.numel() * t.element_size()
    per_e = lambda w, s: 8 * (w.stride(0) * w.element_size() + s.stride(0) * s.element_size())
    return {'proj': lambda: nb(a[0]), 'wo': lambda: nb(a[1]), 'router': lambda: nb(a[3]),
            'w13': lambda: per_e(a[3], a[4]), 'w2': lambda: per_e(a[3], a[4]), 'w2all': lambda: per_e(a[3], a[4]),
            'far_dot': lambda: max(int(a[6].item()) - 1023, 0) * 1152, 'far': lambda: max(int(a[6].item()) - 1023, 0) * 1152,
            'far3': lambda: max(int(a[6].item()) - 1023, 0) * 1152,
            'near_part': lambda: nb(a[3]) + nb(a[2]), 'near_out': lambda: nb(a[4]) + nb(a[5]) + nb(a[2]),
            'gate_z': lambda: grid[0] * 64 * 2 * H * 2, 'gate_out': lambda: grid[0] * 2 * H * 64 * 4,
            }.get(name, lambda: 0)()


# tunable kernels: constexpr -> candidate values, and the grid as a function of (recorded grid, constexprs)
TUNE = {
    'proj': (dict(BN=[2, 4, 8]), lambda g, a, c: ((a[1] + 2 * a[2]) // c['BN'],)),
    'wo': (dict(BN=[4, 8, 16, 32]), lambda g, a, c: (H // c['BN'],)),
    'router': (dict(RB=[2, 4, 8, 16]), lambda g, a, c: (128 // c['RB'],)),
    'w13': (dict(BN=[8, 16, 32, 64], BK=[64, 128, 256]), lambda g, a, c: (NI // c['BN'], 8)),
    'w2': (dict(BN=[4, 8, 16, 32, 64], BK=[32, 64, 128]), lambda g, a, c: (H // c['BN'], 8)),
    'w2_fin': (dict(BN=[64, 128, 256, 512]), lambda g, a, c: (H // c['BN'],)),
    'gate_z': (dict(ZB=[2, 4, 8, 16]), lambda g, a, c: (g[0], 64 // c['ZB'])),
    'gate_out': (dict(BH=[32, 64, 128, 256]), lambda g, a, c: (g[0], H // c['BH'])),
    'near_part': (dict(), lambda g, a, c: g),
    'near_out': (dict(), lambda g, a, c: g),
    'near_fin': (dict(), lambda g, a, c: g),
    'w2all': (dict(BN=[2, 4, 8, 16], BK=[64, 128]), lambda g, a, c: (H // c['BN'],)),
    'far_dot': (dict(BLK=[64, 128]), lambda g, a, c: g),
}
# the block sizes before tuning (2026-11-22 morning): --ab checks the tuned kernels against them on identical inputs
OLD = {'wo': dict(BN=16, num_warps=4), 'router': dict(RB=8, num_warps=4), 'w13': dict(BN=32, BK=128, num_warps=4),
       'gate_z': dict(ZB=8, num_warps=4), 'gate_out': dict(BH=128, num_warps=4)}
META = dict(num_warps=[1, 2, 4, 8], num_stages=[1, 2, 3])


def record(dec, tok):
    """Run one eager fused step, recording (name, kernel, grid, args, kwargs) of every Triton launch."""
    recs, saved = [], []
    for mod, n, k in [(mod, n, k) for mod in (K, D) for n, k in list(vars(mod).items())]:
        if n.endswith('_kernel') and hasattr(k, 'run'):
            class L:
                def __init__(self, k, n):
                    self.k, self.n = k, n

                def __getitem__(self, grid):
                    def run(*a, **kw):
                        recs.append((self.n.strip('_').replace('_kernel', ''), self.k, grid, a, kw))
                        self.k[grid](*a, **kw)
                    return run
            saved.append((mod, n, k))
            setattr(mod, n, L(k, n))
    try:
        dec.step(tok)
        torch.cuda.synchronize()
    finally:
        for mod, n, k in saved:
            setattr(mod, n, k)
    return recs


def launch_all(rs, over=None, gridf=None):
    for name, k, g, a, kw in rs:
        c = dict(kw, **(over or {}))
        k[gridf(g, a, c) if gridf else g](*a, **c)


def time_graph(rs, over=None, gridf=None):
    launch_all(rs, over, gridf)                    # compile + warm
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr, stream=s):
        launch_all(rs, over, gridf)
    torch.cuda.synchronize()
    ts = []
    for _ in range(REPS):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        gr.replay()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    del gr
    return statistics.median(ts)


def state_args(rs):
    """Tensor args small enough to snapshot (outputs, states; weights are never written)."""
    seen = {}
    for _, _, _, a, _ in rs:
        for t in a:
            if isinstance(t, torch.Tensor) and t.numel() * t.element_size() < 4 << 20:
                seen[t.data_ptr(), t.numel()] = t
    return list(seen.values())


def outputs(rs, ts, snap, over=None, gridf=None):
    """Run the launches once from the state snap (clones of ts) and return all state tensors after; restore snap."""
    for t, s in zip(ts, snap):
        t.copy_(s)
    launch_all(rs, over, gridf)
    torch.cuda.synchronize()
    out = [t.clone() for t in ts]
    for t, s in zip(ts, snap):
        t.copy_(s)
    return out


def rel(a, b):
    worst = 0.
    for x, y in zip(a, b):
        x, y = x.float(), y.float()
        d = (x - y).abs().max().item()
        if d:
            worst = max(worst, d / max(y.abs().max().item(), 1e-30))
    return worst


@triton.jit
def _near_old_kernel(qb, rb, proj, y_ptr, rk, rv, rpos, pos_ptr, fo, fl, ns, out,
                 FAR: tl.constexpr, F32: tl.constexpr, BLK: tl.constexpr):
    """The near kernel before the split (reference for --ab). Program = kv head h, its 3 query heads (padded to 16 rows for tl.dot): softmax over the 1024 ring slots (bf16
    scores + rel bias R @ proj gathered by distance, masked by the slot positions) with the sink and the far partials
    (ns splits of (o, lse)) merged by LSE; times sigmoid(gate) -> bf16 out [24*128]. Two passes over the ring so the
    probabilities are exp(S - lse) rounded to bf16 like Model.near_merge."""
    h = tl.program_id(0)
    pos = tl.load(pos_ptr)
    d = tl.arange(0, 128)
    g = tl.arange(0, 16)
    gm = g < 3
    hq = h * 3 + g
    q = tl.load(qb + hq[:, None] * 128 + d[None, :], mask=gm[:, None], other=0.)
    j = tl.arange(0, 16)
    R = tl.load(rb + hq[:, None] * 16 + j[None, :], mask=gm[:, None], other=0.)
    sl0 = tl.arange(0, BLK)
    m = tl.full([16], float('-inf'), tl.float32)
    l = tl.zeros([16], tl.float32)
    for s0 in range(0, 1024, BLK):
        sl = s0 + sl0
        rp = tl.load(rpos + sl)
        dd = pos - rp
        ok = (rp >= 0) & (dd >= 0) & (dd < 1024)
        dist = tl.minimum(tl.maximum(dd, 0), 1023)
        k = tl.load(rk + sl[:, None] * 1024 + h * 128 + d[None, :])
        S = K._r(K._r(K._mma(q, tl.trans(k), tl.zeros([16, BLK], tl.float32), F32), F32) * 0.08838834764831845, F32)
        P = tl.load(proj + j[:, None] * 1024 + dist[None, :])
        S = K._r(S + K._r(K._mma(R, P, tl.zeros([16, BLK], tl.float32), F32), F32), F32)
        S = tl.where(ok[None, :], S, float('-inf'))
        mn = tl.maximum(m, tl.max(S, 1))
        ms = tl.where(mn == float('-inf'), 0., mn)
        l = l * tl.exp(m - ms) + tl.sum(tl.exp(S - ms[:, None]), 1)
        m = mn
    lse = m + tl.log(l)
    sk = tl.load(y_ptr + 3456 + hq, mask=gm, other=0.).to(tl.float32)
    fa = tl.zeros([16, 128], tl.float32)
    tot = tl.full([16], float('-inf'), tl.float32)
    if FAR:
        fs = tl.zeros([16], tl.float32)
        for sp in range(0, ns):
            ls = tl.load(fl + sp * 24 + hq, mask=gm, other=float('-inf'))
            nt = tl.maximum(tot, ls)
            nts = tl.where(nt == float('-inf'), 0., nt)
            a = tl.exp(tot - nts)
            b = tl.exp(ls - nts)
            fs = fs * a + b
            fa = fa * a[:, None] + tl.load(fo + (sp * 24 + hq)[:, None] * 128 + d[None, :], mask=gm[:, None],
                                           other=0.) * b[:, None]
            tot = nt
        fok = fs > 0
        fa = fa / tl.where(fok, fs, 1.)[:, None]
        tot = tl.where(fok, tot + tl.log(tl.where(fok, fs, 1.)), float('-inf'))
    mx = tl.maximum(tl.maximum(lse, sk), tot)
    lse = mx + tl.log(tl.exp(lse - mx) + tl.exp(sk - mx) + tl.exp(tot - mx))
    acc = tl.zeros([16, 128], tl.float32)
    for s0 in range(0, 1024, BLK):
        sl = s0 + sl0
        rp = tl.load(rpos + sl)
        dd = pos - rp
        ok = (rp >= 0) & (dd >= 0) & (dd < 1024)
        dist = tl.minimum(tl.maximum(dd, 0), 1023)
        k = tl.load(rk + sl[:, None] * 1024 + h * 128 + d[None, :])
        S = K._r(K._r(K._mma(q, tl.trans(k), tl.zeros([16, BLK], tl.float32), F32), F32) * 0.08838834764831845, F32)
        P = tl.load(proj + j[:, None] * 1024 + dist[None, :])
        S = K._r(S + K._r(K._mma(R, P, tl.zeros([16, BLK], tl.float32), F32), F32), F32)
        p = K._bf16(tl.exp(tl.where(ok[None, :], S, float('-inf')) - lse[:, None]), F32)
        acc = K._mma(p, tl.load(rv + sl[:, None] * 1024 + h * 128 + d[None, :]), acc, F32)
    o = K._r(acc, F32)
    if FAR:
        o += fa * tl.exp(tot - lse)[:, None]
    gt = tl.sigmoid(tl.load(y_ptr + 3480 + hq[:, None] * 128 + d[None, :], mask=gm[:, None], other=0.).to(tl.float32))
    tl.store(out + hq[:, None] * 128 + d[None, :], K._bf16(o * gt, F32), mask=gm[:, None])


@triton.jit
def _w2_old_kernel(act, idx_ptr, wt_ptr, w_ptr, sc_ptr, gs_ptr, lwe, lwr, lse, lsr, st, cw, r_ptr, h_out,
                   FP4: tl.constexpr, F32: tl.constexpr, H: tl.constexpr, NI: tl.constexpr, BN: tl.constexpr,
                   BK: tl.constexpr):
    """The w2 kernel before the expert split (reference for --ab)."""
    r = tl.program_id(0) * BN + tl.arange(0, BN)
    tot = tl.zeros([BN], tl.float32)
    for t in range(8):
        e = tl.load(idx_ptr + t).to(tl.int64)
        gs = 1.
        if FP4:
            gs = tl.load(gs_ptr + e)
        acc = tl.zeros([BN], tl.float32)
        for k0 in range(0, NI, BK):
            acc = K._deq_dot(acc, w_ptr + e * lwe, sc_ptr + e * lse, r, lwr, lsr, act + t * NI, k0, gs, FP4, BK, F32)
        tot += K._r(acc, F32) * tl.load(wt_ptr + t)
    y = K._r(tot, F32)
    y = K._r(K._conv4(st, r, H, y, tl.load(cw + r * 4), tl.load(cw + r * 4 + 1), tl.load(cw + r * 4 + 2),
                      tl.load(cw + r * 4 + 3), 2.23606797749979), F32)
    tl.store(h_out + r, K._bf16(tl.load(r_ptr + r) + y, F32))


def ab(recs):
    """Tuned / restructured kernels vs the previous ones on identical inputs, per visit: max rel diff."""
    worst = {}
    put = lambda k, e: worst.__setitem__(k, max(worst.get(k, 0.), e))
    groups = {}
    for r in recs:
        groups.setdefault(r[0], []).append(r)
    for nm, over in OLD.items():                       # same kernel, old block sizes
        rs = groups[nm]
        ts = state_args(rs)
        snap = [t.clone() for t in ts]
        for r in rs:
            a = outputs([r], ts, snap)
            b = outputs([r], ts, snap, over, TUNE[nm][1])
            put(nm, rel(b, a))
    parts, outs, fins = groups['near_part'], groups['near_out'], groups['near_fin']
    for p_, o_, f_ in zip(parts, outs, fins):          # split near (3 launches) vs the one-program-per-head kernel
        launch_all([p_, o_, f_])
        a = o_[3]
        qb, rb, proj, y, rk, rv, rpos, pos = a[:8]
        fo, fl, ns, out = f_[3][2], a[10], a[11], f_[3][6]
        new = out.clone()
        out.zero_()
        _near_old_kernel[(8,)](qb, rb, proj, y, rk, rv, rpos, pos, fo, fl, ns, out, FAR=o_[4]['FAR'], F32=False, BLK=64)
        put('near (split vs old)', rel([new], [out.clone()]))
    for w_, f_ in zip(groups['w2'], groups['w2_fin']):  # w2 split over experts + fin vs the old single kernel
        a, b = w_[3], f_[3]
        st, cw, r_, h = b[1], b[2], b[3], b[4]
        st0 = st.clone()
        launch_all([w_, f_])
        hn, sn = h.clone(), st.clone()
        st.copy_(st0)
        _w2_old_kernel[(H // 16,)](*a[:10], st, cw, r_, h, FP4=w_[4]['FP4'], F32=False, H=H, NI=NI, BN=16, BK=128)
        put('w2 (split vs old)', rel([hn, sn], [h.clone(), st.clone()]))
    print('A/B vs previous kernels (max rel diff over visits): ' + ', '.join(f'{k} {v:.1e}' for k, v in worst.items()),
          flush=True)
    return worst


def main():
    argv = [x for x in sys.argv[1:] if not x.startswith('--')]
    n = int(argv[0]) if argv else 1100
    sweep = '--sweep' in sys.argv
    only = None
    if '--only' in sys.argv:
        only = set(sys.argv[sys.argv.index('--only') + 1].split(','))
    from engine import Engine
    eng = Engine('cuda', max_ctx=max(8192, n + 256), vision=False)
    m = eng.model
    M.MOE_IMPL = 'triton'
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(1000, 100000, (n,), generator=g).tolist()
    seq = m.new_seq()
    for a in range(0, n, eng.chunk):
        m.forward(m.embed(ids[a:a + eng.chunk]), seq)
    dec = D.Decoder(m, graph=False, fuse=True)
    dec.load(seq)
    dec.step(ids[-1])
    recs = record(dec, ids[-2])
    print(f'pos {seq.pos}, {len(recs)} launches, BLK {K.BLK}, flags {K.kopts()[0]} W2ALL {K.OPT["W2ALL"]} '
          f'FAR2 {D.FAR2} (nsp {D.FAR_NSP}, {D.FAR_KW})', flush=True)
    names = sorted({r[0] for r in recs}, key=lambda x: [r[0] for r in recs].index(x))
    tot = 0.
    print(f'{"kernel":10s} {"n":>3s} {"ms/step":>8s} {"MB/step":>8s} {"GB/s":>7s} {"%peak":>6s}')
    for nm in names:
        rs = [r for r in recs if r[0] == nm]
        ms = time_graph(rs)
        b = sum(wbytes(nm, r[3], r[2]) for r in rs)
        tot += ms
        print(f'{nm:10s} {len(rs):3d} {ms:8.3f} {b / 1e6:8.1f} {b / ms / 1e6:7.0f} {100 * b / ms / 1e6 / (PEAK / 1e9):5.1f}%',
              flush=True)
    print(f'sum of kernels {tot:.3f} ms; whole step as one graph {time_graph(recs):.3f} ms', flush=True)
    if '--ab' in sys.argv:
        ab(recs)
    if not sweep:
        return
    for nm in names:
        if nm not in TUNE or (only and nm not in only):
            continue
        rs = [r for r in recs if r[0] == nm]
        space, gridf = TUNE[nm]
        ts = state_args(rs)
        snap = [t.clone() for t in ts]
        ref = outputs(rs, ts, snap)
        b = sum(wbytes(nm, r[3], r[2]) for r in rs)
        res = []
        keys = list(space) + list(META)
        for vals in itertools.product(*(list(space.values()) + list(META.values()))):
            over = dict(zip(keys, vals))
            try:
                if nm == 'proj' and any(r[3][1] % over['BN'] or r[3][2] % over['BN'] for r in rs):
                    continue
                ms = time_graph(rs, over, gridf)
                err = rel(outputs(rs, ts, snap, over, gridf), ref)
            except Exception as e:  # noqa: BLE001  (out of resources, bad shapes)
                print(f'  {nm} {over}: skipped ({str(e).splitlines()[0][:80] if str(e) else type(e).__name__})')
                torch.cuda.synchronize()
                continue
            res.append((ms, over, err))
        res.sort(key=lambda x: x[0])
        print(f'\n{nm}: default {time_graph(rs):.3f} ms; best variants:', flush=True)
        for ms, over, err in res[:6]:
            print(f'  {ms:7.3f} ms  {b / ms / 1e6:6.0f} GB/s  {over}  max rel vs default {err:.1e}', flush=True)


if __name__ == '__main__':
    main()
