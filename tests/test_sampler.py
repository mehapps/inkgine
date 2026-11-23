"""Sampler fidelity (~2-3 min on CPU; INK_DEVICE=cuda in a GPU window): every sampler we use, checked against the
exact target distribution over many draws, and the random bits behind its noise.
1. bits: torch exponential_ (fp32 / fp64) on the device: counts of e < 2^-k vs N 2^-k, and whether small e sit on a
   2^-b grid (e ~ u near 0, so a b-bit uniform shows up as e*2^b near-integer). An exponential race argmax p/e can only
   draw a token with p_i / p_max below ~2^-b if e reaches that low, so b bounds how far into the tail it is exact.
2. chi-square (V = 64, random logits, T = 0.7 / top-p 0.9): decode.sample, engine.sample_rows (both also with
   INK_SAMPLE_F64: float64 noise), spec.accept (k = 0: multinomial), spec.accept_dev (k = 0: fp64 race), dspark.race;
   INK_SAMPLE_F64 sample_rows == decode.sample row by row.
3. tail z-test: p = 0.95 on one token, 0.05 spread over 2^20 tokens (each ~4.8e-8 = 2^-24.3, far below the head):
   tail draws vs N 0.05, |z| < 4.5. With a 24-bit uniform behind the noise this tail is (nearly) undrawable.
Exit 1 on a failure. The plain samplers (decode.sample / engine.sample_rows) are another agent's files: a failure
there is reported, not fixed here.
"""
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
DEV = os.environ.get('INK_DEVICE', 'cpu')
if DEV == 'cpu':
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
import torch  # noqa: E402

import dspark  # noqa: E402
import spec  # noqa: E402
from decode import sample  # noqa: E402
from test_spec import chi2_ok  # noqa: E402


def bits(dt, n=1 << 27):
    """-> (min e, counts below 2^-k vs expected, mean distance of e*2^b to an integer for small e (0.25 = no b-bit
    grid; ~0 = the uniform has b bits))."""
    g = torch.Generator(device=DEV).manual_seed(0)
    e = torch.empty(n, dtype=dt, device=DEV).exponential_(generator=g).double()
    cnt = {k: (int((e < 2.0 ** -k).sum()), n * 2.0 ** -k) for k in (20, 23, 24, 25)}
    grid = {}
    for b, lim in ((24, 2.0 ** -16), (32, 2.0 ** -20)):
        s = e[e < lim] * 2.0 ** b
        grid[b] = float((s - s.round()).abs().mean()) if s.numel() else float('nan')
    return float(e.min()), cnt, grid


def samplers():
    """name -> fn(p [V] probs, logits [V], gen) -> token id (int)."""
    import decode
    from engine import sample_rows
    T = 1.0

    def f64(fn):  # the INK_SAMPLE_F64 variant (float64 noise) of a plain sampler
        def run(p, lg, g):
            old, decode.SAMPLE_F64 = decode.SAMPLE_F64, True
            try:
                return fn(p, lg, g)
            finally:
                decode.SAMPLE_F64 = old
        return run
    ds = lambda p, lg, g: int(sample(lg, T, generator=g))
    sr = lambda p, lg, g: int(sample_rows(lg[None], [(T, 1.0, 0, g)])[0])
    return {
        'decode.sample': ds,
        'engine.sample_rows': sr,
        'decode.sample INK_SAMPLE_F64': f64(ds),
        'engine.sample_rows INK_SAMPLE_F64': f64(sr),
        'spec.accept (multinomial)': lambda p, lg, g: spec.accept(p[None], [], gen=g)[0],
        'spec.accept_dev (fp64 race)': lambda p, lg, g: spec.accept_dev(p[None], [], gen=g)[0],
        'dspark.race (fp64 noise)': lambda p, lg, g: int(dspark.race(p[None], torch.empty(
            1, p.numel(), dtype=torch.float64, device=p.device).exponential_(generator=g))[0]),
    }


def f64_rows_equal():
    """INK_SAMPLE_F64: engine.sample_rows row i == decode.sample(row i) with the same generator state (plain, top-p,
    top-k, greedy rows mixed in one call), as without the flag."""
    import decode
    from engine import sample_rows
    old, decode.SAMPLE_F64 = decode.SAMPLE_F64, True
    try:
        lg = torch.randn(6, 5000, generator=torch.Generator().manual_seed(3)).to(DEV) * 3
        cfg = [(0.7, 1.0, 0), (0.7, 0.9, 0), (1.0, 1.0, 40), (0.0, 1.0, 0), (0.7, 0.9, 0), (1.3, 0.5, 0)]
        mk = lambda: [torch.Generator(device=DEV).manual_seed(10 + i) for i in range(6)]
        g1, g2 = mk(), mk()
        got = sample_rows(lg.clone(), [(t, p, k, g) for (t, p, k), g in zip(cfg, g1)]).tolist()
        ref = [int(sample(lg[i], t, p, k, g)) for i, ((t, p, k), g) in enumerate(zip(cfg, g2))]
    finally:
        decode.SAMPLE_F64 = old
    print(f'INK_SAMPLE_F64 sample_rows == decode.sample per row: {got == ref}', flush=True)
    return got == ref


def main():
    torch.set_num_threads(int(os.environ.get('INK_THREADS', 4)))
    ok = True
    print(f'device {DEV}')
    for dt in (torch.float32, torch.float64):
        t0 = time.time()
        mn, cnt, grid = bits(dt)
        good = grid[32] > 0.15
        print(f'exponential_ {dt}: min {mn:.3g}; e<2^-k: ' + ', '.join(f'k={k} {a} (exp {b:.0f})' for k, (a, b) in cnt.items())
              + f'; 2^-b grid distance b=24 {grid[24]:.3f}, b=32 {grid[32]:.3f} -> '
              + ('>= 32 random bits' if good else ('~24 bits' if grid[24] < 0.05 else '< 32 bits'))
              + f' ({time.time() - t0:.0f} s)', flush=True)
        if dt == torch.float64 and not good:   # dspark.race / accept_dev rely on fp64 noise having the bits
            ok = False
    g = torch.Generator(device=DEV).manual_seed(1)
    V, N = 64, 20000
    lg = (torch.randn(V, generator=torch.Generator().manual_seed(2)) * 2).to(DEV)
    p = torch.softmax(lg, -1)
    for name, fn in samplers().items():
        t0 = time.time()
        c = torch.zeros(V)
        for _ in range(N):
            c[fn(p, lg, g)] += 1
        x2, crit = chi2_ok(c, p.cpu(), N)
        ok &= x2 < crit
        print(f'chi2 {name}: {x2:.1f} < {crit:.1f} {"OK" if x2 < crit else "FAIL"} ({time.time() - t0:.0f} s)', flush=True)
    V, N, tail = 1 + (1 << 20), int(os.environ.get('INK_TAIL_N', 1500)), 0.05
    p = torch.full((V,), tail / (V - 1), device=DEV, dtype=torch.float32)
    p[0] = 1 - tail
    lg = p.log()
    for name, fn in samplers().items():
        t0 = time.time()
        k = sum(fn(p, lg, g) != 0 for _ in range(N))
        z = (k - N * tail) / math.sqrt(N * tail * (1 - tail))
        ok &= abs(z) < 4.5
        print(f'tail {name}: {k}/{N} tail draws (expected {N * tail:.0f}, each tail p = {tail / (V - 1):.2g}), '
              f'z {z:+.1f} {"OK" if abs(z) < 4.5 else "FAIL"} ({time.time() - t0:.0f} s)', flush=True)
    ok &= f64_rows_equal()
    print('SAMPLER ALL OK' if ok else 'SAMPLER FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
