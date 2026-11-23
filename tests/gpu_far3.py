"""GPU window only: far decode attention per visit, FAR2 (_far_dot_kernel, the default) vs INK_DEC_FAR3 (_far3_kernel)
tile / stage / warp / split sweep, on a synthetic 1M-row page store (random codes + scales, shuffled page table).
Per far length (131k / 524k / 1040k far keys): ms per visit and achieved GB/s (1152 B per far key: K+V int4 codes
1024 + e4m3 scales 128), timed as 8 launches in one CUDA graph (the 8 full visits of a step, back to back), plus a
plain streaming read of the same bytes (the access-pattern-free ceiling). Exactness: FAR3 merged output vs FAR2
(same splits) and vs an fp32 torch reference on a 64k slice.
--quick: fewer configs. --out FILE: write the best FAR3 config at the longest length as env assignments
(INK_DEC_FAR3_BLK=.. _STAGES=.. _WARPS=.. INK_DEC_FAR3_NSP=..) for the next window step.
Usage (Qwen down): ~/llm/engine/swap_test.sh 30 python tests/gpu_far3.py --out /tmp/ink-far3-best.env"""
import itertools
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

import decode as D  # noqa: E402

PEAK, ROWS, PG = 1792e9, 1 << 20, 256
LENS = (131072, 524288, 1040000)


@triton.jit
def _stream(x, out, n, BLOCK: tl.constexpr):
    """Read n int32 (as 16 B vectors) and keep a xor so the loads stay: the streaming-read ceiling."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], tl.int32)
    for k in range(0, n, BLOCK * tl.num_programs(0)):
        acc ^= tl.load(x + k + i, mask=k + i < n, other=0)
    tl.store(out + tl.program_id(0) * BLOCK + tl.arange(0, BLOCK), acc)


def timed(fn, reps=5):
    """ms per call of fn() (8 launches inside) captured in one graph, median of reps replays."""
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b))
    return sorted(ts)[len(ts) // 2]


def main():
    dev = torch.device('cuda')
    g = torch.Generator(device=dev).manual_seed(0)
    c = torch.randint(0, 256, (2, ROWS, 8, 64), generator=g, dtype=torch.uint8, device=dev)
    s = (torch.rand(2, ROWS, 8, 8, generator=g, device=dev) * 0.4 + 0.02).to(torch.float8_e4m3fn)
    tbl = torch.randperm(ROWS // PG, generator=g, device=dev)
    Q = (torch.randn(24, 128, generator=g, device=dev) * 2).bfloat16()
    print(f'store {(c.numel() + s.numel()) / 1e9:.2f} GB; SMs {torch.cuda.get_device_properties(0).multi_processor_count}',
          flush=True)

    # streaming ceiling over the same number of bytes as a 1040k-key visit
    nb = 1152 * LENS[-1]
    flat = c.view(-1).view(torch.int32)[:nb // 4]
    for progs in (170 * 4, 170 * 8):
        out = torch.empty(progs * 1024, dtype=torch.int32, device=dev)
        ms = timed(lambda: [_stream[(progs,)](flat, out, flat.numel(), BLOCK=1024, num_warps=8) for _ in range(8)]) / 8
        print(f'streaming read {nb / 1e9:.2f} GB ({progs} programs): {ms:.3f} ms -> {nb / ms / 1e6:.0f} GB/s '
              f'({nb / ms / 1e6 / PEAK * 1e9 * 100:.0f}% of {PEAK / 1e9:.0f})', flush=True)

    def bench(far3, nsp, blk, kw, kend):
        """ms per visit of one config (kernel options are read by launch_far at capture) -> (ms, o, lse)."""
        pos = torch.tensor([kend + 1023], device=dev)
        o, lse = torch.empty(nsp, 24, 128, device=dev), torch.empty(nsp, 24, device=dev)
        save = dict(D.FAR_KW), dict(D.FAR3_KW)
        D.FAR_KW.update(num_warps=kw['num_warps'], num_stages=kw['ST'])
        D.FAR3_KW.update(kw)
        try:
            ms = timed(lambda: [D.launch_far(Q, c[0], c[1], s[0], s[1], tbl, pos, o, lse, 4096, blk=blk, nsp=nsp,
                                             far3=far3) for _ in range(8)]) / 8
        finally:
            D.FAR_KW.clear(), D.FAR_KW.update(save[0])
            D.FAR3_KW.clear(), D.FAR3_KW.update(save[1])
        return ms, o, lse

    def merged(o, lse):
        tot = torch.logsumexp(lse, 0)
        return (o * torch.nan_to_num(torch.exp(lse - tot))[..., None]).sum(0), tot

    # exactness on a 64k slice: fp32 torch reference, FAR2, FAR3 (same splits)
    kend = 65536
    rows = (tbl[:kend // PG, None] * PG + torch.arange(PG, device=dev)).reshape(-1)

    def deq(x, sc):
        v = torch.stack(((x & 15).float() - 8, (x >> 4).float() - 8), -1).reshape(*x.shape[:-1], 128)
        return v * sc.float().repeat_interleave(16, -1)
    k, v = deq(c[0, rows], s[0, rows]), deq(c[1, rows], s[1, rows])
    S = torch.einsum('hgd,nhd->hgn', Q.view(8, 3, 128).float(), k) * 128 ** -0.5
    ro = torch.einsum('hgn,nhd->hgd', torch.softmax(S, -1), v).reshape(24, 128)
    del k, v, S
    _, o2, l2 = bench(False, 42, 128, dict(num_warps=4, ST=3), kend)
    _, o3, l3 = bench(True, 42, 128, dict(D.FAR3_KW), kend)
    m2, m3 = merged(o2, l2)[0], merged(o3, l3)[0]
    rel = lambda a, b: ((a - b).abs().max() / b.abs().max()).item()
    print(f'exactness at {kend} far keys (nsp 42, BLK 128): FAR2 vs fp32 {rel(m2, ro):.1e}, FAR3 vs fp32 '
          f'{rel(m3, ro):.1e}, FAR3 vs FAR2 {rel(m3, m2):.1e}, lse diff {(l3 - l2).abs().max():.1e}', flush=True)
    ok = rel(m3, m2) < 1e-3 and rel(m3, ro) < 3 * rel(m2, ro) + 1e-3

    quick = '--quick' in sys.argv
    grid = list(itertools.product((64, 128) if quick else (32, 64, 128), (3,) if quick else (2, 3, 5), (4, 8),
                                  (63, 84) if quick else (42, 63, 84, 126, 170)))
    best = {}
    for kend in LENS:
        fb = 1152 * kend
        print(f'\n== {kend} far keys: {fb / 1e9:.2f} GB per visit, floor {fb / PEAK * 1e3:.3f} ms', flush=True)
        res = []
        for nsp in (42, 63, 84):
            ms = bench(False, nsp, 128, dict(num_warps=4, ST=3), kend)[0]
            res.append((ms, f'FAR2 nsp {nsp} BLK 128 ST 3 w 4', None))
        for blk, st, nw, nsp in grid:
            try:
                ms = bench(True, nsp, blk, dict(BLK=blk, ST=st, num_warps=nw), kend)[0]
            except Exception as e:  # noqa: BLE001 (out of shared memory etc.: skip the config)
                print(f'  FAR3 BLK {blk} ST {st} w {nw} nsp {nsp}: {type(e).__name__} {str(e)[:80]}', flush=True)
                continue
            res.append((ms, f'FAR3 nsp {nsp} BLK {blk} ST {st} w {nw}', (blk, st, nw, nsp)))
        res.sort(key=lambda r: r[0])
        for ms, name, _ in res[:12] + [r for r in res if r[1].startswith('FAR2')]:
            print(f'  {name:32s} {ms:7.3f} ms/visit  {fb / ms / 1e6:6.0f} GB/s  {fb / ms / 1e6 / PEAK * 1e9 * 100:4.1f}%'
                  f'  (8 visits {8 * ms:.2f} ms)', flush=True)
        best[kend] = next(r for r in res if r[2])
    b = best[LENS[-1]]
    blk, st, nw, nsp = b[2]
    env = f'INK_DEC_FAR3_BLK={blk} INK_DEC_FAR3_STAGES={st} INK_DEC_FAR3_WARPS={nw} INK_DEC_FAR3_NSP={nsp}'
    print(f'\nbest FAR3 at {LENS[-1]}: {env} ({b[0]:.3f} ms/visit)', flush=True)
    if '--out' in sys.argv:
        with open(sys.argv[sys.argv.index('--out') + 1], 'w') as f:
            f.write(env + '\n')
    print('PASS' if ok else 'FAIL (exactness)')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
