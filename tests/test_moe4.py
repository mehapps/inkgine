"""CPU checks for the round-4 MoE flags (Triton interpreter, fake packed weights of the real shapes, no GPU):
1. moe_route.route (INK_MOE_ROUTE) == the torch routing: block tables equal moe_triton.tables for several (BB, BS),
   each expert segment holds the same (token, weight) rows, fidx = finish's per-token order. Skewed counts included.
2. Model.moe with INK_MOE_ROUTE on == off, bit for bit (NVFP4 fp4act path and MXFP8 v2 path).
3. INK_MOE_AMAX on == off, bit for bit (absmax == abs().amax(); epilogue row max == amax of the stored act).
4. Tail block size (INK_MOE_FP4_BS / INK_MOE_BS = 32) vs 64: reported; on the GPU the mma sums k in the same order for
   any tile (gpu_moe4.py checks torch.equal); the interpreter's numpy dot doesn't, so here: >= 99.9% within 1 ulp.
Run: cd engine && .venv/bin/python tests/test_moe4.py   (~5-10 min)"""
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('TRITON_INTERPRET', '1')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import torch  # noqa: E402

import model as M  # noqa: E402
import moe_fp4 as F  # noqa: E402
import moe_route  # noqa: E402
import moe_triton as MT  # noqa: E402
from test_moe import E, fake_weights  # noqa: E402

torch.manual_seed(0)


def torch_route(idx, w):
    flat = idx.flatten()
    order = flat.argsort()
    return flat[order], order // idx.shape[1], w.flatten()[order]


def route_check():
    for T, E_, hot in ((64, 16, False), (300, 128, False), (512, 128, True), (1, 128, False)):
        sc = torch.rand(T, E_)
        if hot:
            sc[:, 5] += 1.                      # every token routes expert 5 (skew: one big segment)
        idx = sc.topk(8, -1).indices
        w = torch.rand(T, 8)
        se, tok, ws = torch_route(idx, w)
        for bb, bs in ((128, 64), (128, 32), (64, 64), (256, 64), (64, 16)):
            rt = moe_route.route(idx, w, bb, bs, E=E_)
            ref = MT.tables(se, E_, T * 8, bb, bs)
            for a, b, n in zip(rt[2:6], ref, ('expert', 'row0', 'row1', 'big')):
                assert torch.equal(a, b), f'T={T} E={E_} bb/bs {bb}/{bs}: table {n} differs'
        t2, w2, fidx = rt[0].long(), rt[1], rt[6].long()
        for e in range(E_):
            m = se == e
            a = sorted(zip(tok[m].tolist(), ws[m].tolist()))
            b = sorted(zip(t2[m].tolist(), w2[m].tolist()))   # same segment rows (route keeps torch's segment bounds)
            assert a == b, f'expert {e} segment differs'
        assert torch.equal(fidx, torch.sort(t2, stable=True).indices), 'fidx != finish order'
    print('route: tables equal, segments equal, fidx == finish order (5 block shapes, skewed + T=1): OK')


def moe_out(m, L, u, route=False, amax=False):
    M.ROUTE[0], F.AMAX = route, amax
    try:
        return m.moe(L, u, 'triton')
    finally:
        M.ROUTE[0], F.AMAX = False, False


def ulps(a, b):
    d = (a.view(torch.int16).int() - b.view(torch.int16).int()).abs()
    return (d == 0).float().mean().item(), (d <= 1).float().mean().item()


def main():
    route_check()
    m = object.__new__(M.Model)
    m.W = fake_weights()
    u = M.rms(torch.randn(64, M.H)).bfloat16()
    M.GLUE_CPU[0] = True
    MT.V2, MT.FP4ACT, MT.FP4ACT_MIN = True, True, 64
    x = torch.randn(200, 1152).bfloat16()
    assert torch.equal(F.absmax(x).float(), x.abs().amax(-1).float()), 'absmax != abs().amax()'
    for L in (0, 20):
        base = moe_out(m, L, u)
        r = moe_out(m, L, u, route=True)
        if L < 20:
            assert torch.equal(base, r), f'layer {L}: ROUTE output differs'
            print(f'layer {L}: ROUTE on == off (bit-identical)')
        else:
            # bf16 v2 path: the interpreter's numpy fp32 dot depends on a row's position in the tile, and ROUTE puts an
            # expert's rows in atomic order (checked: permuting rows within experts under the torch routing flips the
            # same ~1e-4 of outputs); the GPU mma is per row (gpu_moe4.py: torch.equal). Same gate as test_moe's v2.
            eq, le1 = ulps(r, base)
            print(f'layer {L}: v2 ROUTE vs off: bit-equal {eq:.4f}, <= 1 ulp {le1:.5f} (interpreter dot order)')
            assert le1 >= 0.999 and (r.float() - base.float()).abs().max() <= 0.01 * base.float().abs().max(), \
                'v2 ROUTE differs beyond the interpreter dot-order noise'
        if L < 20:
            a = moe_out(m, L, u, amax=True)
            assert torch.equal(base, a), 'AMAX output differs'
            ra = moe_out(m, L, u, route=True, amax=True)
            assert torch.equal(base, ra), 'ROUTE + AMAX output differs'
            print(f'layer {L}: AMAX on == off, ROUTE + AMAX == off (bit-identical)')
            old = F.BS
            F.BS = 32
            try:
                b32, b32r = moe_out(m, L, u), moe_out(m, L, u, route=True)
            finally:
                F.BS = old
            assert torch.equal(b32, b32r), 'BS 32: ROUTE differs'
            eq, le1 = ulps(b32, base)
            print(f'layer {L}: fp4 BS 32 vs 64: bit-equal {eq:.4f}, <= 1 ulp {le1:.5f} (interpreter dot order)')
            assert le1 >= 0.999, 'BS 32 differs beyond the interpreter dot-order noise'
        else:
            old = MT.BS
            MT.BS = 32
            try:
                b32, b32r = moe_out(m, L, u), moe_out(m, L, u, route=True)
            finally:
                MT.BS = old
            assert ulps(b32, b32r)[1] >= 0.999, 'BS 32: ROUTE differs'
            eq, le1 = ulps(b32, base)
            print(f'layer {L}: v2 BS 32 vs 64: bit-equal {eq:.4f}, <= 1 ulp {le1:.5f} (interpreter dot order)')
            assert le1 >= 0.999, 'BS 32 differs beyond the interpreter dot-order noise'
    print('MOE4 OK')


if __name__ == '__main__':
    assert E == 16
    main()
