"""CPU checks for the grouped MoE path (moe_triton.py) on fake packed weights of the real shapes.
1. moe_triton.emulate == Model.moe's per-expert math, NVFP4 (layer 0) and MXFP8 (layer 20).
2. The Triton kernels themselves, run by Triton's CPU interpreter (TRITON_INTERPRET=1, no GPU), == emulate.
3. grouped_v2 (INK_MOE_V2) == grouped bit for bit (interpreter: same K tiles, atomics run in expert order there).
4. moe_fp4 (INK_MOE_FP4ACT): quant kernel == quant_ref bit for bit, kernels == moe_fp4.emulate, and the fp4 error
   vs the exact math (printed, with the bf16 kernel's own error next to it).
Run: cd engine && .venv/bin/python tests/test_moe.py      (~15 min; INK_SKIP_INTERP=1 skips 2-4, 30 s)"""
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''      # never touch the GPU from this test
os.environ.setdefault('TRITON_INTERPRET', '1')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402

import model as M  # noqa: E402
import moe_triton  # noqa: E402

torch.manual_seed(0)
E, T = 16, 48   # few experts so it stays small; topk 8 of 16 still routes several blocks per expert


def fake_weights():
    W = object.__new__(M.Weights)
    W.dev, W.e2m1, W.t = torch.device('cpu'), torch.tensor(M.E2M1), {}
    for L in (0, 20):
        p = f'layers.{L}.mlp.'
        W.t[p + 'gate.weight'] = (torch.randn(E, M.H) * 0.05).bfloat16()
        W.t[p + 'gate.bias'] = torch.randn(E) * 0.1
        for w, (rows, cols) in (('w13', (2 * 1152, M.H)), ('w2', (M.H, 1152))):
            q = p + 'experts.' + w + '_weight'
            if L < 20:
                W.t[q] = torch.randint(0, 256, (E, rows, cols // 2), dtype=torch.uint8)
                W.t[q + '_scale'] = (torch.rand(E, rows, cols // 16) * 48 + 16).to(torch.float8_e4m3fn)
                W.t[q + '_amax'] = torch.rand(E) + 0.5
                W.t[q + '_outer_scale_denominator'] = torch.tensor([2688.])
            else:
                W.t[q] = torch.randn(E, rows, cols).to(torch.float8_e4m3fn)
                W.t[q + '_scale'] = torch.randint(118, 124, (E, rows, cols // 32), dtype=torch.uint8)
    return W


def route(m, L, u):
    """The sorted (expert, token, weight) rows exactly as Model.moe builds them."""
    rec = {}
    old, M.MOE_IMPL = M.MOE_IMPL, 'grab'
    m.moe_grouped = lambda L_, u_, se, tok, ws, impl=None: rec.update(args=(se, tok, ws)) or u_
    try:
        m.moe(L, u)
    finally:
        M.MOE_IMPL = old
        del m.moe_grouped
    return rec['args']


def fp4_checks(m, L, u, se, tok, ws, a, ref, emu):
    import moe_fp4 as F
    rel = lambda x, r: ((x - r).abs().mean() / r.abs().mean()).item()   # noqa: E731
    q, s, rs = F.quant(u)
    q2, s2 = F.quant_ref(u, rs)
    assert torch.equal(q, q2) and torch.equal(s.view(torch.uint8), s2.view(torch.uint8)), 'quant kernel != quant_ref'
    for terms in (1, 2):   # quant error alone: the dequantized terms vs u
        d = F.deq(*F.quant_ref(u, rs, terms)).sum(0) * rs[:, None]
        print(f'layer {L}: fp4 activations, {terms} term(s): mean rel err of u {rel(d, u.float()):.2e}')
    exact = m.moe(L, u, 'exact').float()
    fe = F.emulate(u, se, tok, ws, *a).float()
    fk = F.grouped(u, se, tok, ws, *a).float()
    f1 = F.emulate(u, se, tok, ws, *a, terms=1).float()
    print(f'layer {L}: vs exact: bf16 kernel {rel(emu, exact):.2e}, fp4 1 term {rel(f1, exact):.2e}, '
          f'fp4 2 terms {rel(fe, exact):.2e}; fp4 kernel vs its emulate {rel(fk, fe):.2e}')
    assert rel(fk, fe) < 0.25 * rel(fe, exact) + 1e-4, 'fp4 kernel differs from moe_fp4.emulate'
    assert rel(fe, exact) < 0.05, 'fp4 error unexpectedly large'


def tables_check():
    """v2 block table: each routed row in exactly one block of its own expert, blocks of BB rows or BS tails."""
    for n, E_ in ((1000, 16), (8, 128), (32768, 128)):
        se = torch.sort(torch.randint(0, E_, (n,))).values
        if n == 32768:
            se = torch.sort(torch.cat((se[:30000], torch.full((2768,), 5)))).values   # one hot expert
        be, b0, b1, bb = moe_triton.tables(se, E_, n)
        cover = torch.zeros(n, dtype=torch.int)
        for e, a, b, big in zip(be.tolist(), b0.tolist(), b1.tolist(), bb.tolist()):
            if e >= 0:
                b = min(b, a + (moe_triton.BB if big else moe_triton.BS))
                assert (se[a:b] == e).all()
                cover[a:b] += 1
        assert (cover == 1).all(), 'v2 block table does not cover each row once'


def main():
    m = object.__new__(M.Model)
    m.W = fake_weights()
    u = M.rms(torch.randn(T, M.H)).bfloat16()
    interp = os.environ.get('INK_SKIP_INTERP') != '1'
    for L in (0, 20):
        M.MOE_IMPL = 'torch'
        ref = m.moe(L, u).float()
        M.MOE_IMPL = 'emulate'
        emu = m.moe(L, u).float()
        M.MOE_IMPL = 'torch'
        scale = ref.abs().mean().item()
        err = (emu - ref).abs()
        print(f'layer {L}: emulate vs Model.moe max abs {err.max():.4g}, mean abs {err.mean():.3g} (|ref| mean {scale:.3g})')
        assert err.mean() < 0.01 * scale and err.max() < 0.1 * ref.abs().max(), 'emulation differs from Model.moe'
        if interp:
            se, tok, ws = route(m, L, u)
            P, fp4 = f'layers.{L}.mlp.experts.', L < 20
            t = m.W.t
            gs = [t[P + w + '_weight_amax'] / t[P + w + '_weight_outer_scale_denominator'][0] if fp4 else None
                  for w in ('w13', 'w2')]
            got = moe_triton.grouped(u, se, tok, ws, t[P + 'w13_weight'], t[P + 'w13_weight_scale'], gs[0],
                                     t[P + 'w2_weight'], t[P + 'w2_weight_scale'], gs[1], fp4).float()
            e2 = (got - emu).abs()
            print(f'layer {L}: triton (interpreter) vs emulate max abs {e2.max():.4g}, mean abs {e2.mean():.3g}')
            se1, tok1, ws1 = route(m, L, u[:1])                    # decode shape: 1 token, 8 rows, mostly empty blocks
            got1 = moe_triton.grouped(u[:1], se1, tok1, ws1, t[P + 'w13_weight'], t[P + 'w13_weight_scale'], gs[0],
                                      t[P + 'w2_weight'], t[P + 'w2_weight_scale'], gs[1], fp4).float()
            e1 = (got1 - emu[:1]).abs()
            print(f'layer {L}: triton T=1 vs emulate row 0 max abs {e1.max():.4g}')
            assert e1.mean() < 0.002 * scale and e1.max() < 0.02 * ref.abs().max(), 'T=1 kernel differs'
            assert e2.mean() < 0.002 * scale and e2.max() < 0.02 * ref.abs().max(), 'kernel differs from emulation'
            a = (t[P + 'w13_weight'], t[P + 'w13_weight_scale'], gs[0], t[P + 'w2_weight'], t[P + 'w2_weight_scale'],
                 gs[1], fp4)
            # The interpreter's fp32 dot is numpy's, whose summation order depends on the tile shape and the column's
            # position in it (checked: equal tile shapes with gate/up interleaved still flip 0.2%), so v2 can't be
            # bit-equal here; on the GPU the mma sums k in the same order for any tile. Gate: 99.9% within 1 ulp.
            v2 = moe_triton.grouped_v2(u, se, tok, ws, *a)
            v21 = moe_triton.grouped_v2(u[:1], se1, tok1, ws1, *a)
            g16 = got.bfloat16()
            du = (v2.view(torch.int16).int() - g16.view(torch.int16).int()).abs()
            du1 = (v21.view(torch.int16).int() - got1.bfloat16().view(torch.int16).int()).abs()
            print(f'layer {L}: v2 vs grouped bit-equal {(du == 0).float().mean():.4f}, ulps <=1 {(du <= 1).float().mean():.5f}, '
                  f'max {du.max().item()} ulp; T=1 max {du1.max().item()} ulp; same sign {bool(((v2 > 0) == (g16 > 0)).all())}')
            # many-ulp outliers are near-zero outputs (8 terms cancelling): bounded in abs instead
            assert (du <= 1).float().mean() >= 0.999 and (v2.float() - got).abs().max() <= 0.01 * ref.abs().max() \
                and du1.max() <= 1, 'v2 differs from grouped beyond 1 ulp'
            if fp4:
                fp4_checks(m, L, u, se, tok, ws, a[:6], ref, emu)
    tables_check()
    # block table: every routed row in exactly one block of its own expert
    se = torch.sort(torch.randint(0, E, (1000,))).values
    be, b0, b1 = moe_triton.blocks(se, E, 1000 // moe_triton.BM + E)
    cover = torch.zeros(1000, dtype=torch.int)
    for e, a, b in zip(be.tolist(), b0.tolist(), b1.tolist()):
        if e >= 0:
            b = min(b, a + moe_triton.BM)
            assert (se[a:b] == e).all()
            cover[a:b] += 1
    assert (cover == 1).all(), 'block table does not cover each row once'
    print('MOE OK')


if __name__ == '__main__':
    main()
