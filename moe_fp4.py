"""INK_MOE_FP4ACT=1 (off by default; changes numerics): the NVFP4 experts (layers 0-19) on the FP4 tensor cores.

The weights are consumed as stored (e2m1 codes + e4m3 scale per 16, the per-expert amax/denominator applied to the
fp32 accumulator), so there is no weight dequant at all. Activations (u for w13, act for w2) are quantized per row
to e2m1 + e4m3 scale per 16 with an fp32 row scale rs = amax(row) / (6 * 448), as TERMS terms: term 1 = round(x),
term 2 = round(x - term 1) (two FP4 terms give ~8-bit-like accuracy at FP4 speed). Each
term is one tl.dot_scaled (mma ...kind::mxf4nvf4.block_scale ... ue4m3 on sm_120) into the same fp32 accumulator.
Everything after the GEMMs is the bf16 path's math: bf16 GEMM outputs, the §5 activation, bf16 act, fixed-order sum
of the 8 rows with the routing weight (moe_triton.finish).

`quant_ref` / `emulate` are the pure-torch mirror (CPU check + GPU check of the hardware op); under the Triton
interpreter (which only knows e8m0 scales) the kernels dequantize both operands and use an fp32 dot instead.
"""
import os

import torch

import moe_triton as MT
from moe_triton import INTERP, e2m1, finish, tables, tanh, triton

TERMS = int(os.environ.get('INK_MOE_FP4_TERMS', 2))
QMAX = 6. * 448.                       # e2m1 max x e4m3 max: rs maps a row's amax onto the top scale
F8 = torch.float8_e4m3fn
# (BN, BK, num_warps, num_stages); w13 BN = intermediate columns (2*BN stored rows); 'q' = rows per quant program
FCFG = {'w13': (64, 128, 8, 3), 'w2': (128, 128, 8, 3), 'q': 32}
# tile overrides for a window sweep winner, e.g. INK_MOE_FP4_CFG='w13=64,256,8,2;w2=128,128,8,3' (same numbers for any)
for _kv in filter(None, os.environ.get('INK_MOE_FP4_CFG', '').split(';')):
    _k, _v = _kv.split('=')
    FCFG[_k] = tuple(int(x) for x in _v.split(','))
# block rows (big, tail) of this path; INK_MOE_FP4_BS=32 halves the tail padding (~12% of rows at 4k chunks)
BB = int(os.environ.get('INK_MOE_FP4_BB', MT.BB))
BS = int(os.environ.get('INK_MOE_FP4_BS', MT.BS))
# INK_MOE_AMAX=1 (exact): the activation row amax comes from the w13 epilogue (atomic max per row of the stored bf16
# act) and u's from one reduction, instead of abs() + amax() (a [R,1152] bf16 temp written and read: ~4.4 ms/chunk)
AMAX = os.environ.get('INK_MOE_AMAX') == '1'


# ---------------------------------------------------------------- torch mirror
def round_e2m1(d):
    """|d| <= 6 -> e2m1 code (bit 3 = sign), round to nearest, ties to even (the thresholds the kernel uses)."""
    a = d.abs()
    c = sum((x.int() for x in (a > .25, a >= .75, a > 1.25, a >= 1.75, a > 2.5, a >= 3.5, a > 5.)))
    return (c | ((d < 0).int() << 3)).to(torch.uint8)


def rowscale(x, amax=None):
    """amax: max |x| per row if already known (same value, so the same rs)."""
    return (x.abs().amax(-1) if amax is None else amax).float().clamp_min(1e-30) / QMAX


def absmax(x):
    """max |x| per row in one pass (= x.abs().amax(-1), exactly: a max has no rounding)."""
    return torch.linalg.vector_norm(x, float('inf'), -1)


def quant_ref(x, rs, terms=TERMS):
    """x [n,K] -> q [terms,n,K/2] uint8 (element 2j low nibble), s [terms,n,K/16] e4m3. x ~ rs * sum_t deq(q_t, s_t)."""
    n, K = x.shape
    res = x.float()
    qs, ss = [], []
    for _ in range(terms):
        g = (res / rs[:, None]).view(n, K // 16, 16)
        s = (g.abs().amax(-1) / 6).clamp(max=448.).to(F8)
        sf = s.float()[..., None]
        c = round_e2m1(torch.where(sf > 0, g / sf, 0.).clamp(-6., 6.))
        res = res - (e2m1(c) * sf).view(n, K) * rs[:, None]
        c = c.view(n, K // 2, 2)
        qs.append(c[..., 0] | (c[..., 1] << 4))
        ss.append(s)
    return torch.stack(qs), torch.stack(ss)


def deq(q, s):
    """packed e2m1 [.., n, K/2] + e4m3 [.., n, K/16] -> fp32 [.., n, K] (exact)."""
    v = torch.stack((e2m1(q & 15), e2m1(q >> 4)), -1).flatten(-2)
    return v * s.float().repeat_interleave(16, -1)


def emulate(u, sorted_e, tok, ws, w13, s13, gs13, w2, s2, gs2, terms=TERMS):
    """Mirror of grouped(): same quantization, fp32 GEMMs over the exactly dequantized operands."""
    T, H = u.shape
    E, NI = w13.shape[0], w13.shape[1] // 2
    rx = rowscale(u)
    xd = deq(*quant_ref(u, rx, terms)).sum(0)                       # the terms add exactly in fp32
    cnt = torch.bincount(sorted_e, minlength=E).tolist()
    act = torch.empty(len(tok), NI, dtype=torch.bfloat16, device=u.device)
    r0 = 0
    for e, c in enumerate(cnt):
        if c:
            rows = tok[r0:r0 + c]
            gu = (xd[rows] @ deq(w13[e], s13[e]).t() * rx[rows, None] * gs13[e]).bfloat16()
            g, v = gu[:, 0::2].float(), gu[:, 1::2].float()
            act[r0:r0 + c] = (4 * tanh(g / 4) * torch.sigmoid(g) * 25 * tanh(v / 25)).bfloat16()
        r0 += c
    ra = rowscale(act)
    ad = deq(*quant_ref(act, ra, terms)).sum(0)
    out = torch.zeros(T, H, dtype=torch.float32, device=u.device)
    r0 = 0
    for e, c in enumerate(cnt):
        if c:
            y = (ad[r0:r0 + c] @ deq(w2[e], s2[e]).t() * ra[r0:r0 + c, None] * gs2[e]).bfloat16()
            out.index_add_(0, tok[r0:r0 + c], y.float() * ws[r0:r0 + c, None])
        r0 += c
    return out.bfloat16()


# ---------------------------------------------------------------- kernels
if triton is not None:
    import triton.language as tl

    from moe_triton import _act, _bf16, _e2m1

    @triton.jit
    def _quant_kernel(x_ptr, rs_ptr, q_ptr, s_ptr, n, K, TERMS: tl.constexpr, BR: tl.constexpr):
        """One [BR, 64] tile of quant_ref (same ops, same order; launched without FMA contraction)."""
        r = tl.program_id(0) * BR + tl.arange(0, BR)
        rm = r < n
        c = tl.program_id(1) * 64 + tl.arange(0, 64)
        res = tl.load(x_ptr + r[:, None] * K + c[None, :], mask=rm[:, None], other=0.).to(tl.float32)
        rs = tl.load(rs_ptr + r, mask=rm, other=1.)
        for t in tl.static_range(TERMS):
            g = tl.reshape(tl.math.div_rn(res, tl.broadcast_to(rs[:, None], (BR, 64))), (BR, 4, 16))
            s = tl.minimum(tl.math.div_rn(tl.max(tl.abs(g), axis=2), tl.full((BR, 4), 6., tl.float32)), 448.)
            s = _rne_e4m3(s).to(tl.float8e4nv)                       # exact cast (the interpreter's cast misrounds)
            sf = tl.broadcast_to(s.to(tl.float32)[:, :, None], (BR, 4, 16))
            d = tl.where(sf > 0, tl.math.div_rn(g, tl.where(sf > 0, sf, 1.)), 0.)
            d = tl.minimum(tl.maximum(d, -6.), 6.)
            a = tl.abs(d)
            code = ((a > .25).to(tl.int32) + (a >= .75).to(tl.int32) + (a > 1.25).to(tl.int32) +
                    (a >= 1.75).to(tl.int32) + (a > 2.5).to(tl.int32) + (a >= 3.5).to(tl.int32) +
                    (a > 5.).to(tl.int32)) | ((d < 0).to(tl.int32) << 3)
            res = res - tl.reshape(_e2m1(code) * sf, (BR, 64)) * rs[:, None]
            lo, hi = tl.split(tl.reshape(code, (BR, 32, 2)))
            jb = tl.program_id(1) * 32 + tl.arange(0, 32)
            tl.store(q_ptr + t * n * (K // 2) + r[:, None] * (K // 2) + jb[None, :], (lo | (hi << 4)).to(tl.uint8),
                     mask=rm[:, None])
            jg = tl.program_id(1) * 4 + tl.arange(0, 4)
            tl.store(s_ptr + t * n * (K // 16) + r[:, None] * (K // 16) + jg[None, :], s, mask=rm[:, None])

    @triton.jit
    def _rne_e4m3(x):
        """0 <= x <= 448 fp32 -> nearest e4m3 value (ties to even), still fp32."""
        b = x.to(tl.int32, bitcast=True)
        nrm = ((b + 0x7FFFF + ((b >> 20) & 1)) & -1048576).to(tl.float32, bitcast=True)   # keep 3 mantissa bits
        sub = ((x * 512. + 12582912.) - 12582912.) / 512.            # subnormals: multiples of 2^-9 (1.5*2^23 trick)
        return tl.where(x < 0.015625, sub, nrm)

    @triton.jit
    def _deq4(q, s, NR: tl.constexpr, BK: tl.constexpr):
        v = tl.interleave(_e2m1(q & 15), _e2m1(q >> 4))
        return v * tl.reshape(tl.broadcast_to(s.to(tl.float32)[:, :, None], (NR, BK // 16, 16)), (NR, BK))

    @triton.jit
    def _fp4_gemm(q_ptr, s_ptr, xrow, xm, n_x, wp, sp, K,
                  TERMS: tl.constexpr, BM: tl.constexpr, NR: tl.constexpr, BK: tl.constexpr, F32: tl.constexpr):
        """[BM, NR] fp32 = sum over K and terms of deq(x terms rows xrow) @ deq(w rows wp/sp).T (no gs, no rs)."""
        acc = tl.zeros((BM, NR), tl.float32)
        for k0 in range(0, K, BK):
            jb = k0 // 2 + tl.arange(0, BK // 2)
            jg = k0 // 16 + tl.arange(0, BK // 16)
            wq = tl.load(wp[:, None] + jb[None, :])
            wsc = tl.load(sp[:, None] + jg[None, :])
            for t in tl.static_range(TERMS):
                xq = tl.load(q_ptr + t * n_x * (K // 2) + xrow[:, None] * (K // 2) + jb[None, :],
                             mask=xm[:, None], other=0)
                xs = tl.load(s_ptr + t * n_x * (K // 16) + xrow[:, None] * (K // 16) + jg[None, :],
                             mask=xm[:, None], other=0.)
                if F32:
                    acc = tl.dot(_deq4(xq, xs, BM, BK), tl.trans(_deq4(wq, wsc, NR, BK)), acc, input_precision='ieee')
                else:
                    acc = tl.dot_scaled(xq, xs, 'e2m1', tl.trans(wq), wsc, 'e2m1', acc)
        return acc

    @triton.jit
    def _w13_fp4_body(q_ptr, s_ptr, rs_ptr, T, tok_ptr, r0, r1, e, w_ptr, sw_ptr, gs, act_ptr, amax_ptr, K, NI,
                      lwe, lwr, lse, lsr, TERMS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                      BK: tl.constexpr, F32: tl.constexpr, AMAX: tl.constexpr):
        r = r0 + tl.arange(0, BM)
        rm = r < r1
        t = tl.load(tok_ptr + r, mask=rm, other=0)
        j = tl.program_id(1) * 2 * BN + tl.arange(0, 2 * BN)         # stored rows: 2n gate, 2n+1 up (§7)
        acc = _fp4_gemm(q_ptr, s_ptr, t, rm, T, w_ptr + e.to(tl.int64) * lwe + j * lwr,
                        sw_ptr + e.to(tl.int64) * lse + j * lsr, K, TERMS, BM, 2 * BN, BK, F32)
        acc = acc * tl.load(rs_ptr + t, mask=rm, other=0.)[:, None] * gs
        g, v = tl.split(tl.reshape(acc, (BM, BN, 2)))
        a = _bf16(_act(_bf16(g, F32).to(tl.float32), _bf16(v, F32).to(tl.float32)), F32)
        n = tl.program_id(1) * BN + tl.arange(0, BN)
        tl.store(act_ptr + r[:, None] * NI + n[None, :], a, mask=rm[:, None])
        if AMAX:   # row max of |stored act| over this tile's columns; max is exact, so any order gives the same value
            tl.atomic_max(amax_ptr + r, tl.max(tl.abs(a.to(tl.float32)), axis=1), mask=rm)

    @triton.jit
    def _w13_fp4(q_ptr, s_ptr, rs_ptr, T, tok_ptr, be_ptr, b0_ptr, b1_ptr, bb_ptr, w_ptr, sw_ptr, gs_ptr, act_ptr,
                 amax_ptr, K, NI, lwe, lwr, lse, lsr, TERMS: tl.constexpr, BB: tl.constexpr, BS: tl.constexpr,
                 BN: tl.constexpr, BK: tl.constexpr, F32: tl.constexpr, AMAX: tl.constexpr):
        b = tl.program_id(0)
        e = tl.load(be_ptr + b)
        if e >= 0:
            r0, r1, gs = tl.load(b0_ptr + b), tl.load(b1_ptr + b), tl.load(gs_ptr + e)
            if tl.load(bb_ptr + b) != 0:
                _w13_fp4_body(q_ptr, s_ptr, rs_ptr, T, tok_ptr, r0, r1, e, w_ptr, sw_ptr, gs, act_ptr, amax_ptr, K,
                              NI, lwe, lwr, lse, lsr, TERMS, BB, BN, BK, F32, AMAX)
            else:
                _w13_fp4_body(q_ptr, s_ptr, rs_ptr, T, tok_ptr, r0, r1, e, w_ptr, sw_ptr, gs, act_ptr, amax_ptr, K,
                              NI, lwe, lwr, lse, lsr, TERMS, BS, BN, BK, F32, AMAX)

    @triton.jit
    def _w2_fp4_body(q_ptr, s_ptr, rs_ptr, R, r0, r1, e, w_ptr, sw_ptr, gs, y_ptr, K, N, lwe, lwr, lse, lsr,
                     TERMS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, F32: tl.constexpr):
        r = r0 + tl.arange(0, BM)
        rm = r < r1
        n = tl.program_id(0) * BN + tl.arange(0, BN)
        acc = _fp4_gemm(q_ptr, s_ptr, r, rm, R, w_ptr + e.to(tl.int64) * lwe + n * lwr,
                        sw_ptr + e.to(tl.int64) * lse + n * lsr, K, TERMS, BM, BN, BK, F32)
        acc = acc * tl.load(rs_ptr + r, mask=rm, other=0.)[:, None] * gs
        tl.store(y_ptr + r.to(tl.int64)[:, None] * N + n[None, :], _bf16(acc, F32), mask=rm[:, None])

    @triton.jit
    def _w2_fp4(q_ptr, s_ptr, rs_ptr, R, be_ptr, b0_ptr, b1_ptr, bb_ptr, w_ptr, sw_ptr, gs_ptr, y_ptr,
                K, N, lwe, lwr, lse, lsr, TERMS: tl.constexpr, BB: tl.constexpr, BS: tl.constexpr,
                BN: tl.constexpr, BK: tl.constexpr, F32: tl.constexpr):
        b = tl.program_id(1)                                         # n-tiles fastest (act rows stay in L2)
        e = tl.load(be_ptr + b)
        if e >= 0:
            r0, r1, gs = tl.load(b0_ptr + b), tl.load(b1_ptr + b), tl.load(gs_ptr + e)
            if tl.load(bb_ptr + b) != 0:
                _w2_fp4_body(q_ptr, s_ptr, rs_ptr, R, r0, r1, e, w_ptr, sw_ptr, gs, y_ptr, K, N,
                             lwe, lwr, lse, lsr, TERMS, BB, BN, BK, F32)
            else:
                _w2_fp4_body(q_ptr, s_ptr, rs_ptr, R, r0, r1, e, w_ptr, sw_ptr, gs, y_ptr, K, N,
                             lwe, lwr, lse, lsr, TERMS, BS, BN, BK, F32)


def quant(x, terms=TERMS, rs=None):
    """Kernel version of quant_ref (bit-identical): -> q, s, rs. rs: precomputed rowscale(x)."""
    n, K = x.shape
    rs = rowscale(x) if rs is None else rs
    q = torch.empty(terms, n, K // 2, dtype=torch.uint8, device=x.device)
    s = torch.empty(terms, n, K // 16, dtype=F8, device=x.device)
    br = FCFG['q']
    _quant_kernel[(triton.cdiv(n, br), K // 64)](x, rs, q, s, n, K, TERMS=terms, BR=br, enable_fp_fusion=False)
    return q, s, rs


def grouped(u, sorted_e, tok, ws, w13, s13, gs13, w2, s2, gs2, terms=TERMS, fin=None, rt=None):
    """moe_triton.grouped for NVFP4 layers with FP4 activations (same arguments, gs13/gs2 required).
    rt: moe_route.route() output built with (BB, BS) (INK_MOE_ROUTE; sorted_e then unused)."""
    T, H = u.shape
    E, NI = w13.shape[0], w13.shape[1] // 2
    R = len(tok)
    be, b0, b1, bb = rt[2:6] if rt is not None else tables(sorted_e, E, R, BB, BS)
    amax = AMAX
    xq, xs, xr = quant(u, terms, rowscale(u, absmax(u)) if amax else None)
    act = torch.empty(R, NI, dtype=torch.bfloat16, device=u.device)
    am = torch.zeros(R, dtype=torch.float32, device=u.device) if amax else xr   # xr: unused stand-in pointer
    bn, bk, nw, ns = FCFG['w13']
    _w13_fp4[(len(be), NI // bn)](xq, xs, xr, T, tok.int(), be, b0, b1, bb, w13, s13, gs13, act, am, H, NI,
                                  w13.stride(0), w13.stride(1), s13.stride(0), s13.stride(1), TERMS=terms,
                                  BB=BB, BS=BS, BN=bn, BK=bk, F32=INTERP, AMAX=amax, num_warps=nw, num_stages=ns)
    aq, as_, ar = quant(act, terms, rowscale(act, am) if amax else None)
    y = torch.empty(R, H, dtype=torch.bfloat16, device=u.device)
    bn, bk, nw, ns = FCFG['w2']
    _w2_fp4[(H // bn, len(be))](aq, as_, ar, R, be, b0, b1, bb, w2, s2, gs2, y, NI, H,
                                w2.stride(0), w2.stride(1), s2.stride(0), s2.stride(1), TERMS=terms,
                                BB=BB, BS=BS, BN=bn, BK=bk, F32=INTERP, num_warps=nw, num_stages=ns)
    return fin(y, tok, ws, T) if fin else finish(y, tok, ws, T, None if rt is None else rt[6])
