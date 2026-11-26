"""Grouped MoE from the packed expert weights (SPEC §5, §7), no bf16 copy of any expert in memory.

Routed (token, expert) pairs are sorted by expert and cut into blocks of BM rows that each belong to one
expert. Two launches per layer: w13 (+ activation) over all blocks, then w2 (+ routing weight, atomic add
into the fp32 output). Each program dequantizes its weight tile in registers: fp32 value, one cast to
bf16, exactly like Weights.expert, then a bf16 dot with fp32 accumulate.

`emulate` is a pure-torch mirror of the two kernels (same blocks, same K tiles, same dequant formulas) so
the math can be checked on CPU; tests/test_moe.py compares it to Model.moe and, under
TRITON_INTERPRET=1, to the kernels themselves.
"""
import os

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None

BM, BN, BK = 64, 64, 128   # BK in weight elements (64 packed bytes for NVFP4); 2304 and 1152 are multiples
# Triton's CPU interpreter computes bf16 tl.dot wrong (garbage ~1e10) and truncates fp32 -> bf16; with F32 set
# (interpreter only) we upcast before the dot (exact) and round to nearest even by hand, as the GPU does.
INTERP = os.environ.get('TRITON_INTERPRET') == '1'


def blocks(sorted_e, E, n_blk):
    """sorted_e: expert id per sorted row. -> (expert, row0, row1) per block, int32 [n_blk]; expert -1 = unused.
    All on device, no host sync; n_blk is an upper bound (rows / BM + E)."""
    ar = torch.arange(E, device=sorted_e.device)
    start = torch.searchsorted(sorted_e, ar)
    cnt = torch.searchsorted(sorted_e, ar, right=True) - start
    nb = (cnt + BM - 1) // BM
    cb = nb.cumsum(0)                                               # first block of the next expert
    b = torch.arange(n_blk, device=sorted_e.device)
    e = torch.searchsorted(cb, b, right=True)
    ec = e.clamp(max=E - 1)
    r0 = start[ec] + (b - (cb[ec] - nb[ec])) * BM
    r1 = start[ec] + cnt[ec]
    return torch.where(e < E, e, -1).int(), r0.int(), r1.int()


def tanh(x):  # same formula the kernel uses (Triton core has no tanh)
    return 2 * torch.sigmoid(2 * x) - 1


def e2m1(c):
    """fp4 code (bit 3 = sign, 2 exponent bits, 1 mantissa bit) -> fp32, built from bits like the kernel."""
    m = (c & 7).int()
    bits = torch.where(m < 2, m * (126 << 23), (((m >> 1) + 126) << 23) | ((m & 1) << 22)) | ((c & 8).int() << 28)
    return bits.view(torch.float32)


def e8m0(s):
    """2^(s-127) from bits (s = 0 gives 0 instead of 2^-127, which never occurs in real scales)."""
    return (s.int() << 23).view(torch.float32)


def _gemm_emu(x, w, s, gs, fp4):
    """x [n,K] bf16 @ dequant(w)[N,K].T, K-tiled with fp32 accumulate -> bf16 (one expert)."""
    acc = torch.zeros(x.shape[0], w.shape[0])
    for k0 in range(0, x.shape[1], BK):
        k = torch.arange(k0, k0 + BK)
        if fp4:
            j = torch.arange(k0 // 2, (k0 + BK) // 2)
            q, sc = w[:, j], s[:, j // 8].float()
            wt = torch.stack(((e2m1(q & 15) * sc * gs).bfloat16(), (e2m1(q >> 4) * sc * gs).bfloat16()), -1)
            wt = wt.view(w.shape[0], BK)                            # interleave: element 2j low, 2j+1 high
        else:
            wt = (w[:, k].float() * e8m0(s[:, k // 32])).bfloat16()
        acc += x[:, k].float() @ wt.float().t()
    return acc.bfloat16()


def emulate(u, sorted_e, tok, ws, w13, s13, gs13, w2, s2, gs2, fp4, out=None):
    """CPU mirror of grouped(). Same arguments. out: optional fp32 [T,H] accumulator (returns it instead of bf16)."""
    E, NI = w13.shape[0], w13.shape[1] // 2
    be, b0, b1 = (t.tolist() for t in blocks(sorted_e, E, len(tok) // BM + E))
    act = torch.zeros(len(tok), NI, dtype=torch.bfloat16)
    acc = torch.zeros(u.shape, dtype=torch.float32) if out is None else out
    for e, r0, r1 in zip(be, b0, b1):
        if e < 0:
            continue
        gu = _gemm_emu(u[tok[r0:r1]], w13[e], s13[e], gs13[e] if fp4 else None, fp4)
        g, up = gu[:, 0::2].float(), gu[:, 1::2].float()            # rows 2i gate, 2i+1 up (§7)
        act[r0:r1] = (4 * tanh(g / 4) * torch.sigmoid(g) * 25 * tanh(up / 25)).bfloat16()
    for e, r0, r1 in zip(be, b0, b1):
        if e < 0:
            continue
        y = _gemm_emu(act[r0:r1], w2[e], s2[e], gs2[e] if fp4 else None, fp4).float() * ws[r0:r1, None]
        acc.index_add_(0, tok[r0:r1], y)
    return acc.bfloat16() if out is None else acc


if triton is not None:
    @triton.jit
    def _e2m1(c):
        m = (c & 7).to(tl.int32)
        bits = tl.where(m < 2, m * (126 << 23), (((m >> 1) + 126) << 23) | ((m & 1) << 22))
        bits = bits | ((c & 8).to(tl.int32) << 28)
        return bits.to(tl.float32, bitcast=True)

    @triton.jit
    def _bf16(x, F32: tl.constexpr):
        if F32:
            b = x.to(tl.int32, bitcast=True)
            b = (b + 0x7FFF + ((b >> 16) & 1)) & -65536
            x = b.to(tl.float32, bitcast=True)
        return x.to(tl.bfloat16)

    @triton.jit
    def _mma(a, b, acc, F32: tl.constexpr):
        if F32:
            return tl.dot(a.to(tl.float32), b.to(tl.float32), acc, input_precision='ieee')
        return tl.dot(a, b, acc)

    @triton.jit
    def _dot_tile(acc, x_ptr, xrow, xmask, ldx, w_ptr, s_ptr, k0, gs,
                  FP4: tl.constexpr, BK: tl.constexpr, F32: tl.constexpr):
        """acc += x[xrow, k0:k0+BK] @ dequant(w[:, k0:k0+BK]).T; w_ptr/s_ptr: [BN] row pointers."""
        if FP4:
            j = k0 // 2 + tl.arange(0, BK // 2)                      # packed byte = elements 2j (low), 2j+1 (high)
            q = tl.load(w_ptr[:, None] + j[None, :])
            sc = tl.load(s_ptr[:, None] + (j // 8)[None, :]).to(tl.float32)
            lo = _bf16(_e2m1(q & 15) * sc * gs, F32)
            hi = _bf16(_e2m1(q >> 4) * sc * gs, F32)
            wt = tl.interleave(lo, hi)                               # [BN, BK]: element 2j low, 2j+1 high
        else:
            k = k0 + tl.arange(0, BK)
            q = tl.load(w_ptr[:, None] + k[None, :]).to(tl.float32)
            sc = (tl.load(s_ptr[:, None] + (k // 32)[None, :]).to(tl.int32) << 23).to(tl.float32, bitcast=True)
            wt = _bf16(q * sc, F32)
        k = k0 + tl.arange(0, BK)
        x = tl.load(x_ptr + xrow[:, None] * ldx + k[None, :], mask=xmask[:, None], other=0.)
        return _mma(x, tl.trans(wt), acc, F32)

    @triton.jit
    def _w13_kernel(u_ptr, tok_ptr, be_ptr, b0_ptr, b1_ptr, w_ptr, s_ptr, gs_ptr, act_ptr,
                    K, NI, lwe, lwr, lse, lsr,
                    FP4: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, F32: tl.constexpr):
        b = tl.program_id(0)
        e = tl.load(be_ptr + b)
        if e >= 0:
            r = tl.load(b0_ptr + b) + tl.arange(0, BM)
            rm = r < tl.load(b1_ptr + b)
            t = tl.load(tok_ptr + r, mask=rm, other=0)
            n = tl.program_id(1) * BN + tl.arange(0, BN)            # intermediate index: gate row 2n, up row 2n+1
            wg = w_ptr + e.to(tl.int64) * lwe + (2 * n) * lwr
            sg = s_ptr + e.to(tl.int64) * lse + (2 * n) * lsr
            gs = 1.0
            if FP4:
                gs = tl.load(gs_ptr + e)
            ag = tl.zeros((BM, BN), tl.float32)
            au = tl.zeros((BM, BN), tl.float32)
            for k0 in range(0, K, BK):
                ag = _dot_tile(ag, u_ptr, t, rm, K, wg, sg, k0, gs, FP4, BK, F32)
                au = _dot_tile(au, u_ptr, t, rm, K, wg + lwr, sg + lsr, k0, gs, FP4, BK, F32)
            g = _bf16(ag, F32).to(tl.float32)                   # the reference's bf16 GEMM output
            v = _bf16(au, F32).to(tl.float32)
            a = 4 * (2 * tl.sigmoid(g / 2) - 1) * tl.sigmoid(g) * 25 * (2 * tl.sigmoid(v / 25 * 2) - 1)
            tl.store(act_ptr + r[:, None] * NI + n[None, :], _bf16(a, F32), mask=rm[:, None])

    @triton.jit
    def _w2_kernel(act_ptr, tok_ptr, ws_ptr, be_ptr, b0_ptr, b1_ptr, w_ptr, s_ptr, gs_ptr, out_ptr,
                   K, N, lwe, lwr, lse, lsr,
                   FP4: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, F32: tl.constexpr):
        b = tl.program_id(0)
        e = tl.load(be_ptr + b)
        if e >= 0:
            r = tl.load(b0_ptr + b) + tl.arange(0, BM)
            rm = r < tl.load(b1_ptr + b)
            n = tl.program_id(1) * BN + tl.arange(0, BN)
            wp = w_ptr + e.to(tl.int64) * lwe + n * lwr
            sp = s_ptr + e.to(tl.int64) * lse + n * lsr
            gs = 1.0
            if FP4:
                gs = tl.load(gs_ptr + e)
            acc = tl.zeros((BM, BN), tl.float32)
            for k0 in range(0, K, BK):
                acc = _dot_tile(acc, act_ptr, r, rm, K, wp, sp, k0, gs, FP4, BK, F32)
            y = _bf16(acc, F32).to(tl.float32) * tl.load(ws_ptr + r, mask=rm, other=0.)[:, None]
            t = tl.load(tok_ptr + r, mask=rm, other=0)
            tl.atomic_add(out_ptr + t[:, None] * N + n[None, :], y, mask=rm[:, None])


def fp4act(fp4, T):
    return FP4ACT and fp4 and T >= FP4ACT_MIN


def blk(fp4, T):
    """(BB, BS) block sizes of the path grouped() takes (moe_route.route builds its tables with them)."""
    if fp4act(fp4, T):
        import moe_fp4
        return moe_fp4.BB, moe_fp4.BS
    return BB, BS


def grouped(u, sorted_e, tok, ws, w13, s13, gs13, w2, s2, gs2, fp4, fin=None, rt=None, out=None):
    """u [T,H] bf16; sorted_e/tok/ws: expert, token and fp32 routing weight per routed row, sorted by expert.
    w13/s13 [E,2*NI,.], w2/s2 [E,H,.] packed as stored; gs* = amax/denominator [E] (NVFP4) or None.
    fin(y, tok, ws, T): replaces finish on the v2 / fp4 paths (INK_MOETAIL); the atomic path ignores it.
    rt: moe_route.route() output for blk(fp4, T) (INK_MOE_ROUTE; v2 / fp4 paths only; sorted_e then unused).
    out: optional fp32 [T,H] accumulator for the v1 atomic path (returns it instead of bf16)."""
    if fp4act(fp4, u.shape[0]):
        import moe_fp4
        return moe_fp4.grouped(u, sorted_e, tok, ws, w13, s13, gs13, w2, s2, gs2, fin=fin, rt=rt)
    if V2:
        return grouped_v2(u, sorted_e, tok, ws, w13, s13, gs13, w2, s2, gs2, fp4, fin, rt)
    assert rt is None, 'INK_MOE_ROUTE needs the v2 or fp4 path'
    T, H = u.shape
    E, NI = w13.shape[0], w13.shape[1] // 2
    n_blk = len(tok) // BM + E
    be, b0, b1 = blocks(sorted_e, E, n_blk)
    tok32 = tok.int()
    act = torch.empty(len(tok), NI, dtype=torch.bfloat16, device=u.device)
    acc = torch.zeros(T, H, dtype=torch.float32, device=u.device) if out is None else out
    _w13_kernel[(n_blk, NI // BN)](u, tok32, be, b0, b1, w13, s13, gs13 if fp4 else ws, act, H, NI,
                                   w13.stride(0), w13.stride(1), s13.stride(0), s13.stride(1),
                                   FP4=fp4, BM=BM, BN=BN, BK=BK, F32=INTERP)
    _w2_kernel[(n_blk, H // BN)](act, tok32, ws, be, b0, b1, w2, s2, gs2 if fp4 else ws, acc, NI, H,  # ws: unused stand-in pointer
                                 w2.stride(0), w2.stride(1), s2.stride(0), s2.stride(1),
                                 FP4=fp4, BM=BM, BN=BN, BK=BK, F32=INTERP)
    return acc.bfloat16() if out is None else acc


# ------------------------------------------------------------------ v2 (INK_MOE_V2=1)
# Same math as grouped() (same dequant formulas, bf16 dot, fp32 accumulate in k order, bf16 GEMM outputs), faster
# schedule: (1) each expert's rows are cut into BB-row blocks plus up to (BB-1)//BS+1 tail blocks of BS rows (padding
# as with BM=BS, dequant/weight traffic per row halved at BB=128); (2) w13 runs gate and up as one dot over 2*BN
# contiguous stored rows, scales loaded once per 16/32 and broadcast; (3) w2 runs n-tiles fastest so a block's act
# rows stay in L2 for all of H (was: all blocks per n-tile, act re-streamed H/BN times); (4) no atomics: w2 writes
# the bf16 GEMM output per routed row, then _fin_kernel sums each token's 8 rows in expert order (= Model.moe's
# index_add order) with the routing weight: deterministic. +R*H*2 bytes (151 MB at a 4k chunk) while it runs.
V2 = os.environ.get('INK_MOE_V2') == '1'
FP4ACT = os.environ.get('INK_MOE_FP4ACT') == '1'               # moe_fp4.py; NVFP4 layers, prefill-sized calls only
FP4ACT_MIN = int(os.environ.get('INK_MOE_FP4ACT_MIN', 128))    # tokens per call; decode (T=1..8) stays bf16
# block rows (big, tail); INK_MOE_BB/BS for the bf16 v2 path (moe_fp4 has its own): any values give the same numbers
BB, BS = int(os.environ.get('INK_MOE_BB', 128)), int(os.environ.get('INK_MOE_BS', 64))
# per (kernel, fp4): (BN, BK, num_warps, num_stages); w13 BN = intermediate columns (2*BN stored rows per tile)
# BK 128 = grouped()'s K tiles, so the fp32 sums are grouped the same way (bit-identical under the interpreter)
CFG = {('w13', True): (64, 128, 8, 2), ('w13', False): (64, 128, 8, 2),
       ('w2', True): (128, 128, 8, 2), ('w2', False): (128, 128, 8, 2), 'fin': 256}
# MXFP8-layer tile override for a window sweep winner, e.g. INK_MOE_MX_CFG='w13=32,128,4,3;w2=128,64,8,3'
for _kv in filter(None, os.environ.get('INK_MOE_MX_CFG', '').split(';')):
    _k, _v = _kv.split('=')
    CFG[(_k, False)] = tuple(int(x) for x in _v.split(','))


def tables(sorted_e, E, R, BB=None, BS=None):
    """-> (expert, row0, row1, big) per block, int32, experts in order, each expert's BB blocks then its BS tail
    blocks; expert -1 = unused. On device, no host sync; the length depends only on R (CUDA-graph safe)."""
    BB, BS = BB or globals()['BB'], BS or globals()['BS']
    dev = sorted_e.device
    ar = torch.arange(E, device=dev)
    start = torch.searchsorted(sorted_e, ar)
    cnt = torch.searchsorted(sorted_e, ar, right=True) - start
    nbig = cnt // BB
    tail = cnt - nbig * BB
    nb = nbig + (tail + BS - 1) // BS
    n_blk = R // BB + E * ((BB - 1) // BS + 1) if BB > BS else R // BS + E
    cb = nb.cumsum(0)
    b = torch.arange(n_blk, device=dev)
    e = torch.searchsorted(cb, b, right=True)
    ec = e.clamp(max=E - 1)
    i = b - (cb[ec] - nb[ec])                                        # block index within its expert
    big = i < nbig[ec]
    r0 = start[ec] + torch.where(big, i * BB, nbig[ec] * BB + (i - nbig[ec]) * BS)
    return torch.where(e < E, e, -1).int(), r0.int(), (start + cnt)[ec].int(), big.int()


if triton is not None:
    @triton.jit
    def _dot_tile2(acc, x_ptr, xrow, xmask, ldx, w_ptr, s_ptr, k0, gs,
                   FP4: tl.constexpr, NR: tl.constexpr, BK: tl.constexpr, F32: tl.constexpr):
        """_dot_tile with the scales loaded once per 16 (NVFP4) / 32 (MXFP8) elements; same values. NR = len(w_ptr)."""
        if FP4:
            j = k0 // 2 + tl.arange(0, BK // 2)
            q = tl.load(w_ptr[:, None] + j[None, :])
            g = k0 // 16 + tl.arange(0, BK // 16)
            sc = tl.load(s_ptr[:, None] + g[None, :]).to(tl.float32)
            sc = tl.reshape(tl.broadcast_to(sc[:, :, None], (NR, BK // 16, 8)), (NR, BK // 2))
            lo = _bf16(_e2m1(q & 15) * sc * gs, F32)
            hi = _bf16(_e2m1(q >> 4) * sc * gs, F32)
            wt = tl.interleave(lo, hi)
        else:
            k = k0 + tl.arange(0, BK)
            q = tl.load(w_ptr[:, None] + k[None, :]).to(tl.float32)
            g = k0 // 32 + tl.arange(0, BK // 32)
            sc = (tl.load(s_ptr[:, None] + g[None, :]).to(tl.int32) << 23).to(tl.float32, bitcast=True)
            wt = _bf16(q * tl.reshape(tl.broadcast_to(sc[:, :, None], (NR, BK // 32, 32)), (NR, BK)), F32)
        k = k0 + tl.arange(0, BK)
        x = tl.load(x_ptr + xrow[:, None] * ldx + k[None, :], mask=xmask[:, None], other=0.)
        return _mma(x, tl.trans(wt), acc, F32)

    @triton.jit
    def _act(g, v):   # §5 activation on the fp32 values of the bf16 gate/up GEMM outputs (same formula as _w13_kernel)
        return 4 * (2 * tl.sigmoid(g / 2) - 1) * tl.sigmoid(g) * 25 * (2 * tl.sigmoid(v / 25 * 2) - 1)

    @triton.jit
    def _w13_body(u_ptr, tok_ptr, r0, r1, e, w_ptr, s_ptr, gs, act_ptr, K, NI, lwe, lwr, lse, lsr,
                  FP4: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, F32: tl.constexpr):
        r = r0 + tl.arange(0, BM)
        rm = r < r1
        t = tl.load(tok_ptr + r, mask=rm, other=0)
        j = tl.program_id(1) * 2 * BN + tl.arange(0, 2 * BN)         # stored rows: 2n gate, 2n+1 up (§7)
        wp = w_ptr + e.to(tl.int64) * lwe + j * lwr
        sp = s_ptr + e.to(tl.int64) * lse + j * lsr
        acc = tl.zeros((BM, 2 * BN), tl.float32)
        for k0 in range(0, K, BK):
            acc = _dot_tile2(acc, u_ptr, t, rm, K, wp, sp, k0, gs, FP4, 2 * BN, BK, F32)
        g, v = tl.split(tl.reshape(acc, (BM, BN, 2)))
        a = _act(_bf16(g, F32).to(tl.float32), _bf16(v, F32).to(tl.float32))
        n = tl.program_id(1) * BN + tl.arange(0, BN)
        tl.store(act_ptr + r[:, None] * NI + n[None, :], _bf16(a, F32), mask=rm[:, None])

    @triton.jit
    def _w13_v2(u_ptr, tok_ptr, be_ptr, b0_ptr, b1_ptr, bb_ptr, w_ptr, s_ptr, gs_ptr, act_ptr,
                K, NI, lwe, lwr, lse, lsr,
                FP4: tl.constexpr, BB: tl.constexpr, BS: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                F32: tl.constexpr):
        b = tl.program_id(0)
        e = tl.load(be_ptr + b)
        if e >= 0:
            r0, r1 = tl.load(b0_ptr + b), tl.load(b1_ptr + b)
            gs = 1.0
            if FP4:
                gs = tl.load(gs_ptr + e)
            if tl.load(bb_ptr + b) != 0:
                _w13_body(u_ptr, tok_ptr, r0, r1, e, w_ptr, s_ptr, gs, act_ptr, K, NI, lwe, lwr, lse, lsr,
                          FP4, BB, BN, BK, F32)
            else:
                _w13_body(u_ptr, tok_ptr, r0, r1, e, w_ptr, s_ptr, gs, act_ptr, K, NI, lwe, lwr, lse, lsr,
                          FP4, BS, BN, BK, F32)

    @triton.jit
    def _w2_body(act_ptr, r0, r1, e, w_ptr, s_ptr, gs, y_ptr, K, N, lwe, lwr, lse, lsr,
                 FP4: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, F32: tl.constexpr):
        r = r0 + tl.arange(0, BM)
        rm = r < r1
        n = tl.program_id(0) * BN + tl.arange(0, BN)
        wp = w_ptr + e.to(tl.int64) * lwe + n * lwr
        sp = s_ptr + e.to(tl.int64) * lse + n * lsr
        acc = tl.zeros((BM, BN), tl.float32)
        for k0 in range(0, K, BK):
            acc = _dot_tile2(acc, act_ptr, r, rm, K, wp, sp, k0, gs, FP4, BN, BK, F32)
        tl.store(y_ptr + r.to(tl.int64)[:, None] * N + n[None, :], _bf16(acc, F32), mask=rm[:, None])

    @triton.jit
    def _w2_v2(act_ptr, be_ptr, b0_ptr, b1_ptr, bb_ptr, w_ptr, s_ptr, gs_ptr, y_ptr,
               K, N, lwe, lwr, lse, lsr,
               FP4: tl.constexpr, BB: tl.constexpr, BS: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
               F32: tl.constexpr):
        b = tl.program_id(1)                                         # n-tiles fastest: act rows reused from L2
        e = tl.load(be_ptr + b)
        if e >= 0:
            r0, r1 = tl.load(b0_ptr + b), tl.load(b1_ptr + b)
            gs = 1.0
            if FP4:
                gs = tl.load(gs_ptr + e)
            if tl.load(bb_ptr + b) != 0:
                _w2_body(act_ptr, r0, r1, e, w_ptr, s_ptr, gs, y_ptr, K, N, lwe, lwr, lse, lsr, FP4, BB, BN, BK, F32)
            else:
                _w2_body(act_ptr, r0, r1, e, w_ptr, s_ptr, gs, y_ptr, K, N, lwe, lwr, lse, lsr, FP4, BS, BN, BK, F32)

    @triton.jit
    def _fin_kernel(y_ptr, ws_ptr, idx_ptr, out_ptr, N, KR: tl.constexpr, BN: tl.constexpr, F32: tl.constexpr):
        """out[t] = bf16(sum_k fp32(y[row_k]) * ws[row_k]), rows of token t in sorted (= expert) order."""
        t = tl.program_id(0)
        n = tl.program_id(1) * BN + tl.arange(0, BN)
        acc = tl.zeros((BN,), tl.float32)
        for k in tl.static_range(KR):
            i = tl.load(idx_ptr + t * KR + k).to(tl.int64)
            acc += tl.load(y_ptr + i * N + n).to(tl.float32) * tl.load(ws_ptr + i)
        tl.store(out_ptr + t.to(tl.int64) * N + n, _bf16(acc, F32))


def finish(y, tok, ws, T, idx=None):
    """[R,H] bf16 per routed row -> [T,H] bf16 (fixed-order fp32 sum with the routing weight). idx: precomputed
    per-token row order (moe_route fidx)."""
    R, H = y.shape
    assert R % T == 0, 'every token must route the same number of rows'
    if idx is None:
        idx = torch.sort(tok, stable=True).indices.int()            # per token, its rows in expert order
    out = torch.empty(T, H, dtype=torch.bfloat16, device=y.device)
    bf = CFG['fin']
    # no FMA contraction: product then sum, both rounded, like Model.moe's `y.float() * ws` + index_add
    _fin_kernel[(T, H // bf)](y, ws, idx, out, H, KR=R // T, BN=bf, F32=INTERP, enable_fp_fusion=False)
    return out


def grouped_v2(u, sorted_e, tok, ws, w13, s13, gs13, w2, s2, gs2, fp4, fin=None, rt=None):
    """Same arguments and math as grouped(); see the v2 comment above."""
    T, H = u.shape
    E, NI = w13.shape[0], w13.shape[1] // 2
    R = len(tok)
    be, b0, b1, bb = rt[2:6] if rt is not None else tables(sorted_e, E, R)
    tok32 = tok.int()
    act = torch.empty(R, NI, dtype=torch.bfloat16, device=u.device)
    y = torch.empty(R, H, dtype=torch.bfloat16, device=u.device)
    bn, bk, nw, ns = CFG[('w13', fp4)]
    _w13_v2[(len(be), NI // bn)](u, tok32, be, b0, b1, bb, w13, s13, gs13 if fp4 else ws, act, H, NI,
                                 w13.stride(0), w13.stride(1), s13.stride(0), s13.stride(1),
                                 FP4=fp4, BB=BB, BS=BS, BN=bn, BK=bk, F32=INTERP, num_warps=nw, num_stages=ns)
    bn, bk, nw, ns = CFG[('w2', fp4)]
    _w2_v2[(H // bn, len(be))](act, be, b0, b1, bb, w2, s2, gs2 if fp4 else ws, y, NI, H,
                               w2.stride(0), w2.stride(1), s2.stride(0), s2.stride(1),
                               FP4=fp4, BB=BB, BS=BS, BN=bn, BK=bk, F32=INTERP, num_warps=nw, num_stages=ns)
    return fin(y, tok, ws, T) if fin else finish(y, tok, ws, T, None if rt is None else rt[6])
