"""Fused single-token decode (INK_DEC_FUSE=1): one decode step in ~690 Triton launches instead of ~10.5k torch ops.

Same math as Model.forward on a DecodeState (SPEC sections as in model.py), with the same bf16 rounding points; only
reduction orders differ (fp32), so results match the torch path within bf16 noise. Per layer visit:
  attention side   mix (attn-res over the depth entries [+ the 4 loop messages]; RMSNorm -> bf16 inputs when there is
                   no loop gate) | gate_z + gate_out (loop gate MG, 3 controllers per launch) | proj (one GEMV over the
                   fused q|r|sink|gate,k[,taps],v[,taps] matrix, input norm fused when gated) | pre (q/rel clip+scale+tau,
                   K/V short conv + state, K clip, V norm, ring + 4-bit page write) | [far kernel] | near_part + near_out +
                   near_fin (1024-slot ring split over kv head x 64-slot tiles: tile max/sum, then bf16 probabilities
                   with the global lse incl. sink + far, then partials summed + far merged by LSE + output gate) | wo
                   (GEMV + attn short conv + state + residual)
  MLP side         mix [| gate_z + gate_out] | router (GEMV + sigmoid) | w13 (top-8 in every program + 8 experts' gate/up
                   from packed NVFP4/MXFP8 + activation) | w2 (grid rows x 8 experts -> per-expert
                   parts) | w2_fin (parts summed in slot order + mlp short conv + residual)
All buffers are allocated once (Fused.__init__), so a step is CUDA-graph safe; positions come from st.pos.
Depth entries (rms of the block entries) live in B [8,H] fp32: slot 0 = layer 0, pass p: 1+2p (layer 6), 2+2p
(layer 12), 7 = layer 18. Every visit writes its own output row hbuf[v], so the loop's h_pre / prev reads are plain
addresses. tests/test_decode.py checks each kernel against the torch pieces and the whole step against Decoder.
"""
import math
import mmap
import os
import json
import struct

import torch

import decode
import kvpool
from model import CORE, FULL, H, HD, NKV, NQ, VISITS, VOCAB, WIN, rms
from moe_triton import INTERP

import triton
import triton.language as tl
from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait, libdevice
from moe_triton import _bf16, _e2m1, _mma

NI = 1152
CH = 256          # vector chunk (2304 = 9 x 256, 3072 = 12 x 256)
# Block sizes (programs split the output dim). Tests raise them: the CPU interpreter runs programs one by one.
# Picked on the 5090 with tests/tune_decode.py --sweep (2026-11-22); NW = num_warps per kernel (Triton default 4).
BLK = dict(proj=8, wo=4, router=2, w13=8, w2=8, w2f=64, gate=2, gout=32, near=64, kbk=256, kbk2=128, w2a=8)
NW = dict(wo=8, router=2, w13=4, w2=2, gate=8, gout=8, w2a=2)
# Decode speed flags (all off by default; none changes the math or a rounding point). Read when a Fused /
# BatchDecoder is built (tests flip OPT between instances).
#   INK_DEC_UNROLL  fully unroll the K / H loops (all weight loads of a program in flight at once, same op order)
#   INK_DEC_PDL     Programmatic Dependent Launch: every kernel waits for its predecessor (gdc_wait) before touching
#                   activations and lets the next one launch, which prefetches its own weight slice into L2 meanwhile
#   INK_DEC_L2PF    L2 prefetch of the next big weight from the kernel before it: w13 -> its experts' w2, near ->
#                   wo, wo -> router, w2 -> the next visit's projection, proj -> this visit's ring (Fused)
#   INK_DEC_W2ALL   Fused: w2 over all 8 experts in one program per row block, slot-order sum + conv + residual in
#                   the epilogue (replaces w2 + w2_fin and the [8, H] parts round trip)
OPT = dict(UNR=os.environ.get('INK_DEC_UNROLL') == '1', PDL=os.environ.get('INK_DEC_PDL') == '1',
           PF=os.environ.get('INK_DEC_L2PF') == '1', W2ALL=os.environ.get('INK_DEC_W2ALL') == '1')


def kopts():
    """-> (constexprs every decode kernel takes, launch kwargs) for the current OPT. PDL and the prefetches are inline
    PTX: off in the CPU interpreter (they never change results)."""
    gpu = not INTERP
    c = dict(UNR=int(OPT['UNR']), PDL=OPT['PDL'] and gpu, PF=OPT['PF'] and gpu)
    return c, (dict(launch_pdl=True) if c['PDL'] else {})


@triton.jit
def _enter(PDL: tl.constexpr):
    """INK_DEC_PDL: wait until the previous kernel has finished (every activation read/write comes after this), then
    let the next kernel launch (it waits in turn, so a chain stays ordered and only one kernel runs ahead)."""
    if PDL:
        gdc_wait()
        gdc_launch_dependents()


@triton.jit
def _pf(p, n):
    """L2 prefetch of n bytes at p (16-byte aligned, n a multiple of 16; n <= 0: nothing): one bulk prefetch issued by
    thread 0, nothing waits for it. GPU only (inline PTX)."""
    tl.inline_asm_elementwise(
        "{ .reg .pred p, q; .reg .u32 t; mov.u32 t, %tid.x; setp.eq.u32 p, t, 0; setp.gt.s32 q, $2, 0;\n"
        "and.pred p, p, q; @p cp.async.bulk.prefetch.L2.global [$1], $2; mov.u32 $0, 0; }",
        "=r,l,r", [p.to(tl.pointer_type(tl.uint8)).to(tl.int64, bitcast=True), n], dtype=tl.int32, is_pure=False,
        pack=1)


@triton.jit
def _pf_part(p, n, pid, npg):
    """Program pid of npg prefetches its 128-byte-aligned share of the n bytes at p (n a multiple of 16)."""
    ch = (tl.cdiv(n, npg) + 127) // 128 * 128
    off = pid * ch
    _pf(p.to(tl.pointer_type(tl.uint8)) + off, tl.minimum(ch, n - off))


@triton.jit
def _r(x, F32: tl.constexpr):
    """Round fp32 to bf16 (nearest even) and back."""
    return _bf16(x, F32).to(tl.float32)


@triton.jit
def _rint(x):  # round half to even (torch.round)
    r = tl.floor(x + 0.5)
    return tl.where((r - x == 0.5) & (r - 2 * tl.floor(r * 0.5) != 0), r - 1, r)


@triton.jit
def _tanh(x):
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def _inv_rms(p, N: tl.constexpr, CH: tl.constexpr, UNR: tl.constexpr = 0):
    c = tl.arange(0, CH)
    ss = tl.zeros([CH], tl.float32)
    for k0 in tl.range(0, N, CH, loop_unroll_factor=UNR * (N // CH)):
        x = tl.load(p + k0 + c).to(tl.float32)
        ss += x * x
    return 1 / tl.sqrt(tl.sum(ss, 0) / N + 1e-6)


# --------------------------------------------------------------------------------------------- attn-res mix (§4)
@triton.jit
def _mix_kernel(q_ptr, B, eidx, n, x_ptr, bslot, a_ptr, xo_ptr, nwt,
                qm_ptr, midx, nm, xm_ptr, m_ptr, nq,
                FIRST: tl.constexpr, NORM: tl.constexpr, STORE_A: tl.constexpr, F32: tl.constexpr,
                H: tl.constexpr, CH: tl.constexpr, UNR: tl.constexpr = 0, PDL: tl.constexpr = False,
                PF: tl.constexpr = False):
    """Program j < nq: a_j = attn_res(q_j, B[eidx[:n]] + [rms(x)]) (FIRST: a = x); STORE_A -> a_ptr[j];
    NORM -> xo_ptr[j] = bf16(rms(a_j) * nw_j) (nw_j from the pointer table nwt); bslot >= 0: program 0 writes
    B[bslot] = rms(x). Program nq + i (loop messages, nm = 1): m_i = attn_res(qm_i, [B[midx[0]], rms(xm)]) -> m_ptr[i].
    x and xm must have the same dtype."""
    _enter(PDL)
    pid = tl.program_id(0)
    c = tl.arange(0, CH)
    ii = tl.arange(0, 8)
    msg = pid >= nq
    j = tl.where(msg, pid - nq, pid)
    if msg:
        qp = qm_ptr + j * H
        xp = xm_ptr
        ei = midx
        ne = nm
        op = m_ptr + j * H
    else:
        qp = q_ptr + j * H
        xp = x_ptr
        ei = eidx
        ne = n
        op = a_ptr + j * H
    store = (msg.to(tl.int32) + STORE_A) > 0
    inv = _inv_rms(xp, H, CH, UNR)
    rows = tl.load(ei + ii, mask=ii < ne, other=0)
    em = (ii < ne)[:, None]
    dots = tl.zeros([8], tl.float32)
    dv = tl.zeros([CH], tl.float32)
    if not FIRST:
        for k0 in tl.range(0, H, CH, loop_unroll_factor=UNR * (H // CH)):
            qv = tl.load(qp + k0 + c)
            S = tl.load(B + rows[:, None] * H + k0 + c[None, :], mask=em, other=0.)
            dots += tl.sum(S * qv[None, :], 1)
            dv += tl.load(xp + k0 + c).to(tl.float32) * inv * qv
    sq = math.sqrt(H)
    lg = tl.where(ii < ne, dots / sq, float('-inf'))
    ln = tl.sum(dv, 0) / sq
    mx = tl.maximum(tl.max(lg, 0), ln)
    e = tl.exp(lg - mx)
    en = tl.exp(ln - mx)
    tot = tl.sum(e, 0) + en
    w = e / tot
    wn = en / tot
    so = tl.zeros([CH], tl.float32)
    for k0 in tl.range(0, H, CH, loop_unroll_factor=UNR * (H // CH)):
        xv = tl.load(xp + k0 + c).to(tl.float32)
        if FIRST:
            o = xv
        else:
            S = tl.load(B + rows[:, None] * H + k0 + c[None, :], mask=em, other=0.)
            o = tl.sum(S * w[:, None], 0) + wn * (xv * inv)
        if store:
            tl.store(op + k0 + c, o)
        if (pid == 0) & (bslot >= 0):
            tl.store(B + bslot * H + k0 + c, xv * inv)
        so += o * o
    if NORM:
        if not msg:
            io = 1 / tl.sqrt(tl.sum(so, 0) / H + 1e-6)
            nw = tl.load(nwt + j).to(tl.pointer_type(tl.float32))
            for k0 in tl.range(0, H, CH, loop_unroll_factor=UNR * (H // CH)):
                xv = tl.load(xp + k0 + c).to(tl.float32)
                if FIRST:
                    o = xv
                else:
                    S = tl.load(B + rows[:, None] * H + k0 + c[None, :], mask=em, other=0.)
                    o = tl.sum(S * w[:, None], 0) + wn * (xv * inv)
                tl.store(xo_ptr + j * H + k0 + c, _bf16(o * io * tl.load(nw + k0 + c), F32))


# --------------------------------------------------------------------------------------------- loop gate MG (§6)
@triton.jit
def _gate_z_kernel(a_ptr, xs, m_ptr, dtab, z_ptr, F32: tl.constexpr, H: tl.constexpr, CH: tl.constexpr,
                   ZB: tl.constexpr, UNR: tl.constexpr = 0, PDL: tl.constexpr = False, PF: tl.constexpr = False):
    """z_g = bf16(cat(rms(a_g), rms(msg_g)).bf16 @ down_g^T), 64 values per gate; a_g = a_ptr + g*xs, msg_g = m_ptr + g*H."""
    g, zb = tl.program_id(0), tl.program_id(1)
    w = tl.load(dtab + g).to(tl.pointer_type(tl.bfloat16))      # (pointer table: static)
    r = zb * ZB + tl.arange(0, ZB)
    if PDL:
        _pf(w + zb * ZB * (2 * H), ZB * 4 * H)
    _enter(PDL)
    xp = a_ptr + g * xs
    mp = m_ptr + g * H
    ix = _inv_rms(xp, H, CH, UNR)
    im = _inv_rms(mp, H, CH, UNR)
    c = tl.arange(0, CH)
    acc = tl.zeros([ZB, CH], tl.float32)
    for k0 in tl.range(0, H, CH, loop_unroll_factor=UNR * (H // CH)):
        xb = _r(tl.load(xp + k0 + c) * ix, F32)
        acc += tl.load(w + r[:, None] * (2 * H) + k0 + c[None, :]).to(tl.float32) * xb[None, :]
        mb = _r(tl.load(mp + k0 + c) * im, F32)
        acc += tl.load(w + r[:, None] * (2 * H) + H + k0 + c[None, :]).to(tl.float32) * mb[None, :]
    tl.store(z_ptr + g * 64 + r, _r(tl.sum(acc, 1), F32))


@triton.jit
def _gate_out_kernel(a_ptr, xs, m_ptr, gtab, z_ptr, o_ptr, H: tl.constexpr, BH: tl.constexpr, UNR: tl.constexpr = 0, PDL: tl.constexpr = False, PF: tl.constexpr = False):
    """[f|s] = tanh(z @ gcat^T); out_g = (1 - f) * a_g + s * msg_g (fp32)."""
    g, hb = tl.program_id(0), tl.program_id(1)
    ch = hb * BH + tl.arange(0, BH)
    k = tl.arange(0, 64)
    gc = tl.load(gtab + g).to(tl.pointer_type(tl.float32))
    if PDL:
        _pf(gc + hb * BH * 64, BH * 256)
        _pf(gc + (H + hb * BH) * 64, BH * 256)
    _enter(PDL)
    z = tl.load(z_ptr + g * 64 + k)
    f = _tanh(tl.sum(tl.load(gc + ch[:, None] * 64 + k[None, :]) * z[None, :], 1))
    s = _tanh(tl.sum(tl.load(gc + (H + ch)[:, None] * 64 + k[None, :]) * z[None, :], 1))
    x = tl.load(a_ptr + g * xs + ch)
    m = tl.load(m_ptr + g * H + ch)
    tl.store(o_ptr + g * H + ch, (1 - f) * x + s * m)


# --------------------------------------------------------------------------------------------- projections
@triton.jit
def _proj_kernel(w_ptr, n0, n1, x_ptr, xs, nwt, y_ptr, s_ptr, NORM: tl.constexpr, F32: tl.constexpr,
                 H: tl.constexpr, CH: tl.constexpr, BN: tl.constexpr, W8: tl.constexpr = False, UNR: tl.constexpr = 0, PDL: tl.constexpr = False, PF: tl.constexpr = False,
                 pfa=None, pfn=0, pfb=None, pfm=0):
    """y = bf16(x_j @ W^T) over the fused attention matrix; rows [0,n0) use input 0, [n0,n0+n1) input 1, the rest 2
    (BN divides n0 and n1, so a program has one input). NORM: x_j = bf16(rms(x_ptr + j*xs) * nw_j) (fp32 input),
    else x_j = x_ptr + j*xs (bf16). W8 (INK_FP8_DECODE): W is e4m3, row n's fp32 sum is multiplied by s_ptr[n].
    PF: prefetch this program's share of the pfn + pfm bytes at pfa / pfb (the visit's ring K / V)."""
    pid = tl.program_id(0)
    r0 = pid * BN
    if PDL:
        _pf(w_ptr + r0 * H, BN * H * (1 if W8 else 2))
    _enter(PDL)
    if PF:
        _pf_part(pfa, pfn, pid, tl.num_programs(0))
        _pf_part(pfb, pfm, pid, tl.num_programs(0))
    j = (r0 >= n0).to(tl.int32) + (r0 >= n0 + n1).to(tl.int32)
    xp = x_ptr + j * xs
    r = r0 + tl.arange(0, BN)
    c = tl.arange(0, CH)
    inv = 1.
    if NORM:
        inv = _inv_rms(xp, H, CH, UNR)
        nw = tl.load(nwt + j).to(tl.pointer_type(tl.float32))
    acc = tl.zeros([BN, CH], tl.float32)
    for k0 in tl.range(0, H, CH, loop_unroll_factor=UNR * (H // CH)):
        if NORM:
            xb = _r(tl.load(xp + k0 + c) * inv * tl.load(nw + k0 + c), F32)
        else:
            xb = tl.load(xp + k0 + c).to(tl.float32)
        acc += tl.load(w_ptr + r[:, None] * H + k0 + c[None, :]).to(tl.float32) * xb[None, :]
    y = tl.sum(acc, 1)
    if W8:
        y = y * tl.load(s_ptr + r)
    tl.store(y_ptr + r, _bf16(y, F32))


# --------------------------------------------------------------------------------------------- attention (§3.2)
@triton.jit
def _conv4(st, ch, C: tl.constexpr, x, w0, w1, w2, w3, div):
    """Causal kernel-4 conv for one token on channels ch, state st [3,C] bf16 (the last 3 inputs), updated in place.
    Returns the fp32 value before the bf16 rounding."""
    s0 = tl.load(st + ch).to(tl.float32)
    s1 = tl.load(st + C + ch).to(tl.float32)
    s2 = tl.load(st + 2 * C + ch).to(tl.float32)
    y = (s0 * w0 + s1 * w1 + s2 * w2 + x * w3 + x) / div
    tl.store(st + ch, s1.to(tl.bfloat16))
    tl.store(st + C + ch, s2.to(tl.bfloat16))
    tl.store(st + 2 * C + ch, x.to(tl.bfloat16))
    return y


@triton.jit
def _e4m3(x):
    """Round x >= 0 (< 448) to the nearest e4m3 value, ties to even, as fp32 (torch's .to(float8_e4m3fn)). Done by hand:
    the interpreter's fp8 cast rounds exact ties up, and ties are common here (amax of bf16 values / 7)."""
    b = x.to(tl.int32, bitcast=True)
    nrm = ((b + 0x7FFFF + ((b >> 20) & 1)) & -1048576).to(tl.float32, bitcast=True)   # keep 3 mantissa bits
    return tl.where(x < 0.015625, _rint(x * 512.) / 512., nrm)                      # subnormals: steps of 2^-9


@triton.jit
def _quant_put(x, c_ptr, s_ptr, row, h, F32: tl.constexpr):
    """KV4.quant of one head's 128 values (fp32, already bf16-exact) -> codes/scales at page row `row`, kv head h."""
    a = tl.max(tl.reshape(tl.abs(x), (8, 16)), 1)
    s8 = _e4m3(a / 7).to(tl.float8e4nv)                         # exact cast: already an e4m3 value
    sf = tl.reshape(tl.broadcast_to(s8.to(tl.float32)[:, None], (8, 16)), (128,))
    q = tl.where(sf > 0, x / tl.where(sf > 0, sf, 1.), 0.)
    q = (tl.minimum(tl.maximum(_rint(q), -8.), 7.) + 8).to(tl.uint8)
    lo, hi = tl.split(tl.reshape(q, (64, 2)))
    tl.store(c_ptr + row * 512 + h * 64 + tl.arange(0, 64), lo | (hi << 4))
    tl.store(s_ptr + row * 64 + h * 8 + tl.arange(0, 8), s8.to(tl.uint8, bitcast=True))


@triton.jit
def _pre_kernel(y_ptr, koff, voff, tkoff, tvoff, qsc, rsc, kw, vw, stk, stv, rk, rv, rpos, pos_ptr,
                c0, c1, s0, s1, tbl, qo, ro, FULL: tl.constexpr, F32: tl.constexpr, PG: tl.constexpr, UNR: tl.constexpr = 0, PDL: tl.constexpr = False, PF: tl.constexpr = False):
    """Program = kv head h: Q (3 heads) and R clip/scale[/tau], K/V short conv (static kw/vw [1024,4] or dynamic:
    static_w [4,1024] + taps from y) with state update, K l2clip, V rms, ring write at pos % 1024, 4-bit page write."""
    _enter(PDL)
    h = tl.program_id(0)
    pos = tl.load(pos_ptr)
    slot = pos % 1024
    d = tl.arange(0, 128)
    g = tl.arange(0, 4)
    gm = (g < 3)[:, None]
    hq = h * 3 + g
    tau = 1.
    if FULL:
        tau = 1 + 0.1 * tl.log(tl.maximum((pos + 1).to(tl.float32) / 8192, 1.))
    q = tl.load(y_ptr + hq[:, None] * 128 + d[None, :], mask=gm, other=0.).to(tl.float32)
    q = q * tl.load(qsc + hq[:, None] * 128 + d[None, :], mask=gm, other=0.)
    q = _r(q * tl.minimum(22.627416997969522 / (tl.sqrt(tl.sum(q * q, 1)) + 1e-6), 1.)[:, None], F32)
    if FULL:
        q = _r(q * tau, F32)
    tl.store(qo + hq[:, None] * 128 + d[None, :], q.to(tl.bfloat16), mask=gm)
    j = tl.arange(0, 16)
    r = tl.load(y_ptr + 3072 + hq[:, None] * 16 + j[None, :], mask=gm, other=0.).to(tl.float32)
    r = r * tl.load(rsc + hq[:, None] * 16 + j[None, :], mask=gm, other=0.)
    r = _r(r * tl.minimum(8.0 / (tl.sqrt(tl.sum(r * r, 1)) + 1e-6), 1.)[:, None], F32)
    if FULL:
        r = _r(r * tau, F32)
    tl.store(ro + hq[:, None] * 16 + j[None, :], r.to(tl.bfloat16), mask=gm)
    ch = h * 128 + d
    kx = tl.load(y_ptr + koff + ch).to(tl.float32)
    vx = tl.load(y_ptr + voff + ch).to(tl.float32)
    if FULL:  # dynamic conv: W[c, i] = static_w[3-i, c] + taps[h, 3-i]
        tk = y_ptr + tkoff + h * 4
        tv = y_ptr + tvoff + h * 4
        k = _conv4(stk, ch, 1024, kx,
                   tl.load(kw + 3 * 1024 + ch) + tl.load(tk + 3).to(tl.float32),
                   tl.load(kw + 2 * 1024 + ch) + tl.load(tk + 2).to(tl.float32),
                   tl.load(kw + 1024 + ch) + tl.load(tk + 1).to(tl.float32),
                   tl.load(kw + ch) + tl.load(tk).to(tl.float32), 3.0)
        v = _conv4(stv, ch, 1024, vx,
                   tl.load(vw + 3 * 1024 + ch) + tl.load(tv + 3).to(tl.float32),
                   tl.load(vw + 2 * 1024 + ch) + tl.load(tv + 2).to(tl.float32),
                   tl.load(vw + 1024 + ch) + tl.load(tv + 1).to(tl.float32),
                   tl.load(vw + ch) + tl.load(tv).to(tl.float32), 3.0)
    else:
        k = _conv4(stk, ch, 1024, kx, tl.load(kw + ch * 4), tl.load(kw + ch * 4 + 1), tl.load(kw + ch * 4 + 2),
                   tl.load(kw + ch * 4 + 3), 2.23606797749979)
        v = _conv4(stv, ch, 1024, vx, tl.load(vw + ch * 4), tl.load(vw + ch * 4 + 1), tl.load(vw + ch * 4 + 2),
                   tl.load(vw + ch * 4 + 3), 2.23606797749979)
    k = _r(k, F32)
    k = _r(k * tl.minimum(22.627416997969522 / (tl.sqrt(tl.sum(k * k, 0)) + 1e-6), 1.), F32)
    v = _r(v, F32)
    v = _r(v / tl.sqrt(tl.sum(v * v, 0) / 128 + 1e-6), F32)
    tl.store(rk + slot * 1024 + ch, k.to(tl.bfloat16))
    tl.store(rv + slot * 1024 + ch, v.to(tl.bfloat16))
    if h == 0:
        tl.store(rpos + slot, pos)
    if FULL:
        row = tl.load(tbl + pos // PG) * PG + pos % PG
        _quant_put(k, c0, s0, row, h, F32)
        _quant_put(v, c1, s1, row, h, F32)


@triton.jit
def _near_tile(q, R, proj, rk, rpos, pos, h, s0, BLK: tl.constexpr, F32: tl.constexpr):
    """Scores [16, BLK] of ring slots s0..s0+BLK (bf16 rounding points of Model.near_merge), -inf where masked."""
    d = tl.arange(0, 128)
    j = tl.arange(0, 16)
    sl = s0 + tl.arange(0, BLK)
    rp = tl.load(rpos + sl)
    dd = pos - rp
    ok = (rp >= 0) & (dd >= 0) & (dd < 1024)
    dist = tl.minimum(tl.maximum(dd, 0), 1023)
    k = tl.load(rk + sl[:, None] * 1024 + h * 128 + d[None, :])
    S = _r(_r(_mma(q, tl.trans(k), tl.zeros([16, BLK], tl.float32), F32), F32) * 0.08838834764831845, F32)
    P = tl.load(proj + j[:, None] * 1024 + dist[None, :])
    S = _r(S + _r(_mma(R, P, tl.zeros([16, BLK], tl.float32), F32), F32), F32)
    return tl.where(ok[None, :], S, float('-inf'))


@triton.jit
def _near_qr(qb, rb, h):
    d = tl.arange(0, 128)
    g = tl.arange(0, 16)
    j = tl.arange(0, 16)
    gm = (g < 3)[:, None]
    hq = h * 3 + g
    return (tl.load(qb + hq[:, None] * 128 + d[None, :], mask=gm, other=0.),
            tl.load(rb + hq[:, None] * 16 + j[None, :], mask=gm, other=0.))


@triton.jit
def _near_part_kernel(qb, rb, proj, rk, rpos, pos_ptr, pm, pl, F32: tl.constexpr, BLK: tl.constexpr, UNR: tl.constexpr = 0, PDL: tl.constexpr = False, PF: tl.constexpr = False,
                      pfa=None, pfn=0):
    """Split near attention, pass 1. Program (kv head h, ring tile s): max and sum of exp of its BLK slots' scores
    -> pm/pl [NS, 24]. PF: prefetch this program's share of the pfn bytes at pfa (wo)."""
    h, s = tl.program_id(0), tl.program_id(1)
    _enter(PDL)
    if PF:
        _pf_part(pfa, pfn, h * tl.num_programs(1) + s, tl.num_programs(0) * tl.num_programs(1))
    q, R = _near_qr(qb, rb, h)
    S = _near_tile(q, R, proj, rk, rpos, tl.load(pos_ptr), h, s * BLK, BLK, F32)
    m = tl.max(S, 1)
    ms = tl.where(m == float('-inf'), 0., m)
    g = tl.arange(0, 16)
    tl.store(pm + s * 24 + h * 3 + g, m, mask=g < 3)
    tl.store(pl + s * 24 + h * 3 + g, tl.sum(tl.exp(S - ms[:, None]), 1), mask=g < 3)


@triton.jit
def _far_lse(fl, ns, hq, gm, FAR: tl.constexpr, NSF: tl.constexpr):
    """Total lse [16] of the far partials fl [ns, 24] for q heads hq (-inf: none). NSF > 0 (INK_DEC_FAR2, ns <= NSF):
    one vector load and a max/sum instead of the serial merge."""
    tot = tl.full([16], float('-inf'), tl.float32)
    if FAR:
        if NSF > 0:
            sp = tl.arange(0, NSF)
            LS = tl.load(fl + sp[:, None] * 24 + hq[None, :], mask=(sp < ns)[:, None] & gm[None, :],
                         other=float('-inf'))
            fm = tl.max(LS, 0)
            fms = tl.where(fm == float('-inf'), 0., fm)
            fs = tl.sum(tl.exp(LS - fms[None, :]), 0)
            tot = tl.where(fs > 0, fms + tl.log(tl.where(fs > 0, fs, 1.)), float('-inf'))
        else:
            fs = tl.zeros([16], tl.float32)
            for sp in range(0, ns):
                ls = tl.load(fl + sp * 24 + hq, mask=gm, other=float('-inf'))
                nt = tl.maximum(tot, ls)
                nts = tl.where(nt == float('-inf'), 0., nt)
                fs = fs * tl.exp(tot - nts) + tl.exp(ls - nts)
                tot = nt
            fok = fs > 0
            tot = tl.where(fok, tot + tl.log(tl.where(fok, fs, 1.)), float('-inf'))
    return tot


@triton.jit
def _near_out_kernel(qb, rb, proj, y_ptr, rk, rv, rpos, pos_ptr, pm, pl, fl, ns, pacc, plse,
                     FAR: tl.constexpr, F32: tl.constexpr, BLK: tl.constexpr, NS: tl.constexpr, UNR: tl.constexpr = 0, PDL: tl.constexpr = False, PF: tl.constexpr = False,
                     NSF: tl.constexpr = 0):
    """Pass 2. Program (h, s): lse = ring tiles (pm/pl) + sink + far splits (fl) merged; its tile's
    bf16(exp(S - lse)) @ V -> pacc[s] (fp32); program s == 0 stores lse -> plse [24]."""
    _enter(PDL)
    h, s = tl.program_id(0), tl.program_id(1)
    g = tl.arange(0, 16)
    gm = g < 3
    hq = h * 3 + g
    t = tl.arange(0, NS)
    tm = gm[None, :]
    M = tl.load(pm + t[:, None] * 24 + hq[None, :], mask=tm, other=0.)
    L = tl.load(pl + t[:, None] * 24 + hq[None, :], mask=tm, other=0.)
    mx = tl.max(M, 0)                                               # finite: the current token is always in the ring
    lse = mx + tl.log(tl.sum(L * tl.exp(M - mx[None, :]), 0))
    sk = tl.load(y_ptr + 3456 + hq, mask=gm, other=0.).to(tl.float32)
    tot = _far_lse(fl, ns, hq, gm, FAR, NSF)
    mx = tl.maximum(tl.maximum(lse, sk), tot)
    lse = mx + tl.log(tl.exp(lse - mx) + tl.exp(sk - mx) + tl.exp(tot - mx))
    if s == 0:
        tl.store(plse + hq, lse, mask=gm)
    q, R = _near_qr(qb, rb, h)
    S = _near_tile(q, R, proj, rk, rpos, tl.load(pos_ptr), h, s * BLK, BLK, F32)
    p = _bf16(tl.exp(S - lse[:, None]), F32)
    d = tl.arange(0, 128)
    sl = s * BLK + tl.arange(0, BLK)
    acc = _mma(p, tl.load(rv + sl[:, None] * 1024 + h * 128 + d[None, :]), tl.zeros([16, 128], tl.float32), F32)
    tl.store(pacc + (s * 24 + hq)[:, None] * 128 + d[None, :], acc, mask=gm[:, None])


@triton.jit
def _near_fin_kernel(pacc, plse, fo, fl, ns, y_ptr, out, FAR: tl.constexpr, F32: tl.constexpr, NS: tl.constexpr, UNR: tl.constexpr = 0, PDL: tl.constexpr = False, PF: tl.constexpr = False,
                     NSF: tl.constexpr = 0):
    """Program = query head: o = bf16(sum of the tiles' partials) [+ far partials merged by LSE] * sigmoid(gate)."""
    _enter(PDL)
    j = tl.program_id(0)
    d = tl.arange(0, 128)
    acc = tl.zeros([128], tl.float32)
    for s in range(NS):
        acc += tl.load(pacc + (s * 24 + j) * 128 + d)
    o = _r(acc, F32)
    if FAR and NSF > 0:                         # INK_DEC_FAR2: o += sum_sp o_sp * exp(lse_sp - lse), vector loads
        sp = tl.arange(0, NSF)
        LS = tl.load(fl + sp * 24 + j, mask=sp < ns, other=float('-inf'))
        FO = tl.load(fo + (sp * 24 + j)[:, None] * 128 + d[None, :], mask=(sp < ns)[:, None], other=0.)
        o += tl.sum(FO * tl.exp(LS - tl.load(plse + j))[:, None], 0)
    elif FAR:
        lse = tl.load(plse + j)
        fa = tl.zeros([128], tl.float32)
        fs = 0.
        tot = float('-inf')
        for sp in range(0, ns):
            ls = tl.load(fl + sp * 24 + j)
            nt = tl.maximum(tot, ls)
            nts = tl.where(nt == float('-inf'), 0., nt)
            a = tl.exp(tot - nts)
            b = tl.exp(ls - nts)
            fs = fs * a + b
            fa = fa * a + tl.load(fo + (sp * 24 + j) * 128 + d) * b
            tot = nt
        fok = fs > 0
        fa = fa / tl.where(fok, fs, 1.)
        tot = tl.where(fok, tot + tl.log(tl.where(fok, fs, 1.)), float('-inf'))
        o += fa * tl.exp(tot - lse)
    gt = tl.sigmoid(tl.load(y_ptr + 3480 + j * 128 + d).to(tl.float32))
    tl.store(out + j * 128 + d, _bf16(o * gt, F32))


@triton.jit
def _wo_kernel(x_ptr, w_ptr, st, cw, h_ptr, r_ptr, s_ptr, BND: tl.constexpr, F32: tl.constexpr, H: tl.constexpr,
               CH: tl.constexpr, BN: tl.constexpr, W8: tl.constexpr = False, UNR: tl.constexpr = 0, PDL: tl.constexpr = False, PF: tl.constexpr = False, pfa=None, pfn=0):
    """o = bf16(x [3072] @ wo^T) (W8: e4m3 wo, row sums times s_ptr); attn short conv (static, state update);
    r = o (layer % 6 == 0) or h + o (fp32). PF: prefetch this program's share of the pfn bytes at pfa (router)."""
    pid = tl.program_id(0)
    if PDL:
        _pf(w_ptr + pid * BN * 3072, BN * 3072 * (1 if W8 else 2))
    _enter(PDL)
    if PF:
        _pf_part(pfa, pfn, pid, tl.num_programs(0))
    r = pid * BN + tl.arange(0, BN)
    c = tl.arange(0, CH)
    acc = tl.zeros([BN, CH], tl.float32)
    for k0 in tl.range(0, 3072, CH, loop_unroll_factor=UNR * (3072 // CH)):
        acc += tl.load(w_ptr + r[:, None] * 3072 + k0 + c[None, :]).to(tl.float32) * \
            tl.load(x_ptr + k0 + c).to(tl.float32)[None, :]
    o = tl.sum(acc, 1)
    if W8:
        o = o * tl.load(s_ptr + r)
    o = _r(o, F32)
    y = _r(_conv4(st, r, H, o, tl.load(cw + r * 4), tl.load(cw + r * 4 + 1), tl.load(cw + r * 4 + 2),
                  tl.load(cw + r * 4 + 3), 2.23606797749979), F32)
    if not BND:
        y = tl.load(h_ptr + r).to(tl.float32) + y
    tl.store(r_ptr + r, y)


# --------------------------------------------------------------------------------------------- MoE (§5, §7)
@triton.jit
def _router_kernel(x_ptr, nw, u_ptr, gw, s_ptr, NORM: tl.constexpr, F32: tl.constexpr, H: tl.constexpr,
                   CH: tl.constexpr, RB: tl.constexpr, UNR: tl.constexpr = 0, PDL: tl.constexpr = False, PF: tl.constexpr = False):
    """s = sigmoid(u @ gate^T) for RB experts; NORM: u = bf16(rms(x) * nw) from fp32 x (program 0 stores u)."""
    pid = tl.program_id(0)
    if PDL:
        _pf(gw + pid * RB * H, RB * H * 4)
    _enter(PDL)
    e = pid * RB + tl.arange(0, RB)
    c = tl.arange(0, CH)
    inv = 1.
    if NORM:
        inv = _inv_rms(x_ptr, H, CH, UNR)
    acc = tl.zeros([RB, CH], tl.float32)
    for k0 in tl.range(0, H, CH, loop_unroll_factor=UNR * (H // CH)):
        if NORM:
            u = _r(tl.load(x_ptr + k0 + c) * inv * tl.load(nw + k0 + c), F32)
            if pid == 0:
                tl.store(u_ptr + k0 + c, u.to(tl.bfloat16))
        else:
            u = tl.load(u_ptr + k0 + c).to(tl.float32)
        acc += tl.load(gw + e[:, None] * H + k0 + c[None, :]) * u[None, :]
    tl.store(s_ptr + e, tl.sigmoid(tl.sum(acc, 1)))


@triton.jit
def _deq_dot(acc, w_ptr, s_ptr, rows, lwr, lsr, x_ptr, k0, gs, FP4: tl.constexpr, BK: tl.constexpr,
             F32: tl.constexpr):
    """acc[BN] += dequant(w[rows, k0:k0+BK]) . x[k0:k0+BK] (weights rounded to bf16 exactly like Weights.expert)."""
    if FP4:
        jb = k0 // 2 + tl.arange(0, BK // 2)
        q = tl.load(w_ptr + rows[:, None] * lwr + jb[None, :])
        sc = tl.load(s_ptr + rows[:, None] * lsr + (jb // 8)[None, :]).to(tl.float32)
        xe = tl.load(x_ptr + 2 * jb).to(tl.float32)
        xo = tl.load(x_ptr + 2 * jb + 1).to(tl.float32)
        acc += tl.sum(_r(_e2m1(q & 15) * sc * gs, F32) * xe[None, :] + _r(_e2m1(q >> 4) * sc * gs, F32) * xo[None, :], 1)
    else:
        k = k0 + tl.arange(0, BK)
        q = tl.load(w_ptr + rows[:, None] * lwr + k[None, :]).to(tl.float32)
        sc = (tl.load(s_ptr + rows[:, None] * lsr + (k // 32)[None, :]).to(tl.int32) << 23).to(tl.float32, bitcast=True)
        acc += tl.sum(_r(q * sc, F32) * tl.load(x_ptr + k).to(tl.float32)[None, :], 1)
    return acc


@triton.jit
def _route_top8_kernel(s_ptr, bias, idx_ptr, wt_ptr, PDL: tl.constexpr = False):
    """1-program top-8 router (exact match to _w13_kernel's selection): scores = s + bias, lowest index wins ties."""
    _enter(PDL)
    ee = tl.arange(0, 128)
    sv = tl.load(s_ptr + ee)
    sc = sv + tl.load(bias + ee)
    i8 = tl.arange(0, 8)
    ids = tl.zeros([8], tl.int32)
    ws = tl.zeros([8], tl.float32)
    for i in tl.static_range(8):
        mx = tl.max(sc, 0)
        jj = tl.min(tl.where(sc == mx, ee, 128), 0)
        ids = tl.where(i8 == i, jj, ids)
        ws = tl.where(i8 == i, tl.sum(tl.where(ee == jj, sv, 0.), 0), ws)
        sc = tl.where(ee == jj, float('-inf'), sc)
    wn = ws / (tl.sum(ws, 0) + 1e-20) * 8.0
    tl.store(idx_ptr + i8, ids)
    tl.store(wt_ptr + i8, wn)


@triton.jit
def _w13_kernel(s_ptr, bias, u_ptr, w_ptr, sc_ptr, gs_ptr, lwe, lwr, lse, lsr, act, idx_ptr, wt_ptr,
                FP4: tl.constexpr, F32: tl.constexpr, H: tl.constexpr, NI: tl.constexpr, BN: tl.constexpr,
                BK: tl.constexpr, UNR: tl.constexpr = 0, PDL: tl.constexpr = False, PF: tl.constexpr = False,
                pfa=None, pfn=0, pfb=None, pfm=0, PRE_ROUTED: tl.constexpr = False):
    """Program (n block, slot t): top-8 of sigmoid scores + bias (recomputed by every program, 128 values), expert of
    slot t (PRE_ROUTED: slot t indexes directly into the 8 staged experts, skipping routing + idx/wt stores);
    gate/up rows (2n, 2n+1) dot u; act = bf16(softcap-SwiGLU). Program (0,0) stores the 8 ids + weights.
    PF: prefetch this program's share of expert e's w2 (pfn bytes per expert at pfa) and its scales (pfm at pfb)."""
    _enter(PDL)
    nb, t = tl.program_id(0), tl.program_id(1)
    if PRE_ROUTED:
        e = t.to(tl.int64)
    else:
        ee = tl.arange(0, 128)
        sv = tl.load(s_ptr + ee)
        sc = sv + tl.load(bias + ee)
        i8 = tl.arange(0, 8)
        ids = tl.zeros([8], tl.int32)
        ws = tl.zeros([8], tl.float32)
        for i in tl.static_range(8):
            mx = tl.max(sc, 0)
            jj = tl.min(tl.where(sc == mx, ee, 128), 0)
            ids = tl.where(i8 == i, jj, ids)
            ws = tl.where(i8 == i, tl.sum(tl.where(ee == jj, sv, 0.), 0), ws)
            sc = tl.where(ee == jj, float('-inf'), sc)
        wn = ws / (tl.sum(ws, 0) + 1e-20) * 8.0
        if nb == 0 and t == 0:
            tl.store(idx_ptr + i8, ids)
            tl.store(wt_ptr + i8, wn)
        e = tl.sum(tl.where(i8 == t, ids, 0), 0).to(tl.int64)
    if PF:
        _pf_part(pfa.to(tl.pointer_type(tl.uint8)) + e * pfn, pfn, nb, tl.num_programs(0))
        _pf_part(pfb.to(tl.pointer_type(tl.uint8)) + e * pfm, pfm, nb, tl.num_programs(0))
    n = nb * BN + tl.arange(0, BN)
    gs = 1.
    if FP4:
        gs = tl.load(gs_ptr + e)
    wp = w_ptr + e * lwe
    sp = sc_ptr + e * lse
    ag = tl.zeros([BN], tl.float32)
    au = tl.zeros([BN], tl.float32)
    for k0 in tl.range(0, H, BK, loop_unroll_factor=UNR * (H // BK)):
        ag = _deq_dot(ag, wp, sp, 2 * n, lwr, lsr, u_ptr, k0, gs, FP4, BK, F32)
        au = _deq_dot(au, wp, sp, 2 * n + 1, lwr, lsr, u_ptr, k0, gs, FP4, BK, F32)
    gg = _r(ag, F32)
    vv = _r(au, F32)
    a = 4 * _tanh(gg / 4) * tl.sigmoid(gg) * 25 * _tanh(vv / 25)
    tl.store(act + t * NI + n, _bf16(a, F32))


@triton.jit
def _w2_kernel(act, idx_ptr, wt_ptr, w_ptr, sc_ptr, gs_ptr, lwe, lwr, lse, lsr, part,
               FP4: tl.constexpr, F32: tl.constexpr, H: tl.constexpr, NI: tl.constexpr, BN: tl.constexpr,
               BK: tl.constexpr, UNR: tl.constexpr = 0, PDL: tl.constexpr = False, PF: tl.constexpr = False, pfa=None, pfn=0):
    """Program (row block, slot t): part[t, rows] = bf16(act_t @ w2[e_t]^T) * weight_t (fp32). PF: prefetch this
    program's share of the pfn bytes at pfa (the next visit's projection)."""
    _enter(PDL)
    if PF:
        _pf_part(pfa, pfn, tl.program_id(1) * tl.num_programs(0) + tl.program_id(0),
                 tl.num_programs(0) * tl.num_programs(1))
    r = tl.program_id(0) * BN + tl.arange(0, BN)
    t = tl.program_id(1)
    e = tl.load(idx_ptr + t).to(tl.int64)
    gs = 1.
    if FP4:
        gs = tl.load(gs_ptr + e)
    acc = tl.zeros([BN], tl.float32)
    for k0 in tl.range(0, NI, BK, loop_unroll_factor=UNR * (NI // BK)):
        acc = _deq_dot(acc, w_ptr + e * lwe, sc_ptr + e * lse, r, lwr, lsr, act + t * NI, k0, gs, FP4, BK, F32)
    tl.store(part + t * H + r, _r(acc, F32) * tl.load(wt_ptr + t))


@triton.jit
def _w2_fin_kernel(part, st, cw, r_ptr, h_out, F32: tl.constexpr, H: tl.constexpr, BN: tl.constexpr, UNR: tl.constexpr = 0, PDL: tl.constexpr = False, PF: tl.constexpr = False):
    """y = bf16(sum of the 8 slots' parts, in slot order); mlp short conv (state update); h = bf16(r + conv)."""
    _enter(PDL)
    r = tl.program_id(0) * BN + tl.arange(0, BN)
    tot = tl.zeros([BN], tl.float32)
    for t in tl.range(8, loop_unroll_factor=UNR * 8):
        tot += tl.load(part + t * H + r)
    y = _r(tot, F32)
    y = _r(_conv4(st, r, H, y, tl.load(cw + r * 4), tl.load(cw + r * 4 + 1), tl.load(cw + r * 4 + 2),
                  tl.load(cw + r * 4 + 3), 2.23606797749979), F32)
    tl.store(h_out + r, _bf16(tl.load(r_ptr + r) + y, F32))


@triton.jit
def _w2all_kernel(act, idx_ptr, wt_ptr, w_ptr, sc_ptr, gs_ptr, lwe, lwr, lse, lsr, st, cw, r_ptr, h_out,
                  FP4: tl.constexpr, F32: tl.constexpr, H: tl.constexpr, NI: tl.constexpr, BN: tl.constexpr,
                  BK: tl.constexpr, UNR: tl.constexpr = 0, PDL: tl.constexpr = False, PF: tl.constexpr = False,
                  pfa=None, pfn=0):
    """INK_DEC_W2ALL: _w2_kernel for all 8 slots + _w2_fin_kernel in one program per BN rows: the same per-slot
    products (rounded mul, never fused into the sum, as when they went through memory) summed in slot order."""
    _enter(PDL)
    pid = tl.program_id(0)
    if PF:
        _pf_part(pfa, pfn, pid, tl.num_programs(0))
    r = pid * BN + tl.arange(0, BN)
    tot = tl.zeros([BN], tl.float32)
    for t in tl.range(8, loop_unroll_factor=UNR * 8):
        e = tl.load(idx_ptr + t).to(tl.int64)
        gs = 1.
        if FP4:
            gs = tl.load(gs_ptr + e)
        acc = tl.zeros([BN], tl.float32)
        for k0 in tl.range(0, NI, BK, loop_unroll_factor=UNR * (NI // BK)):
            acc = _deq_dot(acc, w_ptr + e * lwe, sc_ptr + e * lse, r, lwr, lsr, act + t * NI, k0, gs, FP4, BK, F32)
        p = _r(acc, F32)
        wt = tl.load(wt_ptr + t)
        if F32:
            p = p * wt
        else:
            p = libdevice.mul_rn(p, wt + tl.zeros([BN], tl.float32))
        tot = tot + p
    y = _r(tot, F32)
    y = _r(_conv4(st, r, H, y, tl.load(cw + r * 4), tl.load(cw + r * 4 + 1), tl.load(cw + r * 4 + 2),
                  tl.load(cw + r * 4 + 3), 2.23606797749979), F32)
    tl.store(h_out + r, _bf16(tl.load(r_ptr + r) + y, F32))


# --------------------------------------------------------------------------------------------- host side
def mmap_experts(W, name):
    """CPU tests: a zero-copy tensor over one expert tensor of the safetensors file (Weights keeps lazy slices on CPU;
    the kernels only touch the routed experts' pages)."""
    if not hasattr(W, '_mm'):
        fn = W.f_path if hasattr(W, 'f_path') else None
        import glob
        from engine import MODEL_DIR
        fn = fn or glob.glob(os.path.join(MODEL_DIR, 'model-*.safetensors'))[0]
        fh = open(fn, 'rb')
        n = struct.unpack('<Q', fh.read(8))[0]
        W._mm = (mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_COPY), json.loads(fh.read(n)), 8 + n)
    mm, hdr, base = W._mm
    meta = hdr['model.llm.' + name]
    dt = {'U8': torch.uint8, 'F8_E4M3': torch.float8_e4m3fn}[meta['dtype']]
    a, b = meta['data_offsets']
    return torch.frombuffer(mm, dtype=torch.uint8, count=b - a, offset=base + a).view(dt).view(meta['shape'])


def kv_ptrs(kv, slot):
    """(K codes, V codes, K scales, V scales, page table) of one full-visit slot: [rows, 8, 64|8] views; a flat KV4
    (tests) is treated as pages of kvpool.PAGE rows with an identity table."""
    if hasattr(kv, 't'):
        c, s, tbl = kv.pool.c[:, slot], kv.pool.s[:, slot], kv.t
    else:
        c, s = kv.c[:, slot], kv.s[:, slot]
        if getattr(kv, 'ident', None) is None:
            kv.ident = torch.arange(-(-kv.cap // kvpool.PAGE), device=c.device)
        tbl = kv.ident
    return c[0], c[1], s[0].view(torch.uint8), s[1].view(torch.uint8), tbl


def visit_plan():
    """Per visit: (L, pass, input visit (-1 = embedding), attn entries, new block slot or -1, mlp entries,
    prev = (B slot, visit whose output is the pass end) for loop messages or None)."""
    plan, v = [], 0
    for L in range(6):
        ent = [] if L == 0 else [0]
        plan.append((L, 0, v - 1, ent, 0 if L == 0 else -1, [0], None))
        v += 1
    for p in range(3):
        for i, L in enumerate(CORE):
            base = [0] + ([1 + 2 * p] if L > 6 else []) + ([2 + 2 * p] if L > 12 else [])
            new = 1 + 2 * p if L == 6 else 2 + 2 * p if L == 12 else -1
            prev = None if p == 0 else (2 + 2 * (p - 1), 6 + 12 * (p - 1) + 11)
            plan.append((L, p, 5 if i == 0 else v - 1, base, new, base + ([new] if new >= 0 else []), prev))
            v += 1
    for L in range(18, 24):
        base = [0, 5, 6] + ([7] if L > 18 else [])
        plan.append((L, 0, v - 1, base, 7 if L == 18 else -1, [0, 5, 6, 7], None))
        v += 1
    return plan


class Fused:
    def __init__(self, m, st, x_raw, out):
        self.m, self.st, self.x_raw, self.out = m, st, x_raw, out
        W, dev = m.W, m.dev
        f32 = lambda *s: torch.zeros(*s, dtype=torch.float32, device=dev)
        bf = lambda *s: torch.zeros(*s, dtype=torch.bfloat16, device=dev)
        i32 = lambda x: torch.tensor(list(x) + [0] * (8 - len(x)), dtype=torch.int32, device=dev)
        ptrs = lambda ts: torch.tensor([t.data_ptr() for t in ts], dtype=torch.int64, device=dev)
        self.h0, self.B, self.hbuf = bf(H), f32(8, H), bf(len(VISITS), H)
        self.A, self.X, self.M, self.G, self.Z = f32(3, H), bf(3, H), f32(4, H), f32(3, H), f32(3, 64)
        self.Y, self.Qb, self.Rb, self.att, self.R = bf(8664), bf(NQ, HD), bf(NQ, 16), bf(NQ * HD), f32(H)
        self.part = f32(8, H)
        self.U, self.S, self.act, self.idx, self.wt = bf(H), f32(128), bf(8, NI), torch.zeros(8, dtype=torch.int32,
                                                                                            device=dev), f32(8)
        self.split = int(os.environ.get('INK_DEC_FAR_BLK', 4096))
        self.far3 = decode.FAR3                                  # INK_DEC_FAR3: FAR2 splits, faster far kernel
        self.nsp = decode.FAR3_NSP if self.far3 else decode.FAR_NSP if decode.FAR2 else 0  # FAR2/3: even far splits
        self.ns = self.nsp or -(-st.kv.cap // self.split)
        self.nsf = triton.next_power_of_2(self.nsp) if self.nsp else 0
        self.fo, self.fl = f32(self.ns, NQ, HD), f32(self.ns, NQ)
        nt = 1024 // BLK['near']
        self.pm, self.pl, self.pacc, self.plse = f32(nt, NQ), f32(nt, NQ), f32(nt, NQ, HD), f32(NQ)
        self.plan, self.vis = visit_plan(), []
        self.o, self.lk = kopts()                       # INK_DEC_* flags at construction (constexprs, launch kwargs)
        self.o_nopf = dict(self.o, PF=False)
        self.w2all = OPT['W2ALL']
        # INK_FP8_DECODE (m.d8.dec at construction): proj / wo / unembed read the e4m3 copies (fp8_dense.Dense8)
        d8 = m.d8 if m.d8 is not None and m.d8.dec else None
        self.w8 = d8 is not None
        offloaded = getattr(W, 'offloaded', ())
        pinned = getattr(W, 'pinned_layers', ())
        keep = []  # tensors the pointer tables point at
        for v, (L, p, src, ent, new, ment, prev) in enumerate(self.plan):
            P, A = f'layers.{L}.', f'layers.{L}.attn.'
            full = L in FULL
            nws = [W[P + f'{n}_input_norm.weight'] for n in 'qkv'] if full else [W[P + 'attn_norm.weight']] * 3
            d = dict(L=L, full=full, src=src, ent=i32(ent), n=len(ent), new=new, ment=i32(ment), mn=len(ment),
                     nwt=ptrs(nws), nws=nws, msg=prev is not None, bnd=L % 6 == 0)
            d['q'] = W[f'attn_res.attn_queries.{L - 1}'].float().reshape(-1, H).contiguous() if L else self.A
            d['mq'] = W[f'attn_res.mlp_queries.{L}'].float().reshape(1, H).contiguous()
            keep += [d['q'], d['mq']]
            if prev is not None:
                d['pslot'], d['pv'] = i32([prev[0]]), prev[1]
                cq = f'layers.{L}.controllers.'
                cs = ('attn', 'k', 'v')
                d['dtab'] = ptrs([W[cq + c + '.down_weight'] for c in cs])
                d['gtab'] = ptrs([W.gcat[cq + c + '.'] for c in cs])
                d['dtab_m'] = ptrs([W[cq + 'mlp.down_weight']])
                d['gtab_m'] = ptrs([W.gcat[cq + 'mlp.']])
                d['mqs'] = m.mq[CORE.index(L)].contiguous()
                keep.append(d['mqs'])
            cat, ns = W.proj[L]
            (cat, d['cs']), _ = d8.proj[L] if d8 else ((cat, self.S), None)   # (bf16: cs is an unused stand-in)
            d['cat'], d['n0'], d['n1'], d['N'] = cat, ns[0], ns[1], sum(ns)  # N: rows (the fp8 matrix has pad rows)
            d['koff'] = ns[0]
            d['tkoff'] = ns[0] + NKV * HD
            d['voff'] = ns[0] + ns[1]
            d['tvoff'] = ns[0] + ns[1] + NKV * HD
            if full:
                d['kw'], d['vw'] = W[A + 'k_sconv.static_w'].float().contiguous(), W[A + 'v_sconv.static_w'].float().contiguous()
            else:
                d['kw'], d['vw'] = (W[A + f'{c}_sconv.weight'].float().reshape(1024, 4).contiguous() for c in 'kv')
            d['qsc'], d['rsc'] = W[A + 'q_l2_clip.scale'].float().contiguous(), W[A + 'rel_l2_clip.scale'].float().contiguous()
            d['proj'] = W[A + 'rel_logits_proj.proj'].contiguous()
            d['wo'], d['wos'] = d8.wo[L] if d8 else (W[A + 'wo_ud.weight'], self.S)
            d['acw'] = W[P + 'attn_sconv.weight'].float().reshape(H, 4).contiguous()
            d['mcw'] = W[P + 'mlp_sconv.weight'].float().reshape(H, 4).contiguous()
            d['mnw'] = W[P + 'mlp_norm.weight'].float().contiguous()
            d['mnwt'] = ptrs([d['mnw']])
            d['gw'], d['gb'] = W[P + 'mlp.gate.weight'].float().contiguous(), W[P + 'mlp.gate.bias'].float().contiguous()
            E = P + 'mlp.experts.'
            mode = 'resident' if L not in offloaded else ('dma' if L in pinned else 'stage')
            d['moe_mode'] = mode
            if mode == 'resident':
                ex = [W[E + n] for n in ('w13_weight', 'w13_weight_scale', 'w2_weight', 'w2_weight_scale')]
                if not isinstance(ex[0], torch.Tensor):  # CPU: lazy safetensors slices -> mmap views
                    ex = [mmap_experts(W, E + n) for n in ('w13_weight', 'w13_weight_scale', 'w2_weight', 'w2_weight_scale')]
                d['ex'] = ex
            else:
                d['ex'] = None
            fp4 = L < 20
            d['fp4'] = fp4
            if fp4:
                d['gs13'], d['gs2'] = ((W[E + w + '_weight_amax'] / W[E + w + '_weight_outer_scale_denominator'][0])
                                       .float().contiguous() for w in ('w13', 'w2'))
            else:
                d['gs13'] = d['gs2'] = self.wt  # unused stand-in pointer
            if full:
                d['kv'] = kv_ptrs(st.kv, m.slot[v])
            self.vis.append(d)
        if any(d['moe_mode'] in ('dma', 'stage') for d in self.vis):
            self.idx07 = torch.arange(8, dtype=torch.int32, device=dev)
            self.gs13_8 = f32(8)
            self.gs2_8 = f32(8)
        nb = lambda t: t.numel() * t.element_size()
        for v, d in enumerate(self.vis):                # INK_DEC_L2PF targets (kernel -> the weight read after it)
            nx = self.vis[v + 1]['cat'] if v + 1 < len(self.vis) else None
            pf_w13 = {}
            if d['moe_mode'] == 'resident':
                w2, s2 = d['ex'][2], d['ex'][3]
                pf_w13 = dict(pfa=w2, pfn=w2.stride(0) * w2.element_size(), pfb=s2,
                              pfm=s2.stride(0) * s2.element_size())
            d['pf'] = dict(near=dict(pfa=d['wo'], pfn=nb(d['wo'])), wo=dict(pfa=d['gw'], pfn=nb(d['gw'])),
                           w13=pf_w13,
                           w2=dict(pfa=nx, pfn=nb(nx)) if nx is not None else dict(pfa=self.S, pfn=0))
            rk, rv = st.ring[v] if getattr(st, 'ring', None) else (self.S, self.S)
            d['pf']['proj'] = dict(pfa=rk, pfn=nb(rk), pfb=rv, pfm=nb(rv)) if rk is not self.S else \
                dict(pfa=self.S, pfn=0, pfb=self.S, pfm=0)
            if not self.o['PF']:
                d['pf'] = dict.fromkeys(d['pf'], {})
        self.keep = keep
        self.final_q = W['attn_res.output_query'].float().reshape(1, H).contiguous()
        self.final_nwt = ptrs([W['norm.weight']])          # fp32 already (F32_ONLY)
        self.final_ent = i32([0, 5, 6, 7])

    def refresh_kv(self):
        """Re-read the page-store pointers (a test swapped st.kv for another flat store)."""
        for v, d in enumerate(self.vis):
            if d['full']:
                d['kv'] = kv_ptrs(self.st.kv, self.m.slot[v])
        self.ns = self.nsp or -(-self.st.kv.cap // self.split)
        if self.fo.shape[0] != self.ns:
            self.fo = torch.zeros(self.ns, NQ, HD, device=self.fo.device)
            self.fl = torch.zeros(self.ns, NQ, device=self.fo.device)

    def near(self, proj, rk, rv, full, pf=None):
        """Near attention split over (kv head, ring tile): 3 launches, 8 x 1024/BLK programs for the two passes."""
        st, F, b, o, lk = self.st, INTERP, BLK['near'], self.o, self.lk
        ns = 1024 // b
        if self.pacc.shape[0] != ns:  # tests change BLK
            self.pm, self.pl, self.pacc = (torch.zeros(ns, *t.shape[1:], device=t.device) for t in (self.pm, self.pl, self.pacc))
        _near_part_kernel[(NKV, ns)](self.Qb, self.Rb, proj, rk, st.rpos, st.pos, self.pm, self.pl, F32=F, BLK=b,
                                     **o, **lk, **(pf or {}))
        _near_out_kernel[(NKV, ns)](self.Qb, self.Rb, proj, self.Y, rk, rv, st.rpos, st.pos, self.pm, self.pl, self.fl,
                                    self.ns, self.pacc, self.plse, FAR=full, F32=F, BLK=b, NS=ns, NSF=self.nsf,
                                    **o, **lk)
        _near_fin_kernel[(NQ,)](self.pacc, self.plse, self.fo, self.fl, self.ns, self.Y, self.att, FAR=full, F32=F,
                                NS=ns, NSF=self.nsf, **o, **lk)

    def run(self):
        m, st = self.m, self.st
        self.h0.copy_(rms(self.x_raw, m.W['embed_norm.weight'])[0])
        for v, d in enumerate(self.vis):
            self.visit(v, d)
        f = self.X[:1]
        _mix_kernel[(1,)](self.final_q, self.B, self.final_ent, 4, self.hbuf[len(VISITS) - 1], -1, self.A, self.X,
                          self.final_nwt, self.final_q, self.final_ent, 1, self.h0, self.M, 1,
                          FIRST=False, NORM=True, STORE_A=False, F32=INTERP, H=H, CH=CH, **self.o, **self.lk)
        self.out.copy_(m.logits(f, self.w8))

    def visit(self, v, d):
        st, F, o, lk, pf = self.st, INTERP, self.o, self.lk, d['pf']
        conv = st.conv[v]
        hin = self.h0 if d['src'] < 0 else self.hbuf[d['src']]
        full, msg = d['full'], d['msg']
        nq = 3 if full else 1
        xs = H if full else 0
        if not msg:
            _mix_kernel[(nq,)](d['q'], self.B, d['ent'], d['n'], hin, d['new'], self.A, self.X, d['nwt'],
                               d['q'], d['ent'], 1, hin, self.M, nq,
                               FIRST=d['L'] == 0, NORM=True, STORE_A=False, F32=F, H=H, CH=CH, **o, **lk)
            _proj_kernel[(d['N'] // BLK['proj'],)](d['cat'], d['n0'], d['n1'], self.X, xs * 1, d['nwt'], self.Y, d['cs'],
                                                   NORM=False, F32=F, H=H, CH=CH, BN=BLK['proj'], W8=self.w8, **o, **lk,
                                                   **pf['proj'])
        else:
            _mix_kernel[(nq + 4,)](d['q'], self.B, d['ent'], d['n'], hin, d['new'], self.A, self.X, d['nwt'],
                                   d['mqs'], d['pslot'], 1, self.hbuf[d['pv']], self.M, nq,
                                   FIRST=False, NORM=False, STORE_A=True, F32=F, H=H, CH=CH, **o, **lk)
            _gate_z_kernel[(3, 64 // BLK['gate'])](self.A, xs, self.M, d['dtab'], self.Z, F32=F, H=H, CH=CH,
                                                   ZB=BLK['gate'], num_warps=NW['gate'], **o, **lk)
            _gate_out_kernel[(3, H // BLK['gout'])](self.A, xs, self.M, d['gtab'], self.Z, self.G, H=H, BH=BLK['gout'],
                                                    num_warps=NW['gout'], **o, **lk)
            _proj_kernel[(d['N'] // BLK['proj'],)](d['cat'], d['n0'], d['n1'], self.G, H, d['nwt'], self.Y, d['cs'],
                                                   NORM=True, F32=F, H=H, CH=CH, BN=BLK['proj'], W8=self.w8, **o, **lk,
                                                   **pf['proj'])
        rk, rv = st.ring[v]
        kv = d['kv'] if full else (self.Y,) * 5
        _pre_kernel[(NKV,)](self.Y, d['koff'], d['voff'], d['tkoff'], d['tvoff'], d['qsc'], d['rsc'], d['kw'], d['vw'],
                            conv['k'], conv['v'], rk, rv, st.rpos, st.pos, *kv, self.Qb, self.Rb,
                            FULL=full, F32=F, PG=kvpool.PAGE, **o, **lk)
        if full:
            c0, c1, s0, s1, tbl = kv
            decode.launch_far(self.Qb, c0, c1, s0.view(torch.float8_e4m3fn), s1.view(torch.float8_e4m3fn), tbl,
                              st.pos, self.fo, self.fl, self.split, nsp=self.nsp, far3=self.far3)
        self.near(d['proj'], rk, rv, full, pf['near'])
        _wo_kernel[(H // BLK['wo'],)](self.att, d['wo'], conv['attn'], d['acw'], hin, self.R, d['wos'], BND=d['bnd'],
                                      F32=F, H=H, CH=CH, BN=BLK['wo'], num_warps=NW['wo'], W8=self.w8, **o, **lk,
                                      **pf['wo'])
        if not msg:
            _mix_kernel[(1,)](d['mq'], self.B, d['ment'], d['mn'], self.R, -1, self.A, self.U, d['mnwt'],
                              d['mq'], d['ment'], 1, self.R, self.M, 1,
                              FIRST=False, NORM=True, STORE_A=False, F32=F, H=H, CH=CH, **o, **lk)
            _router_kernel[(128 // BLK['router'],)](self.A, d['mnw'], self.U, d['gw'], self.S, NORM=False, F32=F,
                                                    H=H, CH=CH, RB=BLK['router'], num_warps=NW['router'], **o, **lk)
        else:
            _mix_kernel[(1,)](d['mq'], self.B, d['ment'], d['mn'], self.R, -1, self.A, self.U, d['mnwt'],
                              d['mq'], d['ment'], 1, self.R, self.M, 1,
                              FIRST=False, NORM=False, STORE_A=True, F32=F, H=H, CH=CH, **o, **lk)
            _gate_z_kernel[(1, 64 // BLK['gate'])](self.A, 0, self.M[3], d['dtab_m'], self.Z, F32=F, H=H, CH=CH,
                                                   ZB=BLK['gate'], num_warps=NW['gate'], **o, **lk)
            _gate_out_kernel[(1, H // BLK['gout'])](self.A, 0, self.M[3], d['gtab_m'], self.Z, self.G, H=H,
                                                    BH=BLK['gout'], num_warps=NW['gout'], **o, **lk)
            _router_kernel[(128 // BLK['router'],)](self.G, d['mnw'], self.U, d['gw'], self.S, NORM=True, F32=F,
                                                    H=H, CH=CH, RB=BLK['router'], num_warps=NW['router'], **o, **lk)
        mode = d['moe_mode']
        if mode == 'resident':
            w13, s13, w2, s2 = d['ex']
            _w13_kernel[(NI // BLK['w13'], 8)](self.S, d['gb'], self.U, w13, s13, d['gs13'], w13.stride(0), w13.stride(1),
                                               s13.stride(0), s13.stride(1), self.act, self.idx, self.wt,
                                               FP4=d['fp4'], F32=F, H=H, NI=NI, BN=BLK['w13'], BK=BLK['kbk'],
                                               num_warps=NW['w13'], **o, **lk, **pf['w13'])
            idx2, gs2 = self.idx, d['gs2']
        else:
            L = d['L']
            _route_top8_kernel[(1,)](self.S, d['gb'], self.idx, self.wt, PDL=o['PDL'], **lk)
            if d['fp4']:
                idx_l = self.idx.to(torch.long)
                torch.index_select(d['gs13'], 0, idx_l, out=self.gs13_8)
                torch.index_select(d['gs2'], 0, idx_l, out=self.gs2_8)
                gs13_s, gs2 = self.gs13_8, self.gs2_8
            else:
                gs13_s = gs2 = self.wt
            ids = self.idx.tolist()
            w13, s13, w2, s2 = (self.m.W.stage_experts_dma(L, ids, cap=8) if mode == 'dma'
                                else self.m.W.stage_experts(L, ids, cap=8))
            _w13_kernel[(NI // BLK['w13'], 8)](self.S, d['gb'], self.U, w13, s13, gs13_s, w13.stride(0), w13.stride(1),
                                               s13.stride(0), s13.stride(1), self.act, self.idx, self.wt,
                                               FP4=d['fp4'], F32=F, H=H, NI=NI, BN=BLK['w13'], BK=BLK['kbk'],
                                               num_warps=NW['w13'], **self.o_nopf, **lk, PRE_ROUTED=True)
            idx2 = self.idx07
        if self.w2all:
            _w2all_kernel[(H // BLK['w2a'],)](self.act, idx2, self.wt, w2, s2, gs2, w2.stride(0), w2.stride(1),
                                              s2.stride(0), s2.stride(1), conv['mlp'], d['mcw'], self.R, self.hbuf[v],
                                              FP4=d['fp4'], F32=F, H=H, NI=NI, BN=BLK['w2a'], BK=BLK['kbk2'],
                                              num_warps=NW['w2a'], **o, **lk, **pf['w2'])
            return
        _w2_kernel[(H // BLK['w2'], 8)](self.act, idx2, self.wt, w2, s2, gs2, w2.stride(0), w2.stride(1),
                                         s2.stride(0), s2.stride(1), self.part, FP4=d['fp4'], F32=F, H=H, NI=NI,
                                         BN=BLK['w2'], BK=BLK['kbk2'], num_warps=NW['w2'], **o, **lk, **pf['w2'])
        _w2_fin_kernel[(H // BLK['w2f'],)](self.part, conv['mlp'], d['mcw'], self.R, self.hbuf[v], F32=F, H=H,
                                           BN=BLK['w2f'], **o, **lk)
