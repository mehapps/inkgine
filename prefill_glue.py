"""Fused prefill glue (`INK_GLUE=triton`, cuda, calls with >= 64 rows): drop-in Triton versions of model.py's fp32
elementwise helpers, which as separate torch ops move ~450 GB per 4096-token chunk (tests/prof_prefill.py --ops).
- sconv(x, st, w=None, dyn=None): §3.3 short conv, one read of x (+3-row halo from st), one bf16 write; st updated.
  Same fp32 products/sum order (taps 0..3 left to right) and the scalar division done like torch on the device
  (cuda: x * fp32(1/d); cpu: x / d).
- rms(x, w=None): one pass per row (sum of squares, rsqrt, scale), fp32 out.
- gate_norm: gate_out + the following rms(., w).bfloat16() in one row pass (what Model.layer uses).
- gate_in / gate_out: the loop gate's rms+cat+bf16 prologue and tanh + (1-f) x + s msg epilogue (one pass each).
- attn_res(q, s): per token: logits q . s_n over H (fp32), / sqrt(H), softmax over n, sum_n w_n s_n; reads the n
  entries twice (no stack copy, no [n,T,H] temp).
Reductions over H (rms mean, attn_res logits) sum in a different order than torch: fp32-ulp differences, gated end
to end by KL vs noise (tests/gpu_fp8.py). tests/test_glue.py checks each against model.py's eager functions.
"""
import math

import numpy as np
import torch
import triton
import triton.language as tl

from moe_triton import INTERP, _bf16, _mma


H = 2304
MAXN = tl.constexpr(8)  # attn_res entries (blocks + 1 <= 7)


_NOFMA = {} if INTERP else dict(enable_fp_fusion=False)  # torch does these multiply-adds as separate ops


def _recip(d):
    return float(np.float32(1) / np.float32(d))


@triton.jit
def _conv_y(x_ptr, st_ptr, w_ptr, tap_ptr, t, c, tm, cm, C, G, scl, DYN: tl.constexpr, RECIP: tl.constexpr,
            BT: tl.constexpr, BC: tl.constexpr):
    """§3.3 conv of rows t, channels c: fp32 value before the bf16 rounding (sconv's math)."""
    m2 = tm[:, None] & cm[None, :]
    acc = tl.zeros([BT, BC], tl.float32)
    cur = tl.zeros([BT, BC], tl.float32)
    for j in tl.static_range(4):                       # tap j reads input row t - (3 - j); tap 3 = current token
        src = t - (3 - j)
        fx = (src >= 0)[:, None]
        xv = tl.load(x_ptr + src[:, None] * C + c[None, :], mask=m2 & fx, other=0.)
        sv = tl.load(st_ptr + (src + 3)[:, None] * C + c[None, :], mask=m2 & ~fx, other=0.)
        v = tl.where(fx, xv, sv).to(tl.float32)
        if DYN:                                        # W[t,c,j] = static_w[3-j, c] + taps[t, c // G, 3-j]
            wv = tl.load(w_ptr + (3 - j) * C + c, mask=cm, other=0.).to(tl.float32)[None, :] + \
                tl.load(tap_ptr + t[:, None] * 32 + (c // G)[None, :] * 4 + (3 - j), mask=m2, other=0.).to(tl.float32)
        else:
            wv = tl.load(w_ptr + c * 4 + j, mask=cm, other=0.).to(tl.float32)[None, :]
        if j == 0:
            acc = v * wv
        else:
            acc = acc + v * wv
        if j == 3:
            cur = v
    y = acc + cur
    return y * scl if RECIP else y / scl


@triton.jit
def _sconv_kernel(x_ptr, st_ptr, w_ptr, tap_ptr, out_ptr, T, C, G, scl,
                  DYN: tl.constexpr, RECIP: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr, F32: tl.constexpr):
    t = tl.program_id(0) * BT + tl.arange(0, BT)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    tm, cm = t < T, c < C
    y = _conv_y(x_ptr, st_ptr, w_ptr, tap_ptr, t, c, tm, cm, C, G, scl, DYN, RECIP, BT, BC)
    tl.store(out_ptr + t[:, None] * C + c[None, :], _bf16(y, F32), mask=tm[:, None] & cm[None, :])


@triton.jit
def _rms_kernel(x_ptr, w_ptr, out_ptr, D, eps, HASW: tl.constexpr, BD: tl.constexpr):
    r = tl.program_id(0)
    d = tl.arange(0, BD)
    dm = d < D
    x = tl.load(x_ptr + r * D + d, mask=dm, other=0.).to(tl.float32)
    inv = tl.math.rsqrt(tl.sum(x * x, 0) / D + eps)
    y = x * inv
    if HASW:
        y = y * tl.load(w_ptr + d, mask=dm, other=0.).to(tl.float32)
    tl.store(out_ptr + r * D + d, y, mask=dm)


@triton.jit
def _res_acc(w, t, h0, hr, nr, s0, s1, s2, s3, s4, s5, s6, s7, N: tl.constexpr, NQP: tl.constexpr, H: tl.constexpr,
             BH: tl.constexpr):
    acc = tl.zeros([NQP, BH], tl.float32)
    for n in tl.static_range(N):
        sp = (s0, s1, s2, s3, s4, s5, s6, s7)[n]
        v = tl.load(sp + t * H + h0 + hr)
        wn = tl.sum(tl.where(nr[None, :] == n, w, 0.), 1)
        acc += wn[:, None] * v[None, :]
    return acc


@triton.jit
def _res_kernel(q_ptr, s0, s1, s2, s3, s4, s5, s6, s7, out_ptr, T, NQ, scl, nw_ptr, eps,
                N: tl.constexpr, NQP: tl.constexpr, RECIP: tl.constexpr, H: tl.constexpr, BH: tl.constexpr,
                NORM: tl.constexpr = False, F32: tl.constexpr = False):
    """One token per program: out[j, t] = sum_n softmax_n(q_j . s_n[t] / sqrt(H)) s_n[t] (fp32); NORM: out =
    bf16(rms(that) * nw[j]) (one more pass over the entries to get the row norm first)."""
    t = tl.program_id(0)
    qr = tl.arange(0, NQP)
    qm = qr < NQ
    nr = tl.arange(0, MAXN)
    hr = tl.arange(0, BH)
    lg = tl.zeros([NQP, MAXN], tl.float32)
    for h0 in range(0, H, BH):
        q = tl.load(q_ptr + qr[:, None] * H + h0 + hr[None, :], mask=qm[:, None], other=0.)
        for n in tl.static_range(N):
            sp = (s0, s1, s2, s3, s4, s5, s6, s7)[n]
            v = tl.load(sp + t * H + h0 + hr)
            lg += tl.where(nr[None, :] == n, tl.sum(q * v[None, :], 1)[:, None], 0.)
    lg = lg * scl if RECIP else lg / scl
    lg = tl.where(nr[None, :] < N, lg, float('-inf'))
    mx = tl.max(lg, 1)
    e = tl.exp(lg - mx[:, None])
    w = e / tl.sum(e, 1)[:, None]
    if NORM:
        ss = tl.zeros([NQP], tl.float32)
        for h0 in range(0, H, BH):
            acc = _res_acc(w, t, h0, hr, nr, s0, s1, s2, s3, s4, s5, s6, s7, N, NQP, H, BH)
            ss += tl.sum(acc * acc, 1)
        inv = tl.math.rsqrt(ss / H + eps)
    for h0 in range(0, H, BH):
        acc = _res_acc(w, t, h0, hr, nr, s0, s1, s2, s3, s4, s5, s6, s7, N, NQP, H, BH)
        o = out_ptr + (qr[:, None] * T + t) * H + h0 + hr[None, :]
        if NORM:
            y = acc * inv[:, None] * tl.load(nw_ptr + qr[:, None] * H + h0 + hr[None, :], mask=qm[:, None], other=0.)
            tl.store(o, _bf16(y, F32), mask=qm[:, None])
        else:
            tl.store(o, acc, mask=qm[:, None])


@triton.jit
def _gate_in_kernel(x_ptr, m_ptr, out_ptr, eps, H: tl.constexpr, BD: tl.constexpr, F32: tl.constexpr):
    r = tl.program_id(0)
    d = tl.arange(0, BD)
    dm = d < H
    for i in tl.static_range(2):
        x = tl.load((x_ptr, m_ptr)[i] + r * H + d, mask=dm, other=0.)
        y = x * tl.math.rsqrt(tl.sum(x * x, 0) / H + eps)
        tl.store(out_ptr + r * 2 * H + i * H + d, _bf16(y, F32), mask=dm)


@triton.jit
def _tanh(x, F32: tl.constexpr):
    if F32:  # interpreter: no libdevice
        return 2 * tl.sigmoid(2 * x) - 1
    return tl.extra.cuda.libdevice.tanh(x)


@triton.jit
def _gate_out_kernel(z_ptr, x_ptr, m_ptr, out_ptr, n, H: tl.constexpr, BLK: tl.constexpr, F32: tl.constexpr):
    i = tl.program_id(0) * BLK + tl.arange(0, BLK)
    ok = i < n
    t, h = i // H, i % H
    f = _tanh(tl.load(z_ptr + t * 2 * H + h, mask=ok, other=0.), F32)
    s = _tanh(tl.load(z_ptr + t * 2 * H + H + h, mask=ok, other=0.), F32)
    a = (1 - f) * tl.load(x_ptr + i, mask=ok, other=0.)
    b = s * tl.load(m_ptr + i, mask=ok, other=0.)
    tl.store(out_ptr + i, a + b, mask=ok)


@triton.jit
def _gate_norm_kernel(z_ptr, x_ptr, m_ptr, w_ptr, out_ptr, eps, H: tl.constexpr, BD: tl.constexpr, F32: tl.constexpr):
    """One row: g = (1 - tanh f) x + tanh(s) msg; out = bf16(g * rsqrt(mean g^2 + eps) * w)."""
    t = tl.program_id(0)
    d = tl.arange(0, BD)
    dm = d < H
    f = _tanh(tl.load(z_ptr + t * 2 * H + d, mask=dm, other=0.), F32)
    s = _tanh(tl.load(z_ptr + t * 2 * H + H + d, mask=dm, other=0.), F32)
    g = (1 - f) * tl.load(x_ptr + t * H + d, mask=dm, other=0.) + s * tl.load(m_ptr + t * H + d, mask=dm, other=0.)
    y = g * tl.math.rsqrt(tl.sum(g * g, 0) / H + eps)
    tl.store(out_ptr + t * H + d, _bf16(y * tl.load(w_ptr + d, mask=dm, other=0.), F32), mask=dm)


def gate_norm(zz, x, msg, nw):
    T = x.shape[0]
    out = torch.empty(T, H, dtype=torch.bfloat16, device=x.device)
    _gate_norm_kernel[(T,)](zz.contiguous(), x.contiguous(), msg.contiguous(), nw, out, 1e-6, H=H, BD=4096,
                            F32=INTERP, num_warps=8, **_NOFMA)
    return out


def gate_in(x, msg):
    T = x.shape[0]
    out = torch.empty(T, 2 * H, dtype=torch.bfloat16, device=x.device)
    _gate_in_kernel[(T,)](x.contiguous(), msg.contiguous(), out, 1e-6, H=H, BD=4096, F32=INTERP, num_warps=8)
    return out


def gate_out(zz, x, msg):
    out = torch.empty_like(x)
    n = x.numel()
    _gate_out_kernel[(triton.cdiv(n, 1024),)](zz.contiguous(), x.contiguous(), msg.contiguous(), out, n, H=H,
                                              BLK=1024, F32=INTERP, num_warps=4, **_NOFMA)
    return out


def sconv(x, st, w=None, dyn=None):
    T, C = x.shape
    x = x.contiguous()
    out = torch.empty_like(x)
    cuda = x.is_cuda
    if dyn is None:
        wt, tap, G, d = w.reshape(C, 4).contiguous(), x, 1, math.sqrt(5)
    else:
        wt, tap = dyn[0].contiguous(), dyn[1].contiguous()
        G, d = C // tap.shape[1], 3.0
    BT, BC = 16, 128
    _sconv_kernel[(triton.cdiv(T, BT), triton.cdiv(C, BC))](
        x, st, wt, tap, out, T, C, G, _recip(d) if cuda else d,
        DYN=dyn is not None, RECIP=cuda, BT=BT, BC=BC, F32=INTERP, num_warps=4, **_NOFMA)
    st.copy_(torch.cat((st, x[-3:]))[-3:])             # after the kernel read st (same stream)
    return out


def rms(x, w=None, eps=1e-6):
    D = x.shape[-1]
    xc = x.contiguous()
    out = torch.empty(x.shape, dtype=torch.float32, device=x.device)
    _rms_kernel[(xc.numel() // D,)](xc, w if w is not None else xc, out, D, eps,
                                    HASW=w is not None, BD=triton.next_power_of_2(D), num_warps=8)
    return out


def attn_res(q, s):
    n, nq, T = len(s), q.shape[0], s[0].shape[0]
    assert n <= MAXN.value and s[0].shape[1] == H
    s = [x.contiguous() for x in s]
    out = torch.empty(nq, T, H, dtype=torch.float32, device=q.device)
    ptrs = s + [s[0]] * (MAXN.value - n)
    _res_kernel[(T,)](q.contiguous(), *ptrs, out, T, nq, _recip(math.sqrt(H)) if q.is_cuda else math.sqrt(H), out, 0.,
                      N=n, NQP=max(2, triton.next_power_of_2(nq)), RECIP=q.is_cuda, H=H, BH=256, num_warps=4)
    return out


# ---------------------------------------------------------------- INK_GLUE2 (model.GLUE2_FNS)
CLIP_QK, CLIP_REL, PG = 22.627416997969522, 8.0, 256


def attn_res_norm(q, s, w):
    """model.attn_res_norm: rms(attn_res(q, s), w).bfloat16(), w [nq,H] fp32 -> [nq,T,H] bf16."""
    n, nq, T = len(s), q.shape[0], s[0].shape[0]
    assert n <= MAXN.value and s[0].shape[1] == H
    s = [x.contiguous() for x in s]
    out = torch.empty(nq, T, H, dtype=torch.bfloat16, device=q.device)
    ptrs = s + [s[0]] * (MAXN.value - n)
    _res_kernel[(T,)](q.contiguous(), *ptrs, out, T, nq, _recip(math.sqrt(H)) if q.is_cuda else math.sqrt(H),
                      w.float().contiguous(), 1e-6, N=n, NQP=max(2, triton.next_power_of_2(nq)), RECIP=q.is_cuda,
                      H=H, BH=256, NORM=True, F32=INTERP, num_warps=4)
    return out


@triton.jit
def _clip_kernel(x_ptr, lx, s_ptr, tau_ptr, out_ptr, n, NH, clip,
                 TAU: tl.constexpr, D: tl.constexpr, BR: tl.constexpr, F32: tl.constexpr):
    """Rows r = (token, head) of D values: x * scale -> l2clip(clip) -> bf16 [-> * tau[token] -> bf16]."""
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    rm = r < n
    tok, hh = r // NH, r % NH
    d = tl.arange(0, D)
    x = tl.load(x_ptr + tok[:, None] * lx + hh[:, None] * D + d[None, :], mask=rm[:, None], other=0.).to(tl.float32)
    x = x * tl.load(s_ptr + hh[:, None] * D + d[None, :], mask=rm[:, None], other=0.)
    f = tl.minimum(tl.math.div_rn(tl.full([BR], clip, tl.float32), tl.math.sqrt_rn(tl.sum(x * x, 1)) + 1e-6), 1.)
    y = _bf16(x * f[:, None], F32)
    if TAU:
        y = _bf16(y.to(tl.float32) * tl.load(tau_ptr + tok, mask=rm, other=0.)[:, None], F32)
    tl.store(out_ptr + r[:, None] * D + d[None, :], y, mask=rm[:, None])


def qr_prep(q, r, qs, rs, tau=None):
    """model.qr_prep (q/r: column views of the projection output, unit column stride)."""
    T = q.shape[0]
    Q = torch.empty(T, 24, 128, dtype=torch.bfloat16, device=q.device)
    R = torch.empty(T, 24, 16, dtype=torch.bfloat16, device=q.device)
    for x, sc, out, D, c, br in ((q, qs, Q, 128, CLIP_QK, 32), (r, rs, R, 16, CLIP_REL, 128)):
        if x.stride(1) != 1:
            x = x.contiguous()
        _clip_kernel[(triton.cdiv(T * 24, br),)](x, x.stride(0), sc.float().contiguous(),
                                                 tau if tau is not None else sc, out, T * 24, 24, c,
                                                 TAU=tau is not None, D=D, BR=br, F32=INTERP, num_warps=4, **_NOFMA)
    return Q, R


@triton.jit
def _rne_e4m3(x):
    """0 <= x <= 448 fp32 -> nearest e4m3 value (ties to even), as fp32 (same as moe_fp4._rne_e4m3)."""
    b = x.to(tl.int32, bitcast=True)
    nrm = ((b + 0x7FFFF + ((b >> 20) & 1)) & -1048576).to(tl.float32, bitcast=True)
    sub = ((x * 512. + 12582912.) - 12582912.) / 512.
    return tl.where(x < 0.015625, sub, nrm)


@triton.jit
def _kv_kernel(x_ptr, st_ptr, w_ptr, tap_ptr, out_ptr, c_ptr, s_ptr, tbl_ptr, T, p0, scl, s7, clip,
               DYN: tl.constexpr, IS_V: tl.constexpr, KV: tl.constexpr, RECIP: tl.constexpr, BT: tl.constexpr,
               F32: tl.constexpr):
    """Program (token block, kv head): conv (bf16) -> K: l2clip, V: rms over the head's 128 -> bf16 out [T,8,128];
    KV: KV4.quant of that bf16 head (e4m3 scale amax/7 per 16, int4 codes) written through the page table."""
    t = tl.program_id(0) * BT + tl.arange(0, BT)
    h = tl.program_id(1)
    c = h * 128 + tl.arange(0, 128)
    tm = t < T
    y = _conv_y(x_ptr, st_ptr, w_ptr, tap_ptr, t, c, tm, c < 1024, 1024, 128, scl, DYN, RECIP, BT, 128)
    yb = _bf16(y, F32).to(tl.float32)
    if IS_V:
        o = yb * tl.math.rsqrt(tl.sum(yb * yb, 1) / 128 + 1e-6)[:, None]
    else:
        f = tl.minimum(tl.math.div_rn(tl.full([BT], clip, tl.float32), tl.math.sqrt_rn(tl.sum(yb * yb, 1)) + 1e-6), 1.)
        o = yb * f[:, None]
    ob = _bf16(o, F32)
    tl.store(out_ptr + t[:, None] * 1024 + c[None, :], ob, mask=tm[:, None])
    if KV:
        g = tl.reshape(ob.to(tl.float32), (BT, 8, 16))
        am = tl.max(tl.abs(g), 2)
        sc = _rne_e4m3(am * s7 if RECIP else am / s7)                       # torch: amax / 7 (cuda: * fp32(1/7))
        sf = tl.broadcast_to(sc[:, :, None], (BT, 8, 16))
        d = tl.where(sf > 0, tl.math.div_rn(g, tl.where(sf > 0, sf, 1.)), 0.)
        d = (d + 12582912.) - 12582912.                                      # round half to even (|d| < 2^22)
        q = (tl.minimum(tl.maximum(d, -8.), 7.) + 8.).to(tl.int32)
        lo, hi = tl.split(tl.reshape(q, (BT, 64, 2)))
        pos = p0 + t
        row = tl.load(tbl_ptr + pos // 256, mask=tm, other=0).to(tl.int64) * 256 + pos % 256
        b = tl.arange(0, 64)
        tl.store(c_ptr + row[:, None] * 512 + h * 64 + b[None, :], (lo | (hi << 4)).to(tl.uint8), mask=tm[:, None])
        e = tl.arange(0, 8)
        tl.store(s_ptr + row[:, None] * 64 + h * 8 + e[None, :], sc.to(tl.float8e4nv), mask=tm[:, None])


def kv_prep(K, V, stk, stv, kw, vw, tk=None, tv=None, kv=None, slot=0, p0=0):
    """model.kv_prep: two launches (K, V) instead of conv + clip/norm + quant + scatter as separate ops."""
    T = K.shape[0]
    cuda = K.is_cuda
    if kv is not None:
        from prefill_attn import store_of
        if hasattr(kv, 'pages') and p0 + T > len(kv.pages) * PG:          # as KVPool.put: grow if not reserved
            assert kv.pool.reserve(kv, p0 + T), 'KV pool full'
        cs, ss, tbl = store_of(kv, slot)
    outs = []
    for j, (x, st, w, tap) in enumerate(((K, stk, kw, tk), (V, stv, vw, tv))):
        x = x.contiguous()
        out = torch.empty(T, 8, 128, dtype=torch.bfloat16, device=x.device)
        if tap is None:
            wt, tp, dv = w.reshape(1024, 4).contiguous(), x, math.sqrt(5)
        else:
            wt, tp, dv = w.contiguous(), tap.contiguous(), 3.0
        kvp = (cs[j], ss[j], tbl) if kv is not None else (out, out, out)
        _kv_kernel[(triton.cdiv(T, 16), 8)](x, st, wt, tp, out, *kvp, T, p0, _recip(dv) if cuda else dv,
                                            _recip(7.) if cuda else 7., CLIP_QK, DYN=tap is not None, IS_V=j == 1,
                                            KV=kv is not None, RECIP=cuda, BT=16, F32=INTERP, num_warps=4, **_NOFMA)
        st.copy_(torch.cat((st, x[-3:]))[-3:])
        outs.append(out)
    return outs


@triton.jit
def _resid_kernel(x_ptr, st_ptr, w_ptr, h_ptr, r_ptr, out_ptr, n_ptr, T, scl, eps,
                  OUT: tl.constexpr, HAS_H: tl.constexpr, RECIP: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr,
                  F32: tl.constexpr):
    """Rows t: y = sconv(x) (bf16 value); OUT=False (res_in): out = [h +] y fp32; OUT=True (res_out): out = bf16(r + y).
    Then n = rms(out) fp32 (second pass re-reads the row just written)."""
    t = tl.program_id(0) * BT + tl.arange(0, BT)
    tm = t < T
    ss = tl.zeros([BT], tl.float32)
    for c0 in range(0, 2304, BC):
        c = c0 + tl.arange(0, BC)
        cm = c < 2304
        y = _bf16(_conv_y(x_ptr, st_ptr, w_ptr, w_ptr, t, c, tm, cm, 2304, 1, scl, False, RECIP, BT, BC),
                  F32).to(tl.float32)
        off = t[:, None] * 2304 + c[None, :]
        if OUT:
            v = _bf16(tl.load(r_ptr + off, mask=tm[:, None], other=0.) + y, F32)
            tl.store(out_ptr + off, v, mask=tm[:, None])
            v = v.to(tl.float32)
        else:
            v = y
            if HAS_H:
                v = tl.load(h_ptr + off, mask=tm[:, None], other=0.).to(tl.float32) + y
            tl.store(out_ptr + off, v, mask=tm[:, None])
        ss += tl.sum(v * v, 1)
    tl.debug_barrier()                                                       # the row values written above
    inv = tl.math.rsqrt(ss / 2304 + eps)
    for c0 in range(0, 2304, BC):
        off = t[:, None] * 2304 + c0 + tl.arange(0, BC)[None, :]
        v = tl.load(out_ptr + off, mask=tm[:, None], other=0.).to(tl.float32)
        tl.store(n_ptr + off, v * inv[:, None], mask=tm[:, None])


def _resid(x, st, w, h, r):
    T, C = x.shape
    x = x.contiguous()
    out = torch.empty(T, C, dtype=torch.bfloat16 if r is not None else torch.float32, device=x.device)
    nrm = torch.empty(T, C, dtype=torch.float32, device=x.device)
    _resid_kernel[(triton.cdiv(T, 8),)](x, st, w.reshape(C, 4).contiguous(), h if h is not None else x,
                                        r if r is not None else nrm, out, nrm, T,
                                        _recip(math.sqrt(5)) if x.is_cuda else math.sqrt(5), 1e-6, OUT=r is not None,
                                        HAS_H=h is not None, RECIP=x.is_cuda, BT=8, BC=256, F32=INTERP, num_warps=4,
                                        **_NOFMA)
    st.copy_(torch.cat((st, x[-3:]))[-3:])
    return out, nrm


def res_in(o, st, w, h=None):
    """model.res_in: -> (r fp32, rms(r) fp32)."""
    return _resid(o, st, w, h.contiguous() if h is not None else None, None)


def res_out(y, st, w, r):
    """model.res_out: -> (h bf16, rms(h) fp32)."""
    return _resid(y, st, w, None, r.contiguous())


# ---------------------------------------------------------------- INK_GATEFUSE (model.gate_down / gate_up_norm)
@triton.jit
def _gate_down_kernel(x_ptr, m_ptr, w_ptr, z_ptr, T, eps, H: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr,
                      F32: tl.constexpr):
    """Rows t: z = bf16([bf16(rms(x)) | bf16(rms(msg))] @ down^T) (down [64, 2H] bf16), fp32 accumulate. The [T,2H]
    bf16 gate input never goes to memory."""
    t = tl.program_id(0) * BT + tl.arange(0, BT)
    tm = t < T
    k, n = tl.arange(0, BK), tl.arange(0, 64)
    acc = tl.zeros([BT, 64], tl.float32)
    for i in tl.static_range(2):
        src = (x_ptr, m_ptr)[i]
        ss = tl.zeros([BT], tl.float32)
        for k0 in range(0, H, BK):
            v = tl.load(src + t[:, None] * H + k0 + k[None, :], mask=tm[:, None], other=0.)
            ss += tl.sum(v * v, 1)
        inv = tl.math.rsqrt(ss / H + eps)
        for k0 in range(0, H, BK):
            a = _bf16(tl.load(src + t[:, None] * H + k0 + k[None, :], mask=tm[:, None], other=0.) * inv[:, None], F32)
            w = tl.load(w_ptr + n[None, :] * (2 * H) + i * H + k0 + k[:, None])          # [BK, 64]
            acc = _mma(a, w, acc, F32)
    tl.store(z_ptr + t[:, None] * 64 + n[None, :], _bf16(acc, F32), mask=tm[:, None])


@triton.jit
def _gate_up_kernel(z_ptr, g_ptr, x_ptr, m_ptr, w_ptr, out_ptr, T, eps, H: tl.constexpr, BT: tl.constexpr,
                    BH: tl.constexpr, F32: tl.constexpr):
    """Rows t: zz = fp32(z) @ gcat^T (fp32 FMA dot, gcat [2H, 64] fp32, [forget | gain]) computed per BH columns,
    g = (1 - tanh f) x + tanh(s) msg; pass 0 sums g^2, pass 1 recomputes g and writes bf16(g * rsqrt(mean g^2 + eps) * w).
    zz [T,2H] fp32 never goes to memory; x / msg are read twice (second time from L2)."""
    t = tl.program_id(0) * BT + tl.arange(0, BT)
    tm = t < T
    n, hr = tl.arange(0, 64), tl.arange(0, BH)
    z = tl.load(z_ptr + t[:, None] * 64 + n[None, :], mask=tm[:, None], other=0.).to(tl.float32)
    ss = tl.zeros([BT], tl.float32)
    inv = tl.zeros([BT], tl.float32)
    for p in tl.static_range(2):
        if p == 1:
            inv = tl.math.rsqrt(ss / H + eps)
        for h0 in range(0, H, BH):
            h = h0 + hr
            gf = tl.load(g_ptr + h[None, :] * 64 + n[:, None])                         # [64, BH]
            gs = tl.load(g_ptr + (H + h)[None, :] * 64 + n[:, None])
            f = _tanh(tl.dot(z, gf, input_precision='ieee'), F32)
            s = _tanh(tl.dot(z, gs, input_precision='ieee'), F32)
            off = t[:, None] * H + h[None, :]
            g = (1 - f) * tl.load(x_ptr + off, mask=tm[:, None], other=0.) + \
                s * tl.load(m_ptr + off, mask=tm[:, None], other=0.)
            if p == 0:
                ss += tl.sum(g * g, 1)
            else:
                y = g * inv[:, None] * tl.load(w_ptr + h)[None, :]
                tl.store(out_ptr + off, _bf16(y, F32), mask=tm[:, None])


def gate_down(x, msg, w):
    """model.gate_down: x, msg fp32 [T,H], w = down_weight [64,2H] bf16 -> z [T,64] bf16."""
    T = x.shape[0]
    z = torch.empty(T, 64, dtype=torch.bfloat16, device=x.device)
    _gate_down_kernel[(triton.cdiv(T, 16),)](x.contiguous(), msg.contiguous(), w.contiguous(), z, T, 1e-6, H=H,
                                             BT=16, BK=128, F32=INTERP, num_warps=4)
    return z


def gate_up_norm(z, x, msg, g, nw):
    """model.gate_up_norm: z [T,64] bf16, gcat g [2H,64] fp32, nw [H] fp32 -> bf16 [T,H]."""
    T = x.shape[0]
    out = torch.empty(T, H, dtype=torch.bfloat16, device=x.device)
    _gate_up_kernel[(triton.cdiv(T, 16),)](z.contiguous(), g.contiguous(), x.contiguous(), msg.contiguous(),
                                           nw.float().contiguous(), out, T, 1e-6, H=H, BT=16, BH=64, F32=INTERP,
                                           num_warps=8, num_stages=1, **_NOFMA)   # sm_120: 36 KB smem, no spills
    return out


# ---------------------------------------------------------------- INK_MOETAIL (model.moe_tail)
@triton.jit
def _fin_rows(y_ptr, ws_ptr, idx_ptr, f_ptr, rows, rm, c, KR: tl.constexpr, NR: tl.constexpr, BC: tl.constexpr,
              F32: tl.constexpr):
    acc = tl.zeros([NR, BC], tl.float32)
    for k in tl.static_range(KR):
        i = tl.load(idx_ptr + rows * KR + k, mask=rm, other=0).to(tl.int64)
        acc += tl.load(y_ptr + i[:, None] * 2304 + c[None, :], mask=rm[:, None], other=0.).to(tl.float32) * \
            tl.load(ws_ptr + i, mask=rm, other=0.)[:, None]
    tl.store(f_ptr + rows[:, None].to(tl.int64) * 2304 + c[None, :], _bf16(acc, F32), mask=rm[:, None])


@triton.jit
def _tail_kernel(y_ptr, ws_ptr, idx_ptr, f_ptr, st_ptr, w_ptr, r_ptr, out_ptr, n_ptr, T, scl, eps,
                 KR: tl.constexpr, RECIP: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr, F32: tl.constexpr):
    """Rows t0..t0+BT: (1) f[t] = bf16(sum_k fp32(y[row_k]) * ws[row_k]) in sorted (= expert) order (moe_triton.finish)
    for those rows and the 3 rows before (halo; the neighbour program writes the same bits), into the [T,H] scratch f;
    (2) res_out on f: h = bf16(r + sconv(f)), n = rms(h). The halo makes the conv local to the program."""
    t0 = tl.program_id(0) * BT
    t = t0 + tl.arange(0, BT)
    tm = t < T
    th = t0 - 4 + tl.arange(0, 4)                                            # halo rows t0-4..t0-1 (t0-4 unused)
    hm = (th >= t0 - 3) & (th >= 0)
    for c0 in range(0, 2304, BC):
        c = c0 + tl.arange(0, BC)
        _fin_rows(y_ptr, ws_ptr, idx_ptr, f_ptr, t, tm, c, KR, BT, BC, F32)
        _fin_rows(y_ptr, ws_ptr, idx_ptr, f_ptr, th, hm, c, KR, 4, BC, F32)
    tl.debug_barrier()                                                       # this program's f rows (incl. halo)
    ss = tl.zeros([BT], tl.float32)
    for c0 in range(0, 2304, BC):
        c = c0 + tl.arange(0, BC)
        cm = c < 2304
        yv = _bf16(_conv_y(f_ptr, st_ptr, w_ptr, w_ptr, t, c, tm, cm, 2304, 1, scl, False, RECIP, BT, BC),
                   F32).to(tl.float32)
        off = t[:, None] * 2304 + c[None, :]
        v = _bf16(tl.load(r_ptr + off, mask=tm[:, None], other=0.) + yv, F32)
        tl.store(out_ptr + off, v, mask=tm[:, None])
        v = v.to(tl.float32)
        ss += tl.sum(v * v, 1)
    tl.debug_barrier()
    inv = tl.math.rsqrt(ss / 2304 + eps)
    for c0 in range(0, 2304, BC):
        off = t[:, None] * 2304 + c0 + tl.arange(0, BC)[None, :]
        v = tl.load(out_ptr + off, mask=tm[:, None], other=0.).to(tl.float32)
        tl.store(n_ptr + off, v * inv[:, None], mask=tm[:, None])


def moe_tail(y, tok, ws, st, w, r):
    """model.moe_tail: y [R,H] bf16 per routed row (sorted by expert), tok/ws per row -> (h bf16, rms(h) fp32), with
    h = bf16(r + sconv(finish(y))) as moe_triton.finish + res_out, in one launch."""
    T, C = r.shape
    R = y.shape[0]
    assert R % T == 0 and C == H
    idx = torch.sort(tok, stable=True).indices.int()                         # as moe_triton.finish
    f = torch.empty(T, C, dtype=torch.bfloat16, device=y.device)            # scratch (L2-hot when re-read)
    out = torch.empty(T, C, dtype=torch.bfloat16, device=y.device)
    nrm = torch.empty(T, C, dtype=torch.float32, device=y.device)
    _tail_kernel[(triton.cdiv(T, 8),)](y.contiguous(), ws.contiguous(), idx, f, st, w.reshape(C, 4).contiguous(),
                                       r.contiguous(), out, nrm, T,
                                       _recip(math.sqrt(5)) if y.is_cuda else math.sqrt(5), 1e-6, KR=R // T,
                                       RECIP=y.is_cuda, BT=8, BC=256, F32=INTERP, num_warps=4, **_NOFMA)
    st.copy_(torch.cat((st, f[-3:]))[-3:])
    return out, nrm
