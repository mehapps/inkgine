"""Batched decode (DESIGN-batching.md sec. 1): one step = R rows (slot, pos, token), R <= 16 = one tensor-core M tile.
Batched decode = R slots x 1 row; speculative verify = 1 slot x T rows (consecutive positions); mixed steps work too.
Same API as loopbatch.LoopBatch (the scheduler's contract):

    bd = BatchDecoder(model, slots=INK_MAX_BATCH)
    bd.load(slot, seq)        take over a prefilled Seq at seq.pos (pool page table, pages reserved up front)
    lg = bd.step(rows)        rows [(slot, pos, token)], a slot's rows consecutive with pos = its next positions
                              -> logits [R, VOCAB] fp32 (static buffer, overwritten by the next step)
    lg = bd.step(rows, tok={r0: ids})  tokens of rows r0.. from a device int64 tensor (e.g. the drafter's ids; those
                              rows' token entries are ignored): no host round trip. With INK_DEC_EMBED_GPU=1 the
                              embedding table is on the device (+0.9 GB) and the row lookup runs inside the step graph;
                              without it ids are read to the host (a sync) and gathered there as usual. Host metadata
                              goes through pinned buffers (non-blocking copies), so step() never waits for the GPU.
    bd.commit({slot: n})      accepted rows of each verify slot (n >= 1); single-row slots commit inside step()
    tp = bd.tap_rows(r0, n)   DSpark taps of step rows r0..r0+n-1: [4, n, H] bf16 view of a static buffer (h entering
                              visits 5, 23, 41 + the final normed state; overwritten by the next step)
    tp = bd.tap_hist(slot, p0, n)  taps of the slot's committed positions p0..p0+n-1 from a per-slot ring of TH
                              positions written inside the step graph (the drafter catches up from it in batches)
    st = bd.state(slot)       Engine.state_of's format without 'f' (rings time-ordered, last min(pos, 1023))
    bd.free(slot); bd.drop(keep=0); bd.busy(); bd.idle(); bd.n; bd.dec

Math = decode_kernels.Fused (same SPEC sections, same bf16 rounding points) with a leading row dimension:
- small activation kernels (attn-res mix, loop gates, router) run Fused's kernels per row (row = a grid axis), so
  they are bit-identical to the single-row path;
- the fused attention projection and wo are small GEMMs: each weight tile is read once for all rows, tl.dot over
  16 rows (padded), fp32 accumulate (reduction order differs from Fused's GEMV: bf16 noise, not bit-exact);
- MoE: an on-device plan (top-8 per row, tie rule as Fused, union of the chosen experts, row -> slot map), then
  w13/w2 over the union: every routed expert is read once per step for all rows that chose it; per-row parts are
  summed in slot order (as Fused);
- attention per row with its slot's ring/page table and position: near over a WR = 1088 slot ring (slot pos % WR:
  verify rows never overwrite a key some row still needs; mask d < 1024 as before), far over the 4-bit pages with
  the rows of one slot packed into the M tile (up to 5 rows x 3 q heads): each K/V page is read once per sequence;
- short convs never write state during a step: inputs of earlier rows of the same slot come from this step
  (Y / the program's own tile), the rest from the slot's state; every conv input is kept in `xh`, and the commit
  kernel builds the new state from it (single-row slots at the end of the step, verify slots in commit()).
Every row's result is independent of R, of its row index and of the other rows (batch invariance), so greedy
speculative decoding gives exactly the plain greedy output. Pad rows (R bucket > rows) write nothing and route to no
expert. All buffers are static and everything that varies is a device tensor (meta), so each R bucket (1, 2, 4, 8,
16) is one CUDA graph (INK_GRAPH=1).
"""
import os
from types import SimpleNamespace

import torch

import kvpool
import decode as D
from decode import DOT_BLK
from decode_kernels import (BLK, CH, NI, NW, Fused, _enter, _far_lse, _gate_out_kernel, _gate_z_kernel, _mix_kernel,
                            _near_fin_kernel, _near_qr, _near_tile, _pf_part, _quant_put, _r, _router_kernel, _tanh)
from model import H, HD, NKV, NQ, VISITS, VOCAB, WIN, rms
from moe_triton import INTERP, _bf16, _dot_tile, _mma

import triton
import triton.language as tl

RM = 16                      # rows per step (one M tile); R buckets below
BUCKETS = (1, 2, 4, 8, 16)
WR = 1088                    # ring slots per visit: 1024 + 64 (DESIGN 3.3)
CC = 6656                    # conv channels per visit: k 1024 | v 1024 | attn 2304 | mlp 2304
CONV = (('k', 0, 1024), ('v', 1024, 1024), ('attn', 2048, H), ('mlp', 4352, H))
NY = 8664                    # Y row stride (widest fused projection)


def mt(R):
    """M tile of the far kernel for R-row steps: up to mt // 3 rows x 3 q heads of one slot share every far K/V tile,
    so a verify of <= 10 rows reads the far KV once (R <= 4 keeps the 16-row tile). shortcut: MT 64 (a 16-row verify in
    one read) spills 7-16 KB per program on sm_120 at any warps/stages, so T 11..16 read it twice; fix the kernel's
    MT 64 layout if long copy rounds at long context matter."""
    return 16 if R <= 4 else 32


def far_spl(pos, nsp):
    """INK_DEC_FAR2 split length of a row at pos (as _bfar_kernel / decode's far kernel compute it; 0: fixed splits)."""
    return -(-(-(-max(pos - 1023, 0) // nsp)) // DOT_BLK) * DOT_BLK if nsp else 0


def far_groups(poss, nsp, fg):
    """Far groups of one slot's consecutive rows at positions poss -> [(first row, rows)]: <= fg rows each, and (FAR2)
    one split length per group, so every row gets exactly the splits of its own single-row step (verify rows == plain
    rows bit for bit; a split-length step inside the group splits it)."""
    out, a = [], 0
    while a < len(poss):
        e, k = a + 1, far_spl(poss[a], nsp)
        while e < len(poss) and e - a < fg and far_spl(poss[e], nsp) == k:
            e += 1
        out.append((a, e - a))
        a = e
    return out
TAPS = (5, 23, 41)           # DSpark drafter taps: h entering these visits (dspark.Cfg.taps[:3]), + final normed state
TH = 32                      # tap history ring per slot (positions); a reader may lag the slot by <= TH - 8 positions
# GEMM tiles (BN output rows per program; tl.dot needs >= 16), K tiles, w2_fin columns, near ring tile (divides WR).
# Picked on the 5090 with tests/tune_batch.py --sweep (2026-11-22; every variant bit-identical to BN 16): proj / wo
# flat across BN/warps/stages; w13u/w2u BN 32 (num_stages NSB); R = 1 uses BK 256 + 3 stages for w13u (W13_R1).
BB = dict(proj=16, wo=16, w13=32, w2=32, kbk=128, kbk2=128, w2f=64, near=64, cb=256)
NWB = dict(proj=4, wo=4, w13=4, w2=4)
NSB = dict(w13=1, w2=1)
W13_R1 = dict(BK=256, num_stages=3)


# --------------------------------------------------------------------------------------------- per-row wrappers
@triton.jit
def _bmix_kernel(q_ptr, B, sB, eidx, n, x_ptr, sx, bslot, a_ptr, sa, xo_ptr, sxo, nwt, qm_ptr, midx, nm, xm_ptr, sxm,
                 m_ptr, sm, nq, FIRST: tl.constexpr, NORM: tl.constexpr, STORE_A: tl.constexpr, F32: tl.constexpr,
                 H: tl.constexpr, CH: tl.constexpr, UNR: tl.constexpr = 0, PDL: tl.constexpr = False, PF: tl.constexpr = False):
    """Fused's _mix_kernel for row program_id(1) (each row has its own depth entries B, inputs and outputs)."""
    _enter(PDL)
    r = tl.program_id(1)
    _mix_kernel(q_ptr, B + r * sB, eidx, n, x_ptr + r * sx, bslot, a_ptr + r * sa, xo_ptr + r * sxo, nwt, qm_ptr,
                midx, nm, xm_ptr + r * sxm, m_ptr + r * sm, nq, FIRST, NORM, STORE_A, F32, H, CH, UNR)


@triton.jit
def _bgate_z_kernel(a_ptr, sa, xs, m_ptr, sm, dtab, z_ptr, sz, F32: tl.constexpr, H: tl.constexpr, CH: tl.constexpr,
                    ZB: tl.constexpr, UNR: tl.constexpr = 0, PDL: tl.constexpr = False, PF: tl.constexpr = False):
    _enter(PDL)
    r = tl.program_id(2)
    _gate_z_kernel(a_ptr + r * sa, xs, m_ptr + r * sm, dtab, z_ptr + r * sz, F32, H, CH, ZB, UNR)


@triton.jit
def _bgate_out_kernel(a_ptr, sa, xs, m_ptr, sm, gtab, z_ptr, sz, o_ptr, so, H: tl.constexpr, BH: tl.constexpr, UNR: tl.constexpr = 0, PDL: tl.constexpr = False, PF: tl.constexpr = False):
    _enter(PDL)
    r = tl.program_id(2)
    _gate_out_kernel(a_ptr + r * sa, xs, m_ptr + r * sm, gtab, z_ptr + r * sz, o_ptr + r * so, H, BH)


@triton.jit
def _brouter_kernel(x_ptr, sx, nw, u_ptr, gw, s_ptr, NORM: tl.constexpr, F32: tl.constexpr, H: tl.constexpr,
                    CH: tl.constexpr, RB: tl.constexpr, UNR: tl.constexpr = 0, PDL: tl.constexpr = False, PF: tl.constexpr = False):
    _enter(PDL)
    r = tl.program_id(1)
    _router_kernel(x_ptr + r * sx, nw, u_ptr + r * H, gw, s_ptr + r * 128, NORM, F32, H, CH, RB, UNR)


@triton.jit
def _bnorm_kernel(G, nwt, X, F32: tl.constexpr, H: tl.constexpr, CH: tl.constexpr, UNR: tl.constexpr = 0, PDL: tl.constexpr = False, PF: tl.constexpr = False):
    """Loop-gated visits: X[r, j] = bf16(rms(G[r, j]) * nw_j) (Fused does this inside _proj_kernel's NORM path)."""
    _enter(PDL)
    j, r = tl.program_id(0), tl.program_id(1)
    xp = G + (r * 3 + j) * H
    c = tl.arange(0, CH)
    ss = tl.zeros([CH], tl.float32)
    for k0 in tl.range(0, H, CH, loop_unroll_factor=UNR * (H // CH)):
        x = tl.load(xp + k0 + c)
        ss += x * x
    inv = 1 / tl.sqrt(tl.sum(ss, 0) / H + 1e-6)
    nw = tl.load(nwt + j).to(tl.pointer_type(tl.float32))
    for k0 in tl.range(0, H, CH, loop_unroll_factor=UNR * (H // CH)):
        tl.store(X + (r * 3 + j) * H + k0 + c, _bf16(tl.load(xp + k0 + c) * inv * tl.load(nw + k0 + c), F32))


# --------------------------------------------------------------------------------------------- small GEMMs
@triton.jit
def _bproj_kernel(w_ptr, n0, n1, N, x_ptr, xs, sx, y_ptr, nr, s_ptr, F32: tl.constexpr, H: tl.constexpr,
                  CH: tl.constexpr, BN: tl.constexpr, RM: tl.constexpr, W8: tl.constexpr = False, UNR: tl.constexpr = 0, PDL: tl.constexpr = False, PF: tl.constexpr = False):
    """Y[r, n] = bf16(x_{r,j(n)} @ W[n]^T) for the RM rows (rows >= nr masked); j(n) = 0 below n0, 1 below n0+n1,
    else 2 (x_{r,j} = x_ptr + r*sx + j*xs, bf16). A program straddling an input boundary does one dot per input
    with the other columns zeroed (adds exact zeros). W8: W is e4m3 (exact in bf16), column n's sum times s_ptr[n]."""
    _enter(PDL)
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    nm = n < N
    j = (n >= n0).to(tl.int32) + (n >= n0 + n1).to(tl.int32)
    j0 = tl.min(tl.where(nm, j, 2), 0)
    j1 = tl.max(tl.where(nm, j, 0), 0)
    rr = tl.arange(0, RM)
    rm = (rr < nr)[:, None]
    c = tl.arange(0, CH)
    acc = tl.zeros([RM, BN], tl.float32)
    for k0 in range(0, H, CH):
        w = tl.load(w_ptr + n[:, None] * H + k0 + c[None, :], mask=nm[:, None], other=0.).to(tl.bfloat16)
        for jj in tl.static_range(3):
            if (j0 <= jj) & (jj <= j1):
                x = tl.load(x_ptr + rr[:, None] * sx + jj * xs + k0 + c[None, :], mask=rm, other=0.)
                acc = _mma(x, tl.trans(tl.where((j == jj)[:, None], w, tl.zeros_like(w))), acc, F32)
    if W8:
        acc = acc * tl.load(s_ptr + n, mask=nm, other=0.)[None, :]
    tl.store(y_ptr + rr[:, None] * 8664 + n[None, :], _bf16(acc, F32), mask=rm & nm[None, :])


@triton.jit
def _prev(cur, xs, st, k, i: tl.constexpr, CC: tl.constexpr, mask):
    """i-th previous conv input (fp32) of a row at index k in its slot group: the row i above in this step
    (cur - i*xs) if k >= i, else the slot's state row 3 - i + k (st = state row 0 at the same channels)."""
    k = tl.where(mask, k, k)                     # broadcast to mask's shape (interpreter: scalar & tensor masks fail)
    a = tl.load(cur - i * xs, mask=mask & (k >= i), other=0.).to(tl.float32)
    b = tl.load(st + (3 - i + k) * CC, mask=mask & (k < i), other=0.).to(tl.float32)
    return tl.where(k >= i, a, b)


@triton.jit
def _cv4(x, p1, p2, p3, w0, w1, w2, w3, div):
    """Fused's _conv4 arithmetic (state rows s0, s1, s2 = p3, p2, p1), no state write."""
    return (p3 * w0 + p2 * w1 + p1 * w2 + x * w3 + x) / div


@triton.jit
def _bwo_kernel(x_ptr, w_ptr, cst, sst, xh, cw, h_ptr, r_ptr, meta, nr, s_ptr, BND: tl.constexpr, F32: tl.constexpr,
                H: tl.constexpr, CH: tl.constexpr, BN: tl.constexpr, RM: tl.constexpr, CC: tl.constexpr,
                W8: tl.constexpr = False, UNR: tl.constexpr = 0, PDL: tl.constexpr = False, PF: tl.constexpr = False, pfa=None, pfn=0):
    """o = bf16(att @ wo^T) for all rows (W8: e4m3 wo, column sums times s_ptr); o -> xh (conv input); attn short
    conv (earlier rows of the slot from xh, written by this program, else the slot's state); r = o (layer % 6 == 0)
    or h + o (fp32). PF: prefetch a share of the pfn bytes at pfa (router)."""
    _enter(PDL)
    if PF:
        _pf_part(pfa, pfn, tl.program_id(0), tl.num_programs(0))
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    rr = tl.arange(0, RM)
    rv = rr < nr
    rm = rv[:, None]
    c = tl.arange(0, CH)
    acc = tl.zeros([RM, BN], tl.float32)
    for k0 in range(0, 3072, CH):
        x = tl.load(x_ptr + rr[:, None] * 3072 + k0 + c[None, :], mask=rm, other=0.)
        w = tl.load(w_ptr + n[:, None] * 3072 + k0 + c[None, :]).to(tl.bfloat16)
        acc = _mma(x, tl.trans(w), acc, F32)
    if W8:
        acc = acc * tl.load(s_ptr + n)[None, :]
    o = _r(acc, F32)
    m2 = rm & (n >= 0)[None, :]
    xo = xh + rr[:, None] * CC + 2048 + n[None, :]
    tl.store(xo, o.to(tl.bfloat16), mask=m2)
    tl.debug_barrier()                                   # earlier rows' o (other threads) visible below
    sl = tl.load(meta + rr, mask=rv, other=0).to(tl.int32)
    k = tl.load(meta + 2 * RM + rr, mask=rv, other=0).to(tl.int32)[:, None]
    st = cst + sl[:, None] * sst + 2048 + n[None, :]
    y = _cv4(o, _prev(xo, CC, st, k, 1, CC, m2), _prev(xo, CC, st, k, 2, CC, m2), _prev(xo, CC, st, k, 3, CC, m2),
             tl.load(cw + n * 4)[None, :], tl.load(cw + n * 4 + 1)[None, :], tl.load(cw + n * 4 + 2)[None, :],
             tl.load(cw + n * 4 + 3)[None, :], 2.23606797749979)
    y = _r(y, F32)
    if not BND:
        y = tl.load(h_ptr + rr[:, None] * H + n[None, :], mask=m2, other=0.).to(tl.float32) + y
    tl.store(r_ptr + rr[:, None] * H + n[None, :], y, mask=m2)


# --------------------------------------------------------------------------------------------- attention
@triton.jit
def _bpre_kernel(Y, koff, voff, tkoff, tvoff, qsc, rsc, kw, vw, cst, sst, xh, rk, rv, rpos, meta,
                 c0, c1, s0, s1, tbl, ts, qo, ro, FULL: tl.constexpr, F32: tl.constexpr, PG: tl.constexpr,
                 WR: tl.constexpr, RM: tl.constexpr, CC: tl.constexpr, UNR: tl.constexpr = 0, PDL: tl.constexpr = False, PF: tl.constexpr = False):
    """Fused's _pre_kernel for row program_id(1): its own pos/tau/slot; K/V conv inputs of earlier rows of the slot
    from Y, else the slot's state (no state write; inputs -> xh); ring slot pos % WR, page row via the slot's table.
    Pad rows (live 0) write nothing."""
    _enter(PDL)
    h, r = tl.program_id(0), tl.program_id(1)
    sl = tl.load(meta + r).to(tl.int32)
    pos = tl.load(meta + RM + r)
    k = tl.load(meta + 2 * RM + r).to(tl.int32)
    lv = tl.load(meta + 3 * RM + r) > 0
    y_ptr = Y + r * 8664
    qo = qo + r * 3072
    ro = ro + r * 384
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
    rq = tl.load(y_ptr + 3072 + hq[:, None] * 16 + j[None, :], mask=gm, other=0.).to(tl.float32)
    rq = rq * tl.load(rsc + hq[:, None] * 16 + j[None, :], mask=gm, other=0.)
    rq = _r(rq * tl.minimum(8.0 / (tl.sqrt(tl.sum(rq * rq, 1)) + 1e-6), 1.)[:, None], F32)
    if FULL:
        rq = _r(rq * tau, F32)
    tl.store(ro + hq[:, None] * 16 + j[None, :], rq.to(tl.bfloat16), mask=gm)
    ch = h * 128 + d
    on = ch >= 0
    kp = y_ptr + koff + ch
    vp = y_ptr + voff + ch
    kx = tl.load(kp).to(tl.float32)
    vx = tl.load(vp).to(tl.float32)
    st = cst + sl * sst + ch
    k1, k2, k3 = _prev(kp, 8664, st, k, 1, CC, on), _prev(kp, 8664, st, k, 2, CC, on), _prev(kp, 8664, st, k, 3, CC, on)
    v1, v2, v3 = (_prev(vp, 8664, st + 1024, k, 1, CC, on), _prev(vp, 8664, st + 1024, k, 2, CC, on),
                  _prev(vp, 8664, st + 1024, k, 3, CC, on))
    if FULL:  # dynamic conv: W[c, i] = static_w[3-i, c] + taps[h, 3-i]
        tk = y_ptr + tkoff + h * 4
        tv = y_ptr + tvoff + h * 4
        kk = _cv4(kx, k1, k2, k3,
                  tl.load(kw + 3 * 1024 + ch) + tl.load(tk + 3).to(tl.float32),
                  tl.load(kw + 2 * 1024 + ch) + tl.load(tk + 2).to(tl.float32),
                  tl.load(kw + 1024 + ch) + tl.load(tk + 1).to(tl.float32),
                  tl.load(kw + ch) + tl.load(tk).to(tl.float32), 3.0)
        vv = _cv4(vx, v1, v2, v3,
                  tl.load(vw + 3 * 1024 + ch) + tl.load(tv + 3).to(tl.float32),
                  tl.load(vw + 2 * 1024 + ch) + tl.load(tv + 2).to(tl.float32),
                  tl.load(vw + 1024 + ch) + tl.load(tv + 1).to(tl.float32),
                  tl.load(vw + ch) + tl.load(tv).to(tl.float32), 3.0)
    else:
        kk = _cv4(kx, k1, k2, k3, tl.load(kw + ch * 4), tl.load(kw + ch * 4 + 1), tl.load(kw + ch * 4 + 2),
                  tl.load(kw + ch * 4 + 3), 2.23606797749979)
        vv = _cv4(vx, v1, v2, v3, tl.load(vw + ch * 4), tl.load(vw + ch * 4 + 1), tl.load(vw + ch * 4 + 2),
                  tl.load(vw + ch * 4 + 3), 2.23606797749979)
    kk = _r(kk, F32)
    kk = _r(kk * tl.minimum(22.627416997969522 / (tl.sqrt(tl.sum(kk * kk, 0)) + 1e-6), 1.), F32)
    vv = _r(vv, F32)
    vv = _r(vv / tl.sqrt(tl.sum(vv * vv, 0) / 128 + 1e-6), F32)
    if lv:
        tl.store(xh + r * CC + ch, kx.to(tl.bfloat16))
        tl.store(xh + r * CC + 1024 + ch, vx.to(tl.bfloat16))
        slot = pos % WR
        tl.store(rk + sl * (WR * 1024) + slot * 1024 + ch, kk.to(tl.bfloat16))
        tl.store(rv + sl * (WR * 1024) + slot * 1024 + ch, vv.to(tl.bfloat16))
        if h == 0:
            tl.store(rpos + sl * WR + slot, pos)
        if FULL:
            row = tl.load(tbl + sl * ts + pos // PG) * PG + pos % PG
            _quant_put(kk, c0, s0, row, h, F32)
            _quant_put(vv, c1, s1, row, h, F32)


@triton.jit
def _bfar_kernel(qb, c0, c1, s0, s1, tbl, ts, meta, o_ptr, l_ptr, ns, scale, SPLIT: tl.constexpr, PG: tl.constexpr,
                 BLK: tl.constexpr, F32: tl.constexpr, RM: tl.constexpr, UNR: tl.constexpr = 0, PDL: tl.constexpr = False, PF: tl.constexpr = False, NSP: tl.constexpr = 0,
                 MT: tl.constexpr = 16):
    """decode._far_dot_kernel for a far group (program_id(2)): up to MT // 3 rows of one slot x 3 q heads in the MT-row
    M tile (MT = mt(R): 16 for R <= 4, else 32: a verify group of up to 10 rows shares one tile), so every
    K/V page is read once per slot per step, however many verify rows; per M row its own far range [0, pos - 1023)
    within the split. Partials per row: o [RM, ns, 24, 128], lse [RM, ns, 24] (empty range: o = 0, lse = -inf)."""
    _enter(PDL)
    h, sp, gi = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    n = tl.load(meta + 5 * RM + gi).to(tl.int32)
    if n > 0:
        r0 = tl.load(meta + 4 * RM + gi).to(tl.int32)
        sl = tl.load(meta + r0).to(tl.int32)
        mm = tl.arange(0, MT)
        i = mm // 3
        mv = i < n
        row = r0 + tl.where(mv, i, 0)
        hq = h * 3 + mm % 3
        kend = (tl.load(meta + RM + row, mask=mv, other=0) - 1023).to(tl.int32)
        if NSP > 0:   # INK_DEC_FAR2: NSP even splits of the group's longest far range (its last row)
            spl = tl.cdiv(tl.cdiv(tl.maximum(tl.max(tl.where(mv, kend, 0), 0), 0), NSP), BLK) * BLK
        else:
            spl = SPLIT
        lo = sp * spl
        hr = tl.where(mv, tl.minimum(lo + spl, kend), lo)              # per M row: keys [lo, hr)
        hi = tl.max(hr, 0)
        d, rk = tl.arange(0, 64), tl.arange(0, BLK)
        j8 = tl.arange(0, 8)
        qp = qb + row[:, None] * 3072 + hq[:, None] * 128 + 2 * d[None, :]
        qe = tl.load(qp, mask=mv[:, None], other=0.)
        qo = tl.load(qp + 1, mask=mv[:, None], other=0.)
        m = tl.full([MT], float('-inf'), tl.float32)
        l = tl.zeros([MT], tl.float32)
        ae = tl.zeros([MT, 64], tl.float32)
        ao = tl.zeros([MT, 64], tl.float32)
        for k0 in range(lo, hi, BLK):
            prow = tl.load(tbl + sl * ts + k0 // PG) * PG + k0 % PG + rk
            km = k0 + rk < hi
            co = prow[:, None] * 512 + h * 64 + d[None, :]
            kc = tl.load(c0 + co, mask=km[:, None], other=0)
            vc = tl.load(c1 + co, mask=km[:, None], other=0)
            if NSP > 0:                                               # scales as [BLK, 8], broadcast (same values)
                s8 = prow[:, None] * 64 + h * 8 + j8[None, :]
                ks = tl.reshape(tl.broadcast_to(tl.load(s0 + s8, mask=km[:, None], other=0.).to(tl.float32)[:, :, None],
                                                (BLK, 8, 8)), (BLK, 64))
                vs = tl.reshape(tl.broadcast_to(tl.load(s1 + s8, mask=km[:, None], other=0.).to(tl.float32)[:, :, None],
                                                (BLK, 8, 8)), (BLK, 64))
            else:
                so = prow[:, None] * 64 + h * 8 + (d // 8)[None, :]
                ks = tl.load(s0 + so, mask=km[:, None], other=0.).to(tl.float32)
                vs = tl.load(s1 + so, mask=km[:, None], other=0.).to(tl.float32)
            klo = (((kc & 15).to(tl.float32) - 8) * ks).to(tl.bfloat16)
            khi = (((kc >> 4).to(tl.float32) - 8) * ks).to(tl.bfloat16)
            s = _mma(qe, tl.trans(klo), tl.zeros([MT, BLK], tl.float32), F32)
            s = _mma(qo, tl.trans(khi), s, F32) * scale
            s = tl.where((k0 + rk)[None, :] < hr[:, None], s, float('-inf'))
            mn = tl.maximum(m, tl.max(s, 1))
            mns = tl.where(mn == float('-inf'), 0., mn)              # a row with no key yet (alone it skips the tile)
            a = tl.exp(m - mns)
            p = tl.exp(s - mns[:, None])
            l = l * a + tl.sum(p, 1)
            pb = _bf16(p, F32)
            ae = _mma(pb, (((vc & 15).to(tl.float32) - 8) * vs).to(tl.bfloat16), ae * a[:, None], F32)
            ao = _mma(pb, (((vc >> 4).to(tl.float32) - 8) * vs).to(tl.bfloat16), ao * a[:, None], F32)
            m = mn
        ok = l > 0
        inv = tl.where(ok, 1 / tl.where(ok, l, 1.), 0.)
        out = (row * ns + sp) * 24 + hq
        op = o_ptr + out[:, None] * 128 + 2 * d[None, :]
        tl.store(op, ae * inv[:, None], mask=mv[:, None])
        tl.store(op + 1, ao * inv[:, None], mask=mv[:, None])
        tl.store(l_ptr + out, tl.where(ok, m + tl.log(tl.where(ok, l, 1.)), float('-inf')), mask=mv)


@triton.jit
def _bnear_part_kernel(qb, rb, proj, rk, rpos, meta, pm, pl, F32: tl.constexpr, BLK: tl.constexpr, WR: tl.constexpr,
                       RM: tl.constexpr, UNR: tl.constexpr = 0, PDL: tl.constexpr = False, PF: tl.constexpr = False, pfa=None, pfn=0):
    """Fused's _near_part_kernel, program (row, kv head, ring tile) over the row's slot ring. PF: prefetch a share of
    the pfn bytes at pfa (wo)."""
    _enter(PDL)
    r, h, s = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if PF:
        _pf_part(pfa, pfn, (r * tl.num_programs(1) + h) * tl.num_programs(2) + s,
                 tl.num_programs(0) * tl.num_programs(1) * tl.num_programs(2))
    if tl.load(meta + 3 * RM + r) > 0:
        sl = tl.load(meta + r).to(tl.int32)
        q, R = _near_qr(qb + r * 3072, rb + r * 384, h)
        S = _near_tile(q, R, proj, rk + sl * (WR * 1024), rpos + sl * WR, tl.load(meta + RM + r), h, s * BLK, BLK, F32)
        m = tl.max(S, 1)
        ms = tl.where(m == float('-inf'), 0., m)
        g = tl.arange(0, 16)
        o = (r * (WR // BLK) + s) * 24 + h * 3 + g
        tl.store(pm + o, m, mask=g < 3)
        tl.store(pl + o, tl.sum(tl.exp(S - ms[:, None]), 1), mask=g < 3)


@triton.jit
def _bnear_out_kernel(qb, rb, proj, Y, rk, rv, rpos, meta, pm, pl, fl, ns, pacc, plse, FAR: tl.constexpr,
                      F32: tl.constexpr, BLK: tl.constexpr, WR: tl.constexpr, RM: tl.constexpr, NTP: tl.constexpr, UNR: tl.constexpr = 0, PDL: tl.constexpr = False, PF: tl.constexpr = False,
                      NSF: tl.constexpr = 0):
    """Fused's _near_out_kernel per (row, kv head, ring tile); WR // BLK tiles (NTP = next power of two)."""
    _enter(PDL)
    r, h, s = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    NT: tl.constexpr = WR // BLK
    if tl.load(meta + 3 * RM + r) > 0:
        sl = tl.load(meta + r).to(tl.int32)
        g = tl.arange(0, 16)
        gm = g < 3
        hq = h * 3 + g
        t = tl.arange(0, NTP)
        tm = (t < NT)[:, None] & gm[None, :]
        po = (r * NT + t)[:, None] * 24 + hq[None, :]
        M = tl.where((t < NT)[:, None], tl.load(pm + po, mask=tm, other=0.), float('-inf'))
        L = tl.load(pl + po, mask=tm, other=0.)
        mx = tl.max(M, 0)                                           # finite: the current token is always in the ring
        lse = mx + tl.log(tl.sum(L * tl.exp(M - mx[None, :]), 0))
        sk = tl.load(Y + r * 8664 + 3456 + hq, mask=gm, other=0.).to(tl.float32)
        tot = _far_lse(fl + r * ns * 24, ns, hq, gm, FAR, NSF)
        mx = tl.maximum(tl.maximum(lse, sk), tot)
        lse = mx + tl.log(tl.exp(lse - mx) + tl.exp(sk - mx) + tl.exp(tot - mx))
        if s == 0:
            tl.store(plse + r * 24 + hq, lse, mask=gm)
        q, R = _near_qr(qb + r * 3072, rb + r * 384, h)
        S = _near_tile(q, R, proj, rk + sl * (WR * 1024), rpos + sl * WR, tl.load(meta + RM + r), h, s * BLK, BLK, F32)
        p = _bf16(tl.exp(S - lse[:, None]), F32)
        d = tl.arange(0, 128)
        slots = s * BLK + tl.arange(0, BLK)
        acc = _mma(p, tl.load(rv + sl * (WR * 1024) + slots[:, None] * 1024 + h * 128 + d[None, :]),
                   tl.zeros([16, 128], tl.float32), F32)
        tl.store(pacc + ((r * NT + s) * 24 + hq)[:, None] * 128 + d[None, :], acc, mask=gm[:, None])


@triton.jit
def _bnear_fin_kernel(pacc, plse, fo, fl, ns, Y, att, meta, FAR: tl.constexpr, F32: tl.constexpr, NT: tl.constexpr,
                      RM: tl.constexpr, UNR: tl.constexpr = 0, PDL: tl.constexpr = False, PF: tl.constexpr = False, NSF: tl.constexpr = 0):
    _enter(PDL)
    r = tl.program_id(1)
    if tl.load(meta + 3 * RM + r) > 0:
        _near_fin_kernel(pacc + r * (NT * 3072), plse + r * 24, fo + r * ns * 3072, fl + r * ns * 24, ns,
                         Y + r * 8664, att + r * 3072, FAR, F32, NT, NSF=NSF)


# --------------------------------------------------------------------------------------------- MoE
@triton.jit
def _plan_kernel(S, bias, meta, nr, ids, wts, eslot, ulist, ucnt, RM: tl.constexpr, UNR: tl.constexpr = 0, PDL: tl.constexpr = False, PF: tl.constexpr = False):
    """One program: top-8 of sigmoid scores + bias per row (Fused's tie rule: lowest index), normalized weights;
    eslot[e, r] = slot of expert e in live row r's top-8 (-1: not chosen); ulist = the chosen experts ascending,
    ucnt = how many. Pad rows choose nothing."""
    _enter(PDL)
    rr = tl.arange(0, RM)
    ee = tl.arange(0, 128)
    lv = (tl.load(meta + 3 * RM + rr) > 0) & (rr < nr)
    sv = tl.load(S + rr[:, None] * 128 + ee[None, :])
    sc = sv + tl.load(bias + ee)[None, :]
    i8 = tl.arange(0, 8)
    idv = tl.zeros([RM, 8], tl.int32)
    ws = tl.zeros([RM, 8], tl.float32)
    es = tl.full([128, RM], -1, tl.int32)
    for i in tl.static_range(8):
        mx = tl.max(sc, 1)
        jj = tl.min(tl.where(sc == mx[:, None], ee[None, :], 128), 1)
        hit = ee[None, :] == jj[:, None]
        idv = tl.where(i8[None, :] == i, jj[:, None], idv)
        ws = tl.where(i8[None, :] == i, tl.sum(tl.where(hit, sv, 0.), 1)[:, None], ws)
        sc = tl.where(hit, float('-inf'), sc)
        es = tl.where((ee[:, None] == jj[None, :]) & lv[None, :], i, es)
    wn = ws / (tl.sum(ws, 1)[:, None] + 1e-20) * 8.0
    tl.store(ids + rr[:, None] * 8 + i8[None, :], idv)
    tl.store(wts + rr[:, None] * 8 + i8[None, :], wn)
    tl.store(eslot + ee[:, None] * RM + rr[None, :], es)
    used = (tl.max(es, 1) >= 0).to(tl.int32)
    tl.store(ulist + tl.cumsum(used, 0) - 1, ee, mask=used > 0)
    tl.store(ucnt, tl.sum(used, 0))


@triton.jit
def _w13u_kernel(U, ulist, ucnt, eslot, w_ptr, s_ptr, gs_ptr, lwe, lwr, lse, lsr, act,
                 FP4: tl.constexpr, F32: tl.constexpr, H: tl.constexpr, NI: tl.constexpr, BN: tl.constexpr,
                 BK: tl.constexpr, RM: tl.constexpr, UNR: tl.constexpr = 0, PDL: tl.constexpr = False, PF: tl.constexpr = False, pfa=None, pfn=0, pfb=None, pfm=0):
    """Program (n block, union entry u): expert ulist[u]'s gate/up rows (2n, 2n+1) dequantized once (as
    Weights.expert), dot the rows that chose it; act[r, slot] = bf16(softcap-SwiGLU)."""
    _enter(PDL)
    nb, u = tl.program_id(0), tl.program_id(1)
    if u < tl.load(ucnt):
        e = tl.load(ulist + u).to(tl.int64)
        rr = tl.arange(0, RM)
        sl = tl.load(eslot + e * RM + rr)
        rm = sl >= 0
        n = nb * BN + tl.arange(0, BN)
        wg = w_ptr + e * lwe + (2 * n) * lwr
        sg = s_ptr + e * lse + (2 * n) * lsr
        gs = 1.
        if FP4:
            gs = tl.load(gs_ptr + e)
        ag = tl.zeros([RM, BN], tl.float32)
        au = tl.zeros([RM, BN], tl.float32)
        for k0 in range(0, H, BK):
            ag = _dot_tile(ag, U, rr, rm, H, wg, sg, k0, gs, FP4, BK, F32)
            au = _dot_tile(au, U, rr, rm, H, wg + lwr, sg + lsr, k0, gs, FP4, BK, F32)
        g = _r(ag, F32)
        v = _r(au, F32)
        a = 4 * _tanh(g / 4) * tl.sigmoid(g) * 25 * _tanh(v / 25)
        tl.store(act + (rr * 8 + tl.where(rm, sl, 0))[:, None] * NI + n[None, :], _bf16(a, F32), mask=rm[:, None])
        if PF:   # this expert's w2 + scales (a share per n block; after the loop: no registers held across it)
            _pf_part(pfa.to(tl.pointer_type(tl.uint8)) + e * pfn, pfn, nb, tl.num_programs(0))
            _pf_part(pfb.to(tl.pointer_type(tl.uint8)) + e * pfm, pfm, nb, tl.num_programs(0))


@triton.jit
def _w2u_kernel(act, ulist, ucnt, eslot, wts, w_ptr, s_ptr, gs_ptr, lwe, lwr, lse, lsr, part,
                FP4: tl.constexpr, F32: tl.constexpr, H: tl.constexpr, NI: tl.constexpr, BN: tl.constexpr,
                BK: tl.constexpr, RM: tl.constexpr, UNR: tl.constexpr = 0, PDL: tl.constexpr = False, PF: tl.constexpr = False, pfa=None, pfn=0):
    """Program (row block, union entry u): part[r, slot, rows] = bf16(act[r, slot] @ w2[e]^T) * weight[r, slot]."""
    _enter(PDL)
    nb, u = tl.program_id(0), tl.program_id(1)
    if PF:   # the next visit's projection (shares over the row blocks of union entries 0 .. 7)
        if u < 8:
            _pf_part(pfa, pfn, u * tl.num_programs(0) + nb, 8 * tl.num_programs(0))
    if u < tl.load(ucnt):
        e = tl.load(ulist + u).to(tl.int64)
        rr = tl.arange(0, RM)
        sl = tl.load(eslot + e * RM + rr)
        rm = sl >= 0
        xr = rr * 8 + tl.where(rm, sl, 0)
        n = nb * BN + tl.arange(0, BN)
        gs = 1.
        if FP4:
            gs = tl.load(gs_ptr + e)
        acc = tl.zeros([RM, BN], tl.float32)
        for k0 in range(0, NI, BK):
            acc = _dot_tile(acc, act, xr, rm, NI, w_ptr + e * lwe + n * lwr, s_ptr + e * lse + n * lsr, k0, gs, FP4,
                            BK, F32)
        wt = tl.load(wts + xr, mask=rm, other=0.)
        tl.store(part + xr[:, None] * H + n[None, :], _r(acc, F32) * wt[:, None], mask=rm[:, None])


@triton.jit
def _bw2_fin_kernel(part, cst, sst, xh, cw, r_ptr, h_out, meta, nr, F32: tl.constexpr, H: tl.constexpr,
                    BN: tl.constexpr, RM: tl.constexpr, CC: tl.constexpr, UNR: tl.constexpr = 0, PDL: tl.constexpr = False, PF: tl.constexpr = False):
    """Per row: y = bf16(sum of its 8 parts in slot order) -> xh; mlp short conv; h = bf16(r + conv)."""
    _enter(PDL)
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    rr = tl.arange(0, RM)
    rv = rr < nr
    m2 = rv[:, None] & (n >= 0)[None, :]
    tot = tl.zeros([RM, BN], tl.float32)
    for t in tl.range(8, loop_unroll_factor=UNR * 8):
        tot += tl.load(part + (rr[:, None] * 8 + t) * H + n[None, :], mask=m2, other=0.)
    y = _r(tot, F32)
    xo = xh + rr[:, None] * CC + 4352 + n[None, :]
    tl.store(xo, y.to(tl.bfloat16), mask=m2)
    tl.debug_barrier()
    sl = tl.load(meta + rr, mask=rv, other=0).to(tl.int32)
    k = tl.load(meta + 2 * RM + rr, mask=rv, other=0).to(tl.int32)[:, None]
    st = cst + sl[:, None] * sst + 4352 + n[None, :]
    y = _cv4(y, _prev(xo, CC, st, k, 1, CC, m2), _prev(xo, CC, st, k, 2, CC, m2), _prev(xo, CC, st, k, 3, CC, m2),
             tl.load(cw + n * 4)[None, :], tl.load(cw + n * 4 + 1)[None, :], tl.load(cw + n * 4 + 2)[None, :],
             tl.load(cw + n * 4 + 3)[None, :], 2.23606797749979)
    y = _r(y, F32)
    tl.store(h_out + rr[:, None] * H + n[None, :], _bf16(tl.load(r_ptr + rr[:, None] * H + n[None, :], mask=m2,
                                                                   other=0.) + y, F32), mask=m2)


@triton.jit
def _commit_kernel(conv, sv, sst, xh, meta, crow0, cn, RM: tl.constexpr, CC: tl.constexpr, CB: tl.constexpr, UNR: tl.constexpr = 0, PDL: tl.constexpr = False, PF: tl.constexpr = False):
    """Program (visit, group, channel block): a group = cn[g] accepted rows starting at row crow0[g] (0: no-op); the
    slot's new conv state = the last 3 of [state ; its rows' inputs in xh]."""
    _enter(PDL)
    v, gi, cb = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    n = tl.load(cn + gi).to(tl.int32)
    if n > 0:
        r0 = tl.load(crow0 + gi).to(tl.int32)
        st = conv + v * sv + tl.load(meta + r0).to(tl.int32) * sst + cb * CB + tl.arange(0, CB)
        x = xh + v * (RM * CC) + cb * CB + tl.arange(0, CB)
        nv = n + tl.zeros([CB], tl.int32)            # vector masks (the interpreter rejects scalar ones)
        a0 = tl.where(nv < 3, tl.load(st + nv * CC, mask=nv < 3, other=0.), tl.load(x + (r0 + nv - 3) * CC, mask=nv >= 3,
                                                                                   other=0.))
        a1 = tl.where(nv < 2, tl.load(st + (nv + 1) * CC, mask=nv < 2, other=0.),
                      tl.load(x + (r0 + nv - 2) * CC, mask=nv >= 2, other=0.))
        a2 = tl.load(x + (r0 + nv - 1) * CC)
        tl.store(st, a0)
        tl.store(st + CC, a1)
        tl.store(st + 2 * CC, a2)


# --------------------------------------------------------------------------------------------- host side
class BatchDecoder:
    """Slots: fixed-address per-sequence decode state (rings [48, 2, slots, WR, 1024] bf16, rpos, conv states, page
    table rows), allocated at the first load (drop() frees them). ~214 MB per slot (rings) + ~2 MB. Lazily per slot
    count: the first load allocates slots 0..slot (na; a lone decode, e.g. at 1M after a solo prefill, holds 1 slot's
    rings, not 4), a load of a slot >= na grows the buffers to all n slots (live slots' state copied, graphs recaptured)."""
    dec = True   # (engine: slot state lives in the batch decoder; drop() frees it when idle)

    def __init__(self, model, slots=4, graph=None):
        self.m, self.n = model, slots
        dev = model.dev
        if graph is None:
            graph = os.environ.get('INK_GRAPH') == '1'
        self.graph = graph and dev.type == 'cuda'
        self.split = int(os.environ.get('INK_DEC_FAR_BLK', 4096))    # far keys per program (as Fused)
        self.seqs = [None] * slots
        self.pend = {}                                                  # slot -> (row0, rows): verify, uncommitted
        self.b, self.na = None, 0                                       # static buffers (lazy, drop() frees), slots in them
        self.graphs, self.pool, self.side = {}, None, None
        # per-visit weight pointers / plan: decode_kernels.Fused's setup (its own small buffers stay unused)
        self.F = Fused(model, SimpleNamespace(kv=model.pool.new_table()), None, None)
        # INK_DEC_EMBED_GPU=1: embedding table on the device, row lookup inside the step (step(rows, tok=...) then needs
        # no host sync); dev_tok tells spec_runner.Spec that device draft ids are worth passing
        self.embed = model.W.embed.to(dev) if os.environ.get('INK_DEC_EMBED_GPU') == '1' else None
        self.dev_tok = self.embed is not None
        self.hm, self.hev, self.hi = None, [None, None], 0              # pinned host meta (+ tokens) x 2, their copies
        # h entering visit k = the output of visit src (none of TAPS is a loop restart: src = k - 1)
        self.tap_src = [self.F.vis[k]['src'] for k in TAPS]
        assert self.tap_src == [k - 1 for k in TAPS], self.tap_src

    # ------------------------------------------------------------------ slots
    def busy(self):
        return any(s is not None for s in self.seqs)

    def idle(self):
        return self.na if self.b is not None and not self.busy() else 0

    @property
    def decs(self):  # LoopBatch compatibility: a slot "has a decoder" while the static buffers hold it
        return [self if self.b is not None and i < self.na else None for i in range(self.n)]

    def new_slots(self, slot):
        """Slots whose buffers load(slot) allocates (engine: VRAM ask before the load): 0 if slot has them."""
        return 0 if self.b is not None and slot < self.na else slot + 1 if self.b is None else self.n

    @property
    def st(self):  # tests/gpu_mem.py: bytes of the static decode state
        return None if self.b is None else SimpleNamespace(conv=[{'conv': self.b.conv}], ring=[(self.b.ring,)])

    def nbytes(self):
        """(bytes per slot, shared bytes) of the static buffers (0, 0 before the first load)."""
        if self.b is None:
            return 0, 0
        per = sum(x.numel() * x.element_size() for x in (self.b.ring, self.b.rpos, self.b.conv, self.b.tbl)) // self.na
        tot = sum(x.numel() * x.element_size() for x in vars(self.b).values() if isinstance(x, torch.Tensor))
        return per, tot - per * self.na

    def _alloc(self, n):
        m, dev = self.m, self.m.dev
        f32 = lambda *s: torch.zeros(*s, dtype=torch.float32, device=dev)
        bf = lambda *s: torch.zeros(*s, dtype=torch.bfloat16, device=dev)
        i32 = lambda *s: torch.zeros(*s, dtype=torch.int32, device=dev)
        b = SimpleNamespace()
        b.ring = bf(len(VISITS), 2, n, WR, NKV * HD)
        b.rpos = torch.full((n, WR), -1, dtype=torch.int64, device=dev)
        b.conv = bf(len(VISITS), n, 3, CC)
        b.tbl = torch.zeros(n, m.pool.n, dtype=torch.int64, device=dev)
        # rows: 0 slot, 1 pos, 2 index in its slot group, 3 live, 4/5 far group first row / rows, 6/7 in-step commit
        # group first row / accepted rows; emeta: commit() groups
        b.meta, b.emeta = torch.zeros(8, RM, dtype=torch.int64, device=dev), torch.zeros(2, RM, dtype=torch.int64,
                                                                                         device=dev)
        b.x_raw, b.h0, b.B, b.hbuf = bf(RM, H), bf(RM, H), f32(RM, 8, H), bf(len(VISITS), RM, H)
        b.tok = torch.zeros(RM, dtype=torch.int64, device=dev)          # row tokens (lookup in the step: self.embed)
        b.A, b.X, b.M, b.G, b.Z = f32(RM, 3, H), bf(RM, 3, H), f32(RM, 4, H), f32(RM, 3, H), f32(RM, 3, 64)
        b.Y, b.Qb, b.Rb, b.att, b.Rr = bf(RM, NY), bf(RM, NQ, HD), bf(RM, NQ, 16), bf(RM, NQ * HD), f32(RM, H)
        b.U, b.S, b.ids, b.wts = bf(RM, H), f32(RM, 128), i32(RM, 8), f32(RM, 8)
        b.eslot, b.ulist, b.ucnt = i32(128, RM), i32(128), i32(1)
        b.act, b.part = bf(RM, 8, NI), f32(RM, 8, H)
        self.ns = self.F.nsp or -(-((m.pool.n - 1) * kvpool.PAGE) // self.split)   # INK_DEC_FAR2: fixed splits
        nt = WR // BB['near']
        b.fo, b.fl = f32(RM, self.ns, NQ, HD), f32(RM, self.ns, NQ)
        b.pm, b.pl, b.pacc, b.plse = f32(RM, nt, NQ), f32(RM, nt, NQ), f32(RM, nt, NQ, HD), f32(RM, NQ)
        b.xh = bf(len(VISITS), RM, CC)
        b.out = f32(RM, VOCAB)
        b.taps, b.tap_idx = bf(4, RM, H), torch.tensor(self.tap_src, device=dev)
        b.thist = bf(4, n * TH + 1, H)                                 # slot s, pos p at s * TH + p % TH; last = trash
        self.b, self.na = b, n

    def _grow(self):
        """Buffers for all n slots; the live slots' state (rings, positions, conv, page tables, tap history) is copied
        over, the graphs (old addresses) are dropped and recaptured on first use."""
        assert not self.pend, 'commit() first'
        o, na = self.b, self.na
        self.b, self.graphs, self.pool, self.side = None, {}, None, None
        self._alloc(self.n)
        b = self.b
        b.ring[:, :, :na].copy_(o.ring)
        b.rpos[:na].copy_(o.rpos)
        b.conv[:, :na].copy_(o.conv)
        b.tbl[:na].copy_(o.tbl)
        b.thist[:, :na * TH].copy_(o.thist[:, :na * TH])
        del o
        if self.m.dev.type == 'cuda':
            torch.cuda.empty_cache()

    @torch.no_grad()
    def load(self, slot, seq):
        """Copy a prefilled Seq into a slot (page table, conv states, rings: time order -> slot pos % WR); the Seq's
        own prefill rings are let go (seq.forward must not run on it afterwards)."""
        assert self.seqs[slot] is None, f'slot {slot} in use'
        assert hasattr(seq.kv, 't'), 'BatchDecoder needs a pool page table'
        if self.b is None:
            self._alloc(slot + 1)
        elif slot >= self.na:
            self._grow()
        b, p = self.b, seq.pos
        b.tbl[slot].copy_(seq.kv.t)
        for v, c in enumerate(seq.conv):
            for key, off, C in CONV:
                b.conv[v, slot, :, off:off + C].copy_(c[key])
        n = max(k.shape[0] for k, _ in seq.ring)                      # min(pos, 1023)
        b.rpos[slot].fill_(-1)
        if n:
            pos = torch.arange(p - n, p, device=b.rpos.device)
            sl = pos % WR
            b.rpos[slot].index_copy_(0, sl, pos)
            for v, (k, x) in enumerate(seq.ring):
                if k.shape[0]:                                         # (unit tests fill only some visits)
                    b.ring[v, 0, slot].index_copy_(0, sl, k.reshape(n, -1))
                    b.ring[v, 1, slot].index_copy_(0, sl, x.reshape(n, -1))
        e = seq.ring[0][0].new_zeros(0, *seq.ring[0][0].shape[1:])
        seq.ring = [(e, e)] * len(seq.ring)
        self.seqs[slot] = seq

    def free(self, slot):
        self.seqs[slot] = None
        self.pend.pop(slot, None)

    def drop(self, keep=0):
        """Free the static buffers and graphs when no slot is in use and keep == 0 -> number of slots freed (they come
        back at the next load)."""
        if self.b is None or keep or self.busy():
            return 0
        n, self.na = self.na, 0
        self.b, self.graphs, self.pool, self.side = None, {}, None, None
        if self.m.dev.type == 'cuda':
            torch.cuda.empty_cache()
        return n

    @torch.no_grad()
    def state(self, slot):
        """Clone of the slot's small state at its pos (Engine.state_of's format without 'f')."""
        assert slot not in self.pend, 'commit() first'
        b, p = self.b, self.seqs[slot].pos
        n = min(p, WIN - 1)
        idx = torch.arange(p - n, p, device=b.rpos.device) % WR
        st = {}
        for v in range(len(VISITS)):
            for key, off, C in CONV:
                st[f'conv.{v}.{key}'] = b.conv[v, slot, :, off:off + C].clone()
            st[f'ring.{v}.k'] = b.ring[v, 0, slot][idx].view(n, NKV, HD)
            st[f'ring.{v}.v'] = b.ring[v, 1, slot][idx].view(n, NKV, HD)
        return st

    # ------------------------------------------------------------------ steps
    def _host_meta(self):
        """A zeroed [9, RM] int64 host buffer (rows 0-7 meta, 8 tokens), pinned on cuda; two alternate, and a buffer
        is reused only after its previous non-blocking copy finished."""
        if self.hm is None:
            cuda = self.m.dev.type == 'cuda'
            self.hm = [torch.zeros(9, RM, dtype=torch.int64, pin_memory=cuda) for _ in range(2)]
        self.hi ^= 1
        if self.hev[self.hi] is not None:
            self.hev[self.hi].synchronize()
        h = self.hm[self.hi]
        h.zero_()
        return h

    @torch.no_grad()
    def step(self, rows, tok=None):
        """rows [(slot, pos, token)], a slot's rows consecutive with pos = seq.pos, seq.pos + 1, ...; -> logits [R, VOCAB]
        fp32 (static buffer). Single-row slots advance by one now; a multi-row (verify) slot waits for commit().
        tok: {row0: int64 device tensor [n]} = the tokens of rows row0 .. row0+n-1 (their token entries are ignored)."""
        assert 0 < len(rows) <= RM, len(rows)
        assert not self.pend, f'commit() the verify rows of slots {sorted(self.pend)} first'
        b, R = self.b, next(x for x in BUCKETS if x >= len(rows))
        hm = self._host_meta()
        meta = hm[:8]
        groups, seen, nf, nc = [], set(), 0, 0
        for i, (slot, pos, _) in enumerate(rows):
            if i and slot == rows[i - 1][0]:
                assert pos == rows[i - 1][1] + 1, f'slot {slot}: rows must have consecutive positions'
                groups[-1][2] += 1
            else:
                seq = self.seqs[slot]
                assert seq is not None and slot not in seen and pos == seq.pos, (slot, pos, seq and seq.pos)
                seen.add(slot)
                groups.append([slot, i, 1])
            meta[0, i], meta[1, i], meta[2, i], meta[3, i] = slot, pos, i - groups[-1][1], 1
        for slot, r0, T in groups:
            kv, last = self.seqs[slot].kv, rows[r0 + T - 1][1]
            assert last < kv.cap, f'context full ({kv.cap} tokens)'
            if last >= len(kv.pages) * kvpool.PAGE:                    # grow (the engine reserves up front)
                assert kv.pool.reserve(kv, last + 1), 'KV pool full'
                b.tbl[slot].copy_(kv.t)
            for a, n in far_groups([p for _, p, _ in rows[r0:r0 + T]], self.F.nsp, mt(R) // 3):
                meta[4, nf], meta[5, nf] = r0 + a, n
                nf += 1
            if T == 1:
                meta[6, nc], meta[7, nc] = r0, 1
                nc += 1
        meta[0, len(rows):] = rows[0][0]                               # pad rows: live 0, a valid slot index
        tok = tok or {}
        dev_rows = {r0 + j for r0, t in tok.items() for j in range(len(t))}
        for i, (_, _, t) in enumerate(rows):
            if i not in dev_rows:
                hm[8, i] = t
        b.meta.copy_(meta, non_blocking=True)
        if self.embed is not None:                                     # lookup inside the step (_run)
            b.tok.copy_(hm[8], non_blocking=True)
            for r0, t in tok.items():
                b.tok[r0:r0 + len(t)].copy_(t)
        else:
            ids = hm[8, :len(rows)].clone()
            for r0, t in tok.items():
                ids[r0:r0 + len(t)] = t.cpu()                          # (a sync: no device table)
            b.x_raw[:len(rows)].copy_(self.m.W.embed[ids])
        if self.m.dev.type == 'cuda':
            self.hev[self.hi] = torch.cuda.Event()
            self.hev[self.hi].record()
        self._replay(R)
        for slot, r0, T in groups:
            if T == 1:
                self.seqs[slot].pos += 1
            else:
                self.pend[slot] = (r0, T)
        return b.out[:len(rows)]

    @torch.no_grad()
    def commit(self, acc=None):
        """acc {slot: accepted rows} for the last step's verify slots (default: all rows); single-row slots were
        committed inside step() (n must be 1 for them). New conv states = last 3 of [state ; accepted inputs]; rings
        and pages need no rollback (rejected positions are masked until rewritten)."""
        acc = {s: T for s, (_, T) in self.pend.items()} if acc is None else acc
        em = torch.zeros(2, RM, dtype=torch.int64)
        ng = 0
        for slot, n in acc.items():
            if slot in self.pend:
                r0, T = self.pend.pop(slot)
                assert 1 <= n <= T, (slot, n, T)
                em[0, ng], em[1, ng] = r0, n
                ng += 1
                self.seqs[slot].pos += n
            else:
                assert n == 1, f'slot {slot}: no verify rows pending'
        if ng:
            b = self.b
            b.emeta.copy_(em)
            _commit_kernel[(len(VISITS), ng, CC // BB['cb'])](b.conv, b.conv.stride(0), 3 * CC, b.xh, b.meta,
                                                              b.emeta[0], b.emeta[1], RM=RM, CC=CC, CB=BB['cb'])

    def tap_rows(self, r0, n):
        """DSpark taps of the last step's rows r0 .. r0+n-1 -> [4, n, H] bf16 (view; the next step overwrites it)."""
        return self.b.taps[:, r0:r0 + n]

    def tap_hist(self, slot, p0, n):
        """Taps of the slot's committed positions p0 .. p0+n-1 -> [4, n, H] bf16 (view, or a copy across the ring
        wrap). Every step writes its rows' taps at their positions (rejected verify rows too: their positions are
        rewritten before they are committed), so a position stays readable until the slot is TH - 8 past it."""
        pos = self.seqs[slot].pos
        assert slot not in self.pend and p0 + n <= pos and pos - p0 <= TH - 8, (slot, p0, n, pos)
        h, i = self.b.thist[:, slot * TH:(slot + 1) * TH], p0 % TH
        return h[:, i:i + n] if i + n <= TH else torch.cat((h[:, i:], h[:, :i + n - TH]), 1)

    def _replay(self, R):
        if not self.graph:
            return self._run(R)
        g = self.graphs.get(R)
        if g is not None:
            return g.replay()
        # first step of this bucket: run it eagerly on a side stream (this IS the step; compiles Triton, initializes
        # cuBLAS), then record the same work (recording executes nothing). One pool for all buckets (never concurrent).
        cur = torch.cuda.current_stream()
        if self.pool is None:
            self.side, self.pool = torch.cuda.Stream(), torch.cuda.graph_pool_handle()
        self.side.wait_stream(cur)
        with torch.cuda.stream(self.side):
            self._run(R)
        cur.wait_stream(self.side)
        cur.synchronize()
        # thread_local: the kvpool writer thread's event sync during a global-mode capture invalidated it (prod 11-23)
        g = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(g, pool=self.pool, stream=self.side, capture_error_mode='thread_local'):
                self._run(R)
        except Exception as e:  # the eager run above was this step: keep its output, capture again next time
            print(f'batch decode capture failed (R={R}, retried next step): {type(e).__name__}: {e}'[:300], flush=True)
            return
        self.graphs[R] = g

    def _run(self, R):
        """The whole step for R rows (pad rows: live 0). Kernel grids depend on R only."""
        b, m, F, Fz = self.b, self.m, INTERP, self.F
        o, lk = Fz.o, Fz.lk
        if self.embed is not None:
            torch.index_select(self.embed, 0, b.tok, out=b.x_raw)
        b.h0.copy_(rms(b.x_raw, m.W['embed_norm.weight']))
        for v, d in enumerate(Fz.vis):
            self._visit(v, d, R)
        _bmix_kernel[(1, R)](Fz.final_q, b.B, 8 * H, Fz.final_ent, 4, b.hbuf[len(VISITS) - 1], H, -1, b.A, 3 * H, b.X,
                             3 * H, Fz.final_nwt, Fz.final_q, Fz.final_ent, 1, b.h0, H, b.M, 4 * H, 1,
                             FIRST=False, NORM=True, STORE_A=False, F32=F, H=H, CH=CH, **o, **lk)
        _commit_kernel[(len(VISITS), R, CC // BB['cb'])](b.conv, b.conv.stride(0), 3 * CC, b.xh, b.meta, b.meta[6],
                                                         b.meta[7], RM=RM, CC=CC, CB=BB['cb'], **o, **lk)
        b.out.copy_(m.logits(b.X[:, 0], Fz.w8))                       # all RM rows: one shape for every R
        torch.index_select(b.hbuf, 0, b.tap_idx, out=b.taps[:3])       # drafter taps (2 small copies, in the graph)
        b.taps[3].copy_(b.X[:, 0])
        mt = b.meta                                                    # tap history: live rows at slot * TH + pos % TH
        b.thist.index_copy_(1, torch.where(mt[3] > 0, mt[0] * TH + mt[1] % TH, self.na * TH), b.taps)

    def _visit(self, v, d, R):
        b, F, meta, Fz = self.b, INTERP, self.b.meta, self.F
        o, lk = Fz.o, Fz.lk
        far_kw = {} if INTERP or not Fz.nsp else D.FAR_KW
        hin = b.h0 if d['src'] < 0 else b.hbuf[d['src']]
        full, msg = d['full'], d['msg']
        nq = 3 if full else 1
        xs = H if full else 0
        cst, xh, sst = b.conv[v], b.xh[v], 3 * CC
        if not msg:
            _bmix_kernel[(nq, R)](d['q'], b.B, 8 * H, d['ent'], d['n'], hin, H, d['new'], b.A, 3 * H, b.X, 3 * H,
                                  d['nwt'], d['q'], d['ent'], 1, hin, H, b.M, 4 * H, nq,
                                  FIRST=d['L'] == 0, NORM=True, STORE_A=False, F32=F, H=H, CH=CH, **o, **lk)
        else:
            _bmix_kernel[(nq + 4, R)](d['q'], b.B, 8 * H, d['ent'], d['n'], hin, H, d['new'], b.A, 3 * H, b.X, 3 * H,
                                      d['nwt'], d['mqs'], d['pslot'], 1, b.hbuf[d['pv']], H, b.M, 4 * H, nq,
                                      FIRST=False, NORM=False, STORE_A=True, F32=F, H=H, CH=CH, **o, **lk)
            _bgate_z_kernel[(3, 64 // BLK['gate'], R)](b.A, 3 * H, xs, b.M, 4 * H, d['dtab'], b.Z, 192, F32=F, H=H,
                                                       CH=CH, ZB=BLK['gate'], num_warps=NW['gate'], **o, **lk)
            _bgate_out_kernel[(3, H // BLK['gout'], R)](b.A, 3 * H, xs, b.M, 4 * H, d['gtab'], b.Z, 192, b.G, 3 * H,
                                                        H=H, BH=BLK['gout'], num_warps=NW['gout'], **o, **lk)
            _bnorm_kernel[(3, R)](b.G, d['nwt'], b.X, F32=F, H=H, CH=CH, **o, **lk)
            xs = H
        N, w8 = d['N'], self.F.w8
        _bproj_kernel[(triton.cdiv(N, BB['proj']),)](d['cat'], d['n0'], d['n1'], N, b.X, xs, 3 * H, b.Y, R, d['cs'], F32=F,
                                                     H=H, CH=CH, BN=BB['proj'], RM=RM, num_warps=NWB['proj'], W8=w8, **o, **lk)
        rk, rv = b.ring[v, 0], b.ring[v, 1]
        c0, c1, s0, s1, _ = d['kv'] if full else (b.Y,) * 5
        _bpre_kernel[(NKV, R)](b.Y, d['koff'], d['voff'], d['tkoff'], d['tvoff'], d['qsc'], d['rsc'], d['kw'], d['vw'],
                               cst, sst, xh, rk, rv, b.rpos, meta, c0, c1, s0, s1, b.tbl, b.tbl.shape[1], b.Qb, b.Rb,
                               FULL=full, F32=F, PG=kvpool.PAGE, WR=WR, RM=RM, CC=CC, **o, **lk)
        if full:
            _bfar_kernel[(NKV, self.ns, R)](b.Qb, c0, c1, s0.view(torch.float8_e4m3fn), s1.view(torch.float8_e4m3fn),
                                            b.tbl, b.tbl.shape[1], meta, b.fo, b.fl, self.ns, HD ** -0.5,
                                            SPLIT=self.split, PG=kvpool.PAGE, BLK=DOT_BLK, F32=F, RM=RM, NSP=Fz.nsp, MT=mt(R),
                                            **far_kw, **o, **lk)
        bl = BB['near']
        nt = WR // bl
        _bnear_part_kernel[(R, NKV, nt)](b.Qb, b.Rb, d['proj'], rk, b.rpos, meta, b.pm, b.pl, F32=F, BLK=bl, WR=WR,
                                         RM=RM, **d['pf']['near'], **o, **lk)
        _bnear_out_kernel[(R, NKV, nt)](b.Qb, b.Rb, d['proj'], b.Y, rk, rv, b.rpos, meta, b.pm, b.pl, b.fl, self.ns,
                                        b.pacc, b.plse, FAR=full, F32=F, BLK=bl, WR=WR, RM=RM,
                                        NTP=triton.next_power_of_2(nt), NSF=Fz.nsf, **o, **lk)
        _bnear_fin_kernel[(NQ, R)](b.pacc, b.plse, b.fo, b.fl, self.ns, b.Y, b.att, meta, FAR=full, F32=F, NT=nt, RM=RM, NSF=Fz.nsf, **o, **lk)
        _bwo_kernel[(H // BB['wo'],)](b.att, d['wo'], cst, sst, xh, d['acw'], hin, b.Rr, meta, R, d['wos'], BND=d['bnd'],
                                      F32=F, H=H, CH=CH, BN=BB['wo'], RM=RM, CC=CC, num_warps=NWB['wo'], W8=w8, **d['pf']['wo'], **o, **lk)
        if not msg:
            _bmix_kernel[(1, R)](d['mq'], b.B, 8 * H, d['ment'], d['mn'], b.Rr, H, -1, b.A, 3 * H, b.U, H, d['mnwt'],
                                 d['mq'], d['ment'], 1, b.Rr, H, b.M, 4 * H, 1,
                                 FIRST=False, NORM=True, STORE_A=False, F32=F, H=H, CH=CH, **o, **lk)
            _brouter_kernel[(128 // BLK['router'], R)](b.A, 3 * H, d['mnw'], b.U, d['gw'], b.S, NORM=False, F32=F, H=H,
                                                       CH=CH, RB=BLK['router'], num_warps=NW['router'], **o, **lk)
        else:
            _bmix_kernel[(1, R)](d['mq'], b.B, 8 * H, d['ment'], d['mn'], b.Rr, H, -1, b.A, 3 * H, b.U, H, d['mnwt'],
                                 d['mq'], d['ment'], 1, b.Rr, H, b.M, 4 * H, 1,
                                 FIRST=False, NORM=False, STORE_A=True, F32=F, H=H, CH=CH, **o, **lk)
            _bgate_z_kernel[(1, 64 // BLK['gate'], R)](b.A, 3 * H, 0, b.M[:, 3], 4 * H, d['dtab_m'], b.Z, 192, F32=F,
                                                       H=H, CH=CH, ZB=BLK['gate'], num_warps=NW['gate'], **o, **lk)
            _bgate_out_kernel[(1, H // BLK['gout'], R)](b.A, 3 * H, 0, b.M[:, 3], 4 * H, d['gtab_m'], b.Z, 192, b.G,
                                                        3 * H, H=H, BH=BLK['gout'], num_warps=NW['gout'], **o, **lk)
            _brouter_kernel[(128 // BLK['router'], R)](b.G, 3 * H, d['mnw'], b.U, d['gw'], b.S, NORM=True, F32=F, H=H,
                                                       CH=CH, RB=BLK['router'], num_warps=NW['router'], **o, **lk)
        _plan_kernel[(1,)](b.S, d['gb'], meta, R, b.ids, b.wts, b.eslot, b.ulist, b.ucnt, RM=RM, **o, **lk)
        w13, s13, w2, s2 = d['ex']
        ug = min(128, 8 * R)
        kw = W13_R1 if R == 1 else dict(BK=BB['kbk'], num_stages=NSB['w13'])
        _w13u_kernel[(NI // BB['w13'], ug)](b.U, b.ulist, b.ucnt, b.eslot, w13, s13, d['gs13'], w13.stride(0),
                                            w13.stride(1), s13.stride(0), s13.stride(1), b.act, FP4=d['fp4'], F32=F,
                                            H=H, NI=NI, BN=BB['w13'], RM=RM, num_warps=NWB['w13'], **kw, **d['pf']['w13'], **o, **lk)
        _w2u_kernel[(H // BB['w2'], ug)](b.act, b.ulist, b.ucnt, b.eslot, b.wts, w2, s2, d['gs2'], w2.stride(0),
                                         w2.stride(1), s2.stride(0), s2.stride(1), b.part, FP4=d['fp4'], F32=F, H=H,
                                         NI=NI, BN=BB['w2'], BK=BB['kbk2'], RM=RM, num_warps=NWB['w2'],
                                         num_stages=NSB['w2'], **d['pf']['w2'], **o, **lk)
        _bw2_fin_kernel[(H // BB['w2f'],)](b.part, cst, sst, xh, d['mcw'], b.Rr, b.hbuf[v], meta, R, F32=F, H=H,
                                           BN=BB['w2f'], RM=RM, CC=CC, **o, **lk)
