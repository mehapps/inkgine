"""Prefill attention kernels (INK_PREFILL_ATTN=triton: far + near; =far: far only; =near: near only (flex far);
=auto (server default on cuda): near + this far kernel only below INK_PA_FAR_MAX far keys (flex is faster beyond
~64k, see model.py); default torch = Model.far + the near loop in Model.attn_core, which stay as the reference).

Both kernels give a program BM consecutive rows of the flattened (token, q-head) order of one kv head, so the 3 query
heads of a kv head share every K/V tile (row r -> token r // 3, q-head 3 * kv_head + r % 3), with fp32 online softmax
(exp2) and fp32 accumulate on tensor cores.
- far(): keys [0, kend) at distance >= 1024 straight from the 4-bit store (codes + e4m3 scales through the page
  table, dequantized to bf16 in registers; exact, int4 x e4m3 fits bf16), causal far mask key <= query - 1024.
  Tiles where every row sees every key skip the mask. Optional split over key ranges (small chunks at long context)
  merged by LSE. Returns Model.far's (o [T,24,128] fp32 normalized, lse [T,24]; no keys: o = 0, lse = -inf).
- near(): keys at distance < 1024 from the bf16 ring + chunk (Kn/Vn), relative bias computed in-kernel: one
  [rows,16] x [16,band] product (R x the slice of rel_logits_proj the tile's distances cover) and a per-row gather,
  then the sink and the far (o, lse) merged by LSE. Same rounding points as attn_core (bf16 scores, bf16 * scale,
  bf16 bias, bf16 add); the near probabilities are rounded to bf16 for P @ V like the torch path.
"""
import os

import torch

from model import HD, NKV, NQ, WIN

try:
    import triton
    import triton.language as tl
    from moe_triton import INTERP, _bf16, _mma
except ImportError:  # pragma: no cover
    triton = None

# tile configs (env for the GPU window sweep). Near defaults = the 2026-11-22 sweep's best (0.66 ms per sliding visit
# vs 1.1-1.2 for the others). Far defaults are what was timed against flex at 4k-1M; the sweep's best at 1M
# (BM 64, BN 64, 4 warps, progs 2720: 344 vs 363 ms) is untimed at short context, where auto uses this kernel.
BM = int(os.environ.get('INK_PA_BM', 128))          # far: rows (token, head) per program
BN = int(os.environ.get('INK_PA_BN', 64))           # far: keys per tile (divides kvpool.PAGE)
WARPS = int(os.environ.get('INK_PA_WARPS', 8))
STAGES = int(os.environ.get('INK_PA_STAGES', 2))
NBM = int(os.environ.get('INK_PA_NEAR_BM', 64))     # near: rows per program
NBN = int(os.environ.get('INK_PA_NEAR_BN', 32))     # near: keys per tile
NWARPS = int(os.environ.get('INK_PA_NEAR_WARPS', 4))
PROGS = int(os.environ.get('INK_PA_PROGS', 1360))
FP8 = os.environ.get('INK_PA_FP8', '0') == '1'      # far: e4m3 P @ V (QK^T stays bf16); numerics change   # far: split keys until the grid has about this many programs
# INK_FARSKIP=<delta nats> (unset = off): far kernel skips a BN-key tile when, for every row of the program, a bound
# on the tile's max score is below the row's running max - delta (each skipped key then carries < e^-delta of the
# running max's weight; the dropped mass of a row is < n_skipped_keys * e^-delta of its far sum). The bound is the
# box bound sum_d max(q_d kmax_d, q_d kmin_d) over per-tile, per-head key min / max of the dequantized 4-bit keys
# (tile_meta, recomputed per call: ~0.1 ms per full visit at 262k vs ~80 ms of attention). 0 = exact mode: skip only
# tiles whose every probability underflows to exactly 0 in fp32 (bound < max - 160 in log2 units): bit-identical.
# PARKED (GPU 2026-11-23, gpu_farskip on 64k-196k source code): the box bound never skips (0.0% at every delta), even
# the exact tile max (oracle) allows only ~5-7% at delta 16 (~30% at delta 12, with ~2e-3 mass lost), and the bound
# costs the kernel 0.62-0.65x speed. Far-attention mass is spread out; tile skipping can't pay here.
_fs = os.environ.get('INK_FARSKIP', '')
SKIP = [float(_fs) if _fs else None]
SKIP_STATS = [None]                                 # tests / gpu_farskip: int64 [2] tensor += (skipped, visited) tiles
PG = 256                                            # kvpool.PAGE (kvpool imports model, so not imported here)
LOG2E, LN2 = 1.4426950408889634, 0.6931471805599453

if triton is not None:
    @triton.jit
    def _deq(c_ptr, s_ptr, co, so, km, BN: tl.constexpr, MASK: tl.constexpr, FP8: tl.constexpr = False):
        """[BN,8 groups,8 bytes] codes + [BN,8] e4m3 scales -> bf16 [BN,128] (low nibble = even element; exact);
        FP8: e4m3 instead (one rounding of the exact fp32 value)."""
        if MASK:
            c = tl.load(c_ptr + co, mask=km[:, :, None], other=0)
            sc = tl.load(s_ptr + so, mask=km, other=0.).to(tl.float32)[:, :, None]
        else:
            c, sc = tl.load(c_ptr + co), tl.load(s_ptr + so).to(tl.float32)[:, :, None]
        x = tl.join(((c & 15).to(tl.float32) - 8) * sc, ((c >> 4).to(tl.float32) - 8) * sc)
        return tl.reshape(x.to(tl.float8e4nv if FP8 else tl.bfloat16), (BN, 128))

    @triton.jit
    def _far_tile(q, m, l, acc, c0, c1, s0, s1, tbl, k0, h, lim, hi, scale,
                  MASK: tl.constexpr, PG: tl.constexpr, BN: tl.constexpr, F32: tl.constexpr, FP8: tl.constexpr):
        """One BN-key tile of the far loop (keys k0.., all in one page). MASK: apply k < lim (per row) and k < hi.
        FP8: P @ V in e4m3 (P scaled by 256; acc carries the 256, the caller divides). QK^T stays bf16: logits reach
        ~45 (l2-clipped q, k), so e4m3 q/k (~3% each) moved lse by ~1 nat in the CPU check."""
        b, gr, kr = tl.arange(0, 8), tl.arange(0, 8), tl.arange(0, BN)
        row = tl.load(tbl + k0 // PG) * PG + k0 % PG + kr          # row in the [pages * PG, 8, ...] store
        co = row[:, None, None] * 512 + h * 64 + gr[None, :, None] * 8 + b[None, None, :]   # [BN, group, byte]
        so = row[:, None] * 64 + h * 8 + gr[None, :]                                        # [BN, group]
        km = (k0 + kr < hi)[:, None]
        k = _deq(c0, s0, co, so, km, BN, MASK)
        s = _mma(q, tl.trans(k), tl.zeros([q.shape[0], BN], tl.float32), F32) * scale   # log2 units
        if MASK:
            s = tl.where((k0 + kr)[None, :] < tl.minimum(lim, hi)[:, None], s, float('-inf'))
            mn = tl.maximum(m, tl.max(s, 1))
            mu = tl.where(mn == float('-inf'), 0., mn)              # rows with no key yet
        else:
            mn = tl.maximum(m, tl.max(s, 1))
            mu = mn
        a = tl.exp2(m - mu)
        p = tl.exp2(s - mu[:, None])
        l = l * a + tl.sum(p, 1)
        if FP8:
            acc = _mma((p * 256.).to(tl.float8e4nv), _deq(c1, s1, co, so, km, BN, MASK, True), acc * a[:, None], F32)
        else:
            acc = _mma(_bf16(p, F32), _deq(c1, s1, co, so, km, BN, MASK), acc * a[:, None], F32)  # V after the softmax
        return mn, l, acc

    @triton.jit
    def _meta_kernel(c_ptr, s_ptr, tbl, meta_ptr, PG: tl.constexpr, BN: tl.constexpr):
        """Program (tile, kv head): per-dimension max / min of the tile's BN dequantized keys -> meta[tile, h, 0/1, :]
        bf16 (exact: the values are bf16-exact)."""
        i, h = tl.program_id(0), tl.program_id(1)
        b, gr, kr = tl.arange(0, 8), tl.arange(0, 8), tl.arange(0, BN)
        k0 = i * BN
        row = tl.load(tbl + k0 // PG) * PG + k0 % PG + kr
        co = row[:, None, None] * 512 + h * 64 + gr[None, :, None] * 8 + b[None, None, :]
        so = row[:, None] * 64 + h * 8 + gr[None, :]
        k = _deq(c_ptr, s_ptr, co, so, so, BN, False).to(tl.float32)
        d = tl.arange(0, 128)
        o = meta_ptr + (i * 8 + h) * 256 + d
        tl.store(o, tl.max(k, 0).to(tl.bfloat16))
        tl.store(o + 128, tl.min(k, 0).to(tl.bfloat16))

    @triton.jit
    def _tile_go(q, m, meta_ptr, k0, h, scale, thr, BN: tl.constexpr):
        """False iff the box bound sum_d max(q_d kmax_d, q_d kmin_d) of tile k0 is below m - thr (log2 units) for
        every row (m = -inf rows: True). (A [BM,128]x[128,16] tensor-core version per 16 tiles spilled more on sm_120.)"""
        d = tl.arange(0, 128)
        o = meta_ptr + ((k0 // BN) * 8 + h) * 256 + d
        kx, kn = tl.load(o).to(tl.float32), tl.load(o + 128).to(tl.float32)
        qf = q.to(tl.float32)
        ub = tl.sum(tl.maximum(qf * kx[None, :], qf * kn[None, :]), 1) * scale
        return tl.max(ub + 0.01 - m, 0) >= -thr      # 0.01: fp32 sum-order slack (|scores| < ~100)

    @triton.jit
    def _far_kernel(q_ptr, c0, c1, s0, s1, tbl, o_ptr, l_ptr, p0, T, kend, split, scale,
                    PG: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, F32: tl.constexpr, FP8: tl.constexpr = False,
                    meta_ptr=None, thr=0., st_ptr=None, SKIP: tl.constexpr = False, STATS: tl.constexpr = False):
        """Program (row tile mt, kv head h, split sp): keys [sp*split, (sp+1)*split) clipped to [0, kend) and to the
        tile's last far key; writes normalized partials o [splits,T,24,128], lse [splits,T,24] (natural log)."""
        mt, h, sp = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        r = mt * BM + tl.arange(0, BM)
        rm = r < 3 * T
        tok = tl.minimum(r, 3 * T - 1) // 3
        qh = h * 3 + r % 3
        t_lo = (mt * BM) // 3
        t_hi = (tl.minimum(mt * BM + BM, 3 * T) - 1) // 3
        lim = p0 + tok - (1024 - 1)                                  # key k is far for this row iff k < lim
        lo = sp * split
        hi = tl.minimum(tl.minimum(lo + split, kend), p0 + t_hi - 1023)
        full = tl.maximum(tl.minimum(hi, p0 + t_lo - 1023) - lo, 0) // BN * BN + lo   # all rows see all keys below
        d = tl.arange(0, 128)
        q = tl.load(q_ptr + (tok * 24 + qh)[:, None] * 128 + d[None, :])
        m = tl.full([BM], float('-inf'), tl.float32)
        l = tl.zeros([BM], tl.float32)
        acc = tl.zeros([BM, 128], tl.float32)
        nsk = tl.full([], 0, tl.int32)
        for k0 in range(lo, full, BN):
            if SKIP:
                go = _tile_go(q, m, meta_ptr, k0, h, scale, thr, BN)
                nsk += tl.where(go, 0, 1)
            else:
                go = True
            if go:
                m, l, acc = _far_tile(q, m, l, acc, c0, c1, s0, s1, tbl, k0, h, lim, hi, scale, False, PG, BN, F32, FP8)
        for k0 in range(full, hi, BN):
            if SKIP:
                go = _tile_go(q, m, meta_ptr, k0, h, scale, thr, BN)
                nsk += tl.where(go, 0, 1)
            else:
                go = True
            if go:
                m, l, acc = _far_tile(q, m, l, acc, c0, c1, s0, s1, tbl, k0, h, lim, hi, scale, True, PG, BN, F32, FP8)
        if STATS:
            tl.atomic_add(st_ptr, nsk)
            tl.atomic_add(st_ptr + 1, tl.maximum(hi - lo, 0) // BN + (tl.maximum(hi - lo, 0) % BN > 0))
        ok = l > 0
        if FP8:
            acc = acc * (1. / 256.)
        out = (sp * T + tok) * 24 + qh
        tl.store(o_ptr + out[:, None] * 128 + d[None, :], acc * tl.where(ok, 1 / tl.where(ok, l, 1.), 0.)[:, None],
                 mask=rm[:, None])
        tl.store(l_ptr + out, tl.where(ok, (m + tl.log2(tl.where(ok, l, 1.))) * 0.6931471805599453, float('-inf')),
                 mask=rm)

    @triton.jit
    def _near_kernel(q_ptr, k_ptr, v_ptr, r_ptr, pj_ptr, sink_ptr, fo_ptr, fl_ptr, o_ptr, p0, T, k0, scale, g_ptr, lg,
                     TAB: tl.constexpr, FAR: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BW: tl.constexpr,
                     F32: tl.constexpr, GATE: tl.constexpr = False):
        """Program (row tile mt, kv head h): near keys (distance 0..1023) of Kn/Vn (row j = position k0 + j).
        Bias: TAB=False -> r_ptr = R [T,24,16] bf16, pj_ptr = rel_logits_proj [16,1024]; TAB=True -> r_ptr = a
        precomputed table [T,24,1024] bf16 (tests). Writes o [T,24,128] fp32 merged with the sink (+ far)."""
        mt, h = tl.program_id(0), tl.program_id(1)
        r = mt * BM + tl.arange(0, BM)
        rm = r < 3 * T
        tok = tl.minimum(r, 3 * T - 1) // 3
        qh = h * 3 + r % 3
        row = tok * 24 + qh
        t_lo = (mt * BM) // 3
        t_hi = (tl.minimum(mt * BM + BM, 3 * T) - 1) // 3
        qpos = p0 + tok
        j_lo = tl.maximum(p0 + t_lo - 1023 - k0, 0)
        j_hi = p0 + t_hi - k0 + 1
        d, kr = tl.arange(0, 128), tl.arange(0, BN)
        q = tl.load(q_ptr + row[:, None] * 128 + d[None, :])
        if not TAB:
            e = tl.arange(0, 16)
            R = tl.load(r_ptr + row[:, None] * 16 + e[None, :])
            u = tl.arange(0, BW)
        m = tl.full([BM], float('-inf'), tl.float32)
        l = tl.zeros([BM], tl.float32)
        acc = tl.zeros([BM, 128], tl.float32)
        for j0 in range(j_lo, j_hi, BN):
            j = j0 + kr
            jm = j < j_hi
            kv = j[:, None] * 1024 + h * 128 + d[None, :]
            K = tl.load(k_ptr + kv, mask=jm[:, None], other=0.)
            s = _mma(q, tl.trans(K), tl.zeros([BM, BN], tl.float32), F32)
            s = _bf16(_bf16(s, F32).to(tl.float32) * scale, F32).to(tl.float32)  # torch: bf16 einsum, bf16 * scale
            dist = qpos[:, None] - (k0 + j)[None, :]
            ok = (dist >= 0) & (dist < 1024) & jm[None, :]
            if TAB:
                bias = tl.load(r_ptr + row[:, None] * 1024 + tl.minimum(tl.maximum(dist, 0), 1023), mask=ok, other=0.)
            else:
                dmin = p0 + t_lo - k0 - (j0 + BN - 1)                # smallest distance in this tile
                col = dmin + u
                P = tl.load(pj_ptr + e[:, None] * 1024 + col[None, :], mask=((col >= 0) & (col < 1024))[None, :],
                            other=0.)
                band = _bf16(_mma(R, P, tl.zeros([BM, BW], tl.float32), F32), F32)  # torch: bf16(R @ proj)
                bias = tl.gather(band, tl.minimum(tl.maximum(dist - dmin, 0), BW - 1), 1)
            s = _bf16(s + bias.to(tl.float32), F32).to(tl.float32)  # torch: bf16 add
            s = tl.where(ok, s * 1.4426950408889634, float('-inf'))
            mn = tl.maximum(m, tl.max(s, 1))
            mu = tl.where(mn == float('-inf'), 0., mn)
            a = tl.exp2(m - mu)
            p = tl.exp2(s - mu[:, None])
            l = l * a + tl.sum(p, 1)
            V = tl.load(v_ptr + kv, mask=jm[:, None], other=0.)
            acc = _mma(_bf16(p, F32), V, acc * a[:, None], F32)
            m = mn
        ln = (m + tl.log2(l)) * 0.6931471805599453                  # finite: every row sees itself
        sk = tl.load(sink_ptr + row)
        mx = tl.maximum(ln, sk)
        if FAR:
            lf = tl.load(fl_ptr + row)
            mx = tl.maximum(mx, lf)
            tot = mx + tl.log(tl.exp(ln - mx) + tl.exp(sk - mx) + tl.exp(lf - mx))
        else:
            tot = mx + tl.log(tl.exp(ln - mx) + tl.exp(sk - mx))
        o = acc * (tl.exp(ln - tot) / l)[:, None]
        if FAR:
            o += tl.load(fo_ptr + row[:, None] * 128 + d[None, :]) * tl.exp(lf - tot)[:, None]
        if GATE:  # INK_GLUE2: bf16(o * sigmoid(g)) as attention()'s torch ops (expf + IEEE divide)
            gv = tl.load(g_ptr + tok[:, None] * lg + qh[:, None] * 128 + d[None, :]).to(tl.float32)
            if F32:
                e = tl.exp(-gv)
            else:
                e = tl.extra.cuda.libdevice.exp(-gv)
            o = _bf16(o * tl.math.div_rn(tl.full(gv.shape, 1., tl.float32), 1. + e), F32)
        tl.store(o_ptr + row[:, None] * 128 + d[None, :], o, mask=rm[:, None])


def store_of(kv, slot):
    """(c [2,rows,8,64] u8, s [2,rows,8,8] e4m3, page table) of a kvpool Table, or of a flat KV4 (tests) read as
    PG-row pages with an identity table (rows past cap are only touched under a key mask)."""
    if hasattr(kv, 't'):
        return kv.pool.c[:, slot], kv.pool.s[:, slot], kv.t
    if getattr(kv, 'ident', None) is None:
        kv.ident = torch.arange(-(-kv.cap // PG), device=kv.c.device)
    return kv.c[:, slot], kv.s[:, slot], kv.ident


def tile_meta(c, s, tbl, kend):
    """Per BN-key tile of keys [0, kend) and kv head: dequantized key max / min per dimension -> bf16 [tiles, 8, 2, 128].
    (Keys past kend in the last tile are included: a looser bound, never a wrong one.)"""
    nt = -(-kend // BN)
    meta = torch.empty(nt, NKV, 2, HD, dtype=torch.bfloat16, device=c.device)
    _meta_kernel[(nt, NKV)](c, s, tbl, meta, PG=PG, BN=BN, num_warps=4)
    return meta


def far(Q, kv, slot, p0, kend, split=None, skip=None):
    """Model.far for a prefill chunk at positions [p0, p0+T): keys [0, kend) at distance >= 1024 from the 4-bit store.
    skip: INK_FARSKIP delta in nats (default SKIP[0]; None = off). -> (o [T,24,128] fp32, lse [T,24] fp32)."""
    T = Q.shape[0]
    Q = Q.contiguous()
    c, s, tbl = store_of(kv, slot)
    skip = SKIP[0] if skip is None else skip
    sk = {}
    if skip is not None and skip >= 0:
        assert PG % BN == 0
        st = SKIP_STATS[0]
        sk = dict(meta_ptr=tile_meta(c[0], s[0], tbl, kend), thr=160. if skip == 0 else skip * LOG2E,
                  st_ptr=st, SKIP=True, STATS=st is not None)
    mt = -(-3 * T // BM)
    if split is None:
        ns = max(1, min(-(-PROGS // (mt * NKV)), -(-kend // 8192)))
        split = -(-kend // ns)
    split = -(-split // BN) * BN
    ns = -(-kend // split)
    o = torch.empty(ns, T, NQ, HD, dtype=torch.float32, device=Q.device)
    lse = torch.empty(ns, T, NQ, dtype=torch.float32, device=Q.device)
    _far_kernel[(mt, NKV, ns)](Q, c[0], c[1], s[0], s[1], tbl, o, lse, p0, T, kend, split, HD ** -0.5 * LOG2E,
                               PG=PG, BM=BM, BN=BN, F32=INTERP, FP8=FP8, num_warps=WARPS, num_stages=STAGES, **sk)
    if ns == 1:
        return o[0], lse[0]
    tot = torch.logsumexp(lse, 0)                                     # -inf where no split had keys
    w = torch.nan_to_num(torch.exp(lse - tot))[..., None]
    return (o * w).sum(0), tot


def near(Q, Kn, Vn, table, sink, p0, k0, far=None, gate=None):
    """attn_core's near part: Q [T,24,128] bf16 at positions [p0, p0+T); Kn/Vn [p0+T-k0,8,128] bf16 (positions k0..);
    table (R [T,24,16], rel_logits_proj [16,1024]) or a [T,24,1024] tensor; sink [T,24]; far (o, lse) or None.
    gate [T,3072] bf16 (pre-sigmoid, unit column stride): -> bf16(o * sigmoid(gate)) instead.
    -> o [T,24,128] fp32 (bf16 with gate)."""
    T = Q.shape[0]
    out = torch.empty(T, NQ, HD, dtype=torch.float32 if gate is None else torch.bfloat16, device=Q.device)
    if gate is not None and gate.stride(1) != 1:
        gate = gate.contiguous()
    tab = not isinstance(table, tuple)
    R, pj = (table, table) if tab else table
    fo, fl = far if far is not None else (out, out)                   # unused stand-ins when FAR=False
    bw = triton.next_power_of_2(NBM // 3 + 2 + NBN)
    _near_kernel[(-(-3 * T // NBM), NKV)](
        Q.contiguous(), Kn.contiguous(), Vn.contiguous(), R.contiguous(), pj.contiguous(), sink.float().contiguous(),
        fo.contiguous(), fl.contiguous(), out, p0, T, k0, HD ** -0.5, out if gate is None else gate,
        0 if gate is None else gate.stride(0),
        TAB=tab, FAR=far is not None, BM=NBM, BN=NBN, BW=bw, F32=INTERP, GATE=gate is not None,
        num_warps=NWARPS, num_stages=STAGES)
    return out
