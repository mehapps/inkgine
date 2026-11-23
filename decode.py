"""Single-token decode with fixed shapes and no host syncs, so one step can be captured in a CUDA graph (INK_GRAPH=1).

Model.forward runs unchanged on a DecodeState instead of a Seq. Differences from the Seq path (prefill + old decode):
- Rings: a fixed 1024-slot bf16 ring per visit; token p lives in slot p % 1024. A step writes its own K/V first,
  then attends over all 1024 slots with a mask from the shared slot->position table (-1 = empty). Same key set as
  the Seq path (1023 previous keys + the current one), only in slot order instead of time order.
- Positions (tau, bias distances, KV page-pool write index) come from the device tensor `pos`.
- Conv states are updated in place (Model's sconv does that for both paths).
- Far keys (full visits, distance >= 1024): on cuda (INK_DEC_FAR=triton, the default there) `far_attn`, a Triton
  kernel that reads the 4-bit pages straight from the pool through the page table, split-K over INK_DEC_FAR_BLK keys
  per program with the valid range read from `pos` on the device; the partial (o, lse) are merged by LSE. Fixed grid
  over the whole pool capacity, so ONE graph serves every position, and scratch is only the partials (~3 MB at 1M).
  The torch path (INK_DEC_FAR=torch, the CPU default) reads [0, far_cap) through Model.far (bf16 blocks) and masks;
  far_cap is a power-of-two bucket, one graph per bucket. Don't graph it at long context: its bf16 block scratch
  (~0.8 GB with 64k blocks) stays pinned in the graph pool (2026-11-21 1M OOM after a 130k request).
  The state's page table is a fixed-address copy of the sequence's (Decoder.load), so graphs work for any sequence.
- MoE: the grouped Triton kernels on cuda (block table built on device). The torch path has a .tolist().
- INK_DEC_FUSE=1: Model.forward is replaced by decode_kernels.Fused (~550 Triton launches per step instead of ~10.5k
  torch ops; same math and rounding points). INK_DEC_FAR_DOT=1: the far kernel on tensor cores (_far_dot_kernel).
  INK_DEC_EMBED_GPU=1: embedding row lookup inside the step + Decoder.stream() (device sampling, step n+1 launched
  before token n is read on the host).

Usage (engine.py hook, one Decoder per Model, reused across requests):
    dec = Decoder(model)             # INK_GRAPH=1 -> CUDA graphs on cuda; otherwise the same code eagerly
    ... prefill with model.forward(x, seq) as today, sample token t from the last row ...
    dec.load(seq)                    # copies conv states + rings in; don't call model.forward(.., seq) afterwards
    logits = dec.step(t)             # [1, VOCAB] fp32 static buffer (overwritten by the next step); seq.pos += 1
    t = int(sample(logits[0], temperature, top_p, top_k, gen))
"""
import hashlib
import os

import torch

import kvpool
from model import H, HD, NKV, NQ, VISITS, VOCAB, WIN, rms

try:
    import triton
    import triton.language as tl
    from moe_triton import INTERP, _bf16, _mma
except ImportError:  # pragma: no cover
    triton = None

# INK_SAMPLE_F64=1: the exponential-race noise in float64 (in case the device's fp32 exponential_ has only ~24 random
# bits, which makes tokens with p / p_max below ~2^-24 undrawable; tests/test_sampler.py measures it). Other RNG stream.
SAMPLE_F64 = os.environ.get('INK_SAMPLE_F64') == '1'
DEC_SPLIT = int(os.environ.get('INK_DEC_FAR_BLK', 4096))  # far keys per kernel program (split-K)
FBLK = 64                                                  # keys per inner tile (divides kvpool.PAGE)

if triton is not None:
    @triton.jit
    def _far_kernel(q_ptr, c0, c1, s0, s1, tbl, pos_ptr, o_ptr, l_ptr, scale,
                    SPLIT: tl.constexpr, PG: tl.constexpr, BLK: tl.constexpr):
        """Program (kv head h, split sp): keys [sp*SPLIT, (sp+1)*SPLIT) clipped to the far range [0, pos-1023),
        for the 3 query heads of h; writes normalized partials (o, lse) (empty range: o = 0, lse = -inf).
        Codes byte i holds elements 2i (low nibble) and 2i+1 (high), so q and o are handled as even/odd halves."""
        h, sp = tl.program_id(0), tl.program_id(1)
        kend = (tl.load(pos_ptr) - 1023).to(tl.int32)
        lo = sp * SPLIT
        hi = tl.minimum(lo + SPLIT, kend)
        g, d, r = tl.arange(0, 4), tl.arange(0, 64), tl.arange(0, BLK)
        gm = g < 3                                              # 3 query heads per kv head (+1 pad row)
        qp = q_ptr + (h * 3 + g)[:, None] * 128 + 2 * d[None, :]
        qe = tl.load(qp, mask=gm[:, None], other=0.).to(tl.float32) * scale
        qo = tl.load(qp + 1, mask=gm[:, None], other=0.).to(tl.float32) * scale
        m = tl.full([4], float('-inf'), tl.float32)
        l = tl.zeros([4], tl.float32)
        ae = tl.zeros([4, 64], tl.float32)
        ao = tl.zeros([4, 64], tl.float32)
        for k0 in range(lo, hi, BLK):
            row = tl.load(tbl + k0 // PG) * PG + k0 % PG + r    # row in the [pages * PG, 8, ...] store
            km = (k0 + r < hi)[:, None]
            co = row[:, None] * 512 + h * 64 + d[None, :]
            so = row[:, None] * 64 + h * 8 + (d // 8)[None, :]
            kc = tl.load(c0 + co, mask=km, other=0)
            ks = tl.load(s0 + so, mask=km, other=0.).to(tl.float32)
            vc = tl.load(c1 + co, mask=km, other=0)
            vs = tl.load(s1 + so, mask=km, other=0.).to(tl.float32)
            klo, khi = ((kc & 15).to(tl.float32) - 8) * ks, ((kc >> 4).to(tl.float32) - 8) * ks
            vlo, vhi = ((vc & 15).to(tl.float32) - 8) * vs, ((vc >> 4).to(tl.float32) - 8) * vs
            s = tl.sum(qe[:, None, :] * klo[None, :, :], 2) + tl.sum(qo[:, None, :] * khi[None, :, :], 2)  # [4,BLK]
            s = tl.where(tl.trans(km), s, float('-inf'))
            mn = tl.maximum(m, tl.max(s, 1))                    # finite: k0 < hi, so the tile has a valid key
            a = tl.exp(m - mn)
            p = tl.exp(s - mn[:, None])
            l = l * a + tl.sum(p, 1)
            ae = ae * a[:, None] + tl.sum(p[:, :, None] * vlo[None, :, :], 1)
            ao = ao * a[:, None] + tl.sum(p[:, :, None] * vhi[None, :, :], 1)
            m = mn
        ok = l > 0
        inv = tl.where(ok, 1 / tl.where(ok, l, 1.), 0.)
        out = sp * 24 + h * 3 + g
        op = o_ptr + out[:, None] * 128 + 2 * d[None, :]
        tl.store(op, ae * inv[:, None], mask=gm[:, None])
        tl.store(op + 1, ao * inv[:, None], mask=gm[:, None])
        tl.store(l_ptr + out, tl.where(ok, m + tl.log(tl.where(ok, l, 1.)), float('-inf')), mask=gm)

    @triton.jit
    def _far_dot_kernel(q_ptr, c0, c1, s0, s1, tbl, pos_ptr, o_ptr, l_ptr, scale,
                        SPLIT: tl.constexpr, PG: tl.constexpr, BLK: tl.constexpr, F32: tl.constexpr,
                        NSP: tl.constexpr = 0):
        """_far_kernel on tensor cores (INK_DEC_FAR_DOT=1): same programs, ranges and outputs, but QK^T and PV are
        tl.dot over BLK-key tiles with the 3 query heads padded to 16 rows. K/V are dequantized in registers to bf16
        (exact: int4 x e4m3 fits bf16), probabilities are rounded to bf16 for PV (like Model.far), fp32 accumulate.
        NSP > 0 (INK_DEC_FAR2): NSP programs per kv head split the far range evenly (keys per program from pos, a
        multiple of BLK), so the GPU is full at every length and the merge reads NSP partials; scales are read as
        [BLK, 8] and broadcast (same values)."""
        h, sp = tl.program_id(0), tl.program_id(1)
        kend = (tl.load(pos_ptr) - 1023).to(tl.int32)
        if NSP > 0:
            spl = tl.cdiv(tl.cdiv(tl.maximum(kend, 0), NSP), BLK) * BLK
        else:
            spl = SPLIT
        lo = sp * spl
        hi = tl.minimum(lo + spl, kend)
        g, d, r = tl.arange(0, 16), tl.arange(0, 64), tl.arange(0, BLK)
        j8 = tl.arange(0, 8)
        gm = g < 3
        qp = q_ptr + (h * 3 + g)[:, None] * 128 + 2 * d[None, :]
        qe = tl.load(qp, mask=gm[:, None], other=0.)
        qo = tl.load(qp + 1, mask=gm[:, None], other=0.)
        m = tl.full([16], float('-inf'), tl.float32)
        l = tl.zeros([16], tl.float32)
        ae = tl.zeros([16, 64], tl.float32)
        ao = tl.zeros([16, 64], tl.float32)
        for k0 in range(lo, hi, BLK):
            row = tl.load(tbl + k0 // PG) * PG + k0 % PG + r
            km = k0 + r < hi
            co = row[:, None] * 512 + h * 64 + d[None, :]
            kc = tl.load(c0 + co, mask=km[:, None], other=0)
            vc = tl.load(c1 + co, mask=km[:, None], other=0)
            if NSP > 0:
                s8 = row[:, None] * 64 + h * 8 + j8[None, :]
                ks = tl.reshape(tl.broadcast_to(tl.load(s0 + s8, mask=km[:, None], other=0.).to(tl.float32)[:, :, None],
                                                (BLK, 8, 8)), (BLK, 64))
                vs = tl.reshape(tl.broadcast_to(tl.load(s1 + s8, mask=km[:, None], other=0.).to(tl.float32)[:, :, None],
                                                (BLK, 8, 8)), (BLK, 64))
            else:
                so = row[:, None] * 64 + h * 8 + (d // 8)[None, :]
                ks = tl.load(s0 + so, mask=km[:, None], other=0.).to(tl.float32)
                vs = tl.load(s1 + so, mask=km[:, None], other=0.).to(tl.float32)
            klo = (((kc & 15).to(tl.float32) - 8) * ks).to(tl.bfloat16)
            khi = (((kc >> 4).to(tl.float32) - 8) * ks).to(tl.bfloat16)
            s = _mma(qe, tl.trans(klo), tl.zeros([16, BLK], tl.float32), F32)
            s = _mma(qo, tl.trans(khi), s, F32) * scale
            s = tl.where(km[None, :], s, float('-inf'))
            mn = tl.maximum(m, tl.max(s, 1))                    # finite: k0 < hi, so the tile has a valid key
            a = tl.exp(m - mn)
            p = tl.exp(s - mn[:, None])
            l = l * a + tl.sum(p, 1)
            pb = _bf16(p, F32)
            ae = _mma(pb, (((vc & 15).to(tl.float32) - 8) * vs).to(tl.bfloat16), ae * a[:, None], F32)
            ao = _mma(pb, (((vc >> 4).to(tl.float32) - 8) * vs).to(tl.bfloat16), ao * a[:, None], F32)
            m = mn
        ok = l > 0
        inv = tl.where(ok, 1 / tl.where(ok, l, 1.), 0.)
        out = sp * 24 + h * 3 + g
        op = o_ptr + out[:, None] * 128 + 2 * d[None, :]
        tl.store(op, ae * inv[:, None], mask=gm[:, None])
        tl.store(op + 1, ao * inv[:, None], mask=gm[:, None])
        tl.store(l_ptr + out, tl.where(ok, m + tl.log(tl.where(ok, l, 1.)), float('-inf')), mask=gm)

    @triton.jit
    def _nib(w, k: tl.constexpr):
        """Nibble k of int32 words w as the bf16 128 + n (bits 0x4300 | n): no int -> float convert."""
        return (((w >> (4 * k)) & 15) | 0x4300).to(tl.int16).to(tl.bfloat16, bitcast=True)

    @triton.jit
    def _dq3(w, sc, BLK: tl.constexpr, F32: tl.constexpr):
        """int4 codes w int32 [BLK, 16] (nibble k of word i = dim 8i+k) x e4m3 scales sc [BLK, 8] (16-dim groups) ->
        bf16 [BLK, 128] (n - 8) * s, exact, in FAR2's column order: column j < 64 = dim 2j, 64 + j = dim 2j + 1."""
        x = tl.join(tl.join(tl.join(_nib(w, 0), _nib(w, 4)), tl.join(_nib(w, 2), _nib(w, 6))),
                    tl.join(tl.join(_nib(w, 1), _nib(w, 5)), tl.join(_nib(w, 3), _nib(w, 7))))  # [BLK,i,a,b,c] dim 8i+4a+2b+c
        x = tl.reshape(tl.permute(x, (0, 4, 1, 2, 3)), (BLK, 128))
        s = tl.reshape(tl.broadcast_to(sc.to(tl.bfloat16)[:, None, :, None], (BLK, 2, 8, 8)), (BLK, 128))
        if F32:                                                  # interpreter: its bf16 fma is wrong; fp32 is exact too
            return (x.to(tl.float32) * s.to(tl.float32) - s.to(tl.float32) * 136.).to(tl.bfloat16)
        return tl.fma(x, s, (s * -136.).to(tl.bfloat16))     # (128 + n) s - 136 s: both terms and the result exact

    @triton.jit
    def _far3_kernel(q_ptr, c0, c1, s0, s1, tbl, pos_ptr, o_ptr, l_ptr, scale, PG: tl.constexpr, BLK: tl.constexpr,
                     F32: tl.constexpr, NSP: tl.constexpr, ST: tl.constexpr):
        """INK_DEC_FAR3: _far_dot_kernel's FAR2 programs, ranges and outputs (NSP even splits per kv head), faster
        body: codes as int32 words (c0/c1 int32 [rows, 8, 16]: 4 B loads instead of 1 B, no 255-register spill)
        through ST cp.async stages, int4 -> bf16 via exponent bits + one bf16 fma (_dq3, no int/fp converts), QK^T as
        one K=128 dot. Same bf16 rounding points as FAR2; the fp32 sum order inside QK^T (and the tile size, default
        64 vs 128) differ, so not bit-identical to FAR2."""
        h, sp = tl.program_id(0), tl.program_id(1)
        kend = (tl.load(pos_ptr) - 1023).to(tl.int32)
        spl = tl.cdiv(tl.cdiv(tl.maximum(kend, 0), NSP), BLK) * BLK
        lo = sp * spl
        hi = tl.minimum(lo + spl, kend)
        g, j, r, w, j8 = tl.arange(0, 16), tl.arange(0, 128), tl.arange(0, BLK), tl.arange(0, 16), tl.arange(0, 8)
        gm = g < 3
        dm = tl.where(j < 64, 2 * j, 2 * j - 127)                    # column -> head dim
        q = tl.load(q_ptr + (h * 3 + g)[:, None] * 128 + dm[None, :], mask=gm[:, None], other=0.)
        m = tl.full([16], float('-inf'), tl.float32)
        l = tl.zeros([16], tl.float32)
        acc = tl.zeros([16, 128], tl.float32)
        for k0 in tl.range(lo, hi, BLK, num_stages=ST):
            row = tl.load(tbl + k0 // PG) * PG + k0 % PG + r
            km = k0 + r < hi
            co = tl.multiple_of(row * 128 + h * 16, 16)[:, None] + w[None, :]   # 64 B per key: 16 B vector loads
            so = tl.multiple_of(row * 64 + h * 8, 8)[:, None] + j8[None, :]
            k = _dq3(tl.load(c0 + co, mask=km[:, None], other=0), tl.load(s0 + so, mask=km[:, None], other=0.), BLK, F32)
            s = _mma(q, tl.trans(k), tl.zeros([16, BLK], tl.float32), F32) * scale
            s = tl.where(km[None, :], s, float('-inf'))
            mn = tl.maximum(m, tl.max(s, 1))                    # finite: k0 < hi, so the tile has a valid key
            a = tl.exp(m - mn)
            p = tl.exp(s - mn[:, None])
            l = l * a + tl.sum(p, 1)
            v = _dq3(tl.load(c1 + co, mask=km[:, None], other=0), tl.load(s1 + so, mask=km[:, None], other=0.), BLK, F32)
            acc = _mma(_bf16(p, F32), v, acc * a[:, None], F32)
            m = mn
        ok = l > 0
        inv = tl.where(ok, 1 / tl.where(ok, l, 1.), 0.)
        out = sp * 24 + h * 3 + g
        tl.store(o_ptr + out[:, None] * 128 + dm[None, :], acc * inv[:, None], mask=gm[:, None])
        tl.store(l_ptr + out, tl.where(ok, m + tl.log(tl.where(ok, l, 1.)), float('-inf')), mask=gm)


FAR_DOT = os.environ.get('INK_DEC_FAR_DOT') == '1'
DOT_BLK = int(os.environ.get('INK_DEC_FAR_TILE', 128))              # keys per tl.dot tile (divides kvpool.PAGE)
# INK_DEC_FAR2=1 (off; Fused + BatchDecoder): far decode attention as FAR_NSP even splits per kv head of the far range
# (work and merge cost follow the context, not the pool size; one graph for every length), tensor-core kernel, and
# the near kernels merge the NSP partials with vector loads instead of a serial loop over pool / 4096 splits.
FAR2 = os.environ.get('INK_DEC_FAR2') == '1'
FAR_NSP = int(os.environ.get('INK_DEC_FAR_NSP', 42))               # 8 x 42 = 336 programs: ~1 wave at 2 CTAs/SM
FAR_KW = dict(num_warps=int(os.environ.get('INK_DEC_FAR_WARPS', 4)), num_stages=int(os.environ.get('INK_DEC_FAR_STAGES', 3)))
# INK_DEC_FAR3=1 (off; Fused only, implies FAR2's splits + merge; BatchDecoder keeps FAR2's kernel at FAR3_NSP splits):
# _far3_kernel. Tunables (window far3 sweep): INK_DEC_FAR3_NSP splits per kv head (63: 504 programs ~ 3 CTAs/SM, its
# 116 registers / 31 KB smem allow 3 vs FAR2's 2), _BLK keys per tile, _STAGES (tl.range stages: the page-table load
# takes 2, so 3 = 1 tile prefetched, 5 = 2), _WARPS.
FAR3 = os.environ.get('INK_DEC_FAR3') == '1'
FAR3_NSP = int(os.environ.get('INK_DEC_FAR3_NSP', 63))
FAR3_KW = dict(BLK=int(os.environ.get('INK_DEC_FAR3_BLK', 64)), ST=int(os.environ.get('INK_DEC_FAR3_STAGES', 3)),
               num_warps=int(os.environ.get('INK_DEC_FAR3_WARPS', 4)))


def launch_far(Q, c0, c1, s0, s1, tbl, pos, o, lse, split, blk=None, nsp=0, far3=False):
    """One far launch (grid (NKV, splits)) with the kernel INK_DEC_FAR_DOT picks; nsp > 0: INK_DEC_FAR2 splits
    (o / lse hold nsp partials; always the tensor-core kernel); far3: INK_DEC_FAR3's kernel for those splits."""
    if nsp and far3:
        kw = dict(FAR3_KW, **({'BLK': blk} if blk else {}))
        if INTERP:
            kw.pop('num_warps')
        _far3_kernel[(NKV, nsp)](Q, c0.view(torch.int32), c1.view(torch.int32), s0, s1, tbl, pos, o, lse, HD ** -0.5,
                                 PG=kvpool.PAGE, F32=INTERP, NSP=nsp, **kw)
    elif nsp:
        _far_dot_kernel[(NKV, nsp)](Q, c0, c1, s0, s1, tbl, pos, o, lse, HD ** -0.5, SPLIT=split, PG=kvpool.PAGE,
                                    BLK=blk or DOT_BLK, F32=INTERP, NSP=nsp, **({} if INTERP else FAR_KW))
    elif FAR_DOT:
        _far_dot_kernel[(NKV, o.shape[0])](Q, c0, c1, s0, s1, tbl, pos, o, lse, HD ** -0.5,
                                           SPLIT=split, PG=kvpool.PAGE, BLK=blk or DOT_BLK, F32=INTERP)
    else:
        _far_kernel[(NKV, o.shape[0])](Q, c0, c1, s0, s1, tbl, pos, o, lse, HD ** -0.5,
                                       SPLIT=split, PG=kvpool.PAGE, BLK=blk or FBLK)


def far_attn(Q, kv, slot, pos, split=None, blk=None):
    """Model.far for one decode token without bf16 K/V: Q [1,24,128] bf16, kv a pool Table (or a flat KV4 in tests),
    pos int64 [1] on the device. -> (o [1,24,128] fp32, lse [1,24]). Fixed shapes (grid over kv.cap), no host sync."""
    split = split or DEC_SPLIT
    if hasattr(kv, 't'):
        c, s, tbl = kv.pool.c[:, slot], kv.pool.s[:, slot], kv.t
    else:  # KV4 [cap,8,..] rows = pages of PAGE rows with an identity table; loads past cap are masked (k < pos-1023)
        c, s = kv.c[:, slot], kv.s[:, slot]
        if getattr(kv, 'ident', None) is None:
            kv.ident = torch.arange(-(-kv.cap // kvpool.PAGE), device=c.device)
        tbl = kv.ident
    ns = -(-kv.cap // split)
    o = torch.empty(ns, NQ, HD, dtype=torch.float32, device=Q.device)
    lse = torch.empty(ns, NQ, dtype=torch.float32, device=Q.device)
    launch_far(Q, c[0], c[1], s[0], s[1], tbl, pos, o, lse, split, blk)
    tot = torch.logsumexp(lse, 0)                                     # [24]; -inf when there are no far keys
    w = torch.nan_to_num(torch.exp(lse - tot))[..., None]
    return (o * w).sum(0, keepdim=True), tot[None]


class DecodeState:
    """Fixed-address decode state for one sequence (graphs capture these tensors' addresses)."""

    def __init__(self, dev, kv):
        self.kv, self.far_cap = kv, 0
        self.kern = os.environ.get('INK_DEC_FAR', 'triton' if dev.type == 'cuda' else 'torch') == 'triton'
        self.moe_impl = 'triton' if dev.type == 'cuda' else None    # None = INK_MOE (CPU tests: same as the Seq path)
        z = lambda c: torch.zeros(3, c, dtype=torch.bfloat16, device=dev)
        self.conv = [{'k': z(1024), 'v': z(1024), 'attn': z(H), 'mlp': z(H)} for _ in VISITS]
        r = lambda: torch.zeros(WIN, NKV, HD, dtype=torch.bfloat16, device=dev)
        self.ring = [(r(), r()) for _ in VISITS]
        self.rpos = torch.full((WIN,), -1, dtype=torch.int64, device=dev)   # position held by each ring slot
        self.pos = torch.zeros(1, dtype=torch.int64, device=dev)

    def positions(self, T):
        """Called once at the start of Model.forward: register this step's slot, build the near mask."""
        assert T == 1, 'DecodeState runs one token per step'
        self.slot = self.pos % WIN
        self.rpos.index_copy_(0, self.slot, self.pos)
        d = self.pos - self.rpos
        self.ok = (self.rpos >= 0) & (d >= 0) & (d < WIN)            # [WIN]
        self.dist = d.clamp(0, WIN - 1)
        return self.pos

    def advance(self, T):  # the host (Decoder.step) owns the position
        pass

    def attend(self, m, Q, K, V, table, sink, v, full):
        """Model.attn_core for one token at device position pos. Q [1,24,128], K/V [1,8,128] bf16, sink [1,24]."""
        rk, rv = self.ring[v]
        rk.index_copy_(0, self.slot, K)
        rv.index_copy_(0, self.slot, V)
        far = None
        if full:
            self.kv.put_at(m.slot[v], self.pos, K, V)
            if self.kern:
                far = far_attn(Q, self.kv, m.slot[v], self.pos)
            elif self.far_cap:
                far = m.far(Q, self.kv, m.slot[v], self.pos, self.far_cap)
        q = Q.view(1, NKV, 3, HD)
        S = torch.einsum('tkgd,skd->kgts', q, rk) * HD ** -0.5        # bf16 [8,3,1,1024]
        tb = (table[0] @ table[1]).permute(1, 0, 2)                   # [24,1,1024]
        bias = tb.gather(2, self.dist.view(1, 1, WIN).expand(tb.shape[0], 1, WIN)).view(NKV, 3, 1, WIN)
        return m.near_merge(S, bias, self.ok, sink, rv, far)


def far_bucket(p, cap):
    """Far key range for a query at position p: keys [0, p-1023) are far; rounded up to a power of two >= 2048."""
    n = p - (WIN - 1)
    return 0 if n <= 0 else min(cap, max(2048, 1 << (n - 1).bit_length()))


class Decoder:
    def __init__(self, model, graph=None, fuse=None, embed_dev=None):
        self.m, dev = model, model.dev
        if graph is None:
            graph = os.environ.get('INK_GRAPH') == '1'
        self.graph = graph and dev.type == 'cuda'
        pool = getattr(model, 'pool', None)                           # unit-test fakes have a KV4 in model.kv
        self.st = DecodeState(dev, pool.new_table() if pool else model.kv)  # fixed-address page table (holds no pages)
        self.x_raw = torch.zeros(1, H, dtype=torch.bfloat16, device=dev)     # raw embedding row (table is on host)
        self.out = torch.zeros(1, VOCAB, dtype=torch.float32, device=dev)
        self.graphs, self.pool, self.seq = {}, None, None
        # INK_DEC_FUSE=1: the whole step as ~550 Triton launches (decode_kernels.py) instead of ~10.5k torch ops
        self.fuse = os.environ.get('INK_DEC_FUSE') == '1' if fuse is None else fuse
        self.fused = None
        # INK_DEC_EMBED_GPU=1: embedding table on the device (+0.9 GB) and the row lookup inside the step, so the
        # token can stay on the device (stream(): step n+1 is launched before token n is read on the host)
        if embed_dev is None:
            embed_dev = os.environ.get('INK_DEC_EMBED_GPU') == '1'
        self.embed = model.W.embed.to(dev) if embed_dev else None
        self.tok = torch.zeros(1, dtype=torch.int64, device=dev)

    @torch.no_grad()
    def load(self, seq):
        """Take over a prefilled Seq: copy its conv states and rings into the static buffers."""
        st, p = self.st, seq.pos
        if hasattr(seq.kv, 't'):  # pool table: graphs read the static copy; pages were reserved for prompt + max_tokens
            st.kv.t.copy_(seq.kv.t)
        elif seq.kv is not st.kv:  # a flat KV4 store (tests): graphs hold its addresses
            st.kv, self.graphs = seq.kv, {}
            if self.fused is not None:
                self.fused.refresh_kv()
        for a, b in zip(st.conv, seq.conv):
            for k in a:
                a[k].copy_(b[k])
        n = max(k.shape[0] for k, _ in seq.ring)                       # min(pos, 1023), the same for every visit
        st.rpos.fill_(-1)
        if n:
            pos = torch.arange(p - n, p, device=st.pos.device)
            slots = pos % WIN
            st.rpos.index_copy_(0, slots, pos)
            for (rk, rv), (k, v) in zip(st.ring, seq.ring):
                if k.shape[0]:                                         # (unit tests fill only some visits)
                    rk.index_copy_(0, slots, k)
                    rv.index_copy_(0, slots, v)
        self.seq = seq

    def _run(self):
        m = self.m
        if self.embed is not None:
            torch.index_select(self.embed, 0, self.tok, out=self.x_raw)
        if self.fuse:
            if self.fused is None:
                import decode_kernels
                self.fused = decode_kernels.Fused(m, self.st, self.x_raw, self.out)
            return self.fused.run()
        x0 = rms(self.x_raw, m.W['embed_norm.weight']).bfloat16()
        self.out.copy_(m.logits(m.forward(x0, self.st)))

    @torch.no_grad()
    def step(self, token):
        """Feed one token at seq.pos (int, or an int64 [1] device tensor with embed_dev); returns logits [1, VOCAB]
        fp32 (static buffer)."""
        m, st, p = self.m, self.st, self.seq.pos
        assert p < st.kv.cap, f'context full ({st.kv.cap} tokens)'
        kv = self.seq.kv
        if hasattr(kv, 'pages') and p >= len(kv.pages) * kvpool.PAGE:  # grow by a page (the engine reserves up front)
            assert kv.pool.reserve(kv, p + 1), 'KV pool full'
            st.kv.t.copy_(kv.t)
        if self.embed is None:
            self.x_raw.copy_(m.W.embed[token:token + 1])
        elif isinstance(token, torch.Tensor):
            if token is not self.tok:
                self.tok.copy_(token)
        else:
            self.tok.fill_(token)
        st.pos.fill_(p)
        cap = 0 if st.kern or self.fuse else far_bucket(p, st.kv.cap)               # the kernel path needs no buckets: one graph
        if not self.graph:
            st.far_cap = cap
            self._run()
        elif cap in self.graphs:
            self.graphs[cap].replay()
        else:
            self._capture(cap)
        self.seq.pos += 1
        return self.out

    @torch.no_grad()
    def stream(self, logits, n, temperature=1.0, top_p=1.0, top_k=0, generator=None):
        """Pipelined decode (needs embed_dev): yields up to n host token ids sampled from `logits` [V] and then from
        each step. Sampling and the next step's embedding stay on the device, and step i+1 is launched before token i
        is copied to the host (pinned buffer + event), so the GPU never waits for Python. Stop iterating at any
        time: the step in flight is wasted (it wrote one KV row past the reply). Grammar masks need token i on the
        host before step i+1's logits are final, so json_schema keeps the plain step() loop."""
        assert self.embed is not None, 'stream() needs Decoder(embed_dev=True) (INK_DEC_EMBED_GPU=1)'
        cuda = self.m.dev.type == 'cuda'
        host = torch.zeros(2, dtype=torch.int64, pin_memory=cuda)
        self.tok.copy_(sample(logits, temperature, top_p, top_k, generator))
        for i in range(n):
            h = host[i % 2:i % 2 + 1]
            h.copy_(self.tok, non_blocking=cuda)
            ev = torch.cuda.Event() if cuda else None
            if ev is not None:
                ev.record()
            if i + 1 < n:  # same stream: the copy above reads token i before this step's sample overwrites it
                self.step(self.tok)
                self.tok.copy_(sample(self.out[0], temperature, top_p, top_k, generator))
            if ev is not None:
                ev.synchronize()
            yield int(h[0])

    def _capture(self, cap):
        """First step of a far bucket: run it eagerly on a side stream (this IS the step; also compiles Triton and
        initializes cuBLAS), then record the same work. Recording executes nothing, so state stays as the eager
        step left it. All graphs share one memory pool: only temporaries live there (inputs/outputs/state are
        allocated outside), and graphs never run concurrently."""
        self.st.far_cap = cap
        cur = torch.cuda.current_stream()
        if self.pool is None:
            self.side, self.pool = torch.cuda.Stream(), torch.cuda.graph_pool_handle()
        self.side.wait_stream(cur)
        with torch.cuda.stream(self.side):
            self._run()
        cur.wait_stream(self.side)
        cur.synchronize()
        torch.cuda.empty_cache()   # drop the warm-up's cached blocks (side-stream cache; matters at 1M context)
        g = torch.cuda.CUDAGraph()
        # thread_local: see decode_batch._replay (a writer-thread CUDA call must not invalidate this capture)
        with torch.cuda.graph(g, pool=self.pool, capture_error_mode='thread_local'):
            self._run()
        self.graphs[cap] = g


def pos_noise(gen, seed, pos, V, device):
    """Exp(1) noise in vocab order [V] for the token at position pos of a request with this seed (float64 with
    INK_SAMPLE_F64, else float32; clamped > 0). gen is reseeded from (seed, pos) first, so the draw depends on nothing
    else: not on how many draws came before (spec rounds of any width, probes, dormant stretches), the batch it ran in,
    or a preemption. Plain sampling, spec verify rows and the drafter all use the noise of the position they sample, so
    a seeded request gives the same tokens with or without speculation, alone or batched."""
    h = hashlib.blake2b(f'{int(seed)}/{int(pos)}'.encode(), digest_size=8).digest()
    gen.manual_seed(int.from_bytes(h, 'little') >> 1)
    dt = torch.float64 if SAMPLE_F64 else torch.float32
    return torch.empty(V, dtype=dt, device=device).exponential_(generator=gen).clamp_(min=torch.finfo(dt).tiny)


@torch.no_grad()
def sample(logits, temperature=1.0, top_p=1.0, top_k=0, generator=None, noise=None):
    """On-device sampling from logits [V] fp32 -> int64 tensor [1] on the same device (no host sync; int() it when
    the token is needed on the host). Same rules as Engine.sample: temperature <= 0 is greedy, top-k, then top-p
    drops tokens whose exclusive cumulative probability exceeds p. Draw: argmax(p / Exp(1)) (exponential race),
    an exact sample from p. noise: Exp(1) [V] in vocab order (pos_noise) instead of a fresh draw from generator.
    shortcut: top-k keeps exactly k tokens (Engine.sample also keeps ties with the k-th value)."""
    if not temperature or temperature <= 0:
        return logits.argmax(-1, keepdim=True)
    x, idx = logits.float() / temperature, None
    if top_k and 0 < top_k < x.numel():
        x, idx = x.topk(top_k)                                         # sorted descending
    if top_p and top_p < 1.0:
        if idx is None:
            x, idx = x.sort(descending=True)
        p = torch.softmax(x, -1)
        p = p.masked_fill(p.cumsum(-1) - p > top_p, 0)
    else:
        p = torch.softmax(x, -1)
    if noise is not None:
        e = noise if idx is None else noise[idx]
        j = (p.double() / e if e.dtype == torch.float64 else p / e).argmax(-1, keepdim=True)
    elif SAMPLE_F64:
        e = torch.empty_like(p, dtype=torch.float64).exponential_(generator=generator).clamp_(min=1e-300)
        j = (p.double() / e).argmax(-1, keepdim=True)
    else:
        e = torch.empty_like(p).exponential_(generator=generator).clamp_(min=torch.finfo(torch.float32).tiny)
        j = (p / e).argmax(-1, keepdim=True)
    return j if idx is None else idx.gather(-1, j)


def sample_rows(lg, rows):
    """lg [R, V] fp32, rows [(temperature, top_p, top_k, generator or noise)] -> int64 [R] on lg's device, row i
    bit-identical to decode.sample(lg[i], *rows[i]) (same math; a generator row draws its exponentials with the same
    shape; a noise row ([V] in vocab order, pos_noise) uses it as sample(noise=) does): greedy rows in one argmax,
    sampled rows without top-k batched per (temperature, top_p), top-k rows one by one through sample. No host sync."""
    R, V = lg.shape
    greedy = [i for i, (t, _, _, _) in enumerate(rows) if not t or t <= 0]
    if len(greedy) == R:
        return lg.argmax(-1)
    out, dev, groups = torch.empty(R, dtype=torch.long, device=lg.device), lg.device, {}
    if greedy:
        g = torch.tensor(greedy, device=dev)
        out[g] = lg.index_select(0, g).argmax(-1)
    for i, (t, p, k, gen) in enumerate(rows):
        if t and t > 0:
            if k and 0 < k < V:
                nz = torch.is_tensor(gen)
                out[i:i + 1] = sample(lg[i], t, p, k, None if nz else gen, gen if nz else None)
            else:
                groups.setdefault((t, p), []).append(i)
    for (t, p), ix in groups.items():
        g = torch.tensor(ix, device=dev)
        x, idx = lg.index_select(0, g).float() / t, None
        if p and p < 1.0:
            x, idx = x.sort(-1, descending=True)
            q = torch.softmax(x, -1)
            # per row, as sample's 1-D cumsum: a batched 2-D scan may sum in another order (unchecked) and move the
            # top-p cut by a last bit
            c = x.new_empty(x.shape)
            for j in range(len(ix)):
                torch.cumsum(q[j], -1, out=c[j])
            q = q.masked_fill(c - q > p, 0)
        else:
            q = torch.softmax(x, -1)
        nz = [torch.is_tensor(rows[i][3]) for i in ix]
        f64 = SAMPLE_F64 or any(nz) and rows[ix[nz.index(True)]][3].dtype == torch.float64
        e = torch.empty_like(q, dtype=torch.float64 if f64 else q.dtype)
        for j, i in enumerate(ix):
            gen = rows[i][3]
            if nz[j]:
                e[j].copy_(gen if idx is None else gen[idx[j]])
            else:
                e[j].exponential_(generator=gen)
        j = ((q.double() / e.clamp_(min=1e-300)) if f64 else (q / e.clamp_(min=torch.finfo(torch.float32).tiny))
             ).argmax(-1, keepdim=True)
        out[g] = (j if idx is None else idx.gather(-1, j)).view(-1)
    return out
