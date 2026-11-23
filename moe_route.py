"""INK_MOE_ROUTE=1 (off by default; exact): the routed-row sort and block tables of the grouped MoE in 3 launches.

Model.moe keeps torch for the router math (fp32 GEMM, sigmoid, top-8, weight normalisation: bit-exact by construction)
and hands the [T,8] expert ids / weights here instead of argsort + gathers, moe_triton.tables (~20 tiny launches) and
finish's stable sort. Per visit: count rows per expert (atomics), cumsum + block tables (each program recomputes the
128-entry cumsum), scatter each (token, expert) into its expert's segment. Rows land in the expert segments in
atomic order, not token order; that changes nothing numerically: every row's GEMMs, activation and quantisation are
per row, and finish sums each token's rows in expert order through `fidx` (= finish's sort result), so outputs are
bit-identical to the torch routing (tests/test_moe4.py; tests/gpu_moe4.py checks torch.equal on the GPU).
"""
import torch

from moe_triton import triton

BT = 64      # tokens per count / scatter program
BLK = 64     # block-table entries per program

if triton is not None:
    import triton.language as tl

    @triton.jit
    def _count(idx_ptr, cnt_ptr, T, K: tl.constexpr, BT: tl.constexpr):
        t = tl.program_id(0) * BT + tl.arange(0, BT)
        m = (t < T)[:, None]
        e = tl.load(idx_ptr + t[:, None] * K + tl.arange(0, K)[None, :], mask=m, other=0).to(tl.int32)
        tl.atomic_add(cnt_ptr + e, tl.full((BT, K), 1, tl.int32), mask=m)

    @triton.jit
    def _pick(v, sel):  # v [E] gathered at one expert per row of sel [n, E] (one-hot)
        return tl.sum(tl.where(sel, v[None, :], 0), axis=1)

    @triton.jit
    def _tables(cnt_ptr, start_ptr, tab_ptr, n_blk, E: tl.constexpr, BB: tl.constexpr, BS: tl.constexpr,
                BLK: tl.constexpr):
        """moe_triton.tables from the counts: (expert, row0, row1, big) per block, same layout and values."""
        ea = tl.arange(0, E)
        c = tl.load(cnt_ptr + ea)
        nbig = c // BB
        nb = nbig + (c - nbig * BB + BS - 1) // BS
        cb = tl.cumsum(nb, 0)                                   # first block of the next expert
        st = tl.cumsum(c, 0) - c                                # first row of each expert
        if tl.program_id(0) == 0:
            tl.store(start_ptr + ea, st)
        b = tl.program_id(0) * BLK + tl.arange(0, BLK)
        e = tl.sum((cb[None, :] <= b[:, None]).to(tl.int32), axis=1)
        sel = ea[None, :] == tl.minimum(e, E - 1)[:, None]     # clamped like tables()' ec
        nbe, nbge, ste = _pick(nb, sel), _pick(nbig, sel), _pick(st, sel)
        i = b - (_pick(cb, sel) - nbe)
        big = i < nbge
        r0 = ste + tl.where(big, i * BB, nbge * BB + (i - nbge) * BS)
        bm = b < n_blk
        tl.store(tab_ptr + b, tl.where(e < E, e, -1), mask=bm)
        tl.store(tab_ptr + n_blk + b, r0, mask=bm)
        tl.store(tab_ptr + 2 * n_blk + b, ste + _pick(c, sel), mask=bm)
        tl.store(tab_ptr + 3 * n_blk + b, big.to(tl.int32), mask=bm)

    @triton.jit
    def _scatter(idx_ptr, w_ptr, start_ptr, fill_ptr, tok_ptr, ws_ptr, fidx_ptr, T, K: tl.constexpr,
                 BT: tl.constexpr):
        t = tl.program_id(0) * BT + tl.arange(0, BT)
        m = (t < T)[:, None]
        o = t[:, None] * K + tl.arange(0, K)[None, :]
        e = tl.load(idx_ptr + o, mask=m, other=0).to(tl.int32)
        slot = tl.load(start_ptr + e, mask=m, other=0) + tl.atomic_add(fill_ptr + e, tl.full((BT, K), 1, tl.int32),
                                                                      mask=m)
        tl.store(tok_ptr + slot, tl.broadcast_to(t[:, None], (BT, K)), mask=m)
        tl.store(ws_ptr + slot, tl.load(w_ptr + o, mask=m, other=0.), mask=m)
        rank = tl.sum((e[:, :, None] > e[:, None, :]).to(tl.int32), axis=2)   # the token's experts are distinct
        tl.store(fidx_ptr + t[:, None] * K + rank, slot, mask=m)


def route(idx, w, bb, bs, E=128):
    """idx [T,K] expert ids (distinct per token), w [T,K] fp32 routing weights -> rt = (tok int32 [R], ws fp32 [R],
    be, b0, b1, bb int32 [n_blk] (moe_triton.tables layout for block sizes bb/bs), fidx int32 [R] (per token, its rows
    in expert order = moe_triton.finish's sort))."""
    T, K = idx.shape
    R, dev = T * K, idx.device
    idx, w = idx.contiguous(), w.contiguous()
    n_blk = R // bb + E * ((bb - 1) // bs + 1) if bb > bs else R // bs + E
    ci = torch.zeros(3 * E, dtype=torch.int32, device=dev)     # counts | fill | start
    cnt, fill, start = ci[:E], ci[E:2 * E], ci[2 * E:]
    tab = torch.empty(4, n_blk, dtype=torch.int32, device=dev)
    tok = torch.empty(R, dtype=torch.int32, device=dev)
    ws = torch.empty(R, dtype=torch.float32, device=dev)
    fidx = torch.empty(R, dtype=torch.int32, device=dev)
    _count[(triton.cdiv(T, BT),)](idx, cnt, T, K=K, BT=BT)
    _tables[(triton.cdiv(n_blk, BLK),)](cnt, start, tab, n_blk, E=E, BB=bb, BS=bs, BLK=BLK)
    _scatter[(triton.cdiv(T, BT),)](idx, w, start, fill, tok, ws, fidx, T, K=K, BT=BT)
    return (tok, ws, tab[0], tab[1], tab[2], tab[3], fidx)
