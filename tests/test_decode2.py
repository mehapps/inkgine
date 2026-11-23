"""CPU (interpreter) checks for the decode speed flags of 2026-11-23 (decode_kernels.OPT / decode.FAR2, all off by
default). UNROLL, PDL and L2PF never change the math (loop unrolling keeps the op order; PDL and the L2 prefetches are
inline PTX, off in the interpreter); what can change results is checked here:
  test_w2all          INK_DEC_W2ALL kernel == _w2_kernel + _w2_fin_kernel bit for bit (synthetic NVFP4 and MXFP8
                      experts; mlp out and conv state)
  test_far2           INK_DEC_FAR2 far kernel (NSP even splits, [BLK, 8] scale loads) merged == the old split kernel
                      merged, both vs an fp64 reference, at several far lengths (incl. none)
  test_far3           INK_DEC_FAR3 far kernel vs fp64 and vs FAR2 at the same splits (tiles 32/64/128, empty splits,
                      shuffled page table)
  test_far2_merge     _near_out / _near_fin with the vector far merge (NSF) == the serial merge (partials incl. -inf)
  test_bfar2          decode_batch._bfar_kernel NSP mode == the old split mode for a verify group (rows with different
                      far ends) and a single row
  test_bfar2_rows     2 slots (different positions + page tables) through the whole batched far + merge: per row vs
                      fp64 and vs the old path
  test_bfar_verify    a verify group of <= 10 rows reads the far KV once (MT 32) and each row == its 1-row step
                      (GPU, TRITON_INTERPRET=0: bit for bit; CPU: 5e-5); FAR2 split-length steps split the group
  INK_REAL=1:
  test_w2all_real     W2ALL == w2 + w2_fin bit for bit on real layers 6 (NVFP4) and 44 (MXFP8)
  test_flags_step     (env INK_DEC_FAR3=1: with the FAR3 kernel) whole fused step, all flags on vs off, lockstep from the same state at 2100+ (far active):
                      KL vs a 1-ulp nudge baseline (gate as test_decode.test_fused_real), top-1 equal or near-tie
  test_flags_batch    BatchDecoder step (3 slots incl. a verify group) all flags on vs off: KL tiny, top-1 equal
Run: cd engine && PATH=$PWD/.venv/bin:$PATH TRITON_INTERPRET=1 python tests/test_decode2.py [names...]
Also run the existing suites with the flags on (see NOTES 2026-11-23 decode2 entry)."""
import os
import sys
import time

os.environ.setdefault('TRITON_INTERPRET', '1')
os.environ.setdefault('INK_DEVICE', 'cpu')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import torch  # noqa: E402

import decode as D  # noqa: E402
import decode_batch as DB  # noqa: E402
import decode_kernels as K  # noqa: E402
import kvpool  # noqa: E402
import model as M  # noqa: E402

H, NI = M.H, K.NI
ALL = dict(UNR=True, PDL=True, PF=True, W2ALL=True)
OFF = dict.fromkeys(ALL, False)


def flags(on, nsp=0):
    """Set the flags read by the next Fused / BatchDecoder."""
    K.OPT.update(ALL if on else OFF)
    D.FAR2, D.FAR_NSP = bool(on), (nsp or D.FAR_NSP)
    D.FAR3, D.FAR3_NSP = bool(on) and os.environ.get('INK_DEC_FAR3') == '1', (nsp or D.FAR3_NSP)  # env: FAR3 kernel too


def fake_experts(fp4, g, E=8):
    if fp4:
        w = torch.randint(0, 256, (E, H, NI // 2), generator=g, dtype=torch.uint8)
        s = (torch.rand(E, H, NI // 16, generator=g) * 0.5 + 0.05).to(torch.float8_e4m3fn)
        gs = torch.rand(E, generator=g) * 0.02 + 0.01
    else:
        w = (torch.randn(E, H, NI, generator=g) * 2).to(torch.float8_e4m3fn)
        s = torch.randint(118, 124, (E, H, NI // 32), generator=g, dtype=torch.uint8)
        gs = torch.zeros(E)
    return w, s, gs


def test_w2all(ex=None, label='synthetic'):
    g = torch.Generator().manual_seed(1)
    K.BLK.update(w2=128, w2a=128, w2f=256)               # fewer interpreter programs (same per-row math)
    for fp4 in (True, False):
        w2, s2, gs = ex[fp4] if ex else fake_experts(fp4, g)
        act = (torch.randn(8, NI, generator=g) * 0.5).bfloat16()
        idx = torch.randperm(w2.shape[0], generator=g)[:8].to(torch.int32)
        wt = torch.rand(8, generator=g) * 2
        R = torch.randn(H, generator=g)
        st0 = torch.randn(3, H, generator=g).bfloat16()
        cw = torch.randn(H, 4, generator=g) * 0.3
        args = (act, idx, wt, w2, s2, gs, w2.stride(0), w2.stride(1), s2.stride(0), s2.stride(1))
        kw = dict(FP4=fp4, F32=True, H=H, NI=NI, BK=K.BLK['kbk2'])
        part, st1, h1 = torch.zeros(8, H), st0.clone(), torch.zeros(H, dtype=torch.bfloat16)
        K._w2_kernel[(H // K.BLK['w2'], 8)](*args, part, BN=K.BLK['w2'], **kw)
        K._w2_fin_kernel[(H // K.BLK['w2f'],)](part, st1, cw, R, h1, F32=True, H=H, BN=K.BLK['w2f'])
        st2, h2 = st0.clone(), torch.zeros(H, dtype=torch.bfloat16)
        K._w2all_kernel[(H // K.BLK['w2a'],)](*args, st2, cw, R, h2, BN=K.BLK['w2a'], UNR=1, **kw)
        same = torch.equal(h1, h2) and torch.equal(st1, st2)
        lab = label if ex is None or fp4 in ex.get('real', (fp4,)) else 'synthetic'
        print(f'w2all {lab} {"NVFP4" if fp4 else "MXFP8"}: mlp out + conv state bit-identical to w2 + w2_fin: {same}'
              + ('' if same else f' (max |dh| {(h1.float() - h2.float()).abs().max():.2e})'), flush=True)
        assert same


def far_store(cap, g):
    """Random flat KV4-like store (rows = positions): codes [2, cap, 8, 64] u8, scales [2, cap, 8, 8] e4m3 and an
    identity page table."""
    c = torch.randint(0, 256, (2, cap, 8, 64), generator=g, dtype=torch.uint8)
    s = (torch.rand(2, cap, 8, 8, generator=g) * 0.4 + 0.02).to(torch.float8_e4m3fn)
    return c, s, torch.arange(-(-cap // kvpool.PAGE))


def far_ref(Q, c, s, kend):
    """fp64 attention of the 24 q heads over keys [0, kend) of the dequantized store -> (o [24,128], lse [24])."""
    def deq(x, sc):
        lo, hi = (x & 15).double() - 8, (x >> 4).double() - 8
        v = torch.stack((lo, hi), -1).reshape(*x.shape[:-1], 128)
        return v * sc.double().repeat_interleave(16, -1)
    k, v = deq(c[0, :kend], s[0, :kend]), deq(c[1, :kend], s[1, :kend])          # [n, 8, 128]
    q = Q.view(8, 3, 128).double()
    S = torch.einsum('hgd,nhd->hgn', q, k) * 128 ** -0.5
    lse = torch.logsumexp(S, -1)
    o = torch.einsum('hgn,nhd->hgd', torch.softmax(S, -1), v)
    return o.reshape(24, 128), lse.reshape(24)


def merge(o, l):
    tot = torch.logsumexp(l, 0)
    w = torch.nan_to_num(torch.exp(l - tot))[..., None]
    return (o * w).sum(0), tot


def test_far2():
    g = torch.Generator().manual_seed(2)
    cap = 3 * 4096
    c, s, tbl = far_store(cap, g)
    Q = (torch.randn(24, 128, generator=g) * 2).bfloat16()
    D.FAR_DOT = True                                         # 'old' = the server's tensor-core split kernel
    for pos in (900, 1100, 5000, cap):
        kend = max(pos - 1023, 0)
        res = {}
        for name, nsp in (('old', 0), ('far2', 5)):
            ns = nsp or -(-cap // 4096)
            o, l = torch.zeros(ns, 24, 128), torch.zeros(ns, 24)
            D.launch_far(Q, c[0], c[1], s[0], s[1], tbl, torch.tensor([pos]), o, l, 4096, nsp=nsp)
            if kend == 0:
                assert torch.isinf(l).all() and (o == 0).all(), name
                continue
            res[name] = merge(o, l)
        if kend == 0:
            print(f'far2 pos {pos}: no far keys -> all partials empty (both)', flush=True)
            continue
        ro, rl = far_ref(Q, c, s, kend)
        err = {n: ((o.double() - ro).abs().max() / ro.abs().max()).item() for n, (o, _) in res.items()}
        el = {n: (l.double() - rl).abs().max().item() for n, (_, l) in res.items()}
        print(f'far2 pos {pos} ({kend} far keys): o rel err vs fp64 old {err["old"]:.1e} far2 {err["far2"]:.1e}; '
              f'lse abs err old {el["old"]:.1e} far2 {el["far2"]:.1e}', flush=True)
        assert err['far2'] < 2e-2 and el['far2'] < 1e-3 and err['far2'] < 3 * err['old'] + 1e-3


def test_far3():
    """INK_DEC_FAR3 kernel (int32 code words, exponent-bit dequant, one K=128 dot) vs fp64 and vs FAR2 at the same
    splits: per-split partials, merged output, empty splits, tile sizes 32/64/128, a short tail and a non-identity
    page table (pages reversed)."""
    g = torch.Generator().manual_seed(3)
    cap = 3 * 4096
    c, s, tbl = far_store(cap, g)
    npg = tbl.numel()
    perm = torch.randperm(npg, generator=g)                  # logical page i lives in physical page perm[i]
    cp, sp_ = torch.empty_like(c), torch.empty_like(s)
    for i in range(npg):
        cp[:, perm[i] * 256:(perm[i] + 1) * 256] = c[:, i * 256:(i + 1) * 256]
        sp_[:, perm[i] * 256:(perm[i] + 1) * 256] = s[:, i * 256:(i + 1) * 256]
    Q = (torch.randn(24, 128, generator=g) * 2).bfloat16()
    for pos, nsp, blk in ((900, 5, 64), (1100, 5, 64), (1023 + 77, 3, 32), (5000, 5, 128), (cap, 7, 64)):
        kend = max(pos - 1023, 0)
        res = {}
        for name, far3, cc, ss, tb in (('far2', False, c, s, tbl), ('far3', True, cp, sp_, perm)):
            o, l = torch.zeros(nsp, 24, 128), torch.zeros(nsp, 24)
            D.launch_far(Q, cc[0], cc[1], ss[0], ss[1], tb, torch.tensor([pos]), o, l, 4096, blk=blk, nsp=nsp,
                         far3=far3)
            res[name] = o, l
        (o2, l2), (o3, l3) = res['far2'], res['far3']
        assert torch.equal(torch.isinf(l2), torch.isinf(l3)) and (o3[torch.isinf(l3)] == 0).all(), 'empty splits'
        if kend == 0:
            print(f'far3 pos {pos}: no far keys -> all partials empty', flush=True)
            continue
        ro, rl = far_ref(Q, c, s, kend)
        (m2, t2), (m3, t3) = merge(o2, l2), merge(o3, l3)
        e3 = ((m3.double() - ro).abs().max() / ro.abs().max()).item()
        e2 = ((m2.double() - ro).abs().max() / ro.abs().max()).item()
        d23 = ((o3 - o2).abs().max() / o2.abs().max()).item()
        dl = (l3 - l2)[~torch.isinf(l2)].abs().max().item()
        print(f'far3 pos {pos} ({kend} far keys, nsp {nsp}, BLK {blk}): o rel err vs fp64 far2 {e2:.1e} far3 {e3:.1e};'
              f' partials far3 vs far2 o {d23:.1e} lse {dl:.1e}; lse vs fp64 {(t3.double() - rl).abs().max():.1e}',
              flush=True)
        assert e3 < 2e-2 and e3 < 3 * e2 + 1e-3 and dl < 1e-4 and (t3.double() - rl).abs().max() < 1e-3


def test_far2_merge():
    """Near kernels on random inputs: vector far merge (NSF) vs the serial loop, same partials."""
    g = torch.Generator().manual_seed(3)
    b = K.BLK['near']
    nt, ns = 1024 // b, 6
    Qb, Rb = (torch.randn(24, 128, generator=g) * 0.3).bfloat16(), (torch.randn(24, 16, generator=g) * 0.3).bfloat16()
    proj = (torch.randn(16, 1024, generator=g) * 0.1).bfloat16()
    rk, rv = (torch.randn(1024, 1024, generator=g)).bfloat16(), torch.randn(1024, 1024, generator=g).bfloat16()
    pos = torch.tensor([3000])
    rpos = torch.empty(1024, dtype=torch.int64)                 # slot p % 1024 holds p
    for p in range(3000 - 1023, 3001):
        rpos[p % 1024] = p
    Y = (torch.randn(8664, generator=g)).bfloat16()
    fo = torch.randn(ns, 24, 128, generator=g)
    fl = torch.randn(ns, 24, generator=g) * 3
    fl[2] = float('-inf')                       # empty splits (o = 0)
    fo[2] = 0
    fl[5, :7] = float('-inf')
    fo[5, :7] = 0
    outs = []
    for nsf in (0, 8):
        pm, pl, pacc, plse = torch.zeros(nt, 24), torch.zeros(nt, 24), torch.zeros(nt, 24, 128), torch.zeros(24)
        att = torch.zeros(3072, dtype=torch.bfloat16)
        K._near_part_kernel[(8, nt)](Qb, Rb, proj, rk, rpos, pos, pm, pl, F32=True, BLK=b)
        K._near_out_kernel[(8, nt)](Qb, Rb, proj, Y, rk, rv, rpos, pos, pm, pl, fl, ns, pacc, plse, FAR=True, F32=True,
                                    BLK=b, NS=nt, NSF=nsf)
        K._near_fin_kernel[(24,)](pacc, plse, fo, fl, ns, Y, att, FAR=True, F32=True, NS=nt, NSF=nsf)
        outs.append((plse, att))
    (l0, a0), (l1, a1) = outs
    dl = (l0 - l1).abs().max().item()
    da = ((a0.float() - a1.float()).abs() / a0.float().abs().clamp_min(1e-3)).max().item()
    ulp = (a0 != a1).float().mean().item()
    print(f'far2 merge: lse max diff {dl:.1e}, attention out max rel {da:.1e} ({100 * ulp:.2f}% of values differ, '
          f'bf16 ulp level)', flush=True)
    assert dl < 1e-5 and da < 1.6e-2


def test_bfar2():
    g = torch.Generator().manual_seed(4)
    RM = DB.RM
    cap = 3 * 4096
    c, s, tbl0 = far_store(cap, g)
    tbl = tbl0[None].clone()                                 # one slot
    qb = (torch.randn(RM, 3072, generator=g) * 2).bfloat16()
    for rows, label in (([5000, 5001, 5002, 5003], 'verify group of 4'), ([9100], 'single row')):
        meta = torch.zeros(8, RM, dtype=torch.int64)
        n = len(rows)
        meta[1, :n] = torch.tensor(rows)
        meta[3, :n] = 1
        meta[4, 0], meta[5, 0] = 0, n
        res = {}
        for name, nsp in (('old', 0), ('far2', 7)):
            ns = nsp or -(-cap // 4096)
            o, l = torch.zeros(RM, ns, 24, 128), torch.zeros(RM, ns, 24)
            DB._bfar_kernel[(8, ns, 1)](qb, c[0], c[1], s[0], s[1], tbl, tbl.shape[1], meta, o, l, ns, 128 ** -0.5,
                                         SPLIT=4096, PG=kvpool.PAGE, BLK=D.DOT_BLK, F32=True, RM=RM, NSP=nsp)
            res[name] = [merge(o[i], l[i]) for i in range(n)]
        worst = 0.
        for i, p in enumerate(rows):
            ro, _ = far_ref(qb[i].view(24, 128), c, s, p - 1023)
            for nm in res:
                worst = max(worst, ((res[nm][i][0].double() - ro).abs().max() / ro.abs().max()).item())
            d = ((res['far2'][i][0] - res['old'][i][0]).abs().max() / res['old'][i][0].abs().max()).item()
            assert d < 2e-2, (label, i, d)
        print(f'bfar2 {label}: far2 == old split within {d:.1e} rel; worst rel err vs fp64 {worst:.1e}', flush=True)
        assert worst < 2e-2


def test_bfar2_rows():
    """Batched FAR2 per row (window final5 'row 1' suspicion): 2 slots at different positions with different page
    tables, through _bfar + _bnear_part/out/fin: far partials vs fp64 per row, and the attention output == the old
    split + serial merge path (bf16 ulp level) per row."""
    import triton
    g = torch.Generator().manual_seed(5)
    RM, WR, bl = DB.RM, DB.WR, DB.BB['near']
    nt, cap, pos = WR // bl, 3 * 4096, [9100, 5000]
    c, s, tbl0 = far_store(cap, g)
    tbl = torch.stack([tbl0, tbl0.flip(0)])                  # slot 1: pages in reverse order
    qb, rb = (torch.randn(RM, 3072, generator=g) * 2).bfloat16(), (torch.randn(RM, 384, generator=g) * .3).bfloat16()
    proj = (torch.randn(16, 1024, generator=g) * .1).bfloat16()
    rk, rv = torch.randn(2, WR, 1024, generator=g).bfloat16(), torch.randn(2, WR, 1024, generator=g).bfloat16()
    rpos = torch.full((2, WR), -1, dtype=torch.int64)
    for sl, p in enumerate(pos):
        for x in range(p - 1023, p + 1):
            rpos[sl, x % WR] = x
    Y = torch.randn(RM, 8664, generator=g).bfloat16()
    meta = torch.zeros(8, RM, dtype=torch.int64)
    meta[0, :2], meta[1, :2], meta[3, :2] = torch.tensor([0, 1]), torch.tensor(pos), 1
    meta[4, :2], meta[5, :2] = torch.tensor([0, 1]), 1      # one far group per row
    out = {}
    for name, nsp in (('old', 0), ('far2', 7)):
        ns, nsf = nsp or -(-cap // 4096), triton.next_power_of_2(nsp) if nsp else 0
        fo, fl = torch.zeros(RM, ns, 24, 128), torch.zeros(RM, ns, 24)
        DB._bfar_kernel[(8, ns, 2)](qb, c[0], c[1], s[0], s[1], tbl, tbl.shape[1], meta, fo, fl, ns, 128 ** -0.5,
                                     SPLIT=4096, PG=kvpool.PAGE, BLK=D.DOT_BLK, F32=True, RM=RM, NSP=nsp)
        for i, p in enumerate(pos):                          # logical store of slot i through its page table
            r = (tbl[i][:, None] * kvpool.PAGE + torch.arange(kvpool.PAGE)).reshape(-1)
            ro, _ = far_ref(qb[i].view(24, 128), c[:, r], s[:, r], p - 1023)
            err = ((merge(fo[i], fl[i])[0].double() - ro).abs().max() / ro.abs().max()).item()
            assert err < 2e-2, (name, i, err)
        pm, pl, pacc, plse = torch.zeros(RM, nt, 24), torch.zeros(RM, nt, 24), torch.zeros(RM, nt, 24, 128), torch.zeros(RM, 24)
        att = torch.zeros(RM, 3072, dtype=torch.bfloat16)
        DB._bnear_part_kernel[(2, 8, nt)](qb, rb, proj, rk, rpos, meta, pm, pl, F32=True, BLK=bl, WR=WR, RM=RM)
        DB._bnear_out_kernel[(2, 8, nt)](qb, rb, proj, Y, rk, rv, rpos, meta, pm, pl, fl, ns, pacc, plse, FAR=True,
                                         F32=True, BLK=bl, WR=WR, RM=RM, NTP=triton.next_power_of_2(nt), NSF=nsf)
        DB._bnear_fin_kernel[(24, 2)](pacc, plse, fo, fl, ns, Y, att, meta, FAR=True, F32=True, NT=nt, RM=RM, NSF=nsf)
        out[name] = plse[:2].clone(), att[:2].float()
    for i in range(2):
        dl = (out['old'][0][i] - out['far2'][0][i]).abs().max().item()
        da = ((out['old'][1][i] - out['far2'][1][i]).abs().max() / out['old'][1][i].abs().max()).item()
        print(f'bfar2 row {i} (pos {pos[i]}, slot {i}): far vs fp64 OK; merged lse diff {dl:.1e}, attention out '
              f'rel {da:.1e} vs old path', flush=True)
        assert dl < 1e-5 and da < 1.6e-2


def test_bfar_verify():
    """Long-context spec enabler: a verify group shares ONE far read (MT 32 tile, <= 10 rows per group); far_groups splits
    a group where the FAR2 split length steps (kend 8960 -> 8961 at nsp 7), so each row's splits are its 1-row step's.
    Each row's far partials vs that row alone in a 1-row step (MT 16): on the GPU (TRITON_INTERPRET=0, window) bit for
    bit (greedy spec == plain at long context); in the CPU interpreter within 5e-5 rel (numpy matmul rounds differently
    per M shape: the old 16-row tile shows the same ulp-level differences between a group of 5 and its single rows)."""
    gpu = os.environ.get('TRITON_INTERPRET') == '0' and torch.cuda.is_available()
    dev = torch.device('cuda' if gpu else 'cpu')
    g = torch.Generator().manual_seed(6)
    RM, nsp, cap = DB.RM, 7, 3 * 4096
    c, s, tbl0 = far_store(cap, g)
    c, s, tbl = c.to(dev), s.to(dev), tbl0[None].to(dev)
    qb = (torch.randn(RM, 3072, generator=g) * 2).bfloat16().to(dev)
    assert DB.mt(1) == DB.mt(4) == 16 and DB.mt(8) == DB.mt(16) == 32

    def run(q, poss, groups, R):
        meta = torch.zeros(8, RM, dtype=torch.int64)
        meta[1, :len(poss)], meta[3, :len(poss)] = torch.tensor(poss), 1
        for j, (a, n) in enumerate(groups):
            meta[4, j], meta[5, j] = a, n
        o, l = torch.zeros(RM, nsp, 24, 128, device=dev), torch.zeros(RM, nsp, 24, device=dev)
        DB._bfar_kernel[(8, nsp, len(groups))](q, c[0], c[1], s[0], s[1], tbl, tbl.shape[1], meta.to(dev), o, l, nsp,
                                               128 ** -0.5, SPLIT=4096, PG=kvpool.PAGE, BLK=D.DOT_BLK, F32=not gpu,
                                               RM=RM, NSP=nsp, MT=DB.mt(R), **(D.FAR_KW if gpu else {}))
        return o.cpu(), l.cpu()
    worst = 0.
    for kends, want in ((range(5000, 5010), [(0, 10)]), (range(8955, 8965), [(0, 6), (6, 4)]),
                        (range(3000, 3012), [(0, 10), (10, 2)])):
        poss = [k + 1023 for k in kends]
        groups = DB.far_groups(poss, nsp, DB.mt(16) // 3)
        assert groups == want, (groups, want)
        o, l = run(qb, poss, groups, 16)
        for i, p in enumerate(poss):                       # row i alone: its q in row 0 of a 1-row step
            q1 = qb.clone()
            q1[0] = qb[i]
            o1, l1 = run(q1, [p], [(0, 1)], 1)
            if gpu:
                assert torch.equal(o[i], o1[0]) and torch.equal(l[i], l1[0]), (kends[0], i)
            else:
                d = max(((o[i] - o1[0]).abs().max() / o1[0].abs().max()).item(), (l[i] - l1[0]).abs().max().item())
                worst = max(worst, d)
                assert d < 5e-5, (kends[0], i, d)
            ro, _ = far_ref(qb[i].cpu().view(24, 128), c.cpu(), s.cpu(), p - 1023)
            err = ((merge(o[i], l[i])[0].double() - ro).abs().max() / ro.abs().max()).item()
            assert err < 2e-2, err
        print(f'bfar verify kend {kends[0]}..{kends[-1]}: groups {groups} (far reads per step {len(groups)}, was '
              f'{-(-len(poss) // 5)}), rows == 1-row steps {"bit for bit (GPU)" if gpu else f"within {worst:.1e} (CPU)"}',
              flush=True)


# --------------------------------------------------------------------------------------------- real weights
def test_w2all_real():
    if os.environ.get('INK_REAL') != '1':
        print('w2all real-weights check skipped (INK_REAL=1)')
        return
    from test_decode import fused_model
    _, m = fused_model()
    dec = D.Decoder(m, graph=False, fuse=True)
    F = K.Fused(m, dec.st, dec.x_raw, dec.out)
    for v in (6, 44):
        d = F.vis[v]
        w13, s13, w2, s2 = d['ex']
        test_w2all({d['fp4']: (w2, s2, d['gs2']), not d['fp4']: fake_experts(not d['fp4'], torch.Generator()),
                    'real': (d['fp4'],)},
                   label=f'layer {d["L"]} real')


def prefilled(m, P):
    g = torch.Generator().manual_seed(3)
    ids = torch.randint(1000, 150000, (P,), generator=g).tolist()
    moe = m.moe
    m.moe = lambda L, u, impl=None: (u.float() * 0.5).bfloat16()
    seq = m.new_seq()
    t0 = time.time()
    for a in range(0, P, 1024):
        m.forward(m.embed(ids[a:a + 1024]), seq)
    m.moe = moe
    print(f'prefill {P} (stub MoE) {time.time() - t0:.0f} s', flush=True)
    return seq


def test_flags_step():
    if os.environ.get('INK_REAL') != '1':
        print('whole-step flags check skipped (INK_REAL=1, ~20 min)')
        return
    from test_decode import fused_model, gemv_mirrors, swap
    _, m = fused_model()
    K.BLK.update(w2a=128)
    P, steps = 2100, 3
    seq = prefilled(m, P)
    decs = {}
    for name, on in (('off', False), ('nud', False), ('on', True)):
        flags(on, nsp=8)
        s2 = M.Seq('cpu', m.pool.fork(seq.kv, P))
        s2.pos, s2.conv, s2.ring = seq.pos, [{k: v.clone() for k, v in c.items()} for c in seq.conv], seq.ring
        assert m.pool.reserve(s2.kv, P + steps + 1)
        d = D.Decoder(m, graph=False, fuse=True)
        d.load(s2)
        d.fused = K.Fused(m, d.st, d.x_raw, d.out)
        decs[name] = d
    flags(False)
    assert decs['on'].fused.w2all and decs['on'].fused.nsp == 8 and not decs['off'].fused.nsp
    ref = decs['off']
    kld = lambda x, y: (torch.log_softmax(x, -1).exp() * (torch.log_softmax(x, -1) - torch.log_softmax(y, -1))).sum().item()
    toks = torch.randint(1000, 150000, (steps,), generator=torch.Generator().manual_seed(8)).tolist()
    for i, t in enumerate(toks):
        for nm in ('nud', 'on'):                                   # lockstep: every step from the reference state
            d = decs[nm]
            for a_, b_ in zip(d.st.conv, ref.st.conv):
                for k in a_:
                    a_[k].copy_(b_[k])
            for (a_, b_), (c_, d_) in zip(d.st.ring, ref.st.ring):
                a_.copy_(c_), b_.copy_(d_)
            d.st.rpos.copy_(ref.st.rpos)
            d.seq.pos = ref.seq.pos
        out = {}
        for nm in ('nud', 'off', 'on'):
            d = decs[nm]
            row = m.W.embed[t].clone()
            if nm == 'nud':
                m.W.embed[t] = (row.float() * (1 + 2 ** -7)).bfloat16()
            restore = swap(K, gemv_mirrors(m, d.fused))
            try:
                t0 = time.time()
                out[nm] = d.step(t).clone()
            finally:
                restore()
                m.W.embed[t] = row
            print(f'  {nm} step {time.time() - t0:.0f} s', flush=True)
        a, b, n_ = out['off'], out['on'], out['nud']
        kl, kn = kld(a, b), kld(a, n_)
        ia, ib = int(a.argmax()), int(b.argmax())
        gap = (a[0, ia] - a[0, ib]).item()
        print(f'flags on vs off, pos {P + i}: KL {kl:.2e} (1-ulp nudge {kn:.2e}), max |dlogit| {(a - b).abs().max():.4f}, '
              f'top-1 {ia} vs {ib} (gap {gap:.3f})', flush=True)
        assert kl <= 2 * kn + 0.05 and gap <= 0.25 and bool(torch.isfinite(b).all()), (i, kl, kn, gap)
    print('whole fused step, all flags on vs off: OK')


def test_flags_batch():
    if os.environ.get('INK_REAL') != '1':
        print('batched flags check skipped (INK_REAL=1, ~30 min)')
        return
    import test_decode_batch as TB
    m, src = TB.setup()
    outs = []
    for on in (False, True):
        flags(on, nsp=4)
        bd = TB.new_bd(m, 'ABD')
        flags(False)
        P = {s: bd.seqs[s].pos for s in range(3)}
        rows = [(0, P[0], 11), (1, P[1], 12), (2, P[2], 13), (2, P[2] + 1, 14), (2, P[2] + 2, 15)]
        lg, dt = TB.run_bd(bd, rows)
        bd.commit()
        outs.append((lg, TB.snap(bd)))
        print(f'  batched step (flags {"on" if on else "off"}) {dt:.0f} s', flush=True)
    (a, sa), (b, sb) = outs
    kl = max(TB.kld(a[i], b[i]) for i in range(len(a)))
    same = torch.equal(a, b)
    print(f'batched step all flags on vs off: logits bit-identical {same}, worst KL {kl:.1e}, top-1 equal '
          f'{bool((a.argmax(-1) == b.argmax(-1)).all())}, state equal {TB.same_state(sa[:3], sb[:3])}', flush=True)
    assert kl < 1e-3 and bool((a.argmax(-1) == b.argmax(-1)).all())


if __name__ == '__main__':
    names = [a for a in sys.argv[1:] if not a.startswith('-')]
    for name, fn in list(globals().items()):
        if name.startswith('test_') and (not names or name in names):
            fn()
    print('DECODE2 ALL OK' if not names else 'OK')
