"""CPU checks for the batched decoder (decode_batch.py). No GPU.
  --compile   offline sm_120 compile of every decode_batch kernel (no interpreter; shared memory <= 101376 B)
  (default)   interpreter tests, real weights: test_api, test_step_tok (s), test_gemm_kernels (~25 min), test_batch_vs_fused (~60 min),
              test_verify_commit (~75 min), test_invariance_pads (~55 min), test_capture_safe (s). Name tests as args to
              run a subset; run them as parallel processes (~7 GB RAM each). The first run prefills 4 states with a
              stub MoE (~5 min) and caches them in INK_TDB_CACHE (/tmp/ink-test-decode-batch.pt).
Whole-step tests run torch mirrors for the three big GEMMs (test_gemm_kernels checks the real kernels against them)
and use WR 1152 / near tile 128 (fewer interpreter programs; the GPU config 1088 / 64 is compiled by --compile).
Run: cd engine && PATH=$PWD/.venv/bin:$PATH python tests/test_decode_batch.py [--compile | test_name ...]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

if '--compile' in sys.argv:
    import compile_sm120 as C  # noqa: E402  (sets CUDA_VISIBLE_DEVICES='' and drops TRITON_INTERPRET first)
    import decode_batch as DB  # noqa: E402

    def cases():
        F, B, I32, I64, U8, F8 = C.F, C.B, C.I32, C.I64, C.U8, C.F8
        c = dict(F32=False, H=C.H, CH=C.CH)
        K, bb = C.K, DB.BB
        mix = dict(q_ptr=F, B=F, sB='i32', eidx=I32, n='i32', x_ptr=B, sx='i32', bslot='i32', a_ptr=F, sa='i32',
                   xo_ptr=B, sxo='i32', nwt=I64, qm_ptr=F, midx=I32, nm='i32', xm_ptr=B, sxm='i32', m_ptr=F, sm='i32',
                   nq='i32')
        for first, norm, sa in ((True, True, False), (False, True, False), (False, False, True)):
            yield f'bmix first={first} norm={norm}', DB._bmix_kernel, mix, dict(c, FIRST=first, NORM=norm, STORE_A=sa)
        yield 'bmix (mlp, fp32 x)', DB._bmix_kernel, dict(mix, x_ptr=F, xm_ptr=F), dict(c, FIRST=False, NORM=True,
                                                                                         STORE_A=False)
        yield 'bgate_z', DB._bgate_z_kernel, dict(a_ptr=F, sa='i32', xs='i32', m_ptr=F, sm='i32', dtab=I64, z_ptr=F,
                                                  sz='i32'), dict(c, ZB=K.BLK['gate']), K.NW['gate']
        yield 'bgate_out', DB._bgate_out_kernel, dict(a_ptr=F, sa='i32', xs='i32', m_ptr=F, sm='i32', gtab=I64, z_ptr=F,
                                                      sz='i32', o_ptr=F, so='i32'), dict(H=C.H, BH=K.BLK['gout']), \
            K.NW['gout']
        for norm in (False, True):
            yield f'brouter norm={norm}', DB._brouter_kernel, dict(x_ptr=F, sx='i32', nw=F, u_ptr=B, gw=F, s_ptr=F), \
                dict(c, NORM=norm, RB=K.BLK['router']), K.NW['router']
        yield 'bnorm', DB._bnorm_kernel, dict(G=F, nwt=I64, X=B), c
        for w8 in (False, True):
            yield f'bproj w8={w8}', DB._bproj_kernel, dict(w_ptr=F8 if w8 else B, n0='i32', n1='i32', N='i32', x_ptr=B,
                                                           xs='i32', sx='i32', y_ptr=B, nr='i32', s_ptr=F), \
                dict(c, BN=bb['proj'], RM=DB.RM, W8=w8), DB.NWB['proj']
            for bnd in (False, True):
                yield f'bwo bnd={bnd} w8={w8}', DB._bwo_kernel, dict(x_ptr=B, w_ptr=F8 if w8 else B, cst=B, sst='i32',
                                                                      xh=B, cw=F, h_ptr=B, r_ptr=F, meta=I64, nr='i32',
                                                                      s_ptr=F), \
                    dict(c, BND=bnd, BN=bb['wo'], RM=DB.RM, CC=DB.CC, W8=w8), DB.NWB['wo']
        pre = dict(Y=B, koff='i32', voff='i32', tkoff='i32', tvoff='i32', qsc=F, rsc=F, kw=F, vw=F, cst=B, sst='i32', xh=B,
                   rk=B, rv=B, rpos=I64, meta=I64, c0=U8, c1=U8, s0=U8, s1=U8, tbl=I64, ts='i32', qo=B, ro=B)
        for full in (False, True):
            yield f'bpre full={full}', DB._bpre_kernel, pre, dict(FULL=full, F32=False, PG=256, WR=DB.WR, RM=DB.RM,
                                                                  CC=DB.CC)
        yield 'bfar', DB._bfar_kernel, dict(qb=B, c0=U8, c1=U8, s0=F8, s1=F8, tbl=I64, ts='i32', meta=I64, o_ptr=F,
                                            l_ptr=F, ns='i32', scale='fp32'), \
            dict(SPLIT=4096, PG=256, BLK=DB.DOT_BLK, F32=False, RM=DB.RM)
        yield f'bfar FAR2 nsp={DB.D.FAR_NSP}', DB._bfar_kernel, dict(qb=B, c0=U8, c1=U8, s0=F8, s1=F8, tbl=I64, ts='i32',
                                                                    meta=I64, o_ptr=F, l_ptr=F, ns='i32', scale='fp32'), \
            dict(SPLIT=4096, PG=256, BLK=DB.DOT_BLK, F32=False, RM=DB.RM, NSP=DB.D.FAR_NSP), DB.D.FAR_KW['num_warps']
        for R in (8,):  # verify groups of up to 10 rows in one M tile (the far KV read once per slot)
            yield f'bfar FAR2 MT={DB.mt(R)}', DB._bfar_kernel, dict(qb=B, c0=U8, c1=U8, s0=F8, s1=F8, tbl=I64, ts='i32',
                                                                 meta=I64, o_ptr=F, l_ptr=F, ns='i32', scale='fp32'), \
                dict(SPLIT=4096, PG=256, BLK=DB.DOT_BLK, F32=False, RM=DB.RM, NSP=DB.D.FAR_NSP, MT=DB.mt(R)), \
                DB.D.FAR_KW['num_warps']
        nt = DB.WR // bb['near']
        yield 'bnear_part', DB._bnear_part_kernel, dict(qb=B, rb=B, proj=B, rk=B, rpos=I64, meta=I64, pm=F, pl=F), \
            dict(F32=False, BLK=bb['near'], WR=DB.WR, RM=DB.RM)
        for far in (False, True):
            yield f'bnear_out far={far}', DB._bnear_out_kernel, dict(qb=B, rb=B, proj=B, Y=B, rk=B, rv=B, rpos=I64,
                                                                     meta=I64, pm=F, pl=F, fl=F, ns='i32', pacc=F,
                                                                     plse=F), \
                dict(FAR=far, F32=False, BLK=bb['near'], WR=DB.WR, RM=DB.RM, NTP=1 << (nt - 1).bit_length())
            yield f'bnear_fin far={far}', DB._bnear_fin_kernel, dict(pacc=F, plse=F, fo=F, fl=F, ns='i32', Y=B, att=B,
                                                                     meta=I64), \
                dict(FAR=far, F32=False, NT=nt, RM=DB.RM)
            yield f'bnear_out FAR2 far={far}', DB._bnear_out_kernel, dict(qb=B, rb=B, proj=B, Y=B, rk=B, rv=B, rpos=I64,
                                                                          meta=I64, pm=F, pl=F, fl=F, ns='i32', pacc=F,
                                                                          plse=F), \
                dict(FAR=far, F32=False, BLK=bb['near'], WR=DB.WR, RM=DB.RM, NTP=1 << (nt - 1).bit_length(), NSF=64)
            yield f'bnear_fin FAR2 far={far}', DB._bnear_fin_kernel, dict(pacc=F, plse=F, fo=F, fl=F, ns='i32', Y=B,
                                                                          att=B, meta=I64), \
                dict(FAR=far, F32=False, NT=nt, RM=DB.RM, NSF=64)
        yield 'plan', DB._plan_kernel, dict(S=F, bias=F, meta=I64, nr='i32', ids=I32, wts=F, eslot=I32, ulist=I32,
                                            ucnt=I32), dict(RM=DB.RM)
        moe = dict(lwe='i64', lwr='i32', lse='i64', lsr='i32')
        for fp4 in (True, False):
            w, s = (U8, F8) if fp4 else (F8, U8)
            yield f'w13u fp4={fp4}', DB._w13u_kernel, dict(moe, U=B, ulist=I32, ucnt=I32, eslot=I32, w_ptr=w, s_ptr=s,
                                                          gs_ptr=F, act=B), \
                dict(FP4=fp4, F32=False, H=C.H, NI=DB.NI, BN=bb['w13'], BK=bb['kbk'], RM=DB.RM), DB.NWB['w13']
            yield f'w13u R=1 fp4={fp4}', DB._w13u_kernel, dict(moe, U=B, ulist=I32, ucnt=I32, eslot=I32, w_ptr=w,
                                                                s_ptr=s, gs_ptr=F, act=B), \
                dict(FP4=fp4, F32=False, H=C.H, NI=DB.NI, BN=bb['w13'], BK=DB.W13_R1['BK'], RM=DB.RM), DB.NWB['w13']
            yield f'w2u fp4={fp4}', DB._w2u_kernel, dict(moe, act=B, ulist=I32, ucnt=I32, eslot=I32, wts=F, w_ptr=w,
                                                        s_ptr=s, gs_ptr=F, part=F), \
                dict(FP4=fp4, F32=False, H=C.H, NI=DB.NI, BN=bb['w2'], BK=bb['kbk2'], RM=DB.RM), DB.NWB['w2']
        yield 'bw2_fin', DB._bw2_fin_kernel, dict(part=F, cst=B, sst='i32', xh=B, cw=F, r_ptr=F, h_out=B, meta=I64,
                                                  nr='i32'), dict(F32=False, H=C.H, BN=bb['w2f'], RM=DB.RM, CC=DB.CC)
        yield 'commit', DB._commit_kernel, dict(conv=B, sv='i32', sst='i32', xh=B, meta=I64, crow0=I64, cn=I64), \
            dict(RM=DB.RM, CC=DB.CC, CB=bb['cb'])

    C.cases = cases
    sys.exit(C.main())

# ------------------------------------------------------------------------------------------------ interpreter tests
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('TRITON_INTERPRET', '1')
import time  # noqa: E402

import torch  # noqa: E402

import decode as D  # noqa: E402
import decode_batch as DB  # noqa: E402
import decode_kernels as K  # noqa: E402
import model as M  # noqa: E402
import moe_triton  # noqa: E402
from test_decode import gemv_mirrors, launch_recorder, op_trace, swap  # noqa: E402

H, RM = M.H, DB.RM
ONLY = [a for a in sys.argv[1:] if not a.startswith('-')]


class Grid:
    def __init__(self, fn):
        self.fn = fn

    def __getitem__(self, grid):
        return self.fn


def batch_mirrors(m, bd):
    """Row-wise torch mirrors of the three big batched GEMM kernels (proj, w13u, w2u) with their call signatures; the
    interpreter needs minutes per launch for them, so whole-step tests run these (test_gemm_kernels checks the real
    kernels against them). Everything else runs the real kernels (plan, pre, far, near, wo, w2_fin, commit, ...)."""
    layer = {d['ex'][j].data_ptr(): d['L'] for d in bd.F.vis for j in (0, 2)}
    w2c = {}

    def proj(w, n0, n1, N, x, xs, sx, y, nr, s_ptr, W8=False, **k):
        X = x.reshape(-1)
        rows = ((0, n0), (n0, n0 + n1), (n0 + n1, N))
        sc = s_ptr if W8 else torch.ones(N)
        for r in range(nr):
            xx = [X[r * sx + j * xs: r * sx + j * xs + H] for j in range(3)]
            y[r, :N] = torch.cat([(w[a:b].float() @ xj.float()) * sc[a:b] for (a, b), xj in zip(rows, xx)])

    def w13(U, ulist, ucnt, eslot, w, s, gs, a1, a2, a3, a4, act, **k):
        L = layer[w.data_ptr()]
        w2c.clear()
        for u in range(int(ucnt[0])):
            e = int(ulist[u])
            w13d, w2c[e] = m.W.expert(L, e)
            for r in range(RM):
                sl = int(eslot[e, r])
                if sl >= 0:
                    gu = (U[r].float() @ w13d.float().t()).bfloat16()
                    g, up = gu[0::2].float(), gu[1::2].float()
                    act[r, sl] = (4 * moe_triton.tanh(g / 4) * torch.sigmoid(g) * 25 * moe_triton.tanh(up / 25)).bfloat16()

    def w2(act, ulist, ucnt, eslot, wts, w, s, gs, a1, a2, a3, a4, part, **k):
        for u in range(int(ucnt[0])):
            e = int(ulist[u])
            for r in range(RM):
                sl = int(eslot[e, r])
                if sl >= 0:
                    part[r, sl] = (act[r, sl].float() @ w2c[e].float().t()).bfloat16().float() * wts[r, sl]
    return {'_bproj_kernel': Grid(proj), '_w13u_kernel': Grid(w13), '_w2u_kernel': Grid(w2)}


_M = {}


def setup():
    """Model (CPU, real weights, 16k pool) + prefilled source sequences (stub MoE: fast; any valid state will do):
    A at 1020 (decode crosses 1024: ring full, far keys start, page boundary), B at 254 (page boundary at 256),
    C at 3 (conv states partly from the prompt), D at 40."""
    if _M:
        return _M['m'], _M['src']
    from engine import MODEL_DIR
    # fewer, bigger programs: the interpreter is serial (~16 ms per program). WR 1152 / near tile 128 here (9 ring
    # tiles instead of 17; same code, the GPU keeps 1088 / 64: its near_out would exceed shared memory at 128)
    DB.BB.update(wo=256, w2f=256, cb=512, near=128)
    DB.WR = 1152
    K.BLK.update(wo=128, router=128, w13=128, w2=128, gate=64, gout=256)
    m = M.Model(MODEL_DIR, 'cpu', max_ctx=16384)
    src, pool = {}, m.pool
    cache = os.environ.get('INK_TDB_CACHE', '/tmp/ink-test-decode-batch.pt')   # prefill states (raw pages + state)
    if os.path.exists(cache):
        for name, (P, conv, ring, c, s) in torch.load(cache).items():
            seq = m.new_seq()
            assert pool.reserve(seq.kv, P)
            pool.c[:, :, seq.kv.pages], pool.s[:, :, seq.kv.pages] = c, s
            seq.pos, seq.conv, seq.ring = P, conv, ring
            src[name] = seq
        print(f'prefill states from {cache}', flush=True)
    else:
        moe = m.moe
        m.moe = lambda L, u, impl=None: (u.float() * 0.5).bfloat16()
        g = torch.Generator().manual_seed(5)
        t0 = time.time()
        for name, P in (('A', 1020), ('B', 254), ('C', 3), ('D', 40)):
            seq = m.new_seq()
            m.forward(m.embed(torch.randint(1000, 150000, (P,), generator=g).tolist()), seq)
            src[name] = seq
        m.moe = moe
        print(f'prefill A/B/C/D (stub MoE) {time.time() - t0:.0f} s', flush=True)
        torch.save({n: (q.pos, q.conv, q.ring, pool.c[:, :, q.kv.pages], pool.s[:, :, q.kv.pages])
                    for n, q in src.items()}, cache + '.tmp')
        os.replace(cache + '.tmp', cache)
    _M.update(m=m, src=src)
    return m, src


def clone(m, seq, extra=24):
    """Independent copy of a prefilled Seq (forked page table: shared full pages, private partial page)."""
    s = M.Seq('cpu', m.pool.fork(seq.kv, seq.pos))
    s.pos = seq.pos
    s.conv = [{k: v.clone() for k, v in d.items()} for d in seq.conv]
    s.ring = list(seq.ring)
    assert m.pool.reserve(s.kv, seq.pos + extra)
    return s


def new_bd(m, names, slots=4):
    _, src = setup()
    bd = DB.BatchDecoder(m, slots, graph=False)
    for i, n in enumerate(names):
        bd.load(i, clone(m, src[n]))
    return bd


def snap(bd):
    b = bd.b
    return b.ring.clone(), b.rpos.clone(), b.conv.clone(), [s.pos if s else None for s in bd.seqs]


def rewind(bd, s):
    b = bd.b
    b.ring.copy_(s[0]), b.rpos.copy_(s[1]), b.conv.copy_(s[2])
    for q, p in zip(bd.seqs, s[3]):
        if q is not None:
            q.pos = p


def slot_state(bd, slot):
    b = bd.b
    return b.ring[:, :, slot].clone(), b.rpos[slot].clone(), b.conv[:, slot].clone()


def same_state(x, y):
    return all(torch.equal(a, b) for a, b in zip(x, y))


TOK = torch.randint(1000, 150000, (64,), generator=torch.Generator().manual_seed(9)).tolist()
kld = lambda x, y: (torch.log_softmax(x, -1).exp() * (torch.log_softmax(x, -1) - torch.log_softmax(y, -1))).sum().item()


def test_commit_kernel():
    """commit kernel alone (no model): new conv state = last 3 of [state ; accepted rows' inputs], n = 1, 2, 3, 5,
    groups at any row offset and slot; other slots and visits untouched."""
    V, n, CC, CB = 2, 3, DB.CC, DB.BB['cb']
    conv = torch.randn(V, n, 3, CC).bfloat16()
    xh = torch.randn(V, RM, CC).bfloat16()
    meta = torch.zeros(8, RM, dtype=torch.int64)
    groups = [(0, 2, 1), (1, 0, 5), (6, 1, 2), (8, 2, 3)]          # (row0, slot, n): slot 2 twice = two calls below
    for calls in (groups[:3], groups[3:]):
        ref = conv.clone()
        c0, cn = torch.zeros(RM, dtype=torch.int64), torch.zeros(RM, dtype=torch.int64)
        for g, (r0, sl, k) in enumerate(calls):
            meta[0, r0:r0 + k] = sl
            c0[g], cn[g] = r0, k
            for v in range(V):
                ref[v, sl] = torch.cat((conv[v, sl], xh[v, r0:r0 + k]))[-3:]
        DB._commit_kernel[(V, RM, CC // CB)](conv, conv.stride(0), 3 * CC, xh, meta, c0, cn, RM=RM, CC=CC, CB=CB)
        assert torch.equal(conv, ref)
    print('commit kernel: n = 1, 2, 3, 5 == last 3 of [state ; rows] OK')


def test_api():
    """API surface (LoopBatch's contract) without running a step."""
    m, src = setup()
    bd = DB.BatchDecoder(m, 3, graph=False)
    assert bd.n == 3 and bd.dec and not bd.busy() and bd.idle() == 0 and bd.decs == [None] * 3
    bd.load(1, clone(m, src['C']))
    assert bd.busy() and bd.decs[1] is bd and bd.drop() == 0          # busy: keeps its buffers
    per, shared = bd.nbytes()
    print(f'memory: {per / 2**20:.1f} MiB per slot (rings {DB.WR}-slot x 48 visits + conv + rpos + page table at a '
          f'16k pool), {shared / 2**20:.1f} MiB shared scratch (far partials scale with the pool: at 1M add '
          f'{16 * 256 * 24 * 129 * 4 / 2**20:.0f} MiB)')
    st = bd.state(1)
    ref = src['C']
    assert all(torch.equal(st[f'conv.{v}.{k}'], ref.conv[v][k]) for v in range(48) for k in ref.conv[v])
    assert all(torch.equal(st[f'ring.{v}.k'], ref.ring[v][0]) and torch.equal(st[f'ring.{v}.v'], ref.ring[v][1])
               for v in range(48))
    bd.free(1)
    assert bd.na == 2 and bd.decs[2] is None                         # lazy: slots 0..1 only (first load at slot 1)
    assert not bd.busy() and bd.idle() == 2 and bd.drop(keep=1) == 0 and bd.drop() == 2 and bd.b is None
    print('api: load/free/state/drop/busy/idle/decs OK, state() == the loaded Seq bit for bit')


def test_step_tok():
    """step(rows, tok={r0: device ids}) == step(rows) on the host side: same meta, same row tokens / embedding rows
    (host table, and INK_DEC_EMBED_GPU's device table + in-step lookup); host meta through the alternating (pinned on
    cuda) buffers. The step itself is not run (_replay recorded)."""
    m, src = setup()
    PA, PB = src['A'].pos, src['B'].pos
    rows = [(0, PA, TOK[0]), (0, PA + 1, TOK[1]), (0, PA + 2, TOK[2]), (1, PB, TOK[3])]
    dev_rows = [(0, PA, TOK[0]), (0, PA + 1, None), (0, PA + 2, None), (1, PB, TOK[3])]
    for emb in (False, True):
        bd = new_bd(m, 'AB', slots=2)
        if emb:
            bd.embed, bd.dev_tok = m.W.embed, True
        rec = []
        bd._replay = lambda R: rec.append((R, bd.b.meta.clone(), bd.b.tok.clone(), bd.b.x_raw[:4].clone()))

        def run(rw, tok=None):
            bd.step(rw, tok=tok)
            bd.pend.clear()                                           # (nothing ran: undo the bookkeeping)
            bd.seqs[1].pos -= 1
        run(rows)
        run(dev_rows, {1: torch.tensor(TOK[1:3])})
        run([(s, p, None) for s, p, _ in rows], {0: torch.tensor(TOK[:4])})
        assert len(rec) == 3 and all(r[0] == 4 for r in rec)
        for _, meta, tk, xr in rec[1:]:
            assert torch.equal(meta, rec[0][1]), 'meta differs'
            if emb:
                assert torch.equal(tk[:4], torch.tensor(TOK[:4])) and torch.equal(tk, rec[0][2])
            else:
                assert torch.equal(xr, rec[0][3]) and torch.equal(xr, m.W.embed[torch.tensor(TOK[:4])])
        if emb:                                                       # the in-step lookup (_run's first op)
            x = torch.index_select(bd.embed, 0, rec[0][2])[:4]
            assert torch.equal(x, m.W.embed[torch.tensor(TOK[:4])])
        assert bd.hi in (0, 1) and len(bd.hm) == 2
    print('step(rows, tok=device ids) == step(rows): meta + tokens / embedding rows (host table and device lookup)')


def test_gemm_kernels():
    """The real batched GEMM kernels (interpreter) == the torch mirrors, and rows are independent of each other
    (bit-identical when other rows change, are removed or the row moves). proj on a full visit (3 inputs, k/v taps,
    blocks straddling input boundaries) and a gated core visit; w13u + w2u on an NVFP4 and an MXFP8 layer."""
    m, _ = setup()
    bd = DB.BatchDecoder(m, 1, graph=False)
    bd._alloc(bd.na)
    b, F = bd.b, bd.F
    mir = batch_mirrors(m, bd)
    g = torch.Generator().manual_seed(1)
    for v in (5, 13):
        d = F.vis[v]
        N, xs = d['N'], H
        b.X.copy_((torch.randn(RM, 3, H, generator=g)).bfloat16())
        ys = {}
        for name, fn in (('kernel', DB._bproj_kernel[(-(-N // DB.BB['proj']),)]), ('mirror', mir['_bproj_kernel'][0])):
            b.Y.zero_()
            fn(d['cat'], d['n0'], d['n1'], N, b.X, xs, 3 * H, b.Y, 3, d['cs'], F32=True, H=H, CH=K.CH, BN=DB.BB['proj'],
               RM=RM, W8=F.w8)
            ys[name] = b.Y[:3, :N].float().clone()
        rel = ((ys['kernel'] - ys['mirror']).abs().max() / ys['mirror'].abs().max()).item()
        x2 = b.X[2].clone()
        b.X.copy_(torch.randn(RM, 3, H, generator=g).bfloat16())
        b.X[0] = x2                                               # row 2 alone at row 0, other rows changed
        b.Y.zero_()
        DB._bproj_kernel[(-(-N // DB.BB['proj']),)](d['cat'], d['n0'], d['n1'], N, b.X, xs, 3 * H, b.Y, 1, d['cs'],
                                                    F32=True, H=H, CH=K.CH, BN=DB.BB['proj'], RM=RM, W8=F.w8)
        inv = torch.equal(b.Y[0, :N].float(), ys['kernel'][2])
        print(f'bproj visit {v} (N {N}, n0 {d["n0"]} n1 {d["n1"]}): kernel vs mirror max rel {rel:.1e}, row moved '
              f'0<-2 with other rows changed: bit-identical {inv}', flush=True)
        assert rel < 1e-2 and inv, (rel, inv)
    meta = torch.zeros(8, RM, dtype=torch.int64)
    for v in (2, 44):                                             # layer 2 NVFP4, layer 20 MXFP8
        d = F.vis[v]
        w13, s13, w2, s2 = d['ex']
        b.U.copy_((torch.randn(RM, H, generator=g) * 0.5).bfloat16())
        b.S.copy_(torch.rand(RM, 128, generator=g))
        outs = {}
        for name, nr, live in (('kernel', 2, (1, 1)), ('mirror', 2, (1, 1)), ('alone', 1, (0, 1))):
            meta[3].zero_()
            meta[3, :len(live)] = torch.tensor(live)
            b.meta.copy_(meta)
            b.act.zero_(), b.part.zero_()
            DB._plan_kernel[(1,)](b.S, d['gb'], b.meta, nr if name != 'alone' else 2, b.ids, b.wts, b.eslot, b.ulist,
                                  b.ucnt, RM=RM)
            k13 = mir['_w13u_kernel'][0] if name == 'mirror' else DB._w13u_kernel[(K.NI // DB.BB['w13'], 16)]
            k2 = mir['_w2u_kernel'][0] if name == 'mirror' else DB._w2u_kernel[(H // DB.BB['w2'], 16)]
            k13(b.U, b.ulist, b.ucnt, b.eslot, w13, s13, d['gs13'], w13.stride(0), w13.stride(1), s13.stride(0),
                s13.stride(1), b.act, FP4=d['fp4'], F32=True, H=H, NI=K.NI, BN=DB.BB['w13'], BK=DB.BB['kbk'], RM=RM)
            k2(b.act, b.ulist, b.ucnt, b.eslot, b.wts, w2, s2, d['gs2'], w2.stride(0), w2.stride(1), s2.stride(0),
               s2.stride(1), b.part, FP4=d['fp4'], F32=True, H=H, NI=K.NI, BN=DB.BB['w2'], BK=DB.BB['kbk2'], RM=RM)
            outs[name] = (b.part[:2].sum(1).clone(), int(b.ucnt[0]), b.ids[:2].clone())
        # reference top-8 per row (Fused's rule) and union size
        sc = b.S[:2] + d['gb']
        ref_ids = torch.stack([torch.tensor(sorted(r.topk(8).indices.tolist(), key=lambda e: -r[e].item())) for r in sc])
        assert torch.equal(outs['kernel'][2].long(), ref_ids), (outs['kernel'][2], ref_ids)
        u = len(set(ref_ids.flatten().tolist()))
        a, mm = outs['kernel'][0], outs['mirror'][0]
        rel = ((a - mm).abs().max() / mm.abs().max()).item()
        inv = torch.equal(outs['alone'][0][1], a[1])
        print(f'w13u+w2u visit {v} ({"NVFP4" if d["fp4"] else "MXFP8"}): union {outs["kernel"][1]} (expected {u}, '
              f'alone {outs["alone"][1]}), kernel vs mirror max rel {rel:.1e}, row 1 alone == with row 0: {inv}',
              flush=True)
        assert outs['kernel'][1] == u and outs['alone'][1] == 8 and rel < 2e-2 and inv
    bd.drop()


def run_bd(bd, rows, mirrors=True):
    restore = swap(DB, batch_mirrors(bd.m, bd)) if mirrors else (lambda: None)
    try:
        t0 = time.time()
        out = bd.step(rows).clone()
        return out, time.time() - t0
    finally:
        restore()


def test_batch_vs_fused():
    """B rows of different slots (A at 1020/1021, B crossing the 256 page boundary, C near the start) == B
    independent single-row Fused (decode_kernels, the tuned T=1 path) steps, 2 steps, no resync. GEMMs are torch mirrors on both
    sides (same math), so differences come from near tiling/ring order (WR 1088 vs 1024), logits GEMM shape and the
    routing-weight sum: expect bf16-noise KL."""
    m, src = setup()
    names = ('A', 'B', 'C')
    bd = new_bd(m, names)
    fz = []
    for n in names:
        d = D.Decoder(m, graph=False, fuse=True)
        d.load(clone(m, src[n]))
        d.fused = K.Fused(m, d.st, d.x_raw, d.out)
        fz.append(d)
    worst = 0
    for i in range(2):
        rows = [(s, bd.seqs[s].pos, TOK[3 * i + s]) for s in range(3)]
        lg, dt = run_bd(bd, rows)
        for s, d in enumerate(fz):
            restore = swap(K, gemv_mirrors(m, d.fused))
            try:
                ref = d.step(rows[s][2]).clone()[0]
            finally:
                restore()
            kl = kld(ref, lg[s])
            ia, ib = int(ref.argmax()), int(lg[s].argmax())
            gap = (ref[ia] - ref[ib]).item()
            worst = max(worst, kl)
            print(f'step {i} slot {names[s]} pos {rows[s][1]}: KL(fused || batch) {kl:.1e}, max |dlogit| '
                  f'{(ref - lg[s]).abs().max():.4f}, top-1 {ia} vs {ib} (gap {gap:.3f})', flush=True)
            assert kl < 2e-2 and gap <= 0.25 and bool(torch.isfinite(lg[s]).all()), (i, s, kl, gap)
        print(f'  (batched step R=3: {dt:.0f} s in the interpreter)', flush=True)
    for s, d in enumerate(fz):
        assert torch.equal(d.st.rpos[d.st.rpos >= 0].sort().values,
                           bd.b.rpos[s][bd.b.rpos[s] >= 0].sort().values), 'ring positions differ'
    print(f'batch == single-row Fused: worst KL {worst:.1e} OK')


def test_verify_commit():
    """T verify rows of one slot == T sequential single steps, bit for bit (logits, conv state, rings, KV pages), in a
    mixed step: A verifies 5 rows 1020..1024 (far keys start, page boundary) (far keys start, page boundary) next to single rows of B and C (B
    crosses 256). Then commit({A: 3}) == the state after 3 single steps, and the next step from there (rejected rows
    still in the ring, masked) == the sequential path. Single-row slots commit inside the step."""
    m, src = setup()
    bd1, bd2 = new_bd(m, 'ABC'), new_bd(m, 'ABC')
    PA, PB, PC = (src[n].pos for n in 'ABC')
    T = 5
    rows = [(0, PA + i, TOK[i]) for i in range(T)] + [(1, PB, TOK[20]), (2, PC, TOK[21])]
    lg1, dt = run_bd(bd1, rows)
    print(f'mixed step: A x {T} verify + B + C (R=8 bucket, 7 rows) {dt:.0f} s', flush=True)
    seqv, s3 = [], None
    for i in range(T):
        seqv.append(run_bd(bd2, [(0, PA + i, TOK[i])])[0][0])
        if i == 2:
            s3 = snap(bd2)
    lb, lc = run_bd(bd2, [(1, PB, TOK[20])])[0][0], run_bd(bd2, [(2, PC, TOK[21])])[0][0]
    ok = [torch.equal(lg1[i], seqv[i]) for i in range(T)] + [torch.equal(lg1[T], lb), torch.equal(lg1[T + 1], lc)]
    print(f'verify rows == sequential steps (bit-exact): {ok}', flush=True)
    assert all(ok), [(lg1[i] - seqv[i]).abs().max().item() for i in range(T)]
    assert same_state(slot_state(bd1, 1), slot_state(bd2, 1)) and same_state(slot_state(bd1, 2), slot_state(bd2, 2)), \
        'single-row slots: in-step commit differs from sequential'
    try:
        bd1.step([(1, PB + 1, 5)])
        raise AssertionError('step with uncommitted verify rows accepted')
    except AssertionError as e:
        assert 'commit' in str(e), e
    full = (bd1.b.ring[:, :, 0].clone(), bd1.b.rpos[0].clone(), bd1.b.conv[:, 0].clone())
    pages = [bd1.seqs[0].kv.get(sl, PA, PA + T) for sl in range(8)], [bd2.seqs[0].kv.get(sl, PA, PA + T) for sl in range(8)]
    assert all(torch.equal(a, b) for x, y in zip(*pages) for a, b in zip(x, y)), 'KV page rows differ'
    seq2 = slot_state(bd2, 0)
    assert same_state(full[:2], seq2[:2]), 'verify rings/rpos != sequential'
    assert not torch.equal(full[2], seq2[2]), 'conv state written during a verify step'
    conv0, pend0 = bd1.b.conv.clone(), dict(bd1.pend)
    bd1.commit()                                                  # accept all T rows
    assert torch.equal(bd1.b.conv[:, 0], seq2[2]) and bd1.seqs[0].pos == PA + T, 'commit(all) != sequential'
    bd1.b.conv.copy_(conv0)                                       # undo, then test a partial accept
    bd1.pend, bd1.seqs[0].pos = pend0, PA
    bd1.commit({0: 3, 1: 1})
    assert bd1.seqs[0].pos == PA + 3
    rewind(bd2, s3)
    c1, c2 = bd1.b.conv[:, 0], bd2.b.conv[:, 0]
    assert torch.equal(c1, c2), 'commit(3) conv state != state after 3 single steps'
    a, _ = run_bd(bd1, [(0, PA + 3, TOK[40])])
    b, _ = run_bd(bd2, [(0, PA + 3, TOK[40])])
    print(f'commit(3) conv == sequential; next step after the rollback == sequential: {torch.equal(a, b)}', flush=True)
    assert torch.equal(a, b)
    st1, st2 = bd1.state(0), bd2.state(0)
    assert all(torch.equal(st1[k], st2[k]) for k in st1), 'state() differs after rollback'
    print('verify + commit OK')


def test_invariance_pads():
    """Batch invariance: slot A's row gives bit-identical logits and state alone (R=1), at row 2 of 3 (+1 pad, R=4)
    and at row 4 of 5 behind D's 4 verify rows (+3 pads, R=8). Pad rows write nothing: the empty slot's buffers and
    the pool's unused pages are unchanged, and the padded steps' states == unpadded single steps."""
    m, src = setup()
    bd = new_bd(m, 'ABCD', slots=5)
    PA, PB, PC, PD = (src[n].pos for n in 'ABCD')
    s0 = snap(bd)
    empty = slot_state(bd, 4)
    pool = m.pool
    used = {p for s in bd.seqs if s for p in s.kv.pages} | {p for s in src.values() for p in s.kv.pages}
    free = sorted(set(range(1, pool.n)) - used)
    pc, ps = pool.c[:, :, free].clone(), pool.s[:, :, free].view(torch.uint8).clone()
    xa = TOK[50]
    res = []
    for label, rows, ia in (('R=1', [(0, PA, xa)], 0),
                            ('R=4 (3 rows + 1 pad), row 2', [(1, PB, TOK[51]), (2, PC, TOK[52]), (0, PA, xa)], 2),
                            ('R=8 (5 rows + 3 pads), row 4', [(3, PD + i, TOK[53 + i]) for i in range(4)] + [(0, PA, xa)],
                             4)):
        rewind(bd, s0)
        lg, dt = run_bd(bd, rows)
        bd.commit()
        res.append((label, lg[ia].clone(), slot_state(bd, 0), lg, rows))
        print(f'{label}: {dt:.0f} s', flush=True)
    for label, l, st, _, _ in res[1:]:
        eq, es = torch.equal(l, res[0][1]), same_state(st, res[0][2])
        print(f'slot A row at {label} vs R=1: logits bit-identical {eq}, slot state identical {es}', flush=True)
        assert eq and es, label
    assert same_state(slot_state(bd, 4), empty), 'pad rows wrote into the empty slot'
    assert torch.equal(pool.c[:, :, free], pc) and torch.equal(pool.s[:, :, free].view(torch.uint8), ps), \
        'pad rows wrote into unused pages'
    # the R=4 step's B and C rows == their own single-row steps (state included)
    _, _, _, lg4, rows4 = res[1]
    after = snap(bd)
    rewind(bd, s0)
    for j, r in enumerate(rows4[:2]):
        l, _ = run_bd(bd, [r])
        assert torch.equal(l[0], lg4[j]), f'row {j} of the padded step != its single step'
    print('batch invariance + pad rows OK')
    del after


def test_capture_safe(emb=False):
    """Within an R bucket the step's torch ops and every kernel launch (grid, tensor addresses, scalars; kernels
    recorded, not run) are identical whatever the slots, positions, verify groups and live rows: one CUDA graph per
    bucket. Also no host sync inside the step. emb: with the device embedding table (INK_DEC_EMBED_GPU: the row lookup
    is inside the step)."""
    m, src = setup()
    bd = new_bd(m, 'ABCD', slots=4)
    if emb:
        bd.embed, bd.dev_tok = m.W.embed, True
    log, restore = launch_recorder([DB])
    tr = []
    bd._replay = lambda R: (log.clear(), tr.append((R, op_trace(lambda: bd._run(R)), list(log))))
    try:
        P = {s: bd.seqs[s].pos for s in range(4)}
        steps = [[(0, P[0], 1), (1, P[1], 2), (2, P[2], 3)],              # 3 rows + pad
                 [(3, P[3], 4), (3, P[3] + 1, 5), (3, P[3] + 2, 6), (3, P[3] + 3, 7)],   # one verify group of 4
                 [(2, P[2] + 1, 8), (0, P[0] + 1, 9)]]                      # bucket 2
        for rows in steps:
            bd.step(rows)
            bd.commit()
        steps2 = [[(1, P[1] + 1, 10), (2, P[2] + 2, 11), (3, P[3] + 4, 12), (0, P[0] + 2, 13)]]
        for rows in steps2:
            bd.step(rows)
            bd.commit()
    finally:
        restore()
    by = {}
    for R, ops, ln in tr:
        by.setdefault(R, []).append((ops, ln))
    for R, xs in sorted(by.items()):
        assert all(x == xs[0] for x in xs), f'bucket {R}: trace differs between steps (not capture-safe)'
        print(f'bucket R={R}: {len(xs)} steps, {len(xs[0][0])} torch ops + {len(xs[0][1])} launches, identical')
    assert len(by[4]) == 3
    print(f'capture safety OK{" (device embedding lookup in the step)" if emb else ""}')


def test_capture_safe_emb():
    test_capture_safe(emb=True)


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_') and (not ONLY or name in ONLY):
            t0 = time.time()
            fn()
            print(f'-- {name} {time.time() - t0:.0f} s', flush=True)
    print('DECODE BATCH OK')
