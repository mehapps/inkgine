"""CPU (Triton interpreter): prefill_glue kernels vs model.py's eager helpers on the same inputs, plus an fp64 reference
to show the kernel is as close to exact as the torch path. ~2-4 min.
sconv must be bit-identical (same products, same sum order, same division) in >= 99.9% of outputs (rest: 1 bf16 ulp);
rms / attn_res: fp32 reduction order differs -> max rel diff vs eager <= 1e-5 and error vs fp64 <= 2x eager's."""
import math
import os
import sys

COMPILE = '--compile' in sys.argv  # offline sm_120 compile (no GPU) instead of the interpreter tests
os.environ['CUDA_VISIBLE_DEVICES'] = ''
if not COMPILE:
    os.environ['TRITON_INTERPRET'] = '1'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402

import model as M  # noqa: E402
import prefill_glue as G  # noqa: E402

g = torch.Generator().manual_seed(0)
r = lambda *s: torch.randn(*s, generator=g)


def ulps(a, b):
    """bf16 tensors -> (fraction equal, max ulp distance)."""
    ia, ib = a.view(torch.int16).int(), b.view(torch.int16).int()
    d = (ia - ib).abs()
    return (d == 0).float().mean().item(), d.max().item()


def rel(a, b):
    return ((a.double() - b.double()).abs().max() / b.double().abs().max()).item()


def test_sconv():
    for T, C, dyn in ((70, 2304, False), (2, 1024, False), (130, 1024, True), (1, 1024, True)):
        x = (r(T, C) * 2).bfloat16()
        st0 = r(3, C).bfloat16()
        if dyn:
            args = dict(dyn=((r(4, C) * 0.3).bfloat16().float(), (r(T, 8, 4) * 0.3).bfloat16()))
        else:
            args = dict(w=(r(C, 1, 4) * 0.3).float())
        st_a, st_b = st0.clone(), st0.clone()
        ref = M.sconv.eager(x, st_a, **args)
        got = G.sconv(x, st_b, **args)
        eq, mu = ulps(got, ref)
        assert torch.equal(st_a, st_b), 'state'
        assert eq >= 0.999 and mu <= 1, (T, C, dyn, eq, mu)
        print(f'sconv T={T} C={C} dyn={dyn}: equal {eq:.5f}, max {mu} ulp, state ok')


def test_rms():
    for shape, w in (((70, 2304), True), ((70, 2304), False), ((33, 8, 128), False)):
        x = (r(*shape) * 3).bfloat16()
        wt = (1 + 0.1 * r(shape[-1])).float() if w else None
        ref, got = M.rms.eager(x, wt), G.rms(x, wt)
        xd = x.double()
        ex = xd * torch.rsqrt(xd.pow(2).mean(-1, keepdim=True) + 1e-6) * (wt.double() if w else 1)
        e_t, e_k, d = rel(ref, ex), rel(got, ex), rel(got, ref)
        assert d <= 1e-5 and e_k <= 2 * e_t + 1e-7, (shape, d, e_t, e_k)
        print(f'rms {shape} w={w}: vs eager {d:.1e}; vs fp64 eager {e_t:.1e} kernel {e_k:.1e}')


def test_attn_res():
    for nq, n, T in ((1, 3, 40), (3, 7, 17), (4, 2, 9), (1, 1, 5)):
        q = (r(nq, 2304) * 0.05).float()
        s = [M.rms.eager(r(T, 2304)) for _ in range(n)]
        ref, got = M.attn_res.eager(q, s), G.attn_res(q, s)
        S = torch.stack(s).double()
        w = torch.softmax(torch.einsum('qh,nth->qtn', q.double(), S) / math.sqrt(2304), -1)
        ex = torch.einsum('qtn,nth->qth', w, S)
        e_t, e_k, d = rel(ref, ex), rel(got, ex), rel(got, ref)
        assert got.shape == ref.shape and d <= 1e-5 and e_k <= 2 * e_t + 1e-7, (nq, n, d, e_t, e_k)
        print(f'attn_res nq={nq} n={n}: vs eager {d:.1e}; vs fp64 eager {e_t:.1e} kernel {e_k:.1e}')


def test_gate():
    T = 70
    x, msg = (r(T, 2304) * 2).float(), (r(T, 2304)).float()
    a, b = M.gate_in.eager(x, msg), G.gate_in(x, msg)
    eq, mu = ulps(b, a)
    assert b.shape == (T, 4608) and eq >= 0.995 and mu <= 1, (eq, mu)   # rms reduction order -> rare 1-ulp ties
    zz = (r(T, 4608) * 2).float()
    a, b = M.gate_out.eager(zz, x, msg), G.gate_out(zz, x, msg)
    d = rel(b, a)
    assert d <= 1e-5, d   # interpreter tanh = 2 sigmoid(2x) - 1 (GPU: libdevice tanhf, as torch)
    nw = (1 + 0.1 * r(2304)).float()
    gn, gr = G.gate_norm(zz, x, msg, nw), M.gate_norm.eager(zz, x, msg, nw)
    eq2, mu2 = ulps(gn, gr)
    # rms sum order + the interpreter's tanh stand-in (2 sigmoid(2x) - 1, imprecise near 0) on near-cancelling blends
    assert eq2 >= 0.999 and rel(gn, gr) <= 4e-3, (eq2, mu2, rel(gn, gr))  # <= 1 bf16 ulp of the row max
    print(f'gate_in: bf16 equal {eq:.5f} (max {mu} ulp); gate_out vs eager {d:.1e}; gate_norm bf16 equal {eq2:.5f} '
          f'(max {mu2} ulp)')


def test_qr_prep():
    T = 70
    y = (r(T, 6552) * 2).bfloat16()                  # projection output; q, r are column views (row stride 6552)
    q, rr = y[:, :3072], y[:, 3072:3456]
    qs, rs = (1 + 0.2 * r(24, 128)).float(), (1 + 0.2 * r(24, 16)).float()
    tau = 1 + 0.1 * torch.log(torch.clamp((torch.arange(9000, 9000 + T) + 1).float() / 8192, min=1))
    for t in (None, tau):
        a, b = M.qr_prep.eager(q, rr, qs, rs, t), G.qr_prep(q, rr, qs, rs, t)
        for x, z, n in zip(a, b, 'QR'):
            eq, mu = ulps(z, x)
            assert z.shape == x.shape and eq >= 0.999 and mu <= 1, (n, eq, mu)   # norm sum order -> rare 1-ulp ties
            print(f'qr_prep {n} tau={t is not None}: bf16 equal {eq:.5f} (max {mu} ulp)')


def _kv_case(T, p0, dyn, store):
    """kv_prep kernel vs eager on the same inputs; store: None | 'kv4' | 'pool' (pages not in position order)."""
    import kvpool
    K, V = (r(T, 1024) * 2).bfloat16(), (r(T, 1024) * 2).bfloat16()
    st = [r(3, 1024).bfloat16() for _ in range(2)]
    if dyn:
        w = [(r(4, 1024) * 0.3).bfloat16().float() for _ in range(2)]
        tap = [(r(T, 32) * 0.3).bfloat16() for _ in range(2)]
    else:
        w, tap = [(r(1024, 1, 4) * 0.3).float() for _ in range(2)], [None, None]

    def mk():
        if store == 'kv4':
            return M.KV4(3, 512, 'cpu')
        if store == 'pool':
            pool = kvpool.KVPool(2048, 'cpu', 'no-disk', cache_dir=None)
            other, t = pool.new_table(), pool.new_table()
            pool.reserve(other, 300)
            pool.reserve(t, 256)
            pool.reserve(other, 600)                     # interleave pages so t's pages are not contiguous
            pool.reserve(t, p0 + T)
            t.keep = other
            return t
        return None
    outs = []
    for f in (M.kv_prep.eager, G.kv_prep):
        kv, s2 = mk(), [x.clone() for x in st]
        o = f(K, V, s2[0], s2[1], w[0], w[1], tap[0], tap[1], kv, 1, p0)
        if kv is None:
            mem = []
        elif store == 'kv4':
            mem = [kv.c, kv.s.view(torch.uint8)]
        else:
            mem = [kv.pool.c, kv.pool.s.view(torch.uint8)]
        outs.append((o, s2, mem))
    (a, sa, ma), (b, sb, mb) = outs
    for x, z, n in zip(a, b, 'KV'):
        eq, mu = ulps(z, x)
        assert eq >= 0.999 and mu <= 1, (n, eq, mu)
        print(f'kv_prep T={T} p0={p0} dyn={dyn} store={store} {n}: bf16 equal {eq:.5f} (max {mu} ulp)', end='; ')
    assert all(torch.equal(x, z) for x, z in zip(sa, sb)), 'conv state'
    for x, z in zip(ma, mb):
        fr = (x == z).float().mean().item()
        assert fr >= 0.9999, fr                          # codes follow K/V bf16 (rare 1-ulp ties)
    print(f'store bytes equal {min([(x == z).float().mean().item() for x, z in zip(ma, mb)] or [1.]):.6f}')


def test_kv_prep():
    _kv_case(70, 0, False, None)
    _kv_case(70, 37, True, 'kv4')
    _kv_case(130, 200, True, 'pool')


def test_resid():
    T = 70
    o, h, y = (r(T, 2304) * 2).bfloat16(), (r(T, 2304) * 3).bfloat16(), (r(T, 2304)).bfloat16()
    w = (r(2304, 1, 4) * 0.3).float()
    st0 = r(3, 2304).bfloat16()
    for hh in (h, None):
        s1, s2 = st0.clone(), st0.clone()
        (ra, na), (rb, nb) = M.res_in.eager(o, s1, w, hh), G.res_in(o, s2, w, hh)
        assert torch.equal(ra, rb) and torch.equal(s1, s2), 'res_in r / state'
        d = rel(nb, na)
        assert d <= 1e-5, d
        print(f'res_in h={hh is not None}: r bit-equal, rms(r) vs eager {d:.1e}')
    rr = ra.clone() + 0.5
    s1, s2 = st0.clone(), st0.clone()
    (ha, na), (hb, nb) = M.res_out.eager(y, s1, w, rr), G.res_out(y, s2, w, rr)
    assert torch.equal(ha, hb) and torch.equal(s1, s2), 'res_out h / state'
    d = rel(nb, na)
    assert d <= 1e-5, d
    print(f'res_out: h bit-equal, rms(h) vs eager {d:.1e}')


def test_attn_res_norm():
    for nq, n, T in ((1, 3, 40), (3, 7, 17), (1, 1, 5)):
        q = (r(nq, 2304) * 0.05).float()
        s = [M.rms.eager(r(T, 2304)) for _ in range(n)]
        w = (1 + 0.1 * r(nq, 2304)).float()
        a, b = M.attn_res_norm.eager(q, s, w), G.attn_res_norm(q, s, w)
        eq, mu = ulps(b, a)
        # fp32 sum order (logits, row norm) -> rare 1-ulp ties; elements near 0 can then differ by a few ulps of their
        # own, so the bound is <= 1 bf16 ulp of the tensor max, as for gate_norm
        assert b.shape == a.shape and eq >= 0.999 and rel(b, a) <= 4e-3, (nq, n, eq, mu, rel(b, a))
        print(f'attn_res_norm nq={nq} n={n}: bf16 equal {eq:.5f} (max {mu} ulp, {rel(b, a):.1e} of max)')


def test_near_gate():
    import prefill_attn as PA
    T = 40
    Q = (r(T, 24, 128)).bfloat16()
    Kn, Vn = r(T, 8, 128).bfloat16(), r(T, 8, 128).bfloat16()
    tab = (r(T, 24, 16).bfloat16(), (r(16, 1024) * 0.3).bfloat16())
    sink = r(T, 24)
    y = (r(T, 4000) * 2).bfloat16()
    g = y[:, 500:3572]                                    # column view, row stride 4000
    o = PA.near(Q, Kn, Vn, tab, sink, 0, 0)
    ref = (o.view(T, -1) * torch.sigmoid(g.float())).bfloat16()
    got = PA.near(Q, Kn, Vn, tab, sink, 0, 0, gate=g).view(T, -1)
    eq, mu = ulps(got, ref)
    assert eq >= 0.999 and mu <= 1, (eq, mu)
    print(f'near + gate: bf16 equal {eq:.5f} (max {mu} ulp)')


def test_gatefuse():
    """INK_GATEFUSE: gate_down / gate_up_norm kernels vs model.py's eager compositions (= the old math)."""
    T = 70
    x, msg = (r(T, 2304) * 2).float(), (r(T, 2304)).float()
    w = (r(64, 4608) * 0.02).bfloat16()
    gc = (r(4608, 64) * 0.1).bfloat16().float()
    nw = (1 + 0.1 * r(2304)).float()
    za, zb = M.gate_down.eager(x, msg, w), G.gate_down(x, msg, w)
    eq, mu = ulps(zb, za)
    ex = (torch.cat((M.rms.eager(x), M.rms.eager(msg)), -1).bfloat16().double() @ w.double().t())
    e_t, e_k = rel(za, ex), rel(zb, ex)
    assert zb.shape == (T, 64) and eq >= 0.98 and mu <= 1 and e_k <= 2 * e_t + 1e-7, (eq, mu, e_t, e_k)
    a, b = M.gate_up_norm.eager(za, x, msg, gc, nw), G.gate_up_norm(za, x, msg, gc, nw)
    eq2, mu2 = ulps(b, a)
    # fp32 dot / rms sum order + the interpreter's tanh stand-in (see test_gate): <= 1 bf16 ulp of the row max
    assert b.shape == (T, 2304) and eq2 >= 0.999 and rel(b, a) <= 4e-3, (eq2, mu2, rel(b, a))
    print(f'gate_down: bf16 equal {eq:.5f} (max {mu} ulp; vs fp64 eager {e_t:.1e} kernel {e_k:.1e}); '
          f'gate_up_norm: bf16 equal {eq2:.5f} (max {mu2} ulp, {rel(b, a):.1e} of max)')


def _routing(T, E=128, k=8):
    """Model.moe's sorted rows: (tok, ws) per routed row, sorted by expert (argsort, not stable, like model.py)."""
    idx = torch.stack([torch.randperm(E, generator=g)[:k] for _ in range(T)])
    flat = idx.flatten()
    order = flat.argsort()
    return order // k, (torch.rand(T * k, generator=g) * 2)[order]


def test_moetail():
    """INK_MOETAIL: moe_tail kernel vs finish + res_out (eager and the existing kernels): h bit-identical."""
    import moe_triton as MT
    w = (r(2304, 1, 4) * 0.3).float()
    for T in (70, 64, 9):
        tok, ws = _routing(T)
        y = (r(T * 8, 2304)).bfloat16()
        rr = (r(T, 2304) * 3).float()
        st0 = r(3, 2304).bfloat16()
        s1, s2, s3 = st0.clone(), st0.clone(), st0.clone()
        ha, na = M.moe_tail.eager(y, tok, ws, s1, w, rr)
        hb, nb = G.moe_tail(y, tok, ws, s2, w, rr)
        hc, nc = G.res_out(MT.finish(y, tok, ws, T), s3, w, rr)
        assert torch.equal(M.finish_ref(y, tok, ws, T), MT.finish(y, tok, ws, T)), 'finish_ref'
        assert torch.equal(ha, hb) and torch.equal(hb, hc) and torch.equal(s1, s2) and torch.equal(s2, s3), T
        d = rel(nb, na)
        assert d <= 1e-5 and torch.equal(nb, nc), (T, d)
        print(f'moe_tail T={T}: h bit-equal to finish + res_out (eager and kernels), state ok, rms(h) vs eager {d:.1e}')


def test_real():
    """`--real` (CPU, real weights, ~15-30 min; not part of the default run): Model.forward over chunks 64 + 80 with a
    stub MoE (cpu expert reads are slow). (1) INK_GLUE2 eager compositions vs the old path: same math, so KL ~0
    (only attn_res's 1-query einsum shape differs); (2) INK_GLUE2 kernels (interpreter, incl. near + gate) vs the old
    path with the same near kernel: KL <= 2x the 1-ulp-nudged-embedding baseline."""
    from tokenizers import Tokenizer

    import prefill_attn  # noqa: F401  (near kernel under the interpreter)
    from engine import MODEL_DIR
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from gpu_check import ulp_nudge
    from gpu_moe import chunk_ids
    m = M.Model(MODEL_DIR, 'cpu', max_ctx=1024)
    M.Model.moe = lambda self, L, u, impl=None: (u.float() * 0.5).bfloat16()
    ids = chunk_ids(Tokenizer.from_file(os.path.join(MODEL_DIR, 'tokenizer.json')), 144)[:144]
    x = m.embed(ids)

    def run(g2, kern, attn, x0=x, gf=False):
        M.GLUE2[0], M.GLUE_CPU[0], M.PREFILL_ATTN, M.GATEFUSE[0] = g2, kern, attn, gf
        try:
            seq, outs = m.new_seq(), []
            for a, b in ((0, 64), (64, 144)):
                outs.append(m.forward(x0[a:b], seq))
            return torch.log_softmax(m.logits(torch.cat(outs)).float(), -1)
        finally:
            M.GLUE2[0], M.GLUE_CPU[0], M.PREFILL_ATTN, M.GATEFUSE[0] = False, False, 'torch', False

    def kl(a, b):
        return (a.exp() * (a - b)).sum(-1).mean().item(), (a.argmax(-1) == b.argmax(-1)).float().mean().item()
    ref = run(False, False, 'torch')
    comp = run(True, False, 'torch')
    nud = run(False, False, 'torch', ulp_nudge(x))
    ref_n = run(False, False, 'triton')
    ker = run(True, True, 'triton')
    gfk = run(True, True, 'triton', gf=True)                       # + INK_GATEFUSE kernels (moe stubbed: no moetail)
    (k1, t1), (k0, t0), (k2, t2), (k3, t3) = kl(ref, comp), kl(ref, nud), kl(ref_n, ker), kl(ref_n, gfk)
    print(f'real: glue2 eager compositions vs old KL {k1:.2e} top-1 {t1:.3f}; 1-ulp nudge KL {k0:.2e} top-1 {t0:.3f}; '
          f'glue2 kernels vs old (near kernel) KL {k2:.2e} top-1 {t2:.3f}; + gatefuse KL {k3:.2e} top-1 {t3:.3f}')
    assert k1 <= 2 * k0 and k2 <= 2 * k0 and k3 <= 2 * k0, (k1, k2, k3, k0)  # k1 is ~0 unless the 1-query einsum rounds differently


def compile_check():
    import subprocess
    import tempfile

    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    ptxas = os.path.join(os.path.dirname(triton.__file__), 'backends/nvidia/bin/ptxas-blackwell')
    B, F = '*bf16', '*fp32'
    sc = dict(x_ptr=B, st_ptr=B, w_ptr=F, tap_ptr=B, out_ptr=B, T='i32', C='i32', G='i32', scl='fp32')
    res = dict(q_ptr=F, **{f's{i}': F for i in range(8)}, out_ptr=F, T='i32', NQ='i32', scl='fp32', nw_ptr=F, eps='fp32')
    cases = [(f'sconv dyn={d}', G._sconv_kernel, sc, dict(DYN=d, RECIP=True, BT=16, BC=128, F32=False), 4)
             for d in (False, True)]
    cases += [(f'rms x={t} w={w}', G._rms_kernel, dict(x_ptr=t, w_ptr=F, out_ptr=F, D='i32', eps='fp32'),
               dict(HASW=w, BD=bd), 8) for t in (B, F) for w in (False, True) for bd in (128, 4096)]
    cases += [('gate_in', G._gate_in_kernel, dict(x_ptr=F, m_ptr=F, out_ptr=B, eps='fp32'), dict(H=2304, BD=4096, F32=False), 8),
              ('gate_out', G._gate_out_kernel, dict(z_ptr=F, x_ptr=F, m_ptr=F, out_ptr=F, n='i32'), dict(H=2304, BLK=1024, F32=False), 4),
              ('gate_norm', G._gate_norm_kernel, dict(z_ptr=F, x_ptr=F, m_ptr=F, w_ptr=F, out_ptr=B, eps='fp32'),
               dict(H=2304, BD=4096, F32=False), 8)]
    cases += [(f'attn_res n={n} nqp={q}', G._res_kernel, res, dict(N=n, NQP=q, RECIP=True, H=2304, BH=256, NORM=False,
               F32=False), 4)
              for n in (1, 2, 7) for q in (2, 4)]
    # INK_GLUE2
    U8, F8, I32 = '*u8', '*fp8e4nv', '*i32'
    resn = dict(res, out_ptr=B, nw_ptr=F, eps='fp32')
    cases += [(f'attn_res_norm n={n} nqp={q}', G._res_kernel, resn, dict(N=n, NQP=q, RECIP=True, H=2304, BH=256,
               NORM=True, F32=False), 4) for n in (2, 7) for q in (2, 4)]
    cases += [(f'qr clip D={d}', G._clip_kernel, dict(x_ptr=B, lx='i32', s_ptr=F, tau_ptr=F, out_ptr=B, n='i32', NH='i32',
               clip='fp32'), dict(TAU=t, D=d, BR=br, F32=False), 4) for d, br in ((128, 32), (16, 128)) for t in (False, True)]
    kvt = dict(x_ptr=B, st_ptr=B, w_ptr=F, tap_ptr=B, out_ptr=B, c_ptr=U8, s_ptr=F8, tbl_ptr=I32, T='i32', p0='i32',
               scl='fp32', s7='fp32', clip='fp32')
    cases += [(f'kv_prep dyn={d} v={v} kv={k}', G._kv_kernel, kvt, dict(DYN=d, IS_V=v, KV=k, RECIP=True, BT=16, F32=False), 4)
              for d, k in ((False, False), (True, True)) for v in (False, True)]
    rt = dict(x_ptr=B, st_ptr=B, w_ptr=F, h_ptr=B, r_ptr=F, out_ptr=F, n_ptr=F, T='i32', scl='fp32', eps='fp32')
    cases += [('res_in', G._resid_kernel, rt, dict(OUT=False, HAS_H=True, RECIP=True, BT=8, BC=256, F32=False), 4),
              ('res_out', G._resid_kernel, dict(rt, out_ptr=B), dict(OUT=True, HAS_H=False, RECIP=True, BT=8, BC=256,
               F32=False), 4)]
    cases += [('gate_down', G._gate_down_kernel, dict(x_ptr=F, m_ptr=F, w_ptr=B, z_ptr=B, T='i32', eps='fp32'),
               dict(H=2304, BT=16, BK=128, F32=False), 4),
              ('gate_up_norm', G._gate_up_kernel, dict(z_ptr=B, g_ptr=F, x_ptr=F, m_ptr=F, w_ptr=F, out_ptr=B, T='i32',
               eps='fp32'), dict(H=2304, BT=16, BH=64, F32=False), (8, 1)),
              ('moe_tail', G._tail_kernel, dict(y_ptr=B, ws_ptr=F, idx_ptr=I32, f_ptr=B, st_ptr=B, w_ptr=F, r_ptr=F,
               out_ptr=B, n_ptr=F, T='i32', scl='fp32', eps='fp32'), dict(KR=8, RECIP=True, BT=8, BC=256, F32=False), 4)]
    bad = 0
    for name, fn, ty, consts, w in cases:
        try:
            sig = {n: 'constexpr' if n in consts else ty[n] for n in fn.arg_names}
            k = triton.compile(ASTSource(fn=fn, signature=sig, constexprs=consts), target=GPUTarget('cuda', 120, 32),
                               options=dict(num_warps=w[0], num_stages=w[1]) if isinstance(w, tuple) else dict(num_warps=w))
            with tempfile.NamedTemporaryFile('w', suffix='.ptx') as f:
                f.write(k.asm['ptx']); f.flush()
                r_ = subprocess.run([ptxas, '-v', '-arch=sm_120a', f.name, '-o', '/dev/null'], capture_output=True, text=True)
            info = ' '.join(x.split('info    : ')[-1].strip() for x in r_.stderr.splitlines() if 'spill' in x or 'registers' in x)
            print(f'{name:28s} smem {k.metadata.shared:6d} | {info}')
            bad += k.metadata.shared > 101376                                 # sm_120 per-block limit
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f'{name:28s} FAILED: {str(e).splitlines()[0] if str(e) else type(e).__name__}')
    print('COMPILE OK' if not bad else f'{bad} FAILED')
    return 1 if bad else 0


if __name__ == '__main__':
    if COMPILE:
        sys.exit(compile_check())
    if '--only' in sys.argv:  # e.g. --only gatefuse,moetail
        for n in sys.argv[sys.argv.index('--only') + 1].split(','):
            globals()['test_' + n]()
        print('GLUE ONLY OK')
        sys.exit(0)
    if '--real' in sys.argv:
        test_real()
        print('GLUE REAL OK')
        sys.exit(0)
    test_sconv()
    test_rms()
    test_attn_res()
    test_gate()
    test_qr_prep()
    test_kv_prep()
    test_resid()
    test_attn_res_norm()
    test_near_gate()
    test_gatefuse()
    test_moetail()
    print('GLUE ALL OK')
