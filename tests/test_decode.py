"""CPU checks for the fixed-shape decode path (decode.py). No GPU.
1. DecodeState.attend (1024-slot ring + mask, KV4 put_at, bucketed far range) == Model.attn_core, token by token,
   from a prefilled state through the ring wrap and with several far blocks.
1b. far_attn (Triton kernel reading 4-bit pages, run by the CPU interpreter) == Model.far / attn_core: across 1024,
   split and page boundaries on a paged pool, and at 200k positions on a random KV4 store.
2. far_bucket, sample() (greedy, top-k, top-p, temperature: frequencies vs the exact distribution).
3. INK_REAL=1: real weights, CPU (~2 min): prefill a few tokens, then 3 decode steps on the Seq path vs Decoder;
   Decoder.stream (device embedding + sampling, pipelined) == the step loop.
4. Fused path (decode_kernels.py, INK_DEC_FUSE): _quant_put == KV4.quant bit for bit (always); with INK_REAL=1 each small
   kernel == the model.py torch code (test_fused_units, ~3 min), the GEMV kernels == torch mirrors (test_fused_gemv,
   ~4 min), and the whole step in lockstep vs the eager Decoder + capture safety (test_fused_real, ~30 min).
   test_far_dot: the tensor-core far kernel (INK_DEC_FAR_DOT) on the far checks.
Run: cd engine && .venv/bin/python tests/test_decode.py"""
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('TRITON_INTERPRET', '1')   # far_attn kernel on CPU
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402

import decode as D  # noqa: E402
import decode_kernels as DK  # noqa: E402
import model as M  # noqa: E402

torch.manual_seed(0)


def fake_model(cap):
    m = object.__new__(M.Model)
    m.kv, m.slot, m.dev = M.KV4(1, cap, 'cpu'), {0: 0}, torch.device('cpu')
    return m


def rand(N):
    Q = (torch.randn(N, M.NQ, M.HD) * 0.5).bfloat16()
    K = M.l2clip(torch.randn(N, M.NKV, M.HD), M.CLIP_QK).bfloat16()
    V = torch.randn(N, M.NKV, M.HD).bfloat16()
    R = torch.randn(N, M.NQ, 16).bfloat16()
    proj = torch.randn(16, M.WIN).bfloat16()
    return Q, K, V, R, proj, torch.randn(N, M.NQ)


def test_attend_vs_attn_core():
    P0, N = 1500, 3400                       # prefill 1500, then 1900 decode steps: ring wraps, far grows past 2 blocks
    Q, K, V, R, proj, sink = rand(N)
    old, M.FAR_BLK = M.FAR_BLK, 1024
    try:
        for full in (False, True):
            mr, md = fake_model(N), fake_model(N)
            sr, sd = M.Seq('cpu', mr.kv), M.Seq('cpu', md.kv)
            for m, s in ((mr, sr), (md, sd)):            # identical prefill into both stores
                m.attn_core(Q[:P0], K[:P0], V[:P0], (R[:P0], proj), sink[:P0], s, 0, full)
                s.pos = P0
            dec = D.Decoder(md, graph=False)
            dec.load(sd)
            st, worst = dec.st, 0.
            for p in range(P0, N):
                sl = slice(p, p + 1)
                ref = mr.attn_core(Q[sl], K[sl], V[sl], (R[sl], proj), sink[sl], sr, 0, full)
                sr.pos += 1
                st.pos.fill_(p)
                st.far_cap = D.far_bucket(p, N)
                st.positions(1)
                got = st.attend(md, Q[sl], K[sl], V[sl], (R[sl], proj), sink[sl], 0, full)
                err = ((got - ref).abs().max() / ref.abs().max()).item()
                mean = ((got - ref).abs().mean() / ref.abs().mean()).item()
                worst = max(worst, err)
                # only the key order differs (slot vs time order; gathering the ring in time order was checked to
                # be bit-exact): ~99.9% of elements bit-equal, rare steps up to ~3e-4 mean rel (bf16 ulp flips)
                assert err < 2e-2 and mean < 1e-3, (full, p, err, mean)
            if full:
                assert torch.equal(mr.kv.c, md.kv.c) and torch.equal(mr.kv.s.view(torch.uint8), md.kv.s.view(torch.uint8)), \
                    'KV4 put_at differs from put'
            print(f'attend vs attn_core ({"full" if full else "sliding"}), {N - P0} steps: worst rel max err {worst:.2e}')
    finally:
        M.FAR_BLK = old


def op_trace(fn):
    """Run fn, return its aten ops as (op, args): tensor args made earlier in fn -> ('tmp', producer index), others ->
    ('ext', address, shape), non-tensors as repr. A CUDA graph replays the first capture's ops with its scalars and
    addresses, so two steps of one far bucket must give identical traces (else: baked-in value or stale address)."""
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_flatten

    class Rec(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            flat = [a for a in tree_flatten((args, kwargs))[0]]
            ins = {a.untyped_storage().data_ptr() for a in flat if isinstance(a, torch.Tensor)}
            sig = [str(func)]
            for a in flat:
                if isinstance(a, torch.Tensor):
                    k = a.untyped_storage().data_ptr()
                    sig.append(('tmp', made[k]) if k in made else ('ext', a.data_ptr(), tuple(a.shape)))
                else:
                    sig.append(repr(a))
            for o in tree_flatten(out)[0]:
                if isinstance(o, torch.Tensor) and o.untyped_storage().data_ptr() not in ins:
                    made[o.untyped_storage().data_ptr()] = len(ops)
            ops.append(tuple(sig))
            return out

    ops, made = [], {}
    with Rec():
        fn()
    assert not any('_local_scalar_dense' in o[0] for o in ops), 'host sync in a captured step'
    return ops


def test_capture_safe():
    """Pool page table (the engine's store), decode from position 1000: crosses 1024 (far part starts, near ring exactly
    1023 keys + current) and the 2048 -> 4096 bucket switch at 3072. Each step == the sequential path, and each far
    bucket's op trace is the same at every step (what a captured graph would replay)."""
    import kvpool
    P0, N = 1000, 3100
    Q, K, V, R, proj, sink = rand(N)
    old, M.FAR_BLK = M.FAR_BLK, 1024
    try:
        pool = kvpool.KVPool(2 * N + 2 * kvpool.PAGE, 'cpu', 'test', cache_dir=None)  # two sequences
        mr, md = fake_model(N), fake_model(N)
        md.pool = pool
        sr, sd = M.Seq('cpu', pool.new_table()), M.Seq('cpu', pool.new_table())
        for m, s in ((mr, sr), (md, sd)):
            assert pool.reserve(s.kv, N)
            m.attn_core(Q[:P0], K[:P0], V[:P0], (R[:P0], proj), sink[:P0], s, 0, True)
            s.pos = P0
        dec = D.Decoder(md, graph=False)
        dec.load(sd)
        st, traces, worst = dec.st, {}, 0.
        B = [x[:1].clone() for x in (Q, K, V, R, sink)]                 # fixed-address inputs, like the graph's
        for p in range(P0, N):
            sl = slice(p, p + 1)
            ref = mr.attn_core(Q[sl], K[sl], V[sl], (R[sl], proj), sink[sl], sr, 0, True)
            sr.pos += 1
            for b, x in zip(B, (Q, K, V, R, sink)):
                b.copy_(x[sl])
            st.pos.fill_(p)
            st.far_cap = cap = D.far_bucket(p, st.kv.cap)
            got = []
            ops = op_trace(lambda: (st.positions(1), got.append(st.attend(md, B[0], B[1], B[2], (B[3], proj), B[4], 0, True))))
            first = traces.setdefault(cap, (p, ops))
            assert first[1] == ops, f'bucket {cap}: step {p} replays differently from its capture at {first[0]}'
            err = ((got[0] - ref).abs().mean() / ref.abs().mean()).item()
            worst = max(worst, err)
            assert err < 1e-3, (p, cap, err)
        assert sorted(traces) == [0, 2048, 4096], sorted(traces)
        print(f'capture-safe decode 1000 -> {N} (buckets {sorted(traces)}): traces fixed per bucket, '
              f'worst mean rel err vs sequential {worst:.1e}')
        # far kernel path (one graph for all positions): the torch ops and the kernel launch (grid, input addresses and
        # scalars; recorded instead of run, the interpreter's own tensor copies would pollute the trace) never change.
        # Args 7, 8 (o, lse partials) are temporaries made in far_attn: graph-pool addresses, fixed at capture.
        launches, kern = [], D._far_kernel
        D._far_kernel = {(M.NKV, -(-st.kv.cap // D.DEC_SPLIT)): lambda *a, **k: launches.append(
            [x.data_ptr() if isinstance(x, torch.Tensor) else x for i, x in enumerate(a) if i not in (7, 8)]
            + sorted(k.items()))}
        st.kern, first = True, None
        try:
            for p in (1100, 1101, 2047, 3000, N - 1):
                st.pos.fill_(p)
                ops = op_trace(lambda: (st.positions(1), st.attend(md, B[0], B[1], B[2], (B[3], proj), B[4], 0, True)))
                first = first or ops
                assert ops == first and launches[-1] == launches[0], f'far kernel path: step {p} differs'
        finally:
            D._far_kernel, st.kern = kern, False
        print(f'capture-safe far kernel path: same trace + launch at 5 positions 1100..{N - 1}')
    finally:
        M.FAR_BLK = old


def test_far_kernel_attend():
    """DecodeState with the far kernel (kern=True) == attn_core, token by token, on a paged pool whose page ids are
    interleaved with another sequence: 1018 -> 1030 (far part starts at 1024) and 1270 -> 1300 (far range crosses
    256 keys: a split boundary with split=256 and a page boundary)."""
    import kvpool
    old = D.DEC_SPLIT
    D.DEC_SPLIT = 256
    try:
        for P0, N in ((1018, 1030), (1270, 1300)):
            Q, K, V, R, proj, sink = rand(N)
            pool = kvpool.KVPool(2 * N + 2 * kvpool.PAGE, 'cpu', 'test', cache_dir=None)
            mr, md = fake_model(N), fake_model(N)
            md.pool = pool
            sr, sd = M.Seq('cpu', pool.new_table()), M.Seq('cpu', pool.new_table())
            for i in range(-(-N // kvpool.PAGE)):                    # alternate pages: sd's ids are not contiguous
                for s in (sr, sd):
                    assert pool.reserve(s.kv, (i + 1) * kvpool.PAGE)
            assert sd.kv.pages != list(range(sd.kv.pages[0], sd.kv.pages[0] + len(sd.kv.pages)))
            for m, s in ((mr, sr), (md, sd)):
                m.attn_core(Q[:P0], K[:P0], V[:P0], (R[:P0], proj), sink[:P0], s, 0, True)
                s.pos = P0
            dec = D.Decoder(md, graph=False)
            dec.load(sd)
            st, worst = dec.st, 0.
            st.kern = True
            for p in range(P0, N):
                sl = slice(p, p + 1)
                ref = mr.attn_core(Q[sl], K[sl], V[sl], (R[sl], proj), sink[sl], sr, 0, True)
                sr.pos += 1
                st.pos.fill_(p)
                st.positions(1)
                got = st.attend(md, Q[sl], K[sl], V[sl], (R[sl], proj), sink[sl], 0, True)
                err = ((got - ref).abs().mean() / ref.abs().mean()).item()
                worst = max(worst, err)
                assert err < 2e-3, (p, err)
            print(f'far kernel attend vs attn_core {P0} -> {N} (paged, split 256): worst mean rel err {worst:.1e}')
    finally:
        D.DEC_SPLIT = old


def test_far_kernel_long():
    """far_attn at 200k positions (default split, 4-bit store filled with random codes/scales) == Model.far, and both
    close to an fp32 reference over the same dequantized keys."""
    cap, p = 200_064, 200_000
    m = fake_model(cap)
    kv = m.kv
    g = torch.Generator().manual_seed(1)
    kv.c.copy_(torch.randint(0, 256, kv.c.shape, dtype=torch.uint8, generator=g))
    kv.s.copy_((torch.rand(kv.s.shape, generator=g) * 0.08 + 0.01).to(torch.float8_e4m3fn))
    Q = (torch.randn(1, M.NQ, M.HD, generator=g) * 0.5).bfloat16()
    pos = torch.tensor([p])
    o, lse = D.far_attn(Q, kv, 0, pos, blk=256)                      # (bigger tiles: the interpreter is slow)
    tol = 5e-3 if D.FAR_DOT else 1e-3                                  # dot kernel: bf16 probabilities, like torch
    ro, rl = m.far(Q, kv, 0, pos, D.far_bucket(p, cap))
    n = p - (M.WIN - 1)
    Kf, Vf = (x.float() for x in kv.get(0, 0, n))                     # exact fp32 reference
    q = Q[0].float().view(M.NKV, 3, M.HD)
    S = torch.einsum('kgd,skd->kgs', q, Kf) * M.HD ** -0.5
    xl = torch.logsumexp(S, -1).reshape(M.NQ)
    xo = torch.einsum('kgs,skd->kgd', torch.softmax(S, -1), Vf).reshape(M.NQ, M.HD)
    del Kf, Vf, S
    rel = lambda a, b: ((a - b).abs().mean() / b.abs().mean()).item()
    ek, et, ekt = rel(o[0], xo), rel(ro[0], xo), rel(o[0], ro[0])
    assert ek < tol and et < 5e-3 and ekt < 5e-3, (ek, et, ekt)
    assert (lse[0] - xl).abs().max() < 1e-3 and (rl[0] - xl).abs().max() < 2e-2
    print(f'far kernel at {p} positions: mean rel err vs fp32 kernel {ek:.1e}, torch path {et:.1e}; kernel vs torch {ekt:.1e}')


def test_far_dot():
    """The same far checks with the tensor-core kernel (INK_DEC_FAR_DOT=1: bf16 tiles, bf16 probabilities)."""
    old = D.FAR_DOT
    D.FAR_DOT = True
    try:
        test_far_kernel_attend()
        test_far_kernel_long()
    finally:
        D.FAR_DOT = old


def test_quant_put():
    """decode_kernels._quant_put (the fused path's 4-bit page write) == KV4.quant bit for bit: random heads, and scales
    that land exactly between two e4m3 values (round to even) or in the e4m3 subnormal range."""
    import triton
    import triton.language as tl

    @triton.jit
    def k(x_ptr, c, s):
        h = tl.program_id(0)
        DK._quant_put(tl.load(x_ptr + h * 128 + tl.arange(0, 128)).to(tl.float32), c, s, 0, h, True)
    tie = torch.zeros(1, M.NKV, M.HD)
    vals = torch.tensor([1.3671875, 0.0009765625, 0.05, 0.0703125, 3.0, 1e-5, 0.109375, 22.0])
    for h in range(M.NKV):
        tie[0, h, ::16] = vals.roll(h)
    for x in (M.l2clip(torch.randn(1, M.NKV, M.HD), M.CLIP_QK), tie):
        x = x.bfloat16()
        c, s = torch.zeros(512, dtype=torch.uint8), torch.zeros(64, dtype=torch.uint8)
        k[(M.NKV,)](x, c, s)
        q, sc = M.KV4.quant(x)
        assert torch.equal(c, q.flatten()) and torch.equal(s, sc.view(torch.uint8).flatten())
    print('fused 4-bit page write == KV4.quant (random, e4m3 ties, subnormals)')


def test_far_bucket():
    cap = 1 << 20
    assert D.far_bucket(0, cap) == 0 and D.far_bucket(1023, cap) == 0
    assert D.far_bucket(1024, cap) == 2048                          # 1 far key (key 0)
    for p in (1024, 3000, 5000, 40000, 66558, 66559, 1_000_000, cap - 1):
        b = D.far_bucket(p, cap)
        assert p - 1023 <= b <= cap and (b == cap or b & (b - 1) == 0), (p, b)
    assert D.far_bucket(3999, 4000) == 4000                          # clipped to the KV4 capacity
    print('far_bucket ok')


def ref_probs(logits, temperature, top_p, top_k):
    """Engine.sample's distribution, computed exactly."""
    x = logits / temperature
    if top_k:
        x = x.masked_fill(x < x.topk(top_k).values[-1], float('-inf'))
    p = torch.softmax(x, -1)
    if top_p < 1:
        sp, si = p.sort(descending=True)
        sp[sp.cumsum(0) - sp > top_p] = 0
        p = torch.zeros_like(p).scatter_(0, si, sp)
    return p / p.sum()


def test_sample():
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(50, generator=g) * 2
    assert int(D.sample(logits, 0)) == int(logits.argmax())
    assert int(D.sample(logits, 0.7, top_k=1, generator=g)) == int(logits.argmax())
    n = 40000
    for temp, top_p, top_k in ((1.0, 1.0, 0), (0.6, 1.0, 5), (1.3, 0.8, 0), (1.0, 0.9, 10)):
        cnt = torch.zeros(50)
        for _ in range(n):
            cnt[int(D.sample(logits, temp, top_p, top_k, g))] += 1
        p = ref_probs(logits, temp, top_p, top_k)
        dev = (cnt / n - p).abs().max().item()
        assert dev < 0.012 and cnt[p == 0].sum() == 0, (temp, top_p, top_k, dev)
        print(f'sample T={temp} top_p={top_p} top_k={top_k}: max freq deviation {dev:.4f}')


def test_real_weights():
    if os.environ.get('INK_REAL') != '1':
        print('real-weights check skipped (INK_REAL=1 to run, ~2 min)')
        return
    from engine import MODEL_DIR
    m = M.Model(MODEL_DIR, 'cpu', max_ctx=4096)
    ids = [200000 + 7, 1000, 2000, 3000, 4000, 5000]
    ids = [i % M.VOCAB for i in ids]
    toks = [11, 333, 4444]
    sa, sb = m.new_seq(), m.new_seq()
    m.forward(m.embed(ids), sa)
    m.forward(m.embed(ids), sb)
    dec = D.Decoder(m, graph=False)
    dec.load(sb)
    for t in toks:
        ref = m.logits(m.forward(m.embed([t]), sa))
        got = dec.step(t).clone()
        lr, lg = torch.log_softmax(ref, -1), torch.log_softmax(got, -1)
        kl = (lr.exp() * (lr - lg)).sum().item()
        print(f'real weights token {t}: max |dlogit| {(ref - got).abs().max():.4f}, KL {kl:.2e}, '
              f'top-1 {int(ref.argmax())} vs {int(got.argmax())}')
        assert kl < 1e-3 and int(ref.argmax()) == int(got.argmax())
    assert sa.pos == sb.pos
    # pipelined stream (embedding on the device, sampling on the device) == step + host sample, greedy and sampled
    for temp in (0, 0.8):
        outs = []
        for embed_dev in (False, True):
            s = m.new_seq()
            f = m.forward(m.embed(ids), s)
            d = D.Decoder(m, graph=False, embed_dev=embed_dev)
            d.load(s)
            gen = torch.Generator().manual_seed(9)
            lg = m.logits(f)[0]
            if embed_dev:
                outs.append(list(d.stream(lg, 3, temp, generator=gen)))
            else:
                got = []
                for i in range(3):
                    got.append(int(D.sample(lg, temp, generator=gen)))
                    if i < 2:
                        lg = d.step(got[-1])[0]
                outs.append(got)
            assert s.pos == len(ids) + 2
        print(f'stream (T={temp}) {outs[1]} == step loop {outs[0]}')
        assert outs[0] == outs[1]
    # whole graph step (_run) at positions 1100..1102 (far bucket 2048, garbage state): identical op traces. The triton
    # MoE kernels can't run on CPU, so they are stubbed after the (traced) routing.
    m.moe_grouped = lambda L, u, se, tok, ws, impl=None: (u.float() * ws.sum()).bfloat16()
    sc = m.new_seq()
    assert m.pool.reserve(sc.kv, 2048)
    sc.pos = 1100
    dec.load(sc)
    dec.st.moe_impl = 'triton'
    tr = []
    for p, t in zip(range(1100, 1103), toks):
        dec.x_raw.copy_(m.W.embed[t:t + 1])
        dec.st.pos.fill_(p)
        dec.st.far_cap = D.far_bucket(p, dec.st.kv.cap)
        tr.append(op_trace(dec._run))
    assert dec.st.far_cap == 2048 and tr[0] == tr[1] == tr[2], 'whole decode step not capture-safe'
    print(f'real weights: whole decode step capture-safe ({len(tr[0])} ops, same trace at 3 positions)')


def launch_recorder(mods):
    """Replace every Triton kernel (name ending in _kernel) in mods by a recorder: launch -> (name, grid, tensor
    addresses, scalars). Returns (log, restore)."""
    log, saved = [], []
    for mod in mods:
        for n, k in list(vars(mod).items()):
            if n.endswith('_kernel') and hasattr(k, 'run'):
                saved.append((mod, n, k))

                class Rec:
                    def __init__(self, n):
                        self.n = n

                    def __getitem__(self, grid, n=n):
                        return lambda *a, **kw: log.append((n, tuple(grid), [x.data_ptr() if isinstance(x, torch.Tensor)
                                                                            else x for x in a], sorted(kw.items())))
                setattr(mod, n, Rec(n))

    def restore():
        for mod, n, k in saved:
            setattr(mod, n, k)
    return log, restore


def gemv_mirrors(m, F):
    """Torch mirrors of the three big GEMV kernels (proj, w13, w2) with the kernels' call signatures: the CPU
    interpreter needs ~1 min per launch for them, so whole-step tests run these instead (test_fused_gemv checks the
    real kernels == these mirrors on single visits). Everything else in the step runs the real kernels."""
    import moe_triton
    nws = {d['nwt'].data_ptr(): d['nws'] for d in F.vis}
    layer = {d['ex'][j].data_ptr(): d['L'] for d in F.vis for j in (0, 2)}
    w2c = {}

    def proj(w, n0, n1, x, xs, nwt, y, s_ptr, NORM, W8=False, **k):
        xx = [x.view(-1)[j * xs: j * xs + M.H] for j in range(3)]
        if NORM:
            xx = [M.rms(a, nw).bfloat16() for a, nw in zip(xx, nws[nwt.data_ptr()])]
        rows = ((0, n0), (n0, n0 + n1), (n0 + n1, n0 + 2 * n1))     # k and v groups have the same size
        sc = s_ptr if W8 else torch.ones(n0 + 2 * n1)
        y[:n0 + 2 * n1].copy_(torch.cat([(w[a:b].float() @ xj.float()) * sc[a:b] for (a, b), xj in zip(rows, xx)]))

    def w13(s, bias, u, w, sc, gs, a1, a2, a3, a4, act, idx, wt, FP4, **k):
        L = layer[w.data_ptr()]
        ids = (s + bias).topk(8).indices
        ws = s[ids]
        idx.copy_(ids)
        wt.copy_(ws / (ws.sum() + 1e-20) * 8.0)
        w2c.clear()
        for t, e in enumerate(ids.tolist()):
            w13d, w2c[e] = m.W.expert(L, e)
            gu = (u.float() @ w13d.float().t()).bfloat16()
            g, up = gu[0::2].float(), gu[1::2].float()
            act[t] = (4 * moe_triton.tanh(g / 4) * torch.sigmoid(g) * 25 * moe_triton.tanh(up / 25)).bfloat16()

    def w2(act, idx, wt, w, sc, gs, a1, a2, a3, a4, part, FP4, **k):  # the real _w2_fin_kernel sums the parts
        for t, e in enumerate(idx.tolist()):
            part[t] = (act[t].float() @ w2c[e].float().t()).bfloat16().float() * wt[t]

    class Grid:
        def __init__(self, fn):
            self.fn = fn

        def __getitem__(self, grid):
            return self.fn
    return {'_proj_kernel': Grid(proj), '_w13_kernel': Grid(w13), '_w2_kernel': Grid(w2)}


def swap(mod, repl):
    old = {n: getattr(mod, n) for n in repl}
    for n, f in repl.items():
        setattr(mod, n, f)
    return lambda: [setattr(mod, n, f) for n, f in old.items()]


def fused_model():
    import decode_kernels as K
    from engine import MODEL_DIR
    K.BLK.update(wo=128, router=128, w13=128, w2=128, gate=64)   # fewer, bigger programs: the interpreter is serial
    return K, M.Model(MODEL_DIR, 'cpu', max_ctx=4096)


def test_fused_gemv():
    """INK_REAL=1 (~4 min): the real proj / w13 / w2 kernels (interpreter) == their torch mirrors on single visits:
    proj on a full visit without loop message (3 bf16 inputs, k/v taps) and on a gated core visit (input RMSNorm
    fused); w13 + w2 (top-8, 8 experts, mlp conv + residual) on an NVFP4 layer and an MXFP8 layer."""
    if os.environ.get('INK_REAL') != '1':
        print('fused GEMV kernel check skipped (INK_REAL=1 to run)')
        return
    K, m = fused_model()
    dec = D.Decoder(m, graph=False, fuse=True)
    F = K.Fused(m, dec.st, dec.x_raw, dec.out)
    mir = gemv_mirrors(m, F)
    g = torch.Generator().manual_seed(5)
    rel = lambda a, b: ((a.float() - b.float()).abs().max() / b.float().abs().max()).item()
    for v, norm in ((5, False), (30, True)):                        # visit 5 = layer 5 (full); 30 = layer 12, pass 2
        d = F.vis[v]
        x = torch.randn(3, M.H, generator=g)
        x = x.bfloat16() if not norm else x
        ys = []
        for fn in (K._proj_kernel[(d['N'] // K.BLK['proj'],)], mir['_proj_kernel'][0]):
            y = torch.zeros(8664, dtype=torch.bfloat16)
            fn(d['cat'], d['n0'], d['n1'], x, M.H, d['nwt'], y, d['cs'], NORM=norm, F32=True, H=M.H, CH=K.CH,
               BN=K.BLK['proj'], W8=F.w8)
            ys.append(y)
        e = rel(ys[0], ys[1])
        print(f'proj kernel visit {v} (norm={norm}): max rel diff vs mirror {e:.1e}', flush=True)
        assert e < 1e-2, e
    for v in (6, 44):                                               # layer 6 NVFP4, layer 20 MXFP8
        d = F.vis[v]
        w13, s13, w2, s2 = d['ex']
        S = torch.sigmoid(torch.randn(128, generator=g))
        U = M.rms(torch.randn(M.H, generator=g)).bfloat16()
        Rr = torch.randn(M.H, generator=g) * 0.1
        st0 = torch.randn(3, M.H, generator=g).bfloat16()
        out = []
        for real in (True, False):
            act, idx, wt = torch.zeros(8, K.NI, dtype=torch.bfloat16), torch.zeros(8, dtype=torch.int32), torch.zeros(8)
            st, h = st0.clone(), torch.zeros(M.H, dtype=torch.bfloat16)
            f13 = K._w13_kernel[(K.NI // K.BLK['w13'], 8)] if real else mir['_w13_kernel'][0]
            f2 = K._w2_kernel[(M.H // K.BLK['w2'], 8)] if real else mir['_w2_kernel'][0]
            part = torch.zeros(8, M.H)
            f13(S, d['gb'], U, w13, s13, d['gs13'], w13.stride(0), w13.stride(1), s13.stride(0), s13.stride(1), act,
                idx, wt, FP4=d['fp4'], F32=True, H=M.H, NI=K.NI, BN=K.BLK['w13'], BK=K.BLK['kbk'])
            f2(act, idx, wt, w2, s2, d['gs2'], w2.stride(0), w2.stride(1), s2.stride(0), s2.stride(1), part,
               FP4=d['fp4'], F32=True, H=M.H, NI=K.NI, BN=K.BLK['w2'], BK=K.BLK['kbk2'])
            K._w2_fin_kernel[(M.H // K.BLK['w2f'],)](part, st, d['mcw'], Rr, h, F32=True, H=M.H, BN=K.BLK['w2f'])
            out.append((act, idx, wt, st, h))
        (a1, i1, w1, s1, h1), (a0, i0, w0, s0, h0) = out
        assert torch.equal(i1, i0), (i1, i0)
        ea, ew, eh, es = rel(a1, a0), rel(w1, w0), rel(h1, h0), rel(s1, s0)
        print(f'w13/w2 kernels layer {d["L"]} ({"NVFP4" if d["fp4"] else "MXFP8"}): experts {i1.tolist()}, rel diff '
              f'act {ea:.1e}, weights {ew:.1e}, mlp out {eh:.1e}, conv state {es:.1e}', flush=True)
        assert ea < 2e-2 and ew < 1e-5 and eh < 2e-2 and es < 1e-2


def test_fused_units():
    """INK_REAL=1 (~3 min): each small fused kernel == the model.py torch code on identical random inputs, real weights:
    attn-res mix (layer 0 FIRST, full visit with 3 queries + norm, gated core visit with the 4 loop messages + new
    block entry), loop gate (sliding + full), q/k/v prologue (pre: clip/scale/tau, static + dynamic conv + state,
    K clip, V norm, ring + 4-bit page write), near attention (in-kernel rel bias, sink, far partial merge, gate), wo
    (+ attn conv + residual), router (+ fused mlp norm)."""
    if os.environ.get('INK_REAL') != '1':
        print('fused unit kernel checks skipped (INK_REAL=1 to run)')
        return
    import kvpool
    K, m = fused_model()
    W, H, F32 = m.W, M.H, True
    dec = D.Decoder(m, graph=False, fuse=True)
    st = dec.st
    seq = m.new_seq()
    assert m.pool.reserve(seq.kv, 2048)
    st.kv.t.copy_(seq.kv.t)
    F = K.Fused(m, st, dec.x_raw, dec.out)
    g = torch.Generator().manual_seed(11)
    rn = lambda *sh: torch.randn(*sh, generator=g)
    rel = lambda a, b: ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp(min=1e-9)).item()
    res = {}

    def check(name, a, b, tol):
        e = rel(a, b)
        res[name] = e
        assert e < tol, (name, e)

    F.B.copy_(M.rms(rn(8, H)))
    # ---- mix
    for v in (0, 5, 30):
        d = F.vis[v]
        hin = rn(H).bfloat16()
        if d['msg']:
            pv = rn(H).bfloat16()
            K._mix_kernel[(1 + 4,)](d['q'], F.B, d['ent'], d['n'], hin, d['new'], F.A, F.X, d['nwt'], d['mqs'], d['pslot'],
                                    1, pv, F.M, 1, FIRST=False, NORM=False, STORE_A=True, F32=F32, H=H, CH=K.CH)
            ent = [F.B[e] for e in d['ent'][:d['n']].tolist()]
            check('mix a (gated)', F.A[0], M.attn_res(d['q'], [x[None] for x in ent] + [M.rms(hin)[None]])[0, 0], 1e-4)
            ref_m = M.attn_res(d['mqs'], [F.B[int(d['pslot'][0])][None], M.rms(pv)[None]])[:, 0]
            check('mix loop messages', F.M, ref_m, 1e-4)
            check('mix new block entry', F.B[d['new']], M.rms(hin), 1e-6)
        else:
            nq = 3 if d['full'] else 1
            K._mix_kernel[(nq,)](d['q'], F.B, d['ent'], d['n'], hin, d['new'], F.A, F.X, d['nwt'], d['q'], d['ent'], 1,
                                 hin, F.M, nq, FIRST=d['L'] == 0, NORM=True, STORE_A=False, F32=F32, H=H, CH=K.CH)
            ent = [F.B[e][None] for e in d['ent'][:d['n']].tolist()]
            a = hin.float()[None] if d['L'] == 0 else M.attn_res(d['q'], ent + [M.rms(hin)[None]])[:, 0]
            for j in range(nq):
                check(f'mix+norm v{v} q{j}', F.X[j], M.rms(a[j], d['nws'][j]).bfloat16(), 1e-2)
    # ---- loop gate
    for v in (29, 30):                                                # layer 17 (full, 3 inputs), layer 12 (sliding)
        d = F.vis[v]
        xs = H if d['full'] else 0
        F.A.copy_(rn(3, H))
        F.M.copy_(rn(4, H))
        K._gate_z_kernel[(3, 64 // K.BLK['gate'])](F.A, xs, F.M, d['dtab'], F.Z, F32=F32, H=H, CH=K.CH, ZB=K.BLK['gate'])
        K._gate_out_kernel[(3, H // K.BLK['gout'])](F.A, xs, F.M, d['gtab'], F.Z, F.G, H=H, BH=K.BLK['gout'])
        for j, c in enumerate(('attn', 'k', 'v')):
            check(f'gate v{v} {c}', F.G[j], m.gate(d['L'], c, F.A[j if d['full'] else 0][None], F.M[j][None])[0], 1e-2)
    # ---- pre (prologue), near, wo, router
    p = 1500
    st.pos.fill_(p)
    q_ = torch.arange(p - 1023, p + 1)
    st.rpos[q_ % 1024] = q_                                          # full ring: positions p-1023 .. p
    for v in (29, 30):
        d = F.vis[v]
        L, A = d['L'], f'layers.{d["L"]}.attn.'
        full = d['full']
        rk, rv = st.ring[v]
        rk.copy_(M.l2clip(rn(1024, 8, 128), M.CLIP_QK).bfloat16())
        rv.copy_(M.rms(rn(1024, 8, 128)).bfloat16())
        cv = st.conv[v]
        for k in cv:
            cv[k].copy_(rn(*cv[k].shape).bfloat16())
        c0 = {k: x.clone() for k, x in cv.items()}
        F.Y.copy_((rn(8664) * 0.5).bfloat16())
        y = F.Y.clone()
        kvp = d['kv'] if full else (F.Y,) * 5
        K._pre_kernel[(8,)](F.Y, d['koff'], d['voff'], d['tkoff'], d['tvoff'], d['qsc'], d['rsc'], d['kw'], d['vw'],
                            cv['k'], cv['v'], rk, rv, st.rpos, st.pos, *kvp, F.Qb, F.Rb, FULL=full, F32=F32, PG=kvpool.PAGE)
        n0, n1 = d['n0'], d['n1']
        Q = M.l2clip(y[:3072].view(1, 24, 128).float() * W[A + 'q_l2_clip.scale'], M.CLIP_QK).bfloat16()
        R = M.l2clip(y[3072:3456].view(1, 24, 16).float() * W[A + 'rel_l2_clip.scale'], M.CLIP_REL).bfloat16()
        Kx, Vx = y[n0:n0 + 1024][None], y[n0 + n1:n0 + n1 + 1024][None]
        sk, sv = c0['k'].clone(), c0['v'].clone()
        if full:
            tau = 1 + 0.1 * torch.log(torch.clamp(torch.tensor([p + 1.]) / 8192, min=1))
            Q, R = (Q.float() * tau).bfloat16(), (R.float() * tau).bfloat16()
            Kc = M.sconv(Kx, sk, dyn=(W[A + 'k_sconv.static_w'], y[n0 + 1024:n0 + n1].view(1, 8, 4)))
            Vc = M.sconv(Vx, sv, dyn=(W[A + 'v_sconv.static_w'], y[n0 + n1 + 1024:].view(1, 8, 4)))
        else:
            Kc, Vc = M.sconv(Kx, sk, W[A + 'k_sconv.weight']), M.sconv(Vx, sv, W[A + 'v_sconv.weight'])
        Kc = M.l2clip(Kc.view(1, 8, 128), M.CLIP_QK).bfloat16()
        Vc = M.rms(Vc.view(1, 8, 128)).bfloat16()
        check(f'pre v{v} Q', F.Qb, Q[0], 1e-2)
        check(f'pre v{v} R', F.Rb, R[0], 1e-2)
        check(f'pre v{v} ring K', rk[p % 1024], Kc[0], 1e-2)
        check(f'pre v{v} ring V', rv[p % 1024], Vc[0], 1e-2)
        check(f'pre v{v} conv state', torch.cat((cv['k'], cv['v'])), torch.cat((sk, sv)), 1e-6)
        if full:
            got = st.kv.get(m.slot[v], p, p + 1)
            for x, r_ in zip(got, (rk[p % 1024][None], rv[p % 1024][None])):  # KV4.quant of what the kernel wrote
                q_, s_ = M.KV4.quant(r_)
                deq = kvpool.deq(q_, s_)
                check(f'pre v{v} 4-bit page row', x, deq, 1e-6)
        # near (+ far partials for the full visit): kernel vs DecodeState.attend's torch math
        F.fo.copy_(rn(*F.fo.shape))
        F.fl.copy_(rn(*F.fl.shape) + 3)
        F.near(d['proj'], rk, rv, full)
        dd = p - st.rpos
        ok = (st.rpos >= 0) & (dd >= 0) & (dd < 1024)
        S = torch.einsum('tkgd,skd->kgts', F.Qb.view(1, 8, 3, 128), rk) * 128 ** -0.5
        tb = (F.Rb @ d['proj'])[:, None]                                # [24,1,1024]
        bias = tb.gather(2, dd.clamp(0, 1023).view(1, 1, 1024).expand(24, 1, 1024)).view(8, 3, 1, 1024)
        far = (F.fo[0][None], F.fl[0][None]) if full else None
        o = m.near_merge(S, bias, ok, y[3456:3480].float()[None], rv, far)
        ref_att = (o.view(1, 3072) * torch.sigmoid(y[3480:6552].float())).bfloat16()
        check(f'near v{v}', F.att, ref_att[0], 2e-2)
        # wo + attn conv + residual
        hin = rn(H).bfloat16()
        sa = cv['attn'].clone()
        K._wo_kernel[(H // K.BLK['wo'],)](F.att, d['wo'], cv['attn'], d['acw'], hin, F.R, d['wos'], BND=d['bnd'],
                                         F32=F32, H=H, CH=K.CH, BN=K.BLK['wo'], W8=F.w8)
        wo_ = (F.att[None].float() @ d['wo'].float().t() * (d['wos'] if F.w8 else 1)).bfloat16()
        oo = M.sconv(wo_, sa, W[f'layers.{L}.attn_sconv.weight'])
        check(f'wo v{v}', F.R, (hin.float() + oo.float())[0] if not d['bnd'] else oo.float()[0], 1e-2)
        check(f'wo v{v} conv state', cv['attn'], sa, 1e-2)
    # router with the fused mlp norm
    d = F.vis[30]
    F.G.copy_(rn(3, H))
    K._router_kernel[(128 // K.BLK['router'],)](F.G, d['mnw'], F.U, d['gw'], F.S, NORM=True, F32=F32, H=H, CH=K.CH,
                                                RB=K.BLK['router'])
    u = M.rms(F.G[0], d['mnw']).bfloat16()
    check('router u', F.U, u, 1e-2)
    check('router scores', F.S, torch.sigmoid(u.float() @ d['gw'].t()), 1e-4)
    print('fused unit kernels vs model.py torch code (max rel diff): ' + ', '.join(f'{k} {v:.0e}' for k, v in res.items()))


def test_fused_capture():
    """INK_REAL=1 (~2 min): the fused step is capture-safe: its torch ops (op_trace) and every Triton launch (grid,
    tensor addresses, scalars; kernels recorded instead of run) are identical at 3 positions, with embed_dev (token
    lookup inside the step) and without."""
    if os.environ.get('INK_REAL') != '1':
        print('fused capture check skipped (INK_REAL=1 to run)')
        return
    K, m = fused_model()
    seq = m.new_seq()
    assert m.pool.reserve(seq.kv, 4096 - 256)
    seq.pos = 1100
    for embed_dev in (False, True):
        fz = D.Decoder(m, graph=False, fuse=True, embed_dev=embed_dev)
        fz.load(seq)
        fz.fused = K.Fused(m, fz.st, fz.x_raw, fz.out)
        log, restore = launch_recorder([K, D])
        try:
            tr, ln = [], []
            for p in (1100, 1101, 3500):
                fz.st.pos.fill_(p)
                fz.tok.fill_(p)
                del log[:]
                tr.append(op_trace(fz._run))
                ln.append(list(log))
            assert tr[0] == tr[1] == tr[2] and ln[0] == ln[1] == ln[2], 'fused step not capture-safe'
            print(f'fused step capture-safe (embed_dev={embed_dev}): {len(tr[0])} torch ops + {len(ln[0])} kernel '
                  f'launches, identical at 3 positions')
        finally:
            restore()


def test_fused_real():
    """INK_REAL=1 (~15 min, CPU interpreter): INK_DEC_FUSE path (decode_kernels.Fused) == eager Decoder.
    State: 1020 prompt tokens prefilled with a stub MoE (fast; any valid state will do), then 5 lockstep decode steps
    with the real MoE, gated by KL vs a 1-ulp-nudge noise baseline: crosses position 1024 (ring full, far part starts, page writes), loop passes,
    NVFP4 + MXFP8 experts. All kernels real except proj/w13/w2 (torch mirrors, checked in test_fused_gemv).
    INK_DEC_FAR_DOT=1 runs it with the tensor-core far kernel."""
    if os.environ.get('INK_REAL') != '1':
        print('fused real-weights check skipped (INK_REAL=1 to run)')
        return
    import time
    K, m = fused_model()
    P, steps = 1020, 5
    g = torch.Generator().manual_seed(3)
    ids = torch.randint(1000, 150000, (P,), generator=g).tolist()
    moe = m.moe
    m.moe = lambda L, u, impl=None: (u.float() * 0.5).bfloat16()
    seq = m.new_seq()
    t0 = time.time()
    m.forward(m.embed(ids), seq)
    m.moe = moe
    print(f'prefill {P} (stub MoE) {time.time() - t0:.0f} s', flush=True)
    s2 = M.Seq('cpu', m.pool.fork(seq.kv, P))
    s2.pos, s2.conv, s2.ring = seq.pos, seq.conv, seq.ring
    for s in (seq, s2):
        assert m.pool.reserve(s.kv, P + steps + 1)
    ref, nud, fz = D.Decoder(m, graph=False, fuse=False), D.Decoder(m, graph=False, fuse=False), \
        D.Decoder(m, graph=False, fuse=True)
    ref.load(seq)
    nud.load(seq)                                                       # writes seq's KV rows first; ref then rewrites them
    fz.load(s2)
    fz.fused = K.Fused(m, fz.st, fz.x_raw, fz.out)
    restore = swap(K, gemv_mirrors(m, fz.fused))
    toks = torch.randint(1000, 150000, (steps,), generator=g).tolist()
    kld = lambda x, y: (torch.log_softmax(x, -1).exp() * (torch.log_softmax(x, -1) - torch.log_softmax(y, -1))).sum().item()
    try:
        for i, t in enumerate(toks):
            # lockstep: every step starts from the reference state; a 1-ulp embedding nudge of the reference is printed
            # as the noise scale (2026-11-21: fused 3.3e-3 vs nudge 4.0e-3 at one step; router near-ties flip under
            # either and the model amplifies them, so tight thresholds don't work on the whole step)
            for d in (nud, fz):
                for a_, b_ in zip(d.st.conv, ref.st.conv):
                    for k in a_:
                        a_[k].copy_(b_[k])
                for (a_, b_), (c_, d_) in zip(d.st.ring, ref.st.ring):
                    a_.copy_(c_)
                    b_.copy_(d_)
                d.st.rpos.copy_(ref.st.rpos)
            row = m.W.embed[t].clone()
            m.W.embed[t] = (row.float() * (1 + 2 ** -7)).bfloat16()
            n_ = nud.step(t).clone()
            nud.seq.pos -= 1                                            # nud shares seq with ref: undo its advance
            m.W.embed[t] = row
            t0 = time.time()
            a = ref.step(t).clone()
            t1 = time.time()
            b = fz.step(t).clone()
            kl, kn = kld(a, b), kld(a, n_)
            ia, ib = int(a.argmax()), int(b.argmax())
            gap = (a[0, ia] - a[0, ib]).item()                          # how far the fused pick is behind in the ref
            print(f'fused step pos {P + i} (lockstep): KL {kl:.2e} (1-ulp nudge baseline {kn:.2e}), max |dlogit| '
                  f'{(a - b).abs().max():.4f}, top-1 {ia} vs {ib} (ref gap {gap:.3f}) (eager {t1 - t0:.0f} s, '
                  f'fused {time.time() - t1:.0f} s)', flush=True)
            # gross-bug gate only (a real bug gives KL >= 1; 1-ulp baselines swing 5e-3..1.7e-1 on this random-token
            # state); per-kernel exactness is test_fused_units' job. A top-1 flip passes only on a near-tie: the
            # fused pick is within 0.25 logits of the ref's (2026-11-22: 12 vs 13 at KL 2e-2 after the near split)
            assert kl <= 2 * kn + 0.05 and gap <= 0.25 and bool(torch.isfinite(b).all()), (P + i, kl, kn, gap)
    finally:
        restore()
    assert torch.equal(fz.st.rpos, ref.st.rpos)
    ekv = max(((a.float() - b.float()).abs().mean() / b.float().abs().mean()).item()
              for sl in range(8) for a, b in zip(s2.kv.get(sl, P, P + steps), seq.kv.get(sl, P, P + steps)))
    print(f'4-bit KV rows written by the fused path vs eager: mean rel {ekv:.1e}')
    assert ekv < 0.05, ekv


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
    print('DECODE OK')
