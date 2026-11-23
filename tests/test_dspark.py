"""CPU: the DSpark drafter (dspark.py) and the spec loop (spec_runner.py). Default (~2 min): real mtp.safetensors
weights with a small fake vocab, fake targets.
1. loader: all 54 tensors with SPEC-dspark.md sec. 2 shapes, bf16.
2. draft: k ids, q rows are distributions; deterministic (greedy, and sampled under a fixed seed); draft() leaves the
   state untouched.
3. context: grows only by what extend() gets (spec loop with a fake target: anchor + accepted rows only) == one
   extend() over the same positions; rings: a 1024-slot sliding ring == an unwrapped one (window masking), the
   layer-1 cap == attention to the most recent `ctx` positions only; chunked extend == one extend.
4. masking: the batched eval path (rows_tf over a longer context, anchor at p) == draft() on a state holding exactly
   the first p rows.
5. serving path: stage() + round() (fixed shapes, single unembed read, U' = U @ hidden_out) == extend() + draft()
   (greedy ids; sampled q == softmax of the reference logits on the same prefix); capture safety: identical op traces
   for rounds with different anchors / positions / staged counts / ring wrap, no host sync inside.
6. spec loop on fake targets (FakeBD: logits a function of the token history, taps a function of (position, token)):
   greedy spec == plain greedy with an oracle drafter that drafts the true continuation with injected errors
   (all acceptance counts), stop ids, max_tokens, json grammar (real xgrammar, random logits): cut at the first
   rejected draft, output == plain; the real drafter in the loop: its context == one extend over the committed
   positions (no rejected row ever enters); sampled + grammar-rejected draft: emitted law == masked target (chi-square).
7. shortlist head (INK_DSPARK_SHORT): q only on static + noted ids, == masked reference softmax, drafts == race(q,
   noise), first draft ~ q (chi-square), dynamic ring; layer-1 context buckets (INK_DSPARK_CTXB) == whole ring;
   capture safety per (k, greedy, bucket) with note() in between; the spec loop with shortlist + buckets + Gate +
   accept_dev: greedy == plain, drafter context == extend, every sequence token in the head's support.
8. device draft ids (decoder dev_tok: bd.step(rows, tok={r0: ids})): greedy == plain incl. a stop id inside the draft,
   drafter context == extend (full head and shortlist); engine wiring on a stub Engine: FP8 decode -> drafter on
   unembed8, shortlist VRAM estimates, min_p, warm_drafter (greedy + sampled graphs, gate: k = 7).
INK_REAL=1 adds (real target weights):
   test_real: position-1 greedy acceptance on the recorded taps (tests/dspark_ablate.py --record, ~2 min);
   test_bd_taps: BatchDecoder tap rows == h entering visits 5/23/41 + final normed state, op traces per bucket stay
     identical (kernels recorded, not run; uses tests/test_decode_batch.py's cached prefill states, ~1 min);
   test_real_spec: greedy spec == plain greedy on a real code prompt (target on the Seq path, verify rows = single-row
     forwards with rollback: the batch-invariance contract of decode_batch), drafter acceptance per position
     (~15-20 min: ~17 s per CPU forward).
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
import torch  # noqa: E402

import chat  # noqa: E402
import dspark as D  # noqa: E402
import spec_runner as S  # noqa: E402
from dspark import BLOCK  # noqa: E402
from model import H  # noqa: E402

V = 1000      # fake vocab: ids < V index the real W1 rows
A1_MIN = 0.7  # floor for position-1 greedy acceptance on code1 (measured 0.88, SPEC-dspark.md "Measured")


def fake_drafter(V=V, dkw=None, **kw):
    g = torch.Generator().manual_seed(0)
    E = (torch.randn(V, H, generator=g) * 0.02).bfloat16()
    U = (torch.randn(V, H, generator=g) * 0.02).bfloat16()
    return D.DSpark(embed=E, unembed=U, embed_norm=torch.ones(H), cfg=D.Cfg(**kw), **(dkw or {}))


def short_drafter(**kw):
    """Shortlist head on the fake vocab: 300 static ids (random rank), 64 dynamic slots."""
    rank = torch.randperm(V, generator=torch.Generator().manual_seed(9))
    return fake_drafter(dkw=dict(short=300, dyn=64, rank=rank, **kw))


def taps(n, seed):
    return torch.randn(4, n, H, generator=torch.Generator().manual_seed(seed)).bfloat16()


def ring_eq(a, b, atol=0.0):
    for L in range(3):
        assert torch.equal(a.pos[L], b.pos[L]), f'layer {L}: ring positions differ'
        for x, y in ((a.k[L][:-1], b.k[L][:-1]), (a.v[L][:-1], b.v[L][:-1])):   # (last slot = trash)
            assert torch.allclose(x.float(), y.float(), atol=atol, rtol=atol), (L, (x.float() - y.float()).abs().max())
    assert torch.allclose(a.e, b.e, atol=1e-4) and a.n == b.n


def test_loader(ds):
    w = ds.w
    assert len(w) == 54, len(w)
    want = {'fuse_weight': (4, 4), 'mask_embed': (H,), 'norm.weight': (H,),
            'markov_head.markov_w1.weight': (201024, 256), 'markov_head.rnn_proj.weight': (768, 5120),
            'markov_head.rnn_proj.bias': (768,), 'markov_head.hidden_out.weight': (H, 256),
            'markov_head.hidden_out.bias': (H,), 'markov_head.dflare_norm.weight': (H,)}
    for L in range(3):
        P = f'layers.{L}.'
        want.update({P + n: (H,) for n in ('input_layernorm.weight', 'post_attention_layernorm.weight',
                                           'dflare_norm.weight')})
        A = P + 'self_attn.'
        want.update({A + 'q_proj.weight': (3072, H), A + 'o_proj.weight': (H, 3072)})
        want.update({A + n: (1024, H) for n in ('k_proj.weight', 'v_proj.weight', 'k_proj_ctx.weight',
                                                'v_proj_ctx.weight')})
        want.update({A + n: (128,) for n in ('q_norm.weight', 'k_norm.weight', 'k_norm_ctx.weight')})
        want.update({P + 'mlp.gate_proj.weight': (9216, H), P + 'mlp.up_proj.weight': (9216, H),
                     P + 'mlp.down_proj.weight': (H, 9216)})
    assert set(w) == set(want), set(w) ^ set(want)
    for k, s in want.items():
        assert tuple(w[k].shape) == s and w[k].dtype == torch.bfloat16, (k, w[k].shape, w[k].dtype)
    assert sum(t.numel() for t in w.values()) == 317_897_872
    print('loader: 54 tensors, shapes + bf16 ok')


def test_draft(ds):
    st = ds.new_state()
    ds.extend(st, taps(40, 1))
    snap = [t.clone() for t in st.k + st.v], st.e.clone(), st.n
    d1, q1 = ds.draft(st, 7, k=7)
    d2, _ = ds.draft(st, 7, k=7)
    assert len(d1) == 7 and q1 is None and d1 == d2 and all(0 <= t < V for t in d1), (d1, d2)
    s1 = ds.draft(st, 7, k=5, temperature=0.8, top_p=0.9, gen=torch.Generator().manual_seed(3))
    s2 = ds.draft(st, 7, k=5, temperature=0.8, top_p=0.9, gen=torch.Generator().manual_seed(3))
    assert s1[0] == s2[0] and torch.equal(s1[1], s2[1]) and s1[1].shape == (5, V)
    assert torch.allclose(s1[1].sum(-1), torch.ones(5), atol=1e-4)
    assert all(float(s1[1][i, t]) > 0 for i, t in enumerate(s1[0]))   # each draft came from its q row
    assert st.n == snap[2] and torch.equal(st.e, snap[1]) and all(torch.equal(a, b) for a, b in zip(st.k + st.v, snap[0]))
    print(f'draft: greedy {d1}, sampled {s1[0]}; deterministic, state untouched')


def test_ctx_growth(ds):
    """Spec loop with a fake target: taps of position t are a fixed function of t. Every round verifies k+1 rows but
    extends by the anchor + accepted rows only."""
    T = taps(200, 2)
    st, pos, acc = ds.new_state(), 30, []
    ds.extend(st, T[:, :pos])
    g = torch.Generator().manual_seed(5)
    for _ in range(6):
        d, _ = ds.draft(st, 11, k=4)
        a = int(torch.randint(0, len(d) + 1, (1,), generator=g))     # accepted drafts this round
        ds.extend(st, T[:, pos:pos + a + 1])                          # anchor + accepted (taps from the verify rows)
        pos += a + 1
        acc.append(a)
        assert st.n == pos and all(int((p >= 0).sum()) == pos for p in st.pos), (st.n, pos)
    ref = ds.new_state()
    ds.extend(ref, T[:, :pos])
    ring_eq(st, ref, atol=1e-2)
    print(f'ctx growth: accepted {acc} -> n = {st.n}, == one extend over the same positions')


def test_rings(ds):
    """n = 1500 > 1024: the sliding layers' 1024-slot rings == unwrapped 4096-slot rings (the window mask hides the
    rest); layer-1 cap 256 == an unwrapped layer 1 with positions < n - 256 masked out; extend in odd chunks == one
    extend (ring layout and values)."""
    n, x0 = 1500, 9
    T = taps(n, 6)
    a, big = ds.new_state(ctx=4096), D.State(ds.dev, ds.dt, (4096, 4096, 4096))
    ds.extend(a, T)
    ds.extend(big, T)
    assert sorted(a.pos[0][a.pos[0] >= 0].tolist()) == list(range(n - 1024, n))
    lr = ds.rows_tf(a, torch.tensor([x0]), torch.tensor([n]), torch.tensor([[x0] * 7]), a.e[None])
    lb = ds.rows_tf(big, torch.tensor([x0]), torch.tensor([n]), torch.tensor([[x0] * 7]), big.e[None])
    close = lambda x, y: (x - y).abs().max() <= 0.02 * y.abs().max() and torch.equal(x.argmax(-1), y.argmax(-1))
    assert close(lr, lb), (lr - lb).abs().max()          # bf16 rounding only (slot order differs)
    c = ds.new_state(ctx=256)
    ds.extend(c, T)
    big.pos[1].masked_fill_(big.pos[1] < n - 256, -1)
    lc = ds.rows_tf(c, torch.tensor([x0]), torch.tensor([n]), torch.tensor([[x0] * 7]), c.e[None])
    lb = ds.rows_tf(big, torch.tensor([x0]), torch.tensor([n]), torch.tensor([[x0] * 7]), big.e[None])
    assert close(lc, lb), (lc - lb).abs().max()
    ch = ds.new_state(ctx=4096)
    for s, e in ((0, 3), (3, 700), (700, 701), (701, 1500)):
        ds.extend(ch, T[:, s:e])
    ring_eq(ch, a, atol=1e-2)
    print('rings: sliding ring == unwrapped + window mask, layer-1 cap == most recent 256 only, chunked extend ok')


def test_mask(ds):
    """rows_tf with context [0, 50) and anchor at p = 37 == draft() on a state with exactly 37 rows."""
    T, p, x0 = taps(50, 4), 37, 9
    short, long_ = ds.new_state(), ds.new_state()
    ds.extend(short, T[:, :p])
    ds.extend(long_, T)
    d, _ = ds.draft(short, x0, k=8 if ds.cfg.row0 else 7)
    lg = ds.rows_tf(long_, torch.tensor([x0]), torch.tensor([p]), torch.tensor([[x0] + d[:-1]]),
                    ds.head_ctx(T)[p - 1:p])
    assert lg.argmax(-1)[0].tolist() == d, (lg.argmax(-1)[0].tolist(), d)
    print('mask: batched eval path == draft() at the same anchor')


def test_round(ds):
    """Serving path == reference over 5 rounds (staged counts 1..8, positions crossing 1024 for the sliding ring)."""
    T = taps(1200, 7)
    st, ref, pos = ds.new_state(), ds.new_state(), 1010
    ds.extend(st, T[:, :pos])
    ds.extend(ref, T[:, :pos])
    g = torch.Generator().manual_seed(1)
    flips = 0
    for r, (n, x0) in enumerate(((0, 5), (3, 17), (8, 400), (1, 3), (6, 999))):
        if n:
            ds.stage(st, T[:, pos:pos + n])
            ds.extend(ref, T[:, pos:pos + n])
            pos += n
        d, q = ds.round(st, x0, 7)
        dr, _ = ds.draft(ref, x0, k=7)
        assert st.n == ref.n == pos
        ring_eq(st, ref, atol=1e-2)
        flips += sum(a != b for a, b in zip(d, dr))
        # sampled: q rows == softmax(reference logits / T) along the drafted prefix
        ds_, qs = ds.round(st, x0, 5, temperature=0.7, gen=g)
        lg = ds.rows_tf(ref, torch.tensor([x0]), torch.tensor([pos]), torch.tensor([[x0] + ds_[:4]]), ref.e[None])[0]
        qr = torch.softmax(lg / 0.7, -1)
        assert qs.shape == (5, V) and torch.allclose(qs.sum(-1), torch.ones(5), atol=1e-4)
        assert all(float(qs[i, t]) > 0 for i, t in enumerate(ds_))
        err = (qs - qr).abs().sum(-1).max().item()
        assert err < 0.1, err                          # bf16 rounding (h + b vs U h + U' tanh o) at T = 0.7
        e = [torch.empty(V).exponential_(generator=g) for _ in range(5)]   # caller's noise (seeded spec): same race
        ds_e, qe = ds.round(st, x0, 5, temperature=0.7, noise=e)
        assert ds_e == [int((qe[i] / e[i]).argmax()) for i in range(5)], ds_e
    assert flips <= 1, flips                           # greedy ids: same up to a rare near-tie
    print(f'round: stage + round == extend + draft over 5 rounds (greedy id flips {flips}/35), sampled q == '
          f'reference softmax (L1 <= {err:.1e})')


def test_capture_safe(ds):
    """Op traces of the device round (_body) identical across rounds with different anchors, positions (ring
    wrap at 1024), staged counts and temperatures; no host sync (no aten._local_scalar_dense / item)."""
    from test_decode import op_trace
    st = ds.new_state()
    ds.extend(st, taps(1020, 8))
    ds._prep(st)
    tr = {}
    for n, x0, k, greedy in ((0, 5, 7, True), (8, 6, 7, True), (2, 900, 7, True), (5, 7, 3, False), (1, 8, 3, False),
                             (3, 9, 0, True), (7, 10, 0, True)):
        if n:
            ds.stage(st, taps(n, 100 + n))
        io = st.io
        io.n.fill_(st.pend), io.x0.fill_(x0), io.xe.copy_(ds.embed[x0]), io.temp.fill_(0.5 + n)
        ops = op_trace(lambda: ds._body(st, k, greedy))
        st.n, st.pend = st.n + st.pend, 0
        tr.setdefault((k, greedy), []).append(ops)
        assert not any('_local_scalar_dense' in o[0] or 'item' in o[0] for o in ops), 'host sync in the round'
    for key, xs in tr.items():
        assert all(x == xs[0] for x in xs), f'{key}: op trace differs between rounds (not capture-safe)'
        print(f'capture: (k, greedy) {key}: {len(xs)} rounds, {len(xs[0])} ops, identical')
    assert st.n == int(st.nd) == 1020 + 8 + 2 + 5 + 1 + 3 + 7


def masked_ref(ds, ref, x0, pos, d, sup):
    """Reference drafter logits along the draft path with everything outside the shortlist support at -inf."""
    lg = ds.rows_tf(ref, torch.tensor([x0]), torch.tensor([pos]), torch.tensor([[x0] + d[:len(d) - 1]]),
                    ref.e[None])[0]
    m = torch.zeros(lg.shape[-1], dtype=torch.bool)
    m[list(sup)] = True
    return lg.masked_fill(~m, float('-inf')), m


def test_short(ds):
    """Shortlist head (static 300 + 64 dynamic slots on the 1000-token fake vocab): q lives exactly on the support
    (static + noted context tokens), == softmax of the masked reference logits, each draft == the exponential race of
    that q with the round's noise, greedy == masked argmax, conf == max masked softmax; the dynamic ring keeps the
    newest 64 tokens; reset empties it; drafts follow q (chi-square over many rounds)."""
    T, pos = taps(60, 7), 60
    st, ref = ds.new_state(), ds.new_state()
    ds.extend(st, T)
    ds.extend(ref, T)
    ds._prep(st)
    static = set(ds.sid.tolist())
    outside = [t for t in range(V) if t not in static]
    ds.note(st, outside[:10] + [next(iter(static))] + outside[:3])       # static / repeated ids add nothing
    assert len(st.dmap) == 10 and st.dptr == 10
    sup = static | set(st.dmap)
    g, flips = torch.Generator().manual_seed(4), 0
    for x0 in (5, 400):
        d, q, conf = ds.round(st, x0, 7, temperature=0.7, gen=g, conf=True)
        lg, m = masked_ref(ds, ref, x0, pos, d, sup)
        assert torch.all(q[:, ~m] == 0) and torch.allclose(q.sum(-1), torch.ones(7), atol=1e-4)
        assert (q - torch.softmax(lg / 0.7, -1)).abs().sum(-1).max() < 0.1
        assert max(abs(c - float(torch.softmax(lg[i], -1).max())) for i, c in enumerate(conf)) < 0.05
        io = st.io
        for i in range(7):                             # the draft is the race of q (pads masked) with this noise
            qs = q[i, io.sid] * (io.mfull > float('-inf'))
            assert int(io.sid[D.race(qs[None], io.noise[i:i + 1])]) == d[i]
        # seeded spec (engine): the caller's per-position noise in vocab order [V]; draft i = argmax over the vocab of
        # q_i / e_i, the same race the target runs at that position (coupled: equal whenever both pick the same token)
        e = [torch.empty(V, dtype=torch.float64).exponential_(generator=g) for _ in range(7)]
        st2 = ds.new_state(); ds.extend(st2, T); ds.note(st2, outside[:10])
        d2, q2 = ds.round(st2, x0, 7, temperature=0.7, conf=False, noise=e)
        assert d2 == [int((q2[i].double() / e[i]).argmax()) for i in range(7)], d2
        dg, _, cg = ds.round(st, x0, 7, conf=True)
        lg, _ = masked_ref(ds, ref, x0, pos, dg, sup)
        flips += sum(a != b for a, b in zip(dg, lg.argmax(-1).tolist()))
        assert all(t in sup for t in d + dg)
    assert flips <= 1, flips
    ds.note(st, outside[10:200])                       # ring of 64: the newest 64 noted ids stay
    assert set(st.dmap) == set(outside[200 - 64:200]) and bool((st.io.md == 0).all())
    assert set(st.io.sid[len(ds.sid):].tolist()) == set(outside[200 - 64:200])
    # drafts follow q end to end: 300 rounds at a fixed state/anchor (q constant), first-draft histogram vs q (coarse;
    # the per-round race check above is the exact one, test_spec checks race + accept with shortlist q at 30k draws)
    from test_spec import chi2_ok
    n, cnt = 300, torch.zeros(V)
    for _ in range(n):
        d, q = ds.round(st, 5, 1, temperature=1.3, gen=g)
        cnt[d[0]] += 1
    x2, crit = chi2_ok(cnt, q[0].double(), n)
    assert x2 < crit, (x2, crit)
    ds.reset(st)
    assert not st.dmap and bool((st.io.md == float('-inf')).all())
    print(f'shortlist: q on the support only, == masked reference softmax, drafts == race(q, noise), greedy flips '
          f'{flips}/14, ring keeps the newest 64; first draft ~ q (chi2 {x2:.1f} < {crit:.1f})')


def test_short_fp8():
    """unembed8 (the target's FP8 unembed, INK_FP8_DECODE): no bf16 unembed needed; the shortlist head built from it ==
    the head built from its dequantized bf16 values (same drafts, same q, same dynamic rows)."""
    from fp8_dense import quant_w
    ref = short_drafter()
    q8, s8 = quant_w(ref.U)
    ref.U = (q8.float() * s8[:, None]).bfloat16()
    f8 = D.DSpark(embed=ref.embed, unembed=None, unembed8=(q8, s8), embed_norm=torch.ones(H), short=300, dyn=64,
                  rank=ref.rank)
    T = taps(40, 3)
    sa, sb = ref.new_state(), f8.new_state()
    ref.extend(sa, T)
    f8.extend(sb, T)
    ref.note(sa, list(range(V)))
    f8.note(sb, list(range(V)))
    for x0 in (5, 600):
        ga, gb = torch.Generator().manual_seed(x0), torch.Generator().manual_seed(x0)
        da, qa = ref.round(sa, x0, 7, temperature=0.9, gen=ga)
        db, qb = f8.round(sb, x0, 7, temperature=0.9, gen=gb)
        assert da == db and torch.equal(qa, qb), (da, db)
        assert ref.round(sa, x0, 7)[0] == f8.round(sb, x0, 7)[0]
    assert f8.U is None and f8.V == V
    print('shortlist on the FP8 unembed == on its dequantized bf16 values (drafts, q); no bf16 unembed held')


def test_short_reuse():
    """11-23 review 2: a shortlist State reused for a second sequence (free list, dormant wake, attach / prompt gap
    resets) keeps working: reset clears the dynamic slots too (note() raised KeyError on stale slots), and the reused
    State drafts like a fresh one."""
    ds = short_drafter()
    st = ds.new_state()
    T = taps(30, 4)
    ds.extend(st, T)
    ds.note(st, list(range(V)))                              # fills (and wraps) the 64 dynamic slots
    ds.round(st, 5, 7)
    ds.reset(st, 0)
    ds.extend(st, T)
    ds.note(st, [999, 998, 997])
    fresh = ds.new_state()
    ds.extend(fresh, T)
    ds.note(fresh, [999, 998, 997])
    g1, g2 = torch.Generator().manual_seed(3), torch.Generator().manual_seed(3)
    a, qa = ds.round(st, 7, 7, temperature=0.8, gen=g1)
    b, qb = ds.round(fresh, 7, 7, temperature=0.8, gen=g2)
    assert a == b and torch.equal(qa, qb) and st.dslot == fresh.dslot and st.dmap == fresh.dmap and st.dmap
    print('shortlist State reuse: reset clears the dynamic slots; reused State == fresh State (drafts, q)')


def test_gate_causal():
    """11-23 review 3: Gate.pick is a stopping rule (verifying d_j may depend only on conf[0..j-1]), so sampled spec
    stays exact even when a later row's confidence depends on the drafted token (the review's toy: conf[2] high only
    when d_2 = 0; the old argmax gave P = 0.901 instead of 0.800)."""
    from spec import Gate, accept
    gen = torch.Generator().manual_seed(1)
    q = torch.tensor([[0.5, 0.5]] * 7)
    p = torch.tensor([[0.5, 0.5], [0.8, 0.2], [0.5, 0.5], [0.5, 0.5]])
    N, hit, n2, ms = 20000, 0, 0, set()
    for _ in range(N):
        d = torch.multinomial(q, 1, generator=gen).view(-1).tolist()
        conf = [0.96, 0.3, 0.99 if d[1] == 0 else 0.05] + [0.05] * 4
        m = Gate().pick(conf, 7)
        ms.add(m)
        out = accept(p[:m + 1], d[:m], q[:m], gen=gen)
        if len(out) >= 2 and out[0] == d[0]:
            n2 += 1
            hit += out[1] == 0
    f = hit / n2
    assert abs(f - 0.8) < 4 * (0.16 / n2) ** 0.5, (f, n2)
    # the rule itself: conf beyond the current width never changes the choice of that width
    cg = torch.Generator().manual_seed(2)
    for _ in range(300):                                     # choosing m reads conf[0..m] only
        c = torch.rand(7, generator=cg).tolist()
        m = Gate().pick(c, 7)
        ms.add(m)
        for j in range(m + 1, 7):
            c2 = list(c)
            c2[j] = float(torch.rand((), generator=cg))
            assert Gate().pick(c2, 7) == m, (c, c2, m, j)
    print(f'gate stopping rule: P(token 2 = 0 | token 1 kept) = {f:.3f} (exact 0.800, n {n2}), widths {sorted(ms)}')


def test_ctx_bucket():
    """INK_DSPARK_CTXB: layer 1 reads only the filled ring prefix (power-of-2 >= 1024) while the context fits; same
    drafts and q as reading the whole ring; buckets per context length."""
    a, b = fake_drafter(dkw=dict(ctxb=True)), fake_drafter(dkw=dict(ctxb=False))
    T = taps(1500, 11)
    sa, sb = a.new_state(ctx=4096), b.new_state(ctx=4096)
    assert a.bucket(sa) == 1024
    a.extend(sa, T)
    b.extend(sb, T)
    assert a.bucket(sa) == 2048 and b.bucket(sb) is None
    g1, g2 = torch.Generator().manual_seed(5), torch.Generator().manual_seed(5)
    for x0 in (3, 77):
        da, qa = a.round(sa, x0, 7, temperature=0.8, gen=g1)
        db, qb = b.round(sb, x0, 7, temperature=0.8, gen=g2)
        assert da == db and (qa - qb).abs().sum(-1).max() < 1e-3, (da, db, (qa - qb).abs().sum(-1).max())
        assert a.round(sa, x0, 7)[0] == b.round(sb, x0, 7)[0]
    st = a.new_state(ctx=4096)
    for n, want in ((0, 1024), (1024, 1024), (1025, 2048), (2049, None), (4096, None)):
        st.n = n
        assert a.bucket(st) == want, (n, a.bucket(st), want)
    st.n, st.pend = 1020, 5
    assert a.bucket(st) == 2048
    print('ctx buckets: layer-1 prefix 2048 of 4097 slots == whole ring (drafts, q); buckets 1024/2048/full')


def test_capture_short():
    """Capture safety with the shortlist + buckets: identical op traces per (k, greedy, bucket) across rounds with
    note() (dynamic slots changing) in between, no host sync."""
    from test_decode import op_trace
    ds = short_drafter(ctxb=True)
    st = ds.new_state()
    ds.extend(st, taps(700, 8))
    ds._prep(st)
    tr = {}
    for i, (n, x0, k, greedy) in enumerate(((0, 5, 7, True), (3, 6, 7, True), (2, 900, 7, False), (5, 7, 7, False),
                                            (1, 8, 3, True), (4, 9, 3, True))):
        ds.note(st, [400 + 37 * i + j for j in range(5)])
        if n:
            ds.stage(st, taps(n, 100 + n))
        io = st.io
        io.n.fill_(st.pend), io.x0.fill_(x0), io.xe.copy_(ds.embed[x0]), io.temp.fill_(0.5 + n)
        nb = ds.bucket(st)
        ops = op_trace(lambda: ds._body(st, k, greedy, nb))
        st.n, st.pend = st.n + st.pend, 0
        tr.setdefault((k, greedy, nb), []).append(ops)
    for key, xs in tr.items():
        assert all(x == xs[0] for x in xs), f'{key}: op trace differs between rounds (not capture-safe)'
        print(f'capture (shortlist, buckets): {key}: {len(xs)} rounds, {len(xs[0])} ops, identical')


# --------------------------------------------------------------------------------------------- spec loop, fakes
class FakeBD:
    """BatchDecoder's step/commit/tap_rows contract for one slot. Logits of a row = f(last 3 tokens of its history),
    taps of a position = f(position, token): wrong positions or rejected rows reaching the drafter show up."""

    def __init__(self, vocab, pos, hist, scale=3.0, dev_tok=False):
        self.V, self.pos, self.hist, self.pend, self.scale = vocab, pos, list(hist), None, scale
        self.steps = self.rows = self.devrows = 0
        self.dev_tok = dev_tok                             # step(rows, tok={r0: ids}) as BatchDecoder's

    def logits(self, h):
        g = torch.Generator().manual_seed(hash(tuple(h[-3:])) % (2 ** 31))
        return torch.randn(self.V, generator=g) * self.scale

    @staticmethod
    def tap(p, t):
        return torch.randn(4, H, generator=torch.Generator().manual_seed(p * 7919 + t)).bfloat16()

    def step(self, rows, tok=None):
        assert self.pend is None, 'commit first'
        rows = list(rows)
        for r0, t in (tok or {}).items():
            assert torch.is_tensor(t)
            self.devrows += len(t)
            for j, x in enumerate(t.tolist()):
                assert rows[r0 + j][2] is None, 'device rows carry no host token'
                rows[r0 + j] = rows[r0 + j][:2] + (x,)
        assert all(t is not None for _, _, t in rows), rows
        assert rows[0][1] == self.pos and all(r[1] == self.pos + i for i, r in enumerate(rows)), (rows, self.pos)
        toks = [t for _, _, t in rows]
        self.steps, self.rows = self.steps + 1, self.rows + len(rows)
        lg = torch.stack([self.logits(self.hist + toks[:i + 1]) for i in range(len(toks))])
        self.taps = torch.stack([self.tap(self.pos + i, t) for i, t in enumerate(toks)], 1)
        if len(rows) == 1:
            self.hist.append(toks[0])
            self.pos += 1
        else:
            self.pend = toks
        return lg

    def commit(self, acc):
        n = acc[0]
        assert self.pend is not None and 1 <= n <= len(self.pend)
        self.hist += self.pend[:n]
        self.pos += n
        self.pend = None

    def tap_rows(self, r0, n):
        return self.taps[:, r0:r0 + n]

    def tap_hist(self, slot, p0, n):
        """Taps of committed positions (from the committed history: a rejected row reaching the drafter shows up)."""
        assert self.pend is None and p0 + n <= self.pos and self.pos - p0 <= 24, (p0, n, self.pos)
        return torch.stack([self.tap(p, self.hist[p]) for p in range(p0, p0 + n)], 1)


class Oracle:
    """Drafter double with DSpark's serving interface: drafts the reference continuation ref[pos + 1 ..] with an
    error at a pseudo-random position (or none); sampled: q = a fixed random distribution per (pos, i), d ~ q."""

    def __init__(self, ref, vocab, err=0.4, seed=0, bad=()):
        self.ref, self.V, self.err, self.g, self.bad = ref, vocab, err, torch.Generator().manual_seed(seed), list(bad)
        self.U, self.dev = torch.zeros(vocab, 1), torch.device('cpu')

    def new_state(self, start=0):
        from types import SimpleNamespace
        return SimpleNamespace(n=start, pend=0)

    def reset(self, st, start=0):
        st.n, st.pend = start, 0

    def extend(self, st, taps):
        st.n += taps.shape[1]

    def stage(self, st, taps):
        st.n, st.pend = st.n + st.pend, taps.shape[1]

    def flush(self, st):
        st.n, st.pend = st.n + st.pend, 0

    def free(self, st):
        pass

    def round(self, st, x0, k, temperature=0.0, gen=None, conf=False, host=True, noise=None):
        d, q = self._round(st, x0, k, temperature)
        if not host:
            return torch.tensor(d), q
        return (d, q, torch.rand(k, generator=self.g).tolist()) if conf else (d, q)

    def _round(self, st, x0, k, temperature):
        st.n, st.pend = st.n + st.pend, 0
        p = st.n
        if temperature > 0:
            q = torch.softmax(torch.randn(k, self.V, generator=self.g) * 2, -1)
            if self.bad:                               # put mass on tokens a grammar rejects
                q[:, self.bad] += 1.0
                q = q / q.sum(-1, keepdim=True)
            return torch.multinomial(q, 1, generator=self.g)[:, 0].tolist(), q
        d = [self.ref[p + 1 + i] if p + 1 + i < len(self.ref) else 0 for i in range(k)]
        if float(torch.rand((), generator=self.g)) < self.err:
            j = int(torch.randint(0, k, (), generator=self.g))
            d[j] = (d[j] + 1 + int(torch.randint(0, self.V - 1, (), generator=self.g))) % self.V
        return d, None


def test_spec_greedy_fake():
    """Greedy spec == plain greedy (oracle drafter with errors: every acceptance count), max_tokens and stop ids."""
    Vf, P, N = 64, 20, 60
    hist = list(range(P))
    plain_bd = FakeBD(Vf, P, hist)
    ref, _ = S.generate(plain_bd, 0, P, 7, N, stop=())
    full = hist + ref
    for seed, err in ((0, 0.0), (1, 0.4), (2, 0.9)):
        bd = FakeBD(Vf, P, hist)
        out, sp = S.generate(bd, 0, P, 7, N, ds=Oracle(full, Vf, err, seed), taps=torch.zeros(4, P, H), stop=(),
                             times=(1.0,) * 8)
        assert out == ref, (seed, out, ref)
        s = sp.stats
        assert bd.pos - BLOCK <= sp.st.n <= bd.pos and sp.st.pend == 0 and s['emitted'] + s['widths'][0] == N - 1
        assert bd.steps == s['rounds'] + s['widths'][0], (bd.steps, s)
        print(f'  oracle err {err}: {s["rounds"]} rounds for {N - 1} tokens, accepted {s["accepted"]}/{s["drafted"]}, '
              f'per position {[f"{a}/{b}" for a, b in zip(s["kept"], s["tried"])]}')
    # stop id inside an accepted draft: output ends there (no bonus after it), same as plain
    stop = ref[25]
    plain, _ = S.generate(FakeBD(Vf, P, hist), 0, P, 7, N, stop=(stop,))
    out, _ = S.generate(FakeBD(Vf, P, hist), 0, P, 7, N, ds=Oracle(full, Vf, 0.0), taps=torch.zeros(4, P, H),
                        stop=(stop,), times=(1.0,) * 8)
    assert out == plain and out[-1] == stop, (out, plain)
    for mt in (1, 2, 9):
        out, _ = S.generate(FakeBD(Vf, P, hist), 0, P, 7, mt, ds=Oracle(full, Vf, 0.0), taps=torch.zeros(4, P, H),
                            stop=(), times=(1.0,) * 8)
        assert out == ref[:mt], (mt, out)
    # device draft ids (bd.dev_tok): the verify gets the ids as a device tensor, the draft isn't cut at a stop id
    # before the verify (settle cuts the output): same outputs
    for seed, err, st_ in ((5, 0.0, (stop,)), (6, 0.5, ()), (7, 0.9, ())):
        bd = FakeBD(Vf, P, hist, dev_tok=True)
        out, sp = S.generate(bd, 0, P, 7, N, ds=Oracle(full, Vf, err, seed), taps=torch.zeros(4, P, H), stop=st_,
                             times=(1.0,) * 8)
        want = plain if st_ else ref
        assert out == want and bd.devrows > 0 and sp.dev_tok, (seed, out, want, bd.devrows)
    for seed, err in ((3, 0.0), (4, 0.6)):             # confidence-gated widths (random confidences)
        out, sp = S.generate(FakeBD(Vf, P, hist), 0, P, 7, N, ds=Oracle(full, Vf, err, seed), taps=torch.zeros(4, P, H),
                             stop=(), times=(1.0,) * 8, gate=True)
        assert out == ref and sp.gate is not None and sp.stats['rounds'] > 0, (seed, out)
        assert sp.lg is not None and sp.lg.shape[1] == Vf  # verify rows' target logits kept for logprobs
        assert set(k for k, n in enumerate(sp.stats['widths']) if n) <= {0, 1, 3, 7}, sp.stats['widths']
    print('spec greedy (fake target): == plain greedy for oracle error 0 / 0.4 / 0.9, stop id, max_tokens 1/2/9, '
          'gated widths {0,1,3,7}, device draft ids (incl. a stop id inside the draft)')


def test_spec_sleep():
    """Dormant spec (INK_SPEC_SLEEP / WAKE): when drafting never pays, after SLEEP rounds off the request decodes
    plain with no drafter call at all (no catch-up, no probes) for WAKE positions, then the drafter context restarts at
    the current position and probing resumes; when drafting pays it never sleeps. Greedy output == plain."""
    Vf, P, N = 64, 20, 400
    hist = list(range(P))
    ref, _ = S.generate(FakeBD(Vf, P, hist), 0, P, 7, N, stop=())
    old = S.SLEEP, S.WAKE
    S.SLEEP, S.WAKE = 16, 64
    try:
        for times, err, want_sleep in (((1.0,) + (100.0,) * 7, 0.9, True), ((1.0,) * 8, 0.0, False)):
            orc = Oracle(hist + ref, Vf, err, 3)
            calls = []
            for name in ('round', 'stage', 'flush', 'reset'):
                f = getattr(orc, name)
                setattr(orc, name, lambda *a, f=f, name=name, **k: (calls.append((name, bd.pos)), f(*a, **k))[1])
            bd = FakeBD(Vf, P, hist)
            out, sp = S.generate(bd, 0, P, 7, N, ds=orc, taps=torch.zeros(4, P, H), stop=(), times=times)
            assert out == ref, (times, out[:20], ref[:20])
            s = sp.stats
            assert (s.get('sleeps', 0) > 0) == want_sleep, s
            if want_sleep:
                assert s['slept'] > N // 2, s
                wakes = [p for n, p in calls if n == 'reset']
                assert wakes, calls[-5:]
                w0 = wakes[0]                              # no drafter call in the WAKE positions before a wake
                assert not [c for c in calls if w0 - S.WAKE + 2 <= c[1] < w0], [c for c in calls if c[1] < w0][-3:]
                print(f'spec sleep: {s["sleeps"]} sleeps, {s["slept"]}/{N} steps dormant (no drafter call), '
                      f'wakes at {wakes[:4]}, output == plain')
    finally:
        S.SLEEP, S.WAKE = old
    print('spec sleep: drafting that pays never sleeps; output == plain both ways')


def test_spec_real_drafter_ctx(ds, dev_tok=False, **kw):
    """The real drafter in the loop on a fake target: after every round the drafter context == one extend over the
    taps of the committed positions (rejected rows never enter), and greedy output == plain. kw: Spec options;
    dev_tok: the decoder takes device draft ids (DSpark.round(host=False))."""
    P, N = 30, 40
    hist = torch.randint(0, V, (P,), generator=torch.Generator().manual_seed(3)).tolist()
    ref, _ = S.generate(FakeBD(V, P, hist), 0, P, 11, N, stop=())
    T0 = torch.stack([FakeBD.tap(p, t) for p, t in enumerate(hist)], 1)
    # flat costs: always drafting; drafting 100x dearer: k = 0 rounds (rows buffered, flushed 8 at a time) + probes
    for times in ((1.0,) * 8, (1.0,) + (100.0,) * 7):
        bd = FakeBD(V, P, hist, dev_tok=dev_tok)
        out, sp = S.generate(bd, 0, P, 11, N, ds=ds, taps=T0, stop=(), kmax=7, times=times, prompt=hist, **kw)
        assert out == ref
        assert (bd.devrows > 0) == (dev_tok and sp.stats['rounds'] > 0 and sp.gate is None), (bd.devrows, dev_tok)
        sp.flush(bd.pos)
        assert sp.st.n == bd.pos, (sp.st.n, bd.pos)
        toks = hist + out
        full = torch.stack([FakeBD.tap(p, t) for p, t in enumerate(toks[:bd.pos])], 1)
        rf = ds.new_state()
        ds.extend(rf, full)
        ring_eq(sp.st, rf, atol=1e-2)
        s = sp.stats
        if ds.short:                                    # every token of the sequence is in the head's support
            sup = set(ds.sid.tolist()) | set(sp.st.dmap)
            assert set(toks[:-1]) <= sup, set(toks[:-1]) - sup   # (the last, plain-sampled token is noted next round)
        print(f'spec + real drafter (fake target){" " + str(kw) if kw else ""}{" shortlist" if ds.short else ""}: == '
              f'plain, context == extend over the {bd.pos} committed positions '
              f'({s["rounds"]} rounds, widths {s["widths"]}, drafted {s["drafted"]}, accepted {s["accepted"]})')


def test_spec_grammar():
    """json_schema on a fake 200k-vocab target with random logits: greedy spec (oracle drafts with errors, some
    of them grammar-invalid) == plain greedy, the reply parses; grammar state consistent after rollbacks."""
    import json

    from tokenizers import Tokenizer

    import grammar
    from engine import MODEL_DIR
    tok = Tokenizer.from_file(os.path.join(MODEL_DIR, 'tokenizer.json'))
    schema = {'type': 'object', 'properties': {'name': {'type': 'string', 'maxLength': 12}, 'n': {'type': 'integer'},
                                               'tags': {'type': 'array', 'items': {'type': 'string', 'maxLength': 6}}},
              'required': ['name', 'n', 'tags']}
    cg = grammar.compile_format(tok, {'type': 'json_schema', 'json_schema': {'schema': schema}})
    Vf, P, N = grammar.VOCAB, 12, 48
    hist = list(range(1000, 1000 + P))

    def run(ds=None, seed=0):
        bd = FakeBD(Vf, P, hist, scale=1.0)
        cons = grammar.Json(cg, in_text=True)
        first_lg = bd.logits(hist + [0])
        cons.mask(first_lg)
        first = int(first_lg.argmax())
        return S.generate(bd, 0, P, first, N, ds=ds, taps=torch.zeros(4, P, H), cons=cons,
                          stop=(chat.EOS, chat.END), times=(1.0,) * 8)

    ref, _ = run()
    full = hist + ref
    for seed, err in ((0, 0.0), (1, 0.7)):
        out, sp = run(Oracle(full, Vf, err, seed), seed)
        assert out == ref, (seed, out[:20], ref[:20])
        s = sp.stats
        print(f'  grammar, oracle err {err}: {s["rounds"]} rounds, accepted {s["accepted"]}/{s["drafted"]}')
    txt = tok.decode([t for t in ref if t < chat.USER])
    if ref[-1] == chat.END:
        json.loads(txt)
    print(f'spec + json grammar: == plain greedy ({len(ref)} tokens, ended {"END" if ref[-1] == chat.END else "length"}): '
          f'{txt[:60]!r}')


def test_spec_sampled_bad():
    """Sampled round where drafts the grammar rejects carry most of q: the first emitted token ~ the masked target
    p (chi-square), i.e. the rejected draft goes through accept() (residual of its row), not a plain bonus draw."""
    from test_spec import chi2_ok
    Vf, P, n = 8, 5, 6000
    bad = [5, 6, 7]

    class Cons:            # grammar double: tokens 5..7 never allowed
        state, done = 'free', False

        def mask(self, x):
            x[bad] = float('-inf')
            return x

        def accept(self, t):
            assert t not in bad

    hist = list(range(P))
    bd0 = FakeBD(Vf, P, hist)
    lg = bd0.logits(hist + [1])
    lg[bad] = float('-inf')
    p = torch.softmax(lg / 0.9, -1)
    counts = torch.zeros(Vf)
    orc = Oracle([], Vf, seed=11, bad=bad)
    gen = torch.Generator().manual_seed(12)
    nb = 0
    for i in range(n):
        bd = FakeBD(Vf, P, hist)
        sp = S.Spec(orc, bd, 0, start=P, kmax=2, temperature=0.9, gen=gen, cons=Cons(), stop=(), times=(1.0,) * 8)
        rows, ctx = sp.propose(P, 1, 10)
        nb += ctx[3] is not None
        out = sp.settle(bd.step(rows), 0, ctx)
        counts[out[0]] += 1
    x2, crit = chi2_ok(counts, p, n)
    assert x2 < crit, (x2, crit, counts / n, p)
    print(f'sampled + grammar-rejected drafts ({nb}/{n} rounds had one): first token ~ masked p, chi2 {x2:.1f} < {crit:.1f}')


class CycBD(FakeBD):
    """FakeBD whose text mostly copies a cycle C (so prompt-lookup drafts get accepted), with detours."""
    C = list(range(7, 19))

    def logits(self, h):
        lg, C = super().logits(h), self.C
        if h[-1] in C and hash(tuple(h[-2:])) % 7:
            lg[C[(C.index(h[-1]) + 1) % len(C)]] += 12
        elif h[-1] not in C:
            lg[C[0]] += 12
        return lg


def test_spec_hybrid(ds):
    """INK_SPEC_HYBRID: copy rounds (no drafter call), DSpark rounds and DSpark + copy (mixed) rounds; greedy == plain
    (oracle drafter with errors, and the real drafter: its context == one extend over the committed positions, so the
    up-to-16-row copy verifies never outrun the decoder's tap ring), max_tokens / stop id inside a copy."""
    Vf, P, N = 64, 24, 150
    hist = CycBD.C * 2
    ref, _ = S.generate(CycBD(Vf, P, hist), 0, P, CycBD.C[0], N, stop=())
    full = hist + ref
    for seed, err, nocopy in ((0, 0.4, False), (1, 0.9, False), (2, 0.3, True)):
        bd = CycBD(Vf, P, hist)
        orc = Oracle(full, Vf, err, seed)
        real_copy = S.Spec._copy
        if nocopy:                                     # no copy-only rounds: every copy goes after DSpark drafts
            S.Spec._copy = lambda self, *a: (S.Spec._sync(self.hy, a[0], a[1], a[4]), None)[1]
        try:
            out, sp = S.generate(bd, 0, P, CycBD.C[0], N, ds=orc, taps=torch.zeros(4, P, H), stop=(),
                                 times=(1.0,) * 8, hybrid=True, prompt=hist)
        finally:
            S.Spec._copy = real_copy
        s = sp.stats
        assert out == ref, (seed, out[:30], ref[:30])
        assert s.get('ext_rounds', 0) > 0 if nocopy else s.get('copy_rounds', 0) > 0, s
        assert bd.steps < N // 2, (bd.steps, s)
        print(f'  hybrid oracle err {err}{" (no copy-only rounds)" if nocopy else ""}: {bd.steps} steps for {N - 1} '
              f'tokens, copy rounds {s.get("copy_rounds", 0)}, DSpark+copy {s.get("ext_rounds", 0)}, '
              f'copy c {[round(c, 2) for c in sp.cpr.c]}')
    stop = ref[60]
    plain, _ = S.generate(CycBD(Vf, P, hist), 0, P, CycBD.C[0], N, stop=(stop,))
    for mt in (2, 17, N):
        st_ = (stop,) if mt == N else ()
        out, _ = S.generate(CycBD(Vf, P, hist), 0, P, CycBD.C[0], mt, ds=Oracle(full, Vf, 0.2, 5),
                            taps=torch.zeros(4, P, H), stop=st_, times=(1.0,) * 8, hybrid=True, prompt=hist)
        assert out == (plain if st_ else ref[:mt]), (mt, out, plain if st_ else ref[:mt])
    # the real drafter: context == extend over the committed positions after copy / mixed rounds
    hist2 = CycBD.C * 2 + [3, 4]
    P2 = len(hist2)
    ref2, _ = S.generate(CycBD(V, P2, hist2), 0, P2, 11, 80, stop=())
    T0 = torch.stack([FakeBD.tap(p, t) for p, t in enumerate(hist2)], 1)
    bd = CycBD(V, P2, hist2)
    out, sp = S.generate(bd, 0, P2, 11, 80, ds=ds, taps=T0, stop=(), times=(1.0,) * 8, hybrid=True, prompt=hist2)
    assert out == ref2 and sp.stats.get('copy_rounds', 0) > 0, sp.stats
    sp.flush(bd.pos)
    rf = ds.new_state()
    ds.extend(rf, torch.stack([FakeBD.tap(p, t) for p, t in enumerate((hist2 + out)[:bd.pos])], 1))
    ring_eq(sp.st, rf, atol=1e-2)
    print(f'spec hybrid: greedy == plain (oracle, max_tokens, stop), real drafter context == extend after '
          f'{sp.stats.get("copy_rounds", 0)} copy rounds ({bd.steps} steps for 79 tokens)')


def test_spec_hybrid_sampled():
    """Sampled mixed rounds through Spec: one DSpark draft (q) followed by a copy of the context continuing after it
    (one-hot rows), many times from the same state: the first emitted token ~ p(row 0), the second (given the most
    common first) ~ p(row after it) (chi-square). Copy-only rounds (point mass) likewise."""
    from test_spec import chi2_ok
    Vf, n, temp = 24, 6000, 0.9
    hist = CycBD.C * 2
    P, last = len(hist), CycBD.C[0]
    bd0 = CycBD(Vf, P, hist)
    p0 = torch.softmax(bd0.logits(hist + [last]) / temp, -1)
    gen = torch.Generator().manual_seed(31)
    orc = Oracle([], Vf, seed=32)
    real_copy = S.Spec._copy
    for mode in ('mixed', 'copy'):
        first, rest, mixed = torch.zeros(Vf), {}, 0
        if mode == 'mixed':
            S.Spec._copy = lambda self, *a: (S.Spec._sync(self.hy, a[0], a[1], a[4]), None)[1]
        try:
            for _ in range(n):
                bd = CycBD(Vf, P, hist)
                sp = S.Spec(orc, bd, 0, start=P, kmax=1, temperature=temp, gen=gen, stop=(), times=(1.0,) * 8,
                            hybrid=True)
                sp.cpr.c = [0.95] * 5                    # any match pays: copies get appended / chosen
                rows, ctx = sp.propose(P, last, 12, kcap=15, hist=lambda: hist + [last])
                mixed += ctx is not None and len(ctx[0]) > 1
                out = sp.settle(bd.step(rows), 0, ctx)
                first[out[0]] += 1
                if len(out) > 1:
                    rest.setdefault(out[0], torch.zeros(Vf))[out[1]] += 1
        finally:
            S.Spec._copy = real_copy
        x2, crit = chi2_ok(first, p0, n)
        assert x2 < crit and mixed > n // 4, (mode, x2, crit, mixed)
        t0, c2 = max(rest.items(), key=lambda kv: kv[1].sum())
        p1 = torch.softmax(bd0.logits(hist + [last, t0]) / temp, -1)
        x2b, critb = chi2_ok(c2, p1, int(c2.sum()))
        assert x2b < critb, (mode, x2b, critb)
        print(f'spec hybrid sampled ({mode}: {mixed}/{n} rounds with copy rows): first token chi2 {x2:.1f} < {crit:.1f}, '
              f'second given {t0} chi2 {x2b:.1f} < {critb:.1f} ({int(c2.sum())} draws)')


def test_fork(ds):
    """DSpark.copy (n > 1: a sibling choice starts from the shared prompt's State): the copy drafts exactly what the
    original drafts (greedy ids, sampled q) and holds the same rings, also into a State that held another sequence;
    with the shortlist the dynamic part comes along."""
    for d in (ds, short_drafter()):
        a, b = d.new_state(), d.new_state()
        d.extend(b, taps(30, 8))                       # b held something else before
        d.round(b, 3, 7)
        d.extend(a, taps(50, 9))
        d.note(a, list(range(400, 460)))
        d.stage(a, taps(3, 10))
        d.copy(a, b)
        assert (b.n, b.pend) == (a.n, a.pend)
        ra = d.round(a, 5, 7)
        rb = d.round(b, 5, 7)
        assert ra[0] == rb[0], (ra[0], rb[0])
        ring_eq(a, b)
        qa = d.round(a, 6, 7, 0.8, torch.Generator().manual_seed(1))
        qb = d.round(b, 6, 7, 0.8, torch.Generator().manual_seed(1))
        assert qa[0] == qb[0] and torch.equal(qa[1], qb[1])
    print('fork: DSpark.copy drafts == the original (greedy ids, sampled q), rings equal, full head + shortlist')


def test_graph_pool():
    """Drafter graphs share one pool only while a live graph holds it: after every State with graphs is gone (idle
    States dropped), the next capture takes a fresh pool (capturing into a released pool id asserted in the
    allocator: window g 11-23). CUDA graph calls faked (CPU)."""
    import contextlib
    import gc
    from types import SimpleNamespace
    ds = fake_drafter()
    pools, used = iter(range(1, 100)), []
    fake = dict(current_stream=lambda: SimpleNamespace(wait_stream=lambda s: None, synchronize=lambda: None),
                Stream=lambda: SimpleNamespace(wait_stream=lambda s: None), graph_pool_handle=lambda: next(pools),
                CUDAGraph=lambda: SimpleNamespace(replay=lambda: None),
                stream=lambda s: contextlib.nullcontext(),
                graph=lambda g, pool=None, stream=None, capture_error_mode=None: used.append(pool) or contextlib.nullcontext())
    real = {k: getattr(torch.cuda, k) for k in fake}
    ds.graph, ds._body = True, lambda st, k, greedy, nb=None: None
    try:
        for k, v in fake.items():
            setattr(torch.cuda, k, v)
        a, b = ds.new_state(), ds.new_state()
        ds._run(a, 3, True)
        ds._run(b, 3, True)              # a's graph alive: same pool
        ds._run(a, 3, True)              # replay, no capture
        del a
        gc.collect()
        ds._run(b, 5, True)              # b still holds a graph: same pool
        b.graphs.clear()
        c = ds.new_state()
        ds._run(c, 3, True)              # no live graph left: fresh pool
        del b
        gc.collect()
        ds._run(c, 7, False)
    finally:
        for k, v in real.items():
            setattr(torch.cuda, k, v)
    assert used == [1, 1, 1, 2, 2], used
    print(f'drafter graph pool: shared while a graph lives, fresh after all are dropped (pools per capture {used})')


def test_engine_wiring():
    """Engine._spec / warm_drafter on a stub engine (fake 1000-token vocab, real drafter weights): with
    INK_FP8_DECODE (m.d8.dec) the drafter gets unembed8 and no bf16 unembed (shortlist on); the shortlist VRAM
    estimates go into free_vram; min_p reaches the Spec; warm_drafter runs the context-only pass + greedy AND sampled
    rounds at every width (gate: 7 only) on the idle State and leaves it reset."""
    from types import SimpleNamespace

    import engine as EN
    from fp8_dense import quant_w
    g = torch.Generator().manual_seed(0)
    W = {'unembed.weight': (torch.randn(V, H, generator=g) * 0.02).bfloat16(), 'embed_norm.weight': torch.ones(H)}
    W = type('W', (dict,), {})(W)
    W.embed = (torch.randn(V, H, generator=g) * 0.02).bfloat16()
    E = EN.Engine.__new__(EN.Engine)
    E.spec, E.ds, E.ds_free, E.parked, E.bd, E.spec_live = 'dspark', None, [], [], FakeBD(V, 0, []), set()
    needs = []
    E.mem_free = lambda: 1 << 62
    real_fv = E.free_vram
    E.free_vram = lambda need=None, **kw: needs.append(need) or real_fv(need, **kw)
    req = SimpleNamespace(spec=None, need=100, temperature=0.7, top_p=1.0, top_k=0, gen=torch.Generator(), cons=None,
                          prefill_text=True, prompt=[1, 2], P=0, min_p=0.05, proc=True, seed=7)
    for fp8 in (True, False):
        q8, s8 = quant_w(W['unembed.weight'])
        E.model = SimpleNamespace(dev=torch.device('cpu'), W=W, d8=SimpleNamespace(dec=fp8, un=(q8, s8)))
        E.ds, E.ds_free, needs[:] = None, [], []
        sp = E._spec(req)
        assert sp is not None and sp.min_p == 0.05 and E.ds.live == 1
        if fp8:
            assert E.ds.U is None and E.ds.U8 is not None and E.ds.short > 0
            short8 = E.ds.short
            assert EN.DS_HEAD + EN.DS_W_MB + EN.DS_SHORT_W_MB in needs, needs
            assert EN.DS_HEAD + EN.DS_ST_MB + EN.DS_SHORT_ST_MB in needs, needs
        else:
            assert E.ds.U is not None and E.ds.U8 is None and not E.ds.short
            assert EN.DS_HEAD + EN.DS_W_MB in needs and EN.DS_HEAD + EN.DS_ST_MB in needs, needs
        E.ds_free.append(sp.st)
        E.ds.live -= 1
    ran = []
    real_run = E.ds._run
    E.ds._run = lambda st, k, greedy, nb=None: (ran.append((k, greedy, nb)), real_run(st, k, greedy, nb))[1]
    for gate in (False, True):
        ran.clear()
        S.GATE = gate
        try:
            keys = E.warm_drafter(n=10)
        finally:
            S.GATE = False
        ks = [7] if gate else list(range(1, 8))
        want = [(0, True, None)] + [(k, gr, None) for gr in (True, False) for k in ks]
        assert keys == want and ran == want, (keys, ran)
        st = E.ds_free[-1]
        assert st.n == 0 and st.pend == 0 and int(st.nd) == 0 and bool((st.pos[1] < 0).all())
    E.ds_free.clear()
    assert E.warm_drafter() == []
    # 11-23 review 1: the State's free_vram must not drop the weights just loaded (crashed with AttributeError -> 500);
    # any failure -> spec off. g2 11-23: the drafter is optional, so VRAM too tight for it (or for a State) -> spec off
    # for this request, and its asks never push a parked prefix out (parked=False); weights stay if loaded.
    E.model = SimpleNamespace(dev=torch.device('cpu'), W=W, d8=None)
    E.bd = SimpleNamespace(drop=lambda keep=0: 0)
    E.parked, unparked = ['p'], []
    E.unpark = lambda e, disk=True: unparked.append(e)
    for free, ok in ((EN.DS_HEAD + EN.DS_W_MB, True), (1 << 30, False)):
        E.mem_free = lambda free=free: free
        for loaded in (False, True):
            if not loaded:
                E.ds = None
            E.ds_free = []
            ds0 = E.ds
            live0 = E.ds.live if E.ds is not None else 0
            sp = E._spec(req)  # (no room: prompt lookup only, no drafter)
            assert (sp.ds is not None) == ok and E.parked == ['p'] and not unparked, (loaded, free, sp, E.parked)
            if ok:
                assert E.ds is not None and E.ds.live == live0 + 1, (loaded, E.ds)
                E.ds.live -= 1
            else:
                assert E.ds is ds0 and (E.ds is None or E.ds.live == live0), (loaded, E.ds)
    E.mem_free = lambda: 1 << 62
    E.ds = None
    assert E._spec(req) is not None
    E.ds.live -= 1
    E.parked = []
    real_ds = D.DSpark
    D.DSpark = lambda **kw: (_ for _ in ()).throw(RuntimeError('boom (injected)'))
    try:
        E.ds = None
        assert E._spec(req) is None and E.ds is None
    finally:
        D.DSpark = real_ds
    print(f'engine wiring: FP8 decode -> drafter on unembed8 (shortlist {short8}), shortlist VRAM estimates, min_p; '
          f'warm_drafter: k=0 + greedy/sampled k=1..7 (gate: 7), State reset')


def test_long_ctx_spec():
    """Long-context spec VRAM decision (Engine, stub model, real drafter weights; BatchDecoder buffers on the CPU):
    a request needing <= INK_SPEC_MAX_CTX positions gets its drafter at start() (prompt taps, as before); a longer one
    holds NO drafter during its prefill (start: no Spec, weights not loaded) and gets it in to_decode if VRAM allows,
    its context starting at the decode position; VRAM short there -> prompt lookup only (no VRAM), parked prefixes
    untouched, weights kept if already loaded. The slot buffers: a lone decode allocates 1 slot's rings (VRAM ask SLOT_NEED only), a
    second slot grows them to all 4 with slot 0's state copied."""
    from types import SimpleNamespace

    import decode_batch as DB
    import engine as EN
    g = torch.Generator().manual_seed(0)
    W = type('W', (dict,), {})({'unembed.weight': (torch.randn(V, H, generator=g) * 0.02).bfloat16(),
                                'embed_norm.weight': torch.ones(H)})
    W.embed = (torch.randn(V, H, generator=g) * 0.02).bfloat16()
    CPU = torch.device('cpu')

    class BD:  # the engine-facing slot contract (dec, decs, new_slots, load, busy, drop, free)
        dec, n = True, 4

        def __init__(self):
            self.na, self.seqs = 0, [None] * 4

        @property
        def decs(self):
            return [self if i < self.na else None for i in range(4)]

        def new_slots(self, slot):
            return 0 if slot < self.na else slot + 1 if not self.na else 4

        def load(self, slot, seq):
            self.na = max(self.na, slot + 1) if not self.na else 4 if slot >= self.na else self.na
            self.seqs[slot] = seq

        def busy(self):
            return any(x is not None for x in self.seqs)

        def drop(self, keep=0):
            return 0
    E = EN.Engine.__new__(EN.Engine)
    P0 = (torch.tensor([99]), None, {'f': None}, True)                         # a parked prompt (must stay)
    E.spec, E.ds, E.ds_free, E.parked, E.spec_live, E.pool = 'dspark', None, [], [P0], set(), None
    E.model = SimpleNamespace(dev=CPU, W=W, d8=None)
    E.bd, needs, unparked = BD(), [], []
    E.unpark = lambda e, disk=True: unparked.append(e)
    real_fv = E.free_vram
    E.free_vram = lambda need=None, **kw: needs.append(need) or real_fv(need, **kw)
    E.restore = lambda keys, need_f=True: (SimpleNamespace(kv=None, pos=0), 0, None)
    E.admit = lambda tbl, n: None
    old_drop = EN.DEC_DROP
    EN.DEC_DROP = 0

    def req(n_ids, spec=None):
        ids = list(range(n_ids % 50 + 3))  # (ids only feed keys; need is what the decision reads)
        r = type('Req', (), {})()  # (hashable: spec_live is a set)
        r.__dict__.update(
            ids=ids, prompt=ids, pre=[], out=[5], keys=torch.tensor(ids), need=n_ids, spec=spec, oomed=False,
            temperature=0.0, top_p=1.0, top_k=0, gen=torch.Generator(), cons=None, prefill_text=True, min_p=0.0, seed=7,
            sp=None, slot=None, seq=None, P=0)
        return r
    try:
        E.mem_free = lambda: 1 << 62
        short, long = req(EN.SPEC_DEFER), req(EN.SPEC_DEFER + 1)
        E._start(short)
        assert short.sp is not None and short in E.spec_live and E.ds.live == 1     # as before: drafter from start()
        E._end_spec(short)
        E.ds = None
        E.ds_free = []
        E._start(long)
        assert long.sp is None and E.ds is None and long not in E.spec_live    # no drafter during the long prefill
        needs.clear()
        E.to_decode(long, 0)
        assert needs[0] == EN.SLOT_NEED, needs                                 # 1 slot's buffers, not 4
        assert long.sp is not None and long in E.spec_live and long.sp.st.n == long.pos == len(long.ids), long.sp
        assert EN.DS_HEAD + EN.DS_W_MB in needs and EN.DS_HEAD + EN.DS_ST_MB in needs and not unparked, needs
        E._end_spec(long)
        E.bd.seqs[0] = None
        # VRAM short at to_decode: lookup only, parked prefix untouched; with the weights loaded, they stay
        for loaded in (False, True):
            if not loaded:
                E.ds = None
            E.ds_free = []
            E.mem_free = lambda: 300 << 20                                       # < every drafter ask
            r = req(1 << 20)
            E._start(r)
            ds0 = E.ds
            E.to_decode(r, 0)
            r.sp.attach(0, 1 << 20)                                            # (far cost table at a 1M position)
            assert r.sp.vbase[0] > S.VERIFY_MS[0] + 5 and r.sp.vt[10] - r.sp.vt[9] > 5, (r.sp.vbase, r.sp.vt)  # 2nd far read past T 10
            assert r.sp.ds is None and r.sp.lk is not None and r not in E.spec_live, (loaded, r.sp)  # lookup only
            assert E.parked == [P0] and not unparked, (loaded, E.parked)
            assert E.ds is ds0, loaded
            E.bd.seqs[0] = None
            if not loaded:  # load the weights for the next round
                E.mem_free = lambda: 1 << 62
                E._end_spec(r)
                q = req(10)
                E._start(q)
                E._end_spec(q)
        E.mem_free = lambda: 1 << 62
        r = req(1 << 20, spec=False)                                           # spec: false -> never
        E._start(r)
        E.to_decode(r, 0)
        assert r.sp is None
        E.bd.seqs[0] = None
        # a second slot: the ask covers all 4 slots' buffers (the load grows them)
        E.bd.seqs[0] = 'busy'
        needs.clear()
        E.to_decode(req(10), 1)
        assert needs[0] == EN.SLOT_NEED + 3 * EN.SLOT_MB, needs
    finally:
        EN.DEC_DROP = old_drop
    # BatchDecoder buffers (real class, CPU tensors): lazy slots + growth keeps slot 0's state
    bd = DB.BatchDecoder.__new__(DB.BatchDecoder)
    bd.m, bd.n, bd.split, bd.tap_src = SimpleNamespace(dev=CPU, pool=SimpleNamespace(n=5)), 4, 4096, [4, 22, 40]
    bd.F, bd.seqs, bd.pend, bd.b, bd.na = SimpleNamespace(nsp=42), [None] * 4, {}, None, 0
    bd.graphs, bd.pool, bd.side = {}, None, None

    def seq(t):
        return SimpleNamespace(kv=SimpleNamespace(t=torch.full((5,), t, dtype=torch.int64)), pos=0,
                               conv=[{k: torch.full((3, C), float(t + v), dtype=torch.bfloat16) for k, _, C in DB.CONV}
                                     for v in range(len(DB.VISITS))],
                               ring=[(torch.zeros(0, 8, 128), torch.zeros(0, 8, 128))] * len(DB.VISITS))
    assert bd.new_slots(0) == 1 and bd.new_slots(2) == 3
    bd.load(0, seq(3))
    assert bd.na == 1 and bd.b.ring.shape[2] == 1 and bd.decs[1:] == [None] * 3 and bd.new_slots(1) == 4
    bd.b.ring[:, :, 0].normal_(generator=g)
    bd.b.rpos[0, :7] = torch.arange(7)
    bd.b.thist[:, :DB.TH].normal_(generator=g)
    keep = [x.clone() for x in (bd.b.ring[:, :, 0], bd.b.rpos[0], bd.b.conv[:, 0], bd.b.tbl[0], bd.b.thist[:, :DB.TH])]
    bd.graphs = {1: 'old graph'}
    bd.load(1, seq(4))
    assert bd.na == 4 and bd.b.ring.shape[2] == 4 and bd.graphs == {} and bd.b.thist.shape[1] == 4 * DB.TH + 1
    now = (bd.b.ring[:, :, 0], bd.b.rpos[0], bd.b.conv[:, 0], bd.b.tbl[0], bd.b.thist[:, :DB.TH])
    assert all(torch.equal(a, b) for a, b in zip(keep, now)), 'slot 0 state lost in the growth'
    assert int(bd.b.tbl[1, 0]) == 4 and bd.new_slots(3) == 0
    bd.free(0)
    bd.free(1)
    assert bd.idle() == 4 and bd.drop() == 4 and bd.na == 0 and bd.b is None
    print('long-context spec: <= INK_SPEC_MAX_CTX drafter from start(); longer: none during prefill, in to_decode '
          'when VRAM allows (context at the decode position), else lookup-only with parked prefixes untouched; lone decode '
          'holds 1 slot of buffers, growth to 4 keeps slot 0 state')


# --------------------------------------------------------------------------------------------- real weights
def test_real():
    import dspark_ablate as A
    if not os.path.exists(A.OUT):
        print(f'real: skipped, no {A.OUT} (run tests/dspark_ablate.py --record)')
        return
    data, ds = A.load()
    name = 'code1'
    r = A.evaluate(ds, {name: data[name]}, rows=1, stride=8)
    assert r['a'][0] >= A1_MIN, r
    print(f'real ({name}): position-1 greedy acceptance {r["a"][0]:.3f} >= {A1_MIN}')


def test_bd_taps():
    """BatchDecoder: tap rows after a step == hbuf of visits 4/22/40 (= h entering 5/23/41) and the final normed
    state, per row; the tap ops sit inside _run (graph) and the per-bucket op traces stay identical."""
    os.environ.setdefault('TRITON_INTERPRET', '1')
    import decode_batch as DB
    import test_decode_batch as TB
    from test_decode import launch_recorder, op_trace
    m, src = TB.setup()
    bd = TB.new_bd(m, 'ABCD', slots=4)
    assert bd.tap_src == [4, 22, 40]
    log, restore = launch_recorder([DB])               # kernels recorded, not run: buffers keep what we put there
    tr = {}
    g = torch.Generator().manual_seed(0)
    try:
        P = {s: bd.seqs[s].pos for s in range(4)}
        for rows in ([(0, P[0], 1), (1, P[1], 2), (2, P[2], 3)],
                     [(3, P[3] + i, 4 + i) for i in range(4)],
                     [(1, P[1] + 1, 9), (3, P[3] + 4, 8)]):
            b = bd.b
            b.hbuf.copy_(torch.randn(b.hbuf.shape, generator=g))
            hb, xx = b.hbuf.clone(), None

            def run(R, b=b):
                log.clear()
                tr.setdefault(R, []).append(op_trace(lambda: bd._run(R)))
            bd._replay = run
            bd.step(rows)
            xx = b.X[:, 0].clone()
            for r in range(len(rows)):
                tp = bd.tap_rows(r, 1)[:, 0]
                assert all(torch.equal(tp[j], hb[v, r]) for j, v in enumerate((4, 22, 40))), r
                assert torch.equal(tp[3], xx[r])
            bd.commit()
            for r, (slot, pos, _) in enumerate(rows):     # tap history: committed positions readable per slot
                assert torch.equal(bd.tap_hist(slot, pos, 1), bd.tap_rows(r, 1)), (slot, pos)
    finally:
        restore()
    for R, xs in tr.items():
        assert all(x == xs[0] for x in xs), R
    print(f'BatchDecoder taps: rows == hbuf[4/22/40] + final normed state == tap_hist; buckets {sorted(tr)} traces identical')


class SeqBD:
    """Real target on the Seq path, one slot: a T-row step = T single-row forwards (state snapshot after each),
    commit(n) restores the snapshot after n rows; taps via forward_taps. Same contract as decode_batch (a verify
    row == the same single-row step) by construction, so greedy spec must equal plain greedy bit for bit."""

    def __init__(self, m, seq):
        self.m, self.seq, self.snaps, self.fw = m, seq, None, 0

    def _snap(self):
        s = self.seq
        return s.pos, [{k: x.clone() for k, x in c.items()} for c in s.conv], list(s.ring)

    def step(self, rows):
        assert self.snaps is None, 'commit first'
        lg, tp, snaps = [], [], []
        for _, pos, t in rows:
            assert pos == self.seq.pos
            f, taps = S.forward_taps(self.m, self.m.embed([t]), self.seq)
            self.fw += 1
            lg.append(self.m.logits(f))
            tp.append(taps)
            snaps.append(self._snap())
        self.taps = torch.cat(tp, 1)
        if len(rows) > 1:
            self.snaps = snaps
        return torch.cat(lg)

    def commit(self, acc):
        n = acc[0]
        pos, conv, ring = self.snaps[n - 1]
        self.seq.pos, self.seq.conv, self.seq.ring = pos, conv, ring
        self.snaps = None

    def tap_rows(self, r0, n):
        return self.taps[:, r0:r0 + n]


def test_real_spec():
    import model as M
    from tokenizers import Tokenizer

    from engine import MODEL_DIR
    N = int(os.environ.get('INK_SPEC_N', 14))
    tok = Tokenizer.from_file(os.path.join(MODEL_DIR, 'tokenizer.json'))
    ids = chat.render(tok, [{'role': 'user', 'content': 'Write a Python function is_prime(n) with a docstring.'}],
                      effort=0)
    m = M.Model(MODEL_DIR, 'cpu', max_ctx=4096)
    ds = D.DSpark(embed=m.W.embed, unembed=m.W['unembed.weight'][:M.VOCAB], embed_norm=m.W['embed_norm.weight'])
    t0 = time.time()
    seq = m.new_seq()
    assert m.pool.reserve(seq.kv, len(ids) + N + 16)
    f, tp = S.forward_taps(m, m.embed(ids), seq)
    first = int(m.logits(f[-1:])[0].argmax())
    print(f'real spec: prompt {len(ids)} tokens prefilled in {time.time() - t0:.0f} s', flush=True)

    def fork():
        s = M.Seq('cpu', m.pool.fork(seq.kv, seq.pos))
        s.pos, s.ring = seq.pos, list(seq.ring)
        s.conv = [{k: v.clone() for k, v in d.items()} for d in seq.conv]
        assert m.pool.reserve(s.kv, seq.pos + N + 16)
        return s
    t0 = time.time()
    bd = SeqBD(m, fork())
    ref, _ = S.generate(bd, 0, len(ids), first, N, stop=(chat.EOS, chat.END))
    tp_plain = time.time() - t0
    print(f'  plain greedy: {len(ref)} tokens, {bd.fw} forwards, {tp_plain:.0f} s: {tok.decode(ref)!r}', flush=True)
    t0 = time.time()
    bd = SeqBD(m, fork())
    out, sp = S.generate(bd, 0, len(ids), first, N, ds=ds, taps=tp, stop=(chat.EOS, chat.END))
    s = sp.stats
    print(f'  spec greedy: {len(out)} tokens, {s["rounds"]} rounds, {bd.fw} forwards, {time.time() - t0:.0f} s '
          f'(drafter {s["draft_s"]:.1f} s); accepted {s["accepted"]}/{s["drafted"]}, per position '
          f'{[f"{a}/{b}" for a, b in zip(s["kept"], s["tried"]) if b]}', flush=True)
    assert out == ref, (out, ref)
    print('real spec: greedy spec == plain greedy (bit-exact)')


if __name__ == '__main__':
    torch.set_num_threads(int(os.environ.get('INK_THREADS', 4)))
    only = [a for a in sys.argv[1:] if not a.startswith('-')]
    ds = fake_drafter()
    tests = [('test_loader', lambda: test_loader(ds)), ('test_draft', lambda: test_draft(ds)),
             ('test_ctx_growth', lambda: test_ctx_growth(ds)), ('test_rings', lambda: test_rings(ds)),
             ('test_mask', lambda: test_mask(ds)),
             ('test_mask2', lambda: test_mask(fake_drafter(head='b', row0=True, bias='pre', tap_norm=True,
                                                           fuse_softmax=False, anchor='raw'))),
             ('test_round', lambda: test_round(ds)), ('test_capture_safe', lambda: test_capture_safe(ds)),
             ('test_spec_greedy_fake', test_spec_greedy_fake), ('test_spec_sleep', test_spec_sleep),
             ('test_spec_real_drafter_ctx', lambda: test_spec_real_drafter_ctx(ds)),
             ('test_short', lambda: test_short(short_drafter())), ('test_ctx_bucket', test_ctx_bucket),
             ('test_capture_short', test_capture_short), ('test_short_fp8', test_short_fp8),
             ('test_short_reuse', test_short_reuse), ('test_gate_causal', test_gate_causal),
             ('test_spec_short_gate', lambda: test_spec_real_drafter_ctx(short_drafter(ctxb=True), gate=True,
                                                                         accept_dev=True)),
             ('test_spec_dev_tok', lambda: test_spec_real_drafter_ctx(ds, dev_tok=True)),
             ('test_spec_short_dev_tok', lambda: test_spec_real_drafter_ctx(short_drafter(ctxb=True), dev_tok=True,
                                                                            accept_dev=True)),
             ('test_spec_grammar', test_spec_grammar), ('test_spec_sampled_bad', test_spec_sampled_bad),
             ('test_engine_wiring', test_engine_wiring), ('test_graph_pool', test_graph_pool),
             ('test_spec_hybrid', lambda: test_spec_hybrid(ds)), ('test_spec_hybrid_sampled', test_spec_hybrid_sampled),
             ('test_spec_hybrid_gate', lambda: test_spec_real_drafter_ctx(short_drafter(ctxb=True), gate=True,
                                                                          accept_dev=True, hybrid=True)),
             ('test_fork', lambda: test_fork(ds)), ('test_long_ctx_spec', test_long_ctx_spec)]
    if os.environ.get('INK_REAL') == '1':
        tests += [('test_real', test_real), ('test_bd_taps', test_bd_taps), ('test_real_spec', test_real_spec)]
    for name, fn in tests:
        if not only or name in only:
            t0 = time.time()
            fn()
            print(f'-- {name} {time.time() - t0:.0f} s', flush=True)
    print('DSPARK ALL OK')
