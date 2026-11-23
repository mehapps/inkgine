"""CPU, no weights (~30 s): spec.py drafting, acceptance and width controller.
1. sampled acceptance is exact: emitted-token distribution == target p (chi-square), point-mass and q drafts,
   including the joint law of the first two tokens.
2. greedy spec == plain greedy on a fake deterministic model (same per-position logits), any draft quality.
3. drafting: hit/accept stats on repetitive vs novel text; edge cases (no match, max_tokens, EOS in draft).
4. controller: k = 0 at low acceptance (with probes), wide at high acceptance.
5. accept_dev (one host sync) has accept's law; shortlist q (zero off a support) drawn by dspark.race; a confidence
   gate on q rows stays exact, a gate on the drawn token's q is caught; Gate widths with Width's on/off, incl. a
   json-like trace (confident structure + unconfident values) where the old gate on/off rule switched drafting off.
6. hybrid: Lookup (longest + most recent match, following a copy, match after uncommitted drafts, 200k-token search
   cost), Copy (width choice by match length), mixed proposals (drafter rows with q, then one-hot copy rows decided
   from the drafts) exact under accept and accept_dev.
"""
import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import torch  # noqa: E402

import spec  # noqa: E402


def chi2_ok(counts, p, n):
    """Pearson chi-square vs expected n*p over cells with expected >= 5 (rest pooled); p-value 1e-3 threshold via
    Wilson-Hilferty. Also every count on a p == 0 cell must be 0."""
    counts, p = counts.double(), p.double()
    assert counts[p == 0].sum() == 0
    e = n * p
    big = e >= 5
    obs = torch.cat((counts[big], counts[~big].sum()[None]))
    exp = torch.cat((e[big], e[~big].sum()[None]))
    keep = exp > 0
    obs, exp = obs[keep], exp[keep]
    x2 = ((obs - exp) ** 2 / exp).sum().item()
    df = len(obs) - 1
    crit = df * (1 - 2 / (9 * df) + 3.09 * math.sqrt(2 / (9 * df))) ** 3
    return x2, crit


def test_sampled_exact():
    g = torch.Generator().manual_seed(0)
    V, N = 12, 40000
    lg = torch.randn(3, V, generator=g) * 2
    for temp, tp, tk in ((1.0, 1.0, 0), (0.7, 0.9, 0), (1.3, 1.0, 5)):
        p = spec.probs(lg, temp, tp, tk)
        assert torch.allclose(p.sum(-1), torch.ones(3))
        for d in (int(p[0].argmax()), int(p[0].argmin()), 3):    # likely, impossible (if filtered), arbitrary draft
            for q in (None, torch.softmax(torch.randn(2, V, generator=g), -1)):
                first, joint = torch.zeros(V), torch.zeros(V)
                d2 = [d, 5]
                if q is not None:
                    q[0, d] = q[0, d] + 0.5
                    q = q / q.sum(-1, keepdim=True)
                for _ in range(N):
                    if q is not None:  # min(1, p/q) is exact only for drafts actually sampled from q
                        d2 = torch.multinomial(q, 1, generator=g).flatten().tolist()
                    out = spec.accept(p, d2, q, gen=g)
                    first[out[0]] += 1
                    if out[0] == d2[0]:
                        joint[out[1]] += 1
                x2, crit = chi2_ok(first, p[0], N)
                assert x2 < crit, ('first', temp, tp, tk, d, q is None, x2, crit)
                # given token 1 == d, token 2 ~ p[1] (row 1 was computed with d in context)
                n2 = int(joint.sum())
                if n2 > 500:
                    x2b, critb = chi2_ok(joint, p[1], n2)
                    assert x2b < critb, ('second', temp, tp, tk, d, x2b, critb)
        print(f'sampled acceptance T={temp} top_p={tp} top_k={tk}: chi2 OK (last {x2:.1f} < {crit:.1f})')


def joint_law(p, draw, acc, N, g):
    """N rounds of acc(p, d, q) with (d, q) = draw(g): chi2 of the first emitted token vs p[0] and, when the first
    draft was kept, of the second vs p[1] (p[1] is the target row after d_1 in context). -> (x2, crit) pairs."""
    V = p.shape[1]
    first, second = torch.zeros(V), torch.zeros(V)
    for _ in range(N):
        d, q = draw(g)
        out = acc(p, d, q, gen=g)
        first[out[0]] += 1
        if d and out[0] == d[0] and len(out) > 1:
            second[out[1]] += 1
    r = [chi2_ok(first, p[0], N)]
    if second.sum() > 500:
        r.append(chi2_ok(second, p[1], int(second.sum())))
    return r


def test_accept_dev_shortlist():
    """accept_dev == accept's law (point mass and q drafts); shortlist q (0 off a support that misses most of p's
    mass) drawn by dspark.race with fp64 noise, through accept and accept_dev; a confidence-gated verify width (Gate
    style: a function of the q rows, which depend on the drafts before them) stays exact, and a gate that looks at the
    drawn token's own q (not allowed) is caught."""
    import dspark
    g = torch.Generator().manual_seed(7)
    V, N = 12, 30000
    p = spec.probs(torch.randn(3, V, generator=g) * 2, 0.8)
    S = torch.tensor([0, 2, 3, 7, 9])                       # shortlist support; p's argmax may be outside it
    qrow = lambda prev, s: torch.zeros(V).index_put_((S,), torch.softmax(
        torch.randn(len(S), generator=torch.Generator().manual_seed(s * 31 + prev)) * 1.5, -1))

    def draw_short(g, gate=None):
        """2 drafts; q_2 depends on d_1 (head fed the draft). gate(q rows, drafts) -> drafts kept for verify."""
        q1 = qrow(-1, 1)
        d1 = int(S[dspark.race(q1[S][None], torch.empty(1, len(S), dtype=torch.float64).exponential_(generator=g))])
        q2 = qrow(d1, 2)
        d2 = int(S[dspark.race(q2[S][None], torch.empty(1, len(S), dtype=torch.float64).exponential_(generator=g))])
        q, d = torch.stack((q1, q2)), [d1, d2]
        m = 2 if gate is None else gate(q, d)
        return d[:m], q[:m]
    cases = [('accept_dev point mass', lambda g: ([int(p[0].argmax()), 4], None), spec.accept_dev),
             ('accept_dev q', lambda g: (lambda q: (torch.multinomial(q, 1, generator=g).flatten().tolist(), q))(
                 torch.softmax(torch.randn(2, V, generator=g), -1)), spec.accept_dev),
             ('shortlist q, accept', draw_short, spec.accept),
             ('shortlist q, accept_dev', draw_short, spec.accept_dev),
             ('shortlist q, conf gate, accept_dev',
              lambda g: draw_short(g, lambda q, d: 1 + int(float(q[1].max()) > 0.45)), spec.accept_dev)]
    for name, draw, acc in cases:
        for x2, crit in joint_law(p, draw, acc, N, g):
            assert x2 < crit, (name, x2, crit)
        print(f'{name}: chi2 OK ({x2:.1f} < {crit:.1f})')
    # mutation: verify d_1 only when its own q is high -> the verified d_1 is not ~ q_1 any more
    bad = lambda g: draw_short(g, lambda q, d: int(float(q[0, d[0]]) > 0.3))
    x2, crit = joint_law(p, bad, spec.accept_dev, N, g)[0]
    assert x2 > crit, ('gate on the drawn token must be detected', x2, crit)
    print(f'gate on the drawn token q(d) (forbidden): detected, chi2 {x2:.0f} > {crit:.1f}')


def gate_run(ctl, gt, acc, conf, n, rng, dms=1.8):
    """Spec's gate loop: Width (ctl) decides whether to draft, Gate picks the verify width from 7 confidences.
    acc(conf) = P(kept | conf); conf(i) draws row i's confidence. -> tokens/ms vs plain, share of drafting rounds."""
    T = spec.Width.STEP_MS
    tok = ms = drafted = 0
    for i in range(n):
        if not ctl.width():
            tok, ms = tok + 1, ms + T[0]
            continue
        drafted += 1
        c = [conf(i) for _ in range(7)]
        m = gt.pick(c, 7)
        assert m in (0, 1, 3, 7), m
        a = 0
        while a < m and rng.random() < acc(c[a]):
            a += 1
        gt.update(c, m, a, ctl)
        tok, ms = tok + a + 1, ms + T[m] + dms
    return tok / ms * T[0], drafted / n


def test_gate():
    """Gate: verify widths only from {1, 3, 7}; high confidence + high acceptance -> wide; low -> drafting off (Width,
    probes back off to pmax); recovery when acceptance comes back; json-like text keeps drafting."""
    rng = random.Random(3)
    T = spec.Width.STEP_MS
    tw = tuple(t + (1.8 if k else 0) for k, t in enumerate(T))   # Width's round cost incl. the drafter (FP8 window)
    gt = spec.Gate()
    assert gt.pick([0.99] * 7, 7) == 7 and gt.pick([0.99, 0.99, 0.1, 0.1, 0.1, 0.1, 0.1], 7) in (1, 3)
    assert gt.pick([0.05] * 7, 7) == 0 and gt.pick([0.99] * 7, 2) in (0, 1)
    W = lambda: spec.Width(kmax=7, step_ms=tw)
    hi, dh = gate_run(W(), spec.Gate(), lambda c: c, lambda i: rng.uniform(0.85, 1.0), 400, rng)
    lo_w = W()
    lo, dl = gate_run(lo_w, spec.Gate(), lambda c: 0.3 * c, lambda i: rng.uniform(0.0, 1.0), 600, rng)
    assert hi > 1.8 and dh > 0.95 and dl < 0.1 and lo > 0.97 and lo_w.gap == lo_w.pmax, (hi, dh, lo, dl, lo_w.gap)
    back, _ = gate_run(lo_w, spec.Gate(), lambda c: c, lambda i: rng.uniform(0.85, 1.0), 400, rng)
    assert back > 1.5, back
    # json-like: stretches of confident structure (keys, punctuation) and unconfident values (names, numbers): the
    # 11-22 gate's own on/off rule (realized tokens/ms EMA per drafting round) drafted in 36/151 rounds here (x1.31)
    js = lambda i: rng.uniform(0.9, 1.0) if (i // 6) % 3 else rng.uniform(0.05, 0.5)
    jx, jd = gate_run(W(), spec.Gate(), lambda c: c, js, 600, rng)
    assert jd > 0.8 and jx > 1.3, (jx, jd)
    print(f'gate: confident text x{hi:.2f} of plain ({dh:.0%} drafting), unconfident x{lo:.3f} ({dl:.0%} drafting), '
          f'recovers to x{back:.2f}, json-like x{jx:.2f} ({jd:.0%} drafting)')


def test_lookup():
    """Lookup: longest suffix match, most recent among the longest; follows a copy without a search; after(d) finds
    the continuation of uncommitted drafts; a 200k-token context costs ~nothing to build and little per search."""
    import time
    lk = spec.Lookup([1, 2, 3, 9, 7, 2, 3, 5, 6, 1, 2, 3])
    assert lk.match() == (3, 3) and lk.draft(3) == [9, 7, 2]   # (1, 2, 3) beats the more recent (2, 3)
    lk = spec.Lookup([4, 2, 3, 5, 6, 7, 2, 3])
    assert lk.draft(2) == [5, 6]
    lk.extend([5, 6])                                          # copying: the match follows (no search)
    assert lk.draft(2) == [7, 2] and lk.last[1:] == (5, 4)
    lk.extend([8])                                             # copy broken: search again (8 never seen)
    assert lk.draft(2) == []
    lk = spec.Lookup([10, 11, 12, 13, 14, 20])
    assert lk.after([11, 12]) == (3, 2) and lk.span(3, 5, stop=(14,)) == [13, 14]
    assert lk.after([99]) == (None, 0)
    rng = random.Random(5)
    big = [rng.randrange(50000) for _ in range(200000)]
    big[::9] = [220] * len(big[::9])                           # one very frequent token
    t0 = time.perf_counter()
    lk = spec.Lookup(big)
    t1 = time.perf_counter()
    lk.extend([220])
    lk.match()
    lk.extend(big[1000:1003])
    j, n = lk.match()
    t2 = time.perf_counter()
    assert j is not None and n >= 3 and lk.draft(4) == big[j:j + 4], (j, n)
    assert t1 - t0 < 0.5 and t2 - t1 < 0.2, (t1 - t0, t2 - t1)
    print(f'lookup: longest/most recent/follow/after OK; 200k ids: build {1e3 * (t1 - t0):.1f} ms, '
          f'2 searches {1e3 * (t2 - t1):.1f} ms')


def test_copy():
    """Copy: no copy at a 1-token match, long copies at long matches; learns from outcomes; verify16 extends 8 -> 16."""
    vt = spec.verify16(spec.Width.STEP_MS)
    assert len(vt) == 16 and vt[:8] == spec.Width.STEP_MS and all(a < b for a, b in zip(vt, vt[1:]))
    cp = spec.Copy()
    assert cp.best(cp.bucket(1), 15, vt)[0] == 0
    m16, r16 = cp.best(cp.bucket(40), 15, vt)
    assert m16 >= 8 and r16 > 1.5 / vt[0], (m16, r16)
    for _ in range(60):
        cp.update(cp.bucket(40), 15, 0)                        # copies of long matches keep failing at once
    assert cp.best(cp.bucket(40), 15, vt)[0] <= 3, cp.c
    # after a drafter lead: copied rows only count if the lead is fully kept
    assert cp.best(4, 8, vt, base=3.0, reach=0.0, k0=7)[0] == 0
    print(f'copy: 1-token match -> no copy, 40-token match -> {m16} tokens (x{r16 * vt[0]:.2f} of plain), learns')


def test_mixed_exact():
    """Hybrid proposals: drafter rows with a real q, then copy rows (one-hot q) that are a deterministic function of
    the drafts before them and included only when a rule on those drafts says so; also copy-only (q None, point
    mass). The emitted tokens follow p exactly (first token; second given the first), accept and accept_dev."""
    g = torch.Generator().manual_seed(21)
    V, N = 10, 30000
    p = spec.probs(torch.randn(3, V, generator=g) * 1.5, 0.9)
    q1 = torch.softmax(torch.randn(V, generator=g), -1)
    copy = lambda d1: (3 * d1 + 1) % V                         # the 'context' continuation after d1

    def draw_mixed(g):
        d1 = int(torch.multinomial(q1, 1, generator=g))
        if d1 % 3 == 0:                                        # no match after this draft: drafter row only
            return [d1], q1[None]
        oh = torch.zeros(V)
        oh[copy(d1)] = 1.0
        return [d1, copy(d1)], torch.stack((q1, oh))
    cases = [('mixed q + one-hot, accept', draw_mixed, spec.accept),
             ('mixed q + one-hot, accept_dev', draw_mixed, spec.accept_dev),
             ('copy only (point mass), accept_dev', lambda g: ([int(p[0].argmax()), 4], None), spec.accept_dev)]
    for name, draw, acc in cases:
        r = joint_law(p, draw, acc, N, g)
        assert len(r) == 2 or name.startswith('copy'), (name, r)
        for x2, crit in r:
            assert x2 < crit, (name, x2, crit)
        print(f'{name}: chi2 OK ({x2:.1f} < {crit:.1f})')


def fake_logits(ctx, V=50):
    """Deterministic 'model': logits for the next token depend on the last 3 ids (so text gets repetitive)."""
    g = torch.Generator().manual_seed(hash(tuple(ctx[-3:])) % (1 << 31))
    return torch.randn(V, generator=g)


def run(prompt, n, spec_on, V=50, eos=None):
    lk, ctl = spec.Lookup(prompt), spec.Width(kmax=6)
    out, rounds = [], 0
    while len(out) < n:
        ctx = prompt + out
        d = lk.draft(ctl.width(), left=n - len(out), stop=() if eos is None else (eos,)) if spec_on else []
        rows = [fake_logits(ctx + d[:i], V) for i in range(len(d) + 1)]  # what a verify step returns
        got = spec.accept_greedy(torch.stack(rows), d)
        ctl.update(len(d), len(got) - 1)
        lk.extend(got)
        out += got
        rounds += 1
        if eos is not None and eos in got:
            out = out[:out.index(eos) + 1]
            break
    assert len(out) <= n, (len(out), n)
    return out, rounds


def test_greedy_equivalence():
    random.seed(1)
    saved = 0
    for trial in range(20):
        V = random.choice((8, 20, 50))  # small vocab: loops -> many matches and accepted drafts
        prompt = [random.randrange(V) for _ in range(random.randint(1, 40))]
        n = random.randint(1, 120)
        a, r_plain = run(prompt, n, False, V)
        b, r_spec = run(prompt, n, True, V)
        assert a == b and len(a) == n, (trial, a, b)
        saved += r_plain - r_spec
    assert saved > 100, saved  # drafts were really accepted, not only rejected
    # with EOS: stops at the same place
    a, _ = run([1, 2, 3, 1, 2, 3], 200, False, 8, eos=4)
    b, rs = run([1, 2, 3, 1, 2, 3], 200, True, 8, eos=4)
    assert a == b
    print(f'greedy spec == plain on 20 random fake-model runs ({saved} verify steps saved) + eos run; last run {len(b)} tokens in {rs} verifies')


def accept_stats(text_ids, k=4):
    """Replay a fixed greedy output through the drafter: drafts accepted vs the true continuation."""
    lk, i, drafted, kept, rounds = spec.Lookup(text_ids[:1]), 1, 0, 0, 0
    while i < len(text_ids):
        d = lk.draft(k, left=len(text_ids) - i)
        n = 0
        while n < len(d) and d[n] == text_ids[i + n]:
            n += 1
        step = text_ids[i:i + n + 1]
        lk.extend(step)
        i, drafted, kept, rounds = i + len(step), drafted + len(d), kept + n, rounds + 1
    return kept / max(drafted, 1), len(text_ids[1:]) / rounds


def test_drafting():
    from tokenizers import Tokenizer
    from engine import MODEL_DIR
    tok = Tokenizer.from_file(os.path.join(MODEL_DIR, 'tokenizer.json'))
    enc = lambda s: tok.encode(s, add_special_tokens=False).ids
    code = enc('def f(x):\n    return x + 1\n\n' * 2 + 'def g(x):\n    return x + 2\n\n' + 'def f(x):\n    return x + 1\n' * 3)
    prose = enc('The lighthouse keeper had not spoken to another person in eleven days when the supply boat finally '
                'rounded the point. She watched it from the gallery, one hand on the cold iron rail, and tried to '
                'remember how conversation worked.')
    ar, tr = accept_stats(code)
    an, tn = accept_stats(prose)
    print(f'drafting: repetitive code accept {ar:.2f}, {tr:.2f} tok/verify; novel prose accept {an:.2f}, '
          f'{tn:.2f} tok/verify')
    assert ar > 0.7 and tr > 2.0 and tn < 1.3, (ar, tr, an, tn)
    # edge cases
    lk = spec.Lookup([1, 2, 3, 4])
    assert lk.draft(4) == []                                    # suffix 4 never seen before: no match
    assert spec.Lookup([]).draft(4) == []
    lk = spec.Lookup([5, 6, 7, 8, 9, 5, 6])
    assert lk.draft(4) == [7, 8, 9, 5]                          # longest match (5, 6), most recent occurrence
    assert lk.draft(4, left=3) == [7, 8]                        # verify yields len + 1 <= left
    assert lk.draft(4, left=1) == [] and lk.draft(0) == []
    assert lk.draft(4, stop=(9,)) == [7, 8, 9]                  # EOS ends the draft (inclusive)
    lk = spec.Lookup([1, 2, 9, 1, 3, 9, 1])
    assert lk.draft(2) == [3, 9]                                # bigram (9, 1): the earlier occurrence (the suffix itself is not indexed)
    lk.extend([2])                                              # (1, 2) seen at the start -> 9, 1
    assert lk.draft(2) == [9, 1]
    assert spec.Lookup([4, 4, 4]).draft(5) == [4]               # draft never runs past the context end
    # greedy accept: all kept -> bonus from last row; first wrong -> correction
    lg = torch.eye(6)[[2, 3, 4]] * 5
    assert spec.accept_greedy(lg, [2, 3]) == [2, 3, 4]
    assert spec.accept_greedy(lg, [2, 5]) == [2, 3]
    assert spec.accept_greedy(lg[:1], []) == [2]
    # sampled accept with a point-mass target at the draft: always kept; residual never empty-crashes
    p = torch.eye(6)[[2, 3, 4]]
    assert spec.accept(p, [2, 3]) == [2, 3, 4]
    assert spec.accept(p, [1]) == [2]
    print('drafting edge cases OK')


def sim(ctl, conds, rounds, rng):
    """Synthetic acceptance trace: draft i is kept with conditional prob conds[i] (all before it kept). ->
    (widths, tokens, ms) with the controller's own cost table."""
    ks, tok, ms = [], 0, 0.0
    for _ in range(rounds):
        k = ctl.width()
        a = 0
        while a < k and rng.random() < conds[a]:
            a += 1
        ctl.update(k, a)
        ks.append(k)
        tok, ms = tok + a + 1, ms + ctl.t[k]
    return ks, tok, ms


def test_width():
    rng = random.Random(0)
    T = tuple(t + (3.0 if k else 0) for k, t in enumerate(spec.Width.STEP_MS))    # verify + ~DSpark round
    plain = 1 / T[0]
    prose, code = (0.40, 0.35, 0.57, 0.3, 0.3, 0.3, 0.3), (0.79, 0.72, 0.75, 0.88, 0.97, 0.95, 0.94)  # GPU 11-22
    lo = spec.Width(kmax=7, step_ms=T)
    ks, tok, ms = sim(lo, prose, 600, rng)
    off = ks[100:].count(0) / 500
    assert off > 0.9 and any(ks[100:]) and tok / ms > 0.98 * plain, (off, tok / ms / plain, ks[:60])
    hi = spec.Width(kmax=7, step_ms=T)
    ks2, tok2, ms2 = sim(hi, code, 300, rng)
    assert ks2[50:].count(0) == 0 and min(ks2[50:]) >= 3 and tok2 / ms2 > 1.8 * plain, (ks2, tok2 / ms2 / plain)
    # content change: prose (drafting off, probes backed off to pmax) -> code: drafting resumes
    sw = spec.Width(kmax=7, step_ms=T)
    sim(sw, prose, 600, rng)
    assert sw.gap == sw.pmax and sw.width() == 0, sw.gap
    ks3, _, _ = sim(sw, code, 200, rng)
    on = next(i for i in range(len(ks3)) if all(ks3[i:i + 20]))
    assert on <= 5 * sw.pmax, (on, ks3)
    # back to prose: off again
    ks4, _, _ = sim(sw, prose, 200, rng)
    assert ks4[100:].count(0) / 100 > 0.9, ks4
    assert spec.Width(kmax=0).width() == 0
    # batched (window g2 11-23): 4 requests in a step. Solo-rule drafting slowed the batch; with the batch rule a request
    # drafts only if the step's tokens/ms go up: prose / mid acceptance -> plain, code-grade acceptance -> still drafts
    tb4 = spec.Width.STEP_MS[3]
    mid = (0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.5)
    for c, solo, b4 in ((prose, False, False), (mid, True, False), ((0.97,) * 7, True, True)):
        w = spec.Width(kmax=7, step_ms=T)
        w.c = list(c)
        k1, k4 = w.width(), w.width(4, tb4)
        assert (k1 > 0) == solo and (k4 > 0) == b4, (c, k1, k4)
        assert w.rates() == w.rates(1, None)                                   # batch 1 == the solo rule exactly
    print(f'width: prose x{tok / ms / plain:.3f} of plain ({off:.1%} rounds k=0, probes at '
          f'{[i for i, k in enumerate(ks) if k][-4:]}), code x{tok2 / ms2 / plain:.2f} (k {sorted(set(ks2[50:]))}), '
          f'prose -> code: on after {on} rounds')


if __name__ == '__main__':
    test_drafting()
    test_width()
    test_greedy_equivalence()
    test_sampled_exact()
    test_accept_dev_shortlist()
    test_gate()
    test_lookup()
    test_copy()
    test_mixed_exact()
    print('SPEC ALL OK')
