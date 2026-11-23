"""Speculative decoding without the kernels (DESIGN-batching.md sec. 3): prompt-lookup drafts + exact acceptance.

A verify step for one sequence feeds T = k + 1 rows: the last committed token and the k draft tokens d_1..d_k.
Logits row i is the model's prediction for the position after row i's token, so row i checks d_{i+1}, and the row
after the last accepted draft gives the bonus token. Every function here is pure on host ids / torch logits, so the
future BatchDecoder verify path only calls them:

    lk = Lookup(prompt_ids)                      # n-gram index over prompt + generated ids
    d = lk.draft(ctl.width(), left=max_tokens - n, stop=(chat.EOS, chat.END))
    lg = bd.step([(slot, pos + i, t) for i, t in enumerate([last] + d)])       # [k+1, V]
    out = accept_greedy(lg, d) if temperature <= 0 else accept(probs(lg, temperature, top_p, top_k), d, gen=gen)
    ctl.update(len(d), len(out) - 1); lk.extend(out); bd.commit({slot: len(out)}); pos += len(out)
"""
import math

import numpy as np
import torch


class Lookup:
    """Prompt-lookup drafter over prompt + generated ids. match(): the longest suffix of the ids (up to n tokens) that
    occurred earlier with a follower, most recent such occurrence -> (j, len): the copied continuation starts at j.
    No index to build (a 200k-token prompt costs nothing up front): a search is one vectorized compare over the ids
    plus filters on the candidates (~0.05-0.3 ms at 200k), and while the text keeps copying (the new ids equal the
    followers of the last match) the match just moves along without a search. The suffix never matches itself."""

    def __init__(self, ids=(), n=32):
        self.n, self.a, self.L, self.last = n, np.zeros(max(1024, 2 * len(ids)), np.int32), 0, None
        self.extend(ids)

    def __len__(self):
        return self.L

    @property
    def ids(self):
        return self.a[:self.L].tolist()

    def extend(self, ids):
        ids = np.asarray(ids, dtype=np.int32).reshape(-1)
        if self.L + len(ids) > len(self.a):
            self.a = np.concatenate((self.a[:self.L], np.zeros(max(len(self.a), len(ids)), np.int32)))
        self.a[self.L:self.L + len(ids)] = ids
        self.L += len(ids)

    def find(self, tail):
        """Longest suffix of `tail` (the last <= n tokens of some sequence) found in the ids with a follower, most
        recent occurrence -> (j, n) with the follower at j, or (None, 0)."""
        a, L = self.a, self.L
        if L < 2 or not len(tail):
            return None, 0
        c, n = np.flatnonzero(a[:L - 1] == tail[-1]) + 1, 1      # follower positions j: a[j - 1] == tail[-1], j < L
        while n < len(tail) and len(c):
            c2 = c[c > n]
            c2 = c2[a[c2 - n - 1] == tail[-n - 1]]
            if not len(c2):
                break
            c, n = c2, n + 1
        return (int(c[-1]), n) if len(c) else (None, 0)

    def match(self):
        """(j, n) for the current ids (see find), following the last match while the text copies it."""
        if self.last is not None:
            L0, j, n = self.last
            dl = self.L - L0
            if dl == 0:
                return j, n
            if j is not None and j + dl < self.L and np.array_equal(self.a[j:j + dl], self.a[L0:self.L]):
                self.last = (self.L, j + dl, min(n + dl, self.n))
                return self.last[1:]
        j, n = self.find(self.a[max(0, self.L - self.n):self.L])
        self.last = (self.L, j, n)
        return j, n

    def after(self, d):
        """(j, n) for the ids followed by the uncommitted tokens d (a draft: a copy continuing after it)."""
        t = np.concatenate((self.a[max(0, self.L - self.n):self.L], np.asarray(d, dtype=np.int32)))
        return self.find(t[-self.n:])

    def span(self, j, k, stop=()):
        """Up to k ids from j (never past the end); ends with the first stop id."""
        d = self.a[j:min(j + k, self.L)].tolist()
        for i, t in enumerate(d):
            if t in stop:
                return d[:i + 1]
        return d

    def draft(self, k, left=None, stop=()):
        """Up to k tokens following the match (see match); [] if none. left: tokens the request may still emit; a
        verify yields up to len(draft) + 1, so the draft is <= left - 1. stop: ids that end generation (EOS/END): the
        draft ends with the first one (accepting it ends the reply)."""
        if left is not None:
            k = min(k, left - 1)
        if k <= 0:
            return []
        j, _ = self.match()
        return [] if j is None else self.span(j, k, stop)


def verify16(t):
    """Verify ms for T = 1..8 rows -> T = 1..16 (BatchDecoder's 16-row bucket): beyond 8 rows log-linear in T with the
    T = 4 -> 8 slope (the MoE expert union keeps growing). shortcut: an extrapolation until a window measures T 9..16."""
    t = tuple(t)
    return t[:8] + tuple(t[7] + (t[7] - t[3]) * math.log2(T / 8) for T in range(9, 17))


class Copy:
    """Acceptance model of prompt-lookup drafts (INK_SPEC_HYBRID): a copied span tends to go on token after token, so
    c[b] = EMA of P(next copied token kept | the ones before it kept), per bucket b of the match length (1, 2-3, 4-7,
    8-15, 16+; longer matches copy more reliably). A draft of m copied tokens after a lead that emits `base` tokens
    and is fully kept with probability `reach` is expected to emit base + reach * (c + c^2 + .. + c^m)."""
    EDGES = (2, 4, 8, 16)
    PRIOR = (0.05, 0.2, 0.45, 0.7, 0.85)   # shortcut: guesses, no probes (a bucket below break-even never learns)

    def __init__(self, ema=0.1, prior=PRIOR):
        self.c, self.ema = list(prior), ema

    def bucket(self, n):
        return sum(n >= e for e in self.EDGES)

    def best(self, b, mcap, t, base=1.0, reach=1.0, k0=0):
        """-> (m, tokens/ms): the m in 0..mcap copied tokens maximizing (base + reach * sum_{j<=m} c^j) / t[k0 + m]."""
        c, tok, best = self.c[b], base, (0, base / t[k0])
        for m in range(1, mcap + 1):
            reach *= c
            tok += reach
            if tok / t[k0 + m] > best[1]:
                best = (m, tok / t[k0 + m])
        return best

    def update(self, b, drafted, accepted):
        for i in range(min(drafted, accepted + 1)):
            self.c[b] += self.ema * ((i < accepted) - self.c[b])


class Width:
    """Adaptive draft width (sec. 3.1). c[i] = EMA of P(draft i+1 kept | drafts 1..i kept), per position; a round
    of width k is expected to emit 1 + sum_{j<=k} c[0]*..*c[j-1] tokens in step_ms[k] ms. Picks the k in 1..kmax with
    the most tokens/ms; drafts only if that beats plain decoding (k = 0: tokens/ms 1 / step_ms[0]) by `margin`, else
    k = 0 (the caller makes no drafter call). Probes keep the estimates fresh: while off, every `gap` rounds one round
    at the best drafting width (gap doubles per probe that leaves drafting off, probe..pmax; resets when drafting
    turns on), so a prose -> code change turns drafting back on; while on, every `probe` rounds one draft deeper
    (k + 1) so a stale low c of a deep position can recover.
    step_ms[k] = time of a round of width k: here the verify step alone (T = k + 1 rows on BatchDecoder, GPU
    2026-11-22: 7.3 ms at T=1, x1.17 / x1.31 / x1.55 at T=2/4/8, log-interpolated); prompt lookup drafts cost ~0.
    spec_runner.step_ms() adds the DSpark drafter round for k >= 1."""
    STEP_MS = (7.3, 8.54, 9.12, 9.56, 10.10, 10.55, 10.96, 11.32)

    def __init__(self, kmax=4, a=0.5, step_ms=STEP_MS, ema=0.1, probe=8, pmax=32, margin=0.02):
        self.kmax, self.t, self.ema, self.margin = min(kmax, len(step_ms) - 1), step_ms, ema, margin
        self.c = [a] * self.kmax
        self.probe, self.pmax, self.gap, self.idle = probe, pmax, probe, 0
        self.on = True                                    # last width(): drafting pays (False: off or a probe)

    def rates(self, batch=1, tb=None):
        """Expected tokens/ms per width k = 0..kmax. batch > 1: of the whole decode step, where the other batch - 1
        requests emit one token each and a plain step of `batch` rows takes tb ms; drafting adds this request's
        drafter round and verify rows (t[k] - t[0]) to the step everyone waits for. (batch 1: tb = t[0], the solo rule.)"""
        b, tb = (0, self.t[0]) if batch <= 1 else (batch - 1, tb)
        out, tok, reach = [(b + 1) / tb], 1.0, 1.0
        for k in range(1, self.kmax + 1):
            reach *= self.c[k - 1]
            tok += reach
            out.append((b + tok) / (self.t[k] if batch <= 1 else tb + self.t[k] - self.t[0]))
        return out

    def expect(self, k):
        """-> (expected tokens of a width-k round, P(all k drafts kept))."""
        tok = reach = 1.0
        for i in range(k):
            reach *= self.c[i]
            tok += reach
        return tok, reach

    def width(self, batch=1, tb=None):
        """Draft width for the next round (0: plain). batch / tb: see rates() (window g2 11-23: 4 parallel requests
        drafting on the solo rule got 137 tok/s aggregate vs 111 solo: every drafter round and verify row stalls the
        whole batch, so a request drafts only if that raises the step's tokens/ms)."""
        if not self.kmax:
            return 0
        r = self.rates(batch, tb)
        k = max(range(1, self.kmax + 1), key=r.__getitem__)
        self.idle += 1
        self.on = r[k] > r[0] * (1 + self.margin)
        if self.on:
            self.gap = self.probe
            if self.idle >= self.probe:
                self.idle, k = 0, min(k + 1, self.kmax)
            return k
        if self.idle >= self.gap:
            self.idle, self.gap = 0, min(2 * self.gap, self.pmax)
            return k
        return 0

    def update(self, drafted, accepted):
        """After a verify: drafted = len(draft) (0: nothing learned), accepted = drafts kept."""
        for i in range(min(drafted, accepted + 1, self.kmax)):  # kept ones, then the first rejected; later unseen
            self.c[i] += self.ema * ((i < accepted) - self.c[i])
        if drafted and accepted == drafted:  # all kept: probe / go deeper next round (fast turn-on after a change)
            self.idle = self.gap


class Gate:
    """Verify width from drafter confidence (INK_SPEC_GATE=1, DSpark only). Whether to draft at all is the Width
    controller's call, as without the gate (spec_runner); a drafting round always drafts 7 and pick() decides how many
    to verify. conf_i = max_v softmax(drafter logits row i) at T = 1. conf_i is a function of row i's q, which depends
    only on the drafts before it (not on d_i), so cutting the draft where confidence drops keeps sampling exact (a
    draft d_i that is verified was still drawn from q_i). a[b] = EMA of P(draft kept | drafts before it kept, conf in
    bin b), prior from the offline eval (tests/dspark_short.py --eval). pick(): verify the m in `widths` (BatchDecoder
    buckets T = 2/4/8) that maximizes expected tokens / verify ms (the drafter is paid by then); m = 0 if none beats a
    plain step; every `probe` verifying rounds one width wider (a bin whose estimate went stale can recover).
    (Until 11-23 the gate had its own on/off rule, an EMA of realized tokens/ms per drafting round vs plain: rounds
    where pick() rightly verified nothing count as below plain, a few in a row switched drafting off, and probes 8 ->
    16 -> 32 rounds apart each moved the EMA only 10% back: json greedy drafted in 36 of 151 rounds, x1.31 vs x1.81
    without the gate.)"""
    EDGES = (0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
    PRIOR = (0.04, 0.19, 0.24, 0.4, 0.4, 0.45, 0.6, 0.7, 0.95)   # CPU eval 11-23, greedy (tests/dspark_short.py --eval)

    def __init__(self, kmax=7, verify=None, widths=(1, 3, 7), ema=0.05, probe=8, prior=PRIOR):
        self.t = Width.STEP_MS if verify is None else verify
        self.kmax, self.widths = min(kmax, len(self.t) - 1), sorted(widths)
        self.ema, self.probe, self.a, self.on = ema, probe, list(prior), 0

    def bin(self, c):
        return sum(c >= e for e in self.EDGES)

    def pick(self, conf, kmax):
        """conf: confidences of the drafted rows -> drafts to verify (0..min(kmax, len(conf))). A stopping rule, so
        sampling stays exact: widths are tried in increasing order and going from m verified drafts to a wider w may
        look only at conf[0..m] (conf[m] is a function of q_{m+1}, i.e. of d_1..d_m); rows beyond m are estimated with
        conf[m]'s bin. Stops at the first width that doesn't improve the expected tokens / ms (11-23 review: an argmax
        over all rows let conf[j] (a function of d_j) decide whether d_j is verified, which biased sampled output)."""
        cap = min(kmax, len(conf), self.kmax)

        def rate(w, m):  # expected tokens / ms at width w, knowing conf[0..m]
            tok, reach = 1.0, 1.0
            for j in range(w):
                reach *= self.a[self.bin(conf[min(j, m)])]
                tok += reach
            return tok / self.t[w]
        bm, best = 0, 1 / self.t[0]
        for w in self.widths:
            if w > cap:
                break
            r = rate(w, bm)
            if r <= best:
                break
            bm, best = w, rate(w, w - 1) if w - 1 < len(conf) else r
        if bm:
            self.on += 1
            if self.on % self.probe == 0:
                bm = next((w for w in self.widths if bm < w <= cap), bm)
        return bm

    def update(self, conf, m, accepted, ctl=None):
        """After a drafting round: m drafts verified (0: drafted, none verified), `accepted` kept. ctl: the Width
        controller that decides whether to draft; it learns from the verified rows, and from a round that verified
        nothing it learns the estimated acceptance of draft 1 (otherwise it would never see that drafting stopped
        paying)."""
        for i in range(min(m, accepted + 1)):
            b = self.bin(conf[i])
            self.a[b] += self.ema * ((i < accepted) - self.a[b])
        if ctl is not None and ctl.kmax:
            if m:
                ctl.update(m, accepted)
            else:
                ctl.c[0] += ctl.ema * (self.a[self.bin(conf[0])] - ctl.c[0])


def accept_greedy(logits, draft):
    """logits [k+1, V] (rows as in the module doc), draft [k] -> emitted ids: the longest draft prefix equal to the
    row argmaxes, then the argmax of the next row. Identical to plain greedy given the same per-row logits."""
    top = logits[:len(draft) + 1].argmax(-1).tolist()
    n = 0
    while n < len(draft) and top[n] == draft[n]:
        n += 1
    return list(draft[:n]) + [top[n]]


def probs(logits, temperature=1.0, top_p=1.0, top_k=0):
    """Target distribution per row, [R, V] fp32, with decode.sample's rules (temperature, top-k keeps exactly k,
    top-p drops tokens whose exclusive cumulative probability exceeds p, renormalized). temperature > 0."""
    x = logits.float() / temperature
    if top_k and 0 < top_k < x.shape[-1]:
        kth = x.topk(top_k, -1)
        x = torch.full_like(x, float('-inf')).scatter_(-1, kth.indices, kth.values)
    p = torch.softmax(x, -1)
    if top_p and top_p < 1.0:
        s, i = p.sort(-1, descending=True)
        p = p.scatter(-1, i, s.masked_fill(s.cumsum(-1) - s > top_p, 0))
        p = p / p.sum(-1, keepdim=True)
    return p


def accept(p, draft, q=None, gen=None):
    """Exact speculative sampling. p [k+1, V] target probs per row; draft [k]; q [k, V] drafter probs (DSpark), or
    None for point-mass drafts (prompt lookup, q = 1 at d). Draft i is kept with prob min(1, p(d)/q(d)); at the first
    rejection the token is sampled from max(p - q, 0) normalized (point mass: p with d removed) and the round ends;
    if all are kept, a bonus token comes from the last row. Emitted tokens are distributed exactly as sampling from p
    row by row. Returns the emitted ids (1..k+1)."""
    out = []
    for i, d in enumerate(draft):
        pi = p[i]
        qd = 1.0 if q is None else float(q[i, d])
        if qd > 0 and float(torch.rand((), generator=gen, device=p.device)) * qd < float(pi[d]):
            out.append(int(d))
            continue
        if q is None:
            r = pi.clone()
            r[d] = 0
        else:
            r = (pi - q[i]).clamp_(min=0)
        if float(r.sum()) <= 0:  # p == q (or p is the point mass at d): residual is empty, only reachable by rounding
            r = pi
        return out + [int(torch.multinomial(r, 1, generator=gen))]
    return out + [int(torch.multinomial(p[len(draft)], 1, generator=gen))]


def accept_dev(p, draft, q=None, gen=None):
    """accept() with one host sync (INK_SPEC_ACCEPT_DEV=1): the same law, vectorized on p's device. Draft i is kept
    iff u_i q_i(d_i) < p_i(d_i) (u fp64 uniform) and all before it were kept; the first rejected row's token comes
    from its residual max(p - q, 0) (point mass: p without d; empty residual: p), else the bonus from row k, drawn by
    an exponential race with fp64 noise. The RNG stream differs from accept(), the distribution does not."""
    k, dev = len(draft), p.device
    if k:
        d, r = torch.tensor(draft, device=dev), torch.arange(k, device=dev)
        pd = p[r, d].double()
        if q is None:
            res, qd = p[:k].clone(), 1.0
            res[r, d] = 0
        else:
            res, qd = (p[:k] - q[:k]).clamp_(min=0), q[r, d].double()
        u = torch.rand(k, generator=gen, device=dev, dtype=torch.float64)
        n = (u * qd < pd).long().cumprod(0).sum()
        row = torch.cat((res, p[k:k + 1])).index_select(0, n.view(1))[0]
        row = torch.where(row.sum() > 0, row, p.index_select(0, n.view(1))[0])
    else:
        n, row = torch.zeros((), dtype=torch.long, device=dev), p[0]
    e = torch.empty(row.shape, dtype=torch.float64, device=dev).exponential_(generator=gen).clamp_(min=1e-300)
    n, t = torch.stack((n, (row.double() / e).argmax())).tolist()
    return list(draft[:n]) + [t]
