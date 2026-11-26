"""Speculative decoding of one sequence with the DSpark drafter (SPEC-dspark.md, DESIGN-batching.md sec. 3).

A slot of a BatchDecoder-like decoder (step(rows) / commit(acc) / tap_hist(slot, p0, n)) verifies, dspark.DSpark
(or a spec.Lookup) drafts, spec.py accepts. One round: Width picks k (<= 7) before drafting; draft k after the pending
token `last` at `pos`; verify T = k + 1 rows [last, d_1..d_k] at pos..pos+k; accept (greedy: accept_greedy; sampled:
accept with the drafter's q); commit the accepted rows; emit 1..k+1 tokens. Greedy output == plain greedy decoding on
the same decoder because verify rows are batch-invariant (decode_batch: a verify row == the same single-row step).
k = 0 rounds are plain decode steps: propose() returns the plain row and ctx None, the caller samples it as usual
(no settle). The drafter context catches up from the decoder's tap history (taps of every committed position,
written inside the step graph): one context-only drafter pass per 8 positions, or with the next drafting round.

    f, taps = forward_taps(m, x, seq)                 # prefill chunk (instead of m.forward): + the drafter taps
    sp = Spec(ds, bd, slot, start=P, temperature=..., cons=req.cons, stop=(chat.EOS,))
    sp.prompt(taps, a)                                # per prefill chunk starting at a (a gap restarts the context)
    bd.load(slot, seq); first = sample(m.logits(f[-1:])[0])
    rows, ctx = sp.propose(pos, last, left)           # rows may share a step with other slots' rows
    lg = bd.step(other_rows + rows)
    ctx None: sample lg[r0] as plain decoding; else out = sp.settle(lg, r0, ctx) -> emitted ids, slot committed
After settle, sp.lg = the verify rows' target logits (json mask applied; emitted token i came from row i, under the
request's temperature/top-p/top-k), e.g. for logprobs; a view of the step buffer, valid until the next step.

Device draft ids (decoders with dev_tok, i.e. BatchDecoder with INK_DEC_EMBED_GPU=1; DSpark without Gate): the
round's ids stay on the device, sp.dtok holds them and the caller passes bd.step(rows, tok={r0 + 1: sp.dtok}); the
host builds the step while the drafter runs and reads the ids only in settle (after the verify was launched; host()).
The draft is then not cut at a stop id before the verify (settle cuts the emitted tokens there, so the output is the
same) nor by the grammar (host() cuts it and builds the verify-row masks then: extra verify rows, same output), and
the drafter round time is not measured (no sync), so Width keeps the DRAFT_MS table.
Flags (off until measured, tests/window_spec2.sh): INK_SPEC_GATE=1 verify width from drafter confidence (spec.Gate;
the drafter always drafts 7, one CUDA graph per (greedy, bucket)); INK_SPEC_ACCEPT_DEV=1 sampled accept on the device
with one host sync (spec.accept_dev). On cuda the drafter's ms per round is measured on the fly (EMA, graph capture
rounds skipped) and fed to the controller's cost table, so shortlist / bucket speedups shift the width choice.
"""
import math
import os
import time

import torch

import chat
from dspark import BLOCK
import grammar
from grammar import ToolCall
from loopbatch import LoopBatch
from model import VOCAB
from decode import pos_noise, sample_rows
from spec import Copy, Gate, Lookup, Width, accept, accept_dev, accept_greedy, probs, verify16

TAP_VISITS = (5, 23, 41)    # dspark.Cfg.taps[:3]: h entering these visits; the 4th tap is the final normed state
VERIFY_MS = Width.STEP_MS   # verify step ms at T = 1..8 rows (BatchDecoder, GPU 2026-11-22)
RMAX = 16                   # rows of one decode step (decode_batch BUCKETS[-1]): a lookup draft is <= RMAX - 1
# INK_SPEC_HYBRID=1 (DSpark mode): prompt lookup alongside the drafter. Per round the best expected tokens/ms of: a plain
# step, a DSpark round (Width), a copied span from the prompt / output so far (<= 15 tokens, no drafter call), and a
# DSpark round of 7 followed by a copy continuing after its drafts (mixed proposal: q rows of the copy are one-hot).
HYBRID = os.environ.get('INK_SPEC_HYBRID') == '1'
COPY_EDGE = float(os.environ.get('INK_SPEC_COPY_EDGE', 1.0))  # a copy round must beat the DSpark round by this factor
# drafter round ms at k = 1..7, graphed, T=0 (GPU 2026-11-22: 2.98 at k=3, 3.28 at k=7, linear); sampled +0.36
DRAFT_MS = (2.83, 2.91, 2.98, 3.06, 3.13, 3.21, 3.28)
SAMPLED_MS = 0.36
# long context: ms per far key of one decode step's far-KV read (GPU 11-23, INK_DEC_FAR2: 131k 162 / 524k 112 / 1M 80
# tok/s -> ~6.9e-6 ms/key). A verify group of <= 10 rows reads it once (decode_batch.mt), so it is added once per step
FAR_MS = 6.9e-6
GATE = os.environ.get('INK_SPEC_GATE') == '1'
ACCEPT_DEV = os.environ.get('INK_SPEC_ACCEPT_DEV') == '1'
# ctx[2] of a round whose json verify-row masks are not built yet: host() builds them after the verify launch
PEND = object()
# DSpark + Width: after SLEEP consecutive rounds where drafting doesn't pay (probes included), the request goes dormant:
# plain rows only, no drafter work at all (no catch-up passes, no probes); WAKE positions later the drafter context
# restarts at the current position and the controller probes again. 0 = never sleep.
SLEEP = int(os.environ.get('INK_SPEC_SLEEP', 128))
WAKE = int(os.environ.get('INK_SPEC_WAKE', 512))


def forward_taps(m, x, seq):
    """model.forward(x, seq) that also returns the drafter taps of every row: -> (final [T, H], taps [4, T, H])
    bf16. Wraps Model.layer on the instance for the call (model.py unchanged)."""
    got = {}
    layer = m.layer

    def tap(L, h, blocks, msg, sq, v, pos):
        if v in TAP_VISITS:
            got[v] = h
        return layer(L, h, blocks, msg, sq, v, pos)
    m.layer = tap
    try:
        f = m.forward(x, seq)
    finally:
        del m.layer
    return f, torch.stack([got[v].bfloat16() for v in TAP_VISITS] + [f])


def step_ms(verify=VERIFY_MS, draft=DRAFT_MS, sampled=False):
    """Width's per-k round time: a (k+1)-row verify step, + the drafter round for k >= 1. k = 0 is exactly a plain
    step (no drafter call; the context catches up from the decoder's tap history, 1/8 of a context pass per token)."""
    return tuple(t + (draft[k - 1] + SAMPLED_MS * sampled if k else 0) for k, t in enumerate(verify))


class Spec:
    """Per sequence: drafter state (ds + st; or lookup: a spec.Lookup, ds None), width controller, sampler settings,
    json constraint (grammar.Json or None) and stats. stop: ids that end the reply (a draft ends with the first one;
    emitted tokens are cut after it). slot may be set later (engine: the Spec exists from prefill on)."""

    def __init__(self, ds, bd, slot, start=0, kmax=7, temperature=0.0, top_p=1.0, top_k=0, gen=None, cons=None,
                 stop=(chat.EOS,), times=None, st=None, lookup=None, vocab=None, dev=None, gate=None, accept_dev=None,
                 min_p=0.0, hybrid=None, seed=None, pred=(), pstats=None):
        # pred: a predicted output (ids) the copy drafter may copy from (lookup= gets it in its own Lookup(pre=));
        # pstats: dict that gets accepted_prediction_tokens / rejected_prediction_tokens (drafts copied from pred)
        self.ds, self.bd, self.slot, self.lk, self.times = ds, bd, slot, lookup, times
        self.pstats = pstats
        self.st = None
        if ds is not None:
            self.st = ds.new_state(start) if st is None else st  # st: a State to reuse (reset; keeps its graphs)
            if st is not None:
                ds.reset(st, start)
        self.ctl = Width(kmax, step_ms=times or (VERIFY_MS if ds is None else step_ms(sampled=temperature > 0)))
        self.gate = None
        if ds is not None and (GATE if gate is None else gate):
            self.gate = Gate(kmax, verify=times or VERIFY_MS)
        # hybrid: ids so far (filled from hist() at the first round, then kept in step), copy acceptance, verify ms
        # of T = 1..16 rows (a lookup round has no drafter cost)
        self.hy = Lookup(pre=pred) if ds is not None and (HYBRID if hybrid is None else hybrid) else None
        self.vbase = times or VERIFY_MS                   # verify-only ms per width (ctl.t minus the drafter)
        self.cpr, self.vt = Copy(), verify16(self.vbase)
        self.src = None                                   # this round: (DSpark drafts, copy bucket or None)
        self.lk_ok = False                                # the lookup's ids end with this round's `last`
        self.acc_dev = ACCEPT_DEV if accept_dev is None else accept_dev
        self.temperature, self.top_p, self.top_k, self.gen, self.cons = temperature, top_p, top_k, gen, cons
        self.min_p = min_p                                # sampled: target rows keep p >= min_p * p_max (as plain)
        # seed (engine): sampled rounds use the Exp(1) noise of each token's position (decode.pos_noise) for the drafter's
        # race AND the target's: row i emits exactly what plain decoding samples at that position, a draft is kept iff it
        # equals it (coupled races). Output = plain sampled output with the same seed, whatever the widths / schedule.
        # None (tests, tools): accept / accept_dev with gen (exact law, but the stream depends on the schedule).
        self.seed, self.nz, self.p0 = seed, {}, 0
        self.stop = set(stop)
        self.V = vocab or (getattr(ds, 'V', None) or ds.U.shape[0] if ds is not None else VOCAB)
        self.dev = dev or (ds.dev if ds is not None else torch.device('cpu'))
        self.short = getattr(ds, 'short', 0)              # shortlist head: the drafter must see this sequence's tokens
        self.noted = False
        self.adapt = times is None and ds is not None and self.dev.type == 'cuda'
        self.dms = None                                   # measured drafter ms per round (EMA, cuda)
        self.conf = self.lg = self.dtok = None
        self.offrun, self.sleep_at = 0, None              # rounds in a row not worth drafting; dormant since pos
        self.dev_tok = getattr(bd, 'dev_tok', False) and ds is not None and self.gate is None
        self.gm = self.gfree = None                       # grammar masks of the verify rows [8, V] (lazy)
        self.gbits = None                                 # their json rows' xgrammar bitmasks (host, pinned on cuda)
        self.stats = dict(rounds=0, drafted=0, accepted=0, emitted=0,    # (rounds/emitted: drafting rounds only)
                          draft_s=0.0, step_s=0.0, settle_s=0.0,   # settle: waits for the verify + accepts
                          tried=[0] * RMAX, kept=[0] * RMAX,   # per draft position: drafted / accepted
                          widths=[0] * 8)                 # rounds per chosen width k

    def prompt(self, taps, a=None):
        """Context rows for prompt positions a .. a+n-1 (taps [4, n, H] from forward_taps); a != st.n (chunks
        skipped or a restored prefix) restarts the context at a."""
        if a is not None and a != self.st.n:
            self.ds.reset(self.st, a)
        self.ds.extend(self.st, taps)

    def attach(self, slot, pos):
        """The sequence is now in decoder slot `slot` at position pos (no taps for some positions: restart there)."""
        self.slot = slot
        if self.st is not None and self.st.n != pos:
            self.ds.reset(self.st, pos)
        self._far(pos)

    def _far(self, pos):
        """Default cost tables at long context: every step (plain or verify) pays the far-KV read once (twice past 10
        rows), so verify rows cost relatively less and drafting pays more often. Measured times (times=) are kept."""
        off = FAR_MS * max(pos - 1023, 0)
        if off <= 0 or self.times is not None:
            return
        self.vbase = tuple(t + off for t in VERIFY_MS)
        self.vt = tuple(t + off * (1 if T <= 10 else 2) for T, t in enumerate(verify16(VERIFY_MS), 1))
        self.ctl.t = self.vbase if self.ds is None else step_ms(self.vbase, (self.dms,) * 7 if self.dms else DRAFT_MS,
                                                                sampled=self.temperature > 0 and self.dms is None)
        if self.gate is not None:
            self.gate.t = self.vbase

    def _catch(self, n):
        """Context-only drafter pass over the next n <= 8 committed positions (taps from the decoder's history)."""
        self.ds.stage(self.st, self.bd.tap_hist(self.slot, self.st.n, n))
        self.ds.flush(self.st)

    def flush(self, pos):
        """Bring the drafter context up to the slot's next position pos (st.n == pos afterwards)."""
        while pos > self.st.n:
            self._catch(min(BLOCK, pos - self.st.n))

    def propose(self, pos, last, left, kcap=7, hist=None, batch=1):
        """Draft -> (rows [(slot, pos + i, token)], ctx for settle), or ([plain row], None) when nothing is drafted
        (k = 0, no lookup match): the caller samples that row like plain decoding. kcap: rows this slot may add to
        the step; hist(): every token so far incl. last (lookup; shortlist: read once for the prompt's tokens);
        batch: requests in this decode step (Width weighs the step's total tokens/ms)."""
        st, self.dtok, self.src, self.lk_ok = self.st, None, None, False
        self.p0, self.nz = pos, {}
        for lk in (self.lk, self.hy):
            if lk is not None:
                lk.from_pred = False  # (set again by a span copied from the prediction this round)
        if self.lk is not None:
            self.lk_ok = self._sync(self.lk, pos, last, hist)
        cp = self._copy(pos, last, left, kcap, hist) if self.hy is not None else None   # hybrid: best copy round
        if self.sleep_at is not None:                     # dormant: no drafter work at all (copies still allowed)
            if pos - self.sleep_at < WAKE:
                self.stats['slept'] = self.stats.get('slept', 0) + 1
                if cp is not None and cp[1] > (1 + self.ctl.margin) / self.ctl.t[0]:
                    return self._copy_round(pos, last, cp, catch=False)
                self.stats['widths'][0] += 1
                return [(self.slot, pos, last)], None
            self.sleep_at, self.offrun, self.noted = None, 0, False
            self.ds.reset(st, pos)                        # (no taps kept for the dormant span: context restarts here)
        if self.short:
            self.ds.note(st, hist() if not self.noted and hist is not None else [last])
            self.noted = True
        tb = self.vbase[min(batch, len(self.vbase)) - 1]  # a plain step of `batch` rows (~ verify T = batch)
        k = min(self.ctl.width(batch, tb), left - 1, kcap)  # (with the gate too: Width decides whether to draft)
        if st is not None and SLEEP:
            self.offrun = 0 if self.ctl.on else self.offrun + 1
            if self.offrun >= SLEEP:
                self.sleep_at, k = pos, 0
                self.stats['sleeps'] = self.stats.get('sleeps', 0) + 1
                if cp is None:
                    self.stats['widths'][0] += 1
                    return [(self.slot, pos, last)], None
        if cp is not None and cp[1] > max(COPY_EDGE * self.ctl.rates()[k] if k > 0 else 0,
                                          (1 + self.ctl.margin) / self.ctl.t[0]):
            return self._copy_round(pos, last, cp, catch=self.sleep_at is None)
        if k <= 0:
            self.stats['widths'][0] += 1
            if st is not None and self.sleep_at is None and pos - st.n >= BLOCK:   # drafter catch-up: 1 pass / 8
                self._catch(BLOCK)
            return [(self.slot, pos, last)], None
        t0 = time.perf_counter()
        if self.lk is not None:
            d, q = self.lk.draft(k, stop=self.stop), None
            self.stats['widths'][k] += 1
        else:
            assert st.pend == 0 and st.n <= pos, (st.n, st.pend, pos)
            while pos - st.n > BLOCK:
                self._catch(BLOCK)
            if pos > st.n:                                # the rest goes into the context with this round
                self.ds.stage(st, self.bd.tap_hist(self.slot, st.n, pos - st.n))
            if self.dev_tok:                              # ids stay on the device (read in settle)
                self.dtok, q = self.ds.round(st, last, k, self.temperature, self.gen, host=False, noise=self._dnoise(k))
                self.stats['widths'][k] += 1
                self.stats['draft_s'] += time.perf_counter() - t0
                rows = [(self.slot, pos, last)] + [(self.slot, pos + 1 + i, None) for i in range(k)]
                return rows, [self.dtok, q, None, None]
            if self.gate is not None:                     # draft 7, verify the confident lead (Gate.pick)
                d, q, self.conf = self.ds.round(st, last, BLOCK - 1, self.temperature, self.gen, conf=True,
                                                noise=self._dnoise(BLOCK - 1))
                k = self.gate.pick(self.conf, min(BLOCK - 1, left - 1, kcap))
                d = d[:k]
            else:
                d, q = self.ds.round(st, last, k, self.temperature, self.gen, noise=self._dnoise(k))
            self.stats['widths'][k] += 1
            self._cost(time.perf_counter() - t0)
            for i, t in enumerate(d):                     # a draft ends with the first stop id
                if t in self.stop:
                    d = d[:i + 1]
                    break
            if self.hy is not None and d and len(d) == k:  # hybrid: a copy continuing after the drafts
                d, q = self._extend(d, q, left, kcap)
        self.stats['draft_s'] += time.perf_counter() - t0
        if not d:
            if self.gate is not None:
                self.gate.update(self.conf, 0, 0, self.ctl)
            return [(self.slot, pos, last)], None
        masks = bad = None
        if self.cons is not None:
            d, bad, masks = self._grammar(d)
        rows = [(self.slot, pos + i, t) for i, t in enumerate([last] + d)]
        return rows, [d, q, masks, bad]

    def _noise(self, pos):
        """Exp(1) [V] of the token at position pos (cached for this round: the drafter and settle share it)."""
        nz = self.nz.get(pos)
        if nz is None:
            nz = self.nz[pos] = pos_noise(self.gen, self.seed, pos, self.V, self.dev)
        return nz

    def _dnoise(self, k):
        """Drafter noise of drafts 1..k (positions p0+1..p0+k), or None (greedy / unseeded: the drafter draws its own)."""
        if self.seed is None or self.temperature <= 0:
            return None
        return [self._noise(self.p0 + 1 + i) for i in range(k)]

    def _cost(self, dt):
        """Measured drafter round (s, incl. its host part and the ids sync) -> EMA -> the controller's cost table."""
        if not self.adapt or self.st.captured:
            return
        ms = 1e3 * dt
        self.dms = ms if self.dms is None else self.dms + 0.1 * (ms - self.dms)
        self.ctl.t = step_ms(self.vbase, (self.dms,) * 7)

    @staticmethod
    def _sync(lk, pos, last, hist):
        """lk's ids = every token up to and including `last` (position pos): one append per round (settle appends the
        committed drafts); rebuilt from hist() only when out of step (first round, resumed decode)."""
        if len(lk) == pos:
            lk.extend([last])
        elif len(lk) != pos + 1 and hist is not None:
            h = hist()
            lk.L, lk.last = lk.base, None  # (a prediction region before base stays)
            lk.extend(h)
        return len(lk) == pos + 1

    def _copy(self, pos, last, left, kcap, hist):
        """Hybrid: the best copy round now -> (m, tokens/ms, j, bucket) or None (no match, or nothing pays)."""
        self.lk_ok = self._sync(self.hy, pos, last, hist)
        if not self.lk_ok:
            return None
        j, n = self.hy.match()
        if j is None:
            return None
        b = self.cpr.bucket(n)
        m, r = self.cpr.best(b, min(kcap, left - 1, RMAX - 1), self.vt)
        return (m, r, j, b) if m else None

    def _copy_round(self, pos, last, cp, catch=True):
        """Verify a copied span (no drafter call; point-mass proposal). The drafter context first catches up to
        < 8 positions behind, so the decoder's tap ring (TH = 32) still holds them after up to 16 verify rows."""
        m, _, j, b = cp
        st = self.st
        while catch and pos - st.n >= BLOCK:
            self._catch(BLOCK)
        d = self.hy.span(j, m, self.stop)
        self.src = (0, b)
        self.stats['copy_rounds'] = self.stats.get('copy_rounds', 0) + 1
        masks = bad = None
        if self.cons is not None:
            d, bad, masks = self._grammar(d)
        if not d and bad is None:
            return [(self.slot, pos, last)], None
        return [(self.slot, pos + i, t) for i, t in enumerate([last] + d)], [d, None, masks, bad]

    def _extend(self, d, q, left, kcap):
        """Hybrid: DSpark drafted d (all k it was asked for, no stop id); if a copy continues after d and pays (the
        copied rows are reached only if all k drafts are kept), append it. The copy is a deterministic function of the
        drafts before it, so its q rows are one-hot (spec.accept's point-mass rule). Exact: this only adds rows after
        the drafter's, decided from d_1..d_k."""
        k = len(d)
        room = min(kcap, left - 1, RMAX - 1) - k
        if room <= 0:
            return d, q
        j, n = self.hy.after(d)
        if j is None:
            return d, q
        b = self.cpr.bucket(n)
        tok, reach = self.ctl.expect(k)
        dms = self.ctl.t[k] - self.vbase[k]               # the drafter round (paid either way)
        m, _ = self.cpr.best(b, room, [t + dms for t in self.vt], base=tok, reach=reach, k0=k)
        x = self.hy.span(j, m, self.stop) if m else []
        if not x:
            return d, q
        self.src = (k, b)
        self.stats['ext_rounds'] = self.stats.get('ext_rounds', 0) + 1
        if q is not None:
            oh = torch.zeros(len(x), q.shape[1], dtype=q.dtype, device=q.device)
            oh[torch.arange(len(x), device=q.device), torch.tensor(x, device=q.device)] = 1.0
            q = torch.cat((q[:k], oh))
        return d + x, q

    def settle(self, lg, r0, ctx):
        """Accept against the step's logits rows r0.. -> emitted ids; commits the slot, advances the json
        constraint. (The emitted ids are cut after a stop id / a finished constraint: the reply ends there.)"""
        t0 = time.perf_counter()
        d, q, masks, bad = self.host(ctx)
        T = len(d) + 1
        lg = lg[r0:r0 + T]
        if masks is not None:
            lg = lg + masks[:T]
        self.lg = lg                                      # (before min_p: the distribution logprobs report)
        if self.temperature <= 0:
            out = accept_greedy(lg, d)
        else:
            # a draft the grammar rejected stays in the draft (p = 0 there: rejected for sure, the token comes from
            # the residual max(p - q, 0) of its row, which is what keeps this exact) but gets no verify row
            if self.min_p:                                # l >= l_max + T ln(min_p), as engine._take
                lg = lg.masked_fill(lg < lg.max(-1, keepdim=True).values + self.temperature * math.log(self.min_p),
                                    float('-inf'))
            dd = d if bad is None else d + [bad]
            if self.seed is not None:  # row i's token = plain sampling's at position p0+1+i; keep drafts while equal
                t = sample_rows(lg, [(self.temperature, self.top_p, self.top_k, self._noise(self.p0 + 1 + i))
                                     for i in range(T)]).tolist()
                n = next((i for i, x in enumerate(dd) if x != t[i]), len(dd))
                out = t[:n + 1]
            else:
                out = (accept_dev if self.acc_dev else accept)(
                    probs(lg, self.temperature, self.top_p, self.top_k), dd, None if q is None else q[:len(dd)], gen=self.gen)
        a = len(out) - 1
        self.ncommit = a + 1                              # rows committed (the slot's position advances by this)
        if T > 1:
            self.bd.commit({self.slot: a + 1})
        kd, b = self.src if self.src is not None else (len(d), None)   # drafter rows, copy bucket (hybrid)
        kd = min(kd, len(d))
        if self.gate is not None and kd:
            self.gate.update(self.conf, kd, min(a, kd), self.ctl)
        elif kd:
            self.ctl.update(kd, min(a, kd))
        if b is not None and a >= kd:                     # the copied rows were reached
            self.cpr.update(b, len(d) - kd, a - kd)
        lk = self.lk if self.lk is not None else self.hy
        if lk is not None and lk.from_pred and self.pstats is not None:  # copied rows came from the prediction
            c0 = 0 if self.lk is not None else kd         # (lookup mode: the whole draft is the copy)
            acc = max(0, min(a, len(d)) - c0)
            for k_, v in (('accepted_prediction_tokens', acc), ('rejected_prediction_tokens', len(d) - c0 - acc)):
                self.pstats[k_] = self.pstats.get(k_, 0) + v
        if lk is not None and self.lk_ok:
            lk.extend(d[:a])                              # committed drafts (last is already in)
        if self.short:
            self.ds.note(self.st, out)
        s = self.stats
        s['rounds'] += 1
        s['drafted'] += len(d)
        s['accepted'] += a
        for i in range(len(d)):
            s['tried'][i] += 1
            s['kept'][i] += i < a
        for i, t in enumerate(out):                       # the reply ends at a stop id (a bonus after it is dropped)
            if t in self.stop:
                out = out[:i + 1]
                break
        if self.cons is not None:
            for i, t in enumerate(out):
                self.cons.accept(t)
                if self.cons.done:                        # json finished: a bonus token after it is dropped
                    out = out[:i + 1]
                    break
        s['emitted'] += len(out)
        s['settle_s'] += time.perf_counter() - t0
        return out

    def host(self, ctx):
        """After the verify launch (idempotent): device draft ids (dev_tok) in ctx -> host ids, cut + masked by the
        constraint, which sees them only now (the verify rows after a cut are just unused); pending json masks
        (PEND) built, overlapping the verify. -> ctx (in place)."""
        if torch.is_tensor(ctx[0]):
            ctx[0] = ctx[0].tolist()
            if self.cons is not None:
                ctx[0], ctx[3], ctx[2] = self._grammar(ctx[0])
        if ctx[2] is PEND:
            ctx[2] = self._fill(ctx[0])
        return ctx

    def _grammar(self, d):
        """Cut the draft at the first token the constraint rejects (or after the one that finishes it) -> (kept,
        rejected token or None, masks): the additive masks (0 / -inf) of the T = len(kept) + 1 verify rows along the
        draft path. Before the launch only the cut (matcher accept_token, allows()) and the non-json rows (rare: call
        headers); json rows (xgrammar bitmask, ~0.25 ms each) make masks PEND: host() fills them after the launch
        (_fill). The constraint is rolled back (settle() advances it by the emitted tokens)."""
        cons = self.cons
        if cons.state == 'free' and isinstance(cons, ToolCall) and chat.MODEL not in d and chat.INVOKE_JSON not in d:
            if self.gfree is None:                        # tool_choice auto outside a call: one fixed mask (every row
                self.gfree = torch.zeros(RMAX + 1, self.V, device=self.dev)  # only bans INVOKE_JSON, ToolCall.mask's
                self.gfree[:, chat.INVOKE_JSON] = float('-inf')               # 'free'), no per-round work, no syncs
            return d, None, self.gfree
        if self.gm is None:
            self.gm = torch.zeros(RMAX + 1, self.V, device=self.dev)
            self.gbits = grammar.xgr.allocate_token_bitmask(RMAX + 1, grammar.VOCAB)
            if self.dev.type == 'cuda':                   # (async copy in _fill)
                self.gbits = self.gbits.pin_memory()
        g, js = self.gm, []
        g[:len(d) + 1].zero_()
        kept, bad = self._walk(d, lambda i: js.append(i) if cons.state == 'json' else cons.mask(g[i]))
        return kept, bad, PEND if js else g

    def _walk(self, d, row):
        """Advance the constraint along d, calling row(i) at verify row i's state (rows 0..len(kept); the bonus row
        unless the constraint finished), up to the first rejected token / after the finishing one -> (kept, bad).
        Rolled back afterwards: state + done + m, m.rollback(json tokens accepted on the original matcher)."""
        cons = self.cons
        state, done, m0, nj, kept, bad = cons.state, cons.done, getattr(cons, 'm', None), 0, [], None
        try:
            for t in d:
                row(len(kept))
                j = cons.state == 'json' and cons.m is m0   # (a later call in the draft gets its own matcher)
                if not cons.accept(t, check=True):
                    bad = t
                    break
                nj += j
                kept.append(t)
                if cons.done:
                    break
            if bad is None and not cons.done:             # (after a finishing token the bonus row is dropped anyway)
                row(len(kept))
        finally:
            if nj:
                m0.rollback(nj)
            cons.state, cons.done, cons.m = state, done, m0
        return kept, bad

    def _fill(self, kept):
        """The json rows of a PEND round's masks: replay the kept draft from the (rolled back) constraint, bitmasks
        into the host buffer, one copy (async on cuda: the verify is still running), unpack, -inf per run of json rows
        (the non-json rows were masked in _grammar). -> the masks."""
        cons, g, b, js = self.cons, self.gm, self.gbits, []

        def row(i):
            if cons.state == 'json':
                cons.m.fill_next_token_bitmask(b, i)
                js.append(i)
        k, bad = self._walk(kept, row)
        assert k == kept and bad is None, (k, kept, bad)
        ok = grammar.unpack(b[:js[-1] + 1], self.dev, non_blocking=self.dev.type == 'cuda')
        a = 0
        for n in range(1, len(js) + 1):
            if n == len(js) or js[n] != js[n - 1] + 1:
                r0, r1 = js[a], js[n - 1] + 1
                g[r0:r1].masked_fill_(~ok[r0:r1], float('-inf'))
                a = n
        return g


class SeqBatch(LoopBatch):
    """CPU stand-in for BatchDecoder's spec contract on LoopBatch's plain Seq path (tests, CPU engine): a slot's T > 1
    rows (verify) = T single-row forwards with a state snapshot after each, commit() restores the accepted one (the
    batch-invariance contract of decode_batch holds by construction); taps of every row via forward_taps, kept per
    slot by position for tap_hist."""

    def __init__(self, model, slots=4, dev_tok=None, taps=True):
        super().__init__(model, slots, dec=False)
        self.pend, self.th, self.fw, self.use_taps = {}, [{} for _ in range(slots)], 0, taps
        # step(rows, tok=...) as BatchDecoder's (tests the Spec device-id path on CPU; INK_DEC_EMBED_GPU=1 turns it on)
        self.dev_tok = os.environ.get('INK_DEC_EMBED_GPU') == '1' if dev_tok is None else dev_tok

    def free(self, slot):
        super().free(slot)
        self.pend.pop(slot, None)
        self.th[slot].clear()

    @torch.no_grad()
    def step(self, rows, tok=None):
        assert not self.pend, f'commit() the verify rows of slots {sorted(self.pend)} first'
        rows = list(rows)
        for r0, t in (tok or {}).items():
            for j, x in enumerate(t.tolist()):
                rows[r0 + j] = rows[r0 + j][:2] + (x,)
        n = {}
        for slot, _, _ in rows:
            n[slot] = n.get(slot, 0) + 1
        lg, tp, m = [], [], self.m
        for slot, pos, t in rows:
            seq = self.seqs[slot]
            assert seq is not None and pos == seq.pos, (slot, pos, seq and seq.pos)
            if self.use_taps:
                f, taps = forward_taps(m, m.embed([t]), seq)
                tp.append(taps)
                self.th[slot][pos] = taps[:, 0]
                if len(self.th[slot]) > 96:
                    for p in [p for p in self.th[slot] if p < pos - 64]:
                        del self.th[slot][p]
            else:
                f = m.forward(m.embed([t]), seq)
            self.fw += 1
            lg.append(m.logits(f))
            if n[slot] > 1:
                self.pend.setdefault(slot, []).append(
                    (seq.pos, [{k: x.clone() for k, x in c.items()} for c in seq.conv], list(seq.ring)))
        self.taps = torch.cat(tp, 1) if tp else None
        return torch.cat(lg)

    def commit(self, acc=None):
        acc = {s: len(v) for s, v in self.pend.items()} if acc is None else acc
        for slot, k in acc.items():
            if slot in self.pend:
                seq = self.seqs[slot]
                seq.pos, seq.conv, seq.ring = self.pend.pop(slot)[k - 1]
            else:
                assert k == 1, f'slot {slot}: no verify rows pending'

    def tap_rows(self, r0, n):
        return self.taps[:, r0:r0 + n]

    def tap_hist(self, slot, p0, n):
        th, pos = self.th[slot], self.seqs[slot].pos
        assert slot not in self.pend and p0 + n <= pos, (slot, p0, n, pos)
        for p in [p for p in th if p < pos - 64]:
            del th[p]
        return torch.stack([th[p] for p in range(p0, p0 + n)], 1)

    def state(self, slot):
        assert slot not in self.pend, 'commit() first'
        return super().state(slot)


def generate(bd, slot, pos, first, max_tokens, ds=None, taps=None, temperature=0.0, top_p=1.0, top_k=0, gen=None,
             cons=None, stop=(chat.EOS,), kmax=7, times=None, st=None, prompt=None, **kw):
    """Decode one loaded slot from position pos with pending token `first` (sampled from the prompt) until a stop id,
    a finished json constraint or max_tokens (first counts). ds None: plain decoding (one row per step, the same
    sampling rules as the engine), else speculative with prompt taps `taps` [4, pos, H] (or None: state starts at
    pos); its k = 0 rounds are exactly the plain steps. prompt: the prompt's ids (shortlist head); kw: Spec options
    (gate, accept_dev). -> (tokens, Spec or None)."""
    from decode import sample
    out = [first]
    if cons is not None:
        cons.accept(first)
    sp = None
    if ds is not None:
        sp = Spec(ds, bd, slot, start=0 if taps is not None else pos, kmax=kmax, temperature=temperature, top_p=top_p,
                  top_k=top_k, gen=gen, cons=cons, stop=stop, times=times, st=st, **kw)
        if taps is not None:
            sp.prompt(taps)
    end = lambda: out[-1] in stop or len(out) >= max_tokens or (cons is not None and cons.done)
    while not end():
        rows, ctx = ([(slot, pos, out[-1])], None) if sp is None else sp.propose(
            pos, out[-1], max_tokens - len(out), kcap=RMAX - 1, hist=lambda: list(prompt or ()) + out)
        t0 = time.perf_counter()
        lg = bd.step(rows) if sp is None or sp.dtok is None else bd.step(rows, tok={1: sp.dtok})
        if ctx is None:
            lg = lg[0]
            pos += 1
            if cons is not None:
                cons.mask(lg)
            t = int(sample(lg, temperature, top_p, top_k, gen))
            if cons is not None:
                cons.accept(t)
            out.append(t)
        else:
            sp.stats['step_s'] += time.perf_counter() - t0
            e = sp.settle(lg, 0, ctx)
            pos += len(e)
            out += e
    return out, sp
