"""DSpark drafter for Inkling 2 Mini (mtp.safetensors), written from SPEC-dspark.md (section numbers refer to it).

Plain PyTorch, any device. The drafter reuses the target's embedding and unembed (passed in). Every UNCONFIRMED [U]
wiring choice of the spec is a field of Cfg; the defaults are the combination tests/dspark_ablate.py measured best
(SPEC-dspark.md "Measured").

    ds = DSpark(MTP, embed=m.W.embed, unembed=m.W['unembed.weight'][:VOCAB], embed_norm=m.W['embed_norm.weight'])
    st = ds.new_state(); ds.extend(st, taps)          # taps [4, n, H]: target states Cfg.taps for positions 0..n-1
    d, q = ds.draft(st, x0, k=7, temperature=t)       # x0 = token at position st.n (not yet seen by the target)
    ... verify [x0] + d at positions st.n .. st.n+k, accept a drafts ...
    ds.extend(st, taps_of_verify_rows[:, :a + 1])     # context rows for the anchor + accepted drafts only

Serving path (GPU, spec_runner.py; Cfg defaults only): fixed-shape, device-only, one CUDA graph per (k, greedy):
    ds.stage(st, taps_of_verify_rows[:, :a + 1])      # copied into st's static input (<= 8 rows)
    d, q = ds.round(st, x0, k, temperature, gen)      # = extend(staged rows) + draft, in one graph replay
Context K/V live in per-layer rings: layers 0/2 the last WIN = 1024 positions (all a sliding layer can see), layer 1
(global) the last `ctx` positions (INK_DSPARK_CTX, default 32768 on cuda: 128 MiB bf16; older context is dropped,
which only costs acceptance on drafts that need a fact from > 32k tokens back; the taps of the recent rows already
carry the target's full-context view). Logits use U h + (U W_out) tanh(o) + U b_out, so a round reads the unembed
once for all 7 rows (base GEMM) plus 7 x the 102 MB U' = U @ hidden_out instead of 7 x 0.92 GB.

Round-cost options (flags, off by default until measured in a window; tests/window_spec2.sh):
- INK_DSPARK_SHORT=M0: shortlist head. Drafter logits only over S = the M0 most frequent tokens (data/dspark_short.pt,
  built by tests/dspark_short.py from local text) + every special + up to INK_DSPARK_DYN (4096) tokens seen in this
  sequence (prompt + output; a ring, newest kept). Rows of U, U' and c for S are copied once (static) / per State
  (dynamic), so a round reads ~0.27 GB of head instead of ~1.6 GB. Exact: q is softmax over S (0 elsewhere), drafts
  are drawn from that q (exponential race with fp64 noise), and accept() gets the same q scattered to [k, V].
- INK_DSPARK_CTXB=1: while the context is shorter than the layer-1 ring, that layer attends over the filled prefix of
  the ring only (power-of-2 bucket >= 1024, one CUDA graph per bucket) instead of all 32769 slots.
- round(..., conf=True) also returns each row's confidence max_v softmax(logits_v) at T = 1 over the drafter's
  support (a function of the row's q only, so a verify width chosen from it keeps sampling exact; spec.Gate).
"""
import os
import weakref
from dataclasses import dataclass
from types import SimpleNamespace

import torch

from model import VOCAB, H, rms

NQ, NKV, HD, BLOCK, R, WIN = 24, 8, 128, 8, 256, 1024
SLIDING = (0, 2)
CTX = int(os.environ.get('INK_DSPARK_CTX', 32768))  # layer-1 (global) context ring on cuda (CPU default 4096: cheap eval)
MTP = os.path.join(os.environ.get('INK_MODEL', os.path.expanduser('~/llm/models/Inkling-2-Mini-Uncensored')),
                   'mtp.safetensors')
SHORT = int(os.environ.get('INK_DSPARK_SHORT', 0))   # static shortlist size (0: full-vocab head)
DYN = int(os.environ.get('INK_DSPARK_DYN', 4096))     # per-sequence shortlist slots for context tokens outside it
CTXB = os.environ.get('INK_DSPARK_CTXB') == '1'
SHORT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'dspark_short.pt')
SPECIAL = 200000                                      # ids >= this are specials: always in the shortlist


def static_short(m0, rank=None, V=VOCAB):
    """Sorted ids of the static shortlist: the m0 top-ranked ids (rank: ids by frequency, default data/dspark_short.pt)
    + every special id < V."""
    rank = (torch.load(SHORT_FILE)['rank'] if rank is None else rank).long()
    rank = rank[(rank < min(V, SPECIAL))][:m0]
    return torch.cat((rank, torch.arange(SPECIAL, max(SPECIAL, V)))).unique()


def race(q, e):
    """Exact draw from each row of q [N, M] given iid Exp(1) noise e [N, M] (fp64): argmax q / e. P(i) = q_i / sum q
    (exponential race); fp64 noise so tokens with tiny q stay drawable (fp32 noise with a 24-bit uniform would cap
    1/e at ~2^24). -> indices [N]."""
    return (q.double() / e).argmax(-1)


@dataclass(frozen=True)
class Cfg:
    taps: tuple = (5, 23, 41, 48)    # target hs indices (sec. 3): k < 48 = h entering visit k, 48 = final normed state
    fuse: str = 'row'                # ctx_c = sum_j F[c, j] tap_j ('row') or F[j, c] ('col')
    fuse_softmax: bool = True        # softmax over each consumer's 4 tap weights
    tap_norm: bool = False           # RMS-normalize (no weight) each tap before mixing
    consumers: tuple = (0, 1, 2, 3)  # fused row that feeds layer 0, layer 1, layer 2, the head
    anchor: str = 'normed'           # anchor row input: target embed row 'raw' or 'normed' (embed_norm)
    mask_scale: float = 1.0          # mask_embed multiplier
    head: str = 'a'                  # 4th 2304-wide rnn input: 'a' head ctx feature at p-1, 'b' dflare_norm(E[prev]),
    #                                  'c' dflare_norm(pre-norm backbone row), 'z' zeros (ablation);
    #                                  'none' = backbone only (no bias)
    zorder: str = 'swhe'             # order of s, W1[prev], h, extra in z
    gco: str = 'gco'                 # order of the gate / candidate / output chunks of rnn_proj
    bias: str = 'post'               # head bias added after the final norm ('post') or before it ('pre')
    softcap: bool = True             # 30 tanh(x / 30) on drafter logits (affects q, not argmax)
    rope: str = 'neox'               # rotate-half ('neox') or interleaved pairs ('gptj')
    win_incl: bool = False           # sliding layers: pos_q - pos_k < 1024 (False) or <= 1024 (True)
    row0: bool = False               # True: row k drafts p+k+1 (DSpark paper); False: DFlash, row 0 dropped, 7 drafts
    ctx_rope: bool = True            # RoPE on context keys (at their own positions + ctx_pos)
    ctx_pos: int = 0                 # context row t sits at position t + ctx_pos


_INV = {}


def rope(x, pos, mode):
    """x [..., heads, 128] (fp32 math), pos [...] absolute positions; theta 1e6 over all 128 dims."""
    inv = _INV.get(x.device)
    if inv is None:   # (cached: 4 fewer launches per call in a round)
        inv = _INV[x.device] = 1e6 ** (-torch.arange(0, HD, 2, device=x.device, dtype=torch.float32) / HD)
    ang = pos[..., None, None].float() * inv                                       # [..., 1, 64]
    c, s, xf = ang.cos(), ang.sin(), x.float()
    if mode == 'neox':
        a, b = xf[..., :HD // 2], xf[..., HD // 2:]
        return torch.cat((a * c - b * s, b * c + a * s), -1).to(x.dtype)
    a, b = xf[..., 0::2], xf[..., 1::2]
    return torch.stack((a * c - b * s, b * c + a * s), -1).flatten(-2).to(x.dtype)


class State:
    """Per-sequence drafter context, fixed addresses (CUDA-graph safe): per layer a ring of post-norm post-RoPE
    context K and V [cap + 1, 8, 128] (slot pos % cap; slot cap = trash for masked writes) with the position held in
    each slot (pos, -1 = empty); e = the head's fused feature of the last context row; n = positions seen (host),
    nd = the same on the device. start > 0: context begins there (e.g. after a restored prefix without taps)."""

    def __init__(self, dev, dt, caps, start=0):
        self.caps, self.n, self.pend, self.io, self.graphs = caps, start, 0, None, {}
        self.dmap, self.dptr, self.captured = {}, 0, False   # shortlist: dynamic id -> slot (host), next slot
        self.k = [torch.zeros(c + 1, NKV, HD, dtype=dt, device=dev) for c in caps]
        self.v = [torch.zeros(c + 1, NKV, HD, dtype=dt, device=dev) for c in caps]
        self.pos = [torch.full((c + 1,), -1, dtype=torch.int64, device=dev) for c in caps]
        self.e = torch.zeros(H, device=dev)
        self.nd = torch.full((1,), start, dtype=torch.int64, device=dev)


class DSpark:
    def __init__(self, path=MTP, embed=None, unembed=None, embed_norm=None, cfg=Cfg(), device='cpu',
                 dtype=torch.bfloat16, graph=None, short=None, dyn=None, rank=None, ctxb=None, unembed8=None):
        from safetensors import safe_open
        self.cfg, self.dev, self.dt = cfg, torch.device(device), dtype
        self.short = SHORT if short is None else short       # static shortlist size, 0 = full vocab
        self.dyn = DYN if dyn is None else dyn
        self.rank, self.sid = rank, None                     # rank: ids by frequency (tests), sid: static ids
        self.ctxb = CTXB if ctxb is None else ctxb
        with safe_open(path, 'pt', device='cpu') as f:
            self.w = {k: f.get_tensor(k).to(self.dev, dtype) for k in f.keys()}
        self.wn = {k: v.float() for k, v in self.w.items() if v.dim() == 1 and 'norm' in k}  # fp32 norm weights
        self.embed, self.embed_norm = embed, embed_norm            # embed may stay on host (row lookups)
        self.U = unembed                                           # [VOCAB, H], on self.dev (None with unembed8)
        # unembed8 = (q e4m3 [V, H], s fp32 [V]): the target's FP8 unembed (Model.d8.un, INK_FP8_DECODE=1). Only the
        # shortlist head can use it (its rows are dequantized once / per new token), so it turns the shortlist on.
        self.U8 = unembed8
        self.V = (unembed8[0] if unembed8 is not None else unembed).shape[0]
        if unembed8 is not None and not self.short:
            self.short = SHORT or 32768
        self.U2 = self.cvec = None                                 # U @ hidden_out [V, 256] bf16, U @ b_out [V] (lazy)
        if graph is None:
            graph = os.environ.get('INK_GRAPH') == '1'
        self.graph = graph and self.dev.type == 'cuda'
        self.side = self.pool = None
        self.gstates = weakref.WeakSet()                            # States that captured graphs (pool users)

    def new_state(self, start=0, ctx=None):
        ctx = ctx or (CTX if self.dev.type == 'cuda' else 4096)
        return State(self.dev, self.dt, (WIN, ctx, WIN), start)

    def free(self, st):
        """Drop st's graphs (their pool memory goes back to the shared pool)."""
        st.graphs.clear()

    def reset(self, st, start=0):
        """Empty st for a new sequence, keeping its buffers and graphs (the engine keeps one State per slot)."""
        for p in st.pos:
            p.fill_(-1)
        st.e.zero_()
        st.nd.fill_(start)
        st.n, st.pend = start, 0
        st.dmap, st.dptr = {}, 0
        if hasattr(st, 'dslot'):                                   # (stale slots made the next note() KeyError)
            st.dslot = [-1] * len(st.dslot)
        if st.io is not None and self.short:
            st.io.md.fill_(float('-inf'))

    @torch.no_grad()
    def copy(self, src, dst):
        """dst := src's context (rings, head feature, positions, staged rows, shortlist dynamic part), keeping dst's
        buffers and graphs. n > 1: each sibling choice starts from the shared prompt's State instead of an empty one."""
        assert src.caps == dst.caps, (src.caps, dst.caps)
        self.reset(dst, src.n)
        for a, b in zip(dst.k + dst.v + dst.pos, src.k + src.v + src.pos):
            a.copy_(b)
        dst.e.copy_(src.e)
        dst.nd.copy_(src.nd)
        dst.pend = src.pend
        if src.io is not None:
            self._prep(dst)
            dst.io.taps.copy_(src.io.taps)
            if self.short:
                for n in ('Ud', 'U2d', 'cd', 'mfull', 'sid'):
                    getattr(dst.io, n).copy_(getattr(src.io, n))
                dst.dmap, dst.dslot, dst.dptr = dict(src.dmap), list(src.dslot), src.dptr

    # ------------------------------------------------------------------ context (sec. 3, 7)
    def fused(self, taps):
        """taps [4, n, H] (Cfg.taps order) -> per consumer (layer 0, 1, 2, head) fused feature [4, n, H] fp32,
        before that consumer's dflare_norm."""
        c = self.cfg
        t = taps.float()
        if c.tap_norm:
            t = rms(t)
        F = self.w['fuse_weight'].float()
        F = F if c.fuse == 'row' else F.t()
        if c.fuse_softmax:
            F = F.softmax(-1)
        x = torch.einsum('cj,jnh->cnh', F, t)
        return x if c.consumers == (0, 1, 2, 3) else x[list(c.consumers)]   # (no host-made index tensor: graphs)

    @torch.no_grad()
    def extend(self, st, taps):
        """Append context rows for the next n positions (st.n .. st.n+n-1): taps [4, n, H]. Called with the prompt's
        taps after prefill (any n, e.g. per prefill chunk), then after every verify with the anchor + accepted rows
        only (rejected rows never enter). Rows older than every ring are skipped."""
        n = taps.shape[1]
        for a in range(max(0, n - max(st.caps)), n, 4096):
            b = min(n, a + 4096)
            self._put(st, taps[:, a:b].to(self.dev), st.n + a, b - a)
        st.n += n
        st.nd.fill_(st.n)

    def _put(self, st, taps, p0, n):
        """Ring writes for taps [4, m, H] at positions p0 + i; rows i >= n (and rows a ring would overwrite within
        this call) go to the trash slot. p0, n: ints, or device [1] tensors (graph path: no host value used)."""
        m = taps.shape[1]
        x = self.fused(taps)
        i = torch.arange(m, device=self.dev)
        tpos = i + p0
        for L in range(3):
            P, cap = f'layers.{L}.', st.caps[L]
            ok = (i < n) & (i >= n - cap)
            slot = torch.where(ok, tpos % cap, cap)
            c = rms(x[L], self.wn[P + 'dflare_norm.weight']).to(self.dt)
            k = rms((c @ self.w[P + 'self_attn.k_proj_ctx.weight'].t()).view(m, NKV, HD),
                    self.wn[P + 'self_attn.k_norm_ctx.weight']).to(self.dt)
            v = (c @ self.w[P + 'self_attn.v_proj_ctx.weight'].t()).view(m, NKV, HD)
            st.k[L].index_copy_(0, slot, rope(k, tpos + self.cfg.ctx_pos, self.cfg.rope) if self.cfg.ctx_rope else k)
            st.v[L].index_copy_(0, slot, v)
            st.pos[L].index_copy_(0, slot, torch.where(ok, tpos, -1))
        j = (torch.as_tensor(n, device=self.dev) - 1).clamp(min=0).view(1)
        e = rms(x[3].index_select(0, j)[0], self.wn['markov_head.dflare_norm.weight'])
        st.e.copy_(torch.where(torch.as_tensor(n, device=self.dev) > 0, e, st.e))

    # ------------------------------------------------------------------ backbone (sec. 4)
    def anchor_embed(self, ids):
        e = self.embed[ids.cpu() if self.embed.device.type == 'cpu' else ids].to(self.dev).float()
        return rms(e, self.embed_norm) if self.cfg.anchor == 'normed' else e

    @torch.no_grad()
    def backbone(self, st, xe, p, nb=None):
        """xe [N, H] anchor rows (anchor_embed), p [N] anchor positions (device; anchor n sees context rows < p[n]:
        eval batches many anchors over one long context, the engine passes p = st.nd). nb: layer 1 reads only ring
        slots < nb (valid while every context position is < nb <= its cap). Returns (pre-norm X, normed H), each
        [N, 8, H] fp32."""
        c, w, N = self.cfg, self.w, xe.shape[0]
        X = torch.cat((xe[:, None], (w['mask_embed'].float() * c.mask_scale).expand(N, BLOCK - 1, H)), 1)
        pos = p[:, None] + torch.arange(BLOCK, device=self.dev)                    # [N, 8]
        for L in range(3):
            tpos, Kc, Vc = st.pos[L], st.k[L], st.v[L]
            if nb is not None and L == 1:
                tpos, Kc, Vc = tpos[:nb], Kc[:nb], Vc[:nb]
            ok = (tpos >= 0) & (tpos[None, None] < p[:, None, None])                # [N, 1, slots]
            P = f'layers.{L}.self_attn.'
            a = rms(X, self.wn[f'layers.{L}.input_layernorm.weight']).to(self.dt)
            q = rope(rms((a @ w[P + 'q_proj.weight'].t()).view(N, BLOCK, NQ, HD), self.wn[P + 'q_norm.weight']).to(self.dt),
                     pos, c.rope).view(N, BLOCK, NKV, 3, HD)
            kb = rope(rms((a @ w[P + 'k_proj.weight'].t()).view(N, BLOCK, NKV, HD), self.wn[P + 'k_norm.weight']).to(self.dt),
                      pos, c.rope)
            vb = (a @ w[P + 'v_proj.weight'].t()).view(N, BLOCK, NKV, HD)
            sc = torch.einsum('nikgd,tkd->nikgt', q, Kc).float()
            sb = torch.einsum('nikgd,njkd->nikgj', q, kb).float()
            m = ok
            if L in SLIDING:
                d = pos[:, :, None] - tpos[None, None]                             # [N, 8, n]
                m = m & ((d <= WIN) if c.win_incl else (d < WIN))
            sc = sc.masked_fill(~m[:, :, None, None], float('-inf'))
            pr = torch.cat((sc, sb), -1).mul_(HD ** -0.5).softmax(-1).to(self.dt)
            nc = Kc.shape[0]
            o = (torch.einsum('nikgt,tkd->nikgd', pr[..., :nc], Vc)
                 + torch.einsum('nikgj,njkd->nikgd', pr[..., nc:], vb))
            X = X + (o.reshape(N, BLOCK, NQ * HD) @ w[P + 'o_proj.weight'].t()).float()
            P = f'layers.{L}.mlp.'
            u = rms(X, self.wn[f'layers.{L}.post_attention_layernorm.weight']).to(self.dt)
            y = torch.nn.functional.silu(u @ w[P + 'gate_proj.weight'].t()) * (u @ w[P + 'up_proj.weight'].t())
            X = X + (y @ w[P + 'down_proj.weight'].t()).float()
        return X, rms(X, self.wn['norm.weight'])

    # ------------------------------------------------------------------ head (sec. 5) and logits
    def head_step(self, s, prev, h, xpre, e):
        """One rnn_hidden step, batched: s [N, 256], prev [N] ids, h / xpre [N, H] (normed / pre-norm row), e [H] or
        [N, H] head ctx feature. Returns (new s, bias [N, H]) fp32."""
        c, w = self.cfg, self.w
        mn = self.wn['markov_head.dflare_norm.weight']
        if c.head == 'a':
            x = e.expand_as(h)
        elif c.head == 'z':
            x = torch.zeros_like(h)
        elif c.head == 'b':
            x = rms(self.embed[prev.cpu() if self.embed.device.type == 'cpu' else prev].to(self.dev), mn)
        else:
            x = rms(xpre, mn)
        part = {'s': s, 'w': w['markov_head.markov_w1.weight'][prev].float(), 'h': h, 'e': x}
        z = torch.cat([part[k] for k in c.zorder], -1).to(self.dt)
        y = (z @ w['markov_head.rnn_proj.weight'].t() + w['markov_head.rnn_proj.bias']).float()
        y = dict(zip(c.gco, y.split(R, -1)))
        gt = torch.sigmoid(y['g'])
        s = gt * s + (1 - gt) * torch.tanh(y['c'])
        b = torch.tanh(y['o']).to(self.dt) @ w['markov_head.hidden_out.weight'].t() + w['markov_head.hidden_out.bias']
        return s, b.float()

    def logits(self, h):
        """h [..., H] -> [..., VOCAB] fp32."""
        x = (h.to(self.U.dtype) @ self.U.t()).float()
        return 30 * torch.tanh(x / 30) if self.cfg.softcap else x

    def final(self, Xk, Hk, b):
        """Row hidden fed to the unembed, with the head bias b (or None)."""
        if b is None:
            return Hk
        return Hk + b if self.cfg.bias == 'post' else rms(Xk + b, self.w['norm.weight'])

    def head_ctx(self, taps):
        """Head ctx feature of every row of taps [4, n, H] -> [n, H] (eval: one per anchor, at row p-1)."""
        return rms(self.fused(taps.to(self.dev))[3], self.w['markov_head.dflare_norm.weight'])

    @torch.no_grad()
    def rows_tf(self, st, x0, p, prev, e):
        """Eval: logits of all draft rows with the head fed teacher-forced previous tokens. x0 [N], p [N],
        prev [N, k] (prev[:, 0] = x0, prev[:, i] = the token drafted at row i-1), e [N, H] head ctx at p-1.
        Returns [N, k, VOCAB]."""
        X, Hn = self.backbone(st, self.anchor_embed(x0), p)
        if not self.cfg.row0:
            X, Hn = X[:, 1:], Hn[:, 1:]
        out, s = [], torch.zeros(x0.shape[0], R, device=self.dev)
        for i in range(prev.shape[1]):
            b = None
            if self.cfg.head != 'none':
                s, b = self.head_step(s, prev[:, i], Hn[:, i], X[:, i], e)
            out.append(self.logits(self.final(X[:, i], Hn[:, i], b)))
        return torch.stack(out, 1)

    @torch.no_grad()
    def draft(self, st, x0, k=7, temperature=0.0, top_p=1.0, top_k=0, gen=None):
        """k <= 7 (8 for row0=True) drafts after anchor x0 at position st.n. Greedy (temperature <= 0): argmax,
        q None. Else d_i ~ q_i = spec.probs(drafter logits) and q [k, VOCAB] is returned for spec.accept."""
        from spec import probs
        c = self.cfg
        assert 0 < k <= BLOCK - (0 if c.row0 else 1)
        X, Hn = self.backbone(st, self.anchor_embed(torch.tensor([x0])), st.nd)
        X, Hn = (X[0], Hn[0]) if c.row0 else (X[0, 1:], Hn[0, 1:])
        s, prev, ids, qs = torch.zeros(1, R, device=self.dev), torch.tensor([x0], device=self.dev), [], []
        for i in range(k):
            b = None
            if c.head != 'none':
                s, b = self.head_step(s, prev, Hn[i:i + 1], X[i:i + 1], st.e)
            lg = self.logits(self.final(X[i:i + 1], Hn[i:i + 1], b))
            if temperature <= 0:
                d = int(lg[0].argmax())
            else:
                qs.append(probs(lg, temperature, top_p, top_k)[0])
                d = int(torch.multinomial(qs[-1], 1, generator=gen))
            ids.append(d)
            prev = torch.tensor([d], device=self.dev)
        return ids, (torch.stack(qs) if qs else None)

    # ------------------------------------------------------------------ serving path (fixed shapes, CUDA graph)
    def _rows(self, ids):
        """Unembed rows [n, H] bf16: from the bf16 unembed, or dequantized from the FP8 one (q * s, rounded to bf16:
        the drafter's logits use these values, and q is computed from them, so sampling stays exact)."""
        if self.U8 is None:
            return self.U[ids]
        q, sc = self.U8
        return (q[ids].float() * sc[ids, None]).to(torch.bfloat16)

    def _prep(self, st):
        c, V, dt = self.cfg, self.V, self.dt
        assert (c.head, c.zorder, c.gco, c.bias, c.row0) == ('a', 'swhe', 'gco', 'post', False), 'serving path: Cfg defaults'
        wo, bo = (self.w[f'markov_head.hidden_out.{n}'] for n in ('weight', 'bias'))
        if self.short and self.sid is None:   # static shortlist rows of U, U' = U @ W_out, c = U @ b_out
            self.sid = static_short(self.short, self.rank, V).to(self.dev)
            self.Us = self._rows(self.sid).contiguous()
            self.U2s = torch.cat([(u.float() @ wo.float()).to(dt) for u in self.Us.split(16384)])
            self.cs = torch.cat([u.float() @ bo.float() for u in self.Us.split(16384)])
            self.ins = bytearray(V)                                          # host: id in the static list
            for t in self.sid.tolist():
                self.ins[t] = 1
        elif not self.short and self.U2 is None:   # fp32 math in row chunks: no 1.8 GB fp32 copy of U
            ch = self.U.split(16384)
            self.U2 = torch.cat([(u.float() @ wo.float()).to(dt) for u in ch])
            self.cvec = torch.cat([u.float() @ bo.float() for u in ch])
        if st.io is None:
            z = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device=self.dev)
            M = len(self.sid) + self.dyn if self.short else V
            st.io = SimpleNamespace(taps=z(4, BLOCK, H, dt=dt), n=z(1, dt=torch.int64), x0=z(1, dt=torch.int64),
                                    xe=z(H, dt=dt), temp=z(1) + 1,
                                    noise=z(BLOCK - 1, M, dt=torch.float64 if self.short else torch.float32) + 1,
                                    ids=z(BLOCK - 1, dt=torch.int64), q=z(BLOCK - 1, V), conf=z(BLOCK - 1))
            if self.short:   # dynamic shortlist part: rows of U / U' / c, id per slot, additive mask (-inf: empty)
                D = self.dyn
                io = st.io
                io.Ud, io.U2d, io.cd = z(D, H, dt=self.Us.dtype), z(D, R, dt=dt), z(D)
                io.mfull = torch.cat((z(len(self.sid)), z(D) - float('inf')))  # additive mask per column
                io.md = io.mfull[len(self.sid):]
                io.sid = torch.cat((self.sid, z(D, dt=torch.int64)))           # shortlist position -> token id
                st.dslot = [-1] * D

    def note(self, st, ids):
        """Shortlist: tokens of this sequence (prompt, then every emitted token) join the dynamic part unless they are
        static or already there; a ring over INK_DSPARK_DYN slots (the oldest goes). Host bookkeeping + a few small
        device copies when something new shows up. No-op without a shortlist."""
        if not self.short:
            return
        self._prep(st)
        D, new = self.dyn, {}
        for t in ids:
            t = int(t)
            if self.ins[t] or t in st.dmap:
                continue
            s = st.dptr % D
            if st.dslot[s] >= 0:
                del st.dmap[st.dslot[s]]
            st.dslot[s], st.dmap[t], st.dptr = t, s, st.dptr + 1
            new[s] = t
        if new:
            io, wo, bo = st.io, self.w['markov_head.hidden_out.weight'], self.w['markov_head.hidden_out.bias']
            sl = torch.tensor(list(new), device=self.dev)
            tt = torch.tensor(list(new.values()), device=self.dev)
            u = self._rows(tt)
            io.Ud.index_copy_(0, sl, u)
            io.U2d.index_copy_(0, sl, (u.float() @ wo.float()).to(self.dt))
            io.cd.index_copy_(0, sl, u.float() @ bo.float())
            io.md.index_fill_(0, sl, 0.0)
            io.sid.index_copy_(0, sl + len(self.sid), tt)

    def stage(self, st, taps):
        """Queue context rows taps [4, n, H] (n <= 8: the anchor + accepted rows of a verify; copied now, so the
        decoder's tap buffer may be overwritten afterwards) for the next round(). Rows already staged are flushed
        first (a round that drafted nothing)."""
        self._prep(st)
        n = taps.shape[1]
        assert 0 < n <= BLOCK, n
        if st.pend:
            self.flush(st)
        st.io.taps[:, :n].copy_(taps)
        st.pend = n

    def flush(self, st):
        """Write the staged rows into the context (no draft)."""
        if st.pend:
            st.io.n.fill_(st.pend)
            self._run(st, 0, True)
            st.n, st.pend = st.n + st.pend, 0

    def bucket(self, st):
        """Layer-1 ring slots the next round reads: all (None), or with INK_DSPARK_CTXB while every context position
        is below the ring size, the power-of-2 prefix (>= 1024) that holds them."""
        need, cap = st.n + st.pend, st.caps[1]
        if not self.ctxb or need > cap:
            return None
        nb = max(WIN, 1 << max(0, need - 1).bit_length())
        return nb if nb < cap else None

    @torch.no_grad()
    def round(self, st, x0, k, temperature=0.0, gen=None, conf=False, host=True, noise=None):
        """Extend by the staged rows, then draft k <= 7 tokens after anchor x0 at position st.n (after the extend):
        one fixed-shape device pass (a CUDA graph per (k, greedy, layer-1 bucket) when self.graph). Greedy: argmax
        drafts, q None. Sampled: d_i ~ q_i = softmax(logits / temperature) over the head's support (full vocab or the
        shortlist; no top-k/top-p: any q is exact as long as accept() gets the q the draft came from), q [k, V] is a
        view of a static buffer (valid until the next round). conf: also return the rows' confidences (list, T = 1),
        read with the ids in one host transfer. host=False: the ids as a device tensor [k] (a view of a static buffer,
        no sync; valid until the next round). noise: sampled, draft i's Exp(1) noise in vocab order [V] (the target's
        noise of that position, spec_runner: coupled races) instead of a fresh draw from gen."""
        self._prep(st)
        assert 0 < k < BLOCK, k
        io, greedy = st.io, temperature <= 0
        io.n.fill_(st.pend)
        io.x0.fill_(int(x0))
        io.xe.copy_(self.embed[int(x0)])
        if not greedy:
            io.temp.fill_(float(temperature))
            if noise is None:
                io.noise.exponential_(generator=gen).clamp_(min=torch.finfo(io.noise.dtype).tiny)
            else:
                for i, e in enumerate(noise):
                    io.noise[i].copy_(e[io.sid] if self.short else e[:io.noise.shape[1]])
        self._run(st, k, greedy, self.bucket(st))
        st.n, st.pend = st.n + st.pend, 0
        if not host:
            return io.ids[:k], (None if greedy else io.q[:k])
        if conf:
            x = torch.cat((io.ids[:k].double(), io.conf[:k].double())).tolist()
            return [int(t) for t in x[:k]], (None if greedy else io.q[:k]), x[k:]
        return io.ids[:k].tolist(), (None if greedy else io.q[:k])

    def _run(self, st, k, greedy, nb=None):
        st.captured = False
        if not self.graph:
            return self._body(st, k, greedy, nb)
        g = st.graphs.get((k, greedy, nb))
        if g is not None:
            return g.replay()
        # first use: run it eagerly on a side stream (this IS the work), then record the same work (recording runs
        # nothing); one graph pool per drafter (rounds never overlap)
        st.captured = True
        cur = torch.cuda.current_stream()
        if self.side is None:
            self.side = torch.cuda.Stream()
        # one pool shared by all States' graphs (rounds never overlap), but only while some live graph holds it: once
        # every graph of the pool is gone (idle States dropped by free_vram / drop_drafter) the allocator releases the
        # pool, and capturing into the stale id fails (window g 11-23: INTERNAL ASSERT use_count > 0 -> 500s)
        if self.pool is None or not any(x.graphs for x in self.gstates):
            self.pool = torch.cuda.graph_pool_handle()
        self.side.wait_stream(cur)
        with torch.cuda.stream(self.side):
            self._body(st, k, greedy, nb)
        cur.wait_stream(self.side)
        cur.synchronize()
        g = torch.cuda.CUDAGraph()
        # thread_local: see decode_batch._replay (a writer-thread CUDA call must not invalidate this capture)
        with torch.cuda.graph(g, pool=self.pool, stream=self.side, capture_error_mode='thread_local'):
            self._body(st, k, greedy, nb)
        st.graphs[(k, greedy, nb)] = g
        self.gstates.add(st)

    def _body(self, st, k, greedy, nb=None):
        """Device-only round: no host values of st/io are read (every varying input is in st.io / st.nd)."""
        io, w, dt, short = st.io, self.w, self.dt, self.short
        self._put(st, io.taps, st.nd, io.n)
        st.nd.add_(io.n)
        if not k:
            return
        xe = io.xe.float()[None]
        if self.cfg.anchor == 'normed':
            if not hasattr(self, 'enf'):
                self.enf = self.embed_norm.float().to(self.dev)
            xe = rms(xe, self.enf)
        X, Hn = self.backbone(st, xe, st.nd, nb)
        Hn = Hn[0, 1:]                                                       # 7 rows (row 0 = anchor, dropped)
        Hb = Hn.to(self.Us.dtype if short else self.U.dtype)
        if short:   # head support = static shortlist ++ dynamic slots (io.sid maps a column to its token id)
            base = torch.cat(((Hb @ self.Us.t()).float() + self.cs, (Hb @ io.Ud.t()).float() + io.cd), -1)
        else:       # the one full unembed read, all rows
            base = (Hb @ self.U.t()).float() + self.cvec
        Wr = w['markov_head.rnn_proj.weight']                                # z = [s | W1[prev] | h | e]
        yhe = ((torch.cat((Hn, st.e.expand(BLOCK - 1, H)), -1).to(dt) @ Wr[:, 2 * R:].t()).float()
               + w['markov_head.rnn_proj.bias'].float())
        s, prev = torch.zeros(1, R, device=self.dev), io.x0
        if not greedy:
            io.q[:k].zero_()
        for i in range(k):
            z = torch.cat((s, w['markov_head.markov_w1.weight'][prev].float()), -1).to(dt)
            g, c, o = (yhe[i:i + 1] + (z @ Wr[:, :2 * R].t()).float()).split(R, -1)
            gt = torch.sigmoid(g)
            s = gt * s + (1 - gt) * torch.tanh(c)
            t = torch.tanh(o).to(dt)
            if short:
                lg = base[i:i + 1] + torch.cat(((t @ self.U2s.t()).float(), (t @ io.U2d.t()).float()), -1)
            else:
                lg = base[i:i + 1] + (t @ self.U2.t()).float()
            if self.cfg.softcap:
                lg = 30 * torch.tanh(lg / 30)
            if short:
                lg = lg + io.mfull                                            # empty dynamic slots: -inf (after softcap)
            mx = lg.max(-1, keepdim=True).values
            io.conf[i:i + 1].copy_((mx - torch.logsumexp(lg, -1, keepdim=True)).exp()[:, 0])
            if greedy:
                d = lg.argmax(-1)
            else:
                q = torch.softmax(lg / io.temp, -1)
                if short:
                    io.q[i].scatter_add_(0, io.sid, q[0])                       # q on the vocab (0 off the shortlist)
                    d = race(q, io.noise[i:i + 1])
                else:
                    io.q[i:i + 1].copy_(q)
                    d = (q / io.noise[i:i + 1]).argmax(-1)                   # exponential race: an exact draw from q
            if short:
                d = io.sid[d]
            io.ids[i:i + 1].copy_(d)
            prev = d


def tv_accept(p, q):
    """Per-row acceptance probability of speculative sampling, sum_v min(p, q) = 1 - TV."""
    return torch.minimum(p, q).sum(-1)

