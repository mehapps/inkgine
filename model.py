"""Inkling 2 Mini text model, written from engine/SPEC.md (section numbers refer to it).

One sequence per forward. KV for the 8 full-attention layer visits lives in a paged 4-bit pool (kvpool.py, KV4's
format) so 1M tokens fit next to the weights; each Seq reads and writes it through its page table (seq.kv); every visit also keeps the last 1023 keys in bf16 (Ring), so all
attention within distance < 1024 (where the relative bias lives) is exact bf16.
"""
import json
import math
import mmap
import os
import struct
import warnings

import torch
import torch.nn.functional as F
from safetensors import safe_open

# Exact GEMM reductions on cuda (set once at import, before any matmul): PyTorch's default lets cuBLAS reduce
# split-K partial sums in bf16/fp16; the reference math assumes fp32 accumulation. Never TF32 for fp32 matmuls.
# Cost: only cuBLAS's split-K choices for some (mostly skinny/decode) bf16 GEMM shapes, expected a few % at most.
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.set_float32_matmul_precision('highest')

H, NQ, NKV, HD, WIN, VOCAB = 2304, 24, 8, 128, 1024, 200058
FULL = (5, 11, 17, 23)
CORE = range(6, 18)
CLIP_QK, CLIP_REL = 22.627416997969522, 8.0
E2M1 = (0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6)
# visit order (§6): layers 0..5, loop core 3x, layers 18..23 -> 48 visits
VISITS = [L for L in range(6)] + [L for _ in range(3) for L in CORE] + [L for L in range(18, 24)]
FULL_VISITS = [i for i, L in enumerate(VISITS) if L in FULL]  # 8 visits with long KV


def rms(x, w=None, eps=1e-6):
    """RMSNorm in fp32 (§2). Caller casts."""
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return y if w is None else y * w.float()


def l2clip(x, c):
    x = x.float()
    return x * torch.clamp(c / (x.norm(dim=-1, keepdim=True) + 1e-6), max=1.0)


def attn_res(q, s):
    """§4. q [nq,H] fp32; s: list of rms(entry) fp32 [T,H]. Callers cache rms(entry): block entries only change every
    6 layers. Returns [nq,T,H] fp32."""
    s = torch.stack(s)                                              # [n,T,H]
    w = torch.softmax(torch.einsum('qh,nth->qtn', q, s) / math.sqrt(H), -1)
    return torch.einsum('qtn,nth->qth', w, s)

# INK_GLUE=triton|compile (default eager): the fp32 elementwise helpers (rms, l2clip, sconv, attn_res: ~450 GB of
# DRAM traffic per 4k prefill chunk as separate torch ops) run fused for big inputs on cuda: `triton` = prefill_glue.py
# kernels (sconv bit-identical; rms/attn_res fp32-ulp; l2clip stays eager), `compile` = torch.compile (inductor) of
# each helper. Small calls (decode, T < 64) keep the eager functions. GLUE[0] switches it at run time (tests A/B).
GLUE = [os.environ.get('INK_GLUE', 'eager')]
GLUE_CPU = [False]  # tests: allow the triton glue on cpu (Triton interpreter)


def _glue(f, big):
    cf = None

    def w(*a, **k):
        nonlocal cf
        mode = GLUE[0]
        x = a[0] if isinstance(a[0], torch.Tensor) else a[1][0]
        if mode != 'eager' and (x.is_cuda or GLUE_CPU[0] and mode == 'triton') and big(a):
            if mode == 'triton':
                import prefill_glue
                fn = getattr(prefill_glue, f.__name__, None)
                if fn is not None:
                    return fn(*a, **k)
            elif mode == 'compile':
                if cf is None:
                    torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 32)  # attn_res: 1-7 entries
                    cf = torch.compile(f, dynamic=True)
                return cf(*a, **k)
        return f(*a, **k)
    w.eager = f
    return w


rms = _glue(rms, lambda a: a[0].shape[0] >= 64)
l2clip = _glue(l2clip, lambda a: a[0].shape[0] >= 64)
attn_res = _glue(attn_res, lambda a: a[1][0].shape[0] >= 64)


class Weights:
    """Text weights. On cuda everything is resident (experts stay packed); on cpu the expert tensors
    are lazy safetensors slices so a forward reads only the experts it routes to."""
    LAZY = ('w13_weight', 'w2_weight', 'w13_weight_scale', 'w2_weight_scale')
    _ST_DTYPE = {'U8': torch.uint8, 'F8_E4M3': torch.float8_e4m3fn, 'BF16': torch.bfloat16,
                 'F16': torch.float16, 'F32': torch.float32}

    def __init__(self, path, device):
        self.dev = torch.device(device)
        self.f_path = os.path.join(path, 'model-00001-of-00001.safetensors')
        f = safe_open(self.f_path, 'pt', device='cpu')
        self.f, self.t = f, {}
        self.offloaded, self._stage, self._stage_ev, self._stage_slot, self._mm = set(), {}, [None, None], 0, None
        for k in f.keys():
            if not k.startswith('model.llm.'):
                continue
            n = k[len('model.llm.'):]
            if n == 'embed.weight':  # 0.9 GB, only row lookups: keep on host
                self.embed = f.get_tensor(k)
            elif self.dev.type == 'cpu' and n.endswith(self.LAZY):
                self.t[n] = f.get_slice(k)
            else:
                self.t[n] = f.get_tensor(k).to(self.dev)
        self.e2m1 = torch.tensor(E2M1, device=self.dev)
        self.fuse()

    def fuse(self):
        """Per layer, concatenate the attention input projections into one [8600|8664, H] matrix
        (q|r|sink|gate, k[|k taps], v[|v taps]); the named weights become row views of it (no extra memory, same
        values), so the prefill path is unchanged and decode runs one GEMM per distinct input. Loop gates get one
        fp32 [forget|gain] [4608, 64] each (cast once here instead of per call; 57 MB in total)."""
        # bf16 weights the math only ever uses via .float(): keep them fp32 (exact upcast, same results, no per-call
        # cast kernel; the router is 1.2 MB per layer per call). +~20 MB.
        for n, x in self.t.items():
            if n.endswith(F32_ONLY) and isinstance(x, torch.Tensor) and x.dtype == torch.bfloat16:
                self.t[n] = x.float()
        self.proj, self.gcat, self.pnames = {}, {}, {}
        for L in range(24):
            A = f'layers.{L}.attn.'
            tap = ['k_sconv.weight_proj'] if L in FULL else []
            groups = [QPROJ, ['wk_dv.weight'] + tap, ['wv_dv.weight'] + [n.replace('k_', 'v_') for n in tap]]
            self.pnames[L] = [A + n for g in groups for n in g]
            self.proj[L] = (self._cat(self.pnames[L]), [sum(self.t[A + n].shape[0] for n in g) for g in groups])
            if L in CORE:
                for c in ('attn', 'k', 'v', 'mlp'):
                    p = f'layers.{L}.controllers.{c}.'
                    self.gcat[p] = torch.cat((self.t[p + 'forget_weight'], self.t[p + 'gain_weight'])).float()

    def drop_dense(self):
        """INK_FP8_DECODE: free the bf16 attention projections + wo (~1.29 GB); every reader then uses Model.d8."""
        for L in range(24):
            for n in self.pnames[L] + [f'layers.{L}.attn.wo_ud.weight']:
                del self.t[n]
            self.proj[L] = (None, self.proj[L][1])

    def _cat(self, names):
        cat, r = torch.cat([self.t[n] for n in names]), 0
        for n in names:
            k = self.t[n].shape[0]
            self.t[n], r = cat[r:r + k], r + k
        return cat

    def __getitem__(self, n):
        return self.t[n]

    def expert(self, L, e):
        """Dequantized (w13 [2304 interleaved gate/up, 2304], w2 [2304,1152]) bf16, §7."""
        p = f'layers.{L}.mlp.experts.'
        out = []
        for w in ('w13', 'w2'):
            q = self.t[p + w + '_weight'][e].to(self.dev)
            s = self.t[p + w + '_weight_scale'][e].to(self.dev)
            if L < 20:  # NVFP4: low nibble = even element
                codes = torch.stack((q & 15, q >> 4), -1).view(q.shape[0], -1).long()
                gs = self.t[p + w + '_weight_amax'][e] / self.t[p + w + '_weight_outer_scale_denominator'][0]
                out.append((self.e2m1[codes] * s.float().repeat_interleave(16, -1) * gs).bfloat16())
            else:  # MXFP8, E8M0 scale per 32
                out.append((q.float() * torch.exp2(s.float() - 127).repeat_interleave(32, -1)).bfloat16())
        return out

    def _mmap_view(self, n):
        """Zero-copy read-only CPU tensor view into the safetensors file (OS-evictable page cache)."""
        mm_info = getattr(self, '_mm', None)
        if mm_info is None:
            fh = open(self.f_path, 'rb')
            hl = struct.unpack('<Q', fh.read(8))[0]
            hdr = json.loads(fh.read(hl))
            mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
            mm_info = self._mm = (fh, mm, hdr, 8 + hl)
        _, mm, hdr, base = mm_info
        meta = hdr['model.llm.' + n]
        a, b = meta['data_offsets']
        dt = self._ST_DTYPE[meta['dtype']]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            return torch.frombuffer(mm, dtype=torch.uint8, count=b - a, offset=base + a).view(dt).view(meta['shape'])

    def _offload_layer(self, L):
        fp = getattr(self, 'f_path', None)
        f = getattr(self, 'f', None)
        for suf in self.LAZY:
            n = f'layers.{L}.mlp.experts.{suf}'
            if n in self.t:
                if fp:
                    self.t[n] = self._mmap_view(n)
                elif f is not None:
                    self.t[n] = f.get_slice('model.llm.' + n)
                else:
                    self.t[n] = self.t[n].cpu()
        self.offloaded.add(L)

    def _load_expert_tensor(self, n):
        t = self.t[n]
        if isinstance(t, torch.Tensor):
            return t.to(self.dev) if t.device != self.dev else t.clone()
        return self.f.get_tensor('model.llm.' + n).to(self.dev)

    def _reload_layers(self, layers):
        loaded = []
        try:
            for L in layers:
                if L not in self.offloaded:
                    continue
                got = {}
                for suf in self.LAZY:
                    n = f'layers.{L}.mlp.experts.{suf}'
                    if n in self.t:
                        got[n] = self._load_expert_tensor(n)
                self.t.update(got)
                self.offloaded.discard(L)
                loaded.append(L)
        except torch.OutOfMemoryError:
            got = None
            for L in loaded:
                self._offload_layer(L)
            if self.dev.type == 'cuda':
                torch.cuda.empty_cache()
            raise

    def offload_experts(self, keep_layers=()):
        """Light mode: replace routed expert tensors for layers not in `keep_layers` with zero-copy read-only mmap
        views (or slices / CPU tensors in tests without a file path); reload any offloaded layers now in `keep_layers`."""
        if not hasattr(self, 'offloaded'):
            self.offloaded = set()
        keep = set(keep_layers)
        for L in range(24):
            if L not in keep and L not in self.offloaded and f'layers.{L}.mlp.experts.w13_weight' in self.t:
                self._offload_layer(L)
        to_reload = [L for L in sorted(self.offloaded) if L in keep]
        if to_reload:
            self._reload_layers(to_reload)
        if not self.offloaded:
            self.clear_stage()

    def reload_experts(self):
        """Reload all offloaded routed expert layers back onto self.dev and drop staging buffers. Rolls back on OOM."""
        self.clear_stage()
        if self.dev.type == 'cuda':
            torch.cuda.empty_cache()
        off = getattr(self, 'offloaded', None)
        if off:
            self._reload_layers(sorted(off))

    def clear_stage(self):
        """Free reusable host/device staging bounce buffers used by streamed offloaded MoE."""
        evs = getattr(self, '_stage_ev', None)
        if isinstance(evs, list):
            for i, ev in enumerate(evs):
                if ev is not None:
                    ev.synchronize()
                    evs[i] = None
        elif evs is not None:
            evs.synchronize()
            self._stage_ev = [None, None]
        self._stage_slot = 0
        stage = getattr(self, '_stage', None)
        if stage:
            stage.clear()

    def stage_experts(self, L, sub_e, cap=None):
        """Fetch packed expert tensors (w13_weight, w13_weight_scale, w2_weight, w2_weight_scale) for expert ids
        `sub_e` onto self.dev. On CUDA, gathers directly from mmap views into double-buffered pinned host bounce
        buffers with per-slot events to overlap CPU gather with GPU execution."""
        P = f'layers.{L}.mlp.experts.'
        names = ('w13_weight', 'w13_weight_scale', 'w2_weight', 'w2_weight_scale')
        srcs = [self.t[P + n] for n in names]
        idx_cpu = torch.tensor(sub_e, dtype=torch.long)
        if self.dev.type != 'cuda':
            return [torch.index_select(s, 0, idx_cpu) if isinstance(s, torch.Tensor) else s[sub_e] for s in srcs]
        k = len(sub_e)
        S = max(k, cap or k)
        stage = getattr(self, '_stage', None)
        if stage is None:
            stage = self._stage = {}
        evs = getattr(self, '_stage_ev', None)
        if not isinstance(evs, list):
            evs = self._stage_ev = [None, None]
        slot = getattr(self, '_stage_slot', 0)
        self._stage_slot = 1 - slot
        ev = evs[slot]
        if ev is not None:
            ev.synchronize()  # previous H2D copy on this bounce slot finished reading hb[slot]
        out = []
        for n, src in zip(names, srcs):
            if isinstance(src, torch.Tensor):
                shape, dt, tmp = tuple(src.shape[1:]), src.dtype, None
            else:
                tmp = src[sub_e]
                shape, dt = tuple(tmp.shape[1:]), tmp.dtype
            key = (n, shape, dt, S)
            buf = stage.get(key)
            if buf is None:
                hbs = (torch.empty((S,) + shape, dtype=dt, pin_memory=True),
                       torch.empty((S,) + shape, dtype=dt, pin_memory=True))
                db = torch.empty((S,) + shape, dtype=dt, device=self.dev)
                buf = stage[key] = (hbs, db)
            hb, db = buf[0][slot], buf[1]
            if tmp is None:
                torch.index_select(src, 0, idx_cpu, out=hb[:k])
            else:
                hb[:k].copy_(tmp)
            db[:k].copy_(hb[:k], non_blocking=True)
            out.append(db[:k])
        if ev is None:
            ev = evs[slot] = torch.cuda.Event()
        ev.record()
        return out


F32_ONLY = ('norm.weight', 'l2_clip.scale', 'sconv.weight', 'static_w', 'mlp.gate.weight')
QPROJ = ('wq_du.weight', 'wr_du.weight', 'attn_sink_proj.weight', 'gate_attn.weight')  # rows 3072|384|24|3072
QSPLIT = (3072, 384, 24, 3072)


class KV4:
    """4-bit KV for the full-attention visits: int4 codes per element, one e4m3 scale per 16. The engine uses the paged
    kvpool (same format, `KV4.quant`); this flat store stays as the reference / test store with the same interface."""

    def __init__(self, n, cap, dev):
        self.cap = cap
        self.c = torch.zeros(2, n, cap, NKV, HD // 2, dtype=torch.uint8, device=dev)
        self.s = torch.zeros(2, n, cap, NKV, HD // 16, dtype=torch.float8_e4m3fn, device=dev)

    @staticmethod
    def quant(x):
        """[n,8,128] -> (codes [n,8,64] uint8, low nibble = even element; scales [n,8,8] e4m3)."""
        n = x.shape[0]
        g = x.float().view(n, NKV, HD // 16, 16)
        s = (g.abs().amax(-1) / 7).to(torch.float8_e4m3fn)
        sf = s.float().unsqueeze(-1)
        q = (torch.where(sf > 0, g / sf, 0).round().clamp(-8, 7) + 8).to(torch.uint8).view(n, NKV, HD)
        return q[..., 0::2] | (q[..., 1::2] << 4), s

    def put(self, slot, p0, K, V):
        n = K.shape[0]
        assert p0 + n <= self.cap, f'context full ({self.cap} tokens)'
        for j, x in enumerate((K, V)):
            self.c[j, slot, p0:p0 + n], self.s[j, slot, p0:p0 + n] = self.quant(x)

    def put_at(self, slot, pos, K, V):
        """Write one token at a device position (pos: int64 [1]); no host sync, CUDA-graph safe. The caller checks
        capacity (engine: prompt + max_tokens <= max_ctx)."""
        for j, x in enumerate((K, V)):
            q, s = self.quant(x)
            self.c[j, slot].index_copy_(0, pos, q)
            self.s[j, slot].view(torch.uint8).index_copy_(0, pos, s.view(torch.uint8))  # no fp8 index_copy kernel

    def get(self, slot, a, b):
        out = []
        for j in range(2):
            c = self.c[j, slot, a:b]
            # bf16 math is exact here: int in [-8,7] times an e4m3 scale needs <= 7 mantissa bits
            q = torch.stack((c & 15, c >> 4), -1).view(b - a, NKV, HD // 16, 16).to(torch.bfloat16).sub_(8)
            out.append(q.mul_(self.s[j, slot, a:b].to(torch.bfloat16).unsqueeze(-1)).view(b - a, NKV, HD))  # in place: one bf16 temp
        return out


class Seq:
    """Per-sequence state: position, conv states (§3.3), bf16 rings, and its KV store (a kvpool.Table, or a KV4 in tests)."""
    moe_impl = None  # None = INK_MOE

    def __init__(self, dev, kv):
        self.pos, self.kv = 0, kv
        z = lambda c: torch.zeros(3, c, dtype=torch.bfloat16, device=dev)
        self.conv = [{'k': z(1024), 'v': z(1024), 'attn': z(H), 'mlp': z(H)} for _ in VISITS]
        e = torch.zeros(0, NKV, HD, dtype=torch.bfloat16, device=dev)
        self.ring = [(e, e) for _ in VISITS]
        self.dev = torch.device(dev)

    def positions(self, T):
        return torch.arange(self.pos, self.pos + T, device=self.dev)

    def advance(self, T):
        self.pos += T


def sconv(x, st, w=None, dyn=None):
    """§3.3 causal kernel-4 conv with residual. Static: w [C,1,4]; dynamic: dyn=(static_w [4,C],
    taps [T,8,4] from the input projection). st [3,C] bf16 = the last 3 inputs, updated in place."""
    T, C = x.shape
    xx = torch.cat((st, x)).float()
    st.copy_(xx[-3:])
    win = xx.unfold(0, 4, 1)                                        # [T,C,4], tap 3 = current token
    if dyn is None:
        return ((win * w[:, 0, :].float()).sum(-1) + x.float()).div(math.sqrt(5)).bfloat16()
    sw, taps = dyn                                                   # lag j = 3 - tap index
    W = sw.float().flip(0).t() + taps.float().repeat_interleave(C // taps.shape[1], 1).flip(-1)
    return ((win * W).sum(-1) + x.float()).div(3.0).bfloat16()


def gate_in(x, msg):
    """Loop gate input: bf16 [rms(x) | rms(msg)] (x, msg fp32 [T,H])."""
    return torch.cat((rms(x), rms(msg)), -1).bfloat16()


def gate_out(zz, x, msg):
    """Loop gate mix: f, s = tanh(zz).chunk(2) (fp32 [T,2H]); (1 - f) * x + s * msg."""
    f, s = torch.tanh(zz).chunk(2, -1)
    return (1 - f) * x + s * msg


def gate_norm(zz, x, msg, nw):
    """gate_out + the RMSNorm that always follows it, bf16 out."""
    return rms(gate_out(zz, x, msg), nw).bfloat16()


sconv = _glue(sconv, lambda a: a[0].shape[0] >= 64)


# INK_GLUE2=1 (off by default): more of the prefill glue fused (prefill_glue.py; Model.layer/attention take these paths
# for Seq calls with >= 64 rows; on cuda they always run as the kernels, on cpu eager unless GLUE_CPU): the attention
# prologue (q/r scale + l2clip + tau; k/v conv + l2clip / value norm + the 4-bit KV write), the output gate inside the
# near kernel, the residual adds with the RMSNorm after them, attn_res with the RMSNorm after it. The eager functions
# below are the old math composed from the same helpers (tests/test_glue.py compares the kernels against them).
GLUE2 = [os.environ.get('INK_GLUE2', '0') == '1']
GLUE2_FNS = ('qr_prep', 'kv_prep', 'res_in', 'res_out', 'attn_res_norm')


def _glue2(f):
    def w(*a, **k):
        if a[0].is_cuda or GLUE_CPU[0]:
            import prefill_glue
            return getattr(prefill_glue, f.__name__)(*a, **k)
        return f(*a, **k)
    w.eager, w.__name__ = f, f.__name__
    return w


def qr_prep(q, r, qs, rs, tau=None):
    """q [T,24*128], r [T,24*16] (row-strided views ok), scales fp32 -> l2-clipped bf16 Q [T,24,128], R [T,24,16];
    full layers: times tau [T] fp32, rounded again."""
    T = q.shape[0]
    Q = l2clip(q.reshape(T, NQ, HD).float() * qs, CLIP_QK).bfloat16()
    R = l2clip(r.reshape(T, NQ, 16).float() * rs, CLIP_REL).bfloat16()
    if tau is not None:
        Q, R = (Q.float() * tau[:, None, None]).bfloat16(), (R.float() * tau[:, None, None]).bfloat16()
    return Q, R


def kv_prep(K, V, stk, stv, kw, vw, tk=None, tv=None, kv=None, slot=0, p0=0):
    """K/V [T,1024] bf16 -> conv (static kw/vw [C,1,4], or dynamic: static_w [4,C] + taps tk/tv [T,32]) ->
    l2clip(K) / rms(V) bf16 [T,8,128]; with kv: written to the 4-bit store at positions p0.. of slot."""
    T = K.shape[0]
    if tk is not None:
        K = sconv(K, stk, dyn=(kw, tk.reshape(T, NKV, 4)))
        V = sconv(V, stv, dyn=(vw, tv.reshape(T, NKV, 4)))
    else:
        K, V = sconv(K, stk, kw), sconv(V, stv, vw)
    K = l2clip(K.view(T, NKV, HD), CLIP_QK).bfloat16()
    V = rms(V.view(T, NKV, HD)).bfloat16()
    if kv is not None:
        kv.put(slot, p0, K, V)
    return K, V


def res_in(o, st, w, h=None):
    """Post-attention conv + residual: r = h + sconv(o) fp32 (h None: block boundary, r = sconv(o)); -> (r, rms(r))."""
    y = sconv(o, st, w)
    r = y.float() if h is None else h.float() + y.float()
    return r, rms(r)


def res_out(y, st, w, r):
    """Post-MLP conv + residual: h = bf16(r + sconv(y)); -> (h, rms(h)) (rms(h) = the next layer's hn)."""
    h = (r + sconv(y, st, w).float()).bfloat16()
    return h, rms(h)


def attn_res_norm(q, s, w):
    """rms(attn_res(q, s), w).bfloat16(), w [nq,H]: one RMSNorm weight per query row -> bf16 [nq,T,H]."""
    return rms(attn_res(q, s), w[:, None]).bfloat16()


def gate_down(x, msg, w):
    """Loop gate input + down projection: bf16(gate_in(x, msg) @ w.t()), w = down_weight [64, 2H] -> z [T,64] bf16."""
    return gate_in(x, msg) @ w.t()


def gate_up_norm(z, x, msg, g, nw):
    """Loop gate up projection (fp32, g = gcat [2H,64]) + tanh/blend + RMSNorm: gate_norm(fp32(z) @ g.t(), ...)."""
    return gate_norm(z.float() @ g.t(), x, msg, nw)


def finish_ref(y, tok, ws, T):
    """moe_triton.finish in torch: per token, its routed rows in expert order, fp32 sum of y * ws from 0, bf16."""
    idx = torch.sort(tok, stable=True).indices
    yy, ww = y[idx].view(T, -1, y.shape[1]).float(), ws[idx].view(T, -1, 1)
    acc = torch.zeros(T, y.shape[1], dtype=torch.float32, device=y.device)
    for k in range(yy.shape[1]):
        acc = acc + yy[:, k] * ww[:, k]
    return acc.bfloat16()


def moe_tail(y, tok, ws, st, w, r):
    """MoE expert sum + post-MLP conv + residual + norm: res_out(finish(y), st, w, r) (y [R,H] bf16 per routed row)."""
    return res_out(finish_ref(y, tok, ws, r.shape[0]), st, w, r)


# INK_GATEFUSE=1 (off): loop gate as 2 kernels (gate_down: norms + down GEMM; gate_up_norm: fp32 up GEMM + tanh/blend
# + RMSNorm; no [T,2H] gate input / [T,2H] fp32 pre-tanh in memory). INK_MOETAIL=1 (off; with INK_GLUE2 and the v2 /
# fp4 grouped MoE): the 8-way expert sum + post-MLP conv + residual + norm in one kernel (moe_tail).
# GPU 2026-11-23 (window_prefill2): both within KL noise. GATEFUSE is a regression, parked: gate_up_norm 102 ms per chunk
# vs ~20 ms for the whole unfused loop gate (fp32 tl.dot on 16-row tiles, 256 programs, 2 passes; cuBLAS SGEMM does
# the up GEMM in ~3 ms). MOETAIL is neutral (the [T,H] round trip it removes is ~1 ms) and OOMs at 8k chunks at 1M.
GATEFUSE = [os.environ.get('INK_GATEFUSE', '0') == '1']
# INK_MOE_ROUTE=1 (off; exact): prefill MoE calls (>= 64 tokens, triton v2 / fp4 path) sort routed rows and build the
# block tables with moe_route.py (3 launches) instead of argsort + moe_triton.tables + finish's sort (~40 launches).
ROUTE = [os.environ.get('INK_MOE_ROUTE', '0') == '1']
MOETAIL = [os.environ.get('INK_MOETAIL', '0') == '1']
qr_prep, kv_prep, res_in, res_out, attn_res_norm, gate_down, gate_up_norm, moe_tail = map(
    _glue2, (qr_prep, kv_prep, res_in, res_out, attn_res_norm, gate_down, gate_up_norm, moe_tail))
gate_norm = _glue(gate_norm, lambda a: a[0].shape[0] >= 64)
gate_in = _glue(gate_in, lambda a: a[0].shape[0] >= 64)
gate_out = _glue(gate_out, lambda a: a[0].shape[0] >= 64)


try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    _flex = torch.compile(flex_attention, dynamic=False)
except Exception:  # pragma: no cover
    _flex = None
FAR_BLK = int(os.environ.get('INK_FAR_BLK', 65536))
_FAR = {}  # per device: tensor [offset, n_valid keys, n_valid queries], updated in place so the mask never recompiles
FLEX_TQ = 4096  # flex query bucket: every prefill chunk <= 4096 rows is one compiled shape (rows >= T masked, skipped)


def _far_params(dev):
    if dev not in _FAR:
        _FAR[dev] = torch.zeros(3, dtype=torch.int64, device=dev)
    return _FAR[dev]


def _far_mask(b, h, qi, ki):
    prm = _FAR[qi.device] if qi.device in _FAR else next(iter(_FAR.values()))
    return (ki <= qi + prm[0]) & (ki < prm[1]) & (qi < prm[2])
FAR_IMPL = os.environ.get('INK_FAR', 'flex')  # flex (cuda) | manual
# torch | triton (prefill_attn far + near) | far (far only) | near (triton near + flex far) | auto (triton near; far:
# triton while the far range is < INK_PA_FAR_MAX keys, flex above; GPU 2026-11-22: triton far 2.5 vs flex 4.1 ms per
# full visit at 4k context, a tie at 64k, slower at 256k/1M (90 vs 81, 358 vs 315 ms)). server.py sets auto on cuda.
PREFILL_ATTN = os.environ.get('INK_PREFILL_ATTN', 'torch')
PA_FAR_MAX = int(os.environ.get('INK_PA_FAR_MAX', 32768))
FARSKIP = os.environ.get('INK_FARSKIP', '') != ''  # far tile skipping (prefill_attn.SKIP): auto uses the Triton far kernel
PA_FP8 = os.environ.get('INK_PA_FP8', '0') == '1'  # fp8 Triton far kernel at every range (auto: no flex above PA_FAR_MAX)
FP8_DENSE = os.environ.get('INK_FP8_DENSE', '0') == '1'  # fp8 attention projections + wo for prefill (fp8_dense.py)
FP8_MIN = int(os.environ.get('INK_FP8_MIN', 64))         # rows; smaller calls: bf16 (INK_FP8_DECODE: weight-only fp8)
# INK_FP8_DECODE=1: decode (fused / batched kernels, unembed) and every dense call without fp8 activations read the
# same fp8 copies weight-only (fp8_dense.mmw); the bf16 attention projections + wo are dropped, and the bf16 unembed
# too (879 MiB: every logits call and the DSpark shortlist head read the fp8 copy then; window g2 11-23 VRAM budget).
# FP8_KEEP_BF16[0] = True (tests) keeps the bf16 copies for A/B in one process.
FP8_DECODE = os.environ.get('INK_FP8_DECODE', '0') == '1'
FP8_KEEP_BF16 = [False]
MOE_IMPL = os.environ.get('INK_MOE', 'torch')  # torch (per-expert dequant) | triton (grouped, packed) | emulate (cpu mirror)
# | exact (torch loop with fp32 GEMMs rounded to bf16 at the §5 points: the reference for tests/gpu_check.py)
LITE_EXPERT_BATCH = int(os.environ.get('INK_LITE_EXPERT_BATCH', 16))  # experts staged per batch when offloaded


class Model:
    def __init__(self, path, device='cuda', max_ctx=1 << 20, pool=None):
        from kvpool import KVPool  # kvpool imports this module
        self.W = Weights(path, device)
        self.dev = self.W.dev
        self.max_ctx = max_ctx
        self.pool = pool or KVPool(max_ctx, self.dev, 'no-disk', cache_dir=None)
        self.slot = {v: i for i, v in enumerate(FULL_VISITS)}
        W = self.W
        self.d8 = None
        if FP8_DENSE or FP8_DECODE:
            from fp8_dense import Dense8
            self.d8 = Dense8(W, FP8_MIN if FP8_DENSE else 1 << 62, W['unembed.weight'][:VOCAB] if FP8_DECODE else None)
            self.d8.dec = FP8_DECODE
            if FP8_DECODE and not FP8_KEEP_BF16[0]:
                W.drop_dense()
                del W.t['unembed.weight']
        self.mq = []  # loop message queries per core layer: (q, k, v, mlp) §6
        cpq, cpkv = W['middle_cycle_cross_pass_queries'], W['middle_cycle_cross_pass_kv_queries']
        for i in range(12):
            self.mq.append(torch.stack((cpq[2 * i], cpkv[2 * i], cpkv[2 * i + 1], cpq[2 * i + 1])).float())

    def new_seq(self, tbl=None):
        return Seq(self.dev, tbl or self.pool.new_table())

    def embed(self, ids):
        e = self.W.embed[torch.as_tensor(ids, dtype=torch.long)].to(self.dev)
        return rms(e, self.W['embed_norm.weight']).bfloat16()

    # ---------------------------------------------------------------- forward
    @torch.no_grad()
    def forward(self, x0, seq):
        """x0: [T,H] bf16 normed embeddings (image rows already replaced). Returns final [T,H] bf16.
        seq: Seq (any T) or decode.DecodeState (T=1, fixed shapes, CUDA-graph safe)."""
        T = x0.shape[0]
        pos = seq.positions(T)
        self._hn = None  # INK_GLUE2: (h, rms(h)) from the last res_out
        # blocks hold rms(entry) fp32: entries only change every 6 layers, attn_res reads them at every layer
        h, blocks, v = x0, [], 0
        for L in range(6):
            h, blocks = self.layer(L, h, blocks, None, seq, v, pos); v += 1
        h_pre, b_pre, prev = h, blocks, None
        for p in range(3):
            h, blocks = h_pre, list(b_pre)
            for i, L in enumerate(CORE):
                # computed per layer: all 12 at once is 12 x [4,T,H] fp32 (7 GB at T=16k)
                msg = None if prev is None else attn_res(self.mq[i], prev)
                h, blocks = self.layer(L, h, blocks, msg, seq, v, pos); v += 1
            prev = [blocks[2], self._rms_h(h)]
        for L in range(18, 24):
            h, blocks = self.layer(L, h, blocks, None, seq, v, pos); v += 1
        f = attn_res(self.W['attn_res.output_query'].float()[None], blocks + [self._rms_h(h)])[0]
        seq.advance(T)
        return rms(f, self.W['norm.weight']).bfloat16()

    def _rms_h(self, h):
        hn = getattr(self, '_hn', None)
        return hn[1] if hn is not None and hn[0] is h else rms(h)

    def logits(self, f, fp8=None):
        """Softcapped logits; fp8 (default: d8.dec) reads the fp8 unembed (INK_FP8_DECODE)."""
        if self.d8 is not None and self.d8.un is not None and (self.d8.dec if fp8 is None else fp8):
            return self.d8.logits(f)
        x = (f @ self.W['unembed.weight'][:VOCAB].t()).float()
        return 30 * torch.tanh(x / 30)

    def lin(self, x, n):
        """Dense weight GEMM x @ W[n].t() (bf16; the hook for FP8 dense weights and the profiler's 'dense' label)."""
        return x @ self.W[n].t()

    def gate(self, L, name, x, msg, nw=None):
        """Loop gate MG (§6). x, msg fp32 [T,H] -> fp32; with nw: rms(gate, nw).bfloat16() (the only use in layer)."""
        p = f'layers.{L}.controllers.{name}.'
        if nw is not None and GATEFUSE[0] and x.shape[0] >= 64:
            return gate_up_norm(gate_down(x, msg, self.W[p + 'down_weight']), x, msg, self.W.gcat[p], nw)
        z = self.lin(gate_in(x, msg), p + 'down_weight')
        zz = z.float() @ self.W.gcat[p].t()                            # pre-tanh fp32 [forget|gain]
        return gate_out(zz, x, msg) if nw is None else gate_norm(zz, x, msg, nw)

    def layer(self, L, h, blocks, msg, seq, v, pos):
        W, P = self.W, f'layers.{L}.'
        full, bnd = L in FULL, L % 6 == 0
        g2 = GLUE2[0] and h.shape[0] >= 64 and not hasattr(seq, 'attend')
        hn = self._rms_h(h)
        if g2 and msg is None and L:  # attn_res + the input RMSNorm(s) in one kernel
            aq = W[f'attn_res.attn_queries.{L - 1}'].float().view(-1, H)
            if full:
                ws = torch.stack([W[P + f'{n}_input_norm.weight'] for n in 'qkv'])
                xq, xk, xv = attn_res_norm(aq, blocks + [hn], ws)
            else:
                xq = xk = xv = attn_res_norm(aq[:1], blocks + [hn], W[P + 'attn_norm.weight'][None])[0]
            a = None
        elif L == 0:
            a = h.float()
        else:
            a = attn_res(W[f'attn_res.attn_queries.{L - 1}'].float().view(-1, H), blocks + [hn])
        if a is None:
            pass
        elif not full:
            a = a[0] if L else a
            nw = W[P + 'attn_norm.weight']
            if msg is None:
                xq = xk = xv = rms(a, nw).bfloat16()
            else:
                xq, xk, xv = (self.gate(L, c, a, msg[j], nw) for j, c in enumerate(('attn', 'k', 'v')))
        else:
            xs = []
            for j, (c, n) in enumerate((('attn', 'q'), ('k', 'k'), ('v', 'v'))):
                w = W[P + f'{n}_input_norm.weight']
                xs.append(rms(a[j], w).bfloat16() if msg is None else self.gate(L, c, a[j], msg[j], w))
            xq, xk, xv = xs
        del a  # VRAM: [3,T,H] fp32 (113 MB at 4k) would otherwise live through attention's far blocks
        st = seq.conv[v]
        o = self.attention(L, full, xq, xk, xv, seq, v, pos, g2)
        if g2:
            r, rn = res_in(o, st['attn'], W[P + 'attn_sconv.weight'], None if bnd else h)
            if bnd:
                blocks = blocks + [hn]
        else:
            o = sconv(o, st['attn'], W[P + 'attn_sconv.weight'])
            if bnd:
                blocks, r = blocks + [hn], o.float()
            else:
                r = h.float() + o.float()
            rn = rms(r)
        nw = W[P + 'mlp_norm.weight']
        mq = W[f'attn_res.mlp_queries.{L}'].float()[None]
        if g2 and msg is None:
            u = attn_res_norm(mq, blocks + [rn], nw[None])[0]
        else:
            m = attn_res(mq, blocks + [rn])[0]
            u = rms(m, nw).bfloat16() if msg is None else self.gate(L, 'mlp', m, msg[3], nw)
            del m
        tail = (st['mlp'], W[P + 'mlp_sconv.weight'], r) if g2 and MOETAIL[0] else None
        y = self.moe(L, u, seq.moe_impl) if tail is None else self.moe(L, u, seq.moe_impl, tail)  # stubs: 3 args
        if g2:
            h, hn = y if isinstance(y, tuple) else res_out(y, st['mlp'], W[P + 'mlp_sconv.weight'], r)
            self._hn = (h, hn)
            return h, blocks
        y = sconv(y, st['mlp'], W[P + 'mlp_sconv.weight'])
        return (r + y.float()).bfloat16(), blocks

    def moe(self, L, u, impl=None, tail=None):
        """§5. u [T,H] bf16. impl overrides INK_MOE (the decode path forces triton on cuda: no host sync).
        tail (st, conv w, r): INK_MOETAIL; the grouped v2/fp4 kernels then return moe_tail's (h, rms(h))."""
        impl = impl or MOE_IMPL
        W, P = self.W, f'layers.{L}.mlp.'
        s = torch.sigmoid(u.float() @ W[P + 'gate.weight'].float().t())
        idx = (s + W[P + 'gate.bias']).topk(8, -1).indices
        w = s.gather(1, idx)
        w = w / (w.sum(-1, keepdim=True) + 1e-20) * 8.0
        off = L in getattr(W, 'offloaded', ())
        if not off and ROUTE[0] and impl == 'triton' and u.shape[0] >= 64 and (u.is_cuda or GLUE_CPU[0]):
            import moe_route
            import moe_triton
            if moe_triton.V2 or moe_triton.fp4act(L < 20, u.shape[0]):
                rt = moe_route.route(idx, w, *moe_triton.blk(L < 20, u.shape[0]), E=s.shape[1])
                return self.moe_grouped(L, u, None, rt[0], rt[1], impl, tail, rt)
        flat = idx.flatten()
        order = flat.argsort()
        tok, ws = order // 8, w.flatten()[order]
        if off:
            return self._moe_stream(L, u, flat[order], tok, ws, impl, tail)
        if impl not in ('torch', 'exact'):
            if tail is None:  # (tests stub moe_grouped with 6 args)
                return self.moe_grouped(L, u, flat[order], tok, ws, impl)
            return self.moe_grouped(L, u, flat[order], tok, ws, impl, tail)
        mm = (lambda a, b: (a.float() @ b.float().t()).bfloat16()) if impl == 'exact' else (lambda a, b: a @ b.t())
        counts = torch.bincount(flat, minlength=128).tolist()
        out = torch.zeros(u.shape, dtype=torch.float32, device=u.device)
        start = 0
        for e, n in enumerate(counts):
            if not n:
                continue
            rows = tok[start:start + n]
            w13, w2 = W.expert(L, e)
            gu = mm(u[rows], w13)
            g, up = gu[:, 0::2].float(), gu[:, 1::2].float()
            act = 4 * torch.tanh(g / 4) * torch.sigmoid(g) * 25 * torch.tanh(up / 25)
            out.index_add_(0, rows, mm(act.bfloat16(), w2).float() * ws[start:start + n, None])
            start += n
        return out.bfloat16()

    def _moe_stream(self, L, u, sorted_e, tok, ws, impl, tail=None):
        """Streamed MoE for offloaded layers: stage active experts in small batches of <= LITE_EXPERT_BATCH, run the
        grouped/emulate/torch path on each batch's rows, accumulate in fp32, and apply tail once at the end."""
        W, P, fp4 = self.W, f'layers.{L}.mlp.experts.', L < 20
        S = max(1, int(LITE_EXPERT_BATCH))
        acc = torch.zeros(u.shape, dtype=torch.float32, device=u.device)
        if len(tok):
            if u.shape[0] == 1:  # decode: 8 distinct experts, one host sync, no unique/repeat_interleave
                ue_t, cnt_t = sorted_e, None
                ue = ue_t.tolist()
                cnts = [1] * len(ue)
            else:
                ue_t, cnt_t = torch.unique_consecutive(sorted_e, return_counts=True)
                ue, cnts = torch.stack((ue_t, cnt_t)).tolist()
            mm = (lambda a, b: (a.float() @ b.float().t()).bfloat16()) if impl == 'exact' else (lambda a, b: a @ b.t())
            if impl not in ('torch', 'exact'):
                import moe_triton
                v2_or_fp4 = impl == 'triton' and (moe_triton.V2 or moe_triton.fp4act(fp4, u.shape[0]))
            row_pos = 0
            for i0 in range(0, len(ue), S):
                sub_e = ue[i0:i0 + S]
                sub_cnts = cnts[i0:i0 + S]
                n_rows = sum(sub_cnts)
                r0, r1 = row_pos, row_pos + n_rows
                row_pos = r1
                sub_tok, sub_ws = tok[r0:r1], ws[r0:r1]
                w13, s13, w2, s2 = W.stage_experts(L, sub_e, cap=S)
                if fp4:
                    ei = ue_t[i0:i0 + S]
                    gs13 = W[P + 'w13_weight_amax'][ei] / W[P + 'w13_weight_outer_scale_denominator'][0]
                    gs2 = W[P + 'w2_weight_amax'][ei] / W[P + 'w2_weight_outer_scale_denominator'][0]
                else:
                    gs13 = gs2 = None
                if impl in ('torch', 'exact'):
                    off_j = 0
                    for j, c in enumerate(sub_cnts):
                        rows = sub_tok[off_j:off_j + c]
                        if fp4:
                            q13, sc13 = w13[j], s13[j]
                            c13 = torch.stack((q13 & 15, q13 >> 4), -1).view(q13.shape[0], -1).long()
                            dw13 = (W.e2m1[c13] * sc13.float().repeat_interleave(16, -1) * gs13[j]).bfloat16()
                            q2, sc2 = w2[j], s2[j]
                            c2 = torch.stack((q2 & 15, q2 >> 4), -1).view(q2.shape[0], -1).long()
                            dw2 = (W.e2m1[c2] * sc2.float().repeat_interleave(16, -1) * gs2[j]).bfloat16()
                        else:
                            dw13 = (w13[j].float() * torch.exp2(s13[j].float() - 127).repeat_interleave(32, -1)).bfloat16()
                            dw2 = (w2[j].float() * torch.exp2(s2[j].float() - 127).repeat_interleave(32, -1)).bfloat16()
                        gu = mm(u[rows], dw13)
                        g, up = gu[:, 0::2].float(), gu[:, 1::2].float()
                        act = 4 * torch.tanh(g / 4) * torch.sigmoid(g) * 25 * torch.tanh(up / 25)
                        acc.index_add_(0, rows, mm(act.bfloat16(), dw2).float() * sub_ws[off_j:off_j + c, None])
                        off_j += c
                else:
                    sub_se = torch.arange(len(sub_e), dtype=sorted_e.dtype, device=sorted_e.device) if cnt_t is None \
                        else torch.repeat_interleave(
                            torch.arange(len(sub_e), dtype=sorted_e.dtype, device=sorted_e.device), cnt_t[i0:i0 + S])
                    if impl != 'triton':
                        moe_triton.emulate(u, sub_se, sub_tok, sub_ws, w13, s13, gs13, w2, s2, gs2, fp4, out=acc)
                    elif v2_or_fp4:
                        moe_triton.grouped(u, sub_se, sub_tok, sub_ws, w13, s13, gs13, w2, s2, gs2, fp4,
                                           fin=lambda y, t_, w_, _T: acc.index_add_(0, t_, y.float() * w_[:, None]))
                    else:
                        moe_triton.grouped(u, sub_se, sub_tok, sub_ws, w13, s13, gs13, w2, s2, gs2, fp4, out=acc)
        out = acc.bfloat16()
        return res_out(out, *tail) if tail is not None else out

    def moe_grouped(self, L, u, sorted_e, tok, ws, impl=None, tail=None, rt=None):
        import moe_triton
        P, fp4 = f'layers.{L}.mlp.experts.', L < 20
        a = [self.W[P + n] for n in ('w13_weight', 'w13_weight_scale', 'w2_weight', 'w2_weight_scale')]
        gs = [self.W[P + w + '_weight_amax'] / self.W[P + w + '_weight_outer_scale_denominator'][0]
              if fp4 else None for w in ('w13', 'w2')]
        if (impl or MOE_IMPL) != 'triton':
            return moe_triton.emulate(u, sorted_e, tok, ws, a[0], a[1], gs[0], a[2], a[3], gs[1], fp4)
        fin = None if tail is None else (lambda y, tok_, ws_, T: moe_tail(y, tok_, ws_, *tail))
        return moe_triton.grouped(u, sorted_e, tok, ws, a[0], a[1], gs[0], a[2], a[3], gs[1], fp4, fin=fin, rt=rt)

    def attention(self, L, full, xq, xk, xv, seq, v, pos, g2=False):
        """§3.2. g2: INK_GLUE2 fused prologue / output gate (prefill Seq calls only)."""
        W, T, A = self.W, xq.shape[0], f'layers.{L}.attn.'
        st = seq.conv[v]
        d8 = self.d8 if self.d8 is not None and (self.d8.dec or T >= self.d8.a8_min) else None
        if d8 is not None:  # fp8 weights (INK_FP8_DECODE: weight-only below a8_min rows; INK_FP8_DENSE: fp8 activations)
            yq, yk, yv = d8.qkv(L, xq, xk, xv)
            q, r, sk, g = yq.split(QSPLIT, -1)
            K, tk, Vv, tv = yk[:, :NKV * HD], yk[:, NKV * HD:], yv[:, :NKV * HD], yv[:, NKV * HD:]
        elif T == 1 and hasattr(W, 'proj'):  # decode: one GEMM per distinct input over the fused matrix (same per-row math)
            cat, n = W.proj[L]
            if xk is xq and xv is xq:
                yq, yk, yv = (xq @ cat.t()).split(n, -1)
            else:
                yq, yk, yv = (x @ cat[r0:r0 + k].t() for x, r0, k in zip((xq, xk, xv), (0, n[0], n[0] + n[1]), n))
            q, r, sk, g = yq.split(QSPLIT, -1)
            K, tk, Vv, tv = yk[:, :NKV * HD], yk[:, NKV * HD:], yv[:, :NKV * HD], yv[:, NKV * HD:]
        else:
            q, r, sk, g = (self.lin(xq, A + w) for w in QPROJ)
            K, Vv = self.lin(xk, A + 'wk_dv.weight'), self.lin(xv, A + 'wv_dv.weight')
            if full:
                tk, tv = self.lin(xk, A + 'k_sconv.weight_proj'), self.lin(xv, A + 'v_sconv.weight_proj')
        if g2:
            return self._attention2(L, full, q, r, sk, g, K, Vv, tk if full else None, tv if full else None, seq, v, pos)
        Q = q.reshape(T, NQ, HD).float() * W[A + 'q_l2_clip.scale'].float()
        Q = l2clip(Q, CLIP_QK).bfloat16()
        R = r.reshape(T, NQ, 16).float() * W[A + 'rel_l2_clip.scale'].float()
        R = l2clip(R, CLIP_REL).bfloat16()
        if full:
            tau = (1 + 0.1 * torch.log(torch.clamp((pos + 1).float() / 8192, min=1)))[:, None, None]
            Q, R = (Q.float() * tau).bfloat16(), (R.float() * tau).bfloat16()
            K = sconv(K, st['k'], dyn=(W[A + 'k_sconv.static_w'], tk.reshape(T, NKV, 4)))
            Vv = sconv(Vv, st['v'], dyn=(W[A + 'v_sconv.static_w'], tv.reshape(T, NKV, 4)))
        else:
            K = sconv(K, st['k'], W[A + 'k_sconv.weight'])
            Vv = sconv(Vv, st['v'], W[A + 'v_sconv.weight'])
        K = l2clip(K.view(T, NKV, HD), CLIP_QK).bfloat16()
        Vv = rms(Vv.view(T, NKV, HD)).bfloat16()
        table = (R, W[A + 'rel_logits_proj.proj'])                     # R @ proj -> [t,24,1024], built per block
        sink = sk.float()                                              # [T,24]
        if hasattr(seq, 'attend'):                                     # decode.DecodeState: fixed shapes
            out = seq.attend(self, Q, K, Vv, table, sink, v, full)
        else:
            out = self.attn_core(Q, K, Vv, table, sink, seq, v, full)  # [T,24,128] fp32
        g = torch.sigmoid(g.float())
        o = (out.view(T, NQ * HD) * g).bfloat16()
        return d8.out(L, o) if d8 is not None else self.lin(o, A + 'wo_ud.weight')

    def _attention2(self, L, full, q, r, sk, g, K, Vv, tk, tv, seq, v, pos):
        """attention() after the projections, INK_GLUE2: fused prologue, 4-bit KV written by kv_prep, gate in near."""
        W, A, st = self.W, f'layers.{L}.attn.', seq.conv[v]
        tau = (1 + 0.1 * torch.log(torch.clamp((pos + 1).float() / 8192, min=1))) if full else None
        Q, R = qr_prep(q, r, W[A + 'q_l2_clip.scale'], W[A + 'rel_l2_clip.scale'], tau)
        if full:
            K, Vv = kv_prep(K, Vv, st['k'], st['v'], W[A + 'k_sconv.static_w'], W[A + 'v_sconv.static_w'], tk, tv,
                            seq.kv, self.slot[v], seq.pos)
        else:
            K, Vv = kv_prep(K, Vv, st['k'], st['v'], W[A + 'k_sconv.weight'], W[A + 'v_sconv.weight'])
        o = self.attn_core(Q, K, Vv, (R, W[A + 'rel_logits_proj.proj']), sk.float(), seq, v, full, kv_done=True, gate=g)
        d8 = self.d8 if self.d8 is not None and (self.d8.dec or Q.shape[0] >= self.d8.a8_min) else None
        return d8.out(L, o) if d8 is not None else self.lin(o, A + 'wo_ud.weight')

    def attn_core(self, Q, K, V, table, sink, seq, v, full, kv_done=False, gate=None):
        """kv_done: K/V already in the 4-bit store; gate [T,3072] (bf16 pre-sigmoid): return the gated bf16 [T,3072]
        (the near kernel applies it) instead of the fp32 [T,24,128] output."""
        T, p0 = Q.shape[0], seq.pos
        rk, rv = seq.ring[v]
        Kn, Vn = torch.cat((rk, K)), torch.cat((rv, V))                # positions [p0-nr, p0+T)
        k0 = p0 - rk.shape[0]
        seq.ring[v] = (Kn[-(WIN - 1):], Vn[-(WIN - 1):])
        far = None
        if full:
            slot = self.slot[v]
            if not kv_done:
                seq.kv.put(slot, p0, K, V)
            if p0 + T - WIN > 0:
                far = self.far(Q, seq.kv, slot, torch.arange(p0, p0 + T, device=Q.device), p0 + T - WIN, p0)
        if PREFILL_ATTN in ('triton', 'near', 'auto'):
            from prefill_attn import near
            o = near(Q, Kn, Vn, table, sink, p0, k0, far, gate=gate)
            return o if gate is None else o.view(T, NQ * HD)
        out = torch.empty(T, NQ, HD, dtype=torch.float32, device=Q.device)
        QB = 512
        for a in range(0, T, QB):
            b = min(T, a + QB)
            ka, kb = max(0, p0 + a - (WIN - 1) - k0), p0 + b - k0     # near keys for these queries
            q = Q[a:b].view(b - a, NKV, 3, HD)
            S = torch.einsum('tkgd,skd->kgts', q, Kn[ka:kb]) * HD ** -0.5   # bf16 [8,3,t,s]
            d = (torch.arange(p0 + a, p0 + b, device=Q.device)[:, None]
                 - torch.arange(k0 + ka, k0 + kb, device=Q.device)[None])
            ok = (d >= 0) & (d < WIN)
            tb = (table[0][a:b] @ table[1] if isinstance(table, tuple) else table[a:b]).permute(1, 0, 2)  # [24,t,1024]
            bias = tb.gather(2, d.clamp(0, WIN - 1)[None].expand(NQ, -1, -1)).view(NKV, 3, b - a, -1)
            out[a:b] = self.near_merge(S, bias, ok, sink[a:b], Vn[ka:kb], None if far is None else
                                       (far[0][a:b], far[1][a:b]))
        if gate is not None:
            return (out.view(T, NQ * HD) * torch.sigmoid(gate.float())).bfloat16()
        return out

    @staticmethod
    def near_merge(S, bias, ok, sink, V, far):
        """Softmax over near keys (bf16 scores + bias, masked) with the sink and the far part merged by LSE.
        S/bias [8,3,t,s], ok [t,s] or [s], sink [t,24], V [s,8,128], far (o [t,24,128], lse [t,24]) or None."""
        t = S.shape[2]
        S = (S + bias).float().masked_fill(~ok, float('-inf'))
        parts = [torch.logsumexp(S, -1), sink.t().reshape(NKV, 3, -1)]
        if far is not None:
            parts.append(far[1].t().reshape(NKV, 3, -1))
        lse = torch.logsumexp(torch.stack(parts), 0)               # [8,3,t]
        o = torch.einsum('kgts,skd->tkgd', torch.exp(S - lse[..., None]).bfloat16(), V).float()
        if far is not None:
            wf = torch.exp(far[1].t().reshape(NKV, 3, -1) - lse).permute(2, 0, 1)[..., None]
            o = o + far[0].view(t, NKV, 3, HD) * wf
        return o.reshape(t, NQ, HD)

    def far(self, Q, kv, slot, qpos, kend, p0=None):
        """Keys [0, kend) at distance >= 1024 (no rel bias) from the 4-bit store kv, blockwise, merged by LSE.
        qpos: query positions (device). p0 (host int, prefill only) enables flex; decode passes a fixed kend
        (its bucket) and masks the rest, so the work has a fixed shape."""
        if p0 is not None and (PREFILL_ATTN in ('triton', 'far') or PREFILL_ATTN == 'auto' and (kend < PA_FAR_MAX or PA_FP8
                                                                                              or FARSKIP)):
            from prefill_attn import far
            return far(Q, kv, slot, p0, kend)
        T = Q.shape[0]
        o_acc = torch.zeros(T, NQ, HD, device=Q.device)
        lse_acc = torch.full((T, NQ), float('-inf'), device=Q.device)
        flex = p0 is not None and FAR_IMPL == 'flex' and Q.is_cuda and _flex is not None and T >= 64
        for b0 in range(0, kend, FAR_BLK):
            b1 = min(kend, b0 + FAR_BLK)
            if flex:  # read a whole block (keys >= b1 are masked, finite: stale rows or the zero page): no pad copy
                Kb, Vb = kv.get(slot, b0, min(kv.cap, b0 + FAR_BLK))
                o, lse = self._far_flex(Q, Kb, Vb, b0, p0, b1 - b0)
            else:
                Kb, Vb = kv.get(slot, b0, b1)
                o, lse = self._far_manual(Q, Kb, Vb, b0, qpos)
            del Kb, Vb  # VRAM: the next block's dequant reuses these 2 x 128 MB instead of stacking on them
            new = torch.logaddexp(lse_acc, lse)
            fa = torch.nan_to_num(torch.exp(lse_acc - new))[..., None]
            fb = torch.nan_to_num(torch.exp(lse - new))[..., None]
            o_acc.mul_(fa).add_(torch.nan_to_num_(o).mul_(fb))  # in place, same ops/rounding as o_acc*fa + o*fb
            lse_acc = new
        return o_acc, lse_acc

    def _far_manual(self, Q, Kb, Vb, b0, qpos):
        # shortcut: materializes [24,T,n] fp32 scores; fine for decode / CPU tests, use flex for big prefill chunks.
        T, n = Q.shape[0], Kb.shape[0]
        kpos = torch.arange(b0, b0 + n, device=Q.device)
        ok = kpos[None] <= (qpos[:, None] - WIN)                       # [T,n]
        q = Q.view(T, NKV, 3, HD)
        S = (torch.einsum('tkgd,skd->kgts', q, Kb) * HD ** -0.5).float().masked_fill(~ok, float('-inf'))
        lse = torch.logsumexp(S, -1)                                   # [8,3,T]
        o = torch.einsum('kgts,skd->tkgd', torch.nan_to_num(torch.exp(S - lse[..., None])).bfloat16(), Vb).float()
        return o.reshape(T, NQ, HD), lse.reshape(NQ, T).t()

    def warm_flex(self):
        """Compile flex far attention + its block mask once (the only far shape: FLEX_TQ queries x FAR_BLK keys), so the
        first prompt past INK_PA_FAR_MAX doesn't pay ~1.5 s; for engine warm-up (cuda only; ~0.3 GB temps, freed)."""
        if not (self.dev.type == 'cuda' and _flex is not None):
            return
        z = torch.zeros(FAR_BLK, NKV, HD, dtype=torch.bfloat16, device=self.dev)
        self._far_flex(torch.zeros(64, NQ, HD, dtype=torch.bfloat16, device=self.dev), z, z, 0, 2048, 1024)
        torch.cuda.synchronize()

    def _far_flex(self, Q, Kb, Vb, b0, p0, n):
        """Kb/Vb: >= n rows (only the first n are keys); rows >= n are masked."""
        T = Q.shape[0]
        if Kb.shape[0] < FAR_BLK:  # (only at the pool's end) fixed shapes avoid recompiles; padded keys are masked
            pad = FAR_BLK - Kb.shape[0]
            Kb = torch.cat((Kb, Kb.new_zeros(pad, NKV, HD)))
            Vb = torch.cat((Vb, Vb.new_zeros(pad, NKV, HD)))
        # one compiled shape per FLEX_TQ rows (was per 1024: up to 4 first-use compiles of ~1.5 s each, e.g. chunk tails;
        # GPU 2026-11-23: the first chunk past INK_PA_FAR_MAX cost ~1.6 s extra in a cold sequential prefill). Pad rows
        # are masked (qi >= T), so their blocks are skipped.
        Tp = -(-T // FLEX_TQ) * FLEX_TQ
        if Tp > T:
            Q = torch.cat((Q, Q.new_zeros(Tp - T, NQ, HD)))
        prm = _far_params(Q.device)
        prm[0], prm[1], prm[2] = p0 - b0 - WIN, n, T                    # key ok iff k <= q + off, k < n, q < T
        bm = create_block_mask(_far_mask, 1, 1, Tp, FAR_BLK, device=Q.device, _compile=Q.is_cuda)
        o, lse = _flex(Q.permute(1, 0, 2)[None], Kb.permute(1, 0, 2)[None], Vb.permute(1, 0, 2)[None],
                       block_mask=bm, scale=HD ** -0.5, enable_gqa=True, return_lse=True)
        lse = lse[0, :, :T].t().float()
        lse = torch.where(lse > -1e30, lse, float('-inf'))  # rows with no valid key
        return o[0, :, :T].permute(1, 0, 2).float(), lse
