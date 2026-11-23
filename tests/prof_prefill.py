"""Per-component prefill time of one 4096-token chunk at several context positions (real weights, real text chunk).

  (default)  GPU window: for each chunk start position P (context already in the pool: random 4-bit pages + random
             bf16 rings, so 1M needs no 1M prefill): plain forward ms (median of 3) + host time of the call (launch
             bound if ~= GPU time), then one instrumented forward with CUDA events around each component (exclusive
             ms, innermost label wins), peak VRAM. Also a real sequential prefill 0..32k (tok/s at 4k/8k/16k/32k) and
             the whole-prompt time at 77k/130k/186k/262k/1M integrated from the chunk curve. `--kernels`: torch.profiler top
             kernels at 8k and the largest P. Flags: --pos 0,8192,...  --skip-seq  --seq N (real cold prefill of N
             tokens instead of 32k; marks at 4k..N incl. 77824 / 186368)  --T N (chunk size, default 4096)
             --syncs (host<->device syncs inside one forward, via torch.cuda.set_sync_debug_mode)
  --ops      CPU, real weights, T=256 at P=0: aten launches and bytes moved (sum of in+out tensor sizes per non-view
             op) per component, GEMM vs other, scaled x16 to a 4096 chunk (attention isn't linear: indicative).
Run (window): ~/llm/engine/swap_test.sh 25 ~/llm/engine/.venv/bin/python \
              ~/llm/engine/tests/prof_prefill.py [--kernels]
Env flags pass through (INK_MOE, INK_MOE_V2, INK_MOE_FP4ACT, INK_PREFILL_ATTN, INK_FP8_DENSE, ...); unset fast paths
default to the server's cuda defaults here (setdefault)."""
import collections
import os
import statistics
import sys
import time

OPS = '--ops' in sys.argv
SMOKE = '--smoke' in sys.argv  # CPU check of the GPU path: T=256, fake events, stub MoE, torch attention
if OPS or SMOKE:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
if OPS:  # cpu: torch MoE (counted as one opaque call below); attention auto/triton = one kernel (see ops())
    os.environ.setdefault('INK_MOE', 'torch')
if SMOKE:
    os.environ.update(INK_PREFILL_ATTN='torch', INK_MOE='torch')
else:
    for k, v in dict(INK_MOE='triton', INK_MOE_V2='1', INK_PREFILL_ATTN='auto',
                     PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True').items():
        os.environ.setdefault(k, v)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import torch  # noqa: E402

import model as M  # noqa: E402
import prefill_attn as PA  # noqa: E402

T = 256 if OPS or SMOKE else int(sys.argv[sys.argv.index('--T') + 1]) if '--T' in sys.argv else 4096
stack = ['forward']
HOOK = [None]  # (name, enter: bool) -> None; set by the mode


def label(name):
    def wrap(fn):
        @torch._dynamo.disable  # INK_GLUE=compile: the event hook must stay outside compiled graphs (graph break here)
        def f(*a, **k):
            stack.append(name)
            if HOOK[0]:
                HOOK[0](name, True)
            try:
                return fn(*a, **k)
            finally:
                if HOOK[0]:
                    HOOK[0](name, False)
                stack.pop()
        return f
    return wrap


def instrument():
    """Labels, innermost wins. 'dense' = every bf16/fp8 weight GEMM (Model.lin)."""
    import kvpool
    M.attn_res = label('attn_res')(M.attn_res)
    M.rms = label('rms')(M.rms)
    M.sconv = label('sconv')(M.sconv)
    M.l2clip = label('l2clip')(M.l2clip)
    for n in M.GLUE2_FNS + ('gate_down', 'gate_up_norm', 'moe_tail'):  # INK_GLUE2 / INK_GATEFUSE / INK_MOETAIL kernels
        setattr(M, n, label(n)(getattr(M, n)))
    PA.near = label('attn near')(PA.near)
    PA.far = label('attn far (triton)')(PA.far)
    kvpool.KVPool.put = label('kv put (quant)')(kvpool.KVPool.put)
    for meth, name in (('layer', 'layer glue'), ('gate', 'loop gate'), ('attention', 'attn glue'),
                       ('attn_core', 'attn_core glue'), ('far', 'attn far (flex+merge)'), ('moe', 'moe routing'),
                       ('moe_grouped', 'moe kernels'), ('lin', 'dense GEMM'), ('logits', 'logits')):
        setattr(M.Model, meth, label(name)(getattr(M.Model, meth)))


def text_ids(n):
    from tokenizers import Tokenizer

    from engine import MODEL_DIR
    from gpu_moe import chunk_ids
    tok = Tokenizer.from_file(os.path.join(MODEL_DIR, 'tokenizer.json'))
    ids = chunk_ids(tok, min(n, 4096))
    return (ids * (-(-n // len(ids))))[:n]


def fake_seq(m, P, g):
    """Seq at position P: pages [0, P+T) reserved (pool already holds random codes), random rings for every visit."""
    seq = m.new_seq()
    assert m.pool.reserve(seq.kv, P + T), 'pool too small'
    seq.pos = P
    nr = min(P, M.WIN - 1)
    if nr:
        r = lambda *s: torch.randn(*s, device=m.dev, generator=g)
        K = M.l2clip(r(nr, M.NKV, M.HD), M.CLIP_QK).bfloat16()
        V = M.rms(r(nr, M.NKV, M.HD)).bfloat16()
        seq.ring = [(K, V) for _ in M.VISITS]
    return seq


# ---------------------------------------------------------------------------------------------------- GPU
def gpu():
    from engine import MODEL_DIR
    from gpu_prefill import fill
    arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
    pos = [int(x) for x in arg('--pos', '0,1300' if SMOKE else '0,8192,32768,131072,262144,524288,1040384').split(',')]
    dev = torch.device('cpu' if SMOKE else 'cuda')
    if SMOKE:
        _smoke_shims()
    flags = {k: v for k, v in os.environ.items() if k.startswith('INK_')}
    print('flags:', ' '.join(f'{k}={v}' for k, v in sorted(flags.items())), flush=True)
    m = M.Model(MODEL_DIR, dev.type, max_ctx=max(max(pos) + T, int(arg('--seq', 0))) + 512)
    if SMOKE:
        M.Model.moe = lambda self, L, u, impl=None: u.clone()
    fill(m.pool)
    print(f'weights + pool: {torch.cuda.memory_allocated() / 2**30:.2f} GiB allocated', flush=True)
    g = torch.Generator(device=dev).manual_seed(0)
    x = m.embed(text_ids(T))

    def plain(P):
        seq = fake_seq(m, P, g)
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter()
        a.record()
        m.forward(x, seq)
        th = time.perf_counter() - t0
        b.record(); torch.cuda.synchronize()
        m.pool.release(seq.kv)
        return a.elapsed_time(b), th * 1e3

    evs, names = [], []

    def hook(name, enter):
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        evs.append((name, enter, e))

    if '--syncs' in sys.argv and not SMOKE:  # cuda only
        import warnings
        seq = fake_seq(m, pos[0], g)
        m.forward(x, seq)                                # warm-up (compiles) outside the count
        m.pool.release(seq.kv)
        seq = fake_seq(m, pos[0], g)
        torch.cuda.synchronize()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            torch.cuda.set_sync_debug_mode(1)
            try:
                m.forward(x, seq)
            finally:
                torch.cuda.set_sync_debug_mode(0)
        m.pool.release(seq.kv)
        where = collections.Counter(str(x.message).splitlines()[0][:90] for x in w)
        print(f'\n== host syncs in one forward at P={pos[0]}: {len(w)}', flush=True)
        for k, v in where.most_common(10):
            print(f'   {v:4d} x {k}')
    curve = {}
    instrumented = False
    for P in pos:
        plain(P)                                         # warm-up / compile for this shape
        torch.cuda.reset_peak_memory_stats()
        runs = [plain(P) for _ in range(3)]
        ms, host = statistics.median(r[0] for r in runs), statistics.median(r[1] for r in runs)
        peak = torch.cuda.max_memory_allocated() / 2**30
        curve[P] = ms
        if not instrumented:
            instrument(); instrumented = True
        evs.clear()
        HOOK[0] = hook
        seq = fake_seq(m, P, g)
        a = torch.cuda.Event(enable_timing=True); a.record()
        m.forward(x, seq)
        b = torch.cuda.Event(enable_timing=True); b.record()
        HOOK[0] = None
        torch.cuda.synchronize()
        m.pool.release(seq.kv)
        # exclusive time: walk the enter/exit events; time between consecutive events goes to the innermost open label
        excl, st, last = collections.Counter(), ['forward'], a
        for name, enter, e in evs:
            excl[st[-1]] += last.elapsed_time(e)
            last = e
            st.append(name) if enter else st.pop()
        excl[st[-1]] += last.elapsed_time(b)
        tot = a.elapsed_time(b)
        print(f'\n== chunk at P={P} ({P / 1024:.0f}k): {ms:.1f} ms GPU ({T / ms * 1e3:.0f} tok/s), host {host:.1f} ms '
              f'per call, peak {peak:.2f} GiB; instrumented {tot:.1f} ms', flush=True)
        for k, v in excl.most_common():
            print(f'   {k:24s} {v:8.1f} ms  {100 * v / tot:5.1f}%')
    if '--kernels' in sys.argv:
        for P in (pos[1] if len(pos) > 1 else pos[0], pos[-1]):
            from torch.profiler import ProfilerActivity, profile
            seq = fake_seq(m, P, g)
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                m.forward(x, seq)
                torch.cuda.synchronize()
            m.pool.release(seq.kv)
            print(f'\n== top kernels at P={P}')
            print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=30, max_name_column_width=70))
    if '--skip-seq' not in sys.argv:
        print('\n== real sequential prefill (fresh seq, 4096-token chunks, real text)', flush=True)
        n_seq = int(arg('--seq', 32768))
        ids = text_ids(2 * T if SMOKE else n_seq)
        seq = m.new_seq()
        assert m.pool.reserve(seq.kv, len(ids)), 'pool too small for --seq (pass --pos up to it)'
        torch.cuda.synchronize()
        t0, marks = time.time(), {}
        for c in range(0, len(ids), T):
            e = min(len(ids), c + T)
            m.forward(m.embed(ids[c:e]), seq)
            if e in (4096, 8192, 16384, 32768, 65536, 77824, 131072, 186368, 262144) or e == len(ids) or SMOKE:
                torch.cuda.synchronize()
                marks[e] = time.time() - t0
        m.pool.release(seq.kv)
        for n, t in marks.items():
            print(f'   {n:6d} tokens: {t:.2f} s, {n / t:.0f} tok/s')
    ps = sorted(curve)
    print('\n== whole-prompt estimate from the chunk curve (linear between measured positions)')
    for n in (77824, 131072, 186368, 262144, 1040384):
        if n > ps[-1] + T:
            continue
        tot = sum(_interp(ps, curve, c) for c in range(0, n, T)) / 1e3
        print(f'   {n:8d} tokens: {tot:.1f} s, {n / tot:.0f} tok/s')
    print(f'peak VRAM overall {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB')


def _smoke_shims():
    class Ev:
        def __init__(self, **k):
            self.t = 0.
        def record(self):
            self.t = time.perf_counter()
        def elapsed_time(self, o):
            return (o.t - self.t) * 1e3
    torch.cuda.Event = Ev
    torch.cuda.synchronize = lambda *a: None
    torch.cuda.reset_peak_memory_stats = lambda *a: None
    torch.cuda.max_memory_allocated = torch.cuda.memory_allocated = lambda *a: 0


def _interp(ps, curve, c):
    for a, b in zip(ps, ps[1:]):
        if c <= b:
            return curve[a] + (curve[b] - curve[a]) * (c - a) / (b - a)
    return curve[ps[-1]]


# ---------------------------------------------------------------------------------------------------- CPU ops
def ops():
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_flatten

    from engine import MODEL_DIR
    GEMM = ('mm', 'addmm', 'bmm', 'baddbmm', '_scaled_mm')

    class Count(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.n, self.b = collections.Counter(), collections.Counter()
            self.mute = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            if self.mute:  # inside a helper that runs as one fused kernel on cuda: counted once by fused()
                return out
            ts_in = [a for a in tree_flatten((args, kwargs))[0] if isinstance(a, torch.Tensor)]
            ts_out = [o for o in tree_flatten(out)[0] if isinstance(o, torch.Tensor)]
            name = func.__name__.split('.')[0]
            ins = {a.untyped_storage().data_ptr() for a in ts_in}
            if not name.endswith('_') and ts_out and all(o.untyped_storage().data_ptr() in ins for o in ts_out):
                return out  # view
            key = (stack[-1], 'GEMM' if name in GEMM else 'other')
            self.n[key] += 1
            self.b[key] += sum(t.numel() * t.element_size() for t in ts_in + ts_out
                               if t.numel() * t.element_size() >= 4096)  # skip scalars/tiny metadata
            return out

    m = M.Model(MODEL_DIR, 'cpu', max_ctx=4096)
    c = Count()

    def fused(fn):
        """The helper runs eager on cpu but as one kernel on cuda (INK_GLUE=triton / INK_GLUE2): count one launch with
        the bytes of its tensor inputs + outputs (each storage once), not its eager ops."""
        def f(*a, **k):
            if c.mute:  # called from inside another fused helper: part of that kernel
                return fn(*a, **k)
            c.mute += 1
            try:
                out = fn(*a, **k)
            finally:
                c.mute -= 1
            ts = {t.untyped_storage().data_ptr(): t for t in tree_flatten((a, k, out))[0] if isinstance(t, torch.Tensor)}
            key = (stack[-1], 'fused')
            c.n[key] += 1
            c.b[key] += sum(t.numel() * t.element_size() for t in ts.values() if t.numel() * t.element_size() >= 4096)
            return out
        return f
    names = []
    if M.GLUE[0] == 'triton':
        names += ['rms', 'sconv', 'attn_res', 'gate_in', 'gate_out', 'gate_norm']
    if M.GLUE2[0]:
        names += list(M.GLUE2_FNS)
    if M.GATEFUSE[0]:
        names += ['gate_down', 'gate_up_norm']
    # INK_MOETAIL: not reached on cpu (torch MoE here, counted as one call); on cuda it removes res_out's [T,H] input read
    for n in names:
        setattr(M, n, fused(getattr(M, n)))
    if os.environ.get('INK_PREFILL_ATTN') in ('auto', 'triton', 'near'):  # kernel attention: count attn_core as one
        M.PREFILL_ATTN = 'torch'                                          # kernel (its math runs eager on cpu)
        M.Model.attn_core = fused(M.Model.attn_core)
        names.append('attn_core')
    M.Model.moe = fused(M.Model.moe)  # MoE: activations in/out only (weights and the cpu torch path's temps not counted)
    if names:
        print('counted as fused kernels:', ' '.join(names))
    instrument()
    x = m.embed(text_ids(T))
    seq = m.new_seq()
    t0 = time.time()
    with c:
        m.forward(x, seq)
    print(f'forward T={T} on cpu: {time.time() - t0:.0f} s; bytes scaled x{4096 // T} to a 4096-token chunk')
    tot = sum(c.b.values()) * (4096 // T)
    for k, v in sorted(c.b.items(), key=lambda kv: -kv[1]):
        gb = v * (4096 // T) / 1e9
        print(f'   {k[0]:24s} {k[1]:5s} launches {c.n[k]:6d}  {gb:8.1f} GB  {100 * gb * 1e9 / tot:5.1f}%  '
              f'(~{gb / 1.6:.1f} ms at 1.6 TB/s)')
    print(f'   total launches {sum(c.n.values())}, {tot / 1e9:.1f} GB')


if __name__ == '__main__':
    sys.exit(ops() if OPS else gpu())  # --smoke: gpu() on the cpu
