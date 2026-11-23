"""GPU window only: the fixed-shape decode path (decode.py, CUDA graphs) vs the current decode, plus decode speed.
For each prompt: prefill once (triton MoE), then decode from clones of that state, teacher-forced on the greedy
tokens of the reference (Seq path + triton MoE, i.e. the current decode with INK_MOE=triton):
  eager-dec  Decoder without graphs (same code as graph; checks the math)         <- gated: KL vs ref
  graph      Decoder with CUDA graphs (what INK_GRAPH=1 gives), informational KL + tok/s
  rerun      reference again (triton atomic-add order: run-to-run noise)          noise baseline
  nudged     reference with every decode input embedding moved +-1 bf16 ulp       noise baseline
  torch      Seq path + torch MoE (the default today), informational
KL gate: mean KL(ref || eager-dec) <= 2x max(noise baselines) + 1e-6, finite logits.
Lockstep (the graph gate, catches capture bugs): every step starts graph, eager2 and eager from the same state (eager's,
copied into the other two) and compares their logits: rel = max|l - l_eager| / max|l_eager|. Same math, so the
median and 95th percentile of graph's per-step rel must stay <= max(1e-3, 2x) those of eager2 (triton MoE atomics
noise; median over eager2's nonzero steps), with top-1 equal on >= 97% of steps, min. 1 flip allowed (lock_gate); prints the first step over 1e-3 with its position and far
bucket. Prompts: 3 short natural chats, ~1k (decode crosses position 1024: a second graph gets
captured and far attention starts), ~40k, and with --long ~130k. Reports median tok/s per path. Exits 1 on a gate
failure, 2 on an exception.
Fused path (decode_kernels.py, INK_DEC_FUSE): 'fused-eager' (gated like eager-dec: KL vs ref) and 'fused-graph'
(tok/s), plus a second lockstep: fused-graph vs fused-eager (capture bugs). --no-fuse skips them.
Per-component timings: one non-graph step per path with CUDA events around each component (plain Decoder: model
methods, inclusive/nested; fused: per kernel). --1m: an extra ~1M-token prompt (prefill ~10 min), speed only
(graph and fused-graph, 16 steps, no KL). INK_DEC_FAR_DOT=1 switches both decoders to the tensor-core far kernel.
Usage (Qwen down, nothing else on the GPU):
  ~/llm/engine/swap_test.sh 30 python tests/gpu_decode.py [--long] [--1m] [--no-fuse]"""
import os
import statistics
import sys
import time
import traceback

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import torch  # noqa: E402

import chat  # noqa: E402
import decode as D  # noqa: E402
import model as M  # noqa: E402
from engine import Engine  # noqa: E402
from gpu_check import PROMPTS, ulp_nudge  # noqa: E402
from oracle import prose  # noqa: E402

LONG = '--long' in sys.argv
ONE_M = '--1m' in sys.argv
FUSE = '--no-fuse' not in sys.argv


def timed_step(dec, t, fused):
    """One eager step with CUDA events per component -> {label: ms} (inclusive; nested labels overlap)."""
    import decode_kernels as K
    ev, stack, saved = [], [], []

    def wrap(obj, name, label):
        fn = getattr(obj, name)
        saved.append((obj, name, fn))

        def f(*a, **k):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            out = fn(*a, **k)
            e.record()
            ev.append((label, s, e))
            return out
        setattr(obj, name, f)

    if fused:
        for n, k in list(vars(K).items()):
            if n.endswith('_kernel') and hasattr(k, 'run'):
                class L:
                    def __init__(self, k, n):
                        self.k, self.n = k, n

                    def __getitem__(self, grid):
                        g = self.k[grid]

                        def run(*a, **kw):
                            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                            s.record()
                            g(*a, **kw)
                            e.record()
                            ev.append((self.n.strip('_').replace('_kernel', ''), s, e))
                        return run
                saved.append((K, n, k))
                setattr(K, n, L(k, n))
        wrap(D, 'launch_far', 'far')
        wrap(dec.m, 'logits', 'logits')
    else:
        for obj, name, label in ((M, 'attn_res', 'attn-res'), (dec.m, 'gate', 'loop gate'), (M, 'sconv', 'sconv'),
                                 (dec.st, 'attend', 'attend (near+far)'), (D, 'far_attn', 'far'),
                                 (dec.m, 'moe', 'moe (routing+experts)'), (dec.m, 'attention', 'attention (all)'),
                                 (dec.m, 'logits', 'logits')):
            wrap(obj, name, label)
    try:
        torch.cuda.synchronize()
        s0, e0 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s0.record()
        dec.step(t)
        e0.record()
        torch.cuda.synchronize()
    finally:
        for obj, name, fn in reversed(saved):
            setattr(obj, name, fn)
    tot = {}
    for label, s, e in ev:
        tot[label] = tot.get(label, 0.) + s.elapsed_time(e)
    tot['TOTAL step (eager)'] = s0.elapsed_time(e0)
    return tot


def clone(seq):
    s = M.Seq.__new__(M.Seq)
    s.__dict__.update(seq.__dict__)
    s.conv = [{k: v.clone() for k, v in d.items()} for d in seq.conv]   # updated in place by sconv
    s.ring = list(seq.ring)                                            # attn_core replaces the tuples
    return s


def seq_stepper(m, seq, impl, nudge=False):
    def step(t):
        old, M.MOE_IMPL = M.MOE_IMPL, impl
        try:
            x = m.embed([t])
            return m.logits(m.forward(ulp_nudge(x, seed=seq.pos) if nudge else x, seq))
        finally:
            M.MOE_IMPL = old
    return step


def dec_stepper(dec, seq):
    dec.load(clone(seq))
    return dec.step


def sync(dst, src):
    """dst Decoder state := src's (in place: graphs hold these addresses). Page tables already match (same seq)."""
    for a, b in zip(dst.st.conv, src.st.conv):
        for k in a:
            a[k].copy_(b[k])
    for (a, b), (c, d) in zip(dst.st.ring, src.st.ring):
        a.copy_(c)
        b.copy_(d)
    dst.st.rpos.copy_(src.st.rpos)


def lockstep(graph, eager, eager2, seq, toks):
    """Per step from eager's state: (step, pos, far bucket, graph rel, eager2 rel, graph top-1 == eager top-1).
    All three write the same KV pages; eager steps last so its K/V is what stays."""
    for d in (graph, eager, eager2):
        d.load(clone(seq))
    rows = []
    for i, t in enumerate(toks):
        p = eager.seq.pos
        sync(graph, eager)
        sync(eager2, eager)
        g, e2, e = graph.step(t).clone(), eager2.step(t).clone(), eager.step(t).clone()
        rel = lambda x: ((x - e).abs().max() / e.abs().max()).item()
        rows.append((i, p, D.far_bucket(p, eager.st.kv.cap), rel(g), rel(e2), int(g.argmax()) == int(e.argmax())))
    return rows


def q(xs, f):
    """Quantile f of xs (linear interpolation)."""
    xs = sorted(xs)
    i = f * (len(xs) - 1)
    return xs[int(i)] + (xs[min(int(i) + 1, len(xs) - 1)] - xs[int(i)]) * (i - int(i))


def lock_gate(label, rows):
    """Robust lockstep gate: median and 95th percentile of the per-step rel diff must stay <= max(1e-3, 2x) the same
    statistic of eager2 (same math, triton MoE atomics; median over its nonzero steps), and top-1 must match on >= 97%
    of steps (at least one flip allowed: 32-step prompts saw a single near-tie flip). A single noisy
    step no longer fails; a capture bug (systematic offset, drift over steps, top-1 flips) still does."""
    g, nz = [r[3] for r in rows], [r[4] for r in rows]
    same = sum(r[5] for r in rows)
    # the noise is bursty (eager2 == eager bit for bit on many steps, MoE atomics flip on others), so the median
    # limit uses the median of the noisy steps only
    lm, lp = max(1e-3, 2 * q([x for x in nz if x > 0] or [0.], .5)), max(1e-3, 2 * q(nz, .95))
    good = q(g, .5) <= lm and q(g, .95) <= lp and len(rows) - same <= max(1, int(0.03 * len(rows)))  # one near-tie flip ok
    bad = [r for r in rows if r[3] > 1e-3]
    print(f'lockstep {label}: rel median {q(g, .5):.1e} / p95 {q(g, .95):.1e} / max {max(g):.1e} (eager2 noise '
          f'{q(nz, .5):.1e} / {q(nz, .95):.1e} / {max(nz):.1e}, limits {lm:.1e} / {lp:.1e}), top-1 same {same}/{len(rows)}, '
          + ('first > 1e-3 at step %d (pos %d, far bucket %d): %.1e' % bad[0][:4] if bad else 'none > 1e-3')
          + ('  OK' if good else '  TOO BIG'), flush=True)
    if not good:
        for r in rows:
            print('  step %2d pos %6d bucket %6d  graph %.1e  eager2 %.1e  top-1 %s' % r, flush=True)
    return good


def run(step, first, n, toks=None):
    """Feed toks (teacher forcing) or greedy tokens from `first` logits. -> (tokens fed, log-probs [n,V], step times)."""
    t = int(first.argmax()) if toks is None else toks[0]
    fed, lps, times = [], [], []
    for i in range(n):
        fed.append(t)
        torch.cuda.synchronize()
        t0 = time.time()
        lg = step(t)
        nxt = int(lg[0].argmax())                                      # host sync, as in real decoding
        times.append(time.time() - t0)
        lps.append(torch.log_softmax(lg[0].float(), -1))
        t = nxt if toks is None else (toks[i + 1] if i + 1 < len(toks) else nxt)
    return fed, torch.stack(lps), times


def prompts(tok):
    from oracle import prose
    out = [(f'natural {i}', chat.render(tok, [{'role': 'user', 'content': p}], effort=0.9), 64)
           for i, p in enumerate(PROMPTS[:3])]
    one_k = chat.render(tok, [{'role': 'user', 'content': prose(1100, tok, 5)}], effort=0.9)[:1000]
    out.append(('1k (crosses 1024)', one_k, 64))
    out.append(('40k', chat.render(tok, [{'role': 'user', 'content': prose(40000, tok, 2)}], effort=0.9), 32))
    if LONG:
        out.append(('130k', chat.render(tok, [{'role': 'user', 'content': prose(130000, tok, 3)}], effort=0.9), 16))
    return out


def main():
    eng = Engine('cuda', max_ctx=1_050_000 if ONE_M else 140_000 if LONG else 48_000, vision=False)
    m = eng.model
    graph, eager, eager2 = D.Decoder(m, graph=True), D.Decoder(m, graph=False), D.Decoder(m, graph=False)
    if FUSE:
        fgraph, feager, feager2 = (D.Decoder(m, graph=g, fuse=True) for g in (True, False, False))
    print(f'loaded: {torch.cuda.memory_allocated() / 2**30:.2f} GiB', flush=True)
    ok, table = True, []
    for name, ids, n in prompts(eng.tok):
        M.MOE_IMPL = 'triton'
        seq = m.new_seq()
        t0 = time.time()
        for a in range(0, len(ids), eng.chunk):
            f = m.forward(m.embed(ids[a:a + eng.chunk]), seq)[-1:]
        first = m.logits(f)[0]
        print(f'\n== {name}: {len(ids)} prompt tokens, prefill {time.time() - t0:.1f} s, {n} decode steps', flush=True)
        toks, ref, tref = run(seq_stepper(m, clone(seq), 'triton'), first, n)
        p_ref = ref.exp()
        res = {'ref (seq+triton)': (0., 1., n, tref)}
        paths = [('rerun', lambda: seq_stepper(m, clone(seq), 'triton')),
                 ('nudged', lambda: seq_stepper(m, clone(seq), 'triton', nudge=True)),
                 ('eager-dec', lambda: dec_stepper(eager, seq)),
                 ('graph', lambda: dec_stepper(graph, seq)),
                 ('torch (default today)', lambda: seq_stepper(m, clone(seq), 'torch'))]
        if FUSE:
            paths[4:4] = [('fused-eager', lambda: dec_stepper(feager, seq)), ('fused-graph', lambda: dec_stepper(fgraph, seq))]
        for pname, mk in paths:
            _, lp, tm = run(mk(), first, n, toks)
            kl = (p_ref * (ref - lp)).sum(-1)
            agree = (lp.argmax(-1) == ref.argmax(-1))
            first_miss = int((~agree).nonzero()[0]) if not agree.all() else n
            fin = bool(torch.isfinite(lp).all())
            res[pname] = (kl.mean().item(), agree.float().mean().item(), first_miss, tm)
            ok &= fin
            if not fin:
                print(f'{pname}: NON-FINITE log-probs', flush=True)
        noise = max(res['rerun'][0], res['nudged'][0])
        for pname, (kl, agree, miss, tm) in res.items():
            gated = pname in ('eager-dec', 'fused-eager')
            good = kl <= 2 * noise + 1e-6
            ok &= good or not gated
            tps = 1 / statistics.median(tm)
            mark = ('OK' if good else 'TOO BIG') if gated else ''
            print(f'{pname:22s} {tps:7.1f} tok/s (median step {statistics.median(tm) * 1e3:7.2f} ms, first {tm[0]:.2f} s)'
                  f'  KL {kl:.2e}  top-1 {agree:.3f}  same greedy for {miss}/{n}  {mark}', flush=True)
            table.append((name, pname, tps, kl))
        print(f'noise baseline {noise:.2e}, gate {2 * noise + 1e-6:.2e}', flush=True)
        ok &= lock_gate('graph vs eager-dec', lockstep(graph, eager, eager2, seq, toks))
        if FUSE:
            ok &= lock_gate('fused-graph vs fused-eager', lockstep(fgraph, feager, feager2, seq, toks))
        for label, d, fz in [('eager-dec', eager, False)] + ([('fused-eager', feager, True)] if FUSE else []):
            d.load(clone(seq))
            ms = timed_step(d, toks[0], fz)
            print(f'components, {label} (one non-graph step, CUDA events, ms):  ' +
                  '  '.join(f'{k} {v:.2f}' for k, v in sorted(ms.items(), key=lambda x: -x[1])), flush=True)
            d.seq = None
        graph.seq = eager.seq = eager2.seq = None
        if FUSE:
            fgraph.seq = feager.seq = feager2.seq = None  # their clones share seq's page table: drop it so its pages go back (--long)
        del seq, ref, p_ref
    if ONE_M:  # VRAM is tight at 1M: keep only the two graph decoders (~200 MB of rings each)
        eager = eager2 = feager = feager2 = None
        torch.cuda.empty_cache()
        ids = chat.render(eng.tok, [{'role': 'user', 'content': prose(1_000_000, eng.tok, 4)}], effort=0.9)
        seq = m.new_seq()
        assert m.pool.reserve(seq.kv, len(ids) + 64)
        t0 = time.time()
        for a in range(0, len(ids), eng.chunk):
            f = m.forward(m.embed(ids[a:a + eng.chunk]), seq)[-1:]
        first = m.logits(f)[0]
        print(f'\n== 1M: {len(ids)} prompt tokens, prefill {time.time() - t0:.0f} s', flush=True)
        for pname, d in [('graph', graph)] + ([('fused-graph', fgraph)] if FUSE else []):
            _, lp, tm = run(dec_stepper(d, seq), first, 16)
            tps = 1 / statistics.median(tm)
            ok &= bool(torch.isfinite(lp).all())
            print(f'{pname:22s} {tps:7.1f} tok/s at {len(ids)} (median step {statistics.median(tm) * 1e3:.2f} ms)', flush=True)
            table.append(('1M', pname, tps, float('nan')))
            d.seq = None
        del seq
    print('\n== summary (median tok/s)')
    for name, pname, tps, kl in table:
        print(f'{name:20s} {pname:22s} {tps:7.1f}')
    print(f'graphs captured (far buckets): {sorted(graph.graphs)}')
    print('PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 2
    sys.exit(rc)
