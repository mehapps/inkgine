"""GPU window only: INK_FP8_DECODE (e4m3 dense weights, one fp32 scale per output row, in decode) vs bf16 in ONE process
(model.FP8_KEEP_BF16 keeps both copies; decoders pick their precision when they are built from m.d8.dec).
  1. kernels on real weights: fused proj / wo W8 kernels and logits8 vs the dequantized fp32 reference (bit-equal share)
  2. decode-only KL (prefill bf16, then teacher-forced on bf16 fused-graph greedy tokens, N steps; 3 chats, ~1k, 40k):
     fp8 fused-graph and fp8 BatchDecoder (B=1, graph) vs noise = max over (bf16 eager unfused, bf16 fused after a
     +-1-ulp nudged prefill) of the per-step KL median / p95 (gpu_batch's rule). Gate: median <= 2x + 1e-4,
     p95 <= 2x + 1e-3, top-1 >= 95%. Lockstep fp8 fused graph vs eager (capture bugs, gpu_decode.lock_gate).
  3. whole pipeline (prefill + logits in fp8 weight-only, and with INK_FP8_DENSE-style fp8 activations) vs bf16 on
     gpu_check's prompts, all positions; gate weight-only <= 2x the 1-ulp embedding-nudge baseline.
  4. speed bf16 vs fp8: fused-graph single stream and BatchDecoder B = 1, 2, 4, 8 at ~1k / 40k (/ 130k with --long);
     verify step T = 1, 2, 4, 8; per-kernel ms of one eager fused step (proj / wo / logits).
--vram: no A/B; load the engine as the server would at 1M (INK_FP8_DECODE as set: bf16 copies dropped when 1), print
     allocated / free, then 32 greedy fused-graph tokens of a chat (text printed) + tok/s. Run with =0 and =1.
Exit 1 on a gate failure, 2 on an exception.
  ~/llm/engine/swap_test.sh 40 env INK_MOE=triton INK_DEC_FAR_DOT=1 python ~/llm/engine/tests/gpu_fp8dec.py [--long]"""
import os
import statistics
import sys
import time
import traceback

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
VRAM = '--vram' in sys.argv
if not VRAM:
    os.environ['INK_FP8_DECODE'] = '1'
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import torch  # noqa: E402

import chat  # noqa: E402
import decode as D  # noqa: E402
import decode_batch as DB  # noqa: E402
import decode_kernels as K  # noqa: E402
import fp8_dense as F8  # noqa: E402
import model as M  # noqa: E402
from engine import Engine  # noqa: E402
from gpu_batch import fork, kl, prefill, speed, teacher, verify_cost  # noqa: E402
from gpu_check import PROMPTS, prompts, ulp_nudge  # noqa: E402
from gpu_decode import lock_gate, lockstep, q, run, timed_step  # noqa: E402
from oracle import prose  # noqa: E402

LONG = '--long' in sys.argv
N = 48


def check(name, got, ref):
    eq = (got.float() == ref.bfloat16().float()).float().mean().item()
    err = ((got.double() - ref).abs().max() / ref.abs().max()).item()
    good = eq >= 0.99 and err < 2 ** -7
    print(f'{name:34s} bit-equal {eq * 100:6.2f}%  max rel {err:.1e}  {"OK" if good else "BAD"}', flush=True)
    return good


def kernels(m):
    d8, ok, g = m.d8, True, torch.Generator(device='cuda').manual_seed(0)
    fz = fdec(m, True, False).fused
    for v in (5, 30):
        d = fz.vis[v]
        (qw, s), ns = d8.proj[d['L']]
        x = torch.randn(3, M.H, device='cuda', generator=g).bfloat16()
        y = torch.zeros(8664, dtype=torch.bfloat16, device='cuda')
        K._proj_kernel[(d['N'] // K.BLK['proj'],)](qw, ns[0], ns[1], x, M.H, d['nwt'], y, s, NORM=False, F32=False,
                                                   H=M.H, CH=K.CH, BN=K.BLK['proj'], W8=True)
        deq = qw.double()[:sum(ns)] * s.double()[:sum(ns), None]
        rows = ((0, ns[0]), (ns[0], ns[0] + ns[1]), (ns[0] + ns[1], sum(ns)))
        ok &= check(f'proj W8 visit {v}', y[:sum(ns)], torch.cat([deq[a:b] @ x[j].double() for j, (a, b) in enumerate(rows)]))
    f = torch.randn(16, M.H, device='cuda', generator=g).bfloat16()
    qu, su = d8.un
    ref = f[:2].double() @ (qu.double() * su.double()[:, None]).t()
    # pre-softcap kernel output directly (inverting the fp32 softcap with atanh amplified tanh's rounding: 38% "equal")
    ok &= check('logits8 (2 rows, pre-softcap)', F8.logits8(f[:2], qu, su, cap=False), ref)
    xw = torch.randn(5, M.H, device='cuda', generator=g).bfloat16()
    qw, s = d8.proj[0][0]
    ok &= check('mmw (torch.mm out_dtype fp32)', F8.mmw(xw, qw, s), xw.double() @ (qw.double() * s.double()[:, None]).t())
    from gpu_moe import cuda_ms
    for T in (1, 16):
        t8, tb = cuda_ms(lambda: F8.logits8(f[:T], qu, su), 20), cuda_ms(lambda: m.logits(f[:T], False), 20)
        print(f'logits T={T}: fp8 kernel {t8:.3f} ms ({qu.numel() / t8 / 1e6:.0f} GB/s) vs bf16 {tb:.3f} ms', flush=True)
    return ok


def fdec(m, fp8, graph):
    """Fused Decoder of one precision: its Fused (built here, not lazily) reads the weights m.d8.dec selects."""
    old, m.d8.dec = m.d8.dec, fp8
    d = D.Decoder(m, graph=graph, fuse=True)
    d.fused = K.Fused(m, d.st, d.x_raw, d.out)
    m.d8.dec = old
    return d


def bdec(m, fp8, slots):
    old, m.d8.dec = m.d8.dec, fp8
    bd = DB.BatchDecoder(m, slots, graph=True)
    m.d8.dec = old
    return bd


def main_ab():
    M.FP8_KEEP_BF16[0] = True
    # pool: 40k prompt + its nudged twin during step 2 (+ 130k with --long) + forks
    eng = Engine('cuda', max_ctx=220_000 if LONG else 100_000, vision=False)
    m = eng.model
    M.MOE_IMPL = 'triton'
    print(f'loaded (bf16 + fp8 copies): {torch.cuda.memory_allocated() / 2**30:.2f} GiB', flush=True)
    print('\n== 1. kernels', flush=True)
    ok = kernels(m)

    print('\n== 2. decode-only KL vs bf16 (prefill bf16)', flush=True)
    m.d8.dec = False                                  # prefill + the unfused eager path in bf16
    fb, eb = fdec(m, False, True), D.Decoder(m, graph=False, fuse=False)
    f8, f8e, f8e2 = fdec(m, True, True), fdec(m, True, False), fdec(m, True, False)
    bd8 = bdec(m, True, 1)
    cases = [(f'chat {i}', chat.render(eng.tok, [{'role': 'user', 'content': p}], effort=0.9)) for i, p in enumerate(PROMPTS[:3])]
    cases.append(('~1k', chat.render(eng.tok, [{'role': 'user', 'content': prose(1100, eng.tok, 5)}], effort=0.9)[:1000]))
    cases.append(('40k', chat.render(eng.tok, [{'role': 'user', 'content': prose(40000, eng.tok, 2)}], effort=0.9)))
    pre = {}
    for name, ids in cases:
        s, f = prefill(m, ids, eng.chunk)
        pre[name] = (s, f)
        r = teacher(fb, s, f, N)
        sq = m.new_seq()
        for a in range(0, len(ids), eng.chunk):
            fn = m.forward(ulp_nudge(m.embed(ids[a:a + eng.chunk])), sq)[-1:]
        ks = [kl(r[1], teacher(eb, s, f, N, r[0])[1]), kl(r[1], teacher(fb, sq, m.logits(fn)[0], N, r[0])[1])]
        del sq
        nm, npp = max(q(k.tolist(), .5) for k in ks), max(q(k.tolist(), .95) for k in ks)
        lp8 = teacher(f8, s, f, N, r[0])[1]
        sb = fork(m, s)
        bd8.load(0, sb)
        lpb = []
        for j in range(N):
            lpb.append(torch.log_softmax(bd8.step([(0, bd8.seqs[0].pos, r[0][j])])[0].float(), -1))
        bd8.free(0)
        for label, lp in (('fp8 fused-graph', lp8), ('fp8 batch B=1', torch.stack(lpb))):
            k = kl(r[1], lp).tolist()
            km, kp = q(k, .5), q(k, .95)
            top = (lp.argmax(-1) == r[1].argmax(-1)).float().mean().item()
            good = km <= 2 * nm + 1e-4 and kp <= 2 * npp + 1e-3 and top >= 0.95 and bool(torch.isfinite(lp).all())
            ok &= good
            print(f'{name:6s} {label:16s} KL median {km:.1e} / p95 {kp:.1e} / mean {statistics.mean(k):.1e} (noise '
                  f'{nm:.1e} / {npp:.1e}), top-1 {top:.3f}  {"OK" if good else "TOO BIG"}', flush=True)
        if name in ('chat 0', '~1k'):
            ok &= lock_gate(f'fp8 fused graph vs eager ({name})', lockstep(f8, f8e, f8e2, s, r[0][:24]))
            for d in (f8, f8e, f8e2):
                d.seq = None
    f8e = f8e2 = eb = None

    print('\n== 3. whole pipeline (prefill + logits) fp8 vs bf16, all positions', flush=True)
    tot = {}
    for pname, ids in prompts(eng.tok):
        x0 = m.embed(ids)

        def lp(x, dec, a8=1 << 62):
            m.d8.dec, m.d8.a8_min = dec, a8
            try:
                return torch.log_softmax(m.logits(m.forward(x, m.new_seq())), -1)
            finally:
                m.d8.dec, m.d8.a8_min = False, 1 << 62
        ref = lp(x0, False)
        p = ref.exp()
        for k, (x, dec, a8) in {'ulp-nudged emb': (ulp_nudge(x0), False, 1 << 62), 'fp8 weight-only': (x0, True, 1 << 62),
                                'fp8 + fp8 act (T>=64)': (x0, True, 64)}.items():
            got = lp(x, dec, a8)
            t = tot.setdefault(k, [0., 0., 0])
            t[0] += (p * (ref - got)).sum().item()
            t[1] += (ref.argmax(-1) == got.argmax(-1)).sum().item()
            t[2] += len(ids)
    for k, (a, b, n) in tot.items():
        print(f'  {k:22s} mean KL {a / n:.3e}  top-1 {b / n:.4f}', flush=True)
    good = tot['fp8 weight-only'][0] <= 2 * tot['ulp-nudged emb'][0]
    ok &= good
    print(f'fp8 weight-only pipeline within 2x the 1-ulp baseline: {good}', flush=True)

    print('\n== 4. speed bf16 vs fp8', flush=True)
    table = []
    ctxs = [('~1k', pre['~1k'][0], pre['~1k'][1]), ('40k', *pre['40k'])]
    if LONG:
        x = chat.render(eng.tok, [{'role': 'user', 'content': prose(130000, eng.tok, 3)}], effort=0.9)
        ctxs.append(('130k', *prefill(m, x, eng.chunk)))
    for name, s, f in ctxs:
        for label, d in (('bf16', fb), ('fp8', f8)):
            d.load(fork(m, s))
            _, _, tm = run(d.step, f, 40)
            d.seq = None
            table.append((name, f'fused-graph {label}', 1, 1 / statistics.median(tm[4:])))
            print(f'{name:5s} fused-graph {label}: {1 / statistics.median(tm[4:]):6.1f} tok/s', flush=True)
    bd8.drop()
    del bd8
    for label, fp8 in (('bf16', False), ('fp8', True)):
        bd = bdec(m, fp8, 8)
        for name, s, f in ctxs:
            for B in (1, 2, 4, 8):
                dt = speed(bd, [fork(m, s) for _ in range(B)])
                table.append((name, f'batch {label}', B, B / dt))
                print(f'{name:5s} batch {label} B={B}: step {dt * 1e3:6.2f} ms, aggregate {B / dt:6.1f} tok/s', flush=True)
            t1 = None
            for T in (1, 2, 4, 8):
                dt = verify_cost(bd, fork(m, s), T)
                t1 = t1 or dt
                print(f'{name:5s} batch {label} verify T={T}: {dt * 1e3:6.2f} ms (x{dt / t1:.2f})', flush=True)
        bd.drop()
        del bd
        torch.cuda.empty_cache()
    for label, fp8 in (('bf16', False), ('fp8', True)):
        d = fdec(m, fp8, False)
        d.load(fork(m, pre['~1k'][0]))
        d.step(1000)                                  # warm (Triton compile) outside the timing
        ms = timed_step(d, 1001, True)
        d.seq = None
        print(f'components {label} (one eager fused step at ~1k, ms): ' +
              '  '.join(f'{k} {v:.3f}' for k, v in sorted(ms.items(), key=lambda x: -x[1])[:12]), flush=True)
    print('\n== summary (tok/s, aggregate)')
    for name, path, B, tps in table:
        print(f'{name:5s} {path:18s} B={B}  {tps:7.1f}')
    print('PASS' if ok else 'FAIL')
    return 0 if ok else 1


def main_vram():
    eng = Engine('cuda', max_ctx=1_050_000, vision=False)
    m = eng.model
    M.MOE_IMPL = 'triton'
    free, total = torch.cuda.mem_get_info()
    print(f'INK_FP8_DECODE={os.environ.get("INK_FP8_DECODE", "0")}: allocated after load (1M pool) '
          f'{torch.cuda.memory_allocated() / 2**30:.2f} GiB, free {free / 2**30:.2f} GiB of {total / 2**30:.2f}', flush=True)
    ids = chat.render(eng.tok, [{'role': 'user', 'content': PROMPTS[0]}], effort=0)
    s, f = prefill(m, ids, eng.chunk)
    d = D.Decoder(m, graph=True, fuse=True)
    d.load(s)
    toks, _, tm = run(d.step, f, 32)
    print(f'greedy 32 tokens: {eng.tok.decode(toks)!r}\n{1 / statistics.median(tm[4:]):.1f} tok/s; peak allocated '
          f'{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB', flush=True)
    return 0


if __name__ == '__main__':
    try:
        rc = main_vram() if VRAM else main_ab()
    except Exception:
        traceback.print_exc()
        rc = 2
    sys.exit(rc)
