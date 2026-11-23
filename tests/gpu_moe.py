"""Prefill MoE: current grouped kernel vs v2 (INK_MOE_V2, same bf16 math) vs fp4act (INK_MOE_FP4ACT, FP4 tensor cores).

  --compile  CPU, no GPU: sm_120 offline compile of the v2 + fp4 kernels with the engine's configs (shared memory <=
             101376 B, fp4 kernels must use the block-scaled FP4 mma).
  --report   CPU, real weights: fp4act error per NVFP4 layer on real activations of a short prompt (1 and 2 terms),
             next to the bf16 kernel's own error, both vs the exact reference. ~20-40 min.
  (default)  GPU window: real activations of a 4096-token chunk (48 visits). Per visit: time old / v2 / fp4, error vs
             the exact reference (INK_MOE=exact), v2 vs old, v2 determinism, fp4 kernel vs its torch mirror, quant
             kernel vs torch; decode shapes (T=1, 8). Whole 4096-token chunk time per path. End-to-end KL vs the old
             path on gpu_check's prompts with noise baselines (old rerun = atomics order, embeddings +-1 ulp).
             Exit 1 on a v2/fp4 bug (mismatch, non-determinism, non-finite); the fp4 KL verdict is printed only
             (INK_MOE_FP4ACT stays off by default either way).
  --sweep    GPU window: also time v2 / fp4 tile configs on 3 visits (outputs checked against the default config).
Usage: ~/llm/engine/swap_test.sh 25 python tests/gpu_moe.py [--sweep]"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
SMEM = 101376


def compile_check():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ.pop('TRITON_INTERPRET', None)
    os.environ.setdefault('TRITON_DEFAULT_FP_FUSION', '1')
    import re

    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    import moe_fp4 as F
    import moe_triton as MT
    B, I32, U8, F8, FP = '*bf16', '*i32', '*u8', '*fp8e4nv', '*fp32'
    st = dict(lwe='i32', lwr='i32', lse='i32', lsr='i32')
    tb = dict(be_ptr=I32, b0_ptr=I32, b1_ptr=I32, bb_ptr=I32)
    cases = []
    for fp4 in (True, False):
        w, s = (U8, F8) if fp4 else (F8, U8)
        for name, fn, ty in (('w13', MT._w13_v2, dict(u_ptr=B, tok_ptr=I32, act_ptr=B, K='i32', NI='i32')),
                             ('w2', MT._w2_v2, dict(act_ptr=B, y_ptr=B, K='i32', N='i32'))):
            bn, bk, nw, ns = MT.CFG[(name, fp4)]
            cases.append((f'v2 {name} fp4={fp4}', fn, dict(ty, **tb, **st, w_ptr=w, s_ptr=s, gs_ptr=FP),
                          dict(FP4=fp4, BB=MT.BB, BS=MT.BS, BN=bn, BK=bk, F32=False), dict(num_warps=nw, num_stages=ns)))
    cases.append(('fin', MT._fin_kernel, dict(y_ptr=B, ws_ptr=FP, idx_ptr=I32, out_ptr=B, N='i32'),
                  dict(KR=8, BN=MT.CFG['fin'], F32=False), dict(enable_fp_fusion=False)))
    cases.append(('fp4 quant', F._quant_kernel, dict(x_ptr=B, rs_ptr=FP, q_ptr=U8, s_ptr=F8, n='i32', K='i32'),
                  dict(TERMS=F.TERMS, BR=F.FCFG['q']), dict(enable_fp_fusion=False)))
    x = dict(q_ptr=U8, s_ptr=F8, rs_ptr=FP, w_ptr=U8, sw_ptr=F8, gs_ptr=FP, **tb, **st)
    for name, fn, ty in (('w13', F._w13_fp4, dict(x, T='i32', tok_ptr=I32, act_ptr=B, amax_ptr=FP, K='i32', NI='i32')),
                         ('w2', F._w2_fp4, dict(x, R='i32', y_ptr=B, K='i32', N='i32'))):
        bn, bk, nw, ns = F.FCFG[name]
        c = dict(TERMS=F.TERMS, BB=F.BB, BS=F.BS, BN=bn, BK=bk, F32=False)
        cases.append((f'fp4 {name}', fn, ty, dict(c, AMAX=False) if name == 'w13' else c,
                      dict(num_warps=nw, num_stages=ns)))
    bad = 0
    for name, fn, ty, consts, opt in cases:
        try:
            sig = {n: ('constexpr' if n in consts else ty[n]) for n in fn.arg_names}
            k = triton.compile(ASTSource(fn=fn, signature=sig, constexprs=consts), target=GPUTarget('cuda', 120, 32),
                               options=opt)
            mma = sorted(set(re.findall(r'mma\.sync\.aligned\.[\w\.:]+', k.asm['ptx'])))
            over = k.metadata.shared > SMEM
            wrong = name.startswith('fp4 w') and not any('mxf4nvf4' in m for m in mma)
            bad += over or wrong
            print(f'{name:18s} {"SHARED > %d" % SMEM if over else "NO FP4 MMA" if wrong else "ok"}  shared '
                  f'{k.metadata.shared}  warps {k.metadata.num_warps}  stages {opt.get("num_stages", "-")}  '
                  f'{" ".join(m.split("aligned.")[1] for m in mma)[:110]}')
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f'{name:18s} FAILED: {str(e).splitlines()[0] if str(e) else type(e).__name__}')
            if os.environ.get('V'):
                import traceback
                traceback.print_exc()
    print('COMPILE OK' if not bad else f'{bad} FAILED')
    return 1 if bad else 0


# ------------------------------------------------------------------ shared helpers (CPU report + GPU)
def route(m, L, u):
    """(sorted_e, tok, ws) exactly as Model.moe builds them."""
    import model as M
    rec = {}
    old, M.MOE_IMPL = M.MOE_IMPL, 'grab'
    m.moe_grouped = lambda L_, u_, se, tok, ws, impl=None: rec.update(args=(se, tok, ws)) or u_
    try:
        m.moe(L, u)
    finally:
        M.MOE_IMPL = old
        del m.moe_grouped
    return rec['args']


def capture(m, x0):
    """Run one forward (current INK_MOE path) and return [(L, u)] for every MoE call (48 visits)."""
    cap, orig = [], m.moe

    def hook(L, u, impl=None):
        cap.append((L, u.clone()))
        return orig(L, u, impl)
    m.moe = hook
    try:
        m.forward(x0, m.new_seq())
    finally:
        del m.moe
    return cap


def rel(a, ref):
    a, ref = a.float(), ref.float()
    return ((a - ref).abs().mean() / ref.abs().mean()).item()


def experts(W, L, full=False):
    P = f'layers.{L}.mlp.experts.'
    get = (lambda n: W.f.get_tensor('model.llm.' + P + n)) if full else (lambda n: W[P + n])
    a = [get(n) for n in ('w13_weight', 'w13_weight_scale', 'w2_weight', 'w2_weight_scale')]
    fp4 = L < 20
    gs = [W[P + w + '_weight_amax'] / W[P + w + '_weight_outer_scale_denominator'][0] if fp4 else None
          for w in ('w13', 'w2')]
    return a[0], a[1], gs[0], a[2], a[3], gs[1], fp4


def report():
    """CPU: fp4act error per NVFP4 layer on real activations (first visit of each layer 0-19)."""
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    from tokenizers import Tokenizer

    import chat
    import model as M
    import moe_fp4 as F
    import moe_triton as MT
    from engine import MODEL_DIR
    from gpu_check import PROMPTS
    torch.manual_seed(0)
    m = M.Model(MODEL_DIR, 'cpu', max_ctx=1024)
    tok = Tokenizer.from_file(os.path.join(MODEL_DIR, 'tokenizer.json'))
    ids = chat.render(tok, [{'role': 'user', 'content': PROMPTS[int(os.environ.get('INK_REPORT_PROMPT', 2))]}],
                      effort=0.9)[:int(os.environ.get('INK_REPORT_T', 48))]
    t0 = time.time()
    M.MOE_IMPL = 'torch'
    cap = capture(m, m.embed(ids))
    print(f'captured {len(cap)} visits of {len(ids)} tokens in {time.time() - t0:.0f} s', flush=True)
    seen, rows = set(), []
    print('layer | bf16 kernel (emulate) vs exact | fp4 1 term | fp4 2 terms | fp4 3 terms | 2-term / bf16'
          ' (mean rel vs exact; max abs / max|ref|)')
    for L, u in cap:
        if L >= 20 or L in seen:
            continue
        seen.add(L)
        ref = m.moe(L, u, 'exact').float()
        se, tk, ws = route(m, L, u)
        a = experts(m.W, L, full=True)
        bf = MT.emulate(u, se, tk, ws, *a).float()
        f1 = F.emulate(u, se, tk, ws, *a[:6], terms=1).float()
        f2 = F.emulate(u, se, tk, ws, *a[:6], terms=2).float()
        f3 = F.emulate(u, se, tk, ws, *a[:6], terms=3).float()
        mx = ref.abs().max().item()
        r = [rel(x, ref) for x in (bf, f1, f2, f3)]
        rows.append(r)
        print(f'{L:5d} | {r[0]:.2e} ({(bf - ref).abs().max() / mx:.1e}) | {r[1]:.2e} ({(f1 - ref).abs().max() / mx:.1e})'
              f' | {r[2]:.2e} ({(f2 - ref).abs().max() / mx:.1e}) | {r[3]:.2e} ({(f3 - ref).abs().max() / mx:.1e})'
              f' | {r[2] / r[0]:.0f}x', flush=True)
        del a
    avg = [sum(c) / len(rows) for c in zip(*rows)]
    print(f'mean  | {avg[0]:.2e} | {avg[1]:.2e} | {avg[2]:.2e} | {avg[3]:.2e} | {avg[2] / avg[0]:.0f}x   '
          f'({time.time() - t0:.0f} s)')


# ------------------------------------------------------------------ GPU window
def cuda_ms(fn, n=5):
    import torch
    fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return sorted(ts)[len(ts) // 2]


def chunk_ids(tok, T):
    import json
    path = os.path.join(HERE, 'oracle.json')
    ids = []
    if os.path.exists(path):
        for o in json.load(open(path))['teacher'].values():
            ids += o['ids']
    if len(ids) < T:
        import chat
        from gpu_check import PROMPTS
        while len(ids) < T:
            for p in PROMPTS:
                ids += chat.render(tok, [{'role': 'user', 'content': p}], effort=0.9)
    return ids[:T]


def gpu(sweep):
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    import torch
    from tokenizers import Tokenizer

    import model as M
    import moe_fp4 as F
    import moe_triton as MT
    from engine import MODEL_DIR
    from gpu_check import prompts, ulp_nudge
    T = 4096
    torch.manual_seed(0)
    m = M.Model(MODEL_DIR, 'cuda', max_ctx=2 * T)
    tok = Tokenizer.from_file(os.path.join(MODEL_DIR, 'tokenizer.json'))
    ok = True
    MT.V2 = MT.FP4ACT = False
    M.MOE_IMPL = 'triton'

    def paths(L, u):
        se, tk, ws = route(m, L, u)
        a = experts(m.W, L)
        return (u, se, tk, ws) + a

    print('== 1. per visit, real activations of a 4096-token chunk', flush=True)
    cap = capture(m, m.embed(chunk_ids(tok, T)))
    tot = {'old': 0., 'v2': 0., 'fp4': 0.}
    print('visit L  | ms old / v2 / fp4 | mean rel vs exact: old, v2, fp4 | v2 vs old: equal, rel | fp4 vs mirror',
          flush=True)
    for v, (L, u) in enumerate(cap):
        A = paths(L, u)
        ref = m.moe(L, u, 'exact').float()
        old = MT.grouped(*A)
        new = MT.grouped_v2(*A)
        new2 = MT.grouped_v2(*A)
        t_old, t_new = cuda_ms(lambda: MT.grouped(*A)), cuda_ms(lambda: MT.grouped_v2(*A))
        tot['old'] += t_old
        tot['v2'] += t_new
        r_old, r_new = rel(old, ref), rel(new, ref)
        det = torch.equal(new, new2)
        good = bool(torch.isfinite(new).all()) and det and r_new <= 1.05 * r_old + 1e-6 and rel(new, old) <= r_old
        line = (f'{v:2d} {L:2d} | {t_old:5.2f} / {t_new:5.2f} / ', f'{r_old:.2e}, {r_new:.2e}',
                f'{(new == old).float().mean():.3f}, {rel(new, old):.1e}{"" if det else " NONDETERMINISTIC"}')
        fl = '   -   '
        if L < 20:
            fq = F.grouped(*A[:10])
            t_fp4 = cuda_ms(lambda: F.grouped(*A[:10]))
            tot['fp4'] += t_fp4
            emu = F.emulate(*A[:10]).float()
            r_fp4, r_mir = rel(fq, ref), rel(fq, emu)
            q, s, rs = F.quant(u)
            q2, s2 = F.quant_ref(u, rs)
            qeq = (q == q2).float().mean().item()
            fgood = bool(torch.isfinite(fq).all()) and r_mir <= max(0.25 * r_fp4, 2e-4) and qeq >= 0.9999
            good &= fgood
            fl = f'{t_fp4:5.2f}'
            line = (line[0] + fl, line[1] + f', {r_fp4:.2e}', line[2] + f' | {r_mir:.1e} (quant eq {qeq:.5f})'
                    + ('' if fgood else ' FP4 BUG?'))
        else:
            line = (line[0] + fl, line[1] + ',    -    ', line[2])
            tot['fp4'] += t_new               # MXFP8 visits keep the bf16 path (v2) under INK_MOE_FP4ACT
        print(' | '.join(line) + ('' if good else '  <- FAIL'), flush=True)
        ok &= good
        if v in (0, 30, 44):                     # decode / mix shapes
            for n in (1, 8):
                B = paths(L, u[:n])
                d = rel(MT.grouped_v2(*B), MT.grouped(*B))
                print(f'   T={n}: v2 vs old rel {d:.1e}', flush=True)
                ok &= d <= 2e-3
        del ref, old, new, new2
    print(f'MoE per chunk (48 visits): old {tot["old"]:.0f} ms, v2 {tot["v2"]:.0f} ms, fp4act {tot["fp4"]:.0f} ms '
          f'(x{tot["old"] / tot["v2"]:.2f} / x{tot["old"] / tot["fp4"]:.2f})', flush=True)

    if sweep:
        print('\n== sweep (visits 0, 20, 44; output vs default config)', flush=True)
        pick = [cap[i] for i in (0, 20, 44)]
        A3 = [paths(L, u) for L, u in pick]
        base = [MT.grouped_v2(*A) for A in A3]
        cfg0, bb0 = dict(MT.CFG), (MT.BB, MT.BS)
        for bb in ((64, 64), (128, 64), (128, 32)):
            for bn13, bn2, bk, nw, ns in ((64, 128, 64, 8, 3), (64, 64, 64, 4, 3), (64, 128, 128, 8, 2),
                                          (32, 128, 64, 8, 4), (64, 128, 64, 8, 4)):
                MT.BB, MT.BS = bb
                for fp4 in (True, False):
                    MT.CFG[('w13', fp4)], MT.CFG[('w2', fp4)] = (bn13, bk, nw, ns), (bn2, bk, nw, ns)
                try:
                    ts = [cuda_ms(lambda: MT.grouped_v2(*A), 3) for A in A3]
                    d = max(rel(MT.grouped_v2(*A), b) for A, b in zip(A3, base))
                    print(f'v2 BB/BS {bb} bn {bn13}/{bn2} bk {bk} w{nw} s{ns}: '
                          + ' '.join(f'{t:.2f}' for t in ts) + f' ms, rel vs default {d:.1e}', flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f'v2 {bb} {bn13}/{bn2} {bk} {nw} {ns}: {type(e).__name__} {str(e)[:80]}', flush=True)
        MT.CFG.update(cfg0)
        MT.BB, MT.BS = bb0
        f0 = dict(F.FCFG)
        A2 = [A for A, (L, _) in zip(A3, pick) if L < 20]
        fb = [F.grouped(*A[:10]) for A in A2]
        for w13c, w2c in (((64, 128, 8, 3), (128, 128, 8, 3)), ((64, 128, 4, 3), (64, 128, 4, 3)),
                          ((64, 256, 8, 2), (128, 256, 8, 2)), ((32, 128, 4, 4), (64, 128, 4, 4)),
                          ((64, 64, 8, 4), (128, 64, 8, 4))):
            F.FCFG['w13'], F.FCFG['w2'] = w13c, w2c
            try:
                ts = [cuda_ms(lambda: F.grouped(*A[:10]), 3) for A in A2]
                d = max(rel(F.grouped(*A[:10]), b) for A, b in zip(A2, fb))
                print(f'fp4 w13 {w13c} w2 {w2c}: ' + ' '.join(f'{t:.2f}' for t in ts) + f' ms, rel vs default {d:.1e}',
                      flush=True)
            except Exception as e:  # noqa: BLE001
                print(f'fp4 {w13c} {w2c}: {type(e).__name__} {str(e)[:80]}', flush=True)
        F.FCFG.update(f0)
    del cap

    print('\n== 2. whole 4096-token chunk', flush=True)
    x = m.embed(chunk_ids(tok, T))
    for name, v2, f4 in (('old', False, False), ('v2', True, False), ('fp4act', True, True)):
        MT.V2, MT.FP4ACT = v2, f4
        m.forward(x, m.new_seq())
        torch.cuda.synchronize()
        ts = []
        for _ in range(3):
            t0 = time.time()
            m.forward(x, m.new_seq())
            torch.cuda.synchronize()
            ts.append(time.time() - t0)
        t = sorted(ts)[1]
        print(f'{name:7s}: {t:.3f} s per chunk, {T / t:.0f} tok/s', flush=True)
    MT.V2 = MT.FP4ACT = False

    print('\n== 3. end to end: mean KL(old || path) and top-1 (fp4act on every prompt: INK_MOE_FP4ACT_MIN=1)', flush=True)
    MT.FP4ACT_MIN = 1
    runs = {'old rerun': (False, False, False), 'old ulp-nudged emb': (False, False, True),
            'v2': (True, False, False), 'fp4act': (True, True, False)}
    tot = {k: [0., 0., 0] for k in runs}
    for pname, ids in prompts(tok):
        x0 = m.embed(ids)

        def lp(x, v2, f4):
            MT.V2, MT.FP4ACT = v2, f4
            try:
                return torch.log_softmax(m.logits(m.forward(x, m.new_seq())), -1)
            finally:
                MT.V2 = MT.FP4ACT = False
        ref = lp(x0, False, False)
        p_ref, top = ref.exp(), ref.argmax(-1)
        line = []
        for k, (v2, f4, nudge) in runs.items():
            got = lp(ulp_nudge(x0) if nudge else x0, v2, f4)
            kl = (p_ref * (ref - got)).sum(-1)
            agree = (top == got.argmax(-1)).float()
            tot[k][0] += kl.sum().item()
            tot[k][1] += agree.sum().item()
            tot[k][2] += len(ids)
            line.append(f'{k} KL {kl.mean():.2e} top1 {agree.mean():.3f}')
        print(f'{pname} ({len(ids)} tok): ' + '; '.join(line), flush=True)
    kl = {k: v[0] / v[2] for k, v in tot.items()}
    for k, v in tot.items():
        print(f'  overall {k:20s}: mean KL {kl[k]:.3e}, top-1 {v[1] / v[2]:.4f}')
    noise = max(kl['old rerun'], kl['old ulp-nudged emb'])
    good = kl['v2'] <= 2 * noise
    print(f'v2 KL {kl["v2"]:.3e} vs noise {noise:.3e} (limit 2x): {"OK" if good else "TOO BIG"}')
    ok &= good
    f4ok = kl['fp4act'] <= 2 * kl['old ulp-nudged emb']
    print(f'fp4act KL {kl["fp4act"]:.3e} vs the 1-ulp baseline {kl["old ulp-nudged emb"]:.3e} (limit 2x): '
          f'{"WITHIN noise" if f4ok else "OUTSIDE noise"} (verdict only; INK_MOE_FP4ACT stays off by default)')
    print('PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    if '--compile' in sys.argv:
        sys.exit(compile_check())
    if '--report' in sys.argv:
        sys.exit(report())
    sys.exit(gpu('--sweep' in sys.argv))
