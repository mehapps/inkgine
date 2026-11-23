"""Round-4 prefill MoE flags: INK_MOE_ROUTE (fused routed-row sort + block tables), INK_MOE_AMAX (activation row amax
from the w13 epilogue, one-pass u amax), tail/big block sizes and tile configs (INK_MOE_FP4_BS/BB/CFG, INK_MOE_BS/BB,
INK_MOE_MX_CFG). All are meant to be exact: same numbers as the current defaults.

  --compile  CPU, no GPU: sm_120 offline compile of the route kernels, the AMAX w13 kernel and every sweep config
             (shared memory vs 101376 B, registers / spills from ptxas). Over-limit configs are skipped on the GPU.
  (default)  GPU window, defaults on (INK_MOE=triton, V2, FP4ACT; glue flags pass through from the environment):
             1. real activations of a 4096-token chunk (48 visits): Model.moe (routing included) per variant, output
                torch.equal to the default path, ms per visit, totals per chunk.
             2. tile sweep (coordinate descent: fp4 w13 cfg, fp4 w2 cfg, fp4 blocks, MXFP8 w13 / w2 / blocks) on 3 fp4 +
                2 MXFP8 visits; a config must be bit-equal and >= 3% faster to win. Winners -> --best FILE (env line).
             3. end to end: mean KL(default || variant) + top-1 + bit-equal logits on gpu_check's prompts and a long
                2 x 4096 real-text prompt (last 1024 rows), noise baselines rerun / embeddings +-1 ulp, gate 2x 1-ulp.
             Exit 1 if an exact variant differs from the default anywhere in 1 or a KL is outside the gate.
Usage: ~/llm/engine/swap_test.sh 30 python tests/gpu_moe4.py [--best ~/llm/logs/ink-moe4-best.env]"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
SMEM = 101376
# (BN, BK, num_warps, num_stages); first entry = current default
FP4_W13 = [(64, 128, 8, 3), (64, 128, 4, 3), (128, 128, 8, 2), (64, 256, 8, 2), (32, 128, 4, 4), (64, 64, 8, 4),
           (128, 64, 8, 3), (64, 128, 8, 4)]
FP4_W2 = [(128, 128, 8, 3), (64, 128, 4, 3), (128, 256, 8, 2), (256, 128, 8, 2), (64, 128, 8, 3), (128, 64, 8, 4),
          (256, 64, 8, 3)]
MX_W13 = [(64, 128, 8, 2), (64, 64, 8, 3), (32, 128, 4, 3), (64, 128, 8, 3), (128, 64, 8, 2), (32, 64, 4, 4)]
MX_W2 = [(128, 128, 8, 2), (64, 128, 4, 3), (128, 64, 8, 3), (256, 64, 8, 2), (64, 64, 4, 4)]
BLOCKS = [(128, 64), (128, 32), (64, 64), (64, 32), (256, 64)]   # (BB, BS)


def compile_check():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ.pop('TRITON_INTERPRET', None)
    os.environ.setdefault('TRITON_DEFAULT_FP_FUSION', '1')
    import re

    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    import moe_fp4 as F
    import moe_route as R
    import moe_triton as MT
    from compile_sm120 import ptxas_info
    B, I32, I64, U8, F8, FP = '*bf16', '*i32', '*i64', '*u8', '*fp8e4nv', '*fp32'
    st = dict(lwe='i32', lwr='i32', lse='i32', lsr='i32')
    tb = dict(be_ptr=I32, b0_ptr=I32, b1_ptr=I32, bb_ptr=I32)
    x = dict(q_ptr=U8, s_ptr=F8, rs_ptr=FP, w_ptr=U8, sw_ptr=F8, gs_ptr=FP, **tb, **st)
    t13 = dict(x, T='i32', tok_ptr=I32, act_ptr=B, amax_ptr=FP, K='i32', NI='i32')
    t2 = dict(x, R='i32', y_ptr=B, K='i32', N='i32')
    cases = [('route count', R._count, dict(idx_ptr=I64, cnt_ptr=I32, T='i32'), dict(K=8, BT=R.BT), {}),
             ('route scatter', R._scatter, dict(idx_ptr=I64, w_ptr=FP, start_ptr=I32, fill_ptr=I32, tok_ptr=I32,
                                                ws_ptr=FP, fidx_ptr=I32, T='i32'), dict(K=8, BT=R.BT), {})]
    for bb, bs in BLOCKS:
        cases.append((f'route tables {bb}/{bs}', R._tables, dict(cnt_ptr=I32, start_ptr=I32, tab_ptr=I32, n_blk='i32'),
                      dict(E=128, BB=bb, BS=bs, BLK=R.BLK), {}))

    def fp4(name, c, bb=F.BB, bs=F.BS, amax=False):
        bn, bk, nw, ns = c
        k = dict(TERMS=F.TERMS, BB=bb, BS=bs, BN=bn, BK=bk, F32=False)
        fn, ty = (F._w13_fp4, t13) if name == 'w13' else (F._w2_fp4, t2)
        cases.append((f'fp4 {name} {c} {bb}/{bs}' + (' AMAX' if amax else ''), fn, ty,
                      dict(k, AMAX=amax) if name == 'w13' else k, dict(num_warps=nw, num_stages=ns)))

    def mx(name, c, bb=MT.BB, bs=MT.BS):
        bn, bk, nw, ns = c
        ty = dict(u_ptr=B, tok_ptr=I32, act_ptr=B, K='i32', NI='i32') if name == 'w13' else \
            dict(act_ptr=B, y_ptr=B, K='i32', N='i32')
        cases.append((f'mx {name} {c} {bb}/{bs}', MT._w13_v2 if name == 'w13' else MT._w2_v2,
                      dict(ty, **tb, **st, w_ptr=F8, s_ptr=U8, gs_ptr=FP),
                      dict(FP4=False, BB=bb, BS=bs, BN=bn, BK=bk, F32=False), dict(num_warps=nw, num_stages=ns)))
    fp4('w13', FP4_W13[0], amax=True)
    for c in FP4_W13:
        fp4('w13', c)
    for c in FP4_W2:
        fp4('w2', c)
    for bb, bs in BLOCKS[1:]:
        fp4('w13', FP4_W13[0], bb, bs)
        fp4('w2', FP4_W2[0], bb, bs)
    for c in MX_W13:
        mx('w13', c)
    for c in MX_W2:
        mx('w2', c)
    for bb, bs in BLOCKS[1:]:
        mx('w13', MX_W13[0], bb, bs)
        mx('w2', MX_W2[0], bb, bs)
    bad = 0
    for name, fn, ty, consts, opt in cases:
        try:
            sig = {n: ('constexpr' if n in consts else ty[n]) for n in fn.arg_names}
            k = triton.compile(ASTSource(fn=fn, signature=sig, constexprs=consts), target=GPUTarget('cuda', 120, 32),
                               options=opt)
            over = k.metadata.shared > SMEM
            mma = set(re.findall(r'mma\.sync\.aligned\.[\w\.:]+', k.asm['ptx']))
            wrong = name.startswith('fp4 w') and not any('mxf4nvf4' in m for m in mma)
            # only the engine's kernels (route, AMAX, defaults) must pass; sweep configs over the limit are skipped
            must = name.startswith('route') or 'AMAX' in name
            bad += must and (over or wrong)
            print(f'{name:36s} {"SHARED > LIMIT (skip)" if over else "NO FP4 MMA" if wrong else "ok"}  shared '
                  f'{k.metadata.shared}  {ptxas_info(k)}', flush=True)
        except Exception as e:  # noqa: BLE001
            bad += name.startswith('route') or 'AMAX' in name
            print(f'{name:36s} FAILED: {str(e).splitlines()[0] if str(e) else type(e).__name__}', flush=True)
    print('COMPILE OK' if not bad else f'{bad} FAILED')
    return 1 if bad else 0


def gpu(best_path):
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    import torch
    from tokenizers import Tokenizer

    import model as M
    import moe_fp4 as F
    import moe_triton as MT
    from engine import MODEL_DIR
    from gpu_check import prompts, ulp_nudge
    from gpu_moe import capture, chunk_ids, cuda_ms
    from prof_prefill import text_ids
    print('flags:', ' '.join(f'{k}={v}' for k, v in sorted(os.environ.items()) if k.startswith('INK_')), flush=True)
    assert M.MOE_IMPL == 'triton' and MT.V2 and MT.FP4ACT, 'needs INK_MOE=triton INK_MOE_V2=1 INK_MOE_FP4ACT=1'
    torch.manual_seed(0)
    m = M.Model(MODEL_DIR, 'cuda', max_ctx=1 << 15)
    tok = Tokenizer.from_file(os.path.join(MODEL_DIR, 'tokenizer.json'))
    d0 = dict(route=M.ROUTE[0], amax=F.AMAX, fbb=F.BB, fbs=F.BS, vbb=MT.BB, vbs=MT.BS, fcfg=dict(F.FCFG),
              vcfg=dict(MT.CFG))

    def setv(route=None, amax=None, fbb=None, fbs=None, vbb=None, vbs=None, fcfg=None, vcfg=None):
        """None = the default (state at start)."""
        M.ROUTE[0] = d0['route'] if route is None else route
        F.AMAX = d0['amax'] if amax is None else amax
        F.BB, F.BS = fbb or d0['fbb'], fbs or d0['fbs']
        MT.BB, MT.BS = vbb or d0['vbb'], vbs or d0['vbs']
        F.FCFG.clear(), F.FCFG.update({**d0['fcfg'], **(fcfg or {})})
        MT.CFG.clear(), MT.CFG.update({**d0['vcfg'], **(vcfg or {})})

    def merge(a, b):
        """setv kwargs b over a (tile dicts merged per key)."""
        return {**a, **b, **{k: {**a.get(k, {}), **b[k]} for k in ('fcfg', 'vcfg') if k in b}}
    ok = True
    run = lambda L, u: m.moe(L, u, 'triton')  # noqa: E731  (routing included)
    print('\n== 1. per visit, real activations of a 4096-token chunk (Model.moe incl. routing)', flush=True)
    cap = capture(m, m.embed(chunk_ids(tok, 4096)))
    var = {'route': dict(route=True), 'amax': dict(amax=True), 'fp4 bs32': dict(fbs=32), 'mx bs32': dict(vbs=32),
           'route+amax': dict(route=True, amax=True), 'route+amax+bs32': dict(route=True, amax=True, fbs=32, vbs=32)}
    tot = {k: 0. for k in ['default', *var]}
    neq = {k: 0 for k in var}
    for v, (L, u) in enumerate(cap):
        setv()
        base = run(L, u)
        t0 = cuda_ms(lambda: run(L, u))
        tot['default'] += t0
        parts = [f'{v:2d} L{L:2d} default {t0:5.2f}']
        for k, kw in var.items():
            setv(**kw)
            eq = torch.equal(run(L, u), base) and torch.equal(run(L, u), base)   # twice: route uses atomics
            t = cuda_ms(lambda: run(L, u))
            tot[k] += t
            neq[k] += not eq
            parts.append(f'{k} {t:5.2f}{"" if eq else " DIFF"}')
        print(' | '.join(parts), flush=True)
    setv()
    for k, t in tot.items():
        print(f'  MoE per chunk {k:16s}: {t:6.1f} ms ({t - tot["default"]:+.1f})' +
              ('' if k == 'default' else f', visits differing from default: {neq[k]}'), flush=True)
    for k in ('route', 'amax', 'route+amax'):
        ok &= neq[k] == 0
    bs_exact = neq['fp4 bs32'] == 0 and neq['mx bs32'] == 0
    print(f'block size 32 exact on the GPU: {bs_exact}', flush=True)

    print('\n== 2. tile sweep (bit-equal and >= 3% faster to win)', flush=True)
    fv = [cap[i] for i in (0, 20, 40)]
    mv = [cap[i] for i in (44, 46)]
    bases = {id(u): run(L, u) for L, u in fv + mv}
    win = {}

    def tsum(vis, **kw):
        setv(**merge(win, kw))
        try:
            t = sum(cuda_ms(lambda: run(L, u), 3) for L, u in vis)
            eq = all(torch.equal(run(L, u), bases[id(u)]) for L, u in vis)
            return t, eq
        except Exception as e:  # noqa: BLE001 (shared memory over the limit, compile failure)
            print(f'    {kw}: {type(e).__name__} {str(e)[:80]}', flush=True)
            return float('inf'), False
        finally:
            setv()

    def sweep(name, vis, opts):
        t_def, _ = tsum(vis)
        bt, bk = t_def, None
        for kw in opts:
            t, eq = tsum(vis, **kw)
            print(f'  {name} {kw}: {t:.2f} ms (default {t_def:.2f}){"" if eq else " NOT EQUAL"}', flush=True)
            if eq and t < 0.97 * bt:
                bt, bk = t, kw
        if bk:
            win.update(merge(win, bk))
        print(f'  -> {name}: {bk or "default"} ({bt:.2f} vs {t_def:.2f} ms)', flush=True)
    sweep('fp4 w13', fv, [dict(fcfg={'w13': c}) for c in FP4_W13[1:]])
    sweep('fp4 w2', fv, [dict(fcfg={'w2': c}) for c in FP4_W2[1:]])
    sweep('fp4 blocks', fv, [dict(fbb=bb, fbs=bs) for bb, bs in BLOCKS[1:]])
    sweep('mx w13', mv, [dict(vcfg={('w13', False): c}) for c in MX_W13[1:]])
    sweep('mx w2', mv, [dict(vcfg={('w2', False): c}) for c in MX_W2[1:]])
    sweep('mx blocks', mv, [dict(vbb=bb, vbs=bs) for bb, bs in BLOCKS[1:]])
    env = []
    f, vc = win.get('fcfg', {}), win.get('vcfg', {})
    if f:
        env.append('INK_MOE_FP4_CFG=' + ';'.join(f'{k}={",".join(map(str, c))}' for k, c in f.items()))
    if vc:
        env.append('INK_MOE_MX_CFG=' + ';'.join(f'{k[0]}={",".join(map(str, c))}' for k, c in vc.items()))
    env += [f'{n}={win[k]}' for k, n in (('fbb', 'INK_MOE_FP4_BB'), ('fbs', 'INK_MOE_FP4_BS'), ('vbb', 'INK_MOE_BB'),
                                          ('vbs', 'INK_MOE_BS')) if k in win]
    print(f'tile winners: {" ".join(env) or "none (defaults stay)"}', flush=True)
    if best_path:
        open(os.path.expanduser(best_path), 'w').write(' '.join(env) + '\n')
    del cap, bases

    print('\n== 3. end to end: mean KL(default || variant), top-1, bit-equal logits', flush=True)

    def lp(x0, kw, chunk=None, rows=None):
        setv(**kw)
        try:
            seq = m.new_seq()
            for a in range(0, x0.shape[0], chunk or x0.shape[0]):
                o = m.forward(x0[a:a + (chunk or x0.shape[0])], seq)
            return torch.log_softmax(m.logits(o[-(rows or o.shape[0]):]), -1)
        finally:
            setv()
    runs = {'rerun': {}, 'ulp-nudged emb': {}, 'route': dict(route=True), 'amax': dict(amax=True),
            'route+amax': dict(route=True, amax=True)}
    if win:
        runs['tiles'] = dict(win)
        runs['route+amax+tiles'] = merge(win, dict(route=True, amax=True))
    sets = [(n, ids, None, None) for n, ids in prompts(tok)] + [('long 2x4096 (last 1024 rows)', text_ids(8192), 4096,
                                                                 1024)]
    tot = {k: [0., 0., 0, 0.] for k in runs}
    for name, ids, chunk, rows in sets:
        x0 = m.embed(ids)
        ref = lp(x0, {}, chunk, rows)
        p, top = ref.exp(), ref.argmax(-1)
        line = []
        for k, kw in runs.items():
            got = lp(ulp_nudge(x0) if k == 'ulp-nudged emb' else x0, kw, chunk, rows)
            kl = (p * (ref - got)).sum(-1)
            n = kl.shape[0]
            tot[k][0] += kl.sum().item()
            tot[k][1] += (top == got.argmax(-1)).float().sum().item()
            tot[k][2] += n
            tot[k][3] += torch.equal(got, ref) * n
            line.append(f'{k} KL {kl.mean():.2e}{" ==" if torch.equal(got, ref) else ""}')
        print(f'{name} ({len(ids)} tok): ' + '; '.join(line), flush=True)
        del ref, p
    kl = {k: v[0] / v[2] for k, v in tot.items()}
    for k, v in tot.items():
        print(f'  overall {k:18s}: mean KL {kl[k]:.3e}, top-1 {v[1] / v[2]:.4f}, bit-equal rows {v[3] / v[2]:.4f}')
    for k in list(runs)[2:]:
        good = kl[k] <= 2 * kl['ulp-nudged emb']
        print(f'{k} KL {kl[k]:.3e} vs 1-ulp {kl["ulp-nudged emb"]:.3e} (limit 2x): '
              f'{"WITHIN noise" if good else "OUTSIDE noise"}{"; bit-identical" if tot[k][3] == tot[k][2] else ""}')
        ok &= good
    print('PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    if '--compile' in sys.argv:
        sys.exit(compile_check())
    bp = sys.argv[sys.argv.index('--best') + 1] if '--best' in sys.argv else None
    sys.exit(gpu(bp))
