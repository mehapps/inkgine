"""GPU window: attention time of one 4096-token prefill chunk at contexts 4k/64k/256k/1M, torch path (flex far + torch
near) vs INK_PREFILL_ATTN=far (Triton far only) vs =triton (Triton far + near), and a check that the outputs match.
No weights and no real prompt: Model.attn_core on random Q/K/V/R/sink, the KV pool filled with random 4-bit codes and
plausible e4m3 scales, random bf16 rings. A chunk = 8 full visits + 40 sliding visits (timed one of each, x8 / x40).
Exit 1 if any output is non-finite, the near kernel (triton vs far-only output) exceeds mean rel 1e-2 / max rel 5e-2
(CPU tests: ~2e-3 / 7e-3), or the Triton far is further from an fp32 reference (32 sampled rows) than 1.5x the torch
far's own error (+1e-3 mean / 5e-3 max), or its lse is off by >= 1e-2.
Run (window): ~/llm/engine/swap_test.sh 20 python tests/gpu_prefill.py [--ctx 4096,65536] [--sweep]
     --sweep: also time far/near tile configs at the largest context (prints the best; set INK_PA_* from it).
     --compile: CPU only, no GPU: offline sm_120 compile of the kernels + ptxas registers/spills.
     --cpu: CPU smoke test of this script (Triton interpreter, T=256, contexts 512/3000; times are meaningless)."""
import os
import statistics
import subprocess
import sys
import tempfile
import time

COMPILE = '--compile' in sys.argv
if COMPILE:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ.pop('TRITON_INTERPRET', None)
    os.environ.setdefault('TRITON_DEFAULT_FP_FUSION', '1')
CPU = '--cpu' in sys.argv
if CPU:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['TRITON_INTERPRET'] = '1'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402

import model as M  # noqa: E402
import prefill_attn as PA  # noqa: E402

T = 256 if CPU else 4096
CTX = [512, 3000] if CPU else [4096, 65536, 262144, 1040000]
FAR_CFGS = [(128, 64, 8, 2), (128, 32, 8, 2), (128, 64, 8, 3), (64, 64, 4, 2), (128, 128, 8, 2)]  # BM, BN, warps, stages
PROGS = [680, 1360, 2720]
NEAR_CFGS = [(64, 32, 4), (128, 32, 8), (64, 64, 4), (128, 64, 8)]  # BM, BN, warps


def compile_check():
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    F, B, I64, U8, F8 = '*fp32', '*bf16', '*i64', '*u8', '*fp8e4nv'
    far = dict(q_ptr=B, c0=U8, c1=U8, s0=F8, s1=F8, tbl=I64, o_ptr=F, l_ptr=F, p0='i32', T='i32', kend='i32',
               split='i32', scale='fp32', meta_ptr=B, thr='fp32', st_ptr=I64)
    near = dict(q_ptr=B, k_ptr=B, v_ptr=B, r_ptr=B, pj_ptr=B, sink_ptr=F, fo_ptr=F, fl_ptr=F, o_ptr=F, p0='i32',
                T='i32', k0='i32', scale='fp32', g_ptr=B, lg='i32')
    ptxas = os.path.join(os.path.dirname(triton.__file__), 'backends/nvidia/bin/ptxas-blackwell')
    cases = [(f'far BM{bm} BN{bn}', PA._far_kernel, far, dict(PG=256, BM=bm, BN=bn, F32=False, FP8=False, SKIP=False, STATS=False), w, st)
             for bm, bn, w, st in FAR_CFGS]
    cases += [(f'far FP8 BM{bm} BN{bn}', PA._far_kernel, far, dict(PG=256, BM=bm, BN=bn, F32=False, FP8=True, SKIP=False, STATS=False), w, st)
              for bm, bn, w, st in FAR_CFGS]
    cases += [(f'far SKIP STATS={sx} BM{bm} BN{bn}', PA._far_kernel, far,   # INK_FARSKIP
               dict(PG=256, BM=bm, BN=bn, F32=False, FP8=False, SKIP=True, STATS=sx), w, st)
              for bm, bn, w, st in FAR_CFGS for sx in (False, True)]
    cases += [('far tile_meta BN64', PA._meta_kernel, dict(c_ptr=U8, s_ptr=F8, tbl=I64, meta_ptr=B), dict(PG=256, BN=64),
               4, 1)]
    cases += [(f'near BM{bm} BN{bn} far={fa} tab={tab}', PA._near_kernel, near,
               dict(TAB=tab, FAR=fa, BM=bm, BN=bn, BW=triton.next_power_of_2(bm // 3 + 2 + bn), F32=False, GATE=False), w, 2)
              for bm, bn, w in NEAR_CFGS for fa in (False, True) for tab in (False, True)]
    cases += [(f'near GATE BM{bm} BN{bn} far={fa}', PA._near_kernel, dict(near, o_ptr=B),   # INK_GLUE2
               dict(TAB=False, FAR=fa, BM=bm, BN=bn, BW=triton.next_power_of_2(bm // 3 + 2 + bn), F32=False, GATE=True), w, 2)
              for bm, bn, w in NEAR_CFGS[:1] for fa in (False, True)]
    bad = 0
    for name, fn, types, consts, w, st in cases:
        try:
            sig = {n: 'constexpr' if n in consts else types[n] for n in fn.arg_names}
            k = triton.compile(ASTSource(fn=fn, signature=sig, constexprs=consts), target=GPUTarget('cuda', 120, 32),
                               options=dict(num_warps=w, num_stages=st))
            with tempfile.NamedTemporaryFile('w', suffix='.ptx') as f:
                f.write(k.asm['ptx']); f.flush()
                r = subprocess.run([ptxas, '-v', '-arch=sm_120a', f.name, '-o', '/dev/null'], capture_output=True,
                                   text=True)
            info = ' '.join(x.split('info    : ')[-1].strip() for x in r.stderr.splitlines() if 'spill' in x or 'registers' in x)
            print(f'{name:36s} warps {w} stages {st} smem {k.metadata.shared:6d} | {info}')
            bad += k.metadata.shared > 101376                                 # sm_120 per-block limit
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f'{name:36s} FAILED: {str(e).splitlines()[0] if str(e) else type(e).__name__}')
    print('COMPILE OK' if not bad else f'{bad} FAILED')
    return 1 if bad else 0


def fill(pool):
    pool.c.random_(0, 256)
    for j, (lo, hi) in enumerate(((0.3, 0.6), (0.2, 0.45))):        # K: norm ~ the 22.6 clip; V: rms ~ 1
        for v in range(pool.s.shape[1]):
            pool.s[j, v].copy_(torch.empty(pool.s[j, v].shape, device=pool.dev).uniform_(lo, hi))
    pool.c[:, :, 0].zero_(); pool.s[:, :, 0].zero_()                  # zero page stays zero


def inputs(ctx, dev, g):
    r = lambda *s: torch.randn(*s, device=dev, generator=g)
    tau = (1 + 0.1 * torch.log(torch.clamp((torch.arange(ctx, ctx + T, device=dev) + 1).float() / 8192, min=1)))
    Q = (M.l2clip(r(T, M.NQ, M.HD) * 3, M.CLIP_QK) * tau[:, None, None]).bfloat16()
    K, V = M.l2clip(r(T, M.NKV, M.HD), M.CLIP_QK).bfloat16(), M.rms(r(T, M.NKV, M.HD)).bfloat16()
    R = (M.l2clip(r(T, M.NQ, 16), M.CLIP_REL) * tau[:, None, None]).bfloat16()
    proj = (r(16, M.WIN) * 0.5).bfloat16()
    nr = min(ctx, M.WIN - 1)
    ring = (M.l2clip(r(nr, M.NKV, M.HD), M.CLIP_QK).bfloat16(), M.rms(r(nr, M.NKV, M.HD)).bfloat16())
    return Q, K, V, (R, proj), r(T, M.NQ) * 3, ring


def run(m, seq, ring, ctx, args, full, impl, n):
    """attn_core n times from the same state; -> (output of the last run, median ms)."""
    M.PREFILL_ATTN = impl
    ts = []
    for _ in range(1 if CPU else n):
        seq.ring[0], seq.pos = ring, ctx
        if CPU:
            t0 = time.perf_counter()
            out = m.attn_core(*args, seq, 0, full)
            ts.append((time.perf_counter() - t0) * 1e3)
            continue
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        out = m.attn_core(*args, seq, 0, full)
        b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return out, statistics.median(ts)


def rel(a, b):
    d = (a - b).abs()
    return (d.max() / b.abs().max()).item(), (d.mean() / b.abs().mean()).item()


def far_ref(Q, kv, qpos, kend, blk=65536):
    """fp32 far attention (keys [0, kend), k <= q - 1024) for a few query rows, blockwise over the pool."""
    n = Q.shape[0]
    q = Q.float().view(n, M.NKV, 3, M.HD)
    o = torch.zeros(n, M.NKV, 3, M.HD, device=Q.device)
    lse = torch.full((n, M.NKV, 3), float('-inf'), device=Q.device)
    for b0 in range(0, kend, blk):
        K, V = (x.float() for x in kv.get(0, b0, min(kend, b0 + blk)))
        S = torch.einsum('tkgd,skd->tkgs', q, K) * M.HD ** -0.5
        ok = torch.arange(b0, b0 + K.shape[0], device=Q.device)[None] <= qpos[:, None] - M.WIN
        S = S.masked_fill(~ok[:, None, None], float('-inf'))
        new = torch.logaddexp(lse, torch.logsumexp(S, -1))
        o = o * torch.nan_to_num(torch.exp(lse - new))[..., None] + torch.einsum(
            'tkgs,skd->tkgd', torch.nan_to_num(torch.exp(S - new[..., None])), V)
        lse = new
    return o.reshape(n, M.NQ, M.HD), lse.reshape(n, M.NQ)


def far_check(m, kv, Q, ctx):
    """Torch far (flex on cuda) and the Triton far vs fp32 on 32 sampled rows -> (torch err, triton err, lse err, ok)."""
    kend = ctx + T - M.WIN
    if kend <= 0:
        return None
    qpos = torch.arange(ctx, ctx + T, device=Q.device)
    outs = []
    for impl in ('torch', 'far'):
        M.PREFILL_ATTN = impl
        outs.append(m.far(Q, kv, 0, qpos, kend, ctx))
    idx = torch.linspace(0, T - 1, 32, device=Q.device).long()
    o0, l0 = far_ref(Q[idx], kv, qpos[idx], kend)
    fin = torch.isfinite(l0)
    (ot, lt), (ox, lx) = ((o[idx][fin], l[idx][fin]) for o, l in outs)
    et, ex = rel(ot, o0[fin]), rel(ox, o0[fin])
    le = (lx - l0[fin]).abs().max().item()
    ok = ex[1] <= 1.5 * et[1] + 1e-3 and ex[0] <= 1.5 * et[0] + 5e-3 and le < 1e-2 \
        and torch.equal(torch.isfinite(outs[1][1][idx]), fin)
    return et, ex, le, ok


def main():
    from kvpool import KVPool
    ctxs = [int(x) for x in sys.argv[sys.argv.index('--ctx') + 1].split(',')] if '--ctx' in sys.argv else CTX
    dev = torch.device('cpu' if CPU else 'cuda')
    pool = KVPool(max(ctxs) + T + 256, dev, 'gpu-prefill', cache_dir=None)
    fill(pool)
    m = object.__new__(M.Model)
    m.slot, m.dev = {0: 0}, dev
    g = torch.Generator(device=dev).manual_seed(0)
    print(f'{"ctx":>8s} {"visit":7s} {"torch ms":>9s} {"far ms":>8s} {"triton ms":>9s}  errors: max/mean rel (near = '
          f'triton vs far-only output; far = vs fp32 on sampled rows)')
    bad, rows = 0, []
    for ctx in ctxs:
        tbl = pool.new_table()
        assert pool.reserve(tbl, ctx + T)
        seq = M.Seq(dev, tbl)
        Q, K, V, table, sink, ring = inputs(ctx, dev, g)
        args = (Q, K, V, table, sink)
        tm = {}
        for full in (True, False):
            outs = {}
            for impl in ('torch', 'far', 'triton') if full else ('torch', 'triton'):
                run(m, seq, ring, ctx, args, full, impl, 2)              # warm-up / compile
                outs[impl], tm[impl, full] = run(m, seq, ring, ctx, args, full, impl, 3)
            if not full:
                tm['far', False] = tm['torch', False]                   # far-only path = torch near on sliding visits
                outs['far'] = outs['torch']
            fe = far_check(m, tbl, Q, ctx) if full else None
            en = rel(outs['triton'], outs['far'])                       # the near kernel alone (same far part)
            ok = all(torch.isfinite(o).all().item() for o in outs.values()) and en[0] < 5e-2 and en[1] < 1e-2 \
                and (fe is None or fe[3])
            bad += not ok
            far_s = '' if fe is None else (f'  far vs fp32 (32 rows): torch {fe[0][0]:.1e}/{fe[0][1]:.1e} triton '
                                           f'{fe[1][0]:.1e}/{fe[1][1]:.1e} lse {fe[2]:.0e}')
            print(f'{ctx:8d} {"full" if full else "sliding":7s} {tm["torch", full]:9.2f} {tm["far", full]:8.2f} '
                  f'{tm["triton", full]:9.2f}  near {en[0]:.1e}/{en[1]:.1e}{far_s}{"" if ok else "   MISMATCH"}')
        chunk = {i: 8 * tm[i, True] + 40 * tm[i, False] for i in ('torch', 'far', 'triton')}
        rows.append((ctx, chunk))
        pool.release(tbl)
        del seq, tbl
    print(f'\nattention per {T}-token chunk (8 full + 40 sliding visits):')
    for ctx, c in rows:
        print(f'  ctx {ctx:8d}: torch {c["torch"] / 1e3:6.3f} s | far {c["far"] / 1e3:6.3f} s | triton '
              f'{c["triton"] / 1e3:6.3f} s  (x{c["torch"] / c["triton"]:.2f})')
    if '--fp8' in sys.argv:
        fp8_far(m, pool, ctxs, dev, g)
    if '--sweep' in sys.argv:
        sweep(m, pool, max(ctxs), dev, g)
    print('PASS' if not bad else f'FAIL ({bad} mismatches)')
    return 1 if bad else 0


def fp8_far(m, pool, ctxs, dev, g):
    """INK_PA_FP8: far kernel bf16 vs fp8 (time per full visit, error vs fp32 on 32 rows) next to flex. Report only."""
    print(f'\nfar kernel bf16 vs FP8 (ms per full visit; max/mean rel err vs fp32, 32 rows)')
    for ctx in ctxs:
        kend = ctx + T - M.WIN
        if kend <= 0:
            continue
        tbl = pool.new_table()
        assert pool.reserve(tbl, ctx + T)
        Q = inputs(ctx, dev, g)[0]
        qpos = torch.arange(ctx, ctx + T, device=dev)
        idx = torch.linspace(0, T - 1, 32, device=dev).long()
        o0, l0 = far_ref(Q[idx], tbl, qpos[idx], kend)
        fin = torch.isfinite(l0)
        line = []
        for name, fn in (('flex', lambda: (setattr(M, 'PREFILL_ATTN', 'torch'), m.far(Q, tbl, 0, qpos, kend, ctx))[1]),
                         ('bf16', lambda: (setattr(PA, 'FP8', False), PA.far(Q, tbl, 0, ctx, kend))[1]),
                         ('fp8', lambda: (setattr(PA, 'FP8', True), PA.far(Q, tbl, 0, ctx, kend))[1])):
            o, l = fn()
            ts = []
            for _ in range(1 if CPU else 3):
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                if CPU:
                    t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1e3)
                    continue
                a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
            e = rel(o[idx][fin], o0[fin])
            le = (l[idx][fin] - l0[fin]).abs().max().item()
            line.append(f'{name} {statistics.median(ts):7.2f} ms err {e[0]:.1e}/{e[1]:.1e} lse {le:.0e}')
        PA.FP8 = False
        print(f'  ctx {ctx:8d}: ' + ' | '.join(line), flush=True)
        pool.release(tbl)


def sweep(m, pool, ctx, dev, g):
    tbl = pool.new_table()
    assert pool.reserve(tbl, ctx + T)
    seq = M.Seq(dev, tbl)
    Q, K, V, table, sink, ring = inputs(ctx, dev, g)
    args = (Q, K, V, table, sink)
    old = PA.BM, PA.BN, PA.WARPS, PA.STAGES, PA.PROGS, PA.NBM, PA.NBN, PA.NWARPS
    print(f'\nsweep at ctx {ctx} (full visit, far only = INK_PREFILL_ATTN=far):')
    best = []
    for bm, bn, w, st in FAR_CFGS:
        for pr in PROGS:
            PA.BM, PA.BN, PA.WARPS, PA.STAGES, PA.PROGS = bm, bn, w, st, pr
            try:
                run(m, seq, ring, ctx, args, True, 'far', 1)
                ms = run(m, seq, ring, ctx, args, True, 'far', 3)[1]
            except Exception as e:  # noqa: BLE001
                print(f'  far BM {bm} BN {bn} warps {w} stages {st} progs {pr}: failed {type(e).__name__}'); continue
            best.append((ms, f'INK_PA_BM={bm} INK_PA_BN={bn} INK_PA_WARPS={w} INK_PA_STAGES={st} INK_PA_PROGS={pr}'))
            print(f'  far BM {bm:3d} BN {bn:3d} warps {w} stages {st} progs {pr:4d}: {ms:8.2f} ms')
    PA.BM, PA.BN, PA.WARPS, PA.STAGES, PA.PROGS = old[:5]
    print('sweep near (sliding visit, triton):')
    nbest = []
    for bm, bn, w in NEAR_CFGS:
        PA.NBM, PA.NBN, PA.NWARPS = bm, bn, w
        try:
            run(m, seq, ring, ctx, args, False, 'triton', 1)
            ms = run(m, seq, ring, ctx, args, False, 'triton', 5)[1]
        except Exception as e:  # noqa: BLE001
            print(f'  near BM {bm} BN {bn} warps {w}: failed {type(e).__name__}'); continue
        nbest.append((ms, f'INK_PA_NEAR_BM={bm} INK_PA_NEAR_BN={bn} INK_PA_NEAR_WARPS={w}'))
        print(f'  near BM {bm:3d} BN {bn:3d} warps {w}: {ms:8.2f} ms')
    PA.NBM, PA.NBN, PA.NWARPS = old[5:]
    if best and nbest:
        print(f'best: {min(best)[1]} {min(nbest)[1]}')
    pool.release(tbl)


if __name__ == '__main__':
    sys.exit(compile_check() if COMPILE else main())
