"""GPU window only: the 2026-11-23 decode speed flags (decode_kernels.OPT: UNROLL / PDL / L2PF / W2ALL; decode.FAR2),
each alone and stacked, against the current kernels, in one process (flags are read when a decoder is built).
Per context mark (~1k, 40k, then one long document prefilled incrementally to 130k [--long], 512k [--512k], 1M [--1m]):
  for every config: fused-graph Decoder tok/s (greedy, host argmax per step as in serving) and BatchDecoder (graphs)
  step ms / aggregate tok/s at B = 1, 2, 4 (+8 up to 40k; rows are forks of the mark's sequence);
  exactness vs the base config, teacher-forced on base's greedy tokens (N steps): bit-identical logits, else gated
  like gpu_batch: median / p95 per-step KL <= 2x those of a 1-ulp embedding-nudge rerun of base + 1e-4 / 1e-3, top-1
  >= 95% (fused); the batched rows (B = 2) the same vs base's batched rows.
Prints the far-attention byte floor per mark (8 full visits x 1152 B per far key, at 1792 GB/s).
--nsp 32,42,64,84: also fused tok/s of the FAR2 / ALL / FAR3 configs at each split count (long marks only).
FAR3 (INK_DEC_FAR3 far kernel, tiles from INK_DEC_FAR3_BLK / _STAGES / _WARPS, splits INK_DEC_FAR3_NSP): fused only.
--configs base,UNR,...: subset. Exit 1 on a gate failure, 2 on an exception.
Usage (Qwen down): ~/llm/engine/swap_test.sh M env INK_MOE=triton INK_DEC_FAR_DOT=1 INK_FP8_DECODE=1 \\
  python tests/gpu_decode2.py --long --512k --1m"""
import contextlib
import os
import statistics
import sys
import time
import traceback

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('TRITON_DEFAULT_FP_FUSION', '1')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import torch  # noqa: E402

import chat  # noqa: E402
import decode as D  # noqa: E402
import decode_batch as DB  # noqa: E402
import decode_kernels as K  # noqa: E402
import model as M  # noqa: E402
from engine import Engine  # noqa: E402
from gpu_batch import fork, kl, speed  # noqa: E402
from gpu_decode import clone, q  # noqa: E402
from oracle import prose  # noqa: E402

N, SPD = 32, 32
CONFIGS = {'base': {}, 'UNR': dict(UNR=True), 'PDL': dict(PDL=True), 'L2PF': dict(PF=True), 'W2ALL': dict(W2ALL=True),
           'FAR2': dict(FAR2=True), 'ALL': dict(UNR=True, PDL=True, PF=True, W2ALL=True, FAR2=True),
           'BEST': dict(UNR=True, PDL=True, FAR2=True),  # window decode2: L2PF + W2ALL were slower
           'FAR3': dict(FAR3=True)}  # fused only: the batch path is FAR2's kernel (at FAR3_NSP splits), not re-run
NSP0 = D.FAR_NSP, D.FAR3_NSP
PEAK = 1792e9


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def apply(cfg, nsp=None):
    K.OPT.update({k: bool(cfg.get(k)) for k in ('UNR', 'PDL', 'PF', 'W2ALL')})
    D.FAR2 = bool(cfg.get('FAR2'))
    D.FAR3 = bool(cfg.get('FAR3'))
    D.FAR_NSP, D.FAR3_NSP = (nsp, nsp) if nsp else NSP0


@contextlib.contextmanager
def nudged(m, toks):
    """m (None: no-op): embedding rows of toks get +1 bf16 ulp inside the block."""
    saved = {}
    if m is not None:
        for x in set(toks):
            saved[x] = m.W.embed[x].clone()
            m.W.embed[x] = (saved[x].float() * (1 + 2 ** -7)).bfloat16()
    try:
        yield
    finally:
        for x, r in saved.items():
            m.W.embed[x] = r


def fused_run(dec, seq, first, n, toks=None, nudge=None):
    """Greedy (toks None) or teacher-forced fused steps from a clone of seq -> (tokens, log-probs [n, V], seconds of
    steps 2..n). nudge: model whose embedding rows of the fed tokens get +1 bf16 ulp during the run."""
    dec.load(clone(seq))
    t = int(first.argmax()) if toks is None else toks[0]
    fed, lps, t0 = [], [], None
    with nudged(nudge, toks):
        for i in range(n):
            if i == 2:
                torch.cuda.synchronize()
                t0 = time.time()
            fed.append(t)
            lg = dec.step(t)[0]
            lps.append(torch.log_softmax(lg.float(), -1))
            t = int(lg.argmax()) if toks is None else (toks[i + 1] if i + 1 < n else 0)
        torch.cuda.synchronize()
    dec.seq = None
    return fed, torch.stack(lps), time.time() - t0


def bd_rows(bd, seqs, toks, n):
    for i, s in enumerate(seqs):
        bd.load(i, s)
    out = []
    for j in range(n):
        lg = bd.step([(i, bd.seqs[i].pos, toks[i][j]) for i in range(len(seqs))])
        out.append(torch.log_softmax(lg.float(), -1))
    for i in range(len(seqs)):
        bd.free(i)
    return torch.stack(out, 1)


def gate(label, ref, lp, noise):
    """ref / lp [n, V] log-probs; noise (median, p95) KL of the nudge baseline. -> ok."""
    if torch.equal(ref, lp):
        print(f'  {label}: bit-identical to base on all {len(ref)} steps  OK', flush=True)
        return True
    ks = kl(ref, lp).tolist()
    km, kp = q(ks, .5), q(ks, .95)
    top = (lp.argmax(-1) == ref.argmax(-1)).float().mean().item()
    lm, lpp = 2 * noise[0] + 1e-4, 2 * noise[1] + 1e-3
    ok = km <= lm and kp <= lpp and top >= 0.95 and bool(torch.isfinite(lp).all())
    print(f'  {label}: NOT bit-identical; KL median {km:.1e} / p95 {kp:.1e} (limits {lm:.1e} / {lpp:.1e}), top-1 '
          f'{top:.3f}  {"OK" if ok else "TOO BIG"}', flush=True)
    return ok


def main():
    cfgs = arg('--configs', ','.join(CONFIGS)).split(',')
    nsps = [int(x) for x in arg('--nsp', '').split(',') if x]
    marks_long = [x for x, f in ((131072, '--long'), (524288, '--512k'), (1040000, '--1m')) if f in sys.argv]
    eng = Engine('cuda', max_ctx=(marks_long[-1] if marks_long else 40000) + 16384, vision=False)
    m, ok = eng.model, True
    M.MOE_IMPL = 'triton'
    print(f'loaded {torch.cuda.memory_allocated() / 2**30:.2f} GiB; configs {cfgs}; FAR_NSP {D.FAR_NSP}; '
          f'FAR_KW {D.FAR_KW}; FAR3_NSP {D.FAR3_NSP} FAR3_KW {D.FAR3_KW}', flush=True)

    def prefill(seq, ids):
        for a in range(0, len(ids), eng.chunk):
            f = m.forward(m.embed(ids[a:a + eng.chunk]), seq)[-1:]
        return m.logits(f)[0]

    def marks():
        for name, n, seed in (('~1k', 1000, 5), ('40k', 40000, 2)):
            ids = chat.render(eng.tok, [{'role': 'user', 'content': prose(n + 200, eng.tok, seed)}], effort=0.9)[:n]
            seq = m.new_seq()
            t0 = time.time()
            first = prefill(seq, ids)
            print(f'\n== {name}: {len(ids)} tokens prefilled in {time.time() - t0:.0f} s', flush=True)
            yield name, seq, first
            del seq
        if marks_long:
            ids = chat.render(eng.tok, [{'role': 'user', 'content': prose(marks_long[-1] + 2000, eng.tok, 7)}],
                              effort=0.9)[:marks_long[-1]]
            seq, done = m.new_seq(), 0
            for n in marks_long:
                t0 = time.time()
                first = prefill(seq, ids[done:n])
                done = n
                print(f'\n== {n // 1000}k: prefilled to {n} in +{time.time() - t0:.0f} s', flush=True)
                yield f'{n // 1000}k', seq, first

    table = []
    for name, seq, first in marks():
        kend = max(seq.pos - 1023, 0)
        fb = 8 * 1152 * kend
        print(f'far attention reads {fb / 1e9:.2f} GB/token -> floor {fb / PEAK * 1e3:.2f} ms/token on top of the '
              f'weights (~3.78 GB fp8 decode -> ~2.11 ms)', flush=True)
        long = seq.pos > 100000
        Bs = (1, 2, 4) if seq.pos > 50000 else (1, 2, 4, 8)
        base = noise = bbase = None
        for cn in cfgs:
            apply(CONFIGS[cn])
            fd = D.Decoder(m, graph=True, fuse=True)
            fused_run(fd, seq, first, 4)                               # build (flags read here) + graph capture
            if base is None:
                assert cn == 'base', 'the first config must be base (the reference)'
                base = fused_run(fd, seq, first, N)
                nz = fused_run(fd, seq, first, N, base[0], nudge=m)
                ks = kl(base[1], nz[1]).tolist()
                noise = (q(ks, .5), q(ks, .95))
                print(f'  noise (1-ulp embedding nudge of base) KL median/p95 {noise[0]:.1e}/{noise[1]:.1e}', flush=True)
                r = base
            else:
                r = fused_run(fd, seq, first, N, base[0])
                ok &= gate(f'{cn} fused', base[1], r[1], noise)
            tps = (N - 2) / r[2]
            table.append((name, cn, 'fused', 1, tps))
            print(f'  {cn:6s} fused-graph {tps:7.1f} tok/s ({1e3 / tps:.2f} ms/step)', flush=True)
            del fd
            if long and nsps and cn in ('FAR2', 'ALL', 'FAR3'):
                for nsp in nsps:
                    apply(CONFIGS[cn], nsp)
                    fd = D.Decoder(m, graph=True, fuse=True)
                    fused_run(fd, seq, first, 4)
                    r = fused_run(fd, seq, first, N, base[0])
                    print(f'  {cn:6s} nsp {nsp:3d}: fused-graph {(N - 2) / r[2]:7.1f} tok/s', flush=True)
                    table.append((name, f'{cn}/nsp{nsp}', 'fused', 1, (N - 2) / r[2]))
                    del fd
                apply(CONFIGS[cn])
            if cn == 'FAR3':
                continue
            bd = DB.BatchDecoder(m, max(Bs), graph=True)
            btoks = [base[0], base[0][::-1]]
            lp = bd_rows(bd, [fork(m, seq) for _ in range(2)], btoks, 12)
            if bbase is None:
                bbase = lp
                # each row's own noise: row 1's reversed tokens are a much less certain stream than base's greedy
                # tokens (window final5: its FAR2 rounding change failed the row-0 based limits)
                with nudged(m, btoks[0][:12] + btoks[1][:12]):
                    nz = bd_rows(bd, [fork(m, seq) for _ in range(2)], btoks, 12)
                bnoise = []
                for i in range(2):
                    ks = kl(bbase[i], nz[i]).tolist()
                    bnoise.append((q(ks, .5), q(ks, .95)))
                    print(f'  batch row {i} noise (1-ulp nudge) KL median/p95 {bnoise[i][0]:.1e}/{bnoise[i][1]:.1e}',
                          flush=True)
            else:
                for i in range(2):
                    ok &= gate(f'{cn} batch row {i}', bbase[i], lp[i], bnoise[i])
            for B in Bs:
                dt = speed(bd, [fork(m, seq) for _ in range(B)], SPD)
                table.append((name, cn, 'batch', B, B / dt))
                print(f'  {cn:6s} batch B={B}: step {dt * 1e3:6.2f} ms, {B / dt:7.1f} tok/s aggregate', flush=True)
            bd.drop()
            del bd
            torch.cuda.empty_cache()
        apply({})

    print('\n== summary (tok/s; x = vs base at the same mark/path/B)')
    ref = {(a, p, B): t for a, c, p, B, t in table if c == 'base'}
    for a, c, p, B, t in table:
        print(f'{a:6s} {c:12s} {p:5s} B={B}  {t:7.1f}  x{t / ref[(a, p, B)]:.3f}')
    print('PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 2
    sys.exit(rc)
