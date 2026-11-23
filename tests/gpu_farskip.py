"""GPU window, measure-first: would skipping far-attention key tiles pay at 64k-200k? (Far attention runs at ~165
TFLOP/s, near the bf16 MMA roof, and FP8-PV gave nothing, so only doing less work can make it faster.)

Cold-prefills real, diverse text (local Python sources, like agent traffic; no repeated block) with the current INK_*
flags, timing it, and at the chunks starting at --at positions captures Q and the 4-bit store of each full visit. Then
offline in torch, per (query row, 64-key tile) of the far range: the exact tile logsumexp, the exact tile max (oracle
bound), a ball bound (q.c + |q| max |k - c| around the tile's key mean c) and a box bound (sum_d max(q_d kmax_d, q_d kmin_d) from the tile's per-dimension key min / max: cheap enough
to precompute per tile when the KV is written, one extra [rows,128] x [128] product per tile in a kernel). For each
delta (nats below the row's far logsumexp; a kernel would use a running max or the near lse instead, so this is
optimistic): the fraction of (row, tile) work below the bound and the probability mass it holds, relative to the far
part alone (an upper bound on the error of dropping it).
  ~/llm/engine/swap_test.sh 20 ~/llm/engine/.venv/bin/python tests/gpu_farskip.py [--at 65536,196608]
  Then (unless --no-kernel) the INK_FARSKIP kernel itself on the same captured data (prefill_attn.far, skip delta
  0 = exact / 12 / 16 / 20 / 24 at BN 64 and 32): ms vs no skipping, fraction of (program, tile) work skipped, and the
  output error vs no skipping, gated by noise = flex vs the Triton far kernel on the same inputs (both accepted far
  paths): error mean and max <= the noise's; delta 0 must be bit-identical.
  --time-only: just the cold prefill with marks at --at (no capture / analysis): end-to-end A/B of INK_FARSKIP.
  --selftest: CPU, random data: bounds are upper bounds, fractions monotone, mass consistent (~10 s)."""
import glob
import os
import sys
import time

SELF = '--selftest' in sys.argv
if SELF:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
for k, v in dict(INK_MOE='triton', INK_MOE_V2='1', INK_PREFILL_ATTN='auto',
                 PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True').items():
    os.environ.setdefault(k, v)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import torch  # noqa: E402

TILE, KB = 64, 16384
DELTAS = (8, 12, 16, 20, 24)


def analyze(Q, K, p0, kend):
    """Q [T,24,128] bf16 (tau applied), K [kend,8,128] bf16 far keys of one visit, chunk at p0. -> {(kind, delta):
    [skipped (row, tile) pairs, valid pairs, sum of skipped mass, list of per-row skipped mass]}."""
    T = Q.shape[0]
    nt = -(-kend // TILE)
    pad = nt * TILE - kend
    out = {}
    for h in range(8):
        q = Q[:, 3 * h:3 * h + 3].reshape(T * 3, 128).float() * 128 ** -0.5
        lim = p0 + torch.arange(T * 3, device=Q.device) // 3 - 1023          # key k is far for row r iff k < lim_r
        k = K[:, h].float()
        if pad:
            k = torch.cat((k, k[-1:].expand(pad, 128)))                      # pad keys masked below (k >= kend)
        kt = k.view(nt, TILE, 128)
        box = q.clamp(min=0) @ kt.amax(1).t() - (-q).clamp(min=0) @ kt.amin(1).t()   # [R, nt]
        cen = kt.mean(1)                                                     # ball: q.c + |q| max_k |k - c|
        ball = q @ cen.t() + q.norm(dim=1, keepdim=True) * (kt - cen[:, None]).norm(dim=2).amax(1)[None]
        tl, tm = [], []
        for b0 in range(0, nt * TILE, KB):
            b1 = min(nt * TILE, b0 + KB)
            s = q @ k[b0:b1].t()
            kp = torch.arange(b0, b1, device=Q.device)
            s = s.masked_fill((kp[None] >= lim[:, None]) | (kp[None] >= kend), float('-inf')).view(len(q), -1, TILE)
            tl.append(torch.logsumexp(s, -1))
            tm.append(s.amax(-1))
        tl, tm = torch.cat(tl, 1), torch.cat(tm, 1)
        valid = tm > float('-inf')
        lse = torch.logsumexp(tl, 1, keepdim=True)
        ok = lse.squeeze(1) > float('-inf')                                   # rows with far keys at all
        mass = torch.exp(tl - lse).nan_to_num(0.)
        for kind, bnd in (('exact', tm), ('box', box), ('ball', ball)):
            assert bool((bnd[valid] >= tm[valid] - 1e-3).all()), f'{kind} bound below the tile max'
            for d in DELTAS:
                skip = valid & (bnd < lse - d)
                m = (mass * skip).sum(1)[ok]
                a = out.setdefault((kind, d), [0, 0, 0., []])
                a[0] += int(skip.sum()); a[1] += int(valid.sum()); a[2] += float(m.sum()); a[3].append(m.cpu())
    return out


def kernel(m, orig, Q, kv, slot, p0, kend):
    """Time / check prefill_attn.far with tile skipping on one captured far call -> bool (all gates passed)."""
    import model as M
    import prefill_attn as PA
    from gpu_moe import cuda_ms
    bn0, pa0 = PA.BN, M.PREFILL_ATTN
    ref = PA.far(Q, kv, slot, p0, kend, skip=-1)
    t_ref = cuda_ms(lambda: PA.far(Q, kv, slot, p0, kend, skip=-1), 3)
    M.PREFILL_ATTN = 'torch'                                                 # Model.far -> flex blocks
    try:
        qp = torch.arange(p0, p0 + Q.shape[0], device=Q.device)
        fx = orig(m, Q, kv, slot, qp, kend, p0)
        t_fx = cuda_ms(lambda: orig(m, Q, kv, slot, qp, kend, p0), 3)
    finally:
        M.PREFILL_ATTN = pa0
    fin = torch.isfinite(ref[1])

    def err(a):
        do = (a[0] - ref[0]).abs()[fin]
        return do.mean().item(), do.max().item(), (a[1] - ref[1]).abs()[fin].max().item()
    nz = err(fx)
    print(f'    kernel: no skip {t_ref:.1f} ms (flex {t_fx:.1f} ms: auto uses flex here without INK_FARSKIP); noise (flex vs triton far): o mean {nz[0]:.1e} max {nz[1]:.1e}, '
          f'lse max {nz[2]:.1e}', flush=True)
    ok = True
    for bn in (64, 32):
        PA.BN = bn
        try:
            for d in (0, 12, 16, 20, 24):
                PA.SKIP_STATS[0] = st = torch.zeros(2, dtype=torch.int64, device=Q.device)
                got = PA.far(Q, kv, slot, p0, kend, skip=d)
                PA.SKIP_STATS[0] = None
                t = cuda_ms(lambda: PA.far(Q, kv, slot, p0, kend, skip=d), 3)
                e = err(got)
                if d == 0 and bn == bn0:
                    good = torch.equal(got[0], ref[0]) and torch.equal(got[1], ref[1])
                else:
                    good = e[0] <= nz[0] and e[1] <= nz[1] and e[2] <= max(nz[2], 1e-6)
                ok &= good
                print(f'    BN {bn} delta {d:2d}: {t:6.1f} ms ({t_ref / t:.2f}x), skipped {st[0].item() / max(st[1].item(), 1):6.1%};'
                      f' err o mean {e[0]:.1e} max {e[1]:.1e} lse {e[2]:.1e} '
                      f'{"(bit-identical)" if d == 0 and bn == bn0 and good else ""}{"OK" if good else "OUTSIDE noise"}',
                      flush=True)
        finally:
            PA.BN, PA.SKIP_STATS[0] = bn0, None
    return ok


def report(name, st):
    print(f'  {name}: (row, tile) work skippable / mass skipped (per row, rel. to the far part: mean, p99.9, max)')
    for (kind, d), (n, v, _, ms) in st.items():
        m = torch.cat(ms)
        print(f'    {kind:5s} delta {d:2d}: skip {n / max(v, 1):6.1%}  mass mean {m.mean():.1e} p99.9 '
              f'{m.quantile(0.999) if m.numel() < 1 << 24 else m.max():.1e} max {m.max():.1e}', flush=True)


def code_ids(tok, n):
    """n token ids of local Python sources (sorted file order, deterministic, no repeats until exhausted)."""
    root = os.path.join(os.path.dirname(HERE), '.venv', 'lib')
    ids = []
    for f in sorted(glob.glob(os.path.join(root, 'python3*', 'site-packages', '**', '*.py'), recursive=True)):
        try:
            ids += tok.encode(open(f, encoding='utf-8', errors='ignore').read()).ids
        except OSError:
            continue
        if len(ids) >= n:
            return ids[:n]
    raise RuntimeError(f'only {len(ids)} tokens of source text')


def gpu():
    from tokenizers import Tokenizer

    import model as M
    from engine import MODEL_DIR
    arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
    at = [int(x) for x in arg('--at', '65536,131072,196608').split(',')]
    T = 4096
    print('flags:', ' '.join(f'{k}={v}' for k, v in sorted(os.environ.items()) if k.startswith('INK_')), flush=True)
    m = M.Model(MODEL_DIR, 'cuda', max_ctx=max(at) + T + 512)
    ids = code_ids(Tokenizer.from_file(os.path.join(MODEL_DIR, 'tokenizer.json')), max(at) + T)
    cap, orig = [], M.Model.far

    def far(self, Q, kv, slot, qpos, kend, p0=None):
        if p0 is not None and p0 in at and '--time-only' not in sys.argv:
            cap.append((Q.clone(), kv, slot, p0, kend))
        return orig(self, Q, kv, slot, qpos, kend, p0)
    M.Model.far = far
    seq = m.new_seq()
    assert m.pool.reserve(seq.kv, len(ids))
    t0, ok = time.time(), True
    for c in range(0, len(ids), T):
        m.forward(m.embed(ids[c:c + T]), seq)
        if c in at:
            torch.cuda.synchronize()
            print(f'cold prefill of source text: {c + T} tokens in {time.time() - t0:.1f} s '
                  f'({(c + T) / (time.time() - t0):.0f} tok/s; capture forwards included)', flush=True)
            for Q, kv, slot, p0, kend in cap:
                Kf = kv.get(slot, 0, kend)[0]
                report(f'chunk at {p0}, full visit slot {slot} (far keys {kend})', analyze(Q, Kf, p0, kend))
                del Kf
                if not ({'--no-kernel', '--time-only'} & set(sys.argv)):
                    ok &= kernel(m, orig, Q, kv, slot, p0, kend)
            cap.clear()
    if not ({'--no-kernel', '--time-only'} & set(sys.argv)):
        print('FARSKIP KERNEL', 'PASS' if ok else 'FAIL (some delta outside noise; see lines above)')
    return 0 if ok else 1


def selftest():
    g = torch.Generator().manual_seed(0)
    T, p0 = 40, 3000
    kend = p0 + T - 1024
    Q = (torch.randn(T, 24, 128, generator=g) * 2).bfloat16()
    K = torch.randn(kend, 8, 128, generator=g)
    K[100:164] += 3 * Q[0, 0].float()                                         # one hot tile for row 0
    st = analyze(Q, K.bfloat16(), p0, kend)
    for kind in ('exact', 'box', 'ball'):
        fr = [st[(kind, d)][0] / st[(kind, d)][1] for d in DELTAS]
        assert all(a >= b for a, b in zip(fr, fr[1:])), (kind, fr)
        assert all(torch.cat(st[(kind, d)][3]).max() <= 1 + 1e-5 for d in DELTAS)
    for d in DELTAS:                                                          # the oracle skips at least as much
        assert st[('exact', d)][0] >= max(st[('box', d)][0], st[('ball', d)][0])
    report('selftest', st)
    print('FARSKIP SELFTEST OK')
    return 0


if __name__ == '__main__':
    sys.exit(selftest() if SELF else gpu())
