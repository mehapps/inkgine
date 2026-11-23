"""GPU window: flex far attention with one compiled query shape (model.FLEX_TQ; pad rows masked) for chunk tails.
Model.far (flex) vs prefill_attn.far (Triton) on a random 4-bit pool at ~100k far keys for T = 4096, 2048, 777, 64:
first-call ms (a compile shows up as seconds) and the error between the two accepted far paths (window 2026-11-23:
o max ~3e-2, lse ~1e-5 at T=4096). Exit 1 if a tail's error exceeds 2x the T=4096 one or the empty-row pattern differs.
  ~/llm/engine/swap_test.sh 10 ~/llm/engine/.venv/bin/python ~/llm/engine/tests/gpu_flextail.py"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import torch  # noqa: E402

import model as M  # noqa: E402
import prefill_attn as PA  # noqa: E402
from gpu_prefill import fill  # noqa: E402
from kvpool import KVPool  # noqa: E402


def main():
    dev, g = torch.device('cuda'), torch.Generator(device='cuda').manual_seed(0)
    p0 = 100352
    pool = KVPool(p0 + 4096 + 512, dev, 'no-disk', cache_dir=None)
    fill(pool)
    kv = pool.new_table()
    assert pool.reserve(kv, p0 + 4096)
    m = object.__new__(M.Model)
    m.slot, m.dev = {}, dev
    ok, base = True, None
    for T in (4096, 2048, 777, 64):
        Q = (M.l2clip(torch.randn(T, M.NQ, M.HD, device=dev, generator=g) * 3, M.CLIP_QK) * 1.2).bfloat16()
        kend, qpos = p0 + T - M.WIN, torch.arange(p0, p0 + T, device=dev)
        M.PREFILL_ATTN = 'torch'
        torch.cuda.synchronize()
        t0 = time.time()
        of, lf = m.far(Q, kv, 0, qpos, kend, p0)
        torch.cuda.synchronize()
        t1 = time.time() - t0
        ot, lt = PA.far(Q, kv, 0, p0, kend, skip=-1)
        fin = torch.isfinite(lt)
        same = torch.equal(fin, torch.isfinite(lf))
        eo, el = (of - ot).abs()[fin].max().item(), (lf - lt).abs()[fin].max().item()
        base = base or (eo, el)
        good = same and eo <= 2 * base[0] + 1e-3 and el <= 2 * base[1] + 1e-4
        ok &= good
        print(f'T={T:4d}: flex first call {t1 * 1e3:7.0f} ms; flex vs triton o max {eo:.1e} lse max {el:.1e} '
              f'{"OK" if good else "BAD"}', flush=True)
    print('FLEXTAIL', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
