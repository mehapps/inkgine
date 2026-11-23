"""GPU window only: the kvpool writer thread vs graph capture on the main thread (prod 11-23: during a 1M prefill the
writer's ready() -- a cudaEventSynchronize -- ran while the main thread captured a BatchDecoder / DSpark graph in
global mode: the write failed ("cache write skipped ... not permitted when stream is capturing") and the capture was
invalidated (drafter failed / 500 from decode_batch._replay)).

Loop: queue a burst of big checkpoint writes on a scratch KVPool (its own writer thread, temp dir: every file starts
with ready()), then drop the BatchDecoder's and the drafter State's graphs and decode a step, so both recapture while
the writer drains. Counts writer ready() calls that overlap a capture (must be > 0 or the run proves nothing).
Pass: no "cache write skipped", no drafter failure, no capture failure, every write landed, spec still on, overlaps > 0.
--global: force capture_error_mode='global' (the old behaviour): should FAIL, proving the repro reaches the race.
Exit 0 pass, 1 fail, 3 no overlap (inconclusive).
Usage (Qwen down, nothing else on the GPU):
  ~/llm/engine/swap_test.sh 15 python tests/gpu_capture_race.py [--iters 20] [--global]
"""
import os
import shutil
import sys
import tempfile
import threading

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import torch  # noqa: E402

import decode_batch  # noqa: E402
import dspark  # noqa: E402
import engine as E  # noqa: E402
import kvpool  # noqa: E402

ITERS = int(sys.argv[sys.argv.index('--iters') + 1]) if '--iters' in sys.argv else 20
errs, capturing, overlaps, readies = [], threading.Event(), [0], [0]


def rec(*a, **k):  # the modules' prints: keep them, and remember the failures
    s = ' '.join(map(str, a))
    print(s, flush=True)
    if any(x in s for x in ('cache write skipped', 'drafter failed', 'capture failed')):
        errs.append(s)


for mod in (kvpool, E, decode_batch, dspark):
    mod.print = rec
if '--global' in sys.argv:
    real_graph = torch.cuda.graph
    torch.cuda.graph = lambda *a, **k: real_graph(*a, **{**k, 'capture_error_mode': 'global'})


def flagged(fn):  # capture window (incl. its eager run: an over-count, overlaps are a lower bound only for 'global')
    def w(*a, **k):
        capturing.set()
        try:
            return fn(*a, **k)
        finally:
            capturing.clear()
    return w


decode_batch.BatchDecoder._replay = (lambda real: lambda self, R: real(self, R) if R in self.graphs
                                     else flagged(real)(self, R))(decode_batch.BatchDecoder._replay)
dspark.DSpark._run = (lambda real: lambda self, st, k, g, nb=None: real(self, st, k, g, nb)
                      if (k, g, nb) in st.graphs else flagged(real)(self, st, k, g, nb))(dspark.DSpark._run)

eng = E.Engine('cuda')
eng.warmup(min(2048, eng.chunk))
tmp = tempfile.mkdtemp(prefix='inkrace')
side = kvpool.KVPool(kvpool.PAGE * 8, 'cuda', 'race', cache_dir=tmp, cap_gb=2, min_free_gb=1)  # (ckpts evict ckpts: ~5 GB written in all)
real_host = side._host


def host(tensors):  # count writer ready() calls that land inside a capture window
    out, ready = real_host(tensors)

    def r():
        readies[0] += 1
        overlaps[0] += capturing.is_set()
        ready()
    return out, r


side._host = host
blob = {f't{i}': torch.randn(16 << 20, device='cuda') for i in range(2)}  # 2 x 64 MiB per checkpoint
try:
    reqs = []
    for s in range(2):
        r = eng.request([{'role': 'user', 'content': f'Count from {s} to 5000, one number per line.'}],
                        max_tokens=100000, temperature=0)
        eng.start(r)
        while not eng.prefill_chunk(r):
            pass
        eng.to_decode(r, s)
        reqs.append(r)
    assert all(r.sp is not None for r in reqs), 'spec is off: no drafter capture to race'
    n = 0
    for it in range(ITERS):
        for _ in range(4):  # a burst: the writer drains it while the steps below recapture
            n += 1
            side.save_ckpt(torch.arange(n * kvpool.PAGE, (n + 1) * kvpool.PAGE), blob)
        for _ in range(3):
            eng.bd.graphs.clear()
            for r in reqs:
                if r.sp is not None:
                    r.sp.st.graphs.clear()
            eng.decode_step(reqs)
            if any(r.done for r in reqs):
                raise SystemExit('a request finished early: raise max_tokens')
        if it % 10 == 9:
            print(f'iter {it + 1}: writer ready() {readies[0]}, during a capture {overlaps[0]}, errors {len(errs)}',
                  flush=True)
    side.flush()
    landed = sum(os.path.exists(p) for p in side.files)
    for r in reqs:
        errs.extend([f'request {r.slot}: spec turned off'] if r.sp is None else [])
        eng.finish(r)
    print(f'writes queued {n}, landed {landed}; writer ready() {readies[0]}, during a capture {overlaps[0]}; '
          f'errors {len(errs)}', flush=True)
    if landed != len(side.files):
        errs.append(f'{len(side.files) - landed} writes missing')
finally:
    shutil.rmtree(tmp, ignore_errors=True)
if errs:
    print('FAIL', *errs[:5], sep='\n')
    sys.exit(1)
if not overlaps[0]:
    print('INCONCLUSIVE: no writer ready() inside a capture')
    sys.exit(3)
print('CAPTURE RACE OK')
