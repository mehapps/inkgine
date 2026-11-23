"""GPU memory/speed probe: load the engine, prefill a synthetic prompt in chunks, print peak memory per chunk,
then run the 16.7 MP gate image through the vision tower. Usage: python tests/memprobe.py [TOKENS] [CHUNK]"""
import os
import sys
import time

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import Engine  # noqa: E402
from vision import load_image, preprocess  # noqa: E402

n = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
if len(sys.argv) > 2:
    os.environ['INK_CHUNK'] = sys.argv[2]
G = 2 ** 30
eng = Engine('cuda')
m = eng.model
print(f'loaded: allocated {torch.cuda.memory_allocated() / G:.2f} GiB, chunk {eng.chunk}', flush=True)
ids = eng.tok.encode(' '.join(f'line {i} the quick brown fox {i * 7 % 101}' for i in range(n // 9)),
                     add_special_tokens=False).ids[:n]
seq = m.new_seq()
t0 = time.time()
for a in range(0, len(ids), eng.chunk):
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    m.forward(m.embed(ids[a:a + eng.chunk]), seq)
    torch.cuda.synchronize()
    print(f'chunk at {a:>7}: {time.time() - t:5.2f} s, peak {torch.cuda.max_memory_allocated() / G:.2f} GiB, '
          f'reserved {torch.cuda.memory_reserved() / G:.2f} GiB', flush=True)
print(f'prefill {len(ids)} tokens in {time.time() - t0:.1f} s ({len(ids) / (time.time() - t0):.0f} tok/s)')
torch.cuda.reset_peak_memory_stats()
t = time.time()
f = m.forward(m.embed([1000]), seq)
torch.cuda.synchronize()
print(f'decode step at {seq.pos}: {time.time() - t:.2f} s, peak {torch.cuda.max_memory_allocated() / G:.2f} GiB')
torch.cuda.reset_peak_memory_stats()
t = time.time()
im = load_image(open(os.path.expanduser('~/llm/gate/img16mp.jpg'), 'rb').read())
y = eng.vision(*preprocess(im))
torch.cuda.synchronize()
print(f'vision 16.7MP: {y.shape[0]} tokens, {time.time() - t:.1f} s, peak {torch.cuda.max_memory_allocated() / G:.2f} GiB')
