"""GPU window: VRAM headroom of a long prefill through the engine (same Engine + generate path as the server).
Loads the engine (env as the server: INK_MOE, INK_CHUNK, INK_GRAPH, ...), runs a short warm-up request (captures the
decode graph when INK_GRAPH=1), then one TOKENS-token prompt with 8 decode tokens. A thread samples cudaMemGetInfo
every 10 ms (= what nvidia-smi / other processes see); a preempt hook records per prefill chunk: allocated at start,
peak allocated / reserved, min free. One late chunk runs with the allocator trace on and prints which code lines held
the memory at its peak (transient contributors). Exits 1 if min free < INK_MEM_FLOOR_MB (700).
Usage (window, Qwen down): INK_MOE=triton INK_CHUNK=4096 INK_GRAPH=1 python tests/gpu_mem.py [TOKENS=1040000]
CPU check of the trace analysis: python tests/gpu_mem.py --selftest"""
import os
import sys
import threading
import time

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')  # as server.py
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
MiB = 2 ** 20
FLOOR = int(os.environ.get('INK_MEM_FLOOR_MB', 700))


def peak_holders(events, root=os.path.dirname(HERE)):
    """Replay allocator trace events -> (peak bytes above the trace start, [(bytes, 'file:line fn'), ...] live at the
    peak, grouped by the innermost engine frame)."""
    cur = peak = peak_i = 0
    live = {}
    for i, e in enumerate(events):
        if e['action'] == 'alloc':
            live[e['addr']] = e
            cur += e['size']
            if cur > peak:
                peak, peak_i = cur, i
        elif e['action'] == 'free_completed' and e['addr'] in live:
            cur -= live.pop(e['addr'])['size']
    live = {}
    for e in events[:peak_i + 1]:
        if e['action'] == 'alloc':
            live[e['addr']] = e
        elif e['action'] == 'free_completed':
            live.pop(e['addr'], None)
    by = {}
    for e in live.values():
        fr = next((f for f in e.get('frames', []) if f['filename'].startswith(root) and '/tests/' not in f['filename']),
                  None)
        k = f'{os.path.basename(fr["filename"])}:{fr["line"]} {fr["name"]}' if fr else '?'
        by[k] = by.get(k, 0) + e['size']
    return peak, sorted(((v, k) for k, v in by.items()), reverse=True)


def nbytes(ts):
    """Bytes of distinct cuda storages among tensors ts (views of one storage count once)."""
    seen = {}
    for t in ts:
        if isinstance(t, torch.Tensor) and t.is_cuda:
            seen[t.untyped_storage().data_ptr()] = t.untyped_storage().nbytes()
    return sum(seen.values())


def graph_pool_bytes():
    """Reserved bytes in private (CUDA graph) pools."""
    return sum(s['total_size'] for s in torch.cuda.memory._snapshot()['segments'] if tuple(s.get('segment_pool_id', (0, 0))) != (0, 0))


def prompt(eng, target):
    n = target // 12
    for _ in range(4):
        text = '\n'.join(f'Entry {i}: the {("red", "blue", "green")[i % 3]} fox counted {i * 7 % 1013} stones.' for i in range(n))
        msgs = [{'role': 'user', 'content': text + '\nHow many entries are there? Answer with one number.'}]
        got = len(eng.prepare(msgs, effort=0)[0]) + 2
        if abs(got - target) < target * 0.003:
            break
        n = int(n * target / got)
    return msgs, got


def main():
    target = int(next((a for a in sys.argv[1:] if a.isdigit()), 1040000))
    import engine
    from engine import Engine
    total = torch.cuda.mem_get_info()[1]
    eng = Engine('cuda')
    m = eng.model
    torch.cuda.synchronize()
    free0 = torch.cuda.mem_get_info()[0]
    alloc0, res0 = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
    w = nbytes(list(m.W.t.values()) + [m.W.e2m1] + [c for c, _ in m.W.proj.values()] + list(m.W.gcat.values()) + m.mq)
    pool = eng.pool.c.nbytes + eng.pool.s.nbytes
    print(f'loaded (chunk {eng.chunk}, INK_MOE={os.environ.get("INK_MOE", "torch")}, INK_GRAPH={int(eng.graph)}, '
          f'margin {engine.MARGIN // MiB} MiB, dec drop >= {engine.DEC_DROP} tokens): total {total / MiB:.0f} MiB, '
          f'free {free0 / MiB:.0f}, allocated {alloc0 / MiB:.0f}, reserved {res0 / MiB:.0f}, '
          f'non-torch (context, cuBLAS, kernels) {(total - free0 - res0) / MiB:.0f}', flush=True)
    dec = nbytes([x for d in eng.dec.st.conv for x in d.values()] + [x for r in eng.dec.st.ring for x in r]) if eng.dec else 0
    print(f'  text weights {w / MiB:.0f} MiB, KV pool {pool / MiB:.0f}, decoder static {dec / MiB:.0f}, other allocated '
          f'(vision, ...) {(alloc0 - w - pool - dec) / MiB:.0f}', flush=True)

    state = dict(min_free=free0, min_free_all=free0, stop=False)

    def sampler():
        torch.cuda.set_device(m.dev)
        while not state['stop']:
            f = torch.cuda.mem_get_info()[0]
            state['min_free'] = min(state['min_free'], f)
            state['min_free_all'] = min(state['min_free_all'], f)
            time.sleep(0.01)
    threading.Thread(target=sampler, daemon=True).start()

    st = {}
    list(eng.generate([{'role': 'user', 'content': 'Say hi.'}], max_tokens=8, temperature=0, effort=0, stats=st,
                      prefill_text=True))
    torch.cuda.synchronize()
    gp = graph_pool_bytes()
    print(f'warm-up done: graph pool {gp / MiB:.0f} MiB, parked {eng.parked_bytes() / MiB:.0f} MiB, free '
          f'{torch.cuda.mem_get_info()[0] / MiB:.0f} MiB, min free so far {state["min_free_all"] / MiB:.0f}', flush=True)

    msgs, n = prompt(eng, target)
    n_chunks = -(-n // eng.chunk)
    trace_at = max(1, n_chunks - 2)                       # a late chunk: far blocks at full size, most of them
    rec, t0 = [], time.time()
    cur = dict(i=0, alloc=torch.cuda.memory_allocated())

    def hook():                                           # called before every prefill chunk (and nothing else)
        torch.cuda.synchronize()
        i = cur['i']
        if i:
            rec.append((i - 1, cur['alloc'], torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved(),
                        state['min_free'], eng.parked_bytes(), graph_pool_bytes() if i % 16 == 0 else None))
            r = rec[-1]
            if i % 16 == 0 or i <= 2:
                print(f'chunk {r[0]:4d}/{n_chunks} {time.time() - t0:5.0f} s: start alloc {r[1] / MiB:.0f} MiB, peak alloc '
                      f'+{(r[2] - r[1]) / MiB:.0f}, peak reserved {r[3] / MiB:.0f}, min free {r[4] / MiB:.0f}, parked '
                      f'{r[5] / MiB:.0f}' + (f', graph pool {r[6] / MiB:.0f}' if r[6] is not None else ''), flush=True)
        if i == trace_at + 1:
            snap = torch.cuda.memory._snapshot()
            torch.cuda.memory._record_memory_history(enabled=None)
            peak, holders = peak_holders(snap['device_traces'][m.dev.index or 0])
            print(f'chunk {trace_at} transient peak {peak / MiB:.0f} MiB above its start; live at the peak by code line:')
            for b, k in holders[:15]:
                print(f'  {b / MiB:8.1f} MiB  {k}')
        if i == trace_at:
            torch.cuda.memory._record_memory_history(enabled='all', context='alloc', stacks='python',
                                                     max_entries=2_000_000)
        state['min_free'] = torch.cuda.mem_get_info()[0]
        torch.cuda.reset_peak_memory_stats()
        cur['i'], cur['alloc'] = i + 1, torch.cuda.memory_allocated()
        return False

    print(f'prompt {n} tokens, {n_chunks} chunks', flush=True)
    st, out = {}, []
    for k, d in eng.generate(msgs, max_tokens=8, temperature=0, effort=0, stats=st, prefill_text=True, preempt=hook):
        if not out:
            pre_min, state['min_free'] = state['min_free'], torch.cuda.mem_get_info()[0]  # decode phase from here
        out.append(d)
    torch.cuda.synchronize()
    state['stop'] = True
    dec_min = state['min_free']
    worst = max(rec, key=lambda r: r[2] - r[1])
    lo = min(rec, key=lambda r: r[4])
    print(f'prefill {st.get("prefill_s", 0):.0f} s, decode {st.get("completion_tokens")} tokens {st.get("decode_s", 0):.1f} s: '
          f'{"".join(out)!r}')
    print(f'worst chunk transient: chunk {worst[0]} +{(worst[2] - worst[1]) / MiB:.0f} MiB over {worst[1] / MiB:.0f}; '
          f'lowest free in chunk {lo[0]}: {lo[4] / MiB:.0f} MiB (peak reserved {lo[3] / MiB:.0f})')
    mf = state['min_free_all']
    print(f'min free: whole run {mf / MiB:.0f} MiB, decode {dec_min / MiB:.0f}, last chunk + park + decoder rebuild '
          f'{pre_min / MiB if out else float("nan"):.0f}; graph pool now {graph_pool_bytes() / MiB:.0f} MiB, '
          f'ooms {eng.ooms}')
    ok = mf >= FLOOR * MiB and eng.ooms == 0
    print('PASS' if ok else 'FAIL', f'(want min free >= {FLOOR} MiB, no OOM)')
    sys.exit(0 if ok else 1)


def selftest():
    R = os.path.dirname(HERE)
    fr = lambda f, ln: [{'filename': f'{R}/{f}', 'line': ln, 'name': 'fn'}, {'filename': '/usr/lib/torch/x.py', 'line': 1, 'name': 'g'}]
    ev = [dict(action='alloc', addr=1, size=100, frames=fr('model.py', 10)),
          dict(action='alloc', addr=2, size=50, frames=fr('kvpool.py', 5)),
          dict(action='free_completed', addr=1, size=100),
          dict(action='alloc', addr=3, size=120, frames=fr('model.py', 10)),
          dict(action='alloc', addr=4, size=10, frames=fr('model.py', 20)),   # peak: 50 + 120 + 10 = 180
          dict(action='free_completed', addr=3, size=120),
          dict(action='alloc', addr=5, size=60, frames=[])]
    peak, h = peak_holders(ev, R)
    assert peak == 180 and h == [(120, 'model.py:10 fn'), (50, 'kvpool.py:5 fn'), (10, 'model.py:20 fn')], (peak, h)
    print('selftest OK')


if __name__ == '__main__':
    selftest() if '--selftest' in sys.argv else main()
