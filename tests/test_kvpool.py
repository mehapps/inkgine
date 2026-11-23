"""CPU checks for kvpool.py: paged put/get == KV4, isolation, shared-prefix refcounts, eviction, disk spill/restore,
key mismatch, crash safety, size cap / free-space guard, mid-prompt checkpoints (pending writes, eviction order). Run: python tests/test_kvpool.py (~10 s)."""
import os
import shutil
import sys
import tempfile

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import kvpool
from kvpool import PAGE, KVPool
from model import KV4, HD, NKV

torch.manual_seed(0)
D = tempfile.mkdtemp(prefix='inkcache-test-')


def pool(n_tokens=8 * PAGE, wid='w-A', **kw):
    kw.setdefault('min_free_gb', 0)
    return KVPool(n_tokens, 'cpu', wid, cache_dir=kw.pop('cache_dir', D), **kw)


def kv(n):
    return torch.randn(n, NKV, HD).bfloat16(), torch.randn(n, NKV, HD).bfloat16()


def fill(p, tbl, data, p0=0):
    for slot, (K, V) in enumerate(data):
        p.put(tbl, slot, p0, K, V)


def same(p, tbl, data, a, b):
    return all(torch.equal(x, y) for slot, (K, V) in enumerate(data)
               for x, y in zip(p.get(tbl, slot, a, b), KV4_ref(K, V, a, b)))


def KV4_ref(K, V, a, b):
    ref = KV4(1, K.shape[0], 'cpu')
    ref.put(0, 0, K, V)
    return ref.get(0, a, b)


def test_roundtrip_isolation():
    p = pool()
    t1, t2 = p.new_table(), p.new_table()
    assert p.reserve(t1, 600) and p.reserve(t2, 300)
    assert len(t1.pages) == 3 and len(t2.pages) == 2 and not set(t1.pages) & set(t2.pages)
    d1, d2 = [kv(600) for _ in range(8)], [kv(300) for _ in range(8)]
    fill(p, t2, [(K[:100], V[:100]) for K, V in d2])           # interleaved, chunked writes
    fill(p, t1, d1)
    fill(p, t2, [(K[100:], V[100:]) for K, V in d2], p0=100)
    for a, b in ((0, 600), (17, 300), (255, 257), (512, 600), (0, 1)):
        assert same(p, t1, d1, a, b), (a, b)
    assert same(p, t2, d2, 0, 300)
    # decode write at a device position
    K, V = kv(1)
    p.put(t1, 3, torch.tensor([599]), K, V)
    ref = torch.cat((d1[3][0][:599], K)), torch.cat((d1[3][1][:599], V))
    assert all(torch.equal(x, y) for x, y in zip(p.get(t1, 3, 590, 600), KV4_ref(*ref, 590, 600)))
    # unused entries point at the zero page: fixed-size reads past the end are finite zeros
    assert torch.equal(p.get(t2, 0, 512, 1024)[0], torch.zeros(512, NKV, HD, dtype=torch.bfloat16))


def test_refcount_eviction():
    p = pool(cache_dir=None)
    keys = torch.arange(3 * PAGE + 10)
    a = p.new_table(); p.reserve(a, len(keys))
    data = [kv(len(keys)) for _ in range(8)]
    fill(p, a, data)
    p.publish(a, keys, len(keys))
    assert len(p.nodes) == 3 and [p.ref[x] for x in a.pages] == [2, 2, 2, 1]
    # a second request with the same first 2 pages, then a different token
    k2 = torch.cat((keys[:2 * PAGE + 5], torch.tensor([10 ** 6]), keys[2 * PAGE + 6:]))
    b, n = p.match(k2)
    assert n == 2 * PAGE and b.pages == a.pages[:2] and [p.ref[x] for x in a.pages[:3]] == [3, 3, 2]
    assert same(p, b, [(K[:n], V[:n]) for K, V in data], 0, n)
    free0 = p.free_pages()
    p.release(a)
    assert p.free_pages() == free0 + 1                         # only the private tail page came back
    assert p.free_pages() + len(p._evictable()) == 6 and p.can_admit(6 * PAGE) and not p.can_admit(7 * PAGE + 1)
    c = p.new_table()
    assert not p.reserve(c, 8 * PAGE) and c.pages == []        # too big: nothing allocated
    assert p.reserve(c, 6 * PAGE)                              # evicts the unshared cached page 3
    assert len(p.nodes) == 2 and p.free_pages() == 0
    p.release(b); p.release(c)
    assert p.free_pages() == 6 and p.reserve(c, 8 * PAGE) and not p.nodes   # evicts the rest


def test_disk_spill_restore():
    p = pool()
    keys = torch.arange(1000, 1000 + 2 * PAGE + 77)
    t = p.new_table(); p.reserve(t, len(keys) + 50)
    data = [kv(len(keys)) for _ in range(8)]
    fill(p, t, data)
    state = {'ring.0.k': torch.randn(1023, NKV, HD).bfloat16(), 'conv.3.mlp': torch.randn(3, 2304).bfloat16(),
             'f': torch.randn(1, 2304).bfloat16(), 'empty': torch.zeros(0, NKV, HD).bfloat16()}
    p.publish(t, keys, len(keys), keep=True)
    assert p.save_state(t, keys, state)
    assert not [f for f in os.listdir(p.dir) if f.endswith('.page')]   # pages go to disk only on eviction
    p.release(t)
    x = p.new_table(); assert p.reserve(x, 8 * PAGE)           # evicts both cached pages -> spilled
    p.flush()                                                   # (writes are queued on the writer thread)
    assert len([f for f in os.listdir(p.dir) if f.endswith('.page')]) == 2
    p.release(x)
    q = pool()                                                  # fresh process view of the same cache dir
    longer = torch.cat((keys, torch.tensor([7, 8, 9])))
    tbl, P, st = q.restore(longer)
    assert P == len(keys) and st.keys() == state.keys() and all(torch.equal(st[k], state[k]) for k in state)
    assert same(q, tbl, data, 0, P)
    assert q.restore(keys[:-1])[1] == 0                        # not a prefix of the saved state
    # pages loaded from disk are cached on the GPU now: a plain match shares them
    m, n = q.match(longer)
    assert n == 2 * PAGE and m.pages == tbl.pages[:2]


def test_key_mismatch():
    p = pool(wid='w-A')
    keys = torch.arange(PAGE + 5)
    t = p.new_table(); p.reserve(t, len(keys)); fill(p, t, [kv(len(keys)) for _ in range(8)])
    p.publish(t, keys, len(keys), keep=True); p.save_state(t, keys, {'f': torch.zeros(1, 4)})
    p.persist(t, keys, len(keys))
    assert pool(wid='w-B').restore(keys) == (None, 0, None)    # other weights: different root, no hit
    # a file from other weights placed under the right name (the old "Muffins" incident) is rejected and deleted
    b = pool(wid='w-B')
    src = [f for f in os.listdir(p.dir) if f.endswith('.page')][0]
    dst = b._page_path(b.chain(keys, 1)[1])
    shutil.copy(os.path.join(p.dir, src), dst)
    b = pool(wid='w-B')
    assert b.match(keys)[1] == 0 and not os.path.exists(dst)
    assert pool(wid='w-A').restore(keys)[1] == len(keys)       # the right weights still hit


def test_crash_safety():
    p = pool(wid='w-C')
    keys = torch.arange(50, 50 + PAGE + 3)
    t = p.new_table(); p.reserve(t, len(keys)); fill(p, t, [kv(len(keys)) for _ in range(8)])
    p.publish(t, keys, len(keys)); p.persist(t, keys, len(keys))
    p.save_state(t, keys, {'f': torch.ones(1, 4)}); p.flush()
    page = p._page_path(p.chain(keys, 1)[1])
    open(page + '.123.tmp', 'wb').write(b'half a file')       # crashed mid-write
    with open(page, 'r+b') as f:                                # torn file
        f.truncate(os.path.getsize(page) // 2)
    q = pool(wid='w-C')
    assert not os.path.exists(page + '.123.tmp')
    assert q.restore(keys) == (None, 0, None) and q.match(keys)[1] == 0 and not os.path.exists(page)
    # flipped byte -> checksum
    p._drop_file(page); p.persist(t, keys, len(keys))       # p's index still listed the deleted file
    with open(page, 'r+b') as f:
        f.seek(5000); b = f.read(1); f.seek(5000); f.write(bytes([b[0] ^ 1]))
    assert kvpool._read(page, 'cpu') is None and not os.path.exists(page)


def test_cap_and_free_floor():
    d = tempfile.mkdtemp(prefix='inkcache-cap-')
    page_mb = (2 * 8 * PAGE * NKV * (HD // 2 + HD // 16) + 8 * PAGE) / 2 ** 20
    p = pool(wid='w-D', cache_dir=d, cap_gb=2.5 * page_mb / 1024)
    t = p.new_table(); p.reserve(t, 3 * PAGE); fill(p, t, [kv(3 * PAGE) for _ in range(8)])
    keys = torch.arange(3 * PAGE)
    for i in range(3):                                         # three pages, room for two
        k = keys.clone(); k[0] = i
        p.persist(t, k, PAGE)
    files = sorted(os.listdir(p.dir))
    assert len(files) == 2 and p.used_bytes <= p.cap
    assert not os.path.exists(p._page_path(p.chain(torch.cat((torch.tensor([0]), keys[1:])), 1)[1]))  # oldest gone
    g = pool(wid='w-D', cache_dir=d, min_free_gb=1e9)          # free-space floor: no writes, no deletes
    g.persist(t, keys, PAGE)
    assert sorted(os.listdir(p.dir)) == files and not g.save_state(t, keys[:5], {'f': torch.zeros(1)})
    shutil.rmtree(d)


def test_fork_and_kv4_interface():
    p = pool(cache_dir=None)
    live = p.new_table(); p.reserve(live, 3 * PAGE)
    P = PAGE + 40
    data = [kv(P) for _ in range(8)]
    for slot, (K, V) in enumerate(data):
        live.put(slot, 0, K, V)                                # KV4-style calls through the table
    snap = p.fork(live, P)
    assert snap.pages[0] == live.pages[0] and snap.pages[1] != live.pages[1] and p.ref[live.pages[0]] == 2
    live.put_at(0, torch.tensor([P]), *kv(1))                  # live seq keeps decoding into its tail page
    p.release(live)
    r = p.fork(snap, P); p.reserve(r, P + 10)                  # restore: the new owner writes rows >= P
    r.put(0, P, *kv(10))
    assert same(p, snap, data, 0, P) and same(p, r, data, 0, P) and snap.cap == r.cap == 8 * PAGE


def test_weights_id():
    md = tempfile.mkdtemp(prefix='inkmodel-')
    open(os.path.join(md, 'model-00001-of-00001.safetensors'), 'wb').write(b'abc')
    open(os.path.join(md, 'tokenizer.json'), 'w').write('{}')
    a = kvpool.weights_id(md, D)
    assert kvpool.weights_id(md, D) == a
    open(os.path.join(md, 'model-00001-of-00001.safetensors'), 'wb').write(b'abd')
    assert kvpool.weights_id(md, D) != a
    shutil.rmtree(md)


def test_read_oom_write_oserror():
    """11-23 review: a device failure while reading a good file (CUDA OOM under VRAM pressure) keeps the file and
    returns None; a disk write error (ENOSPC/EIO) in a spill or state save returns False (no exception, no tmp left)."""
    p = pool(wid='w-E')
    keys = torch.arange(70, 70 + PAGE)
    t = p.new_table(); p.reserve(t, len(keys)); fill(p, t, [kv(len(keys)) for _ in range(8)])
    p.publish(t, keys, len(keys)); p.persist(t, keys, len(keys))
    page = p._page_path(p.chain(keys, 1)[1])
    real_to = torch.Tensor.to

    def oom_to(self, *a, **k):
        raise torch.OutOfMemoryError('CUDA out of memory (injected)')
    torch.Tensor.to = oom_to
    try:
        assert kvpool._read(page, 'cpu') is None
    finally:
        torch.Tensor.to = real_to
    assert os.path.exists(page) and kvpool._read(page, 'cpu') is not None
    real_w = kvpool._write

    def full(path, meta, tensors, pause=None):
        open(f'{path}.{os.getpid()}.tmp', 'wb').write(b'partial')
        raise OSError(28, 'No space left on device (injected)')
    kvpool._write = full
    try:
        p.save_state(t, keys, {'f': torch.ones(1, 4)})           # state save path: no exception
        k2 = torch.arange(900, 900 + PAGE)
        t2 = p.new_table(); p.reserve(t2, PAGE); fill(p, t2, [kv(PAGE) for _ in range(8)])
        p.publish(t2, k2, PAGE, True)
        p.persist(t2, k2, PAGE)                                  # spill path: no exception
    finally:
        kvpool._write = real_w
    assert not [f for f in os.listdir(p.dir) if f.endswith('.tmp')]


def test_ckpt_eviction_and_pending():
    """Background writes + mid-prompt checkpoints: restore of a still-queued write reads its host copy; a checkpoint evicts only other
    checkpoints (oldest first) and is skipped rather than evicting an end state or a page; an end state or page spill
    evicts checkpoints first; no room even then -> skipped, nothing deleted; a restart indexes c-/s- files."""
    import threading
    import time as _t
    d = tempfile.mkdtemp(prefix='inkcache-ck-')
    st = {'f': torch.randn(1, 2304).bfloat16(), 'ring.0.k': torch.randn(1023, NKV, HD).bfloat16()}
    sz = sum(kvpool._pad(v.numel() * v.element_size()) for v in st.values()) + 8192
    page_b = 2 * 8 * PAGE * NKV * (HD // 2 + HD // 16) + 8 * PAGE * 8 + 8192
    p = pool(wid='w-K', cache_dir=d, n_tokens=16 * PAGE, cap_gb=(3 * sz + 0.5 * page_b) / 2 ** 30)
    keys = torch.arange(5000, 5000 + 6 * PAGE)
    t = p.new_table(); p.reserve(t, len(keys)); fill(p, t, [kv(len(keys)) for _ in range(8)])
    p.publish(t, keys, len(keys), keep=True)
    gate, real_w = threading.Event(), kvpool._write

    def slow(path, meta, tensors, pause=None):
        gate.wait(5)
        real_w(path, meta, tensors)
    kvpool._write = slow
    try:
        assert p.save_ckpt(keys[:PAGE], st) and p.pending and not os.listdir(p.dir)
        threading.Timer(0.3, gate.set).start()
        t0 = _t.time()
        tbl, P, got = p.restore(keys[:PAGE + 3])                # still queued: read from its host copy, no wait
        dt, on_disk = _t.time() - t0, os.listdir(p.dir)
        p.flush()
    finally:
        kvpool._write = real_w
    assert P == PAGE and dt < 0.25 and not on_disk and torch.equal(got['ring.0.k'], st['ring.0.k']), (P, dt, on_disk)
    assert os.listdir(p.dir) and not p.mem                     # written after all; the host copy is gone
    p.release(tbl)
    assert not p.save_ckpt(keys[:PAGE], st)                     # exists: nothing queued
    for i in (2, 3):
        assert p.save_ckpt(keys[:i * PAGE], st)
        p.flush(); _t.sleep(0.02)
    p.save_state(t, keys[:4 * PAGE], st); p.flush()             # end state: room comes from the oldest checkpoint
    names = lambda: sorted(f.split('-')[0] + f.split('-')[1] for f in os.listdir(p.dir))
    assert names() == ['c' + str(2 * PAGE), 'c' + str(3 * PAGE), 's' + str(4 * PAGE)], names()
    p.persist(t, keys, PAGE)                                    # a page spill: evicts the next checkpoint
    assert sum(f.startswith('c-') for f in os.listdir(p.dir)) == 1 and sum(f.endswith('.page') for f in os.listdir(p.dir)) == 1
    p.persist(t, keys, 2 * PAGE)                                # second page: the last checkpoint goes
    assert not [f for f in os.listdir(p.dir) if f.startswith('c-')]
    before = sorted(os.listdir(p.dir))
    assert not p.save_ckpt(keys[:5 * PAGE], st)                 # only an end state + pages left: skipped
    p.flush()
    assert sorted(os.listdir(p.dir)) == before
    q = pool(wid='w-K', cache_dir=d, n_tokens=16 * PAGE)        # restart: c-/s- files indexed
    assert set(q.states) == {4 * PAGE} and q.used_bytes == p.used_bytes, (q.states, q.used_bytes, p.used_bytes)
    shutil.rmtree(d)


def test_batched_page_load_and_evict_set():
    """match() loads a run of disk pages in one batch (one allocation, reader threads) == the original data; a corrupt
    page in the middle ends the prefix there (and is deleted); the incremental evictable set == a full scan."""
    d = tempfile.mkdtemp(prefix='inkcache-batch-')
    p = pool(wid='w-L', cache_dir=d, n_tokens=12 * PAGE)
    keys = torch.arange(300, 300 + 6 * PAGE)
    data = [kv(len(keys)) for _ in range(8)]
    t = p.new_table(); p.reserve(t, len(keys)); fill(p, t, data)
    p.publish(t, keys, len(keys), keep=True); p.release(t)
    scan = lambda: sorted(n.key for n in p.nodes.values() if p.ref[n.page] == 1)
    assert scan() == sorted(p.evict) and len(p.evict) == 6
    x = p.new_table(); assert p.reserve(x, 12 * PAGE); p.release(x)   # evict all 6 -> spilled (batched)
    p.flush()
    assert not p.nodes and not p.evict and len([f for f in os.listdir(p.dir) if f.endswith('.page')]) == 6
    m, n = p.match(keys)
    assert n == 6 * PAGE and same(p, m, [(K[:n], V[:n]) for K, V in data], 0, n) and scan() == sorted(p.evict) == []
    p.release(m)
    assert scan() == sorted(p.evict) and len(p.evict) == 6
    x = p.new_table(); assert p.reserve(x, 12 * PAGE); p.release(x); p.flush()
    bad = p._page_path(p.chain(keys, 4)[4])                    # page 4 corrupt
    with open(bad, 'r+b') as f:
        f.seek(9000); b = f.read(1); f.seek(9000); f.write(bytes([b[0] ^ 1]))
    m, n = p.match(keys)
    assert n == 3 * PAGE and not os.path.exists(bad) and bad not in p.files, n
    assert same(p, m, [(K[:n], V[:n]) for K, V in data], 0, n) and scan() == sorted(p.evict)
    p.release(m)
    assert p.free_pages() + len(p.evict) == p.n - 1             # no page leaked (pages 5, 6 loaded, cached)
    shutil.rmtree(d)


def test_restore_never_waits_behind_writes():
    """Window 11-23 (checkpoints on: disk restore 0.7 s vs 0.2 s): a restore must not wait for unrelated queued writes.
    Slow writer (0.4 s per file); 3 checkpoints of another chat queued first, then the target's end state, then its pages
    spilled by an eviction: restore() gets the state and every page from their host copies at once (== the data), while
    the writer still works through its queue. Reads go first: the writer pauses (between tensors) while one runs."""
    import threading
    import time as _t
    d = tempfile.mkdtemp(prefix='inkcache-q-')
    p = pool(wid='w-Q', cache_dir=d, n_tokens=8 * PAGE)
    st = {'f': torch.randn(1, 2304).bfloat16(), 'ring.0.k': torch.randn(1023, NKV, HD).bfloat16()}
    real_w, slept = kvpool._write, []

    def slow(path, meta, tensors, pause=None):
        _t.sleep(0.4)
        slept.append(os.path.basename(path)[:2])
        real_w(path, meta, tensors, pause)
    kvpool._write = slow
    try:
        other = torch.arange(9000, 9000 + 4 * PAGE)
        t2 = p.new_table(); p.reserve(t2, len(other)); fill(p, t2, [kv(len(other)) for _ in range(8)])
        for i in (1, 2, 3):
            p.publish(t2, other[:i * PAGE], i * PAGE, keep=True)
            assert p.save_ckpt(other[:i * PAGE], st)
        p.release(t2)
        keys = torch.arange(100, 100 + 2 * PAGE + 5)
        data = [kv(len(keys)) for _ in range(8)]
        t = p.new_table(); p.reserve(t, len(keys)); fill(p, t, data)
        p.publish(t, keys, len(keys), keep=True)
        assert p.save_state(t, keys, st)
        p.release(t)
        x = p.new_table(); assert p.reserve(x, 8 * PAGE); p.release(x)  # evict all: the target's 2 pages spill (queued)
        assert len(p.pending) >= 6 and not p.nodes, len(p.pending)
        t0 = _t.time()
        tbl, P, got = p.restore(keys)
        dt = _t.time() - t0
        busy = len([f for f in p.pending.values() if not f.done()])
        assert P == len(keys) and dt < 0.3 and busy >= 3, (P, dt, busy)
        assert torch.equal(got['ring.0.k'], st['ring.0.k']) and same(p, tbl, data, 0, P), 'restored data'
        p.release(tbl)
        p.flush()
        # reads first: with a read running, a queued write doesn't start its tensors
        k3 = torch.arange(20000, 20000 + PAGE)
        with p._reading():
            assert p.save_ckpt(k3[:PAGE], st)
            path = next(reversed(p.pending)) if p.pending else None
            _t.sleep(0.8)
            assert path is not None and not os.path.exists(path), 'writer paused while a read runs'
        p.flush()
        assert os.path.exists(path)
    finally:
        kvpool._write = real_w
    shutil.rmtree(d)


def test_capture_vs_writer_thread():
    """Prod 11-23: the writer thread's ready() (cudaEventSynchronize) during a global-mode capture on the main thread
    failed the write and invalidated the capture (500s). Every engine capture must be thread_local, and the writer
    must make no CUDA call of its own besides ready()."""
    import glob as _g, inspect, re
    eng = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    n = 0
    for f in _g.glob(os.path.join(eng, '*.py')):
        for m in re.finditer(r'torch\.cuda\.graph\(([^)]*)\)', open(f).read()):
            n += 1
            assert "capture_error_mode='thread_local'" in m.group(1), f'{os.path.basename(f)}: {m.group(0)}'
    assert n >= 3, n
    for fn in (kvpool._write_bg, kvpool._write):
        assert 'cuda' not in inspect.getsource(fn).split('"""')[-1], fn.__name__


if __name__ == '__main__':
    try:
        for name, fn in list(globals().items()):
            if name.startswith('test_'):
                fn(); print(name, 'OK')
        print('ALL OK')
    finally:
        shutil.rmtree(D)
