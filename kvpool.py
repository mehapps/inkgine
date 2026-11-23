"""Paged 4-bit KV pool for the 8 full-attention visits, plus a disk prefix cache (no host-RAM tier, by decision).

GPU: fixed pages of PAGE tokens. One page id covers a token range for all 8 full visits, so a sequence has one page
table (Table) shared by the visits. Per token and visit the format is KV4's (int4 codes, low nibble = even element,
e4m3 scale per 16 values; 9 KB/token over the 8 visits), and get() dequantizes with KV4.get's exact math.
Page 0 is a zero page that is never allocated: unused table entries point at it, so fixed-size reads past a sequence's
end (decode far buckets) gather finite values that the caller masks.

Prefix tree: chain-hashed full pages, key_i = H(key_{i-1}, token keys of page i), key_0 = H(weights id, FORMAT). The
cache holds one ref on every published page; a page nobody else references is evictable (LRU, deepest first) and is
spilled to disk first if it was published with keep=True. Nodes exist only for GPU-resident pages; disk presence is
the file index. A node found by key is correct on its own (the key encodes the whole prefix, tokens are re-checked).

Disk (INK_CACHE_DIR, default ~/llm/inkcache/<root hex>/): one file per page (~2.3 MB) or per sequence state, written
to a .tmp name then renamed; header holds weights id / FORMAT / key; crc32 of everything at the end (integrity only:
keys, weights and format are checked in the header); header and tensors 64-byte aligned, so a read is one (pinned)
buffer whose tensors are views.
Truncated, corrupt or mismatching files are deleted when read. LRU by mtime over every file under the cache dir (old
weights' roots go first), capped at INK_CACHE_GB and never writing below INK_CACHE_MIN_FREE_GB free on the filesystem.
One server process per cache dir (startup deletes leftover .tmp files).
Writes never block on the disk: every file (page spill, end state, checkpoint) is copied device -> pinned host memory with
non-blocking copies on the current stream (so they read the data before any later kernel can overwrite or free it)
and written by one writer thread; the index (files, states) is updated at once on the caller's thread, and a reader or
eviction of a file still being written waits for it (_settle). Host bytes in flight are capped (INK_CACHE_PIN_MB).
Page loads (match) are batched: one allocation for the missing run, files read by 4 threads into pinned buffers,
non-blocking copies into the pool. Evictable pages are tracked incrementally (no scan of the tree per eviction).
Mid-prompt checkpoints (save_ckpt, engine INK_CKPT_EVERY): a state file like an end state (`c-` instead of `s-`, same key
and restore path). Checkpoints are evicted first (LRU among them), and a checkpoint save evicts only other checkpoints.
"""
import glob
import hashlib
import heapq
import json
import math
import os
import shutil
import struct
import threading
import time
import zlib
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor

import torch

from model import FULL_VISITS, HD, KV4, NKV

PAGE = 256
NV = len(FULL_VISITS)
FORMAT = f'kv4-int4-e4m3x16-page{PAGE}-visits{NV}-v1'  # bump on any change to the page/state layout or KV math
CACHE_DIR = os.environ.get('INK_CACHE_DIR', os.path.expanduser('~/llm/inkcache'))
CACHE_GB = float(os.environ.get('INK_CACHE_GB', 200))
MIN_FREE_GB = float(os.environ.get('INK_CACHE_MIN_FREE_GB', 100))
MAGIC = b'INKC2\n'  # INKC1 files (blake2b, unaligned) fail the magic check and are deleted on read
PIN_MB = int(os.environ.get('INK_CACHE_PIN_MB', 1024))  # pinned host bytes of queued writes before the caller waits


def H(*parts):
    h = hashlib.blake2b(digest_size=16)
    for p in parts:
        h.update(p)
    return h.digest()


def deq(c, s):
    """codes [n,8,64] uint8 + scales [n,8,8] e4m3 -> bf16 [n,8,128]; the exact math of KV4.get."""
    n = c.shape[0]
    q = torch.empty(n, NKV, HD // 2, 2, dtype=torch.bfloat16, device=c.device)
    q[..., 0], q[..., 1] = c & 15, c >> 4  # nibbles straight into the bf16 output (no uint8 stack temp; VRAM at 64k rows)
    q = q.view(n, NKV, HD // 16, 16).sub_(8)
    return q.mul_(s.to(torch.bfloat16).unsqueeze(-1)).view(n, NKV, HD)


def weights_id(model_dir, cache_dir=CACHE_DIR):
    """blake2b of every *.safetensors + tokenizer.json in model_dir (full contents). Memoized in cache_dir by
    (name, size, mtime_ns, inode), so only the first start after a weights change pays the ~10 s read."""
    files = sorted(f for f in os.listdir(model_dir) if f.endswith('.safetensors') or f == 'tokenizer.json')
    stat = [[f, s.st_size, s.st_mtime_ns, s.st_ino] for f in files for s in [os.stat(os.path.join(model_dir, f))]]
    memo = os.path.join(cache_dir, 'weights_id.json')
    try:
        with open(memo) as fh:
            m = json.load(fh)
        if m['dir'] == os.path.abspath(model_dir) and m['stat'] == stat:
            return m['id']
    except (OSError, ValueError, KeyError, TypeError):
        pass
    h = hashlib.blake2b(digest_size=16)
    for f in files:
        h.update(f.encode())
        with open(os.path.join(model_dir, f), 'rb') as fh:
            while b := fh.read(16 << 20):
                h.update(b)
    wid = h.hexdigest()
    os.makedirs(cache_dir, exist_ok=True)
    with open(memo + '.tmp', 'w') as fh:
        json.dump({'dir': os.path.abspath(model_dir), 'stat': stat, 'id': wid}, fh)
    os.replace(memo + '.tmp', memo)
    return wid


ALIGN = 64  # header and every tensor start on a 64-byte boundary: a read is one buffer, tensors are views of it


def _pad(n):
    return -(-n // ALIGN) * ALIGN


def _write(path, meta, tensors, pause=None):
    """Header (padded) + raw tensors (each padded) + crc32 of everything before it, then an atomic rename. tensors: host,
    contiguous (written straight from their memory, no copies). No fsync: a file torn by a power loss fails its
    checksum and is deleted on read. pause(): called before each tensor (the writer yields the disk to reads)."""
    hb = json.dumps(dict(meta, t=[(k, str(v.dtype).split('.')[-1], list(v.shape)) for k, v in tensors.items()])).encode()
    hb += b' ' * (_pad(len(MAGIC) + 4 + len(hb)) - len(MAGIC) - 4 - len(hb))
    tmp, crc = f'{path}.{os.getpid()}.tmp', 0
    with open(tmp, 'wb') as f:
        for v in [MAGIC, struct.pack('<I', len(hb)), hb] + list(tensors.values()):
            if pause is not None:
                pause()
            b = v if isinstance(v, bytes) else v.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
            crc = zlib.crc32(b, crc)
            f.write(b)
            if (z := _pad(len(b)) - len(b)) and not isinstance(v, bytes):
                crc = zlib.crc32(bytes(z), crc)
                f.write(bytes(z))
        f.write(struct.pack('<I', crc))
    os.replace(tmp, path)


def _read_host(path, pin=False):
    """-> (meta, {name: host tensor}: views of one buffer, pinned if pin), or None if the file is missing, truncated or
    corrupt (then it is deleted). Safe on a reader thread (file I/O and crc32 release the GIL; no index access)."""
    try:
        f = open(path, 'rb')
    except FileNotFoundError:
        return None
    except OSError:
        return None
    with f:
        n = os.fstat(f.fileno()).st_size
        buf = torch.empty(n, dtype=torch.uint8, pin_memory=pin)  # (an allocation failure keeps the file: raises)
        try:
            mv = memoryview(buf.numpy())
            if f.readinto(mv) != n or n < len(MAGIC) + 8 or mv[:len(MAGIC)] != MAGIC:
                raise ValueError('bad magic / short')
            o = len(MAGIC) + 4 + struct.unpack_from('<I', mv, len(MAGIC))[0]
            if o > n - 4 or zlib.crc32(mv[:n - 4]) != struct.unpack_from('<I', mv, n - 4)[0]:
                raise ValueError('checksum')
            meta, out = json.loads(bytes(mv[len(MAGIC) + 4:o])), {}
            for name, dt, shape in meta.pop('t'):
                dtype = getattr(torch, dt)
                nb = math.prod(shape) * dtype.itemsize
                if o + nb > n - 4:
                    raise ValueError('truncated')
                out[name] = buf[o:o + nb].view(dtype).view(shape)
                o += _pad(nb)
            if o != n - 4:
                raise ValueError('size')
            return meta, out
        except (OSError, ValueError, KeyError, TypeError, AttributeError, struct.error):
            pass
    try:
        os.remove(path)
    except OSError:
        pass
    return None


def _read(path, dev):
    """-> (meta, {name: tensor on dev}), or None if the file is missing, truncated or corrupt (then it is deleted).
    cuda: read into pinned memory, checked on the host, then non-blocking copies on the current stream; a device
    failure (CUDA OOM under VRAM pressure) keeps the good file."""
    dev = torch.device(dev)
    try:
        r = _read_host(path, pin=dev.type == 'cuda')
        return r and (r[0], {k: v.to(dev, non_blocking=True) for k, v in r[1].items()})
    except RuntimeError as e:
        print(f'cache read skipped (file kept): {type(e).__name__}: {e}'[:200], flush=True)
        return None


class Table:
    """A sequence's page table: host list + a fixed-address device copy (CUDA-graph safe; unused entries = page 0).
    Also the sequence's KV store with KV4's interface (put / put_at / get by full-visit slot, cap), so model code and
    test fakes keep calling seq.kv.*."""

    def __init__(self, pool):
        self.pool, self.cap = pool, (pool.n - 1) * PAGE
        self.pages, self.t = [], torch.zeros(pool.n, dtype=torch.long, device=pool.dev)

    def put(self, slot, p0, K, V):
        self.pool.put(self, slot, p0, K, V)

    put_at = put

    def get(self, slot, a, b):
        return self.pool.get(self, slot, a, b)

    def __del__(self):
        # shortcut: a dropped Seq gives its pages back (tests/tools build seqs straight from Model); the engine releases
        # explicitly. Upgrade to explicit-only if tables ever end up in reference cycles.
        try:
            for p in self.pages:
                self.pool._unref(p)
        except Exception:
            pass


class Node:
    __slots__ = ('key', 'keys', 'depth', 'page', 'used', 'keep')

    def __init__(self, key, keys, depth, page):
        self.key, self.keys, self.depth, self.page, self.used, self.keep = key, keys, depth, page, 0, False


class KVPool:
    def __init__(self, n_tokens, dev, wid, cache_dir=CACHE_DIR, cap_gb=CACHE_GB, min_free_gb=MIN_FREE_GB):
        self.dev = torch.device(dev)
        self.n = -(-n_tokens // PAGE) + 1                      # + the zero page
        self.c = torch.zeros(2, NV, self.n, PAGE, NKV, HD // 2, dtype=torch.uint8, device=self.dev)
        self.s = torch.zeros(2, NV, self.n, PAGE, NKV, HD // 16, dtype=torch.float8_e4m3fn, device=self.dev)
        self.ref = [0] * self.n
        self.free = list(range(self.n - 1, 0, -1))              # pop() hands out low ids first
        self.wid, self.root = wid, H(wid.encode(), FORMAT.encode())
        self.nodes, self.clock = {}, 0                          # key -> Node (GPU-resident published pages)
        self.dir, self.files, self.states, self.used_bytes = None, {}, {}, 0
        self.pnode, self.evict = {}, {}                         # page -> its Node; key -> Node with ref 1 (evictable)
        self.pending, self.writer, self.readers = {}, None, None  # path -> Future of its background write
        self.mem = {}                                           # path -> (meta, host tensors, ready) of a queued write
        self.rd, self.rdc = 0, threading.Condition()           # reads in progress (the writer pauses for them)
        if cache_dir:
            self.base, self.cap, self.min_free = cache_dir, int(cap_gb * 2 ** 30), int(min_free_gb * 2 ** 30)
            self.dir = os.path.join(cache_dir, self.root.hex())
            os.makedirs(self.dir, exist_ok=True)
            for d, _, fs in os.walk(cache_dir):                 # every root: other weights' files count toward the cap
                for f in fs:
                    p = os.path.join(d, f)
                    if f.endswith('.tmp'):
                        os.remove(p)                            # a write that crashed before its rename
                    elif f.endswith(('.page', '.state')):
                        st = os.stat(p)
                        self.files[p] = [st.st_size, st.st_mtime]
                        self.used_bytes += st.st_size
                        if d == self.dir and f[:2] in ('s-', 'c-'):
                            P, k = f[2:-6].split('-')
                            self.states.setdefault(int(P), set()).add(k)

    # ------------------------------------------------------------------ pages
    def free_pages(self):
        return len(self.free)

    def _evictable(self):
        return list(self.evict.values())

    def can_admit(self, n_tokens, tbl=None):
        """Could reserve(tbl, n_tokens) succeed (free pages + evictable cached pages)?"""
        need = -(-n_tokens // PAGE) - (len(tbl.pages) if tbl else 0)
        return need <= len(self.free) + len(self.evict)

    def new_table(self):
        return Table(self)

    def fork(self, tbl, P):
        """New table for positions [0, P) of tbl: full pages shared (refcount), the partial last page copied into a
        private page (the new owner writes rows >= P there). None if no page is free. Used to park an end-of-prompt
        snapshot on the GPU while the live sequence keeps decoding, and to restore it later."""
        k = P // PAGE
        ids = self._alloc(1) if P > k * PAGE else []
        if ids is None:
            return None
        if ids:
            self.c[:, :, ids[0]], self.s[:, :, ids[0]] = self.c[:, :, tbl.pages[k]], self.s[:, :, tbl.pages[k]]
        for p in tbl.pages[:k]:
            self._ref(p)
        new = self.new_table()
        self._push(new, tbl.pages[:k] + ids)
        return new

    def _alloc(self, k):
        if len(self.free) < k:
            self._evict(k - len(self.free))
        if len(self.free) < k:
            return None
        ids = [self.free.pop() for _ in range(k)]
        for i in ids:
            self.ref[i] = 1
        return ids

    def _ref(self, p, d=1):
        """ref[p] += d, keeping the evictable set (cached pages only the cache holds) and the free list current."""
        self.ref[p] += d
        if (nd := self.pnode.get(p)) is not None:
            if self.ref[p] == 1:
                self.evict[nd.key] = nd
            else:
                self.evict.pop(nd.key, None)
        if self.ref[p] == 0:
            self.free.append(p)

    def _unref(self, p):
        self._ref(p, -1)

    def _add_node(self, nd):
        """Register a cached page; the caller holds the cache's ref (refcount already counts it)."""
        self.nodes[nd.key], self.pnode[nd.page] = nd, nd
        self._ref(nd.page, 0)

    def _drop_node(self, nd):
        """Remove a cached page from the tree and drop the cache's ref."""
        del self.nodes[nd.key]
        del self.pnode[nd.page]
        self.evict.pop(nd.key, None)
        self._unref(nd.page)

    def _push(self, tbl, ids):
        a = len(tbl.pages)
        tbl.pages += ids
        if ids:
            tbl.t[a:a + len(ids)] = torch.tensor(ids, dtype=torch.long)

    def reserve(self, tbl, n_tokens):
        """Grow tbl to cover positions [0, n_tokens) (prompt + max_tokens); evicts cached pages if needed.
        False (nothing allocated) if the pool is short: the scheduler waits or preempts."""
        ids = self._alloc(max(0, -(-n_tokens // PAGE) - len(tbl.pages)))
        if ids is None:
            return False
        self._push(tbl, ids)
        return True

    def release(self, tbl):
        for p in tbl.pages:
            self._unref(p)
        tbl.pages = []
        tbl.t.zero_()

    def _evict(self, k):
        """Free k cached pages (LRU, deepest first); kept ones are queued for disk first (batched, never blocking on it)."""
        out = heapq.nsmallest(k, self.evict.values(), key=lambda n: (n.used, -n.depth))
        spill = [n for n in out if n.keep and self.dir and self._page_path(n.key) not in self.files]
        for i in range(0, len(spill), 64):  # (one host copy per <= 64 pages, ~150 MB pinned)
            self._save_many([self._page_item(n.key, n.page, n.keys) for n in spill[i:i + 64]])
        for n in out:
            self._drop_node(n)

    # ------------------------------------------------------------------ K/V
    def put(self, tbl, slot, p0, K, V):
        """Write K/V [n,8,128] bf16 at positions p0.. of full-visit slot (0..7). p0: int, or an int64 device
        tensor [1] for decode (no host sync, CUDA-graph safe). Pages must be reserved."""
        n = K.shape[0]
        if isinstance(p0, int) and p0 + n > len(tbl.pages) * PAGE:  # grow (the engine reserves up front; tools may not)
            assert self.reserve(tbl, p0 + n), f'KV pool full ({self.n - 1} pages)'
        pos = torch.arange(n, device=self.dev) + p0
        idx = tbl.t[pos // PAGE] * PAGE + pos % PAGE
        for j, x in enumerate((K, V)):
            q, s = KV4.quant(x)
            self.c[j, slot].view(-1, NKV, HD // 2).index_copy_(0, idx, q)
            self.s[j, slot].view(torch.uint8).view(-1, NKV, HD // 16).index_copy_(0, idx, s.view(torch.uint8))

    def get(self, tbl, slot, a, b):
        """Dequantized bf16 (K, V) [b-a,8,128] for positions [a, b) of slot: one gather per tensor over the page ids."""
        p0 = a // PAGE
        idx = tbl.t[p0:-(-b // PAGE)]
        lo, hi = a - p0 * PAGE, b - p0 * PAGE
        return [deq(self.c[j, slot].index_select(0, idx).view(-1, NKV, HD // 2)[lo:hi],
                    self.s[j, slot].view(torch.uint8).index_select(0, idx).view(-1, NKV, HD // 16)[lo:hi]
                    .view(torch.float8_e4m3fn)) for j in range(2)]

    # ------------------------------------------------------------------ prefix tree
    def chain(self, keys, k):
        """[key_0 .. key_k] for the first k full pages of keys (int64 cpu tensor: token ids, images as negative hashes)."""
        out, b, w = [self.root], keys[:k * PAGE].numpy().tobytes(), PAGE * 8
        for i in range(k):
            out.append(H(out[-1], b[i * w:(i + 1) * w]))
        return out

    def publish(self, tbl, keys, n, keep=False):
        """Register tbl's full pages covering keys[:n] (end of a prompt or turn; all 8 visits written). keep: worth a
        disk copy when evicted (long or likely-continued conversation). A block already cached under another page
        stays private to tbl."""
        k = n // PAGE
        chain = self.chain(keys, k)
        self.clock += 1
        for i in range(k):
            nd = self.nodes.get(chain[i + 1])
            if nd is None and tbl.pages[i] not in self.pnode:  # (a page is cached under one key only)
                nd = Node(chain[i + 1], keys[i * PAGE:(i + 1) * PAGE].clone(), i, tbl.pages[i])
                self.ref[nd.page] += 1
                self._add_node(nd)
            if nd is not None:
                nd.used, nd.keep = self.clock, nd.keep or keep

    def match(self, keys, k=None):
        """-> (new Table sharing the longest cached page prefix of keys (GPU, else loaded from disk), tokens covered)."""
        k = len(keys) // PAGE if k is None else k
        chain, plan = self.chain(keys, k), []
        self.clock += 1
        for i in range(k):
            blk = keys[i * PAGE:(i + 1) * PAGE]
            nd = self.nodes.get(chain[i + 1])
            if nd is not None:
                if not torch.equal(nd.keys, blk):
                    break
                self._ref(nd.page)                               # matched pages are never evicted by later loads
                plan.append(nd)
            elif self.dir and (p := self._page_path(chain[i + 1])) in self.files:
                plan.append((chain[i + 1], p, blk, i))
            else:
                break
        disk = [x for x in plan if not isinstance(x, Node)]
        got = self._load_page(disk) if disk else {}
        ids, ok = [], True
        for x in plan:
            nd = x if isinstance(x, Node) else got.get(x[0])
            if ok and nd is not None:
                if not isinstance(x, Node):
                    self._ref(nd.page)
                nd.used = self.clock
                ids.append(nd.page)
            else:  # the prefix ends at the first page that couldn't be loaded
                ok = False
                if isinstance(x, Node):
                    self._unref(x.page)
        tbl = self.new_table()
        self._push(tbl, ids)
        return tbl, len(ids) * PAGE

    # ------------------------------------------------------------------ disk
    def _page_path(self, key):
        return os.path.join(self.dir, key.hex() + '.page')

    def _page_item(self, key, page, blk):
        return self._page_path(key), {'kind': 'page', 'key': key.hex()}, \
            {'c': self.c[:, :, page], 's': self.s[:, :, page], 'keys': blk}

    def persist(self, tbl, keys, n):
        """Write tbl's full pages for keys[:n] to disk now (skips pages already there; waits for the writes). Normally
        pages reach disk only when evicted (publish keep=True); use this e.g. before a planned shutdown."""
        chain = self.chain(keys, n // PAGE)
        todo = [self._page_item(chain[i + 1], tbl.pages[i], keys[i * PAGE:(i + 1) * PAGE]) for i in range(n // PAGE)
                if self._page_path(chain[i + 1]) not in self.files]
        for i in range(0, len(todo), 64):
            self._save_many(todo[i:i + 64])
        self.flush()

    def _load_page(self, items):
        """Load cached page files [(key, path, keys block, depth)] into the pool: one allocation (as many as fit), 4
        reader threads into pinned buffers, non-blocking copies. -> {key: Node} for the pages loaded (cache ref only).
        A missing, corrupt or mismatching file is dropped; a host allocation failure keeps it."""
        if not items:
            return {}
        ids = self._alloc(min(len(items), len(self.free) + len(self.evict))) or []
        self.readers = self.readers or ThreadPoolExecutor(4, thread_name_prefix='inkread')
        pin, got = self.dev.type == 'cuda', {}
        with self._reading():
            self._load_run(items, ids, pin, got)
        for pid in ids[len(items):]:
            self._unref(pid)
        return got

    def _load_run(self, items, ids, pin, got):
        for j in range(0, len(ids), 32):  # (pinned staging bounded to ~32 pages in flight)
            # a page still queued for writing comes from its host copy (no wait behind the writer's queue)
            futs = [self._mem(p) or self.readers.submit(_read_host, p, pin) for _, p, _, _ in items[j:j + 32]]
            for (key, p, blk, depth), pid, fu in zip(items[j:j + 32], ids[j:j + 32], futs):
                try:
                    r = fu if isinstance(fu, tuple) else fu.result()
                except RuntimeError as e:  # (pinned allocation): keep the file
                    print(f'cache page read skipped (file kept): {e}'[:200], flush=True)
                    self._unref(pid)
                    continue
                if r is None or not self._meta_ok(r[0], 'page', key.hex()) or r[1]['c'].shape != self.c[:, :, 0].shape \
                        or r[1]['s'].shape != self.s[:, :, 0].shape or not torch.equal(r[1]['keys'], blk):
                    self._drop_file(p)
                    self._unref(pid)
                    continue
                try:
                    self.c[:, :, pid].copy_(r[1]['c'], non_blocking=True)
                    self.s[:, :, pid].copy_(r[1]['s'], non_blocking=True)
                except RuntimeError as e:  # (device failure under VRAM pressure): keep the file
                    print(f'cache page read skipped (file kept): {type(e).__name__}: {e}'[:200], flush=True)
                    self._unref(pid)
                    continue
                self._touch(p)
                nd = got[key] = Node(key, blk.clone(), depth, pid)  # the alloc ref is the cache's ref
                nd.keep = True
                self._add_node(nd)

    @contextmanager
    def _reading(self):
        """Disk reads (restores) go first: the writer thread pauses between tensors while one runs."""
        with self.rdc:
            self.rd += 1
        try:
            yield
        finally:
            with self.rdc:
                self.rd -= 1
                self.rdc.notify_all()

    def _pause(self):
        with self.rdc:  # (bounded: no reader waits on the writer inside _reading, but never hang on it)
            self.rdc.wait_for(lambda: not self.rd, timeout=2.0)

    def _mem(self, path):
        """A write still queued for path: (meta, host tensors) straight from its host copy, else None. Restores of a
        just-unparked state or a just-spilled page never wait for the writer's queue (other files ahead of it)."""
        m = self.mem.get(path) if path in self.pending else None
        if m is None:
            return None
        if m[2] is not None:
            m[2]()                                               # its device -> host copy has landed
        return m[0], m[1]

    def _meta_ok(self, meta, kind, key):
        return meta.get('kind') == kind and meta.get('weights') == self.wid and meta.get('format') == FORMAT \
            and meta.get('key') == key

    def _state_key(self, keys, P, chain):
        return H(chain[P // PAGE], keys[P // PAGE * PAGE:P].numpy().tobytes(), str(P).encode()).hex()

    def save_state(self, tbl, keys, tensors):
        """Disk snapshot of a sequence at P = len(keys): `tensors` (rings, conv states, last hidden; any device) plus
        the KV rows of its partial last page. Call at end of prompt/turn only for conversations likely to continue
        (SSD wear); publish(keep=True) the full pages too, or restore() will find no pages after they are evicted.
        Queued for the writer thread (the device -> host copy is enqueued now: the caller may free or reuse the
        tensors and pages right away)."""
        if not self.dir:
            return False
        P = len(keys)
        k = self._state_key(keys, P, self.chain(keys, P // PAGE))
        path = os.path.join(self.dir, f's-{P}-{k}.state')
        if have := self._state_path(P, k):                     # (a checkpoint at the same point has the same content)
            self._touch(have)
            return True
        a, t = P // PAGE * PAGE, {}
        if P > a:
            pg = tbl.pages[P // PAGE]
            t = {'tc': self.c[:, :, pg, :P - a], 'ts': self.s[:, :, pg, :P - a]}
        ok = bool(self._save_many([(path, {'kind': 'state', 'key': k, 'P': P},
                                    {**t, **{'x.' + n: v for n, v in tensors.items()}})]))
        if ok:
            self.states.setdefault(P, set()).add(k)
        return ok

    def restore(self, keys, min_len=0):
        """-> (tbl, P, tensors) for the longest saved state (P > min_len) whose keys are a prefix of `keys` and whose full pages are
        on the GPU or disk; tbl covers [0, P) (shared pages + a private page with the state's tail rows). Else
        (None, 0, None). The caller then reserve()s prompt + max_tokens and prefills keys[P:]."""
        if not self.dir:
            return None, 0, None
        chain = self.chain(keys, len(keys) // PAGE)
        for P in sorted((P for P in self.states if min_len < P <= len(keys)), reverse=True):
            k = self._state_key(keys, P, chain)
            if k not in self.states[P] or not all(chain[i] in self.nodes or self._page_path(chain[i]) in self.files
                                                  for i in range(1, P // PAGE + 1)):
                continue
            tbl, n = self.match(keys, P // PAGE)
            path = self._state_path(P, k)
            if n == P // PAGE * PAGE and self.reserve(tbl, P):
                r = self._mem(path)                              # queued for writing: from memory, no wait
                if r is not None:
                    r = r[0], {name: v.to(self.dev, non_blocking=True, copy=True) for name, v in r[1].items()}
                elif self._settle(path):
                    with self._reading():
                        r = _read(path, self.dev)
                    if r is None or not (self._meta_ok(r[0], 'state', k) and r[0].get('P') == P):
                        self._drop_file(path)
                        r = None
                if r is not None:
                    if P > n:
                        self.c[:, :, tbl.pages[-1], :P - n], self.s[:, :, tbl.pages[-1], :P - n] = r[1]['tc'], r[1]['ts']
                    self._touch(path)
                    return tbl, P, {name[2:]: v for name, v in r[1].items() if name.startswith('x.')}
            self.release(tbl)
        return None, 0, None

    def _state_path(self, P, k):
        """Path of the saved state (end state or checkpoint) for key k at P, or None."""
        for x in 'sc':
            if (p := os.path.join(self.dir, f'{x}-{P}-{k}.state')) in self.files:
                return p
        return None

    def save_ckpt(self, keys, tensors):
        """Mid-prompt checkpoint at P = len(keys) (a multiple of PAGE; the caller published the full pages, keep=True):
        tensors as for save_state (any device; copied now, written in the background). Skipped if a state exists at this
        key or there's no room (other checkpoints are evicted for it, nothing else). -> queued."""
        P = len(keys)
        assert self.dir and P % PAGE == 0, P
        k = self._state_key(keys, P, self.chain(keys, P // PAGE))
        if have := self._state_path(P, k):
            self._touch(have)
            return False
        path = os.path.join(self.dir, f'c-{P}-{k}.state')
        if not self._save_many([(path, {'kind': 'state', 'key': k, 'P': P}, {'x.' + n: v for n, v in tensors.items()})],
                               ckpt=True):
            return False
        self.states.setdefault(P, set()).add(k)
        return True

    def _settle(self, path):
        """Wait for path's background write, if any; a failed write is dropped from the index. -> the file is usable."""
        f = self.pending.pop(path, None)
        self.mem.pop(path, None)
        if f is None:
            return path in self.files
        if not f.result():
            self._drop_file(path)
            return False
        if path in self.files:                                   # estimated size -> real size
            n = os.path.getsize(path)
            self.used_bytes += n - self.files[path][0]
            self.files[path][0] = n
        return True

    def _reap(self):
        for p in [p for p, f in self.pending.items() if f.done()]:
            self._settle(p)

    def flush(self):
        """Wait for every pending checkpoint write (tests, shutdown)."""
        for p in list(self.pending):
            self._settle(p)

    def _room(self, size, ckpt=False):
        """Make room for a new file: delete cache files (checkpoints first, then the rest; LRU within each class) until
        it fits under the cap and above the free-space floor. ckpt: may delete only checkpoints. Deletes nothing (and
        returns False) if it would not fit even then."""
        self._reap()
        free = shutil.disk_usage(self.base).free - sum(self.files[p][0] for p in self.pending if p in self.files)
        cand = sorted((p for p in self.files if not ckpt or _is_ckpt(p)), key=lambda p: (not _is_ckpt(p), self.files[p][1]))
        gain = sum(self.files[p][0] for p in cand)
        if size > self.cap or free + gain - size < self.min_free or self.used_bytes - gain + size > self.cap:
            return False
        for p in cand:
            if self.used_bytes + size <= self.cap and free - size >= self.min_free:
                break
            free += self.files[p][0]
            self._drop_file(p)
        return True

    def _save(self, path, meta, tensors):
        return bool(self._save_many([(path, meta, tensors)]))

    def _save_many(self, items, ckpt=False):
        """Queue files [(path, meta, tensors)] for the writer thread: room for all of them (LRU; ckpt: only checkpoints
        may go), index entries now, one host copy (pinned, non-blocking, current stream) for all. -> paths queued
        ([] if there's no room: nothing written, nothing deleted)."""
        self._reap()
        size = lambda t: sum(_pad(v.numel() * v.element_size()) for v in t.values()) + 8192
        total = sum(size(t) for _, _, t in items)
        if not items or not self._room(total, ckpt):
            return []
        while self.pending and self._inflight() + total > PIN_MB << 20:
            self._settle(next(iter(self.pending)))               # oldest write first (one FIFO writer)
        try:
            host, ready = self._host({(i, n): v for i, (_, _, t) in enumerate(items) for n, v in t.items()})
        except RuntimeError as e:  # (pinned allocation / device failure): the writes are skipped
            print(f'cache write skipped: {type(e).__name__}: {e}'[:200], flush=True)
            return []
        self.writer = self.writer or ThreadPoolExecutor(1, thread_name_prefix='inkwrite')
        for i, (path, meta, t) in enumerate(items):
            self.files[path] = [size(t), time.time()]
            self.used_bytes += size(t)
            m = self.mem[path] = (dict(meta, format=FORMAT, weights=self.wid), {n: host[i, n] for n in t}, ready)
            self.pending[path] = self.writer.submit(_write_bg, path, m[0], m[1], ready, self._pause)
        return [p for p, _, _ in items]

    def _host(self, tensors):
        """Host copies -> (dict, ready). cuda: one pinned buffer (64-byte aligned slots), non-blocking copies on the
        current stream, ready = wait for them (writer thread); cpu: the same layout, plain copies, ready None."""
        cuda = self.dev.type == 'cuda'
        buf = torch.empty(sum(_pad(v.numel() * v.element_size()) for v in tensors.values()), dtype=torch.uint8,
                          pin_memory=cuda)
        out, o = {}, 0
        for k, v in tensors.items():
            nb = v.numel() * v.element_size()
            out[k] = buf[o:o + nb].view(v.dtype).view(v.shape)
            out[k].copy_(v, non_blocking=cuda)
            o += _pad(nb)
        if not cuda:
            return out, None
        ev = torch.cuda.Event()
        ev.record()
        return out, ev.synchronize

    def _inflight(self):
        return sum(self.files[p][0] for p, f in self.pending.items() if not f.done() and p in self.files)

    def _touch(self, path):
        self.files[path][1] = time.time()
        try:
            os.utime(path)                                       # LRU order survives restarts
        except OSError:
            pass

    def _drop_file(self, path):
        if path in self.pending:
            self._settle(path)                                   # (a failed write has dropped it already)
        try:
            os.remove(path)
        except OSError:
            pass
        if path in self.files:
            self.used_bytes -= self.files.pop(path)[0]
        f = os.path.basename(path)
        if os.path.dirname(path) == self.dir and f[:2] in ('s-', 'c-') and f.endswith('.state'):
            P, k = f[2:-6].split('-')
            if self._state_path(int(P), k) is None:
                self.states.get(int(P), set()).discard(k)


def _is_ckpt(path):
    return os.path.basename(path).startswith('c-')


def _write_bg(path, meta, tensors, ready, pause=None):
    """Writer thread: ready() (device -> host copy done), then the file (pausing while restores read). -> written.
    Its only CUDA calls are ready() (an event sync) and freeing pinned blocks: legal while the main thread captures a
    graph only because every capture is thread_local (global mode: the call fails and invalidates the capture)."""
    try:
        if ready is not None:
            ready()
        _write(path, meta, tensors, pause)
        return True
    except Exception as e:  # (ENOSPC, EIO, a device error in ready, ...): no file, never a failed request
        print(f'cache write skipped: {type(e).__name__}: {e}'[:200], flush=True)
        for f in glob.glob(f'{path}.*.tmp'):
            try:
                os.remove(f)
            except OSError:
                pass
        return False
