"""Engine: tokenizer + vision + model + sampling, split into steps that server.Scheduler interleaves (DESIGN-batching.md
2): request() -> start() -> prefill_chunk()... -> to_decode(slot) -> decode_step([reqs])... -> finish(); a Request
holds everything a paused request needs (parser, stop matcher, grammar, RNG, tokens so far). Decode runs through a
batch decoder, INK_MAX_BATCH slots: decode_batch.BatchDecoder on CUDA with INK_GRAPH (INK_BATCH_DECODER=0: LoopBatch;
INK_SOLO_FUSED=1: a lone decode on the single-row fused Decoder), else loopbatch.LoopBatch. generate() and
prefill() are single-request wrappers (tests, tools).
KV for the full-attention visits lives in a paged pool (kvpool.py), one page table per sequence.
Prefix reuse, GPU + disk only (no host-RAM tier: host RAM is shared with video renders):
- at the end of each prompt (and at a preemption) the sequence is *parked* on the GPU: its full pages are shared with
  the prefix tree (refcount) and the small state (rings, conv states, last hidden; ~200 MB) is cloned. INK_PARK entries,
  LRU. A later prompt that starts with a parked prompt forks it and prefills only the rest (bit-exact vs cold when the
  chunk boundaries match).
- a parked entry pushed out of the GPU is written to disk (state file; its full pages when the pool evicts them) only
  for chats likely to continue (assistant/tool turns or tools, >= INK_DISK_MIN tokens) and preempted prompts; one-off
  and short prompts never touch the SSD. kvpool.restore finds them there.
- INK_CKPT_EVERY=N (0 = off; a multiple of the 256-token page): while prefilling a disk-worthy prompt (keep), every N
  tokens a mid-prompt checkpoint (the same state, ~194 MiB) goes to disk, its full pages published (keep), so an edit
  inside an earlier prompt (compaction, trimmed tool output) restores from the deepest checkpoint before the edit
  instead of re-prefilling from 0. Prefill chunks end on checkpoint boundaries. The state is copied device -> pinned
  host memory (non-blocking, current stream) and written by kvpool's writer thread (no GPU-resident copy).
Speculative decoding (INK_SPEC=dspark|lookup|off; default dspark on cuda with the BatchDecoder, else off; a request's
`spec: false` turns it off for that request): each spec request holds a spec_runner.Spec from start() (requests needing
> INK_SPEC_MAX_CTX positions: from to_decode, after their prefill, if VRAM allows) to finish() with its own drafter State (DSpark: ~0.14 GiB, taken from a free list of reused States, so their CUDA graphs survive;
drafter weights ~0.7 GiB loaded on first use, dropped with the idle States under VRAM pressure). Prompt taps come from
each prefill chunk (forward_taps); a restored prefix has no taps, so the drafter context starts at the restored
position (Spec start=P). A parked decode gives its State back: on resume the drafter starts empty at the resume
position (acceptance recovers within a few hundred tokens). A decode step's rows: per request 1 plain row, or k + 1
verify rows (k cut so the step stays <= 16 rows); then each spec slot accepts + commits its rows and the plain rows
are sampled as before (so a k = 0 round is exactly a plain step). Requests with logit processors (penalties,
logit_bias, min_tokens) or min_p speculate too: verify row i is processed for the history out + d_1..d_i (as plain
decoding would process that position) and min_p filters the target rows in settle, so greedy == plain and sampled
acceptance stays exact. Spec flags (INK_DSPARK_SHORT / INK_DSPARK_CTXB / INK_SPEC_GATE / INK_SPEC_ACCEPT_DEV, read by
dspark.py / spec_runner.py at import) apply here as in tests/gpu_spec.py; with INK_FP8_DECODE=1 the drafter reads the
target's FP8 unembed (shortlist head), not the bf16 one.
"""
import base64
import gc
import hashlib
import math
import os
import threading
import time
import urllib.request

import torch
from tokenizers import Tokenizer

import chat
import decode
import grammar
from decode import pos_noise, sample, sample_rows  # noqa: F401  (sample_rows: re-exported for tests/tools)
from kvpool import KVPool, weights_id
from loopbatch import LoopBatch, SoloFused
from model import H, Model
from spec import Lookup
from spec_runner import SeqBatch, Spec, forward_taps
from vision import MAX_SIDE, TILE, Vision, load_image, preprocess

MODEL_DIR = os.environ.get('INK_MODEL', os.path.expanduser('~/llm/models/Inkling-2-Mini-Uncensored'))


PARK = int(os.environ.get('INK_PARK', 2))                      # end-of-prompt states parked on the GPU (~200 MB each)
SNAP_MIN = int(os.environ.get('INK_SNAP_MIN', 512))            # shorter prompts are cheaper to re-prefill: not parked
DISK_MIN = int(os.environ.get('INK_DISK_MIN', 2048))           # disk only for continued chats at least this long
CKPT = int(os.environ.get('INK_CKPT_EVERY', 0))                 # mid-prompt disk checkpoints every N tokens (0 = off)
assert CKPT % 256 == 0, 'INK_CKPT_EVERY must be a multiple of the 256-token page'
# free VRAM wanted before a full INK_CHUNK prefill chunk: its transient peak (measured on GPU 11-22, 4k chunks, current
# prefill flags: +1.5 GiB at 0 context, +2.8-2.9 GiB from 32k on) + a ~200 MB floor. A chunk of n tokens asks
# MARGIN * max(n, chunk / 2) / chunk (the scheduler's 2k chunks while decodes run). Below it, idle decoder buffers and
# drafter States, then parked states, then the drafter (spec off for running requests) are pushed out. (Was 2048: under
# the 4 KB/token decodes + 4 drafter States of window g2's queue case a 2k chunk OOMed with 139 MiB free.)
MARGIN = int(os.environ.get('INK_VRAM_MARGIN_MB', 3072)) << 20
# solo mode: a prompt this long admitted while nothing decodes runs alone (nothing else is admitted until it ends) and
# drops the idle slots' decode state (graphs + ~0.2 GB static state per slot, rebuilt lazily, ~1-2 s against >= ~30 s
# of prefill), whose memory prefill can't use otherwise. 0 = never
DEC_DROP = int(os.environ.get('INK_DEC_DROP_MIN', 65536))
MAX_BATCH = int(os.environ.get('INK_MAX_BATCH', 4))           # decode slots (DESIGN-batching.md 1.5: ~0.2 GB each)
# free VRAM wanted before a slot builds its Decoder (~0.2 GB static state + its first, captured step) + the floor
SLOT_NEED = int(os.environ.get('INK_SLOT_NEED_MB', 512)) << 20
SLOT_MB = 224 << 20  # BatchDecoder: static state per extra slot (rings 204 MiB at WR 1088 + conv; all slots at once)
RMAX = 16            # rows per decode step (decode_batch.RM): spec verify rows of all slots share it
# requests needing more positions get their drafter only after the prefill (to_decode; VRAM-driven like every drafter
# ask: room -> spec, else plain), so a long prefill's margin never competes with it; their drafter context starts at the
# decode position (as after a prefix restore). (Env name kept from the old hard cap "longer decode plain": deploy sets it.)
SPEC_DEFER = int(os.environ.get('INK_SPEC_MAX_CTX', 262144))
DS_W_MB = 800 << 20  # drafter weights (0.6 GB) + U @ hidden_out (0.1 GB) + its graph pool, est.
DS_ST_MB = 160 << 20  # drafter State: layer-1 ring 128 MiB at INK_DSPARK_CTX 32768 + sliding rings + io buffers, est.
DS_SHORT_W_MB = 70 << 20   # shortlist head: static rows of U / U' / c (168 MB) replace the full U' (102 MB), est.
DS_SHORT_ST_MB = 16 << 20  # shortlist head: dynamic rows per State (~21 MB at INK_DSPARK_DYN 4096; noise is fp64), est.
# headroom a drafter load/State must leave: fixed, not MARGIN/2 (window final 11-23: MARGIN 3072 starved the drafter; a
# prefill chunk short of its own MARGIN turns spec off via free_vram(spec=True) instead)
DS_HEAD = int(os.environ.get('INK_DS_HEAD_MB', 256)) << 20  # (pre-11-23 the ask was only a target: load anyway, OOM -> spec off)


def _bytes_to_unicode():
    """The byte-level BPE alphabet (bytes 0-255 <-> printable characters), the standard GPT-2 table."""
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, map(chr, cs)))


_B2U_INV = {c: b for b, c in _bytes_to_unicode().items()}
_BYTES = {}  # token id -> raw bytes (logprob entries)


class BadRequest(ValueError):
    pass


class Preempted(Exception):
    """Raised between prefill chunks when preempt() says so. The caller restarts the request later; the progress so
    far is parked, so the restart resumes where it stopped."""


class OutOfMemory(Exception):
    """CUDA OOM that freeing the parked states didn't fix (server: 503). The request's pages are released."""


class NoRoom(Exception):
    """The KV pool can't hold prompt + max_tokens right now (pages held by running sequences); requeue and retry."""


def image_bytes(part):
    url = part.get('image_url', part.get('image'))
    url = url.get('url') if isinstance(url, dict) else url
    if not isinstance(url, str):
        raise BadRequest('image part without url')
    if url.startswith('data:'):
        return base64.b64decode(url.split(',', 1)[1])
    if url.startswith(('http://', 'https://')):
        with urllib.request.urlopen(url, timeout=30) as r:
            return r.read(64 << 20)
    raise BadRequest('image url must be data: or http(s)')


def image_grid(im):
    """(rows, cols) of tiles vision.preprocess will make of im, from the header size alone (no pixel decode)."""
    w, h = im.size
    if MAX_SIDE and max(w, h) > MAX_SIDE:  # (same arithmetic as preprocess's opt-in resize)
        k = MAX_SIDE / max(w, h)
        w, h = max(1, round(w * k)), max(1, round(h * k))
    return math.ceil(h / TILE), math.ceil(w / TILE)


class Request:
    """One generation, driven by server.Scheduler through the Engine step functions (or by Engine.generate). Holds
    what a paused request needs to continue its stream unchanged: parser, stop matcher, grammar matcher, RNG and the
    tokens so far. emit(kind, value) gets ('delta', (reasoning|content, text)); the scheduler adds done/errors."""
    sp = spec = None

    def __init__(self, eng, ids, pre, keep=False, max_tokens=0, temperature=1.0, top_p=1.0, top_k=0, seed=None,
                 prefill_text=False, response_format=None, stop=None, stats=None, emit=None, prio=0, cancel=None,
                 spec=None, logprobs=None, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0,
                 logit_bias=None, min_p=0.0, min_tokens=0, stop_ids=None, tool_force=None, one_call=False,
                 prediction=None):
        self.pred = list(prediction or ())          # predicted output ids: a source of copy drafts (speculation only)
        self.prompt = self.ids = ids                # ids: tokens to prefill (a parked decode: prompt + fed tokens)
        self.pre, self.keep, self.need = pre, keep, len(ids) + max_tokens   # need: positions reserved
        self.max_tokens, self.temperature, self.top_p, self.top_k = max_tokens, temperature, top_p, top_k
        self.prefill_text, self.emit = prefill_text, emit or (lambda *a: None)
        self.stats = {} if stats is None else stats
        self.prio, self.cancel = prio, cancel or threading.Event()
        # sampled tokens use the Exp(1) noise of their position (decode.pos_noise(gen, seed, pos)), so a seeded request
        # is reproducible however its decode was scheduled (spec widths, batching, preemption)
        self.seed = seed if seed is not None else int.from_bytes(os.urandom(8), 'little')
        self.gen = torch.Generator(device=eng.model.dev)
        self.gen.manual_seed(self.seed & ((1 << 63) - 1))
        self.parser = chat.Parser(eng.tok)
        if prefill_text:
            self.parser.kind = 'content'
        self.stopper = chat.Stop(stop) if stop else None
        # forced call: reasoning (one block) gets at most half the budget, so the call always fits
        self.cons = grammar.ToolCall(tool_force, think_max=max_tokens // 2) if tool_force is not None else \
            grammar.Json(response_format, prefill_text) if response_format is not None else None
        self.lp, self.lp_pend = logprobs, []        # logprobs: None (off) or the top_logprobs count
        self.pen = (presence_penalty, frequency_penalty, repetition_penalty)
        self.pen = None if self.pen == (0, 0, 1) else self.pen
        self.cnt, self.ncnt = None, 0               # output token counts [V] on device (penalties), tokens counted
        # host tensors here (Request is built on an executor thread, possibly while the scheduler captures a CUDA
        # graph); moved to the device on first use in _process (scheduler thread)
        self.bias = None if not logit_bias else (torch.tensor(list(logit_bias)),
                                                 torch.tensor(list(logit_bias.values()), dtype=torch.float))
        self.min_p, self.min_tokens, self.stop_ids, self.one_call = min_p, min_tokens, set(stop_ids or ()), one_call
        self.ban = [chat.EOS] + [chat.END] * prefill_text + sorted(self.stop_ids)  # held off until min_tokens
        self.proc = bool(self.pen or self.bias is not None or min_tokens)  # logit processors (_process)
        self.out = []                               # sampled ids; while decoding the last one is pending (not fed)
        self.n_reason = 0                           # of those, tokens inside a thinking block (its body + closing END)
        self.err = None                             # an error that failed this request alone in a decode step
        self.spec, self.sp = spec, None             # spec: False = no speculative decoding; sp: its Spec while running
        self.seq = self.f = self.slot = self.img = None
        self.P = self.a = self.pos = 0              # restored prefix, next prefill position, next decode position
        self.done = self.hit = self.oomed = self.solo = False
        self.t_start = self.t1 = time.time()


class Engine:
    def __init__(self, device='cuda', max_ctx=None, vision=True):
        max_ctx = max_ctx or int(os.environ.get('INK_MAX_CTX', 1 << 20))
        self.tok = Tokenizer.from_file(os.path.join(MODEL_DIR, 'tokenizer.json'))
        self.pool = KVPool(max_ctx, device, weights_id(MODEL_DIR))
        self.model = Model(MODEL_DIR, device, max_ctx, pool=self.pool)
        self.graph = os.environ.get('INK_GRAPH') == '1'
        if self.model.dev.type == 'cuda' and self.graph and os.environ.get('INK_BATCH_DECODER', '1') != '0':
            from decode_batch import BatchDecoder  # (Triton-only kernels: CPU keeps LoopBatch)
            self.bd = BatchDecoder(self.model, MAX_BATCH)
            if os.environ.get('INK_SOLO_FUSED') == '1':
                self.bd = SoloFused(self.bd, self.model)
        else:
            self.bd = LoopBatch(self.model, MAX_BATCH, dec=self.graph)  # INK_GRAPH: a decode.Decoder per slot
        dflt = 'dspark' if self.model.dev.type == 'cuda' and not isinstance(self.bd, (LoopBatch, SoloFused)) else 'off'
        self.spec = os.environ.get('INK_SPEC', dflt)
        assert self.spec in ('dspark', 'lookup', 'off'), self.spec
        if self.spec != 'off' and isinstance(self.bd, LoopBatch):  # CPU: verify rows on the Seq path
            self.bd = SeqBatch(self.model, MAX_BATCH)
        assert self.spec == 'off' or hasattr(self.bd, 'tap_hist'), 'INK_SPEC needs the BatchDecoder (not SoloFused)'
        if self.spec != 'off':
            flags = {k: v for k, v in os.environ.items() if k.startswith(('INK_DSPARK', 'INK_SPEC', 'INK_DEC_EMBED_GPU'))}
            print(f'spec {self.spec}, flags {flags or "none (defaults)"}', flush=True)
        self.ds, self.ds_free = None, []  # DSpark drafter (lazy) and its idle States (reused: their graphs stay)
        self.spec_live = set()            # requests holding a drafter State (spec_off frees them under VRAM pressure)
        self.vision = Vision(self.model.W.f, self.model.dev) if vision else None
        self.chunk = int(os.environ.get('INK_CHUNK', 4096 if self.model.dev.type == 'cuda' else 256))
        self.max_ctx = max_ctx
        self.parked = []  # GPU-parked prompt states, oldest first: (keys, page table, state tensors, keep on disk)
        self.mem_free = (lambda: torch.cuda.mem_get_info()[0]) if self.model.dev.type == 'cuda' else None  # tests fake it
        self.ooms = 0     # CUDA OOMs caught (prefill restarts + decode failures)
        self.vision_images = 0  # images run through the vision encoder (restored-prefix images skip it)
        self._margin = None  # None: use module MARGIN; set in light mode (INK_LITE_MARGIN_MB)
        self.lite, self.lite_keep_core, self.lite_ram_experts, self._full_state = False, 0, 'mmap', None
        grammar.compiler(self.tok)  # build the xgrammar vocab once at startup (~3 s)

    @property
    def slots(self):
        return self.bd.n

    @property
    def batched(self):  # shared static decode state for all slots (BatchDecoder), not a Decoder per slot
        return not isinstance(self.bd, LoopBatch)

    @property
    def dec(self):  # slot 0's Decoder (BatchDecoder: itself, while its buffers exist) (tests/gpu_mem.py reads its static state)
        return self.bd.decs[0]

    def prepare(self, messages, tools=None, effort=0.9):
        """-> (ids, [(tiles, rows, cols, pixel-bytes hash)] in prompt order). Images of the new input (after the last
        assistant message) are decoded + tiled here (executor thread); older ones, normally inside the cached prefix,
        only get their size from the header and keep the raw bytes as `tiles`: _start decodes them if they turn out to
        lie past the restored prefix (_pixels)."""
        pre = []
        last = max((i for i, m in enumerate(messages) if m.get('role') == 'assistant'), default=-1)
        new = {id(p) for m in messages[last + 1:] if isinstance(m.get('content'), list) for p in m['content']}

        def count(part):
            if self.vision is None:
                raise BadRequest('images disabled')
            try:
                raw = image_bytes(part)
                if id(part) in new:
                    tiles, rows, cols = preprocess(load_image(raw))
                else:
                    (rows, cols), tiles = image_grid(load_image(raw)), raw
            except BadRequest:
                raise
            except Exception as e:
                raise BadRequest(f'bad image: {e}')
            pre.append((tiles, rows, cols, int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), 'little')))
            return rows * cols
        ids = chat.render(self.tok, messages, tools, effort, count)
        return ids, pre

    def count(self, messages, tools=None, effort=0.9, memo=None):
        """Prompt length in tokens, images sized from their headers only (no decode): /v1/responses/input_tokens and
        truncation. memo: {id(part): tiles} across calls on the same messages (an http image is fetched once)."""
        memo = {} if memo is None else memo

        def tiles(part):
            if self.vision is None:
                raise BadRequest('images disabled')
            if id(part) not in memo:
                try:
                    r, c = image_grid(load_image(image_bytes(part)))
                except BadRequest:
                    raise
                except Exception as e:
                    raise BadRequest(f'bad image: {e}')
                memo[id(part)] = r * c
            return memo[id(part)]
        return len(chat.render(self.tok, messages, tools, effort, tiles))

    @staticmethod
    def _pixels(req):
        """Decode + tile the images of req.pre that are still raw bytes and end past the restored prefix req.P."""
        end, pos = 0, None
        for i, (src, r, c, h) in enumerate(req.pre):
            end += r * c
            if isinstance(src, bytes) and pos is None:
                pos = (req.keys < 0).nonzero().flatten()  # image tile positions in prompt order
            if isinstance(src, bytes) and int(pos[end - 1]) >= req.P:
                try:
                    tiles, r2, c2 = preprocess(load_image(src))
                except Exception as e:
                    raise BadRequest(f'bad image: {e}')
                assert (r2, c2) == (r, c), ('image header size != decoded size', (r, c), (r2, c2))
                req.pre[i] = (tiles, r, c, h)

    @staticmethod
    def keys(ids, pre):
        """Prefix-cache key per position: the token id, or for image tiles -(1 + (image hash + tile index) % 2^62)."""
        k = torch.tensor(ids, dtype=torch.long)
        pos, off = (k == chat.IMG_TOK).nonzero().flatten(), 0
        for _, r, c, h in pre:
            k[pos[off:off + r * c]] = -1 - (h + torch.arange(r * c)) % (1 << 62)
            off += r * c
        return k

    @staticmethod
    def state_of(seq, f):
        """Clone of the small per-sequence state at seq.pos (sconv updates conv states in place)."""
        st = {'f': f.clone()}
        for v, (d, (rk, rv)) in enumerate(zip(seq.conv, seq.ring)):
            st.update({f'conv.{v}.{k}': x.clone() for k, x in d.items()})
            st[f'ring.{v}.k'], st[f'ring.{v}.v'] = rk.clone(), rv.clone()
        return st

    def _ckpt(self, req, b):
        """Mid-prompt checkpoint of req at b (INK_CKPT_EVERY): publish its full pages (keep) and queue the state for disk
        unless one exists there. No clones: kvpool enqueues the device -> pinned host copy on the current stream, ahead of
        the next chunk's in-place conv updates."""
        if not self.pool.dir:
            return
        self.pool.publish(req.seq.kv, req.keys[:b], b, keep=True)
        st = {'f': req.f}
        for v, (d, (rk, rv)) in enumerate(zip(req.seq.conv, req.seq.ring)):
            st.update({f'conv.{v}.{k}': x for k, x in d.items()})
            st[f'ring.{v}.k'], st[f'ring.{v}.v'] = rk, rv
        if self.pool.save_ckpt(req.keys[:b], st):
            req.stats['ckpts'] = req.stats.get('ckpts', 0) + 1

    def seq_from(self, tbl, P, st):
        seq = self.model.new_seq(tbl)
        seq.pos = P
        seq.conv = [{k: st[f'conv.{v}.{k}'].clone() for k in d} for v, d in enumerate(seq.conv)]
        seq.ring = [(st[f'ring.{v}.k'], st[f'ring.{v}.v']) for v in range(len(seq.ring))]  # replaced, never mutated
        return seq, P, st['f'].clone() if 'f' in st else None

    def park(self, keys, seq, f, keep, st=None):
        """Keep the state at len(keys) on the GPU (shared pages + cloned small state). st: a parked decode's state from
        the batch decoder (no 'f'; it doesn't supersede the prompt's own entry)."""
        P = len(keys)
        if getattr(self, 'lite', False):  # light mode: publish pages and spill directly to disk (0 GPU-parked states)
            self.pool.publish(seq.kv, keys, P, keep)
            if keep and self.pool.dir:
                try:
                    st = self.state_of(seq, f) if st is None else st
                    tbl = self.pool.fork(seq.kv, P)
                except torch.OutOfMemoryError:
                    print(f'park skipped at {P} tokens: CUDA OOM', flush=True)
                    return
                if tbl is not None:
                    try:
                        self.pool.save_state(tbl, keys, st)
                    finally:
                        self.pool.release(tbl)
            return
        try:  # near the 1M limit the ~200 MB clone can OOM: skip parking, keep the finished prefill
            st = self.state_of(seq, f) if st is None else st
            tbl = self.pool.fork(seq.kv, P)
        except torch.OutOfMemoryError:
            print(f'park skipped at {P} tokens: CUDA OOM', flush=True)
            return
        if tbl is None:
            return
        self.pool.publish(seq.kv, keys, P, keep)
        # entries hold tensors: filter by identity, never list.remove/index (== on tuples compares tensors)
        old = [e for e in self.parked if 'f' in st and len(e[0]) <= P and torch.equal(keys[:len(e[0])], e[0])]
        self.parked = [e for e in self.parked if not any(e is o for o in old)]
        for e in old:  # an earlier turn of this chat: superseded, no disk write
            self.unpark(e, disk=False)
        self.parked.append((keys.clone(), tbl, st, keep))
        while len(self.parked) > PARK:
            self.unpark(self.parked.pop(0))

    def unpark(self, e, disk=True):
        keys, tbl, st, keep = e
        if disk and keep:
            self.pool.save_state(tbl, keys, st)
        self.pool.release(tbl)

    def parked_bytes(self):
        return sum(x.numel() * x.element_size() for e in self.parked for x in e[2].values())

    def free_vram(self, need=None, parked=True, spec=False):
        """Free VRAM below `need` bytes: return cached allocator blocks, then drop idle slots' Decoders (rebuilt on
        their next use, ~1 s; BatchDecoder: the shared buffers of all slots, only while no slot decodes) and idle drafter
        States, then push parked states out (LRU; to disk if worth keeping: preempted prefills, continued chats), then
        the drafter weights if no request uses them, then (spec) turn speculative decoding off for the running requests
        (their States, graphs and the drafter weights go; they decode plain). parked=False: an optional allocation (a
        drafter State) never pushes a cached prefix out. need=None: all of it (OOM recovery). -> enough free now."""
        short = lambda: need is None or self.mem_free() < need
        if need is not None and (self.mem_free is None or not short()):
            return True
        torch.cuda.empty_cache()  # (no-op without CUDA)
        if short():
            self.bd.drop()
            self.drop_drafter(weights=False)  # idle States first; weights only after the parked states (reload + captures)
        while parked and self.parked and short():
            self.unpark(self.parked.pop(0))
            torch.cuda.empty_cache()
        if short():
            self.drop_drafter()
        if (spec or need is None) and short() and self.spec_live:
            self.spec_off()
        return need is None or not short()

    def spec_off(self, keep=None):
        """VRAM pressure: every running request except `keep` decodes plain from here; their drafter States (with
        graphs) are dropped, then the drafter weights. -> States dropped."""
        n = 0
        for r in [r for r in self.spec_live if r is not keep]:
            sp, r.sp = r.sp, None
            self.spec_live.discard(r)
            if sp is not None and sp.st is not None:
                sp.st.graphs.clear()
                self.ds.live -= 1
                n += 1
        if n:
            print(f'VRAM: speculative decoding off for {n} running request(s)', flush=True)
        self.drop_drafter()
        if self.model.dev.type == 'cuda':
            torch.cuda.empty_cache()
        return n

    def drop_drafter(self, keep=0, weights=True):
        """Free idle drafter States beyond `keep`; with none left and no spec request running (and weights), the
        drafter weights."""
        n = len(self.ds_free)
        del self.ds_free[keep:]
        if weights and not keep and self.ds is not None and self.ds.live == 0:
            self.ds, n = None, n + 1
        if n > len(self.ds_free) and self.model.dev.type == 'cuda':
            torch.cuda.empty_cache()

    def _bd_spec(self):
        """Lightweight descriptor to recreate self.bd on exit_lite without retaining any reference to it or its
        Fused weight pointers (which otherwise pin offloaded expert tensors in VRAM)."""
        bd = self.bd
        cls = type(bd)
        if isinstance(bd, SoloFused):
            return ('solo', bd.bd.n, bd.bd.graph)
        if isinstance(bd, SeqBatch):
            return ('seq', bd.n, bd.dev_tok, getattr(bd, 'use_taps', True))
        if isinstance(bd, LoopBatch):
            return ('loop', cls, bd.n, bd.dec)
        return ('batch', cls, bd.n, getattr(bd, 'graph', self.graph))

    def _build_bd(self, spec):
        kind = spec[0]
        if kind == 'solo':
            from decode_batch import BatchDecoder
            return SoloFused(BatchDecoder(self.model, spec[1], graph=spec[2]), self.model)
        if kind == 'seq':
            return SeqBatch(self.model, spec[1], dev_tok=spec[2], taps=spec[3])
        if kind == 'loop':
            cls, slots, dec = spec[1], spec[2], spec[3]
            return LoopBatch(self.model, slots, dec=dec) if cls is LoopBatch else cls(self.model, slots)
        cls, slots, graph = spec[1], spec[2], spec[3]
        return cls(self.model, slots, graph=graph)

    @staticmethod
    def _lite_ram_layers(ram_experts, keep_layers):
        if ram_experts == 'mmap':
            return set()
        if ram_experts == 'core':
            return set(range(6, 18)) - keep_layers
        if ram_experts == 'all':
            return set(range(24)) - keep_layers
        raise ValueError(f"ram_experts must be 'mmap', 'core', or 'all', got {ram_experts!r}")

    def enter_lite(self, keep_core_layers=0, chunk=None, margin_mb=None, ram_experts='mmap'):
        """Enter (or reconfigure) light mode: drop decoder/drafter objects (freeing BatchDecoder.F's references to
        expert tensors), offload routed MoE experts to zero-copy read-only mmap views or owned CPU RAM tensors, flush
        GPU-parked states, and serve single-stream with prompt-lookup speculation."""
        if ram_experts not in ('mmap', 'core', 'all'):
            raise ValueError(f"ram_experts must be 'mmap', 'core', or 'all', got {ram_experts!r}")
        assert not self.bd.busy(), 'cannot enter light mode while decode slots are busy'
        keep_core = max(0, min(12, int(keep_core_layers)))
        keep_layers = set(range(6, 6 + keep_core))
        ram_layers = self._lite_ram_layers(ram_experts, keep_layers)
        if not getattr(self, 'lite', False):
            self._full_state = (self._bd_spec(), self.chunk, self.spec, getattr(self, '_margin', None))
            self.spec_off()
            self.drop_drafter()
            self.bd.drop()
            self.bd = SeqBatch(self.model, 1, taps=False)  # drop old bd/Fused refs before offloading experts
            while self.parked:
                self.unpark(self.parked.pop(0))
            self.model.W.offload_experts(keep_layers, ram_layers=ram_layers)
            full_chunk, full_spec = self._full_state[1], self._full_state[2]
            self.spec = 'lookup' if full_spec != 'off' else 'off'
        else:
            while self.parked:
                self.unpark(self.parked.pop(0))
            self.model.W.offload_experts(keep_layers, ram_layers=ram_layers)
            full_chunk = self._full_state[1]
        c = int(chunk if chunk is not None else os.environ.get('INK_LITE_CHUNK', 1024))
        self.chunk = max(1, min(c, full_chunk))
        mb = int(margin_mb if margin_mb is not None else os.environ.get('INK_LITE_MARGIN_MB', 512))
        self._margin = max(0, mb) << 20
        self.lite, self.lite_keep_core, self.lite_ram_experts = True, keep_core, ram_experts
        gc.collect()
        if self.model.dev.type == 'cuda':
            torch.cuda.empty_cache()
        return self.lite_status()

    def exit_lite(self):
        """Leave light mode: reload all offloaded routed experts onto the device and rebuild full-mode decoder/config.
        On CUDA OOM mid-reload or decoder rebuild, rolls back to light mode and re-raises."""
        if not getattr(self, 'lite', False):
            return self.lite_status()
        assert not self.bd.busy(), 'cannot exit light mode while decode slots are busy'
        keep_layers = set(range(6, 6 + self.lite_keep_core))
        ram_layers = self._lite_ram_layers(getattr(self, 'lite_ram_experts', 'mmap'), keep_layers)
        self.model.W.reload_experts()
        bd_spec, chunk, spec, margin = self._full_state
        try:
            new_bd = self._build_bd(bd_spec)
        except torch.OutOfMemoryError:
            self.model.W.offload_experts(keep_layers, ram_layers=ram_layers)
            gc.collect()
            if self.model.dev.type == 'cuda':
                torch.cuda.empty_cache()
            raise
        self.bd, self.chunk, self.spec, self._margin = new_bd, chunk, spec, margin
        self._full_state = None
        self.lite, self.lite_keep_core, self.lite_ram_experts = False, 0, 'mmap'
        gc.collect()
        if self.model.dev.type == 'cuda':
            torch.cuda.empty_cache()
        return self.lite_status()

    def lite_status(self):
        """Current light-mode state and VRAM stats (None when not on CUDA)."""
        cuda = self.model.dev.type == 'cuda' and torch.cuda.is_available()
        alloc_mb = res_mb = free_mb = None
        if cuda:
            free_b, _ = torch.cuda.mem_get_info(self.model.dev)
            free_mb = round(free_b / (1 << 20), 1)
            alloc_mb = round(torch.cuda.memory_allocated(self.model.dev) / (1 << 20), 1)
            res_mb = round(torch.cuda.memory_reserved(self.model.dev) / (1 << 20), 1)
        return {
            'lite': bool(getattr(self, 'lite', False)),
            'keep_core_layers': int(getattr(self, 'lite_keep_core', 0)),
            'offloaded_layers': len(getattr(self.model.W, 'offloaded', ())),
            'ram_experts': getattr(self, 'lite_ram_experts', 'mmap'),
            'ram_layers': len(getattr(self.model.W, 'ram_layers', ())),
            'slots': self.slots,
            'chunk': self.chunk,
            'spec': self.spec,
            'vram_allocated_mb': alloc_mb,
            'vram_reserved_mb': res_mb,
            'vram_free_mb': free_mb,
        }

    def _spec(self, req, start=None):
        """req's Spec for this run (None: plain decoding), its drafter context starting at `start` (default req.P).
        DSpark: loads the drafter / builds a State if VRAM allows (never pushing a parked prefix out); no room -> prompt
        lookup only (copied spans, no VRAM: agent copies still speculate); a failure there (OOM, ...) -> plain."""
        # (logit processors / min_p: decode_step processes each verify row for its draft prefix, Spec applies min_p)
        mode = self.spec if req.spec is not False else 'off'
        if mode == 'off':
            return None
        stop = (chat.EOS, chat.END) if req.prefill_text else (chat.EOS,)
        kw = dict(temperature=req.temperature, top_p=req.top_p, top_k=req.top_k, gen=req.gen, cons=req.cons, stop=stop,
                  min_p=req.min_p, seed=req.seed, pred=req.pred, pstats=req.stats)
        lookup = lambda: Spec(None, self.bd, None, lookup=Lookup(req.prompt, pre=req.pred), dev=self.model.dev, **kw)
        if mode == 'lookup':
            return lookup()
        try:
            m = self.model
            d8 = getattr(m, 'd8', None)
            u8 = d8.un if d8 is not None and d8.dec and d8.un is not None else None  # INK_FP8_DECODE: FP8 unembed
            import dspark
            short = bool(dspark.SHORT) or u8 is not None  # (unembed8 turns the shortlist head on)
            if self.ds is None:  # (optional: never evicts parked prefixes; no room -> this request decodes plain)
                if not self.free_vram(DS_HEAD + DS_W_MB + DS_SHORT_W_MB * short, parked=False):
                    print(f'spec lookup-only for a request: VRAM short for the drafter (free {self.mem_free() >> 20} MiB, '
                          f'alloc {torch.cuda.memory_allocated() >> 20}, reserved {torch.cuda.memory_reserved() >> 20}, '
                          f'need {(DS_HEAD + DS_W_MB + DS_SHORT_W_MB * short) >> 20}, parked {len(self.parked)}, '
                          f'bd {getattr(self.bd, "busy", lambda: "?")()})', flush=True)
                    return lookup()
                self.ds = dspark.DSpark(embed=m.W.embed, unembed=None if u8 is not None else
                                        m.W['unembed.weight'][:grammar.VOCAB], unembed8=u8,
                                        embed_norm=m.W['embed_norm.weight'], device=m.dev)
                self.ds.live = 0
            ds = self.ds
            ds.live += 1  # reserved first: the free_vram below must not drop the weights we're about to use
            try:
                st = self.ds_free.pop() if self.ds_free else None
                if st is None and not self.free_vram(DS_HEAD + DS_ST_MB + DS_SHORT_ST_MB * short, parked=False):
                    # (window g2 11-23: this ask evicted the parked prompt n > 1 siblings were about to restore)
                    ds.live -= 1
                    print('spec lookup-only for a request: VRAM short for a drafter State', flush=True)
                    return lookup()
                return Spec(ds, self.bd, None, start=req.P if start is None else start, st=st, **kw)
            except BaseException:
                ds.live -= 1
                raise
        except Exception as e:  # (OOM or anything else: spec off for this request, never a failed request)
            print(f'spec off for a request: building the drafter failed: {type(e).__name__}: {e}'[:300], flush=True)
            return None

    def restore(self, keys, need_f=True):
        """-> (seq, P, f): the longest parked (GPU) or saved (disk) state that is a prefix of keys, else a fresh seq.
        A parked decode has no last hidden (f None): a new prompt (need_f) matches it only if it continues past it."""
        ok = lambda e: len(e[0]) <= len(keys) and torch.equal(keys[:len(e[0])], e[0]) and \
            (len(e[0]) < len(keys) or not need_f or 'f' in e[2])
        best = max(filter(ok, self.parked), key=lambda e: len(e[0]), default=None)
        n = len(best[0]) if best else 0
        tbl, P, st = self.pool.restore(keys, min_len=n)
        if tbl is not None and need_f and P == len(keys) and 'f' not in st:
            self.pool.release(tbl)
            tbl, P, st = self.pool.restore(keys[:-1], min_len=n)
        if tbl is not None:
            return self.seq_from(tbl, P, st)
        if best is not None and (tbl := self.pool.fork(best[1], n)) is not None:
            self.parked = [e for e in self.parked if e is not best] + [best]  # LRU touch (by identity, see park)
            return self.seq_from(tbl, n, best[2])
        return self.model.new_seq(), 0, None

    def admit(self, tbl, n):
        """Reserve pages for positions [0, n); pushes parked states out of the GPU if the pool is short."""
        while not self.pool.reserve(tbl, n):
            if not self.parked:
                raise NoRoom(f'{n} tokens')
            self.unpark(self.parked.pop(0))

    def request(self, messages, max_tokens=512, effort=0.9, tools=None, prefill_text=False, prompt_ids=None, **kw):
        """-> Request (tokenized, images preprocessed; BadRequest if it can't fit). kw: Request's sampling, seed,
        response_format (grammar.compile_format), stop, stats, emit, prio, cancel. prompt_ids: a raw prompt instead of
        rendered messages (/v1/completions): no chat template; with prefill_text the reply is plain text to END/EOS."""
        if prompt_ids is not None:
            ids, pre = list(prompt_ids), []
        else:
            ids, pre = self.prepare(messages, tools, effort)
        if prefill_text and prompt_ids is None:  # thinking off: the model's turn starts as a text message
            ids += [chat.MODEL, chat.TEXT]
        if len(ids) + max_tokens > self.max_ctx:
            raise BadRequest(f'prompt {len(ids)} + max_tokens {max_tokens} > context {self.max_ctx}')
        # SSD wear: only chats likely to continue (earlier assistant/tool turns, or tools) are worth a disk copy
        keep = len(ids) >= DISK_MIN and bool(tools or any(x.get('role') in ('assistant', 'tool') for x in messages or ()))
        req = Request(self, ids, pre, keep, max_tokens, prefill_text=prefill_text, **kw)
        req.keys = self.keys(ids, pre)  # (here, on the executor thread, not in the scheduler's _start)
        return req

    def can_start(self, req):
        """Cheap pre-check for start (no restore/disk reads while a request waits for pages)."""
        return bool(self.parked) or self.pool.can_admit(req.need)

    def start(self, req):
        """Restore the longest parked/saved prefix of req.ids and reserve pages for prompt + max_tokens (NoRoom if the
        pool is short; nothing held then). A long prompt admitted while nothing decodes runs solo (DEC_DROP)."""
        try:
            self._start(req)
        except torch.OutOfMemoryError as e:
            self._oom(req, e)
            self.start(req)

    def uncached(self, req):
        """For the scheduler: tokens req still has to prefill. Running: exact. Waiting: the prompt minus its longest
        GPU-parked prefix (disk states and tree pages are not looked at: an upper bound)."""
        if req.seq is not None:
            return len(req.ids) - req.a
        k = self._keys(req)
        return len(k) - max((len(e[0]) for e in self.parked if len(e[0]) <= len(k) and torch.equal(k[:len(e[0])], e[0])),
                            default=0)

    def _keys(self, req):
        k = getattr(req, 'keys', None)
        if k is None or len(k) != len(req.ids):  # (a parked decode's ids grew by its fed tokens)
            k = req.keys = self.keys(req.ids, req.pre)
        return k

    def _start(self, req):
        req.keys, req.t_start = self._keys(req), time.time()
        seq, P, f = self.restore(req.keys, need_f=not req.out)  # a parked decode resumes with its pending token
        try:
            self.admit(seq.kv, req.need)
        except NoRoom:
            self.pool.release(seq.kv)
            raise
        req.seq, req.P, req.a, req.f, req.img = seq, P, P, f, None
        self._pixels(req)
        self._end_spec(req)
        # (after the restore: the drafter context starts at P; after an OOM the restart runs without the drafter, which
        # the OOM recovery just dropped)
        req.sp = self._spec(req) if not req.oomed and req.need <= SPEC_DEFER else None  # (longer: in to_decode)
        if req.sp is not None and req.sp.st is not None:
            self.spec_live.add(req)
        if req.out:  # a resumed decode: its parked entry has done its job (don't write it to disk later)
            done = [e for e in self.parked if 'f' not in e[2] and torch.equal(e[0], req.keys)]
            self.parked = [e for e in self.parked if not any(e is d for d in done)]
            for e in done:
                self.unpark(e, disk=False)
        # solo only for a big *uncached* prefill (a warm turn of a long chat prefills a few k: keep decoder + drafter)
        req.solo = bool(DEC_DROP and len(req.ids) - P >= DEC_DROP and not self.bd.busy())
        if req.solo:
            if self.bd.dec:
                self.bd.drop()
            self.drop_drafter()

    def _oom(self, req, e):
        """Prefill OOM: release the pages, push every parked state out; the caller restarts once, then OutOfMemory."""
        self.ooms += 1
        if req.seq is not None:
            self.pool.release(req.seq.kv)
            req.seq = None
        if req.oomed:
            self.free_vram()
            raise OutOfMemory(f'out of GPU memory (prompt {len(req.ids)} tokens)') from e
        req.oomed = True
        print(f'prefill OOM ({len(req.ids)} tokens), freeing {self.bd.idle()} idle slot decoders + {len(self.parked)} '
              f'parked states + {len(self.spec_live)} drafter States (spec off) and retrying: {e}'[:300], flush=True)
        self.free_vram()

    @torch.no_grad()
    def prefill_chunk(self, req, n=None):
        """One prefill chunk of n tokens (default INK_CHUNK) -> True when the prompt is done (then parked, if worth it).
        A CUDA OOM restarts the prefill once from the best parked/saved point (a chunk can't be retried: forward
        updates conv states in place); twice -> OutOfMemory. The restart may raise NoRoom."""
        try:
            return self._chunk(req, n or self.chunk)
        except torch.OutOfMemoryError as e:
            self._oom(req, e)
        self.start(req)
        return False

    def _chunk(self, req, n):
        m, ids, a = self.model, req.ids, req.a
        if req.img is None:  # vision only for images not covered by the restored prefix
            t0 = time.time()
            img_pos, rows, poss, off = (torch.tensor(ids) == chat.IMG_TOK).nonzero().flatten(), [], [], 0
            for tiles, r, c, _ in req.pre:
                p = img_pos[off:off + r * c]; off += r * c
                if p[-1] >= req.P:
                    rows.append(self.vision(tiles, r, c)); poss.append(p)
                    self.vision_images += 1
            req.img = (torch.cat(rows), torch.cat(poss)) if rows else ()
            req.stats['vision_s'] = time.time() - t0
        if a < len(ids):
            b = min(len(ids), a + n)
            ck = CKPT and req.keep and self.pool.dir
            if ck:  # end chunks on checkpoint boundaries (any split gives the same result)
                b = min(b, (a // CKPT + 1) * CKPT)
            margin = MARGIN if getattr(self, '_margin', None) is None else self._margin
            self.free_vram(margin * max(b - a, self.chunk // 2) // self.chunk, spec=True)
            x = m.embed(ids[a:b])
            if req.img:
                img, img_pos = req.img
                sel = (img_pos >= a) & (img_pos < b)
                if sel.any():
                    x[(img_pos[sel] - a).to(x.device)] = img[sel.nonzero().flatten().to(img.device)].to(x.dtype)
            if req.sp is not None and req.sp.st is not None:
                f, taps = forward_taps(m, x, req.seq)
                try:
                    req.sp.prompt(taps, a)
                except Exception as e:  # (drafter only: the target's prefill above is done and kept)
                    self._spec_fail(req, e)
                req.f, req.a = f[-1:], b
            else:
                req.f, req.a = m.forward(x, req.seq)[-1:], b
            if ck and b % CKPT == 0 and b < len(ids):  # (the prompt's end is parked as usual)
                self._ckpt(req, b)
            if len(ids) > 8 * n and (a // n) % 16 == 15:  # progress for long prompts
                print(f'prefill {b}/{len(ids)} tokens, {time.time() - req.t_start:.0f} s', flush=True)
            if b < len(ids):
                return False
        if req.P < len(ids) and len(ids) >= SNAP_MIN:
            self.park(req.keys, req.seq, req.f, req.keep)
        if not req.out:  # (a resumed decode keeps its first prefill's numbers)
            req.stats.update(prompt_tokens=len(ids), cached_tokens=req.P, prefill_s=time.time() - req.t_start)
        return True

    def preempt_prefill(self, req, park=True):
        """Stop a prefill between chunks: the progress is parked (the next start resumes there), pages released."""
        if park and req.a > req.P:
            self.park(req.keys[:req.a], req.seq, req.f, keep=req.a >= DISK_MIN)
        self.finish(req)

    @torch.no_grad()
    def to_decode(self, req, slot):
        """Prompt done: load the sequence into decode slot `slot`. A fresh request samples its first token here (from
        the prompt's last hidden); a resumed one goes on with its pending token. May finish the request (req.done)."""
        req.solo = False  # solo frees memory for the prefill only; holding it through decode would block admission
        try:
            req.slot, req.pos = slot, len(req.ids)
            if self.bd.dec and self.bd.decs[slot] is None:  # BatchDecoder: the load allocates slots 0..slot (or all)
                k = self.bd.new_slots(slot) if hasattr(self.bd, 'new_slots') else self.slots
                self.free_vram(SLOT_NEED + (k - 1) * SLOT_MB if self.batched else SLOT_NEED)
            self.bd.load(slot, req.seq)
            if req.sp is None and getattr(req, 'need', 0) > SPEC_DEFER and not req.oomed:  # long: drafter only now
                req.sp = self._spec(req, start=req.pos)
                if req.sp is not None and req.sp.st is not None:
                    self.spec_live.add(req)
            if req.sp is not None:
                req.sp.attach(slot, req.pos)
            if not req.out:
                req.t1 = time.time()
                self._take([req], self.model.logits(req.f))
        except torch.OutOfMemoryError as e:
            self._decode_oom(e)

    @torch.no_grad()
    def decode_step(self, reqs):
        """One batched step: every request feeds its pending token and samples the next one (own grammar mask,
        sampler settings and RNG); a spec request may instead verify k drafts (rows <= RMAX in all) and emit 1..k+1
        tokens. Finished requests get req.done; the caller finish()es them."""
        try:
            if not any(r.sp is not None for r in reqs):
                lg = self.bd.step([(r.slot, r.pos, r.out[-1]) for r in reqs])
                for r in reqs:
                    r.pos += 1
                return self._take(reqs, lg)
            rows, ctxs, spare, tok = [], [], RMAX - len(reqs), {}
            for r in reqs:
                if r.sp is not None:
                    try:
                        rw, ctx = r.sp.propose(r.pos, r.out[-1], r.max_tokens - len(r.out), kcap=spare,
                                               hist=lambda r=r: r.prompt + r.out, batch=len(reqs))
                    except Exception as e:  # a drafter failure (graph capture, OOM) never fails the request
                        self._spec_fail(r, e)
                if r.sp is None:
                    rw, ctx = [(r.slot, r.pos, r.out[-1])], None
                else:
                    spare -= len(rw) - 1
                    if r.sp.dtok is not None:  # device draft ids (no host round trip before the verify)
                        tok[len(rows) + 1] = r.sp.dtok
                ctxs.append((len(rows), ctx))
                rows += rw
            lg = self.bd.step(rows, tok=tok) if tok else self.bd.step(rows)
            plain = [(r, r0) for r, (r0, ctx) in zip(reqs, ctxs) if ctx is None]
            for r, (r0, ctx) in zip(reqs, ctxs):  # every request's json masks before the first settle syncs
                if ctx is not None:
                    try:
                        r.sp.host(ctx)
                    except torch.OutOfMemoryError:
                        raise
                    except Exception as e:
                        self._fail_one(r, e)
            for r, (r0, ctx) in zip(reqs, ctxs):
                if ctx is not None and not r.done:
                    try:
                        self._settle(r, lg, r0, ctx)
                    except torch.OutOfMemoryError:
                        raise
                    except Exception as e:  # this request fails, the batch goes on
                        self._fail_one(r, e)
            if plain:
                ix = [r0 for _, r0 in plain]
                for r, _ in plain:
                    r.pos += 1
                self._take([r for r, _ in plain], lg if ix == list(range(len(rows))) else
                           lg.index_select(0, torch.tensor(ix, device=lg.device)))
        except torch.OutOfMemoryError as e:  # tokens may be out already and the state is half-updated: no retry
            self._decode_oom(e)

    def _fail_one(self, r, e):
        """An error while finishing r's part of a decode step: only r fails (r.err, r.done; the caller ends it with
        the error); a verify slot's uncommitted rows are dropped (finish frees the slot)."""
        import traceback
        traceback.print_exc()
        print(f'request failed in decode: {type(e).__name__}: {e}'[:300], flush=True)
        r.err, r.done = e, True
        pend = getattr(self.bd, 'pend', None)
        if isinstance(pend, dict) and r.slot is not None:
            pend.pop(r.slot, None)

    def _settle(self, r, lg, r0, ctx):
        """A spec request's part of a step: processors / logprobs of its verify rows, accept + commit, feed."""
        r.sp.host(ctx)  # (device draft ids: read, and cut / masked by the grammar)
        if r.proc:  # verify row i: the processors for the history out + d_1..d_i (as plain decoding)
            self._process_rows(r, lg[r0:r0 + len(ctx[0]) + 1], ctx[0])
        if r.lp is not None:  # target logprobs of the verify rows (+ the draft-path grammar masks)
            T = len(ctx[0]) + 1
            x = lg[r0:r0 + T].float()
            lp = torch.log_softmax(x if ctx[2] is None else x + ctx[2][:T], -1)
        out = r.sp.settle(lg, r0, ctx)
        r.pos += r.sp.ncommit
        infos = self._lp_info(lp[:len(out)], out, [r.lp] * len(out)) if r.lp is not None else ()
        for i, t in enumerate(out):  # (settle advanced the grammar past all of out: done only at the end)
            if not r.done:
                self._feed(r, t, infos[i] if infos else None, last=i == len(out) - 1)

    def _decode_oom(self, e):
        self.ooms += 1
        self.free_vram()
        raise OutOfMemory('out of GPU memory during decode') from e

    def _take(self, reqs, lg):
        """Logit processors, mask + sample all rows on device (grammar.mask_rows, sample_rows), one host sync, then feed
        the tokens. lg [R, VOCAB] may be the decoder's static buffer: modified + sampled before the next step.
        logprobs (requests that asked) come from the processed + masked logits, before temperature / top-p / top-k /
        min-p: the model's distribution as OpenAI reports it (independent of the sampling settings)."""
        for i, r in enumerate(reqs):
            if r.proc:
                try:
                    self._process(r, lg[i])
                except torch.OutOfMemoryError:
                    raise
                except Exception as e:
                    self._fail_one(r, e)
        try:
            self._sample_feed(reqs, lg)
        except torch.OutOfMemoryError:
            raise
        except Exception as e:  # the batched mask / sample part (not attributable): these rows' requests fail
            for r in reqs:
                if not r.done:
                    self._fail_one(r, e)

    def _sample_feed(self, reqs, lg):
        if any(r.cons is not None for r in reqs):
            grammar.mask_rows(lg, [r.cons for r in reqs])
        li = [i for i, r in enumerate(reqs) if r.lp is not None]
        if li:
            lp = torch.log_softmax(lg[li].float(), -1)
        for i, r in enumerate(reqs):
            if r.min_p and r.temperature and r.temperature > 0:  # p >= min_p * p_max  <=>  l >= l_max + T ln(min_p)
                row = lg[i]
                row.masked_fill_(row < row.max() + r.temperature * math.log(r.min_p), float('-inf'))
        V, dev = lg.shape[1], lg.device
        toks = sample_rows(lg, [(r.temperature, r.top_p, r.top_k, pos_noise(r.gen, r.seed, len(r.prompt) + len(r.out), V, dev)
                                 if r.temperature and r.temperature > 0 else None) for r in reqs]).tolist()
        infos = dict(zip(li, self._lp_info(lp, [toks[i] for i in li], [reqs[i].lp for i in li]))) if li else {}
        for i, (r, t) in enumerate(zip(reqs, toks)):
            if r.done:  # (failed in _process)
                continue
            try:
                if r.cons is not None:
                    r.cons.accept(t)
                self._feed(r, t, infos.get(i))
            except Exception as e:
                self._fail_one(r, e)

    def _process(self, r, row):
        """Per-request logit processors on one row [VOCAB], in place: logit_bias, then presence / frequency
        penalties (OpenAI: minus pres * [count > 0] + freq * count over the output so far), repetition penalty
        (divide positive / multiply negative logits of tokens already in the output), and the stop tokens held off
        until min_tokens."""
        if r.bias is not None:
            if r.bias[0].device != row.device:
                r.bias = tuple(x.to(row.device) for x in r.bias)
            row.index_add_(0, r.bias[0], r.bias[1].to(row.dtype))
        if r.pen is not None and r.out:
            if r.cnt is None:
                r.cnt = torch.zeros(row.shape[0], device=row.device)
            if r.ncnt < len(r.out):
                new = torch.tensor(r.out[r.ncnt:], device=row.device)
                r.cnt.index_add_(0, new, torch.ones(len(new), device=row.device))
                r.ncnt = len(r.out)
            pres, freq, rep = r.pen
            seen = r.cnt > 0
            if pres or freq:
                row.sub_((freq * r.cnt + pres * seen).to(row.dtype))
            if rep != 1:
                row.copy_(torch.where(seen, torch.where(row > 0, row / rep, row * rep), row))
        if len(r.out) < r.min_tokens:
            row[r.ban] = float('-inf')

    def _process_rows(self, r, lg, d):
        """Verify rows lg [T, V] in place: row i gets r's logit processors as if d[:i] were already in the output
        (the position plain decoding would process with that history); r's counts are left as they were."""
        out0, cnt0, n0 = r.out, None if r.cnt is None else r.cnt.clone(), r.ncnt
        try:
            for i in range(lg.shape[0]):
                r.out = out0 + list(d[:i])
                self._process(r, lg[i])
        finally:
            r.out, r.cnt, r.ncnt = out0, cnt0, n0 if cnt0 is not None else 0

    def _lp_info(self, lp, toks, ks):
        """lp [n, V] log-probs, toks [n] chosen ids, ks [n] top_logprobs counts -> per row (logprob, [(id, logprob)]),
        one host copy each for the chosen and the top-k values."""
        n = len(toks)
        ch = lp[torch.arange(n, device=lp.device), torch.tensor(toks, device=lp.device)].tolist()
        k = max(ks, default=0)
        top = [[]] * n
        if k:
            v, ix = lp.topk(k, -1)
            v, ix = v.tolist(), ix.tolist()
            top = [list(zip(ix[j][:ks[j]], v[j][:ks[j]])) for j in range(n)]
        return list(zip(ch, top))

    def tok_entry(self, t, lp):
        """OpenAI logprob entry of token t: text, logprob (-inf -> -9999.0, OpenAI's floor), utf-8 bytes."""
        b = _BYTES.get(t)
        if b is None:
            s = self.tok.id_to_token(t) or ''
            b = _BYTES[t] = bytes(_B2U_INV.get(c, 0) for c in s) if t < chat.USER else s.encode()
        return {'token': b.decode('utf-8', 'replace'), 'logprob': max(lp, -9999.0), 'bytes': list(b)}

    def _feed(self, r, t, li=None, last=True):
        """Emit token t. last: no more tokens of this step follow (a spec round feeds several; r.cons already
        accepted all of them, so its done flag holds only for the last one)."""
        r.out.append(t)
        n, st = len(r.out), r.stopper
        r.n_reason += r.parser.kind == 'reasoning'
        if li is not None and r.parser.kind == 'content' and t < chat.USER:  # a content token: its logprob entry
            r.lp_pend.append(dict(self.tok_entry(t, li[0]), top_logprobs=[self.tok_entry(i, v) for i, v in li[1]]))

        def emit(k, d):
            if k == 'content' and r.lp_pend:
                r.emit('delta', (k, d, r.lp_pend))
                r.lp_pend = []
            else:
                r.emit('delta', (k, d))
        if t != chat.EOS and t not in r.stop_ids:
            for k, d in r.parser.feed(t):
                if st is not None:
                    if k != 'content':  # the held content tail goes out before other kinds (order kept)
                        if held := st.flush():
                            emit('content', held)
                    else:
                        d, r.hit = st.feed(d)
                if d:
                    emit(k, d)
                if r.hit:
                    break
        cdone = r.cons is not None and r.cons.done and last
        one = r.one_call and r.parser.tool_calls  # parallel_tool_calls false: the reply ends after its first call
        # stop string; thinking off: the pre-filled answer message is the whole reply; json done; no step for a token
        # that would never be sampled
        if t == chat.EOS or t in r.stop_ids or r.hit or (r.prefill_text and t == chat.END) or cdone or one or \
                n >= r.max_tokens:
            r.done = True
            if st is not None and (held := st.flush()):
                emit('content', held)
            if r.lp_pend:  # entries of tokens whose text was held back or cut (stop string)
                emit('content', '')
            natural = r.hit or t in (chat.EOS, chat.END) or t in r.stop_ids or cdone or one
            r.stats.update(completion_tokens=n, reasoning_tokens=r.n_reason, decode_s=time.time() - r.t1, tool_calls=r.parser.tool_calls,
                           finish_reason='length' if n >= r.max_tokens and not natural else 'stop')

    def drop_parked(self, req):
        """req was cancelled while waiting after a preemption: unpark its entry (a prefix of req's keys parked by
        that preemption), no disk write."""
        keys = self.keys(req.ids, req.pre)
        mine = [e for e in self.parked if len(e[0]) > req.P and len(e[0]) <= len(keys) and torch.equal(e[0], keys[:len(e[0])])
                and (len(e[0]) < len(keys) or 'f' not in e[2])]  # (a finished prompt's own entry stays: still cache)
        self.parked = [e for e in self.parked if not any(e is x for x in mine)]
        for e in mine:
            self.unpark(e, disk=False)

    def preempt_decode(self, req):
        """Pause a decode at its position, no recompute later: keys = prompt + fed tokens, pages + the batch decoder's
        state parked (keep: to disk if the pool needs the pages), slot and pages released. The next start restores it
        and to_decode continues with the pending token (without a parked state it re-prefills those tokens)."""
        req.ids = req.prompt + req.out[:-1]
        assert len(req.ids) == req.pos, (len(req.ids), req.pos)
        try:
            st = self.bd.state(req.slot)
        except torch.OutOfMemoryError:
            st = None
            print(f'decode park skipped at {req.pos} tokens: CUDA OOM', flush=True)
        if st is not None:
            self.park(self.keys(req.ids, req.pre), req.seq, None, True, st)
        self.finish(req)

    def _spec_fail(self, req, e):
        """The drafter failed for req (capture error, CUDA OOM, ...): req decodes plain from here; its State is
        dropped (not reused: it may be half-updated), its graphs with it."""
        print(f'spec off for a request: drafter failed: {type(e).__name__}: {e}'[:300], flush=True)
        sp, req.sp = req.sp, None
        self.spec_live.discard(req)
        self.spec_fails = getattr(self, 'spec_fails', 0) + 1
        if sp is not None and sp.st is not None:
            sp.st.graphs.clear()
            self.ds.live -= 1
        if self.model.dev.type == 'cuda':
            torch.cuda.empty_cache()

    def _end_spec(self, req):
        """Give the request's drafter State back (reused by the next spec request with its graphs)."""
        sp, req.sp = req.sp, None
        self.spec_live.discard(req)
        if sp is not None and sp.st is not None:
            self.ds_free.append(sp.st)
            self.ds.live -= 1

    def finish(self, req):
        """Release the request's slot and pages (done, error, cancel, or after parking)."""
        self._end_spec(req)
        if req.slot is not None:
            self.bd.free(req.slot)
            req.slot = None
            if not self.bd.busy():  # idle: keep one warm Decoder, so idle VRAM doesn't depend on past concurrency
                self.bd.drop(keep=1)
                self.drop_drafter(keep=1)
        req.img = None  # (vision rows of a request waiting in line; start() recomputes what it needs)
        if req.seq is not None:
            self.pool.release(req.seq.kv)  # decoded rows are not cached: the next turn re-renders the reply differently
            req.seq = None

    def _prefill_all(self, req, preempt):
        self.start(req)
        while True:
            if req.a < len(req.ids) and preempt is not None and (why := preempt()):
                self.preempt_prefill(req, park=why != 'cancel')  # a client that hung up won't rerun it: no park
                raise Preempted
            if self.prefill_chunk(req):
                return

    @torch.no_grad()
    def prefill(self, ids, pre, stats=None, preempt=None, max_tokens=0, keep=False):
        """-> (seq, f [1,H]) after the prompt (tests, tools): pages for prompt + max_tokens reserved, the caller
        releases them (pool.release(seq.kv)). keep: worth a disk copy when it leaves the GPU. preempt() as in generate."""
        req = Request(self, list(ids), pre, keep, max_tokens, stats=stats)
        self._prefill_all(req, preempt)
        return req.seq, req.f

    @torch.no_grad()
    def warmup(self, mix=2048):
        """Startup warm-up (server, before it listens): synthetic requests through the paths whose first use compiles
        Triton kernels (int args specialize on 16-alignment), captures CUDA graphs or loads cuBLAS/cuDNN kernels, so the
        first real request doesn't pay it. Covers: prefill chunks of INK_CHUNK, `mix` (the chunk while decodes run) and
        short tails, at 16-aligned and unaligned start/length; near + far prefill kernels (far keys up to ~6k); short
        prompts without a far part; park + restore from the GPU; decode graph capture in slots 0 and 1, both rows in
        one step; greedy and top-p/top-k sampling; a json_object grammar mask; a decode park + resume; vision on a tiny
        image. Disk is off meanwhile (no SSD reads/writes); afterwards no parked entry and no prefix-tree page remain
        and all pages are free; one warm slot Decoder stays (the normal idle state after any traffic). -> seconds.
        Flex far attention: Model.warm_flex (one fixed shape). Not warmed:
        graph capture of slots >= 2 (kernels already compiled, capture only), disk restore, per-schema json_schema
        compiles, image shapes other than the tiny one, Triton specializations of int args not hit here."""
        t0, C, pool = time.time(), self.chunk, self.pool
        parked0, nodes0, disk = list(self.parked), set(pool.nodes), pool.dir
        unit = self.tok.encode('The quick brown fox jumps over the lazy dog. ').ids
        reqs = []

        def req(n, rot=0, **kw):  # same rot: a prefix of the longer prompts (r2 restores r1's parked state)
            u = unit[rot:] + unit[:rot]
            reqs.append(Request(self, (u * (n // len(u) + 1))[:n], [], False, 64, **kw))
            return reqs[-1]

        def step(rs):
            if live := [r for r in rs if not r.done]:
                self.decode_step(live)
        pool.dir = None  # no disk restore, spill or state file while warming
        try:
            warm_flex = getattr(self.model, 'warm_flex', None)
            if warm_flex is not None:  # flex far attention compiles once here, not on the first prompt past INK_PA_FAR_MAX
                warm_flex()
            if self.vision is not None:
                import io
                from PIL import Image
                b = io.BytesIO()
                Image.new('RGB', (80, 120), (200, 30, 30)).save(b, 'PNG')
                self.vision(*preprocess(load_image(b.getvalue())))
            for n in (96, 97):  # short prompts (not prefixes of r1): near kernel without a far part
                r = req(n, rot=n - 95)
                self.start(r)
                self.prefill_chunk(r)
                self.finish(r)
            r1 = req(C + mix + 37, temperature=0)
            self.start(r1)
            ns = iter((C, mix))
            while not self.prefill_chunk(r1, next(ns, C)):  # C at 0, mix at C, then the 37-token tail
                pass
            self.to_decode(r1, 0)
            step([r1])  # slot 0: Decoder + graph capture
            r2 = req(len(r1.ids) + mix + 21, temperature=1.0, top_p=0.9, top_k=40,
                     response_format=grammar.compile_format(self.tok, {'type': 'json_object'}))
            self.start(r2)  # restores r1's parked prompt: chunks at an unaligned start
            assert r2.P == len(r1.ids), (r2.P, len(r1.ids))
            while not self.prefill_chunk(r2, mix):
                step([r1])  # interleaved, as the scheduler does
            self.to_decode(r2, 1)
            step([r1, r2])  # slot 1's capture, then 2 rows in one step
            if not r2.done:
                self.preempt_decode(r2)  # park a decode at its position, then resume it from the GPU
                self.start(r2)
                self.to_decode(r2, 1)
                step([r1, r2])
        finally:
            for r in reqs:
                self.finish(r)
            for e in [e for e in self.parked if not any(e is p for p in parked0)]:
                self.parked = [x for x in self.parked if x is not e]
                self.unpark(e, disk=False)
            for k in set(pool.nodes) - nodes0:  # the warm-up's prefix-tree pages: dropped, not left as cache
                pool._drop_node(pool.nodes[k])
            pool.dir = disk
        self.warm_drafter()
        if self.model.dev.type == 'cuda':
            torch.cuda.empty_cache()
        return time.time() - t0

    @torch.no_grad()
    def warm_drafter(self, n=None):
        """Capture the drafter graphs of the idle State kept after the warm-up (the next spec request reuses it):
        greedy AND sampled, every width k = 1..7 (INK_SPEC_GATE: k = 7 only, the gate always drafts 7), plus the
        context-only pass (k = 0), at a context of n positions (default 1000: the smallest layer-1 bucket with
        INK_DSPARK_CTXB; other buckets are captured on first use). The State is reset afterwards. -> keys run."""
        if self.ds is None or not self.ds_free:
            return []
        import spec_runner
        ds, st, dev = self.ds, self.ds_free[-1], self.model.dev
        n = 1000 if n is None else n
        ks = (7,) if spec_runner.GATE else range(1, 8)
        gen, keys = torch.Generator(device=dev), []
        gen.manual_seed(0)
        ds.reset(st, n)
        try:
            ds.stage(st, torch.zeros(4, 1, H, dtype=ds.dt, device=dev))
            ds.flush(st)
            keys.append((0, True, ds.bucket(st)))
            for temp in (0.0, 1.0):
                for k in ks:
                    ds.round(st, 0, k, temp, gen, conf=bool(spec_runner.GATE))
                    keys.append((k, temp <= 0, ds.bucket(st)))
        finally:
            ds.reset(st, 0)
        return keys

    @torch.no_grad()
    def generate(self, messages, preempt=None, **kw):
        """Single-request wrapper over the step functions (tests, tools; the server runs server.Scheduler). Yields
        (kind, text) deltas, kind in reasoning|content; stats (kw) gets usage/timings. preempt(): checked before each
        prefill chunk; truthy parks the progress and raises Preempted ('cancel': no park), nothing yielded yet.
        kw: request()'s (max_tokens, temperature, top_p, top_k, effort, tools, seed, stats, prefill_text,
        response_format, stop)."""
        out = []
        req = self.request(messages, emit=lambda kind, v: out.append(v), **kw)
        try:
            self._prefill_all(req, preempt)
            self.to_decode(req, 0)
            while True:
                yield from (v for v in out if v[0] in ('reasoning', 'content'))  # (not the call_* deltas)
                out.clear()
                if getattr(req, 'err', None) is not None:
                    raise req.err
                if req.done:
                    break
                self.decode_step([req])
        finally:
            self.finish(req)
