"""OpenAI-compatible HTTP server for the Inkling engine. Scheduler: priority queue + continuous batching (up to
INK_MAX_BATCH requests decode together, one prefill chunk at a time interleaved with batched decode steps).
Run: cd engine && PATH=$PWD/.venv/bin:$PATH .venv/bin/python server.py   (env: INK_HOST, INK_PORT, INK_MAX_CTX, INK_SERVED,
INK_SPEC=dspark|lookup|off; body field `spec: false` turns speculative decoding off for one request)"""
import asyncio
import base64
import heapq
import itertools
import json
import os

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')  # less fragmentation near the 32 GB limit
# Verified fast paths (GPU windows 2026-11-21/22) are the defaults on cuda; set any of these to go back, e.g.
# INK_GRAPH=0, INK_MOE=torch, INK_PREFILL_ATTN=torch, INK_DEC_FUSE=0, INK_MOE_V2=0. INK_DEC_EMBED_GPU stays off (+0.9 GB
# VRAM); INK_MOE_FP4ACT stays off (numerics change, waits for the user's OK: deploy/CUTOVER.md).
# 2026-11-23 (user-approved): + fused prefill glue (GLUE/GLUE2, KL within noise, ~1.4x prefill), FP4-activation MoE,
# FP8 decode weights (gate/needles/evals pass), DSpark shortlist + ctx buckets + device accept (window spec2).
# Off on purpose: INK_FP8_DENSE (+3%, 1.5x noise), INK_SPEC_GATE (hurts json), INK_MOE_FP4_TERMS, GATEFUSE/MOETAIL/FARSKIP.
CUDA_DEFAULTS = dict(INK_MOE='triton', INK_GRAPH='1', INK_CHUNK='4096', INK_PREFILL_ATTN='auto', INK_DEC_FUSE='1',
                     INK_DEC_FAR_DOT='1', INK_MOE_V2='1', INK_GLUE='triton', INK_GLUE2='1', INK_MOE_FP4ACT='1',
                     INK_FP8_DECODE='1', INK_DSPARK_SHORT='32768', INK_DSPARK_CTXB='1', INK_SPEC_ACCEPT_DEV='1',
                     # 2026-11-23 window prefill4: bit-identical MoE routing/amax + tile winners (-4% prefill, no spills)
                     INK_MOE_ROUTE='1', INK_MOE_AMAX='1', INK_MOE_FP4_CFG='w13=128,128,8,2',
                     INK_MOE_MX_CFG='w13=128,64,8,2;w2=128,64,8,3', INK_MOE_FP4_BB='64', INK_MOE_FP4_BS='32',
                     # window spec4: copy+DSpark hybrid + fixed gate, edge 1.5 (agent tool 4.4x, edit 5.8x, greedy == plain)
                     INK_SPEC_HYBRID='1', INK_SPEC_GATE='1', INK_SPEC_COPY_EDGE='1.5',
                     INK_VIT_GPU_PRE='1',
                     INK_CKPT_EVERY='32768',
                     INK_DEC_FAR2='1')  # window far2 11-23: +17-20% decode at 131k-1M; per-row KL ~1-2.7x own 1-ulp noise  # window final5: 150k edited at 100k -> resume at 98k, 7.6 s vs 16.6 s cold   # window vision: bit-identical, -160 ms at 16.7 MP (more on a loaded host)
if __name__ == '__main__' and os.environ.get('INK_DEVICE', 'cuda') == 'cuda':  # before the imports below read them
    for k, v in CUDA_DEFAULTS.items():
        os.environ.setdefault(k, v)
import threading
import time
import uuid
import weakref
from types import SimpleNamespace

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

import chat
import grammar
from engine import BadRequest, Engine, NoRoom, OutOfMemory

SERVED = os.environ.get('INK_SERVED', 'inkling-2-mini')
NOROOM_S = float(os.environ.get('INK_NOROOM_S', 30))  # NoRoom with nothing running this long: fail instead
DECODE_SHARE = float(os.environ.get('INK_DECODE_SHARE', 0.5))  # wall-time share of decode steps vs prefill chunks
MIX_CHUNK = int(os.environ.get('INK_MIX_CHUNK', 2048))  # prefill chunk while decodes run (caps their stall)
# short-first (default on; 0 = strict FIFO prefills as before): a waiting request whose uncached suffix is <= SHORT_MAX
# preempts a same-priority prefill with > 4x that left (parked on the GPU, resumes later); a prefill gets >= 1 chunk
# after each (re)start before it can be preempted this way again. Suffixes < SHORT_MIX get full chunks and decode share
# 0.3 while decodes run (a warm agent turn is a few k tokens: finish it fast).
SHORT_FIRST = os.environ.get('INK_SHORT_FIRST', '1') == '1'
SHORT_MAX = int(os.environ.get('INK_SHORT_MAX', 16384))
SHORT_MIX = int(os.environ.get('INK_SHORT_MIX', 8192))
# INK_PARK_AUTO (default on; 0 = keep INK_PARK): after warm-up, parked chat states kept on the GPU = what free VRAM
# beyond INK_VRAM_MARGIN_MB holds, clamped to [INK_PARK, 6] (free_vram still pushes them out under pressure)
PARK_AUTO = os.environ.get('INK_PARK_AUTO', '1') == '1'
app = FastAPI()
eng = None
M = dict(requests=0, preemptions=0, prompt_tokens=0, cached_tokens=0, completion_tokens=0, prefill_s=0.0, decode_s=0.0,
         last_prefill_tps=0.0, last_decode_tps=0.0, decode_steps=0, decode_rows=0, warmup_seconds=0.0,
         decode_step_s=0.0, decode_gap_s=0.0,  # in decode_step / between back-to-back decode steps (host loop cost)
         # turn-latency breakdown (timing only; per request too, body `timings: true`, see _instrument / timings()):
         tokenize_s=0.0, queue_s=0.0, start_s=0.0, restore_s=0.0, disk_lookup_s=0.0, disk_state=0, disk_pages=0,
         disk_page_s=0.0, prefill_chunk_s=0.0, sync_write_s=0.0, sync_writes=0, decoder_allocs=0, decoder_alloc_s=0.0, solo_starts=0,
         short_preemptions=0,
         last_ttft_s=0.0, last_queue_s=0.0, last_restore_s=0.0)
_cur = None  # the request whose engine call runs on the scheduler thread now (timing attribution), else None


def _tm(r):
    st = getattr(r, 'stats', None)
    return st.setdefault('tm', {}) if isinstance(st, dict) else {}


def _add(k, v, r=None):
    """Add v to timing k of request r (default: the one whose engine call is running) and to the /metrics counter."""
    r = r or _cur
    if r is not None:
        t = _tm(r)
        t[k] = t.get(k, 0) + v
    M[k] = M.get(k, 0) + v


def _instrument(eng):
    """Timing-only wrappers (no behaviour change: same args, result, exceptions) on the engine / pool methods where turn
    latency hides: prefix restore (GPU fork or disk), disk state + page reads, synchronous disk writes (page spills on
    eviction, parked states saved on unpark) and BatchDecoder buffer re-allocations (after a solo drop). Missing
    methods (test fakes) are skipped."""
    if getattr(eng, '_tm_wrapped', False):
        return

    def wrap(obj, name, after):
        fn = getattr(obj, name, None)
        if not callable(fn):
            return

        def w(*a, **k):
            t0, out = time.perf_counter(), None
            try:
                out = fn(*a, **k)
                return out
            finally:
                after(time.perf_counter() - t0, out)
        setattr(obj, name, w)
    pool, bd = getattr(eng, 'pool', None), getattr(eng, 'bd', None)
    wrap(eng, 'restore', lambda dt, out: _add('restore_s', dt))
    if pool is not None:
        wrap(pool, 'restore', lambda dt, out: (_add('disk_lookup_s', dt),
                                               out and out[0] is not None and _add('disk_state', 1)))
        wrap(pool, '_load_page', lambda dt, out: out is not None and (_add('disk_pages', 1), _add('disk_page_s', dt)))
        wrap(pool, '_save', lambda dt, out: (_add('sync_write_s', dt), _add('sync_writes', 1)))
    if bd is not None:
        wrap(bd, '_alloc', lambda dt, out: (_add('decoder_allocs', 1), _add('decoder_alloc_s', dt)))
    eng._tm_wrapped = True


def timings(st):
    """Per-request latency breakdown (seconds) for the response when the body asks `timings: true`: tokenize_s (render +
    images, executor), queue_s (submit -> start), start_s (restore + page reservation + spec setup), restore_s (prefix
    lookup + GPU fork or disk read), disk_* (disk state restore, pages read), prefill_chunk_s (this request's prefill chunks,
    incl. the end-of-prompt park), first_token_s (to_decode), ttft_s (arrival -> first token sampled), sync_write_s
    (disk writes on the scheduler thread during this request's start / prefill), step_max_s (longest decode step it was
    in), src (cold | gpu | disk | disk+pages)."""
    t = dict(st.get('tm', {}))
    arr, at = t.pop('arrive', None), t.pop('ttft_at', None)
    t.pop('submit', None)
    if arr and at:
        t['ttft_s'] = at - arr
    t['src'] = 'cold' if not st.get('cached_tokens') else 'disk+pages' if t.get('disk_pages') else \
        'disk' if t.get('disk_state') else 'gpu'
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in t.items()}


class Outbox:
    """Scheduler thread -> event loop: messages queue here and one call_soon_threadsafe delivers all that piled up (one
    loop wake per burst, e.g. a decode step's deltas of every request, not one per token). Order is kept."""
    boxes = weakref.WeakKeyDictionary()  # loop -> Outbox

    def __init__(self, loop):
        self.loop, self.items, self.lock = loop, [], threading.Lock()

    @classmethod
    def of(cls, loop):
        return cls.boxes.get(loop) or cls.boxes.setdefault(loop, cls(loop))

    def put(self, q, m):
        with self.lock:
            self.items.append((q, m))
            first = len(self.items) == 1
        if first:
            self.loop.call_soon_threadsafe(self.drain)

    def drain(self):
        with self.lock:
            items, self.items = self.items, []
        for q, m in items:
            q.put_nowait(m)


class Scheduler:
    """Continuous batching on one worker thread (DESIGN-batching.md 2.2). Waiting requests sit in a heap ordered by
    (priority desc, arrival). Running: at most one prefill (chunked) plus the decodes, together <= eng.slots.
    Each iteration:
    - cancelled requests end (slot + pages freed, no park);
    - admission: one prefill at a time. Priority never pauses running work: a running prefill or decode runs to the
      end, and a higher-priority request only jumps to the front of the queue (next prefill, next free slot / pages).
      The one exception is short-first (below), within a level. Then the head of the queue starts. If it gets
      NoRoom it blocks and nothing behind it starts until running requests free pages (protects a big request from
      a stream of small ones). NoRoom with nothing running gives up after INK_NOROOM_S;
    - one step: a prefill chunk (INK_CHUNK; INK_MIX_CHUNK while decodes run) or one batched decode step, by time
      share: decodes get INK_DECODE_SHARE of the wall time (0.8 when they outrank the prefill, 0.3 when it outranks
      them), tracked as measured time (whoever is behind runs next).
    Within a level every decode gets a row every step; prefills are FIFO, except short-first (INK_SHORT_FIRST): a
    short warm turn preempts a same-priority long prefill (see _short_first). A solo prefill (big uncached suffix,
    engine DEC_DROP) runs alone, as every prefill does; solo ends when it starts decoding."""

    def __init__(self, eng):
        self.eng, self.cv, self.heap, self.n = eng, threading.Condition(), [], itertools.count()
        self.prefill, self.decoding = None, []
        self.tp = self.td = 0.0  # prefill / decode time while both kinds run (time share)
        self.retry = 0.0         # blocked head: no admission attempt before this (reset when anything changes)
        self.dend = 0.0          # end of the last decode step if nothing else ran since (decode_gap_s)
        _instrument(eng)
        threading.Thread(target=self._worker, daemon=True).start()

    def submit(self, req):
        _tm(req).setdefault('submit', time.time())  # (a preempted request keeps its first submit)
        with self.cv:
            if getattr(req, 'order', None) is None:  # a preempted request keeps its place in line
                req.order = next(self.n)
            heapq.heappush(self.heap, (-req.prio, req.order, req))
            self.retry = 0.0
            self.cv.notify()

    def depth(self):
        return len(self.heap)

    def running(self):
        return len(self.decoding) + (self.prefill is not None)

    def active(self):
        return ([self.prefill] if self.prefill is not None else []) + self.decoding

    def _worker(self):
        while True:
            try:
                self._iterate()
            except Exception:  # requests report their own errors; never let the worker die
                import traceback; traceback.print_exc()
                time.sleep(0.5)

    def _iterate(self):
        with self.cv:
            while not self.heap and not self.running():
                self.cv.wait()
        for r in self.active():
            if r.cancel.is_set():
                self._end(r, None)  # client gone: no park
        self._admit()
        if self.running():
            return self._step()
        self.dend = 0.0
        with self.cv:  # blocked or only cancelled waiters: until a submit or the next retry
            if self.heap:
                self.cv.wait(max(0.01, self.retry - time.time()) if self.retry else 1.0)

    # ------------------------------------------------------------------ admission
    def _admit(self):
        eng, jump = self.eng, None
        while True:
            with self.cv:
                gone = [it for it in self.heap if it[2].cancel.is_set()]
                if gone:
                    self.heap = [it for it in self.heap if not it[2].cancel.is_set()]
                    heapq.heapify(self.heap)
                q = [it[2] for it in sorted(self.heap, key=lambda it: it[:2])]
            for it in gone:  # a preempted request's parked progress is useless now (no disk write either)
                drop = getattr(eng, 'drop_parked', None)
                if drop is not None and getattr(it[2], 'keys', None) is not None:
                    drop(it[2])
            if not q:
                return
            if jump is not None and any(x is jump for x in q):  # short-first: it goes before the prefill it paused
                q = [jump] + [x for x in q if x is not jump]
            head, pf, jump = q[0], self.prefill, None
            if pf is not None:
                if self._short_first(head, pf):
                    self._preempt(pf)
                    jump = head
                    continue
                return  # one prefill at a time
            if self.running() >= eng.slots:
                return
            if time.time() < self.retry:
                return
            e = self._start(head) if eng.can_start(head) else NoRoom('no free pages')
            if e is None:
                return
            head.noroom0 = getattr(head, 'noroom0', None) or time.time()
            if not self.running() and time.time() - head.noroom0 >= NOROOM_S:
                pool = getattr(eng, 'pool', None)
                print(f'NoRoom for {NOROOM_S:g} s ({e}): free pages {pool.free_pages()}, parked {len(eng.parked)}, '
                      f'tree nodes {len(pool.nodes)}', flush=True)
                self._end(head, 'error', f'KV cache full: no room for {e} after {NOROOM_S:g} s')
                continue
            # head-of-line: nobody behind a blocked head starts (they are all of the same or lower priority); with no
            # overcommit a page-blocked head leaves no pages that could be used without delaying it
            self.retry = time.time() + 1.0  # (or sooner: any end / preemption / submit resets it)
            return

    def _start(self, r):
        """-> None if r is now the prefill (or failed and ended), else the NoRoom it got."""
        global _cur
        t0, _cur, tm = time.time(), r, _tm(r)
        try:
            self.eng.start(r)
        except NoRoom as e:
            return e
        except Exception as e:
            self._fail(r, e)
            return None
        finally:
            _cur = None
            _add('start_s', time.time() - t0, r)
        if 'queue_s' not in tm and 'submit' in tm:
            _add('queue_s', t0 - tm['submit'], r)
            M['last_queue_s'] = round(t0 - tm['submit'], 4)
        if getattr(r, 'solo', False):
            tm['solo'] = 1
            M['solo_starts'] += 1
        with self.cv:
            self.heap = [it for it in self.heap if it[2] is not r]
            heapq.heapify(self.heap)
        self.prefill, r.noroom0, r.chunks_run = r, None, 0
        f = getattr(self.eng, 'uncached', None)
        r.suffix = f(r) if f else None  # (prefill tokens left at start = its uncached suffix)
        return None

    def _short_first(self, head, pf):
        """Short-first: head (same priority as the running prefill pf) has a short uncached suffix and pf has > 4x that
        left, and pf ran a chunk since its (re)start (so a stream of short turns can't starve it)."""
        f = getattr(self.eng, 'uncached', None)
        if not SHORT_FIRST or f is None or head.prio != pf.prio or not getattr(pf, 'chunks_run', 0):
            return False
        n = f(head)
        if n > SHORT_MAX or f(pf) <= 4 * n:
            return False
        M['short_preemptions'] += 1
        return True

    def _preempt(self, r):
        global _cur
        _cur = r
        try:
            if r.cancel.is_set():  # the client is gone: free it, don't park anything for a rerun that won't come
                return self._end(r, None)
            self.prefill = None  # only the running prefill is ever preempted
            self.eng.preempt_prefill(r)
        except Exception as e:
            return self._fail(r, e)
        finally:
            _cur = None
        M['preemptions'] += 1
        self.submit(r)

    # ------------------------------------------------------------------ steps
    def _step(self):
        pf, dec = self.prefill, list(self.decoding)
        small = pf is not None and SHORT_FIRST and getattr(pf, 'suffix', None) is not None and pf.suffix < SHORT_MIX
        if pf is not None and dec:
            top = max(r.prio for r in dec)
            share = 0.8 if top > pf.prio else 0.3 if pf.prio > top or small else DECODE_SHARE
            decode = self.td * (1 - share) <= self.tp * share
        else:
            self.tp = self.td = 0.0
            decode = bool(dec)
        t0 = time.time()
        if decode:
            if self.dend:
                M['decode_gap_s'] += t0 - self.dend
            self._decode(dec)
            self.dend = time.time()
            M['decode_step_s'] += self.dend - t0
            self.td += self.dend - t0
        else:
            self._prefill(pf, min(MIX_CHUNK, self.eng.chunk) if dec and not small else self.eng.chunk)
            self.tp += time.time() - t0
            self.dend = 0.0

    def _prefill(self, pf, n):
        global _cur
        _cur, tm = pf, _tm(pf)
        try:
            t0 = time.time()
            try:
                if not self.eng.prefill_chunk(pf, n):
                    return
            finally:
                _add('prefill_chunk_s', time.time() - t0, pf)
                tm['chunks'] = tm.get('chunks', 0) + 1
                pf.chunks_run = getattr(pf, 'chunks_run', 0) + 1
            self.prefill = None
            t0 = time.time()
            self.eng.to_decode(pf, min(set(range(self.eng.slots)) - {r.slot for r in self.decoding}))
            tm['first_token_s'], tm['ttft_at'] = time.time() - t0, time.time()
            if 'arrive' in tm:
                M['last_ttft_s'] = round(tm['ttft_at'] - tm['arrive'], 4)
            M['last_restore_s'] = round(tm.get('restore_s', 0), 4)
        except NoRoom:  # an OOM restart couldn't get its pages back: wait in line again
            self.prefill = None
            self.eng.finish(pf)
            return self.submit(pf)
        except Exception as e:
            return self._fail(pf, e)
        finally:
            _cur = None
        if getattr(pf, 'err', None) is not None:
            self._fail(pf, pf.err)
        elif pf.done:
            self._end(pf)
        else:
            self.decoding.append(pf)

    def _decode(self, dec):
        t0 = time.time()
        try:
            self.eng.decode_step(dec)
        except Exception as e:  # (OutOfMemory: the whole batch fails with 503)
            for r in dec:
                self._fail(r, e)
            return
        dt = time.time() - t0
        for r in dec:  # a slow step = graph capture after a decoder re-alloc, a spill, ...
            tm = _tm(r)
            tm['step_max_s'] = max(tm.get('step_max_s', 0.0), dt)
        M['decode_steps'] += 1
        M['decode_rows'] += len(dec)
        for r in dec:
            if getattr(r, 'err', None) is not None:  # failed alone in the step (engine contained it)
                self._fail(r, r.err)
            elif r.done:
                self._end(r)

    # ------------------------------------------------------------------ ends
    def _fail(self, r, e):
        if isinstance(e, BadRequest):
            self._end(r, 'bad', str(e))
        elif isinstance(e, OutOfMemory):  # pages already released by the engine; the client may retry later
            self._end(r, 'oom', str(e))
        else:
            import traceback; traceback.print_exc()
            self._end(r, 'error', f'{type(e).__name__}: {e}')

    def _end(self, r, kind='done', msg=None):
        """r leaves: slot + pages freed, result to the client (kind None: client gone), metrics."""
        if r is self.prefill:
            self.prefill = None
        elif r in self.decoding:
            self.decoding.remove(r)
        with self.cv:
            self.heap = [it for it in self.heap if it[2] is not r]
            heapq.heapify(self.heap)
            self.retry = 0.0
        self.eng.finish(r)
        if kind is not None:
            r.emit(kind, msg)
        st = r.stats
        if st.get('prompt_tokens') is not None:
            M['requests'] += 1
            for k in ('prompt_tokens', 'cached_tokens', 'completion_tokens', 'prefill_s', 'decode_s'):
                M[k] += st.get(k, 0)
            if st.get('prefill_s'):
                M['last_prefill_tps'] = round((st['prompt_tokens'] - st.get('cached_tokens', 0)) / st['prefill_s'], 1)
            if st.get('decode_s'):  # per request (wall time from its first token, shared with the batch)
                M['last_decode_tps'] = round(st.get('completion_tokens', 0) / st['decode_s'], 2)


sched = None


def priority_of(body):
    p = body.get('priority', (body.get('extra_body') or {}).get('priority', 0))  # LiteLLM: extra_body.priority
    return int(p or 0)


def effort_of(body):
    kw = body.get('chat_template_kwargs') or {}
    if kw.get('enable_thinking') is False or body.get('reasoning_effort') == 'none':
        return 0, True                     # thinking off: pre-fill a text message (no reasoning block)
    e = body.get('reasoning_effort', kw.get('reasoning_effort', 0.9))
    if isinstance(e, str) and e not in chat.EFFORT:
        raise BadRequest(f'unknown reasoning_effort {e!r}')
    return e, False


@app.get('/health')
def health():
    return {'ok': eng is not None}


@app.get('/metrics')
def metrics():
    from fastapi.responses import PlainTextResponse
    g = dict(M, queue_depth=sched.depth() if sched else 0, running=sched.running() if sched else 0,
             batch_size=len(sched.decoding) if sched else 0, slots=getattr(eng, 'slots', 0))
    if getattr(eng, 'pool', None) is not None:
        g.update(kv_free_pages=eng.pool.free_pages(), parked=len(eng.parked), parked_bytes=eng.parked_bytes(),
                 ooms=eng.ooms, slot_decoders=sum(d is not None for d in getattr(getattr(eng, 'bd', None), 'decs', ())),
                 vram_allocated_bytes=torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
                 batch_decoder=int(getattr(eng, 'batched', False)), vision_images=getattr(eng, 'vision_images', 0))
    lines = []
    for k, v in g.items():
        kind = 'gauge' if k.startswith('last_') or k in ('queue_depth', 'running', 'batch_size', 'slots',
                                                         'kv_free_pages', 'parked', 'parked_bytes', 'slot_decoders',
                                                         'vram_allocated_bytes', 'warmup_seconds', 'batch_decoder') else 'counter'
        name = f'inkling_{k}' + ('_total' if kind == 'counter' else '')
        lines += [f'# TYPE {name} {kind}', f'{name} {v}']
    return PlainTextResponse('\n'.join(lines) + '\n')


def model_card():
    return {'id': SERVED, 'object': 'model', 'created': 0, 'owned_by': 'local', 'max_model_len': eng.max_ctx if eng else None}


@app.get('/v1/models')
def models():
    return {'object': 'list', 'data': [model_card()]}


@app.get('/v1/models/{model_id:path}')
def model(model_id: str):
    return model_card() if model_id == SERVED else JSONResponse(
        {'error': {'message': f'model {model_id!r} not found', 'type': 'invalid_request_error', 'code': 'model_not_found'}}, 404)


def err(code, msg):
    return JSONResponse({'error': {'message': msg, 'type': 'invalid_request_error', 'code': code}}, code)


# Known OpenAI / SGLang parameters we don't implement: 400 instead of silently ignoring them. Unknown fields and
# metadata-only ones (user, store, metadata, LiteLLM's own) are ignored, as SGLang does.
UNSUPPORTED = ('ignore_eos', 'regex', 'ebnf', 'audio', 'web_search_options', 'logit_processor', 'echo', 'best_of',
               'suffix')  # (echo / best_of: /v1/completions; suffix: insertion, which the model wasn't trained for)
MAX_N = 8
DEFAULT_MAX_TOKENS = 4096
tool_name = lambda t: (t.get('function') or t.get('custom') or t).get('name') if isinstance(t, dict) else None


def chat_tools(body):
    """tools + tool_choice -> (tools as rendered, tool_choice for grammar.compile_tools, custom tool names).
    Custom (free-form text) tools become functions taking one string `input`, the call format the model knows: a regex
    format is enforced through the string's pattern, a Lark grammar only goes into the description (shortcut: Lark is
    not compiled to a constraint; compile it if a client depends on it). allowed_tools renders only the allowed tools."""
    choice = 'auto' if body.get('tool_choice') is None else body['tool_choice']
    tools = body.get('tools') or []
    if choice == 'none':  # no tools rendered
        return None, 'none', set()
    if not isinstance(tools, list):
        raise BadRequest('tools must be a list')
    if isinstance(choice, dict) and choice.get('type') == 'allowed_tools':
        a = choice.get('allowed_tools') or {}
        allow = {tool_name(t) for t in a.get('tools') or []}
        tools, choice = [t for t in tools if tool_name(t) in allow], a.get('mode', 'auto')
        if choice not in ('auto', 'required'):
            raise BadRequest("allowed_tools mode must be 'auto' or 'required'")
    if isinstance(choice, dict) and choice.get('type') == 'custom':
        choice = {'type': 'function', 'function': {'name': (choice.get('custom') or {}).get('name')}}
    out, custom = [], set()
    for t in tools:
        if not isinstance(t, dict) or t.get('type', 'function') not in ('function', 'custom'):
            raise BadRequest(f'unsupported tool type {t.get("type") if isinstance(t, dict) else t!r}')
        if t.get('type') != 'custom':
            out.append(t)
            continue
        c = t.get('custom') or {}
        if not isinstance(c.get('name'), str) or not c['name']:
            raise BadRequest('custom tool needs a name')
        fmt, desc = c.get('format') or {'type': 'text'}, c.get('description') or ''
        s = {'type': 'string'}
        if fmt.get('type') == 'grammar' and fmt.get('syntax') == 'regex' and isinstance(fmt.get('definition'), str):
            s['pattern'] = fmt['definition']
        elif fmt.get('type') == 'grammar' and fmt.get('syntax') == 'lark' and isinstance(fmt.get('definition'), str):
            desc += f'\n\nThe input must match this Lark grammar:\n{fmt["definition"]}'
        elif fmt.get('type') != 'text':
            raise BadRequest("custom tool format must be text or a grammar with syntax 'regex' or 'lark'")
        custom.add(c['name'])
        out.append({'type': 'function', 'function': {'name': c['name'], 'description': desc, 'parameters': {
            'type': 'object', 'properties': {'input': s}, 'required': ['input']}}})
    return out or None, choice, custom


def custom_input(tc):
    """A parsed call of a custom tool -> its raw input (the `input` string chat_tools declared)."""
    try:
        v = json.loads(tc['arguments']).get('input', '')
    except (ValueError, AttributeError):  # (a call cut short by max_tokens)
        return ''
    return v if isinstance(v, str) else json.dumps(v)


class CustomInput:
    """A custom tool's raw input, streamed out of its args text {"input":"..."} as that arrives: the JSON string is
    unescaped piece by piece, cut only at safe points (never inside an escape or a surrogate pair), so the pieces add up
    to the decoded string. Args written another way stream nothing (finish() gives it all)."""
    PRE = '{"input":"'

    def __init__(self):
        self.raw, self.pos, self.out, self.closed = '', len(self.PRE), 0, False  # pos: raw index decoded up to

    def feed(self, d):
        self.raw += d
        r, i = self.raw, self.pos
        if self.closed or not r.startswith(self.PRE):
            return ''
        while i < len(r):
            c = r[i]
            if c == '"':
                self.closed = True
                break
            if c != '\\':
                i += 1
            elif i + 1 >= len(r):
                break
            elif r[i + 1] != 'u':
                i += 2
            elif i + 6 > len(r) or (r[i + 2] in 'dD' and r[i + 3] in '89abAB' and i + 12 > len(r)):
                break  # \uXXXX not complete yet, or a high surrogate waiting for its pair
            else:
                i += 6
        try:
            s = json.loads('"' + r[self.pos:i] + '"')
        except ValueError:
            self.closed = True  # (not a JSON string after all: the rest comes with finish)
            return ''
        self.pos, self.out = i, self.out + len(s)
        return s

    def finish(self, final):
        """The rest of the final input after what feed emitted."""
        return final[self.out:]


SYSTEM_FP = None


def system_fingerprint():
    """fp_ + the weights id (changes when the weights do), as OpenAI's per-backend-config fingerprint."""
    global SYSTEM_FP
    if SYSTEM_FP is None:
        wid = getattr(getattr(eng, 'pool', None), 'wid', None)
        SYSTEM_FP = 'fp_' + (wid[:12] if isinstance(wid, str) else 'inkgine')
    return SYSTEM_FP


def legacy_functions(body):
    """The deprecated functions / function_call -> tools / tool_choice (one call per reply, as that API had); role
    'function' messages -> tool messages and assistant function_call -> a tool call (ids f'fn{n}' link them).
    -> (body, legacy?)."""
    fns, fc = body.get('functions'), body.get('function_call')
    msgs = body.get('messages') or []
    if not fns and fc is None and not any(isinstance(m, dict) and (m.get('role') == 'function' or m.get('function_call'))
                                         for m in msgs):
        return body, False
    if body.get('tools') and fns:
        raise BadRequest('use tools or functions, not both')
    out, n = [], 0
    for m in msgs:
        if isinstance(m, dict) and m.get('function_call'):
            n += 1
            m = dict({k: v for k, v in m.items() if k != 'function_call'}, tool_calls=[
                {'id': f'fn{n}', 'type': 'function', 'function': m['function_call']}])
        elif isinstance(m, dict) and m.get('role') == 'function':
            m = {'role': 'tool', 'tool_call_id': f'fn{n}', 'name': m.get('name'), 'content': m.get('content') or ''}
        out.append(m)
    body = {k: v for k, v in body.items() if k not in ('functions', 'function_call')}
    body['messages'] = out
    if fns:
        body.update(tools=[{'type': 'function', 'function': f} for f in fns], parallel_tool_calls=False)
        if fc is not None:
            body['tool_choice'] = fc if isinstance(fc, str) else {'type': 'function', 'function': {'name': fc.get('name')}}
    return body, bool(fns)


def chat_messages(msgs):
    """Assistant custom tool calls -> function calls with arguments {"input": ...} (as chat_tools declares them)."""
    fix = lambda c: {'id': c.get('id'), 'type': 'function', 'function': {
        'name': (c.get('custom') or {}).get('name'), 'arguments': json.dumps({'input': (c.get('custom') or {}).get('input', '')})}}
    return [dict(m, tool_calls=[fix(c) if isinstance(c, dict) and c.get('type') == 'custom' else c for c in m['tool_calls']])
            if isinstance(m, dict) and isinstance(m.get('tool_calls'), list) else m for m in msgs]


def parse(body):
    """Request body -> (n, engine kwargs per request without stats/emit/cancel, logprobs on, custom tool names).
    BadRequest / ValueError / TypeError on a bad request. Compiles grammars, so it runs off the event loop."""
    g = lambda k, d=None: d if body.get(k) is None else body[k]  # null == absent (clients send nulls)
    bad = [k for k in UNSUPPORTED if body.get(k) not in (None, False, [], {})]
    if bad or (body.get('modalities') or ['text']) != ['text']:
        raise BadRequest(f'unsupported parameter(s): {", ".join(bad) or "modalities"}')
    effort, no_think = effort_of(body)
    n = int(g('n', 1))
    if not 1 <= n <= MAX_N:
        raise BadRequest(f'n must be 1..{MAX_N}')
    lp, top = g('logprobs', False), g('top_logprobs')
    if not isinstance(lp, bool) or (top is not None and not lp):
        raise BadRequest('logprobs must be true or false; top_logprobs needs logprobs: true')
    top = int(top or 0)
    if not 0 <= top <= 20:
        raise BadRequest('top_logprobs must be 0..20')
    stop = g('stop', [])
    stop = [stop] if isinstance(stop, str) else stop
    if not isinstance(stop, list) or len(stop) > 4 or not all(isinstance(x, str) for x in stop):
        raise BadRequest('stop must be a string or a list of up to 4 strings')
    sids = g('stop_token_ids', [])
    if not isinstance(sids, list) or not all(isinstance(x, int) and 0 <= x < grammar.VOCAB for x in sids):
        raise BadRequest('stop_token_ids must be a list of token ids')
    tools, choice, custom = chat_tools(body)
    parallel = g('parallel_tool_calls', True)
    if not isinstance(parallel, bool):
        raise BadRequest('parallel_tool_calls must be true or false')
    spec = body.get('spec', (body.get('extra_body') or {}).get('spec'))  # false: no speculative decoding
    if spec is not None and not isinstance(spec, bool):
        raise BadRequest('spec must be true or false')
    force = grammar.compile_tools(eng.tok, tools, choice, think=not no_think, parallel=parallel) if tools else None
    if choice not in (None, 'auto', 'none') and not tools:
        raise BadRequest('tool_choice needs tools')
    rf = g('response_format')
    rf = grammar.compile_format(eng.tok, rf) if rf and rf.get('type', 'text') != 'text' else None
    pres, freq, rep = float(g('presence_penalty', 0)), float(g('frequency_penalty', 0)), float(g('repetition_penalty', 1))
    min_p, min_tokens = float(g('min_p', 0)), int(g('min_tokens', 0))
    temperature, top_p = float(g('temperature', 1.0)), float(g('top_p', 1.0))
    if not (-2 <= pres <= 2 and -2 <= freq <= 2 and 0 < rep <= 2 and 0 <= min_p <= 1 and min_tokens >= 0
            and temperature >= 0 and 0 < top_p <= 1):
        raise BadRequest('presence/frequency_penalty -2..2, repetition_penalty (0, 2], min_p 0..1, min_tokens >= 0, '
                         'temperature >= 0, top_p (0, 1]')
    if force is not None and force.lazy and (rf is not None or min_tokens):
        force = None  # tool_choice auto: response_format holds the reply to its JSON (no call); min_tokens as before
    if force is not None and rf is not None:
        raise BadRequest('response_format together with a forced tool_choice is not supported')
    if min_tokens and (force is not None or rf is not None):
        raise BadRequest('min_tokens together with response_format / a forced tool_choice is not supported')
    bias = g('logit_bias', {})
    if not isinstance(bias, dict):
        raise BadRequest('logit_bias must be an object {token id: bias}')
    bias = {int(k): float(v) for k, v in bias.items()}
    if not all(0 <= k < grammar.VOCAB and -100 <= v <= 100 for k, v in bias.items()):
        raise BadRequest('logit_bias: token ids 0..vocab, biases -100..100')
    seed = g('seed')
    if seed is not None and not -2 ** 63 <= int(seed) < 2 ** 64:
        raise BadRequest('seed must be a 64-bit integer')
    pred = g('prediction')  # predicted output: copy-draft source for speculation (same output, faster where it matches)
    if pred is not None:
        pc = pred.get('content') if isinstance(pred, dict) and pred.get('type') == 'content' else None
        pc = pc if isinstance(pc, str) else ''.join(p.get('text', '') for p in pc if isinstance(p, dict)
                                                    and p.get('type') == 'text') if isinstance(pc, list) else None
        if pc is None:
            raise BadRequest('prediction must be {"type": "content", "content": a string or text parts}')
        pred = eng.tok.encode(pc, add_special_tokens=False).ids
    mt = g('max_completion_tokens') if g('max_completion_tokens') is not None else g('max_tokens')
    if mt is not None and int(mt) < 1:
        raise BadRequest('max_tokens must be >= 1')
    kw = dict(max_tokens=int(mt or DEFAULT_MAX_TOKENS), temperature=temperature,
              top_p=top_p, top_k=max(int(g('top_k', 0)), 0), effort=effort, tools=tools,
              seed=None if seed is None else int(seed), prefill_text=no_think and not tools,  # tools may follow text
              response_format=rf, stop=[x for x in stop if x] or None, spec=spec, logprobs=top if lp else None,
              presence_penalty=pres, frequency_penalty=freq, repetition_penalty=rep, logit_bias=bias or None,
              min_p=min_p, min_tokens=min_tokens, stop_ids=sids or None, tool_force=force,
              one_call=bool(tools) and not parallel, prediction=pred or None)
    return n, kw, lp, custom


async def submit(body, t_arr, prompts=None):
    """Chat-format body -> SimpleNamespace(reqs, stats, events, cancel, lp, custom) for the endpoints to format, or an
    error response. events() submits the requests and yields (choice, kind, value); leaving it early cancels them.
    prompts: raw prompts (lists of ids, /v1/completions) instead of body['messages']: n requests per prompt, plain text
    replies, choice index = prompt * n + i."""
    try:
        prio = priority_of(body)
    except (TypeError, ValueError) as e:
        return err(400, f'bad parameter: {e}')
    q, cancel = asyncio.Queue(), threading.Event()  # one cancel for all n choices (the client is gone for all)
    loop = asyncio.get_running_loop()
    box = Outbox.of(loop)

    def build():
        """n requests sharing the prompt (the second and later restore the first's parked prompt state, so their
        prefill is ~free; prompts < INK_SNAP_MIN tokens aren't parked and are simply re-prefilled). Choice i's seed is
        seed + i, so seeded choices differ but are reproducible."""
        n, kw, lp, custom = parse(body)
        if prompts is None:
            msgs, todo = chat_messages(body['messages']), [None]
        else:
            msgs, todo = None, prompts
            kw.update(prefill_text=True, tools=None, tool_force=None, one_call=False)
        reqs = []
        for p in todo:
            for _ in range(n):
                i = len(reqs)
                k = dict(kw, seed=None if kw['seed'] is None else kw['seed'] + i)
                emit = lambda *m, i=i: box.put(q, (i,) + m)  # from the scheduler thread
                reqs.append(eng.request(msgs, stats={}, emit=emit, prio=prio, cancel=cancel, prompt_ids=p, **k))
        return reqs, lp, custom
    try:  # parsing, grammar compiles, tokenize + image preprocessing: off the event loop and the scheduler thread
        reqs, want_lp, custom = await loop.run_in_executor(None, build)
    except BadRequest as e:
        return err(400, str(e))
    except (TypeError, ValueError) as e:
        return err(400, f'bad parameter: {e}')
    stats = [r.stats for r in reqs]
    tok_s = time.time() - t_arr  # body parse + render/tokenize + images + grammar compiles
    M['tokenize_s'] += tok_s
    for s in stats:
        s.setdefault('tm', {}).update(arrive=t_arr, tokenize_s=tok_s)

    async def events():
        for r in reqs:
            sched.submit(r)
        left = len(reqs)
        try:
            while left:
                i, kind, val = await q.get()
                yield i, kind, val
                if kind != 'delta':
                    left -= 1
                    if kind != 'done':
                        break
        finally:
            cancel.set()  # queued: skipped; running: ends at the next step (slot + pages freed)
    return SimpleNamespace(reqs=reqs, stats=stats, events=events, cancel=cancel, lp=want_lp, custom=custom)


async def watch(request, cancel):
    """Non-stream requests: a client that hangs up frees its slot (only streams get cancelled by Starlette)."""
    while not cancel.is_set():
        if await request.is_disconnected():
            cancel.set()
            return
        await asyncio.sleep(1.0)


@app.post('/v1/chat/completions')
async def completions(request: Request):
    t_arr = time.time()
    try:
        body = await request.json()
    except ValueError:
        return err(400, 'body is not JSON')
    if not isinstance(body, dict) or not isinstance(body.get('messages'), list) or not body['messages']:
        return err(400, 'messages must be a non-empty list')
    rid, created = 'chatcmpl-' + uuid.uuid4().hex[:24], int(time.time())
    try:
        body, legacy = legacy_functions(body)  # (legacy: reply with message.function_call, the old API's shape)
    except BadRequest as e:
        return err(400, str(e))
    run = await submit(body, t_arr)
    if isinstance(run, JSONResponse):
        return run
    reqs, stats, events, cancel, want_lp = run.reqs, run.stats, run.events, run.cancel, run.lp
    want_tm = body.get('timings') is True  # non-standard: latency breakdown in the response (benchmarks)
    head = {'id': rid, 'created': created, 'model': body.get('model', SERVED), 'system_fingerprint': system_fingerprint(),
            'service_tier': body.get('service_tier') or 'default'}

    def usage():  # the prompt is billed once (OpenAI), completions summed over the choices
        p, c = stats[0].get('prompt_tokens', 0), sum(s.get('completion_tokens', 0) for s in stats)
        det = {'reasoning_tokens': sum(s.get('reasoning_tokens', 0) for s in stats)}
        if body.get('prediction'):
            det.update({k: sum(s.get(k, 0) for s in stats) for k in ('accepted_prediction_tokens', 'rejected_prediction_tokens')})
        return {'prompt_tokens': p, 'completion_tokens': c, 'total_tokens': p + c,
                'prompt_tokens_details': {'cached_tokens': stats[0].get('cached_tokens', 0)}, 'completion_tokens_details': det}

    def call_entry(tc, cid=None):
        cid = cid or f'call_{uuid.uuid4().hex[:24]}'
        if tc['name'] in run.custom:
            return {'id': cid, 'type': 'custom', 'custom': {'name': tc['name'], 'input': custom_input(tc)}}
        return {'id': cid, 'type': 'function', 'function': tc}

    def finish(i):
        if stats[i].get('tool_calls'):
            return 'function_call' if legacy else 'tool_calls'
        return stats[i].get('finish_reason', 'stop')

    if not body.get('stream'):
        watcher = asyncio.create_task(watch(request, cancel))
        text, reasoning, lps = ([[] for _ in reqs] for _ in range(3))
        try:
            async for i, kind, val in events():
                if kind == 'delta' and val[0] in ('reasoning', 'content'):  # (calls: from the stats at the end)
                    (reasoning if val[0] == 'reasoning' else text)[i].append(val[1])
                    lps[i] += val[2] if len(val) > 2 else []
                elif kind in ('bad', 'error', 'oom'):
                    return err({'bad': 400, 'oom': 503}.get(kind, 500), val)
        finally:
            watcher.cancel()
        choices = []
        for i in range(len(reqs)):
            msg = {'role': 'assistant', 'content': ''.join(text[i]), 'reasoning_content': ''.join(reasoning[i]) or None,
                   'refusal': None}
            calls = stats[i].get('tool_calls') or []
            if calls and legacy:
                msg['function_call'] = calls[0]
            elif calls:
                msg['tool_calls'] = [call_entry(tc) for tc in calls]
            choices.append({'index': i, 'message': msg, 'logprobs': {'content': lps[i]} if want_lp else None,
                            'finish_reason': finish(i)})
        out = dict(head, object='chat.completion', choices=choices, usage=usage())
        if want_tm:
            out['timings'] = [timings(s) for s in stats]
        return out

    include_usage = (body.get('stream_options') or {}).get('include_usage')

    def chunk(i, delta, fin=None, lp=None):
        c = {'index': i, 'delta': delta, 'finish_reason': fin}
        if want_lp:
            c['logprobs'] = {'content': lp} if lp else None
        return 'data: ' + json.dumps(dict(head, object='chat.completion.chunk', choices=[c])) + '\n\n'

    open_call, n_calls = [None] * len(reqs), [0] * len(reqs)  # per choice: the streaming call, calls so far

    def call_delta(i, kind, val):
        """A call_* delta of choice i -> the chunk's delta dict (None: nothing to send yet)."""
        c = open_call[i]
        if kind == 'call_start':
            c = open_call[i] = {'id': f'call_{uuid.uuid4().hex[:24]}', 'custom': val in run.custom, 'dec': CustomInput()}
            body_ = {'name': val, 'input' if c['custom'] else 'arguments': ''}
            if legacy:
                return {'function_call': body_}
            return {'tool_calls': [{'index': n_calls[i], 'id': c['id'], 'type': 'custom' if c['custom'] else 'function',
                                    'custom' if c['custom'] else 'function': body_}]}
        if kind == 'call_args':
            d = c['dec'].feed(val) if c['custom'] else val
        else:  # call_end
            open_call[i], n_calls[i] = None, n_calls[i] + 1
            if c is None:  # parsed at its end: the whole call at once
                return {'function_call': val} if legacy else {'tool_calls': [dict(call_entry(val), index=n_calls[i] - 1)]}
            d = c['dec'].finish(custom_input(val)) if c['custom'] else ''
            if not d:
                return None
        if legacy:
            return {'function_call': {'arguments': d}}
        return {'tool_calls': [{'index': n_calls[i] - (kind == 'call_end'),
                                'custom' if c['custom'] else 'function': {'input' if c['custom'] else 'arguments': d}}]}

    async def sse():
        for i in range(len(reqs)):
            yield chunk(i, {'role': 'assistant', 'content': '', 'refusal': None})
        async for i, kind, val in events():
            if kind == 'delta' and val[0] in ('reasoning', 'content'):
                yield chunk(i, {'reasoning_content' if val[0] == 'reasoning' else 'content': val[1]},
                            lp=val[2] if len(val) > 2 else None)
            elif kind == 'delta':
                if (d := call_delta(i, val[0], val[1])) is not None:
                    yield chunk(i, d)
            elif kind in ('bad', 'error', 'oom'):
                yield 'data: ' + json.dumps({'error': {'message': val, 'code': {'bad': 400, 'oom': 503}.get(kind, 500)}}) + '\n\n'
                break
            else:
                yield chunk(i, {}, finish(i))
        else:
            if include_usage:
                yield 'data: ' + json.dumps(dict(head, object='chat.completion.chunk', choices=[], usage=usage(),
                                                 **({'timings': [timings(s) for s in stats]} if want_tm else {}))) + '\n\n'
        yield 'data: [DONE]\n\n'

    return StreamingResponse(sse(), media_type='text/event-stream')


# Legacy /v1/completions: raw prompts, no chat template (the model is chat-tuned: a raw prompt continues as plain text
# to its end-of-message). Sampling parameters as chat. suffix (insertion) and prompt logprobs (echo + logprobs) aren't
# supported: the model has no insertion training, and prefill keeps no per-position logits.
COMPL_PASS = ('model', 'stream', 'stream_options', 'temperature', 'top_p', 'seed', 'stop', 'presence_penalty',
              'frequency_penalty', 'logit_bias', 'user', 'priority', 'extra_body', 'spec', 'top_k', 'min_p',
              'repetition_penalty', 'min_tokens', 'stop_token_ids')
MAX_PROMPTS = 16


def compl_prompts(p):
    """prompt (a string, strings, token ids, lists of token ids) -> [(ids, text)]."""
    tok_ok = lambda x: isinstance(x, list) and x and all(isinstance(t, int) and 0 <= t < grammar.VOCAB and
                                                         t != chat.IMG_TOK for t in x)
    if isinstance(p, str) or tok_ok(p):
        p = [p]
    if not isinstance(p, list) or not p or len(p) > MAX_PROMPTS:
        raise BadRequest(f'prompt must be a string, a list of up to {MAX_PROMPTS} strings, or token id list(s)')
    out = []
    for x in p:
        if isinstance(x, str) and x:
            out.append((eng.tok.encode(x, add_special_tokens=False).ids, x))
        elif tok_ok(x):
            out.append((x, eng.tok.decode(x, skip_special_tokens=False)))
        else:
            raise BadRequest('each prompt must be a non-empty string or a list of valid token ids')
    return out


def compl_logprobs(ents, offset):
    """Chat logprob entries -> the completions format (tokens, token_logprobs, top_logprobs, text_offset)."""
    offs = []
    for e in ents:
        offs.append(offset)
        offset += len(e['token'])
    return {'tokens': [e['token'] for e in ents], 'token_logprobs': [e['logprob'] for e in ents],
            'top_logprobs': [{t['token']: t['logprob'] for t in e.get('top_logprobs') or []} for e in ents],
            'text_offset': offs}


@app.post('/v1/completions')
async def text_completions(request: Request):
    t_arr = time.time()
    try:
        body = await request.json()
    except ValueError:
        return err(400, 'body is not JSON')
    if not isinstance(body, dict):
        return err(400, 'body must be an object')
    try:
        prompts = compl_prompts(body.get('prompt'))
        n = int(body.get('n') or 1)
        best = int(body.get('best_of') or n)
        lpn, echo, stream = body.get('logprobs'), bool(body.get('echo')), bool(body.get('stream'))
        if body.get('suffix'):
            raise BadRequest('suffix (insertion) is not supported')
        if not 1 <= n <= best <= MAX_N:
            raise BadRequest(f'need 1 <= n <= best_of <= {MAX_N}')
        if stream and best > n:
            raise BadRequest('best_of > n cannot stream')
        if lpn is not None and (isinstance(lpn, bool) or not isinstance(lpn, int) or not 0 <= lpn <= 20):
            raise BadRequest('logprobs must be an integer 0..20')
        if echo and lpn is not None:
            raise BadRequest('echo with logprobs (prompt logprobs) is not supported')
    except (BadRequest, TypeError, ValueError) as e:
        return err(400, str(e))
    cb = {k: body[k] for k in COMPL_PASS if k in body}
    cb.update(messages=[], n=best, max_tokens=body.get('max_tokens') or 16)  # (OpenAI's default for this endpoint)
    if lpn is not None or best > n:  # (best_of ranks candidates by their mean token logprob)
        cb.update(logprobs=True, top_logprobs=lpn or 0)
    rid, created = 'cmpl-' + uuid.uuid4().hex[:24], int(time.time())
    run = await submit(cb, t_arr, prompts=[ids for ids, _ in prompts])
    if isinstance(run, JSONResponse):
        return run
    stats = run.stats
    head = {'id': rid, 'object': 'text_completion', 'created': created, 'model': body.get('model', SERVED),
            'system_fingerprint': system_fingerprint()}
    pre = lambda i: prompts[i // best][1] if echo else ''

    def usage():  # every prompt and every candidate is billed (best_of included)
        p = sum(stats[k * best].get('prompt_tokens', 0) for k in range(len(prompts)))
        c = sum(s.get('completion_tokens', 0) for s in stats)
        return {'prompt_tokens': p, 'completion_tokens': c, 'total_tokens': p + c}

    if not stream:
        watcher = asyncio.create_task(watch(request, run.cancel))
        text, lps = ([[] for _ in run.reqs] for _ in range(2))
        try:
            async for i, kind, val in run.events():
                if kind == 'delta' and val[0] == 'content':
                    text[i].append(val[1])
                    lps[i] += val[2] if len(val) > 2 else []
                elif kind in ('bad', 'error', 'oom'):
                    return err({'bad': 400, 'oom': 503}.get(kind, 500), val)
        finally:
            watcher.cancel()
        choices = []
        for k in range(len(prompts)):
            cand = list(range(k * best, (k + 1) * best))
            if best > n:
                cand = sorted(cand, key=lambda i: -sum(e['logprob'] for e in lps[i]) / max(len(lps[i]), 1))[:n]
            for i in cand:
                choices.append({'text': pre(i) + ''.join(text[i]), 'index': len(choices),
                                'logprobs': compl_logprobs(lps[i], len(pre(i))) if lpn is not None else None,
                                'finish_reason': stats[i].get('finish_reason', 'stop')})
        return dict(head, choices=choices, usage=usage())

    offs = [len(pre(i)) for i in range(len(run.reqs))]

    def chunk(i, text, lp=None, fin=None):
        c = {'text': text, 'index': i, 'logprobs': None, 'finish_reason': fin}
        if lpn is not None and lp:
            c['logprobs'] = compl_logprobs(lp, offs[i])
            offs[i] += sum(len(e['token']) for e in lp)
        return 'data: ' + json.dumps(dict(head, choices=[c])) + '\n\n'

    async def sse():
        for i in range(len(run.reqs)):
            if echo:
                yield chunk(i, pre(i))
        async for i, kind, val in run.events():
            if kind == 'delta' and val[0] == 'content':
                yield chunk(i, val[1], val[2] if len(val) > 2 else None)
            elif kind in ('bad', 'error', 'oom'):
                yield 'data: ' + json.dumps({'error': {'message': val, 'code': {'bad': 400, 'oom': 503}.get(kind, 500)}}) + '\n\n'
                break
            elif kind != 'delta':
                yield chunk(i, '', fin=stats[i].get('finish_reason', 'stop'))
        else:
            if (body.get('stream_options') or {}).get('include_usage'):
                yield 'data: ' + json.dumps(dict(head, choices=[], usage=usage())) + '\n\n'
        yield 'data: [DONE]\n\n'

    return StreamingResponse(sse(), media_type='text/event-stream')


# OpenAI Responses API, stateless: each request carries the whole conversation in `input` (LiteLLM's Anthropic
# /v1/messages bridge works this way). Stored responses (previous_response_id, conversation, background, GET by id)
# are rejected; hosted tools are dropped. The request is mapped to a chat body and runs the chat path (function and
# custom tools, allowed_tools, json_schema, logprobs); truncation 'auto' drops the oldest turns; /input_tokens counts.
RESP_PASS = ('model', 'stream', 'temperature', 'top_p', 'parallel_tool_calls', 'seed', 'priority', 'extra_body', 'spec',
             'top_k', 'min_p', 'presence_penalty', 'frequency_penalty', 'repetition_penalty', 'logit_bias', 'stop',
             'chat_template_kwargs')  # (the last ones are non-standard here, as in our chat API)
RESP_STATEFUL = ('previous_response_id', 'conversation', 'background', 'prompt')
TEXT_PARTS = ('input_text', 'output_text', 'text', 'summary_text', 'reasoning_text')


def resp_parts(c):
    """Responses content (string or parts) -> chat content."""
    if isinstance(c, str):
        return c
    if not isinstance(c, list):
        raise BadRequest('content must be a string or a list of parts')
    out = []
    for p in c:
        t = p.get('type') if isinstance(p, dict) else None
        if t in TEXT_PARTS or t == 'refusal':
            out.append({'type': 'text', 'text': p.get('text', p.get('refusal')) or ''})
        elif t == 'input_image':
            if not p.get('image_url'):
                raise BadRequest('input_image needs image_url (file_id is not supported)')
            out.append({'type': 'image_url', 'image_url': p['image_url']})
        else:
            raise BadRequest(f'unsupported content part type {t!r}')
    return out


def resp_to_chat(body):
    """Responses request body -> chat completions body (BadRequest if it can't be mapped)."""
    bad = [k for k in RESP_STATEFUL if body.get(k) not in (None, False)]
    if bad:
        raise BadRequest(f'unsupported parameter(s): {", ".join(bad)} (stateless: send the whole conversation as input)')
    if body.get('truncation') not in (None, 'disabled', 'auto'):
        raise BadRequest("truncation must be 'auto' or 'disabled'")
    msgs =[{'role': 'system', 'content': body['instructions']}] if body.get('instructions') else []
    inp = body.get('input')
    inp = [{'type': 'message', 'role': 'user', 'content': inp}] if isinstance(inp, str) else inp
    if not isinstance(inp, list) or not inp:
        raise BadRequest('input must be a string or a non-empty list of items')
    cur = None  # the assistant turn being assembled: reasoning, text, then calls (chat.render's order)

    def turn(stage):  # a later item of an earlier stage starts a new assistant turn
        nonlocal cur
        if cur is None or cur['stage'] > stage:
            cur = {'role': 'assistant', 'content': '', 'stage': stage}
            msgs.append(cur)
        cur['stage'] = stage
        return cur
    for it in inp:
        if not isinstance(it, dict):
            raise BadRequest('input items must be objects')
        t = it.get('type', 'message')
        if t == 'reasoning':  # the full text: our encrypted_content, else the raw content, else the summary
            text = dec_reasoning(it.get('encrypted_content'))  # (another model's encrypted_content: dropped)
            if text is None:
                text = ''.join(p.get('text', '') for p in (it.get('content') or it.get('summary') or [])
                               if isinstance(p, dict) and p.get('type') in TEXT_PARTS)
            if text:
                m = turn(0)
                m['reasoning_content'] = m.get('reasoning_content', '') + text
        elif t == 'message' and it.get('role') == 'assistant':
            c = resp_parts(it.get('content'))
            turn(1)['content'] += c if isinstance(c, str) else ''.join(p.get('text', '') for p in c)
        elif t == 'function_call':
            if not isinstance(it.get('name'), str) or not it['name']:
                raise BadRequest('function_call needs a name')
            turn(2).setdefault('tool_calls', []).append(
                {'id': it.get('call_id'), 'type': 'function',
                 'function': {'name': it['name'], 'arguments': it.get('arguments') or '{}'}})
        elif t == 'custom_tool_call':
            if not isinstance(it.get('name'), str) or not it['name']:
                raise BadRequest('custom_tool_call needs a name')
            turn(2).setdefault('tool_calls', []).append(
                {'id': it.get('call_id'), 'type': 'custom', 'custom': {'name': it['name'], 'input': it.get('input', '')}})
        elif t in ('function_call_output', 'custom_tool_call_output'):
            cur = None
            msgs.append({'role': 'tool', 'tool_call_id': it.get('call_id'), 'content': resp_parts(it.get('output', ''))})
        elif t == 'message':
            if it.get('role') not in ('user', 'system', 'developer'):
                raise BadRequest(f'unknown message role {it.get("role")!r}')
            cur = None
            msgs.append({'role': it['role'], 'content': resp_parts(it.get('content'))})
        else:
            raise BadRequest(f'unsupported input item type {t!r}')
    for m in msgs:
        m.pop('stage', None)
    chat_body = {k: body[k] for k in RESP_PASS if k in body}
    chat_body['messages'] = msgs
    # function + custom tools in chat form; hosted tools (web search, file search, ...) are dropped: the model just
    # doesn't see them (a 400 would fail every turn of a client that always declares one, e.g. web_search_preview from
    # LiteLLM's Anthropic bridge). max_tool_calls caps hosted tool calls only, so it has nothing to cap here.
    keys = ('name', 'description', 'parameters', 'format')
    as_chat = lambda t: {'type': t['type'], t['type']: {k: t[k] for k in keys if k in t}}
    tools = [as_chat(t) for t in body.get('tools') or [] if isinstance(t, dict) and t.get('type') in ('function', 'custom')]
    if tools:
        chat_body['tools'] = tools
    tc = body.get('tool_choice')
    if isinstance(tc, dict) and tc.get('type') == 'allowed_tools':
        tc = {'type': 'allowed_tools', 'allowed_tools': {'mode': tc.get('mode', 'auto'), 'tools': [
            as_chat(t) for t in tc.get('tools') or [] if isinstance(t, dict) and t.get('type') in ('function', 'custom')]}}
    elif isinstance(tc, dict):
        if tc.get('type') not in ('function', 'custom') or not tc.get('name'):
            raise BadRequest('tool_choice must be auto, none, required, allowed_tools or a function / custom tool')
        tc = as_chat(tc)
    if tc is not None:
        chat_body['tool_choice'] = tc
    if 'message.output_text.logprobs' in (body.get('include') or []) or body.get('top_logprobs'):
        chat_body.update(logprobs=True, top_logprobs=body.get('top_logprobs') or 0)
    fmt = (body.get('text') or {}).get('format') or {}
    if fmt.get('type') == 'json_schema':
        chat_body['response_format'] = {'type': 'json_schema', 'json_schema': {
            'name': fmt.get('name', 'output'), 'schema': fmt.get('schema'), 'strict': fmt.get('strict')}}
    elif fmt.get('type') not in (None, 'text'):
        chat_body['response_format'] = {'type': fmt['type']}  # json_object (anything else: compile_format says no)
    effort = (body.get('reasoning') or {}).get('effort')
    if effort is not None:
        chat_body['reasoning_effort'] = effort
    if body.get('max_output_tokens') is not None:
        chat_body['max_tokens'] = body['max_output_tokens']
    return chat_body


ENC_PRE = 'ink1:'  # our reasoning items' encrypted_content: opaque, not secret (base64 of the client's own reasoning)


def enc_reasoning(text):
    return ENC_PRE + base64.b64encode(text.encode('utf-8')).decode('ascii')


def dec_reasoning(s):
    """Our encrypted_content -> the reasoning text; None for another model's (dropped)."""
    if not isinstance(s, str) or not s.startswith(ENC_PRE):
        return None
    try:
        return base64.b64decode(s[len(ENC_PRE):], validate=True).decode('utf-8')
    except ValueError:
        raise BadRequest('reasoning item: bad encrypted_content')


class RespOut:
    """Engine deltas -> Responses output items, plus the stream events that build them (the non-stream path drops
    the events). Reasoning: the raw text always goes in the item's content (reasoning_text); `summary` only when the
    request asks for one (reasoning.summary; there's no summarizer, so it's the same full text); include
    reasoning.encrypted_content adds it as encrypted_content (ENC_PRE), which carries the full reasoning to the next turn
    through clients that keep only that (LiteLLM's Anthropic bridge: a thinking block's signature)."""

    def __init__(self, custom=(), summary=False, enc=False):
        self.items, self.cur, self.acc, self.lps, self.custom, self.cs = [], None, '', [], custom, None
        self.summary, self.enc = summary, enc

    def _ev(self, type, **kw):
        return dict(type=type, output_index=len(self.items) - 1, **kw)

    def delta(self, kind, text, lp=()):
        lp = list(lp or ())
        if not text and not (lp and self.cur and self.cur['type'] == 'message'):
            return []  # (an empty content delta: no empty message item; one carrying held-back logprobs joins the open one)
        ev = []
        want = 'reasoning' if kind == 'reasoning' else 'message'
        if self.cur is None or self.cur['type'] != want:
            ev += self.close()
            if want == 'reasoning':
                self.cur = {'id': 'rs_' + uuid.uuid4().hex[:24], 'type': 'reasoning', 'summary': [], 'content': []}
                self.items.append(self.cur)
                ev += [self._ev('response.output_item.added', item=dict(self.cur)),
                       self._ev('response.content_part.added', item_id=self.cur['id'], content_index=0,
                                part={'type': 'reasoning_text', 'text': ''})]
                if self.summary:
                    ev.append(self._ev('response.reasoning_summary_part.added', item_id=self.cur['id'], summary_index=0,
                                       part={'type': 'summary_text', 'text': ''}))
            else:
                self.cur = {'id': 'msg_' + uuid.uuid4().hex[:24], 'type': 'message', 'status': 'in_progress',
                            'role': 'assistant', 'content': []}
                self.items.append(self.cur)
                ev += [self._ev('response.output_item.added', item=dict(self.cur)),
                       self._ev('response.content_part.added', item_id=self.cur['id'], content_index=0,
                                part={'type': 'output_text', 'text': '', 'annotations': []})]
        self.acc += text
        if want == 'reasoning':
            ev.append(self._ev('response.reasoning_text.delta', item_id=self.cur['id'], content_index=0, delta=text))
            if self.summary:
                ev.append(self._ev('response.reasoning_summary_text.delta', item_id=self.cur['id'], summary_index=0,
                                   delta=text))
        else:
            self.lps += lp
            ev.append(self._ev('response.output_text.delta', item_id=self.cur['id'], content_index=0, delta=text,
                               logprobs=lp))
        return ev

    def close(self):
        c, text, lps, self.cur, self.acc, self.lps = self.cur, self.acc, self.lps, None, '', []
        if c is None or c['type'] not in ('reasoning', 'message'):  # (a call left open: the stream ended inside it)
            self.cs = None
            return []
        i = dict(item_id=c['id'], output_index=self.items.index(c))
        if c['type'] == 'reasoning':
            raw = {'type': 'reasoning_text', 'text': text}
            c['content'] = [raw]
            ev = [dict(type='response.reasoning_text.done', content_index=0, text=text, **i),
                  dict(type='response.content_part.done', content_index=0, part=raw, **i)]
            if self.summary:
                part = {'type': 'summary_text', 'text': text}
                c['summary'] = [part]
                ev += [dict(type='response.reasoning_summary_text.done', summary_index=0, text=text, **i),
                       dict(type='response.reasoning_summary_part.done', summary_index=0, part=part, **i)]
            if self.enc:
                c['encrypted_content'] = enc_reasoning(text)
        else:
            part = {'type': 'output_text', 'text': text, 'annotations': [], 'logprobs': lps}
            c.update(status='completed', content=[part])
            ev = [dict(type='response.output_text.done', content_index=0, text=text, logprobs=lps, **i),
                  dict(type='response.content_part.done', content_index=0, part=part, **i)]
        return ev + [dict(type='response.output_item.done', output_index=i['output_index'], item=c)]

    def call(self, kind, val):
        """A parser call_* delta -> events. A streamed call opens its item at call_start and sends argument (custom:
        decoded input) deltas; one parsed only at its end goes out whole at call_end."""
        ev = []
        if kind == 'call_start' or (kind == 'call_end' and self.cs is None):
            ev += self.close()
            name = val if kind == 'call_start' else val['name']
            custom = name in self.custom
            t, field = ('custom_tool_call', 'input') if custom else ('function_call', 'arguments')
            self.cur = {'id': ('ctc_' if custom else 'fc_') + uuid.uuid4().hex[:24], 'type': t, 'status': 'in_progress',
                        'call_id': 'call_' + uuid.uuid4().hex[:24], 'name': name, field: ''}
            self.items.append(self.cur)
            ev.append(self._ev('response.output_item.added', item=dict(self.cur)))
            self.cs = CustomInput()  # (the open call's input decoder; function calls just use its presence)
            if kind == 'call_start':
                return ev
        it = self.cur
        custom, field = it['type'] == 'custom_tool_call', 'input' if it['type'] == 'custom_tool_call' else 'arguments'
        if kind == 'call_args':
            d = self.cs.feed(val) if custom else val
        else:
            final = custom_input(val) if custom else val['arguments']
            d = self.cs.finish(final) if custom else final[len(it[field]):]
        if d:
            it[field] += d
            ev.append(self._ev(f'response.{it["type"]}_{field}.delta', item_id=it['id'], delta=d))
        if kind == 'call_end':
            self.cs = None
            it['status'] = 'completed'
            ev += [self._ev(f'response.{it["type"]}_{field}.done', item_id=it['id'], **{field: it[field]},
                            **({} if custom else {'name': it['name']})),
                   self._ev('response.output_item.done', item=dict(it))]
            self.cur = None
        return ev


def prompt_len(cb, msgs=None, memo=None):
    """Chat body (messages: msgs if given) -> prompt length in tokens as the engine renders it (images sized from
    their headers)."""
    tools, _, _ = chat_tools(cb)
    effort, no_think = effort_of(cb)
    n = eng.count(chat_messages(cb['messages'] if msgs is None else msgs), tools, effort, memo)
    return n + 2 * (no_think and not tools)  # (thinking off: the pre-filled answer header)


def fit(cb):
    """truncation 'auto': the messages without the fewest oldest turns (a user or assistant message plus the tool
    results after it; system / developer messages stay) such that prompt + max_tokens fits the context. The last turn
    always stays (if even that doesn't fit, the engine's 400 says so)."""
    msgs, mt, memo = cb['messages'], int(cb.get('max_tokens') or DEFAULT_MAX_TOKENS), {}
    starts = [i for i, m in enumerate(msgs) if m.get('role') in ('user', 'assistant')]
    without = lambda k: msgs if k == 0 else \
        [m for i, m in enumerate(msgs) if i >= starts[k] or m.get('role') in ('system', 'developer')]
    ok = lambda k: prompt_len(cb, without(k), memo) + mt <= eng.max_ctx
    if len(starts) < 2 or ok(0):
        return msgs
    lo, hi = 1, len(starts) - 1  # smallest k that fits (binary search: each try renders the whole prompt)
    if not ok(hi):
        return without(hi)
    while lo < hi:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if ok(mid) else (mid + 1, hi)
    return without(lo)


async def resp_body(request):
    """-> (Responses body, chat body with truncation applied) or an error response."""
    try:
        body = await request.json()
    except ValueError:
        return err(400, 'body is not JSON'), None
    if not isinstance(body, dict):
        return err(400, 'body must be an object'), None
    try:
        cb = resp_to_chat(body)
        if body.get('truncation') == 'auto':
            cb['messages'] = await asyncio.get_running_loop().run_in_executor(None, fit, cb)
    except (BadRequest, TypeError, ValueError, AttributeError) as e:
        return err(400, str(e)), None
    return body, cb


@app.post('/v1/responses/input_tokens')
async def input_tokens(request: Request):
    body, cb = await resp_body(request)
    if cb is None:
        return body
    try:
        n = await asyncio.get_running_loop().run_in_executor(None, prompt_len, cb)
    except (BadRequest, TypeError, ValueError) as e:
        return err(400, str(e))
    return {'object': 'response.input_tokens', 'input_tokens': n}


@app.post('/v1/responses')
async def responses(request: Request):
    t_arr = time.time()
    body, chat_body = await resp_body(request)
    if chat_body is None:
        return body
    rid, created = 'resp_' + uuid.uuid4().hex[:24], int(time.time())
    run = await submit(chat_body, t_arr)
    if isinstance(run, JSONResponse):
        return run
    st, out = run.stats[0], RespOut(run.custom, summary=(body.get('reasoning') or {}).get('summary') is not None,
                                    enc='reasoning.encrypted_content' in (body.get('include') or []))

    def response(status, error=None):
        p, c = st.get('prompt_tokens', 0), st.get('completion_tokens', 0)
        length = status == 'completed' and st.get('finish_reason') == 'length' and not st.get('tool_calls')
        return {'id': rid, 'object': 'response', 'created_at': created, 'model': body.get('model', SERVED),
                'status': 'incomplete' if length else status, 'error': error,
                'incomplete_details': {'reason': 'max_output_tokens'} if length else None,
                'output': out.items if status != 'in_progress' else [],
                'usage': None if status == 'in_progress' else {
                    'input_tokens': p, 'input_tokens_details': {'cached_tokens': st.get('cached_tokens', 0)},
                    'output_tokens': c, 'output_tokens_details': {'reasoning_tokens': st.get('reasoning_tokens', 0)}, 'total_tokens': p + c},
                'instructions': body.get('instructions'), 'max_output_tokens': body.get('max_output_tokens'),
                'parallel_tool_calls': body.get('parallel_tool_calls', True), 'previous_response_id': None,
                'reasoning': body.get('reasoning') or {'effort': None, 'summary': None}, 'store': False,
                'temperature': body.get('temperature', 1.0), 'top_p': body.get('top_p', 1.0),
                'text': body.get('text') or {'format': {'type': 'text'}}, 'tool_choice': body.get('tool_choice', 'auto'),
                'tools': body.get('tools') or [], 'truncation': body.get('truncation') or 'disabled',
                'top_logprobs': body.get('top_logprobs') or 0, 'max_tool_calls': body.get('max_tool_calls'),
                'service_tier': body.get('service_tier') or 'default', 'metadata': body.get('metadata') or {},
                'user': body.get('user')}

    async def run_events():  # -> Responses events; ends with ('fail', code, message) on an engine error
        async for _, kind, val in run.events():
            if kind == 'delta':
                evs = out.call(val[0], val[1]) if val[0].startswith('call_') else \
                    out.delta(val[0], val[1], val[2] if len(val) > 2 else ())
                for ev in evs:
                    yield ev
            elif kind in ('bad', 'error', 'oom'):
                out.close()
                yield ('fail', {'bad': 400, 'oom': 503}.get(kind, 500), val)
            else:
                for ev in out.close():
                    yield ev

    if not body.get('stream'):
        watcher = asyncio.create_task(watch(request, run.cancel))
        try:
            async for ev in run_events():
                if isinstance(ev, tuple):
                    return err(ev[1], ev[2])
        finally:
            watcher.cancel()
        return response('completed')

    async def sse():
        seq = itertools.count()

        def put(ev):
            ev['sequence_number'] = next(seq)
            return f'event: {ev["type"]}\ndata: {json.dumps(ev)}\n\n'
        yield put({'type': 'response.created', 'response': response('in_progress')})
        yield put({'type': 'response.in_progress', 'response': response('in_progress')})
        async for ev in run_events():
            if isinstance(ev, tuple):
                code = {400: 'invalid_request_error', 503: 'server_is_overloaded'}.get(ev[1], 'server_error')
                yield put({'type': 'response.failed', 'response': response('failed', {'code': code, 'message': ev[2]})})
                return
            yield put(ev)
        r = response('completed')
        yield put({'type': 'response.' + r['status'], 'response': r})

    return StreamingResponse(sse(), media_type='text/event-stream')


def park_auto(e):
    """INK_PARK_AUTO: size engine.PARK by the free VRAM after warm-up (one parked state ~200 MB)."""
    import engine
    per = e.parked_bytes() // len(e.parked) if e.parked else 200 << 20
    keep = engine.MARGIN  # leave room for the drafter + one State per decode slot (window final 11-23: 4 parked starved it)
    if getattr(e, 'spec', None) == 'dspark':
        keep += engine.DS_W_MB + engine.DS_SHORT_W_MB + engine.MAX_BATCH * (engine.DS_ST_MB + engine.DS_SHORT_ST_MB)
    n = min(6, max(engine.PARK, int((e.mem_free() - keep) // max(per, 1))))
    print(f'INK_PARK_AUTO: {n} parked states (was {engine.PARK}; free {e.mem_free() >> 20} MiB, {per >> 20} MiB each)',
          flush=True)
    engine.PARK = n
    return n


def main():
    global eng, sched
    e = Engine(os.environ.get('INK_DEVICE', 'cuda'))
    if os.environ.get('INK_WARMUP', '1') != '0':  # first-use compiles/captures now, not on the first user request
        try:
            M['warmup_seconds'] = round(e.warmup(min(MIX_CHUNK, e.chunk)), 1)
            print(f'warm-up done in {M["warmup_seconds"]} s', flush=True)
        except Exception:  # never block serving; the paths warm up on first use instead
            import traceback; traceback.print_exc()
    if PARK_AUTO and e.mem_free is not None:
        park_auto(e)
    eng, sched = e, Scheduler(e)  # (the HTTP server, /health included, starts only after this)
    uvicorn.run(app, host=os.environ.get('INK_HOST', '127.0.0.1'), port=int(os.environ.get('INK_PORT', 30000)), log_level='info')


if __name__ == '__main__':
    main()
