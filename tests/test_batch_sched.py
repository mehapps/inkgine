"""CPU, no model: server.Scheduler (continuous batching) against a fake step engine. Priority admission order, decode
fairness, time share (+ VIP-aware), chunk size drop while decodes run, VIP preemption of a decode among several
running (resumes with no gap/duplicate), head-of-line blocking, cancellation, NoRoom (waits while others run, gives up
alone), solo mode, no page leak. FakeEngine is also used by test_sched.py / test_chat.py (HTTP level).
Run: cd engine && PATH=$PWD/.venv/bin:$PATH python tests/test_batch_sched.py"""
import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import server
from engine import NoRoom, OutOfMemory


class FakeReq:
    def __init__(self, spec, stats, emit, prio, cancel):
        name, chunks, toks, *pg = spec.split(':')
        self.name, self.chunks, self.toks, self.need = name, int(chunks), int(toks), int(pg[0]) if pg else 1
        self.stats, self.emit, self.prio, self.cancel = stats, emit or (lambda *a: None), prio, cancel or threading.Event()
        self.a, self.out, self.slot, self.held, self.done, self.solo = 0, [], None, 0, False, False


class FakeEngine:
    """Prompt 'name:chunks:tokens[:pages]'. A prefill chunk takes chunk_s, a decode step step_s whatever the batch
    size. Names: never (always NoRoom), full* (NoRoom the first time), oom (OutOfMemory in prefill), solo* (solo)."""
    tok, parked, ooms = None, [], 0
    parked_bytes = staticmethod(lambda: 123)

    def __init__(self, slots=4, pages=100, chunk_s=0.05, step_s=0.05, chunk=4096):
        self.slots, self.total, self.free, self.chunk_s, self.step_s, self.chunk = slots, pages, pages, chunk_s, step_s, chunk
        self.log, self.steps, self.kw, self.lock = [], [], [], threading.Lock()
        eng = self

        class pool:  # what /metrics and the NoRoom give-up log read
            nodes = {}
            free_pages = staticmethod(lambda: eng.free)
        self.pool = pool

    def L(self, *e):
        with self.lock:
            self.log.append(e + (time.time(),))

    def has(self, *e):
        return any(x[:len(e)] == e for x in self.log)

    def request(self, messages, stats=None, emit=None, prio=0, cancel=None, **kw):
        self.kw.append(kw)
        return FakeReq(messages[-1]['content'], {} if stats is None else stats, emit, prio, cancel)

    def uncached(self, r):  # (scheduler short-first: prefill tokens left, chunk = eng.chunk tokens)
        return (r.chunks - r.a) * self.chunk

    def can_start(self, r):
        return True  # (start decides; the real engine pre-checks pages)

    def start(self, r):
        self.L('start', r.name)
        if r.name == 'never':
            raise NoRoom('never')
        if r.name.startswith('full') and not self.has('noroom', r.name):
            self.L('noroom', r.name); raise NoRoom(r.name)
        if r.need > self.free:
            self.L('noroom', r.name); raise NoRoom(f'{r.need} pages')
        self.free -= r.need
        r.held, r.solo = r.need, r.name.startswith('solo')

    def prefill_chunk(self, r, n):
        if r.name == 'oom':
            raise OutOfMemory('out of GPU memory (test)')
        if r.a < r.chunks:
            t0 = time.time()
            time.sleep(self.chunk_s)
            r.a += 1
            self.L('chunk', r.name, n, t0)
        if r.a < r.chunks:
            return False
        if not r.out:
            r.stats.update(prompt_tokens=r.chunks, prefill_s=self.chunk_s * r.chunks)
        return True

    def to_decode(self, r, slot):
        r.slot, r.solo = slot, False  # (engine: solo ends with the prefill)
        if not r.out:
            self._tok(r)

    def decode_step(self, reqs):
        assert len({r.slot for r in reqs}) == len(reqs) and all(0 <= r.slot < self.slots for r in reqs), 'slot clash'
        t0 = time.time()
        time.sleep(self.step_s)
        self.steps.append([r.name for r in reqs])
        self.L('step', len(reqs), t0)
        for r in reqs:
            self._tok(r)

    def _tok(self, r):
        r.out.append(len(r.out))
        r.emit('delta', ('content', f'{r.name}{len(r.out) - 1} '))
        if len(r.out) >= r.toks:
            r.done = True
            r.stats.update(completion_tokens=len(r.out), decode_s=self.step_s * len(r.out), finish_reason='stop')

    def preempt_prefill(self, r, park=True):
        self.L('preempted', r.name)  # progress kept (parked): resumes at r.a
        self.finish(r)

    def preempt_decode(self, r):
        self.L('parked', r.name)     # r.out kept: resumes with its pending token, no recompute
        self.finish(r)

    def finish(self, r):
        self.free += r.held
        r.held, r.slot = 0, None


def run(eng, specs, sched=None):
    """specs: [(delay, 'name:chunks:toks[:pages]', prio)] submitted directly to a Scheduler. -> {name: result}"""
    sched = sched or server.Scheduler(eng)
    res = {}

    def one(delay, spec, prio):
        time.sleep(delay)
        name, ev, out = spec.split(':')[0], threading.Event(), []

        def emit(kind, val):
            out.append(val[1] if kind == 'delta' else None)
            if kind != 'delta':
                res[name] = dict(kind=kind, msg=val, text=''.join(x for x in out if x), t=time.time())
                ev.set()
        req = eng.request([{'content': spec}], emit=emit, prio=prio)
        res[name + ':req'] = req
        sched.submit(req)
        ev.wait(60)
    th = [threading.Thread(target=one, args=s) for s in specs]
    for t in th:
        t.start()
    for t in th:
        t.join()
    return res


def order(res):
    return sorted((k for k in res if ':' not in k), key=lambda k: res[k]['t'])


def full_text(name, n):
    return ''.join(f'{name}{i} ' for i in range(n))


if __name__ == '__main__':
    # 1. admission order: priority desc, FIFO within a level (a prio-10 blocker no one can preempt holds the 1 slot)
    eng = FakeEngine(slots=1)
    res = run(eng, [(0, 'blocker:1:40', 10), (0.1, 'a:1:1', 0), (0.4, 'b:1:1', 5), (0.7, 'c:1:1', 0), (1.0, 'd:1:1', 5)])
    assert order(res) == ['blocker', 'b', 'd', 'a', 'c'], order(res)
    print('1 admission order (priority, then FIFO):', order(res))

    # 2. fairness: 3 concurrent decodes each get a row every step, finish together; prefills FIFO
    eng = FakeEngine(slots=4, step_s=0.02)
    res = run(eng, [(0, 'x:1:20', 0), (0.01, 'y:1:20', 0), (0.02, 'z:1:20', 0)])
    assert [e[1] for e in eng.log if e[0] == 'start'] == ['x', 'y', 'z']
    assert ['x', 'y', 'z'] in [sorted(s) for s in eng.steps] and max(len(s) for s in eng.steps) == 3
    for n in 'xyz':
        idx = [i for i, s in enumerate(eng.steps) if n in s]
        assert idx == list(range(idx[0], idx[-1] + 1)), (n, idx)  # never skipped once decoding
        assert res[n]['text'] == full_text(n, 20)
    spread = max(res[n]['t'] for n in 'xyz') - min(res[n]['t'] for n in 'xyz')
    assert spread < 0.5, spread
    print(f'2 fairness: 3 rows per step, finish spread {spread:.2f}s')

    # 3. time share while a long prefill and decodes both run, + chunk size drop (INK_MIX_CHUNK) while decodes run
    def share(dec_prio, pf_prio, want):
        eng = FakeEngine(slots=4, chunk_s=0.03, step_s=0.01)
        res = run(eng, [(0, 'dec:1:500', dec_prio), (0.2, 'long:30:1', pf_prio)])
        ch = [e for e in eng.log if e[0] == 'chunk' and e[1] == 'long']  # (chunk, name, n, t0, t_end)
        a, b = ch[0][3], ch[-1][4]  # window: first to last chunk of the long prefill
        tp = sum(e[4] - e[3] for e in ch)
        td = sum(min(e[3], b) - max(e[2], a) for e in eng.log if e[0] == 'step' and e[3] > a and e[2] < b)
        got = td / (td + tp)
        assert abs(got - want) <= 0.1, (dec_prio, pf_prio, got, want)
        assert {e[2] for e in ch} == {2048} and res['dec']['text'] == full_text('dec', 500)
        assert res['long']['t'] < res['dec']['t']
        return got
    s1, s2, s3 = share(0, 0, 0.5), share(10, 0, 0.8), share(0, 10, 0.3)
    eng = FakeEngine(slots=4)
    run(eng, [(0, 'alone:2:1', 0)])
    assert [e[2] for e in eng.log if e[0] == 'chunk'] == [4096, 4096]
    print(f'3 time share decode {s1:.2f} (want 0.5), VIP decode {s2:.2f} (0.8), VIP prefill {s3:.2f} (0.3); '
          f'chunk 2048 with decodes, 4096 alone')

    # 4. VIP with all slots decoding: the newest low decode is parked (others keep a row every step), VIP runs, the
    #    parked one resumes with no gap or duplicate; equal priority never preempts
    eng = FakeEngine(slots=3, step_s=0.02)
    res = run(eng, [(0, 'l1:1:30', 0), (0.01, 'l2:1:30', 0), (0.02, 'l3:1:30', 0), (0.3, 'same:1:3', 0),
                    (0.4, 'vip:2:3', 10)])
    newest = max(('l1', 'l2', 'l3'), key=lambda n: res[n + ':req'].order)
    assert [e[1] for e in eng.log if e[0] == 'parked'] == [newest], (newest, eng.log)
    assert res['vip']['t'] < min(res[n]['t'] for n in ('l1', 'l2', 'l3', 'same')), order(res)
    for n in ('l1', 'l2', 'l3'):
        assert res[n]['text'] == full_text(n, 30), res[n]['text']
    for n in {'l1', 'l2', 'l3'} - {newest}:
        idx = [i for i, s in enumerate(eng.steps) if n in s]
        assert idx == list(range(idx[0], idx[-1] + 1)), (n, 'paused')
    assert [e[1] for e in eng.log if e[0] == 'start'].count(newest) == 2 and not eng.has('preempted')
    print('4 VIP parked the newest of 3 running decodes; resumed stream continuous:', order(res))

    # 5. head-of-line: a big head blocked on pages; a later small request (same or lower priority) waits behind it
    eng = FakeEngine(slots=4, pages=100, step_s=0.02)
    res = run(eng, [(0, 'r1:1:30:60', 0), (0.1, 'big:1:2:50', 0), (0.15, 'small:1:2:5', 0), (0.2, 'lower:1:2:5', -1)])
    starts = [e[1] for e in eng.log if e[0] == 'start']
    first_ok = [e[1] for e in eng.log if e[0] == 'chunk']
    assert eng.has('noroom', 'big') and first_ok.index('big') < first_ok.index('small') < first_ok.index('lower'), \
        (starts, first_ok)
    assert res['r1']['t'] < res['big']['t']
    print('5 head-of-line: blocked big head not passed by small ones:', first_ok)

    # 6. cancel: a decoding request (slot + pages freed, no park, others go on) and a queued one (never started)
    eng = FakeEngine(slots=1, step_s=0.02)
    sched = server.Scheduler(eng)
    reqs = [eng.request([{'content': s}]) for s in ('c1:1:500:10', 'c2:1:500:10', 'c3:1:2:10')]
    for r in reqs:
        sched.submit(r)
    time.sleep(0.3)
    reqs[1].cancel.set()  # queued
    reqs[0].cancel.set()  # decoding
    t0 = time.time()
    while not reqs[2].done and time.time() - t0 < 5:
        time.sleep(0.02)
    assert reqs[2].done and not reqs[0].done and not eng.has('start', 'c2') and not eng.has('parked')
    time.sleep(0.1)
    assert eng.free == eng.total and sched.running() == 0 and sched.depth() == 0, (eng.free, sched.running())
    print('6 cancel: decoding + queued requests dropped, pages back')

    # 7. NoRoom: waits (beyond INK_NOROOM_S) while something runs, gives up when alone; OOM -> 'oom'
    server.NOROOM_S = 0.5
    eng = FakeEngine(slots=4, pages=100, step_s=0.02)
    res = run(eng, [(0, 'r1:1:60:80', 0), (0.1, 'wait:1:2:50', 0), (0.2, 'never:1:1', -1), (0.2, 'oom:1:1', 0)])
    assert res['wait']['kind'] == 'done' and res['wait']['t'] > res['r1']['t'], res['wait']
    assert res['never']['kind'] == 'error' and 'KV cache full' in res['never']['msg'], res['never']
    assert res['oom']['kind'] == 'oom', res['oom']
    assert eng.free == eng.total
    server.NOROOM_S = 30
    print('7 NoRoom: waited while r1 ran, gave up alone; OOM -> oom')

    # 8. solo: a long cold prompt's prefill runs alone (nothing starts before its last chunk); solo ends there, so the
    #    others are admitted while it decodes (S1: solo no longer blocks admission for the whole reply)
    eng = FakeEngine(slots=4, step_s=0.02)
    res = run(eng, [(0, 'solo1:3:10', 0), (0.05, 'small1:1:2', 0), (0.06, 'small2:1:2', 0)])
    t_last = max(e[-1] for e in eng.log if e[:2] == ('chunk', 'solo1'))
    assert all(e[-1] > t_last for e in eng.log if e[0] == 'start' and e[1] != 'solo1'), eng.log
    assert any('solo1' in s and len(s) > 1 for s in eng.steps), eng.steps
    assert res['solo1']['t'] > max(res['small1']['t'], res['small2']['t'])
    print('8 solo prefill ran alone; the others decoded alongside its reply')

    # 9. VRAM policy (real Engine methods + LoopBatch, fake Decoders and a fake mem_get_info computed from what is
    # held): pressure drops idle slot Decoders first, then parked states (LRU); never a busy slot's Decoder; a new
    # slot's Decoder is preceded by a margin check; finishing the last running request keeps one warm Decoder; the
    # loaded Seq's prefill rings are released; free() forgets the Seq
    import torch
    import engine
    import loopbatch

    class Dec:
        graph = False

        def __init__(self, m, embed_dev=None):
            self.seq = None

        def load(self, seq):
            self.seq = seq

    class Seq:
        def __init__(self):
            self.ring = [(torch.zeros(5, 2, 3), torch.zeros(5, 2, 3))] * 3
            self.kv = None
    loopbatch.Decoder = Dec
    GB = 1 << 30
    e = engine.Engine.__new__(engine.Engine)
    e.bd = lb = loopbatch.LoopBatch(type('M', (), {'dev': torch.device('cpu')})(), 4, dec=True)
    e.ooms, e.parked, gone, e.ds, e.ds_free, e.spec_live = 0, [], [], None, [], set()
    e.unpark = lambda ent, disk=True: gone.append(ent)
    # idle 5 GB of headroom; each Decoder holds 0.4 GB, each parked state 0.2 GB
    e.mem_free = lambda: 5 * GB - int(0.4 * GB) * sum(d is not None for d in lb.decs) - int(0.2 * GB) * len(e.parked)
    for slot in range(4):
        lb.load(slot, Seq())
    assert all(sq.ring[0][0].numel() == 0 for sq in lb.seqs), 'prefill rings kept after load'
    for slot in (1, 2, 3):
        lb.free(slot)
    assert lb.decs[1].seq is None and lb.idle() == 3
    e.parked = ['p0', 'p1', 'p2']                       # free now 5 - 1.6 - 0.6 = 2.8 GB
    e.free_vram(3 * GB)                                 # idle Decoders cover it: no parked state touched
    assert lb.decs[0] is not None and lb.decs[1:] == [None] * 3 and not gone, (lb.decs, gone)
    for slot in (1, 2):
        lb.load(slot, Seq()); lb.free(slot)             # 2 idle again: free 3.2 GB
    e.free_vram(int(4.5 * GB))                          # idle Decoders (-> 4.0), then parked LRU (4.2, 4.4, 4.6)
    assert lb.decs[1:] == [None] * 3 and gone == ['p0', 'p1', 'p2'] and lb.decs[0] is not None, (lb.decs, gone)
    gone.clear(); e.parked = ['q0', 'q1']
    e.free_vram(int(4.5 * GB))                          # free 4.2, only parked left: LRU until met (4.4, 4.6)
    assert gone == ['q0', 'q1'] and lb.decs[0] is not None
    e.free_vram(1 * GB)                                 # already met: nothing happens
    lb.load(1, Seq()); lb.free(1); e.parked = ['r0']
    gone.clear(); e.free_vram()                         # OOM path: every idle Decoder + every parked state
    assert lb.decs[1:] == [None] * 3 and gone == ['r0'] and lb.decs[0] is not None
    # slot admission: a new Decoder only after free_vram(SLOT_NEED); an existing idle one needs no check
    for slot in (2, 3):
        lb.load(slot, Seq()); lb.free(slot)
    e.parked, gone[:] = ['s0'], []
    old, engine.SLOT_NEED = engine.SLOT_NEED, int(3.7 * GB)  # free 5 - 1.2 - 0.2 = 3.6 GB: below it
    req = engine.Request.__new__(engine.Request)
    req.ids, req.seq, req.out = [1, 2], Seq(), [7]      # a resumed decode (no first-token sampling)
    e.to_decode(req, 1)
    assert lb.decs[1] is not None and lb.decs[2:] == [None, None] and gone == [], (lb.decs, gone)
    engine.SLOT_NEED = old
    # idle trim: the last running request finishing keeps one warm Decoder (the lowest free slot's)
    e.pool = type('P', (), {'release': staticmethod(lambda t: None)})()
    lb.load(2, Seq()); lb.load(3, Seq())
    r1, r2 = engine.Request.__new__(engine.Request), engine.Request.__new__(engine.Request)
    for r, slot in ((r1, 0), (r2, 1)):
        r.slot, r.seq, r.img = slot, None, None
    lb.free(2); lb.free(3)
    e.finish(r2)                                        # slot 0 still decodes: idle Decoders stay
    assert sum(d is not None for d in lb.decs) == 4
    e.finish(r1)                                        # nothing runs: one warm Decoder kept
    assert sum(d is not None for d in lb.decs) == 1 and lb.decs[0] is not None, lb.decs
    print('9 VRAM policy: idle slot Decoders before parked states, busy slots untouched, slot margin, idle trim')
    body = server.metrics().body.decode()
    assert '# TYPE inkling_warmup_seconds gauge\ninkling_warmup_seconds 0.0' in body, body
    print('10 /metrics has the inkling_warmup_seconds gauge (engine warm-up itself: test_engine_cpu)')

    # 11. BatchDecoder semantics (fake): one set of static buffers for all slots, allocated at the first load, drop()
    # frees them only when no slot is in use and keep == 0; decs = [itself] * n while allocated. Real Engine paths:
    # to_decode's margin (all slots at once), free_vram while busy (buffers stay, parked states go), finish keeps
    # them, free_vram when idle frees them, a solo prompt drops them; then SoloFused's moves (fake both sides)
    class FakeBD:
        dec = True

        def __init__(self, n):
            self.n, self.b, self.seqs, self.pend, self.steps = n, None, [None] * n, {}, []

        def busy(self):
            return any(q is not None for q in self.seqs)

        def idle(self):
            return self.n if self.b is not None and not self.busy() else 0

        @property
        def decs(self):
            return [self if self.b is not None else None] * self.n

        def load(self, slot, seq):
            assert self.seqs[slot] is None
            self.b = self.b or object()
            self.seqs[slot] = seq
            seq.st = (seq.conv[0]['k'].clone(), seq.ring[0][0].clone())  # the state this slot continues from

        def free(self, slot):
            self.seqs[slot] = None

        def drop(self, keep=0):
            if self.b is None or keep or self.busy():
                return 0
            self.b = None
            return self.n

        def state(self, slot):
            q = self.seqs[slot]
            return {'conv.0.k': q.st[0] + q.pos, 'ring.0.k': q.st[1] + q.pos, 'ring.0.v': q.st[1] - q.pos}

        def step(self, rows):
            self.steps.append([s for s, _, _ in rows])
            for s, p, _ in rows:
                assert self.seqs[s].pos == p
                self.seqs[s].pos += 1
            return torch.zeros(len(rows), 4)

        def commit(self, acc):
            pass

    class BSeq:
        def __init__(self, pos=0):
            self.pos, self.kv = pos, None
            self.conv, self.ring = [{'k': torch.zeros(3)}], [(torch.zeros(2), torch.zeros(2))]
    e = engine.Engine.__new__(engine.Engine)
    e.bd = bd = FakeBD(4)
    e.pool = type('P', (), {'release': staticmethod(lambda t: None)})()
    e.ooms, e.parked, gone, needs, e.ds, e.ds_free, e.spec_live = 0, ['p0', 'p1'], [], [], None, [], set()
    e.unpark = lambda ent, disk=True: gone.append(ent)
    e.mem_free = lambda: 5 * GB - 900 * 2**20 * (bd.b is not None) - int(0.2 * GB) * len(e.parked)
    real_fv = e.free_vram
    e.free_vram = lambda need=None, **kw: (needs.append(need), real_fv(need, **kw))
    rq = lambda slot=None: type('R', (), dict(ids=[1, 2], seq=BSeq(2), out=[7], slot=slot, img=None, sp=None))()
    r0, r1 = rq(), rq()
    e.to_decode(r0, 0)                                    # first load: margin for every slot's buffers
    assert needs == [engine.SLOT_NEED + 3 * engine.SLOT_MB] and bd.b is not None, needs
    e.to_decode(r1, 1)                                    # buffers exist: no margin check
    assert needs == [engine.SLOT_NEED + 3 * engine.SLOT_MB], needs
    e._take = lambda reqs, lg: None                      # (sampling: not this test's business)
    e.decode_step([r0, r1])
    assert bd.steps == [[0, 1]] and (r0.pos, r1.pos) == (3, 3)
    e.free_vram()                                         # busy: the shared buffers stay, parked states go
    assert bd.b is not None and gone == ['p0', 'p1'] and not e.parked
    e.finish(r1)
    e.finish(r0)                                          # idle: drop(keep=1) keeps them (the warm state)
    assert bd.b is not None and bd.idle() == 4 and not bd.busy()
    e.free_vram()                                         # idle + pressure: freed
    assert bd.b is None and bd.decs == [None] * 4
    e.to_decode(rq(), 0)                                  # realloc; then a solo prompt while slot 0 decodes: no drop
    e.keys, e.admit = (lambda ids, pre: ids), (lambda tbl, n: None)
    e.restore = lambda keys, need_f=True: (BSeq(), 0, None)
    old, engine.DEC_DROP = engine.DEC_DROP, 2
    long = type('R', (), dict(ids=[1, 2, 3], pre=[], out=[], need=4, sp=None, spec=False, oomed=False))()
    e._start(long)
    assert not getattr(long, 'solo', False) and bd.b is not None
    bd.free(0)
    e._start(long)                                        # nothing decodes: solo, buffers dropped
    assert long.solo and bd.b is None
    e.model = type('M', (), {'logits': staticmethod(lambda f: None)})()
    e.to_decode(long, 0)                                  # solo ends with the prefill (S1): admission reopens
    assert not long.solo and bd.b is not None
    bd.free(0)
    e.restore = lambda keys, need_f=True: (BSeq(), 2, None)  # warm turn: 1 uncached token < DEC_DROP
    e._start(long)                                        # nothing decodes, long prompt, but warm: no solo, no drop
    assert not long.solo and bd.b is not None
    engine.DEC_DROP = old
    # SoloFused: alone -> the fused side; a second load moves it to the batch side (state -> load), the last one
    # left moves back; steps go to whichever side holds the rows; an OOM on the way back leaves it batched
    sf = loopbatch.SoloFused(FakeBD(4), None)
    sf.one = FakeBD(1)
    a, b = BSeq(5), BSeq(9)
    sf.load(2, a)
    assert sf.at == 2 and sf.one.seqs[0] is a and not sf.bd.busy() and sf.busy()
    sf.step([(2, 5, 0)])
    assert sf.one.steps == [[0]] and a.pos == 6
    sf.load(0, b)                                         # a moves with its state at pos 6
    assert sf.at is None and sf.bd.seqs[2] is a and sf.bd.seqs[0] is b and not sf.one.busy()
    assert float(a.conv[0]['k'][0]) == 6 and float(a.ring[0][0][0]) == 6 and float(a.ring[0][1][0]) == -6
    sf.step([(0, 9, 0), (2, 6, 0)])
    assert sf.bd.steps == [[0, 2]] and (a.pos, b.pos) == (7, 10)
    sf.free(0)                                            # a alone again: back to the fused side
    assert sf.at == 2 and sf.one.seqs[0] is a and not sf.bd.busy() and float(a.conv[0]['k'][0]) == 13
    assert sf.drop(keep=1) == 0 and sf.idle() == 4        # (one is busy; the batch side's buffers stay with keep)
    sf.free(2)
    assert not sf.busy() and sf.drop() == 5               # both sides freed when idle
    sf.load(1, a); sf.load(3, b)                          # a batched again; b's departure -> a back, but OOM
    sf.one.load = lambda *x: (_ for _ in ()).throw(torch.OutOfMemoryError('fake'))
    sf.free(3)
    assert sf.at is None and sf.bd.seqs[1] is a, 'OOM on the way back lost the request'
    sf.step([(1, a.pos, 0)])
    print('11 BatchDecoder semantics (fake): shared buffers, margin for all slots, drop only when idle, solo drop; '
          'SoloFused moves both ways, step routing, OOM on the way back')

    # 12. short-first (INK_SHORT_FIRST): a short warm turn (uncached suffix <= INK_SHORT_MAX) preempts a same-priority
    #     long cold prefill (parked, resumes where it was); the long one gets >= 1 chunk between such preemptions;
    #     never across priorities, never for a suffix over SHORT_MAX, off with the flag; small suffixes get full
    #     chunks while decodes run
    sp0 = server.M['short_preemptions']
    eng = FakeEngine(slots=4, chunk_s=0.03, step_s=0.01)
    res = run(eng, [(0, 'cold:30:2', 0), (0.2, 'warm:2:3', 0)])
    ch = [e for e in eng.log if e[0] == 'chunk']
    assert res['warm']['t'] < res['cold']['t'] and eng.has('preempted', 'cold'), order(res)
    assert [e[1] for e in ch].index('warm') < max(i for i, e in enumerate(ch) if e[1] == 'cold')
    assert sum(e[1] == 'cold' for e in ch) == 30 and res['cold']['text'] == full_text('cold', 2)
    assert server.M['short_preemptions'] == sp0 + 1, server.M['short_preemptions']
    eng = FakeEngine(slots=4, chunk_s=0.03, step_s=0.01)
    res = run(eng, [(0, 'cold:20:1', 0), (0.15, 'w1:1:1', 0), (0.15, 'w2:1:1', 0), (0.15, 'w3:1:1', 0)])
    ev = [e[1] if e[0] == 'chunk' else 'P' for e in eng.log if e[:2] in (('chunk', 'cold'), ('preempted', 'cold'))]
    assert ev.count('P') == 3 and 'P' not in ''.join(x[0] for x in ev).replace('cP', ''), ev  # a cold chunk before each P
    assert all(res[n]['t'] < res['cold']['t'] for n in ('w1', 'w2', 'w3'))
    for specs in ([(0, 'cold:20:1', 5), (0.15, 'warm:1:1', 0)],   # lower priority: waits (VIP rules unchanged)
                  [(0, 'cold:20:1', 0), (0.15, 'mid:5:1', 0)]):   # 5 chunks = 20480 > SHORT_MAX: FIFO
        eng = FakeEngine(slots=4, chunk_s=0.02, step_s=0.01)
        res = run(eng, specs)
        assert not eng.has('preempted') and order(res)[0] == 'cold', (specs, order(res))
    server.SHORT_FIRST = False
    eng = FakeEngine(slots=4, chunk_s=0.02, step_s=0.01)
    res = run(eng, [(0, 'cold:20:1', 0), (0.15, 'warm:1:1', 0)])
    assert not eng.has('preempted') and order(res) == ['cold', 'warm'], order(res)
    server.SHORT_FIRST = True
    eng = FakeEngine(slots=4, chunk_s=0.03, step_s=0.01)
    run(eng, [(0, 'dec:1:300', 0), (0.2, 'warm:1:1', 0), (0.4, 'long:4:1', 0)])
    assert [e[2] for e in eng.log if e[:2] == ('chunk', 'warm')] == [4096], eng.log
    assert {e[2] for e in eng.log if e[:2] == ('chunk', 'long')} == {2048}
    print('12 short-first: warm turn preempted a long cold prefill (which resumed and finished), >= 1 chunk between '
          'preemptions, not across priorities / over SHORT_MAX / with the flag off; small suffix gets full chunks')
    # park_auto: free VRAM beyond the margin / state size, clamped to [INK_PARK, 6]
    import engine as E_
    fe = type('FE', (), dict(parked=[], parked_bytes=lambda self: 0))()
    for free_mb, want in ((1000, E_.PARK), ((E_.MARGIN >> 20) + 4 * 200, 4), (20000, 6)):
        old_park, fe.mem_free = E_.PARK, (lambda f=free_mb: f << 20)
        assert server.park_auto(fe) == want, (free_mb, E_.PARK, want)
        E_.PARK = old_park
    print('12b INK_PARK_AUTO sizing')
    print('ALL OK')
