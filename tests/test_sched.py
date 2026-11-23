"""CPU, no model: priority queue + preemption through the real HTTP handlers with a fake step engine (one slot;
batching itself: test_batch_sched.py).
Run: cd engine && PATH=$PWD/.venv/bin:$PATH python tests/test_sched.py"""
import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from fastapi.testclient import TestClient

import server
from test_batch_sched import FakeEngine

server.eng = FakeEngine(slots=1)  # one slot: the old one-at-a-time scenarios, now with decode preemption
server.sched = server.Scheduler(server.eng)
LOG = server.eng.log
client = TestClient(server.app)
done = {}


def ask(name, chunks, toks, delay, **extra):
    time.sleep(delay)
    r = client.post('/v1/chat/completions', json=dict(messages=[{'role': 'user', 'content': f'{name}:{chunks}:{toks}'}],
                                                      **extra))
    assert r.status_code == 200, r.text
    assert r.json()['choices'][0]['message']['content'] == ''.join(f'{name}{i} ' for i in range(toks)), r.json()
    done[name] = time.time()


def scenario(reqs):
    LOG.clear(); done.clear()
    th = [threading.Thread(target=ask, args=a, kwargs=k) for a, k in reqs]
    for t in th: t.start()
    for t in th: t.join()
    return sorted(done, key=done.get)


def has(*e):
    return server.eng.has(*e)


def count(*e):
    return sum(x[:len(e)] == e for x in LOG)


# 1. gate/priority.py adapted to one slot: a blocker decoding, then spoof (prio 0) queued, then VIP (prio 10, sent the
#    LiteLLM way as extra_body) 0.1 s later: the VIP parks the blocker's decode and runs; the blocker resumes (full,
#    continuous output: ask() checks every token) before spoof (its place in line is kept).
order = scenario([(('blocker', 1, 20, 0), {}), (('spoof', 1, 2, 0.2), {'priority': 0}),
                  (('vip', 1, 2, 0.3), {'extra_body': {'priority': 10}})])
assert order == ['vip', 'blocker', 'spoof'], order
assert has('parked', 'blocker') and count('start', 'blocker') == 2
print('1 VIP parked the decoding blocker, which resumed before spoof:', order)

# 2. VIP arrives while a low-priority request is in prefill: low is paused between chunks, VIP runs, low restarts.
order = scenario([(('low', 40, 2, 0), {}), (('vip', 2, 2, 0.3), {'priority': 10})])
assert order == ['vip', 'low'] and has('preempted', 'low'), (order, LOG)
assert [e[:2] for e in LOG].index(('preempted', 'low')) < [e[:2] for e in LOG].index(('start', 'vip'))
assert count('start', 'low') == 2
print('2 prefill preempted:', [e[:2] for e in LOG if e[0] != 'chunk'])

# 3. FIFO within a level, higher level first: queued behind a blocker that nobody may preempt (priority 9)
order = scenario([(('blocker', 1, 40, 0), {'priority': 9}), (('a', 1, 1, 0.1), {'priority': 0}),
                  (('b', 1, 1, 0.4), {'priority': 5}), (('c', 1, 1, 0.7), {'priority': 0}),
                  (('d', 1, 1, 1.0), {'priority': 5})])
assert order == ['blocker', 'b', 'd', 'a', 'c'], order
print('3 levels + FIFO:', order)

# 4. equal priority never preempts (INK_SHORT_FIRST=0: strict FIFO prefills, the old rule) ...
server.SHORT_FIRST = False
order = scenario([(('low', 10, 1, 0), {'priority': 3}), (('same', 1, 1, 0.1), {'priority': 3})])
assert order == ['low', 'same'] and not has('preempted', 'low'), LOG
server.SHORT_FIRST = True
print('4 equal priority waits (short-first off)')
# 4b. ... except short-first (default): a short same-priority turn pauses a long prefill (parked), which then resumes
order = scenario([(('low', 10, 1, 0), {'priority': 3}), (('same', 1, 1, 0.1), {'priority': 3})])
assert order == ['same', 'low'] and has('preempted', 'low') and count('chunk', 'low') == 10, LOG
print('4b short-first: the short same-priority turn went first, the long prefill resumed')

# 5. KV pool short: requeued (not an error, not a preemption), then served
order = scenario([(('full1', 1, 1, 0), {})])
assert order == ['full1'] and count('start', 'full1') == 2, LOG
print('5 NoRoom requeued')

m = client.get('/metrics').text
assert 'inkling_requests_total 15' in m and 'inkling_preemptions_total 3' in m and 'inkling_queue_depth 0' in m, m
assert 'inkling_running 0' in m and 'inkling_batch_size 0' in m and 'inkling_slots 1' in m, m
assert 'inkling_parked_bytes 123' in m and 'inkling_ooms_total 0' in m, m
print('metrics ok:', ' '.join(l for l in m.splitlines() if not l.startswith('#'))[:200])
server.NOROOM_S = 1.5  # 6. never fits: a clear error after a bounded wait, not a retry loop forever
t0 = time.time()
r = client.post('/v1/chat/completions', json=dict(messages=[{'role': 'user', 'content': 'never:1:1'}]))
assert r.status_code == 500 and 'KV cache full' in r.text and time.time() - t0 < 10, (r.status_code, r.text)
print('6 NoRoom gives up')
r = client.post('/v1/chat/completions', json=dict(messages=[{'role': 'user', 'content': 'oom:1:1'}]))
assert r.status_code == 503 and 'out of GPU memory' in r.text, (r.status_code, r.text)
r = client.post('/v1/chat/completions', json=dict(messages=[{'role': 'user', 'content': 'oom:1:1'}], stream=True))
assert '"code": 503' in r.text and '[DONE]' in r.text, r.text
print('7 OOM -> 503 (plain + stream)')
r = client.post('/v1/chat/completions', json=dict(messages=[{'role': 'user', 'content': 'x:1:1'}], priority='hi'))
assert r.status_code == 400, r.text
print('ALL OK')
