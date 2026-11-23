"""CPU: server.py turn-latency timings (body `timings: true`, /metrics counters, _instrument attribution). ~5 s."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fastapi.testclient import TestClient  # noqa: E402

import server  # noqa: E402
from test_batch_sched import FakeEngine  # noqa: E402


class Pool:  # the kvpool methods _instrument wraps; restore hits disk once with 2 pages, then saves (a spill)
    nodes = {}
    free_pages = staticmethod(lambda: 1)

    def restore(self, keys, min_len=0):
        self._load_page(1)
        self._load_page(2)
        self._save('x')
        return ('tbl', 512, {})

    def _load_page(self, i):
        return i

    def _save(self, p):
        return True


class Fake(FakeEngine):
    def request(self, messages, **kw):
        return super().request([{'content': 'ok:1:2'}], **{k: kw[k] for k in ('stats', 'emit', 'prio', 'cancel')})

    def start(self, r):
        super().start(r)
        self.restore()
        r.stats['cached_tokens'] = 512

    def restore(self):
        return self.pool.restore(None)


eng = Fake(chunk_s=0, step_s=0)
eng.pool = Pool()
server.eng, server.sched = eng, server.Scheduler(eng)
c = TestClient(server.app)
body = dict(messages=[{'role': 'user', 'content': 'hi'}])
r = c.post('/v1/chat/completions', json=dict(body, timings=True)).json()
t = r['timings'][0]
assert t['src'] == 'disk+pages' and t['disk_pages'] == 2 and t['disk_state'] == 1 and t['sync_writes'] == 1, t
for k in ('tokenize_s', 'queue_s', 'start_s', 'restore_s', 'prefill_chunk_s', 'first_token_s', 'ttft_s', 'step_max_s',
          'disk_lookup_s'):
    assert isinstance(t.get(k), (int, float)) and t[k] >= 0, (k, t)
assert t['ttft_s'] >= t['queue_s'] and 'arrive' not in t and 'submit' not in t, t
assert 'timings' not in c.post('/v1/chat/completions', json=body).json()
s = c.post('/v1/chat/completions', json=dict(body, timings=True, stream=True, stream_options={'include_usage': True})).text
last = [json.loads(x[6:]) for x in s.split('\n\n') if x.startswith('data: {')][-1]
assert last['usage'] and last['timings'][0]['src'] == 'disk+pages', last
m = c.get('/metrics').text
for k in ('restore_s_total', 'disk_pages_total 6', 'sync_writes_total 3', 'queue_s_total', 'last_ttft_s', 'tokenize_s_total'):
    assert f'inkling_{k}' in m, (k, m)
server._instrument(eng)  # idempotent: no double wrapping
assert eng.pool.restore(None)[1] == 512 and server.M['disk_pages'] == 8, server.M['disk_pages']
print('TIMINGS OK')
