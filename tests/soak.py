#!/usr/bin/env python3
"""Soak test: mixed, realistic traffic against a running engine (OpenAI API on :30000) for hours, with pass/fail.

  python tests/soak.py [--duration 2h] [--concurrency 6] [--base http://127.0.0.1:30000] [--out FILE]
  python tests/soak.py --smoke        # ~3 min, small prompts: does the harness + server path work at all
  python tests/soak.py --selftest     # CPU only: runs the harness against a built-in mock server (no engine, no GPU)

Stdlib only. Prompts are synthetic (random common words, generated PNGs); no user content anywhere.

TRAFFIC (shapes from the LiteLLM spend logs, counts/percentiles only, 2026-11-23: 20k qwen rows, 13% with images,
peak ~560 req/h; param frequencies from the API agent's audit in NOTES.md 2026-11-23):
- Each worker runs agent *sessions* back to back: a base context (prompt length drawn from the logged percentiles,
  p10 1.8k / p50 77k / p90 186k tokens, x --len-scale, capped at --max-prompt), then a multi-turn loop (geometric,
  mean 6 turns; an assumption, the logs have no session ids). Every turn resends the whole history plus the model's
  reply and a synthetic tool result / user message, so turns 2+ should hit the prefix cache. Half the sessions start
  with one of 3 shared system prompts (cross-session prefix, like a coding agent's fixed preamble).
- Per session: VIP priority 10 (20%), reasoning_effort (84%; low/medium/high/xhigh/minimal/none), tools (83%),
  stream (77%, include_usage on), an image in the first user turn (13%, 512 px .. 4096 px).
- Per request: max_completion_tokens vs max_tokens (~50/50, value from the logged completion percentiles), temperature
  (19%), response_format json_schema (2.2%) / json_object (0.5%), named tool_choice (1%), required (0.5%), none (0.2%),
  enable_thinking=false (2%), stop / seed / min_tokens / n=2+logprobs (0.5% each: real rates are ~0.1%, floored so a
  2 h run sees each), a deliberately unsupported param (0.3%, must get 400), and a client disconnect mid-stream (3% of
  streams).

PASS/FAIL THRESHOLDS (all must hold; the report lists each with its value):"""
THRESHOLDS = {
    'http_5xx': 0,                  # 500/503 responses, incl. in-stream error chunks after a 200
    'transport_errors': 0,          # refused / reset / timeout that we did not cause by disconnecting
    'unexpected_4xx_rate': 0.001,   # 4xx other than the deliberate bad-param probes, per request
    'probe_not_400': 0,             # a deliberate unsupported-param probe that did not get 400
    'check_failures': 0,            # bad json_schema JSON, forced tool call missing, stream without [DONE]/usage, n/logprobs shape
    'ooms': 0,                      # /metrics inkling_ooms_total delta over the run
    'server_restarts': 0,           # /metrics inkling_requests_total went backwards, or /health failed at the end
    'ttft_light_vip_p95_s': 5.0,    # TTFT of VIP requests with < 4096 uncached prompt tokens (VIP jumps the queue)
    'ttft_light_p50_s': 3.0,        # TTFT of all light requests (queueing behind big prefills is allowed in the tail)
    'decode_tps_p50': 30.0,         # per-stream completion tok/s after the first token (requests with >= 32 tokens)
    'followup_cache_hit_min': 0.6,  # sum(cached_tokens) / sum(prompt_tokens) over turns 2+ of sessions
    'rss_drift_gib': 1.0,           # server RSS, last idle point - first idle point after traffic
    'vram_drift_mib': 512,          # nvidia-smi memory.used, same idle points
    'vram_min_free_mib': 300,       # worst free VRAM seen (same floor as cutover.sh)
    'cache_dir_max_gb': 200.0,      # INK_CACHE_GB: the disk tier must stay under its cap
}
HIGHER = {'decode_tps_p50', 'followup_cache_hit_min', 'vram_min_free_mib'}  # value must be >= the limit
__doc__ += '\n' + '\n'.join(f'  {k} {">=" if k in HIGHER else "<="} {v}' for k, v in THRESHOLDS.items())

import argparse, base64, http.client, json, math, os, random, re, struct, subprocess, sys, threading, time, zlib
from urllib.parse import urlparse

P_IN = [(0, 200), (.1, 1771), (.5, 76573), (.75, 136066), (.9, 186182), (.99, 220837), (1, 270406)]   # prompt tokens
P_OUT = [(0, 16), (.1, 85), (.5, 313), (.75, 640), (.9, 1600), (.99, 7517), (1, 16000)]              # completion tokens
EFFORTS = ['low', 'medium', 'high', 'xhigh', 'minimal', 'none']
WORDS = ('the of and to in is that it for on was with as by at from this be are or have not but had they which you '
         'one all were we when there can an their more if will would about out so up into time some could them other '
         'than then now only its over also after use two how our work first well way even new because any these give '
         'day most us system data file value result table report river stone garden market window engine signal paper '
         'number letter yellow green orange purple silver copper ocean forest valley bridge castle planet rocket '
         'library kitchen station summer winter morning evening history science music theory method process').split()
TOOLS = [{'type': 'function', 'function': {'name': n, 'description': d, 'parameters': {
    'type': 'object', 'properties': p, 'required': list(p)}}} for n, d, p in [
    ('get_time', 'Current time in a timezone.', {'tz': {'type': 'string'}}),
    ('search_notes', 'Search the notes for a phrase.', {'query': {'type': 'string'}, 'limit': {'type': 'integer'}}),
    ('propose_memories', 'Propose facts worth remembering.', {'items': {'type': 'array', 'items': {'type': 'string'}}}),
    ('suggest_follow_ups', 'Suggest follow-up questions.', {'questions': {'type': 'array', 'items': {'type': 'string'}}})]]
SCHEMA = {'type': 'object', 'properties': {'title': {'type': 'string'}, 'score': {'type': 'integer'},
                                           'tags': {'type': 'array', 'items': {'type': 'string'}}},
          'required': ['title', 'score', 'tags']}
BAD_PROBES = [{'best_of': 2}, {'echo': True}, {'ignore_eos': True}, {'functions': [{'name': 'f', 'parameters': {}}]}]


def quantile(knots, u):
    """Log-linear interpolation between (quantile, value) knots."""
    for (q0, v0), (q1, v1) in zip(knots, knots[1:]):
        if u <= q1:
            return math.exp(math.log(v0) + (math.log(v1) - math.log(v0)) * (u - q0) / (q1 - q0))
    return knots[-1][1]


def text(rng, ntok):  # ~1.1 tokens per word for these common words
    return ' '.join(rng.choices(WORDS, k=max(1, int(ntok / 1.1))))


_png_cache = {}


def png_url(side, seed):
    """Blocky synthetic PNG (compresses well; the engine's cost is pixels, not bytes), cached per (side, seed)."""
    key = (side, seed % 4)
    if key not in _png_cache:
        r = random.Random(side * 10 + seed % 4)
        cols = [bytes(r.randrange(256) for _ in range(3)) for _ in range(16)]
        rows = []
        for y in range(side):
            if y % 64 == 0:
                line = b''.join(r.choice(cols) * 64 for _ in range(side // 64)) + cols[0] * (side % 64)
            rows.append(b'\0' + line)
        chunk = lambda t, d: struct.pack('>I', len(d)) + t + d + struct.pack('>I', zlib.crc32(t + d) & 0xffffffff)
        data = (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', side, side, 8, 2, 0, 0, 0))
                + chunk(b'IDAT', zlib.compress(b''.join(rows), 6)) + chunk(b'IEND', b''))
        _png_cache[key] = 'data:image/png;base64,' + base64.b64encode(data).decode()
    return _png_cache[key]


class Session:
    def __init__(self, rng, a, sid):
        self.rng, self.a, self.sid, self.turn = rng, a, sid, 0
        self.vip = rng.random() < .20
        self.tools = rng.random() < .83
        self.stream = rng.random() < .77
        self.effort = rng.choice(EFFORTS) if rng.random() < .84 else None
        self.turns = min(30, 1 + int(rng.expovariate(1 / 5)))
        base = min(a.max_prompt, max(64, int(quantile(P_IN, rng.random()) * a.len_scale)))
        msgs = []
        if rng.random() < .5:  # shared preamble: same text for every session that picks it
            k = rng.randrange(3)
            msgs.append({'role': 'system', 'content': text(random.Random(1000 + k), int((4000 + 8000 * k) * a.len_scale) + 32)})
        user = [{'type': 'text', 'text': f'Session {sid}. Notes:\n' + text(rng, base) +
                 '\nSummarise the notes above, then answer briefly.'}]
        self.image = rng.random() < .13
        if self.image:
            side = min(a.max_image_side, rng.choices([512, 1024, 1536, 2048, 4096], [.3, .3, .2, .15, .05])[0])
            user.insert(0, {'type': 'image_url', 'image_url': {'url': png_url(side, sid)}})
        msgs.append({'role': 'user', 'content': user})
        self.msgs = msgs

    def body(self):
        rng, a = self.rng, self.a
        b = {'model': 'inkling-2-mini', 'messages': self.msgs, 'stream': self.stream, 'store': False}
        tags = []
        mt = min(a.max_out, max(16, int(quantile(P_OUT, rng.random()))))
        b['max_completion_tokens' if rng.random() < .53 else 'max_tokens'] = mt
        if self.stream:
            b['stream_options'] = {'include_usage': True}
        if self.vip:
            b['priority'] = 10
        if self.effort:
            b['reasoning_effort'] = self.effort
        if self.tools:
            b['tools'] = TOOLS
        if rng.random() < .19:
            b['temperature'] = round(rng.uniform(0, 1), 2)
        u = rng.random()
        if u < .022:
            b['response_format'] = {'type': 'json_schema', 'json_schema': {'name': 'card', 'schema': SCHEMA}}; tags.append('json_schema')
        elif u < .027:
            b['response_format'] = {'type': 'json_object'}; tags.append('json_object')
        elif self.tools and u < .037:
            b['tool_choice'] = {'type': 'function', 'function': {'name': rng.choice(TOOLS)['function']['name']}}; tags.append('named_tool')
        elif self.tools and u < .042:
            b['tool_choice'] = 'required'; tags.append('required_tool')
        elif self.tools and u < .044:
            b['tool_choice'] = 'none'
        if rng.random() < .02:
            b['chat_template_kwargs'] = {'enable_thinking': False}
        for p, f in (('stop', lambda: ['\n\n\n', 'END_OF_ANSWER']), ('seed', lambda: rng.randrange(1 << 30)),
                     ('min_tokens', lambda: 8)):
            if rng.random() < .005:
                b[p] = f()
        if rng.random() < .005 and 'response_format' not in b and 'tool_choice' not in b:
            b.update(n=2, logprobs=True, top_logprobs=3); tags.append('n2_logprobs')
        if rng.random() < .003:
            b.update(rng.choice(BAD_PROBES)); tags.append('probe')
        cut = self.stream and 'probe' not in tags and rng.random() < .03
        return b, tags, (rng.randint(1, 40) if cut else None)

    def advance(self, res):
        """Append the model's reply (as returned) plus the next user/tool turn. False = session over."""
        self.turn += 1
        msg = res.get('message') or {}
        if self.turn >= self.turns or not res.get('ok') or res.get('cut') or 'n2_logprobs' in res['tags']:
            return False
        reply = {'role': 'assistant', 'content': msg.get('content') or ''}
        if msg.get('tool_calls'):
            reply['tool_calls'] = msg['tool_calls']
            self.msgs.append(reply)
            for tc in msg['tool_calls']:
                self.msgs.append({'role': 'tool', 'tool_call_id': tc.get('id', ''),
                                  'content': 'Result: ' + text(self.rng, self.rng.randint(100, 3000) * self.a.len_scale + 16)})
        else:
            self.msgs.append(reply)
            self.msgs.append({'role': 'user', 'content': 'Next: ' + text(self.rng, self.rng.randint(20, 1500) * self.a.len_scale + 8)})
        return approx_tokens(self.msgs) < self.a.max_prompt - self.a.max_out


def approx_tokens(msgs):
    n = 0
    for m in msgs:
        c = m.get('content')
        for part in (c if isinstance(c, list) else [{'text': c or ''}]):
            n += len((part.get('text') or '').split()) * 1.1 + (1000 if 'image_url' in part else 0)
    return n


def send(base, body, cut_after, timeout):
    """One request. Returns dict: status, ok, err, ttft, lat, ptok, ctok, cached, message, cut, checks failed."""
    u = urlparse(base)
    r = {'status': None, 'ok': False, 'err': None, 'ttft': None, 'cut': False, 'ptok': 0, 'ctok': 0, 'cached': 0}
    t0 = time.time()
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)
    try:
        conn.request('POST', '/v1/chat/completions', json.dumps(body), {'Content-Type': 'application/json'})
        resp = conn.getresponse()
        r['status'] = resp.status
        if resp.status != 200:
            r['err'] = resp.read(300).decode('utf-8', 'replace')
            return r
        if not body.get('stream'):
            out = json.loads(resp.read())
            r['lat'] = time.time() - t0
            r['choices'] = out.get('choices', [])
            r['message'] = r['choices'][0]['message'] if r['choices'] else {}
            r['finish'] = r['choices'][0].get('finish_reason') if r['choices'] else None
            usage(r, out.get('usage'))
            r['ok'] = True
            return r
        content, calls, chunks, done, use = [], [], 0, False, None
        while True:
            line = resp.readline()
            if not line:
                break
            if not line.startswith(b'data: '):
                continue
            if line.strip() == b'data: [DONE]':
                done = True; break
            d = json.loads(line[6:])
            if 'error' in d:
                r['status'] = int(d['error'].get('code') or 500); r['err'] = str(d['error'].get('message'))[:300]
                return r
            if d.get('usage'):
                use = d['usage']
            for c in d.get('choices', []):
                dl = c.get('delta') or {}
                if r['ttft'] is None and (dl.get('content') or dl.get('reasoning_content') or dl.get('tool_calls')):
                    r['ttft'] = time.time() - t0
                if c.get('index', 0) == 0:
                    content.append(dl.get('content') or '')
                    calls += dl.get('tool_calls') or []
                    r['finish'] = c.get('finish_reason') or r.get('finish')
            chunks += 1
            if cut_after and chunks >= cut_after:
                r.update(cut=True, ok=True, lat=time.time() - t0)
                return r                                     # finally: close mid-stream = client disconnect
        r['lat'] = time.time() - t0
        r['message'] = {'content': ''.join(content), 'tool_calls': [
            {k: v for k, v in tc.items() if k != 'index'} for tc in calls] or None}
        r['done'] = done
        usage(r, use)
        r['ok'] = done
        if not done:
            r['err'] = 'stream ended without [DONE]'
        return r
    except (OSError, http.client.HTTPException, ValueError) as e:
        r['err'] = f'{type(e).__name__}: {e}'[:300]
        r['lat'] = time.time() - t0
        return r
    finally:
        conn.close()


def usage(r, u):
    if u:
        r['ptok'], r['ctok'] = u.get('prompt_tokens', 0), u.get('completion_tokens', 0)
        r['cached'] = (u.get('prompt_tokens_details') or {}).get('cached_tokens', 0)
        r['has_usage'] = True


def checks(body, tags, r):
    """Functional checks on a successful, uncut response. Returns the list of failed check names."""
    bad = []
    if body.get('stream') and body.get('stream_options', {}).get('include_usage') and not r.get('has_usage'):
        bad.append('stream_usage_missing')
    m = r.get('message') or {}
    if 'json_schema' in tags and r.get('finish') == 'stop':
        try:
            j = json.loads(m.get('content') or '')
            if not all(k in j for k in SCHEMA['required']):
                bad.append('json_schema_keys')
        except ValueError:
            bad.append('json_schema_invalid')
    if 'json_object' in tags and r.get('finish') == 'stop':
        try:
            json.loads(m.get('content') or '')
        except ValueError:
            bad.append('json_object_invalid')
    if 'named_tool' in tags and r.get('finish') != 'length':
        want = body['tool_choice']['function']['name']
        if not any((tc.get('function') or {}).get('name') == want for tc in m.get('tool_calls') or []):
            bad.append('named_tool_missing')
    if 'required_tool' in tags and r.get('finish') != 'length' and not m.get('tool_calls'):
        bad.append('required_tool_missing')
    if 'n2_logprobs' in tags and not body.get('stream'):
        ch = r.get('choices') or []
        if len(ch) != 2 or any(c.get('logprobs') is None for c in ch):
            bad.append('n2_logprobs_shape')
    return bad


# ---------------------------------------------------------------- server-side samplers
def get(base, path, timeout=10):
    u = urlparse(base)
    c = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)
    try:
        c.request('GET', path)
        resp = c.getresponse()
        return resp.status, resp.read().decode()
    finally:
        c.close()


def metrics(base):
    try:
        st, body = get(base, '/metrics')
        return {k[8:]: float(v) for k, v in re.findall(r'^(inkling_\w+) (\S+)$', body, re.M)} if st == 200 else {}
    except (OSError, http.client.HTTPException, ValueError):
        return {}


def nvsmi(a):
    if a.no_gpu:
        return None
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=10).stdout.split('\n')[0]
        used, total = (int(x) for x in out.split(','))
        return {'used': used, 'free': total - used}
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def server_pid(a):
    """--pid, else the GPU process whose cmdline mentions server.py (host pid, also for a container), else pgrep."""
    if a.pid:
        return a.pid
    cands = []
    if not a.no_gpu:
        try:
            cands = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                                   capture_output=True, text=True, timeout=10).stdout.split()
        except (OSError, subprocess.SubprocessError):
            pass
    cands += subprocess.run(['pgrep', '-f', 'server.py'], capture_output=True, text=True).stdout.split()
    for p in cands:
        try:
            if b'server.py' in open(f'/proc/{p}/cmdline', 'rb').read():
                return int(p)
        except OSError:
            pass
    return None


def rss_mib(pid):
    try:
        for line in open(f'/proc/{pid}/status'):
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) / 1024
    except (OSError, TypeError):
        return None


def dir_gb(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total / 1e9


def pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


# ---------------------------------------------------------------- the run
class Soak:
    def __init__(self, a):
        self.a, self.recs, self.samples, self.idles = a, [], [], []
        self.lock, self.stop, self.paused = threading.Lock(), threading.Event(), threading.Event()
        self.inflight, self.restarts, self.sid, self.harness_errors, self.pid = 0, 0, 0, 0, server_pid(a)
        self.t0 = time.time()

    def worker(self, wid):
        rng = random.Random(self.a.seed * 1000 + wid)
        while not self.stop.is_set():
            try:
                self.session(rng)
            except Exception:  # a harness bug must not silently lower the concurrency
                import traceback; traceback.print_exc()
                with self.lock:
                    self.harness_errors += 1

    def session(self, rng):
        with self.lock:
            self.sid += 1; sid = self.sid
        s = Session(rng, self.a, sid)
        while not self.stop.is_set():
            while self.paused.is_set() and not self.stop.is_set():
                time.sleep(.2)
            if self.stop.is_set():
                return
            body, tags, cut = s.body()
            with self.lock:
                self.inflight += 1
            t = time.time() - self.t0
            r = send(self.a.base, body, cut, self.a.timeout)
            with self.lock:
                self.inflight -= 1
            r['tags'] = tags
            rec = {'t': round(t, 2), 'sid': sid, 'turn': s.turn, 'vip': s.vip, 'stream': s.stream, 'tags': tags,
                   'status': r['status'], 'err': r['err'], 'cut': r['cut'], 'ttft': r['ttft'], 'lat': r.get('lat'),
                   'ptok': r['ptok'], 'ctok': r['ctok'], 'cached': r['cached'], 'image': s.image and s.turn == 0, 'checks': []}
            if r['ok'] and not r['cut'] and 'probe' not in tags:
                rec['checks'] = checks(body, tags, r)
            if r['ttft'] is not None and r.get('lat') and r['ctok'] >= 32 and r['lat'] - r['ttft'] > .05:
                rec['tps'] = round(r['ctok'] / max(1e-3, r['lat'] - r['ttft']), 1)
            with self.lock:
                self.recs.append(rec)
            if 'probe' in tags or not s.advance(r):
                break
            time.sleep(rng.expovariate(1 / self.a.think_s))

    def sample(self):
        m, g = metrics(self.a.base), nvsmi(self.a)
        prev = next((x['m'] for x in reversed(self.samples) if x['m']), None)
        if prev and m and m.get('requests_total', 0) < prev.get('requests_total', 0):
            self.restarts += 1
            self.pid = server_pid(self.a)
        x = {'t': round(time.time() - self.t0, 1), 'm': m, 'gpu': g, 'rss_mib': rss_mib(self.pid), 'inflight': self.inflight}
        self.samples.append(x)
        return x

    def idle_point(self, label):
        """Stop new requests, let in-flight ones and the server drain, settle, then sample RSS/VRAM/cache."""
        self.paused.set()
        t_end = time.time() + self.a.drain_s
        while self.inflight and time.time() < t_end:
            time.sleep(1)
        while time.time() < t_end:
            m = metrics(self.a.base)
            if not m or (m.get('running', 0) == 0 and m.get('queue_depth', 0) == 0):
                break
            time.sleep(2)
        time.sleep(self.a.settle_s)
        x = self.sample()
        x.update(label=label, cache_gb=round(dir_gb(self.a.cache_dir), 3), drained=self.inflight == 0)
        self.idles.append(x)
        print(f'[{x["t"]:7.0f}s] idle {label}: rss {x["rss_mib"]} MiB, vram used {(x["gpu"] or {}).get("used")} MiB, '
              f'kv_free_pages {x["m"].get("kv_free_pages")}, parked {x["m"].get("parked")}, cache {x["cache_gb"]} GB',
              flush=True)
        self.paused.clear()

    def run(self):
        a = self.a
        print(f'soak: {a.base} for {a.duration} s, concurrency {a.concurrency}, len-scale {a.len_scale}, '
              f'server pid {self.pid}, cache dir {a.cache_dir}', flush=True)
        self.idle_point('start')
        threads = [threading.Thread(target=self.worker, args=(i,), daemon=True) for i in range(a.concurrency)]
        for t in threads:
            t.start()
        end, next_idle, next_sample, next_log = time.time() + a.duration, time.time() + a.idle_every, 0, time.time() + a.log_s
        while time.time() < end:
            now = time.time()
            if now >= next_idle and end - now > a.idle_every / 2:
                self.idle_point(f'{(now - self.t0) / 60:.0f}min'); next_idle = time.time() + a.idle_every
            if now >= next_sample:
                x = self.sample(); next_sample = now + a.sample_s
            if now >= next_log:
                with self.lock:
                    n, e = len(self.recs), sum(1 for r in self.recs if not ok_rec(r))
                print(f'[{now - self.t0:7.0f}s] requests {n}, not-ok {e}, inflight {self.inflight}, '
                      f'vram used {(x["gpu"] or {}).get("used")} MiB, ooms {x["m"].get("ooms_total")}', flush=True)
                next_log = now + a.log_s
            time.sleep(.5)
        self.stop.set()
        for t in threads:
            t.join(a.timeout + 5)
        self.idle_point('end')
        try:
            st, body = get(a.base, '/health')
            healthy = st == 200 and json.loads(body).get('ok')
        except (OSError, http.client.HTTPException, ValueError):
            healthy = False
        return self.report(healthy)

    def report(self, healthy):
        a, R = self.a, self.recs
        is5 = lambda r: (r['status'] or 0) >= 500
        probes = [r for r in R if 'probe' in r['tags']]
        normal = [r for r in R if 'probe' not in r['tags']]
        transport = [r for r in normal if r['status'] is None or (r['status'] == 200 and r['err'] and not r['cut'])]
        u4 = [r for r in normal if r['status'] and 400 <= r['status'] < 500]
        good = [r for r in normal if r['status'] == 200 and not r['err']]
        light = [r for r in good if r['ttft'] is not None and r['ptok'] - r['cached'] < 4096]
        heavy = [r for r in good if r['ttft'] is not None and r['ptok'] - r['cached'] >= 4096]
        fol = [r for r in good if r['turn'] > 0 and not r['cut']]
        m0 = next((x['m'] for x in self.samples if x['m']), {})
        m1 = next((x['m'] for x in reversed(self.samples) if x['m']), {})
        dm = lambda k: m1.get(k, 0) - m0.get(k, 0)
        idl = self.idles[1:] if len(self.idles) > 2 else self.idles  # drift from the first idle after traffic
        drift = lambda f: (f(idl[-1]) - f(idl[0])) if len(idl) >= 2 and f(idl[0]) is not None and f(idl[-1]) is not None else None
        frees = [x['gpu']['free'] for x in self.samples if x.get('gpu')]
        val = {
            'http_5xx': sum(map(is5, R)),
            'transport_errors': len(transport),
            'unexpected_4xx_rate': round(len(u4) / max(1, len(normal)), 5),
            'probe_not_400': sum(1 for r in probes if r['status'] != 400),
            'check_failures': sum(len(r['checks']) for r in R),
            'ooms': dm('ooms_total') if 'ooms_total' in m1 else None,
            'server_restarts': self.restarts + (0 if healthy else 1),
            'ttft_light_vip_p95_s': pct([r['ttft'] for r in light if r['vip']], .95),
            'ttft_light_p50_s': pct([r['ttft'] for r in light], .5),
            'decode_tps_p50': pct([r.get('tps') for r in good], .5),
            'followup_cache_hit_min': round(sum(r['cached'] for r in fol) / sum(r['ptok'] for r in fol), 3)
            if sum(r['ptok'] for r in fol) else None,
            'rss_drift_gib': None if drift(lambda x: x['rss_mib']) is None else round(drift(lambda x: x['rss_mib']) / 1024, 3),
            'vram_drift_mib': drift(lambda x: (x['gpu'] or {}).get('used')),
            'vram_min_free_mib': min(frees) if frees else None,
            'cache_dir_max_gb': max(x['cache_gb'] for x in self.idles),
        }
        verdict = {}
        for k, lim in THRESHOLDS.items():
            v = val[k]
            verdict[k] = None if v is None else (v >= lim if k in HIGHER else v <= lim)
        hours = max(1e-9, (time.time() - self.t0) / 3600)
        rep = {
            'args': {k: v for k, v in vars(a).items()}, 'thresholds': THRESHOLDS, 'values': val, 'verdict': verdict,
            'pass': all(v is not False for v in verdict.values()) and bool(good) and not self.harness_errors,
            'harness_errors': self.harness_errors,
            'counts': {'requests': len(R), 'ok': len(good), 'cut': sum(r['cut'] for r in R), 'probes': len(probes),
                       'sessions': len({r['sid'] for r in R}), 'images': sum(1 for r in R if r['image']),
                       'by_status': {str(s): sum(1 for r in R if r['status'] == s) for s in {r['status'] for r in R}},
                       'by_tag': {t: sum(1 for r in R if t in r['tags']) for t in {t for r in R for t in r['tags']}},
                       'check_fail_kinds': {c: sum(1 for r in R if c in r['checks']) for c in {c for r in R for c in r['checks']}}},
            'latency': {'ttft_light': [pct([r['ttft'] for r in light], q) for q in (.5, .95, .99)],
                        'ttft_light_vip': [pct([r['ttft'] for r in light if r['vip']], q) for q in (.5, .95, .99)],
                        'ttft_heavy': [pct([r['ttft'] for r in heavy], q) for q in (.5, .95, .99)],
                        'decode_tps': [pct([r.get('tps') for r in good], q) for q in (.1, .5, .9)],
                        'prompt_tokens': [pct([r['ptok'] for r in good], q) for q in (.1, .5, .9, 1)],
                        'completion_tokens': [pct([r['ctok'] for r in good], q) for q in (.1, .5, .9, 1)]},
            'server_metrics_delta': {k: dm(k) for k in ('requests_total', 'prompt_tokens_total', 'cached_tokens_total',
                                                         'completion_tokens_total', 'preemptions_total', 'ooms_total')},
            'server_cache_hit': round(dm('cached_tokens_total') / dm('prompt_tokens_total'), 3) if dm('prompt_tokens_total') else None,
            'cache_dir_growth_gb_per_h': round((self.idles[-1]['cache_gb'] - self.idles[0]['cache_gb']) / hours, 3),
            'idles': self.idles, 'errors': [r for r in R if r['err'] and not r['cut']][:200],
            'samples': self.samples, 'requests': R,
        }
        return rep


def ok_rec(r):
    return r['status'] == 200 and not r['err'] or 'probe' in r['tags'] and r['status'] == 400


def summarize(rep):
    print('\n== soak summary')
    print(json.dumps(rep['counts']), 'harness errors', rep['harness_errors'])
    for k, v in rep['latency'].items():
        print(f'  {k}: {v}')
    if rep['values']['rss_drift_gib'] is None:
        print('  WARNING: server pid not found (pass --pid): RSS leak check skipped')
    print(f"  server cache hit {rep['server_cache_hit']}, cache dir growth {rep['cache_dir_growth_gb_per_h']} GB/h")
    for k, lim in THRESHOLDS.items():
        v, ok = rep['values'][k], rep['verdict'][k]
        print(f"  {'PASS' if ok else ('n/a ' if ok is None else 'FAIL')} {k}: {v} (limit {lim})")
    print('SOAK PASSED' if rep['pass'] else 'SOAK FAILED', flush=True)


# ---------------------------------------------------------------- mock server (for --selftest; CPU only)
def mock_server(fail_rate=0.0, leak_kib=0, port=0):
    """OpenAI-ish stub on a free port: streams, usage + cached_tokens (prefix hashes), tool calls, JSON, 400 probes,
    /metrics in the engine's names. fail_rate injects 500s; leak_kib grows this process's RSS per request."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import hashlib
    st = {'requests': 0, 'prompt': 0, 'cached': 0, 'seen': set(), 'leak': [], 'lock': threading.Lock()}
    rng = random.Random(7)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, code, obj, ctype='application/json'):
            data = obj.encode() if isinstance(obj, str) else json.dumps(obj).encode()
            self.send_response(code); self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(data))); self.end_headers(); self.wfile.write(data)

        def do_GET(self):
            if self.path == '/health':
                return self.send(200, {'ok': True})
            if self.path == '/metrics':
                return self.send(200, ''.join(f'inkling_{k} {v}\n' for k, v in [
                    ('requests_total', st['requests']), ('prompt_tokens_total', st['prompt']),
                    ('cached_tokens_total', st['cached']), ('ooms_total', 0), ('running', 0), ('queue_depth', 0),
                    ('kv_free_pages', 4000), ('parked', 2)]), 'text/plain')
            self.send(404, {})

        def do_POST(self):
            b = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if any(k in b for k in ('best_of', 'echo', 'ignore_eos', 'functions')):
                return self.send(400, {'error': {'message': 'unsupported'}})
            if rng.random() < fail_rate:
                return self.send(500, {'error': {'message': 'injected'}})
            msgs = b['messages']
            sizes = [len(json.dumps(m)) // 4 for m in msgs]
            hs = [hashlib.sha1(json.dumps(msgs[:k + 1]).encode()).hexdigest() for k in range(len(msgs))]
            with st['lock']:
                cached = max([sum(sizes[:k + 1]) for k in range(len(msgs) - 1) if hs[k] in st['seen']] or [0])
                st['seen'].update(hs); st['requests'] += 1; st['prompt'] += sum(sizes); st['cached'] += cached
                st['leak'].append(b'x' * (leak_kib << 10))
            use = {'prompt_tokens': sum(sizes), 'completion_tokens': 40, 'total_tokens': sum(sizes) + 40,
                   'prompt_tokens_details': {'cached_tokens': cached}}
            tc, content = None, ' '.join(rng.choices(WORDS, k=40))
            rf, tch = b.get('response_format') or {}, b.get('tool_choice')
            if rf.get('type') in ('json_schema', 'json_object'):
                content = json.dumps({'title': 'x', 'score': 1, 'tags': []})
            elif isinstance(tch, dict) or tch == 'required' or (b.get('tools') and tch != 'none' and rng.random() < .3):
                name = tch['function']['name'] if isinstance(tch, dict) else 'get_time'
                tc, content = [{'id': 'call_1', 'type': 'function', 'function': {'name': name, 'arguments': '{}'}}], ''
            n = b.get('n') or 1
            lp = {'content': []} if b.get('logprobs') else None
            if not b.get('stream'):
                msg = {'role': 'assistant', 'content': content, 'tool_calls': tc}
                return self.send(200, {'choices': [{'index': i, 'message': msg, 'logprobs': lp,
                                                    'finish_reason': 'tool_calls' if tc else 'stop'} for i in range(n)], 'usage': use})
            self.send_response(200); self.send_header('Content-Type', 'text/event-stream'); self.end_headers()
            ev = lambda d: self.wfile.write(b'data: ' + json.dumps(d).encode() + b'\n\n')
            try:
                ev({'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ''}}]})
                for w in content.split(' ') if content else []:
                    ev({'choices': [{'index': 0, 'delta': {'content': w + ' '}}]}); self.wfile.flush(); time.sleep(.002)
                if tc:
                    ev({'choices': [{'index': 0, 'delta': {'tool_calls': [dict(tc[0], index=0)]}}]})
                ev({'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'tool_calls' if tc else 'stop'}]})
                if (b.get('stream_options') or {}).get('include_usage'):
                    ev({'choices': [], 'usage': use})
                self.wfile.write(b'data: [DONE]\n\n')
            except (BrokenPipeError, ConnectionResetError):
                pass

    srv = ThreadingHTTPServer(('127.0.0.1', port), H)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, st


def selftest(a):
    import tempfile
    base_args = ['--no-gpu', '--duration', '20', '--concurrency', '4', '--len-scale', '0.01', '--idle-every', '7',
                 '--settle-s', '0.2', '--drain-s', '10', '--think-s', '0.01', '--sample-s', '1', '--log-s', '10',
                 '--max-image-side', '256', '--pid', str(os.getpid())]
    with tempfile.TemporaryDirectory() as td:
        open(os.path.join(td, 'f'), 'wb').write(b'x' * 1000)
        srv, st = mock_server()
        args = parse(base_args + ['--base', f'http://127.0.0.1:{srv.server_port}', '--cache-dir', td,
                                  '--out', os.path.join(td, 'r.json')])
        rep = Soak(args).run(); summarize(rep); srv.shutdown()
        c = rep['counts']
        assert rep['pass'], rep['verdict']
        assert c['requests'] > 200 and c['sessions'] > 20, c
        assert rep['values']['followup_cache_hit_min'] > .6, rep['values']   # session turns reuse the prefix
        assert len(rep['idles']) >= 3 and all(x['drained'] for x in rep['idles']), rep['idles']
        assert rep['values']['http_5xx'] == 0 and rep['values']['probe_not_400'] == 0
        assert c['cut'] > 0 and c['images'] > 0, c
        # the same run with injected 500s and a leaking server must FAIL on exactly those thresholds
        srv, st = mock_server(fail_rate=.05, leak_kib=256)   # ~0.1 GB over the run (shared host: keep it small)
        THRESHOLDS['rss_drift_gib'] = 0.03
        args = parse(base_args + ['--base', f'http://127.0.0.1:{srv.server_port}', '--cache-dir', td, '--seed', '2'])
        rep2 = Soak(args).run(); srv.shutdown()
        bad = {k for k, v in rep2['verdict'].items() if v is False}
        assert {'http_5xx', 'rss_drift_gib'} <= bad, rep2['verdict']
        assert not rep2['pass']
        # a session's history never exceeds --max-prompt, and checks catch a broken json_schema reply
        assert checks({'stream': False}, ['json_schema'], {'finish': 'stop', 'message': {'content': '{"title": 1}'}}) == ['json_schema_keys']
        assert checks({'stream': False, 'tool_choice': {'function': {'name': 'get_time'}}}, ['named_tool'],
                      {'finish': 'stop', 'message': {}}) == ['named_tool_missing']
        assert abs(quantile(P_IN, .5) - 76573) < 1 and abs(quantile(P_OUT, 0) - 16) < 1e-6
    print('SELFTEST OK: mock run PASS (cache reuse, disconnects, idles, probes), 500s + leak -> FAIL on', sorted(bad))


def dur(s):
    m = re.fullmatch(r'([\d.]+)([smh]?)', str(s))
    return float(m[1]) * {'': 1, 's': 1, 'm': 60, 'h': 3600}[m[2]]


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--base', default=os.environ.get('INK_BASE', 'http://127.0.0.1:30000'))
    p.add_argument('--duration', type=dur, default='2h')
    p.add_argument('--concurrency', type=int, default=6)
    p.add_argument('--len-scale', type=float, default=1.0, help='multiply prompt lengths (1 = logged distribution)')
    p.add_argument('--max-prompt', type=int, default=262144, help='cap on a session history, tokens')
    p.add_argument('--max-out', type=int, default=4096, help='cap on max_tokens')
    p.add_argument('--max-image-side', type=int, default=4096)
    p.add_argument('--think-s', type=float, default=2.0, help='mean client think time between turns')
    p.add_argument('--idle-every', type=dur, default='30m', help='pause traffic this often for an idle sample')
    p.add_argument('--drain-s', type=float, default=900, help='max wait for in-flight requests at an idle point')
    p.add_argument('--settle-s', type=float, default=20)
    p.add_argument('--sample-s', type=float, default=10)
    p.add_argument('--log-s', type=float, default=60)
    p.add_argument('--timeout', type=float, default=1800)
    p.add_argument('--pid', type=int, help='server pid for RSS (default: auto from nvidia-smi / pgrep)')
    p.add_argument('--cache-dir', default=os.environ.get('INK_CACHE_DIR', os.path.expanduser('~/llm/inkcache')))
    p.add_argument('--no-gpu', action='store_true', help='skip nvidia-smi sampling')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--out', default=os.path.expanduser(f'~/llm/logs/ink-soak-{time.strftime("%Y%m%d-%H%M")}.json'))
    p.add_argument('--smoke', action='store_true', help='3 min, concurrency 3, prompts x0.02, idle every 1 min')
    p.add_argument('--selftest', action='store_true', help='CPU: run against the built-in mock server')
    a = p.parse_args(argv)
    if a.smoke:
        a.duration, a.concurrency, a.len_scale, a.idle_every, a.settle_s = 180, 3, .02, 60, 5
        a.max_out, a.max_image_side, a.think_s = 256, 1024, .5
    a.duration, a.idle_every = dur(a.duration), dur(a.idle_every)
    return a


def main():
    a = parse()
    if a.selftest:
        return selftest(a)
    rep = Soak(a).run()
    summarize(rep)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w') as f:
        json.dump(rep, f)
    print(f'report: {a.out}')
    sys.exit(0 if rep['pass'] else 1)


if __name__ == '__main__':
    main()
