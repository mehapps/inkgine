#!/usr/bin/env python3
"""Agent-loop turn latency: long agent sessions against a running engine (OpenAI API, default :30000).

  python tests/gpu_agentloop.py [--start-server] [--ctx 32k,128k,256k,1m] [--turns 8] [--multi 4] [--out FILE]
  python tests/gpu_agentloop.py --smoke         # ~2-3 min: one 16k session (GPU, disk-state, disk-pages turns) + 2x8k multi
  python tests/gpu_agentloop.py --selftest      # CPU only: the harness against a built-in stub server (~20 s)
  GPU window: ~/llm/engine/swap_test.sh 45 ~/llm/engine/.venv/bin/python \
      ~/llm/engine/tests/gpu_agentloop.py --start-server          (~25-35 min incl. server start)

A turn N+1 = turn N's prompt (cached: parked on the GPU or saved on disk) + its reply + a new tool output. Per session:
  cold turn 0 (base context: synthetic repo dump = code, prose, logs, tool JSON; unique per run), then --turns warm turns,
  tool outputs cycling 300 / 2k / 8k tokens (turn 1 = 300, turn 2 = 2k, ...) -> phase gpu (restored from the GPU park);
  then INK_PARK + 1 short non-chat evictor prompts push the session's parked state to disk and one 2k-tool turn runs
  -> phase disk (state from disk, pages still on the GPU); then one evictor reserves the whole KV pool (small prompt,
  max_tokens = context - 2k, cancelled after its first token), which unparks everything and evicts every cached page
  (chat pages are spilled to disk), and one 2k-tool turn runs -> phase disk+pages (state and pages from disk).
Multi-session mode (--multi N): N sessions (contexts --multi-ctx) run concurrently, staggered by --stagger s, with a
  client think time between turns, so parked states spill (INK_PARK=2 < N) and, when the contexts add up past the pool,
  pages spill too; a cold first turn of one session competes with the warm turns of the others.
Per turn: prompt / cached tokens (cache hit = cached >= 98% of the previous turn's prompt), client TTFT, server timings
  (body `timings: true`, server.py: src cold|gpu|disk|disk+pages, queue_s, restore_s, prefill_chunk_s, sync_write_s,
  disk_pages, step_max_s, tokenize_s, solo), decode tok/s after the first token. Prints a table + PASS/FAIL per target;
  JSON report to --out. Thinking off, temperature 0, --max-tokens 128 per reply (decode speed is measured, not stressed).
TARGETS (client TTFT, median of the matching turns; see NOTES.md 2026-11-23 agent-loop analysis for the byte math):
  gpu 256k 2k-tool <= 1.5 s | disk (state) 256k 2k <= 2.0 s | disk+pages 256k 2k <= 3.0 s (expected FAIL until the
  parallel/pinned page read fix: ~4-6 s today) | gpu 1M 2k <= 5.0 s | gpu 128k 2k <= 1.0 s | gpu 32k 2k <= 0.6 s;
  every warm turn a cache hit; GPU restore_s <= 0.15 s (zero-copy fork); multi: warm-turn TTFT p50 <= 3 s; 0 errors.
Stdlib only (the tokenizer, if importable, only calibrates text length)."""
import argparse, http.client, json, os, random, signal, statistics, subprocess, sys, threading, time
from urllib.parse import urlparse

SIZES = (300, 2000, 8000)
CTX = {'8k': 8000, '16k': 16000, '32k': 32000, '128k': 128000, '256k': 256000, '1m': 960000}  # 1m: room for 8 turns
TARGETS = [  # (session, src, tool tokens, TTFT limit s)
    ('256k', 'gpu', 2000, 1.5), ('256k', 'disk', 2000, 2.0), ('256k', 'disk+pages', 2000, 3.0),
    ('1m', 'gpu', 2000, 5.0), ('128k', 'gpu', 2000, 1.0), ('32k', 'gpu', 2000, 0.6), ('16k', 'gpu', 2000, 1.0)]
TOOLS = [{'type': 'function', 'function': {'name': n, 'description': d, 'parameters': {
    'type': 'object', 'properties': p, 'required': list(p)}}} for n, d, p in [
    ('read_file', 'Read a file from the repository.', {'path': {'type': 'string'}}),
    ('run_tests', 'Run the test suite, optionally filtered.', {'filter': {'type': 'string'}}),
    ('search_code', 'Search the repository for a pattern.', {'pattern': {'type': 'string'}, 'limit': {'type': 'integer'}})]]
WORDS = ('the of and to in is that it for on was with as by at from this be are or have not but had they which you one '
         'all were we when there can an their more if will would about out so up into time some could them other than '
         'then now only its over also after use two how our work first well way even new because any these give day '
         'system data file value result table report signal buffer cache page token request session engine queue').split()
IDS = ('cfg', 'buf', 'row', 'col', 'idx', 'key', 'val', 'req', 'resp', 'node', 'page', 'tbl', 'pos', 'seq', 'ctx', 'out')


class Gen:
    """Deterministic synthetic agent text (prose, Python, logs, tool JSON) sized in tokens (tokens/char calibrated with
    the model tokenizer when importable, else ~0.27)."""

    def __init__(self, seed, tpc=None):
        self.r = random.Random(seed)
        self.tpc = tpc or self.calibrate()

    def calibrate(self):
        try:
            from tokenizers import Tokenizer
            t = Tokenizer.from_file(os.path.join(os.environ.get('INK_MODEL', os.path.expanduser(
                '~/llm/models/Inkling-2-Mini-Uncensored')), 'tokenizer.json'))
            s = Gen(1, tpc=1).mix(60000)
            return len(t.encode(s, add_special_tokens=False).ids) / len(s)
        except Exception:
            return 0.27

    def prose(self, n):
        return ' '.join(self.r.choices(WORDS, k=n)).capitalize() + '.\n'

    def code(self):
        a, b = self.r.sample(IDS, 2)
        f = f'{self.r.choice(IDS)}_{self.r.randrange(10000)}'
        return (f'def {f}({a}, {b}=None):\n    """{self.prose(8).strip()}"""\n    if {b} is None:\n        {b} = {{}}\n'
                f'    for i, x in enumerate({a}):\n        {b}[x] = i * {self.r.randrange(97)} + len({b})\n    return {b}\n\n')

    def log(self):
        return (f'2026-11-22T{self.r.randrange(24):02d}:{self.r.randrange(60):02d}:{self.r.randrange(60):02d} '
                f'{self.r.choice(["INFO", "WARN", "DEBUG", "ERROR"])} {self.r.choice(IDS)}.{self.r.choice(IDS)}: '
                f'{self.prose(6).strip()} ({self.r.randrange(1000)} ms)\n')

    def js(self):
        return json.dumps({'path': f'src/{self.r.choice(IDS)}/{self.r.choice(IDS)}.py', 'line': self.r.randrange(2000),
                           'match': self.prose(7).strip(), 'score': round(self.r.random(), 3)}) + '\n'

    def mix(self, ntok, kinds=('code', 'prose', 'log', 'js')):
        out, n, want = [], 0, int(ntok / self.tpc)
        while n < want:
            k = self.r.choice(kinds)
            s = (f'\n=== src/{self.r.choice(IDS)}/{self.r.choice(IDS)}_{self.r.randrange(99)}.py ===\n' + self.code()
                 if k == 'code' else self.prose(self.r.randrange(20, 80)) if k == 'prose' else getattr(self, k)())
            out.append(s)
            n += len(s)
        return ''.join(out)[:want]


# ---------------------------------------------------------------- HTTP
def post(base, body, timeout=3600, cut=False):
    """Streamed request -> dict(ok, status, err, ttft, lat, dec_tps, content, tool_calls, usage, timings)."""
    u = urlparse(base)
    r = {'ok': False, 'status': None, 'err': None, 'ttft': None, 'usage': {}, 'timings': {}, 'content': '',
         'tool_calls': []}
    t0 = time.time()
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)
    try:
        conn.request('POST', '/v1/chat/completions', json.dumps(body), {'Content-Type': 'application/json'})
        resp = conn.getresponse()
        r['status'] = resp.status
        if resp.status != 200:
            r['err'] = resp.read(300).decode('utf-8', 'replace')
            return r
        content, calls, tlast = [], [], None
        while True:
            line = resp.readline()
            if not line:
                break
            if not line.startswith(b'data: '):
                continue
            if line.strip() == b'data: [DONE]':
                r['ok'] = True
                break
            d = json.loads(line[6:])
            if 'error' in d:
                r['err'] = str(d['error'])[:300]
                return r
            if d.get('usage'):
                r['usage'], r['timings'] = d['usage'], (d.get('timings') or [{}])[0]
            for c in d.get('choices', []):
                dl = c.get('delta') or {}
                if dl.get('content') or dl.get('reasoning_content') or dl.get('tool_calls'):
                    tlast = time.time()
                    if r['ttft'] is None:
                        r['ttft'] = tlast - t0
                        if cut:
                            r['ok'] = True
                            return r                                   # finally closes: the server cancels it
                content.append(dl.get('content') or '')
                calls += dl.get('tool_calls') or []
        r['lat'] = time.time() - t0
        r['content'] = ''.join(content)
        r['tool_calls'] = [{k: v for k, v in tc.items() if k != 'index'} for tc in calls]
        n = r['usage'].get('completion_tokens', 0)
        if r['ttft'] is not None and tlast and n > 1 and tlast > t0 + r['ttft']:
            r['dec_tps'] = (n - 1) / (tlast - t0 - r['ttft'])
        if not r['ok']:
            r['err'] = 'stream ended without [DONE]'
        return r
    except (OSError, http.client.HTTPException, ValueError) as e:
        r['err'] = f'{type(e).__name__}: {e}'[:300]
        return r
    finally:
        conn.close()


def get(base, path):
    u = urlparse(base)
    c = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=10)
    try:
        c.request('GET', path)
        return c.getresponse().read().decode()
    finally:
        c.close()


def metrics(base):
    out = {}
    for line in get(base, '/metrics').splitlines():
        if line and not line.startswith('#'):
            k, v = line.rsplit(' ', 1)
            out[k.replace('inkling_', '').replace('_total', '')] = float(v)
    return out


# ---------------------------------------------------------------- sessions
class Session:
    def __init__(self, a, name, base_tok, sid, nonce):
        self.a, self.name, self.sid, self.n = a, name, sid, 0
        self.g = Gen(hash((nonce, sid)) & 0xffffffff, a.tpc)
        self.prev = 0  # previous turn's prompt tokens
        self.msgs = [{'role': 'system', 'content': 'You are a coding agent. Use the tools to inspect the repository '
                                                   'and answer briefly.'},
                     {'role': 'user', 'content': f'[run {nonce} session {sid}] Repository dump follows.\n'
                                                 + self.g.mix(int(base_tok * a.scale))
                                                 + '\nFind why the cache tests fail. Start by reading a file.'}]
        self.reply = None

    def body(self):
        return dict(model='x', messages=self.msgs, tools=TOOLS, stream=True, stream_options={'include_usage': True},
                    max_tokens=self.a.max_tokens, temperature=0, chat_template_kwargs={'enable_thinking': False},
                    timings=True)

    def add_tool(self, ntok):
        """Append the last reply (with its tool call, or a synthetic one) and a tool output of ~ntok tokens."""
        r = self.reply or {}
        calls = r.get('tool_calls') or [{'id': f'call_{self.sid}_{self.n}', 'type': 'function', 'function': {
            'name': TOOLS[self.n % 3]['function']['name'], 'arguments': json.dumps({'path': 'src/cache.py'})}}]
        self.msgs.append({'role': 'assistant', 'content': r.get('content') or '', 'tool_calls': calls[:1]})
        kinds = (('code',), ('log',), ('js',))[self.n % 3]
        self.msgs.append({'role': 'tool', 'tool_call_id': calls[0].get('id', ''),
                          'content': self.g.mix(int(ntok * self.a.scale), kinds)})

    def turn(self, phase, tool):
        """One request (tool None: the cold first turn) -> row."""
        if tool is not None:
            self.add_tool(tool)
        r = post(self.a.base, self.body(), self.a.timeout)
        self.reply = r if r['ok'] else None
        u, tm = r['usage'], r['timings']
        p, c = u.get('prompt_tokens', 0), (u.get('prompt_tokens_details') or {}).get('cached_tokens', 0)
        row = dict(sess=self.name, sid=self.sid, turn=self.n, phase=phase, tool=tool or 0, prompt=p, cached=c,
                   suffix=p - c, hit=None if tool is None else bool(self.prev and c >= 0.98 * self.prev),
                   ttft=r['ttft'], dec_tps=r.get('dec_tps'), ctok=u.get('completion_tokens', 0), err=r['err'],
                   src=tm.get('src', '?'), t=time.time(), **{k: tm.get(k) for k in (
                       'queue_s', 'tokenize_s', 'restore_s', 'start_s', 'prefill_chunk_s', 'first_token_s', 'ttft_s',
                       'sync_write_s', 'disk_pages', 'step_max_s', 'solo')})
        self.prev, self.n = p or self.prev, self.n + 1
        return row


def evict_parked(a, nonce, k):
    """k short one-off prompts (>= INK_SNAP_MIN, no tools: parked but never written to disk) push older parked states
    out of the GPU (to disk if they are chats)."""
    g = Gen(7, a.tpc)
    for i in range(k):
        post(a.base, dict(model='x', stream=True, max_tokens=1, temperature=0, chat_template_kwargs={
            'enable_thinking': False}, messages=[{'role': 'user', 'content': f'[evict {nonce} {i}] ' + g.mix(800)
                                                  + '\nReply with OK.'}]), a.timeout)


def evict_pool(a, nonce):
    """Reserve the whole KV pool (small prompt + huge max_tokens; cancelled after its first token): parked states are
    unparked (chat states saved to disk) and every evictable page is evicted (chat pages spilled). -> server seconds
    of synchronous disk writes it caused (metrics delta), or None."""
    max_ctx = json.loads(get(a.base, '/v1/models'))['data'][0].get('max_model_len') or 1 << 20
    m0 = metrics(a.base)
    post(a.base, dict(model='x', stream=True, max_tokens=max_ctx - 2000, temperature=0, chat_template_kwargs={
        'enable_thinking': False}, messages=[{'role': 'user', 'content': f'[pool {nonce}] Say OK.'}]), a.timeout, cut=True)
    t0 = time.time()
    while time.time() - t0 < 120:  # until the cancelled request has left
        m1 = metrics(a.base)
        if not m1.get('running') and not m1.get('queue_depth'):
            break
        time.sleep(0.2)
    return round(m1.get('sync_write_s', 0) - m0.get('sync_write_s', 0), 3) if 'sync_write_s' in m0 else None


def run_single(a, name, nonce, sid, rows, log):
    s = Session(a, name, CTX[name], sid, nonce)
    log(s.turn('cold', None))
    for i in range(a.turns):
        log(s.turn('gpu', SIZES[i % 3]))
    if a.no_disk:
        return
    evict_parked(a, nonce, a.park + 1)
    log(s.turn('disk', 2000))
    if name == '1m' and not a.disk_1m:
        return
    w = evict_pool(a, f'{nonce}-{sid}')
    print(f'  pool evictor: {w} s of synchronous disk writes (page spills + state saves)', flush=True)
    rows.append(dict(sess=name, phase='evict', sync_write_s=w))
    log(s.turn('disk+pages', 2000))
    log(s.turn('gpu', 2000))


def run_multi(a, nonce, rows, log):
    names = (a.multi_ctx.split(',') * a.multi)[:a.multi]

    def one(i, name):
        time.sleep(i * a.stagger)
        s = Session(a, name, CTX[name], 100 + i, nonce)
        rng = random.Random(i)
        log(s.turn('multi-cold', None))
        for t in range(a.turns):
            time.sleep(rng.uniform(0.5, 2 * a.think))
            log(s.turn('multi', SIZES[(t + i) % 3]))
    th = [threading.Thread(target=one, args=(i, n)) for i, n in enumerate(names)]
    for t in th:
        t.start()
    for t in th:
        t.join()


# ---------------------------------------------------------------- report
COLS = [('sess', 5, '{}'), ('turn', 4, '{}'), ('phase', 10, '{}'), ('tool', 5, '{}'), ('prompt', 8, '{}'),
        ('cached', 8, '{}'), ('suffix', 6, '{}'), ('hit', 3, '{}'), ('src', 10, '{}'), ('ttft', 6, '{:.2f}'),
        ('ttft_s', 6, '{:.2f}'), ('tokenize_s', 6, '{:.2f}'), ('queue_s', 6, '{:.2f}'), ('restore_s', 6, '{:.3f}'),
        ('prefill_chunk_s', 6, '{:.2f}'), ('sync_write_s', 6, '{:.2f}'), ('disk_pages', 5, '{}'),
        ('step_max_s', 6, '{:.2f}'), ('dec_tps', 6, '{:.0f}'), ('solo', 4, '{}')]
HEAD = dict(ttft='TTFT', ttft_s='srvTT', tokenize_s='tok', queue_s='queue', restore_s='rstor', prefill_chunk_s='prefl',
            sync_write_s='write', disk_pages='pgs', step_max_s='stepX', dec_tps='tok/s')


def fmt(row):
    out = []
    for k, w, f in COLS:
        v = row.get(k)
        s = '-' if v is None else ('y' if v else 'N') if k == 'hit' else f.format(v)
        out.append(s.rjust(w)[:max(w, len(s))])
    return ' '.join(out)


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def evaluate(rows, targets):
    """-> [(name, value, limit, verdict True/False/None)]."""
    turns = [r for r in rows if r.get('phase') != 'evict']
    res = []
    for sess, src, tool, lim in targets:
        m = [r['ttft'] for r in turns if r['sess'] == sess and r['src'] == src and r['tool'] == tool
             and not r['phase'].startswith('multi')]
        if not any(r['sess'] == sess for r in turns):
            continue
        v = med(m)
        res.append((f'TTFT {src} {sess} {tool}-tok tool (median of {len(m)})', v, lim, None if v is None else v <= lim))
    warm = [r for r in turns if r['hit'] is not None]
    miss = [f"{r['sess']}#{r['turn']}" for r in warm if not r['hit']]
    res.append((f'warm turns hitting the cache ({len(warm) - len(miss)}/{len(warm)}; misses {miss[:6]})',
                len(warm) - len(miss), len(warm), not miss))
    errs = [r for r in turns if r['err']]
    res.append((f'errors ({[r["err"][:60] for r in errs[:3]]})', len(errs), 0, not errs))
    g = [r['restore_s'] for r in turns if r['src'] == 'gpu' and r.get('restore_s') is not None]
    if g:
        res.append(('max restore_s of GPU-parked turns (zero-copy fork)', max(g), 0.15, max(g) <= 0.15))
    mw = [r['ttft'] for r in turns if r['phase'] == 'multi']
    if mw:
        v = med(mw)
        res.append(('multi-session warm-turn TTFT p50', v, 3.0, None if v is None else v <= 3.0))
        xs = sorted(x for x in mw if x is not None)
        p90 = xs[int(0.9 * (len(xs) - 1))] if xs else None
        res.append(('multi-session warm-turn TTFT p90 (info: head-of-line behind cold prefills)', p90, None, None))
    for sess in dict.fromkeys(r['sess'] for r in turns):
        d = med([r['dec_tps'] for r in turns if r['sess'] == sess and not r['phase'].startswith('multi')])
        res.append((f'decode tok/s {sess} (info, median)', d, None, None))
    return res


def report(rows, res, out):
    print('\n' + ' '.join(HEAD.get(k, k).rjust(w) for k, w, _ in COLS))
    for r in rows:
        if r.get('phase') == 'evict':
            print(f"  [{r['sess']} pool evictor: sync disk writes {r['sync_write_s']} s]")
        else:
            print(fmt(r))
    print('\nTARGETS')
    for name, v, lim, ok in res:
        vs = '-' if v is None else f'{v:.2f}' if isinstance(v, float) else str(v)
        print(f"  {'PASS' if ok else 'n/a ' if ok is None else 'FAIL'} {name}: {vs}" + (f' (limit {lim})' if lim is not None else ''))
    passed = all(ok is not False for _, _, _, ok in res)
    print('AGENTLOOP PASSED' if passed else 'AGENTLOOP FAILED', flush=True)
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, 'w') as f:
            json.dump({'rows': rows, 'targets': res, 'pass': passed}, f, default=str)
        print(f'report: {out}')
    return passed


def run(a, targets=TARGETS):
    nonce = a.nonce or f'{int(time.time())}'
    rows, lock = [], threading.Lock()

    def log(r):
        with lock:
            rows.append(r)
            print(fmt(r), flush=True)
    print(f'agent loop vs {a.base}: sessions {a.ctx}, {a.turns} turns, multi {a.multi}x{a.multi_ctx}, '
          f'scale {a.scale}, tokens/char {a.tpc:.3f}, nonce {nonce}', flush=True)
    print(' '.join(HEAD.get(k, k).rjust(w) for k, w, _ in COLS), flush=True)
    for i, name in enumerate(a.ctx.split(',') if a.ctx else []):
        if name == '1m' and a.multi:  # 1M needs the whole pool: run it after the multi phase (see main)
            continue
        run_single(a, name, nonce, i, rows, log)
    if a.multi:
        run_multi(a, nonce, rows, log)
    if a.ctx and '1m' in a.ctx.split(',') and a.multi:
        run_single(a, '1m', nonce, 99, rows, log)
    return report(rows, evaluate(rows, targets), a.out)


# ---------------------------------------------------------------- server start (GPU window) / stub (selftest)
def start_server(a):
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PATH=os.path.join(here, '.venv/bin') + ':' + os.environ.get('PATH', ''))
    logp = os.path.expanduser(f'~/llm/logs/ink-server-agentloop-{time.strftime("%H%M%S")}.log')
    p = subprocess.Popen([os.path.join(here, '.venv/bin/python'), os.path.join(here, 'server.py')], cwd=here, env=env,
                         stdout=open(logp, 'w'), stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < 900:
        if p.poll() is not None:
            raise SystemExit(f'server exited ({p.returncode}), log {logp}')
        try:
            if '"ok":true' in get(a.base, '/health').replace(' ', ''):
                print(f'server up after {time.time() - t0:.0f} s, log {logp}', flush=True)
                return p
        except OSError:
            pass
        time.sleep(2)
    p.terminate()
    raise SystemExit(f'server did not come up in 900 s, log {logp}')


def stub_server(slow_pages=0.0, park=2):
    """Prefix-caching OpenAI stub: tokens = chars / 4; entries keyed by message-prefix hashes: parked (LRU of `park`,
    src gpu), pushed out -> disk if a chat (tools), pool evictor (max_tokens > 100k) -> chat pages to disk
    (src disk+pages, + slow_pages s). Streams 8 tokens, usage + timings like server.py."""
    import hashlib
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    st = {'ent': {}, 'park': [], 'lock': threading.Lock(), 'm': {'sync_write_s': 0.0, 'running': 0, 'queue_depth': 0}}

    class H(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.0'

        def log_message(self, *a):
            pass

        def reply(self, obj, ctype='application/json'):
            d = obj.encode() if isinstance(obj, str) else json.dumps(obj).encode()
            self.send_response(200); self.send_header('Content-Type', ctype); self.end_headers(); self.wfile.write(d)

        def do_GET(self):
            if self.path == '/health':
                return self.reply({'ok': True})
            if self.path == '/v1/models':
                return self.reply({'data': [{'id': 'x', 'max_model_len': 1 << 20}]})
            return self.reply(''.join(f'inkling_{k} {v}\n' for k, v in st['m'].items()), 'text/plain')

        def do_POST(self):
            b = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            msgs, chat = b['messages'], bool(b.get('tools'))
            hs = [hashlib.sha1(json.dumps(msgs[:k + 1]).encode()).hexdigest() for k in range(len(msgs))]
            toks = sum(len(json.dumps(m)) for m in msgs) // 4
            with st['lock']:
                hit = next((st['ent'][h] for h in reversed(hs) if h in st['ent']), None)
                cached, src, extra = 0, 'cold', 0.0
                if hit:
                    cached = hit['tok']
                    src = 'gpu' if hit['where'] == 'park' else 'disk+pages' if hit['pages'] == 'disk' else 'disk'
                    extra = {'gpu': 0.0, 'disk': 0.05, 'disk+pages': 0.1 + slow_pages}[src]
                    st['ent'].pop(hit['h'], None)                        # superseded by this turn
                    st['park'] = [h for h in st['park'] if h != hit['h']]
                if b.get('max_tokens', 0) > 100000:                      # pool evictor
                    for e in list(st['ent'].values()):
                        if e['chat']:
                            e['where'], e['pages'] = 'disk', 'disk'
                        else:
                            st['ent'].pop(e['h'])
                    st['park'], st['m']['sync_write_s'] = [], st['m']['sync_write_s'] + 0.01
                e = st['ent'][hs[-1]] = dict(h=hs[-1], tok=toks, chat=chat, where='park', pages='gpu')
                st['park'].append(e['h'])
                while len(st['park']) > park:
                    old = st['ent'].get(st['park'].pop(0))
                    if old and old['chat']:
                        old['where'] = 'disk'
                    elif old:
                        st['ent'].pop(old['h'])
            time.sleep(0.01 + extra + (toks - cached) * 2e-6)
            use = {'prompt_tokens': toks, 'completion_tokens': 8, 'total_tokens': toks + 8,
                   'prompt_tokens_details': {'cached_tokens': cached}}
            tm = {'src': src, 'queue_s': 0.0, 'restore_s': 0.001 + extra, 'prefill_chunk_s': 0.01, 'sync_write_s': 0.0}
            self.send_response(200); self.send_header('Content-Type', 'text/event-stream'); self.end_headers()
            ev = lambda d: self.wfile.write(b'data: ' + json.dumps(d).encode() + b'\n\n')
            try:
                for i in range(min(8, b.get('max_tokens', 8))):
                    ev({'choices': [{'index': 0, 'delta': {'content': f'w{i} '}}]}); self.wfile.flush(); time.sleep(0.002)
                ev({'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})
                ev({'choices': [], 'usage': use, **({'timings': [tm]} if b.get('timings') else {})})
                self.wfile.write(b'data: [DONE]\n\n')
            except (BrokenPipeError, ConnectionResetError):
                pass

    srv = ThreadingHTTPServer(('127.0.0.1', 0), H)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def selftest():
    base = ['--scale', '0.002', '--turns', '4', '--ctx', '256k,32k', '--multi', '3', '--multi-ctx', '128k',
            '--stagger', '0.1', '--think', '0.05', '--tpc', '0.25', '--out', '']
    srv = stub_server()
    a = parse(base + ['--base', f'http://127.0.0.1:{srv.server_port}'])
    assert run(a), 'fast stub must PASS'
    srv.shutdown()
    # slow disk pages (3.5 s): exactly the disk+pages target fails
    srv = stub_server(slow_pages=3.5)
    a = parse(base + ['--base', f'http://127.0.0.1:{srv.server_port}', '--multi', '0', '--ctx', '256k', '--turns', '3'])
    rows = []
    s_log = rows.append
    run_single(a, '256k', 'st2', 0, rows, s_log)
    res = evaluate(rows, TARGETS)
    srv.shutdown()
    bad = [n for n, _, _, ok in res if ok is False]
    assert len(bad) == 1 and 'disk+pages 256k' in bad[0], res
    srcs = [r['src'] for r in rows if r.get('phase') != 'evict']
    assert srcs == ['cold', 'gpu', 'gpu', 'gpu', 'disk', 'disk+pages', 'gpu'], srcs
    assert all(r['hit'] for r in rows if r.get('hit') is not None), rows
    g = Gen(3, 0.25)
    assert abs(len(g.mix(1000)) - 4000) <= 1 and Gen(3, 0.25).mix(500) == Gen(3, 0.25).mix(500)
    print('SELFTEST OK: fast stub PASS (single 256k/32k + 3-session multi), slow-pages stub FAILS only on', bad)


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--base', default=os.environ.get('INK_BASE', 'http://127.0.0.1:30000'))
    p.add_argument('--ctx', default='32k,128k,256k,1m', help=f'single sessions, comma list of {",".join(CTX)}')
    p.add_argument('--turns', type=int, default=8)
    p.add_argument('--multi', type=int, default=4, help='concurrent sessions (0 = off)')
    p.add_argument('--multi-ctx', default='256k,256k,256k,128k', help='their contexts, cycled')
    p.add_argument('--stagger', type=float, default=20.0, help='s between multi-session starts')
    p.add_argument('--think', type=float, default=1.5, help='mean client think/tool time between multi turns, s')
    p.add_argument('--max-tokens', type=int, default=128)
    p.add_argument('--park', type=int, default=int(os.environ.get('INK_PARK', 2)), help="server's INK_PARK")
    p.add_argument('--no-disk', action='store_true', help='skip the disk / disk+pages turns')
    p.add_argument('--disk-1m', action='store_true', help='also run disk+pages at 1M (~9 GB spilled + read back)')
    p.add_argument('--scale', type=float, default=1.0, help='multiply every token size (stub/self-test)')
    p.add_argument('--tpc', type=float, default=None, help='tokens per char (default: calibrate with the tokenizer)')
    p.add_argument('--nonce', default=None)
    p.add_argument('--timeout', type=float, default=3600)
    p.add_argument('--start-server', action='store_true', help='start server.py first (GPU window), stop it at the end')
    p.add_argument('--out', default=os.path.expanduser(f'~/llm/logs/ink-agentloop-{time.strftime("%Y%m%d-%H%M")}.json'))
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--selftest', action='store_true')
    a = p.parse_args(argv)
    if a.smoke:
        a.ctx, a.turns, a.multi, a.multi_ctx, a.stagger, a.think = '16k', 3, 2, '8k', 3.0, 0.5
    a.tpc = a.tpc or Gen(1, 1).calibrate()
    return a


def main():
    a = parse()
    if a.selftest:
        return selftest()
    srv = start_server(a) if a.start_server else None
    try:
        ok = run(a)
    finally:
        if srv is not None:
            srv.send_signal(signal.SIGTERM)
            try:
                srv.wait(60)
            except subprocess.TimeoutExpired:
                srv.kill()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
