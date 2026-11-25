"""GPU window: json_schema, VIP priority (queue jump, no preemption), prefix reuse (also with a 16.7 MP image: imgcache), stop sequences,
concurrency (continuous batching), speculative decoding (spec: DSpark on vs `spec: false`), against our running server (default :30000).
Usage: python tests/gpu_features.py [json|priority|prefix|imgcache|stop|concurrency|spec|logprobs|n|tools|params|specproc|all|ckpt] [BASE]
(ckpt: not part of all; start the server with INK_CKPT_EVERY=32768)  exit 0 = all checks passed
Start the server first (window.sh start_server, or python server.py; defaults INK_PARK=2, INK_DISK_MIN=2048)."""
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.expanduser('~/llm/gate'))
import gate  # noqa: E402  (chat(), filler(), check())

what = sys.argv[1] if len(sys.argv) > 1 else 'all'
gate.BASE = BASE = sys.argv[2] if len(sys.argv) > 2 else 'http://localhost:30000'
check = gate.check


def metric(name):
    raw = urllib.request.urlopen(BASE + '/metrics', timeout=10).read().decode()
    m = re.search(rf'^inkling_{name}(?:_total)? ([\d.e+-]+)$', raw, re.M)
    return float(m.group(1)) if m else None


def post(body, timeout=1800):
    req = urllib.request.Request(BASE + '/v1/chat/completions', json.dumps(dict(model='x', **body)).encode(),
                                 {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


SCHEMA = {'type': 'object', 'properties': {'city': {'type': 'string'}, 'population_millions': {'type': 'number'}},
          'required': ['city', 'population_millions'], 'additionalProperties': False}
Q = 'Give the capital of France and its rough metro population in millions.'


def t_json():
    print('== json_schema (gate.py image test, json part)', flush=True)
    j = gate.chat([{'role': 'user', 'content': Q}], max_tokens=80,
                  response_format={'type': 'json_schema', 'json_schema': {'name': 'c', 'schema': SCHEMA, 'strict': True}})
    ok = False
    try:
        d = json.loads(j['text']); ok = d['city'].lower() == 'paris' and isinstance(d['population_millions'], (int, float))
    except Exception:
        pass
    check(j['status'] == 200 and ok, f"json_schema thinking off: {j['text'].strip()[:80]!r}")
    r = post(dict(messages=[{'role': 'user', 'content': Q}], max_tokens=1500, temperature=0, reasoning_effort='low',
                  response_format={'type': 'json_schema', 'json_schema': {'name': 'c', 'schema': SCHEMA}}))
    msg = r['choices'][0]['message']
    try:
        d = json.loads(msg['content']); ok = d['city'].lower() == 'paris'
    except Exception:
        ok = False
    check(ok and r['choices'][0]['finish_reason'] == 'stop',
          f"json_schema thinking on: reasoning {len(msg.get('reasoning_content') or '')} chars, content {msg['content'][:80]!r}")
    r = post(dict(messages=[{'role': 'user', 'content': 'Describe a cat as a JSON object.'}], max_tokens=200,
                  temperature=0, chat_template_kwargs={'enable_thinking': False}, response_format={'type': 'json_object'}))
    try:
        ok = isinstance(json.loads(r['choices'][0]['message']['content']), dict)
    except Exception:
        ok = False
    check(ok, f"json_object: {r['choices'][0]['message']['content'][:80]!r}")


def ask(out, name, max_tokens, content='Count upward from 1. ({})', **extra):
    t0 = time.time()
    try:
        r = post(dict(max_tokens=max_tokens, temperature=0, messages=[{'role': 'user', 'content': content.format(name)}],
                      chat_template_kwargs={'enable_thinking': False}, **extra))
        out[name], out[name + ':usage'] = time.time(), r['usage']
    except Exception as e:
        out[name] = None; print(name, 'error:', repr(e)[:300])
    print(f'{name}: done after {time.time() - t0:.1f}s', flush=True)


def run_all(specs):
    done, th = {}, []
    for delay, args, kw in specs:
        time.sleep(delay)
        th.append(threading.Thread(target=ask, args=(done,) + args, kwargs=kw)); th[-1].start()
    for t in th:
        t.join()
    return done


def t_priority():
    print('== priority (gate/priority.py adapted): every decode slot busy with a blocker, spoof (priority 0, as the '
          'LiteLLM hook rewrites non-VIPs) queued, VIP (10) 1 s later: no blocker is paused; the VIP takes the first free '
          'slot ahead of spoof; every blocker completes all its tokens', flush=True)
    slots, n = int(metric('slots') or 1), 600
    # greedy + a list far longer than max_tokens: a blocker can't finish early (a stray EOS -> inconclusive, reported)
    long_list = 'Write the numbers from 1 to 3000 separated by commas, nothing else. ({})'
    idle0 = (metric('vram_allocated_bytes') or 0) - (metric('parked_bytes') or 0)
    done, th = {}, []
    for i in range(slots):
        # spec: false: with DSpark a 600-token count finishes in ~3.5 s, before the VIP arrives (window g 11-23)
        th.append(threading.Thread(target=ask, args=(done, f'blocker{i}', n), kwargs={'content': long_list,
                                                                                    'spec': False}))
        th[-1].start()
        time.sleep(0.5)
    t0 = time.time()
    while not (metric('running') == slots == metric('batch_size') and metric('queue_depth') == 0) \
            and time.time() - t0 < 120:
        time.sleep(0.2)
    time.sleep(2)  # blockers decoding
    p0 = metric('preemptions') or 0
    full = metric('running') == slots == metric('batch_size') and not any(k.startswith('blocker') for k in done)
    print(f'before spoof + VIP: running {metric("running")}, batch {metric("batch_size")}, slots {slots}, '
          f'blockers done {sorted(k for k in done if ":" not in k)}', flush=True)
    for name, prio in (('spoof', 0), ('vip', 10)):
        th.append(threading.Thread(target=ask, args=(done, name, 8), kwargs={'priority': prio})); th[-1].start()
        time.sleep(1)
    for t in th:
        t.join()
    toks = [(done.get(f'blocker{i}:usage') or {}).get('completion_tokens') for i in range(slots)]
    early = [f'blocker{i}: {t}' for i, t in enumerate(toks) if t is not None and t < n]
    if not full or early:
        print(f'INCONCLUSIVE priority test: {"not all slots were decoding blockers when spoof + VIP arrived; " if not full else ""}'
              f'{"blockers stopped early (model ended the list): " + ", ".join(early) if early else ""}', flush=True)
    check(full and not early, f'conclusive: all {slots} slots decoding blockers when spoof + VIP arrived, and no blocker '
          f'stopped before {n} tokens (else INCONCLUSIVE, not a scheduler verdict): {toks}')
    ok = full and all(done.get(k) for k in ['vip', 'spoof'] + [f'blocker{i}' for i in range(slots)])
    check(ok and done['vip'] < done['spoof'], 'VIP jumped the queue: finished before spoof')
    check((metric('preemptions') or 0) == p0, 'no decode was preempted (preemptions metric unchanged)')
    check(all(t == n for t in toks), f'every blocker completed all {n} tokens: {toks}')
    # idle trim: after N concurrent requests the server keeps one warm slot Decoder (~0.2 GB) + parked states
    idle1, dec1, pb = (metric(k) or 0 for k in ('vram_allocated_bytes', 'slot_decoders', 'parked_bytes'))
    # (BatchDecoder: slot_decoders == slots, one shared set of buffers that stays allocated)
    check((dec1 <= 1 or metric('batch_decoder')) and idle1 - pb <= idle0 + 0.5 * 2**30,
          f'idle allocation after {slots} concurrent: {idle1 / 2**30:.2f} GiB (parked {pb / 2**30:.2f}) vs before '
          f'{idle0 / 2**30:.2f} GiB (minus parked), slot decoders {dec1:.0f}')
    print('== no priority preemption: VIP arrives while a ~90k-token low-priority prompt is prefilling', flush=True)
    # ~90k tokens and the VIP at 2 s: prefill got fast enough (11-23 glue) that a 36k prompt finished before a 4 s VIP
    doc, _ = gate.filler('prio' + str(int(time.time())), 150000, needles=1)
    p0 = metric('preemptions')
    done = run_all([(0, ('low', 8), {'content': doc + '\n\nSummarize in one line. ({})', 'priority': 0}),
                    (2, ('vip2', 8), {'priority': 10})])
    check(done.get('vip2') and done.get('low'), 'both finished')
    check((metric('preemptions') or 0) == (p0 or 0), f"prefill not paused by the VIP (preemptions {p0} -> {metric('preemptions')})")


def t_prefix():
    print('== prefix reuse: ~20k-token chat; turn 2 warm from the GPU-parked turn 1, then again from disk (startup '
          f'warm-up took {metric("warmup_seconds")} s; 0 = INK_WARMUP=0 or failed)', flush=True)
    doc, codes = gate.filler('pfx' + str(int(time.time())), 20000, needles=2)
    name, code = next(iter(codes.items()))
    t1 = [{'role': 'user', 'content': doc + '\n\nReply with OK.'}]
    # same max_tokens on every turn, so the 2x checks compare prefill cost, not decode length
    body = dict(max_tokens=16, temperature=0, chat_template_kwargs={'enable_thinking': False})
    t0 = time.time(); r1 = post(dict(body, messages=t1)); tc = time.time() - t0   # cold 20k prefill
    t2 = t1 + [{'role': 'assistant', 'content': r1['choices'][0]['message']['content']},
               {'role': 'user', 'content': f'What is the access code for {name}? Answer with the code only.'}]
    t0 = time.time(); warm = post(dict(body, messages=t2)); tw = time.time() - t0
    cw = warm['usage'].get('prompt_tokens_details', {}).get('cached_tokens', 0)
    for i in range(3):  # push turn 2's parked state off the GPU (INK_PARK=2): a chat turn, so it goes to disk
        post(dict(body, max_tokens=1, messages=[{'role': 'user', 'content': gate.filler(f'evict{i}{t0}', 2000, 1)[0]}]))
    t0 = time.time(); disk = post(dict(body, messages=t2)); td = time.time() - t0
    cd = disk['usage'].get('prompt_tokens_details', {}).get('cached_tokens', 0)
    a, b = warm['choices'][0]['message']['content'], disk['choices'][0]['message']['content']
    print(f'turn 1 cold {tc:.1f}s; warm {tw:.1f}s cached {cw}/{warm["usage"]["prompt_tokens"]}: {a!r}\n'
          f'disk {td:.1f}s cached {cd}: {b!r}; metrics parked {metric("parked")} free pages {metric("kv_free_pages")}')
    check(cw > 0.9 * r1['usage']['prompt_tokens'], 'turn 2 reused the parked turn 1')
    check(tw < 0.5 * tc, f'warm turn 2 at least 2x faster than the cold turn 1 ({tw:.1f}s vs {tc:.1f}s)')
    check(cd == disk['usage']['prompt_tokens'], 'turn 2 again: whole prompt restored from disk (no host-RAM tier exists)')
    check(td < 0.5 * tc, f'disk restore at least 2x faster than cold ({td:.1f}s vs {tc:.1f}s)')
    # info only: triton's atomic-add order can flip a near-tie; bit-exact restore is proven in test_engine_cpu
    print(f'  INFO disk-restored answer {"==" if a == b else "!="} warm answer', flush=True)
    check(code in a, f'needle answer correct ({code})')


def t_imgcache():
    print('== image in a cached agent chat: 16.7 MP document + ~20k tokens; turn 2 warm from the GPU-parked turn 1 '
          '(image tiles cached, vision not re-run), again from disk; a changed image at the same position misses', flush=True)
    import base64
    import io
    from PIL import Image, ImageDraw
    T0 = time.time()
    path = os.path.expanduser('~/llm/gate/img16mp_doc.jpg')  # gate/mkdoc.py: Locker 37 = 7Q4K-2938, checked 2026-11-17
    raw = open(path, 'rb').read()
    im = Image.open(io.BytesIO(raw)).convert('RGB')
    ImageDraw.Draw(im).rectangle([0, 0, 199, 199], fill=(0, 0, 0))  # same size (same token count), other pixels
    b = io.BytesIO(); im.save(b, 'JPEG', quality=90)
    url = lambda x: {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + base64.b64encode(x).decode()}}
    doc, _ = gate.filler('img' + str(int(time.time())), 20000, needles=1)
    body = dict(max_tokens=20, temperature=0, chat_template_kwargs={'enable_thinking': False})
    q1 = '\n\nWhat is the access code for Locker 37 in the image? Answer with the code only.'
    q2 = 'On what date was Locker 37 checked, per the image? Answer with the date only.'
    user1 = lambda x: {'role': 'user', 'content': [url(x), {'type': 'text', 'text': doc + q1}]}  # image first
    chat2 = lambda x: [user1(x), {'role': 'assistant', 'content': a1}, {'role': 'user', 'content': q2}]

    def turn(msgs):
        v0, t0 = metric('vision_images'), time.time()
        r = post(dict(body, messages=msgs))
        cached = r['usage'].get('prompt_tokens_details', {}).get('cached_tokens', 0)
        return r, time.time() - t0, metric('vision_images') - v0, cached
    r1, tc, v1, _ = turn([user1(raw)])
    a1, p1 = r1['choices'][0]['message']['content'], r1['usage']['prompt_tokens']
    t2 = chat2(raw)
    warm, tw, vw, cw = turn(t2)
    for i in range(3):  # push turn 2's parked state off the GPU (INK_PARK=2): a continued chat, so it goes to disk
        post(dict(body, max_tokens=1, messages=[{'role': 'user', 'content': gate.filler(f'evict{i}{T0}', 2000, 1)[0]}]))
    disk, td, vd, cd = turn(t2)
    other, to, vo, co = turn(chat2(b.getvalue()))
    aw, ad, ao = (r['choices'][0]['message']['content'] for r in (warm, disk, other))
    print(f'turn 1 cold {tc:.1f}s ({p1} tokens, vision runs {v1:.0f}): {a1!r}\n'
          f'warm {tw:.1f}s cached {cw}/{warm["usage"]["prompt_tokens"]} vision {vw:.0f}: {aw!r}\n'
          f'disk {td:.1f}s cached {cd}/{disk["usage"]["prompt_tokens"]} vision {vd:.0f}: {ad!r}\n'
          f'changed image {to:.1f}s cached {co} vision {vo:.0f}: {ao!r}', flush=True)
    check(v1 == 1 and '7Q4K-2938' in a1.upper(), f'turn 1: vision ran once, code read correctly ({a1.strip()[:30]!r})')
    check(cw > 0.9 * p1, f'warm turn 2 reused the parked turn 1 incl. the image tiles (cached {cw}, turn 1 {p1})')
    check(vw == 0 and tw < 0.5 * tc, f'warm turn 2: vision not re-run, at least 2x faster ({tw:.1f}s vs {tc:.1f}s)')
    check('11-17' in aw, f'warm turn 2 reads the image (Locker 37 checked 2026-11-17): {aw.strip()[:30]!r}')
    check(cd == disk['usage']['prompt_tokens'] and vd == 0 and td < 0.5 * tc,
          f'disk turn 2: whole prompt restored ({cd}), vision not re-run, at least 2x faster ({td:.1f}s vs {tc:.1f}s)')
    check('11-17' in ad, f'disk turn 2 reads the image: {ad.strip()[:30]!r}')
    check(co < 64 and vo == 1, f'changed image at the same position: no reuse past it (cached {co}), vision ran ({vo:.0f})')
    print(f'  imgcache took {time.time() - T0:.0f} s', flush=True)


def t_stop():
    print('== stop sequences (plain + streamed): counting stops before "7", nothing after it leaks', flush=True)
    body = dict(model='x', max_tokens=200, temperature=0, chat_template_kwargs={'enable_thinking': False}, stop=['7', 'xyz'],
                messages=[{'role': 'user', 'content': 'Count from 1 to 20, separated by commas. Numbers only.'}])
    r = post({k: v for k, v in body.items() if k != 'model'})
    c, fin = r['choices'][0]['message']['content'], r['choices'][0]['finish_reason']
    check('6' in c and '7' not in c and fin == 'stop' and r['usage']['completion_tokens'] < 40, f'plain: {c!r} {fin}')
    req = urllib.request.Request(BASE + '/v1/chat/completions', json.dumps(dict(body, stream=True)).encode(),
                                 {'Content-Type': 'application/json'})
    text, fin = '', None
    with urllib.request.urlopen(req, timeout=300) as resp:
        for line in resp:
            line = line.decode().strip()
            if line.startswith('data: {'):
                ch = json.loads(line[6:])['choices']
                if ch:
                    text += ch[0]['delta'].get('content') or ''
                    fin = ch[0]['finish_reason'] or fin
    check(text == c and fin == 'stop', f'streamed == plain: {text!r} {fin}')


QS = ['Name three rivers in Europe.', 'What is the boiling point of water in Celsius? Explain briefly.',
      'Write a haiku about the sea.', 'List the planets of the solar system in order.']


def parallel(bodies):
    """Post all bodies at once -> ([response or None], wall s, peak inkling_batch_size seen)."""
    out, peak, stop = [None] * len(bodies), [0], []

    def one(i):
        try:
            out[i] = post(bodies[i])
        except Exception as e:
            print(i, 'error:', repr(e)[:300])

    def sampler():
        while not stop:
            peak[0] = max(peak[0], metric('batch_size') or 0)
            time.sleep(0.2)
    threading.Thread(target=sampler, daemon=True).start()
    th = [threading.Thread(target=one, args=(i,)) for i in range(len(bodies))]
    t0 = time.time()
    for t in th:
        t.start()
    for t in th:
        t.join()
    stop.append(1)
    return out, time.time() - t0, peak[0]


def first_diff(a, b):
    """First differing char of two texts (len of the shorter if one is a prefix; len(a) if equal)."""
    return next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def t_concurrency():
    print('== concurrency: 4 short requests one by one (twice: run-to-run noise baseline), then all at once (outputs '
          'vs solo, aggregate tok/s)', flush=True)
    # spec: false on both sides: this measures batching itself (11-23: drafting made solo ~2x faster, while at B=4 the
    # batch-aware width drafts less on purpose, so a spec-on ratio compares drafted solo with mostly plain batches)
    body = lambda q, n=96: dict(max_tokens=n, temperature=0, chat_template_kwargs={'enable_thinking': False},
                                messages=[{'role': 'user', 'content': q}], spec=False)
    solo, solo2, t_solo, n_solo = [], [], 0.0, 0
    for q in QS:
        t0 = time.time(); r = post(body(q)); t_solo += time.time() - t0
        solo.append(r['choices'][0]['message']['content']); n_solo += r['usage']['completion_tokens']
    for q in QS:  # greedy reruns still differ sometimes: triton MoE atomic-add order flips near ties
        solo2.append(post(body(q))['choices'][0]['message']['content'])
    par, wall, peak = parallel([body(q) for q in QS])
    ok = all(par)
    texts = [r['choices'][0]['message']['content'] if r else '' for r in par]
    n_par = sum(r['usage']['completion_tokens'] for r in par if r)
    base = sum(a == b for a, b in zip(solo, solo2))
    same = sum(a == b for a, b in zip(solo, texts))
    d_solo = [first_diff(a, b) for a, b in zip(solo, solo2)]  # == len(a) when identical
    d_par = [first_diff(a, b) for a, b in zip(solo, texts)]
    for i, (a, b, c) in enumerate(zip(solo, solo2, texts)):
        for what, x, k in (('solo rerun', b, d_solo[i]), ('parallel', c, d_par[i])):
            if x != a:
                print(f'  INFO prompt {i}: {what} differs from solo at char {k}: {a[k:k + 40]!r} vs {x[k:k + 40]!r}')
    earlier = sum(p < q for p, q in zip(d_par, d_solo))
    print(f'solo: {n_solo} tokens in {t_solo:.1f}s ({n_solo / t_solo:.1f} tok/s); parallel: {n_par} tokens in '
          f'{wall:.1f}s ({n_par / wall:.1f} tok/s aggregate), peak batch {peak}\n'
          f'identical vs solo: solo rerun {base}/4 (noise baseline), parallel {same}/4; first divergence (char, '
          f'len if identical): solo rerun {d_solo}, parallel {d_par}', flush=True)
    check(ok, 'all 4 parallel requests completed')
    check(peak >= 2, f'decodes ran together (peak inkling_batch_size {peak})')
    if metric('batch_decoder'):  # one step reads the weights once for all rows (LoopBatch: single-stream speed)
        tps_solo, tps_par = n_solo / t_solo, n_par / wall
        check(tps_par >= 1.5 * tps_solo, f'batched: aggregate {tps_par:.1f} tok/s >= 1.5 x solo {tps_solo:.1f}')
    else:
        print('  INFO LoopBatch server (inkling_batch_decoder 0): no aggregate speed-up expected, not checked')
    # greedy batch rows are independent, so parallel may differ from solo only as much as solo differs from itself
    check(same >= base - 1, f'parallel == solo no worse than the solo rerun noise ({same}/4 vs {base}/4, one tolerated)')
    check(earlier <= 4 - base + 1, f'parallel divergences not systematically earlier than the noise ({earlier} prompts '
          f'diverge earlier than their solo rerun; allowed {4 - base + 1})')
    print('== queue (gate queue-like): a ~30k-token prompt, then 4 short ones 0.5 s later', flush=True)
    doc, codes = gate.filler('conc' + str(int(time.time())), 30000, needles=1)
    name, code = next(iter(codes.items()))
    long_body = body(doc + f'\n\nWhat is the access code for {name}? Answer with the code only.', 32)
    res, done = {}, {}

    def go(key, b):
        t0 = time.time()
        try:
            res[key] = post(b)
        except Exception as e:
            print(key, 'error:', repr(e)[:300])
        done[key] = time.time() - t0
    th = [threading.Thread(target=go, args=('long', long_body))]
    th[0].start()
    time.sleep(0.5)
    th += [threading.Thread(target=go, args=(f's{i}', body(q, 48))) for i, q in enumerate(QS)]
    for t in th[1:]:
        t.start()
    for t in th:
        t.join()
    print('  latency ' + ', '.join(f'{k} {v:.1f}s' for k, v in sorted(done.items())), flush=True)
    check(len(res) == 5, f'long + 4 short all completed ({sorted(res)})')
    ans = res.get('long', {}).get('choices', [{}])[0].get('message', {}).get('content', '')
    check(code in ans, f'long prompt needle correct ({code} in {ans[:60]!r})')


CODE = ('Write a Python module with a class LRUCache (get, put, capacity eviction, type hints, docstrings) and '
        'pytest unit tests for it.')
PROSE = 'Write a short story (about 300 words) about a lighthouse keeper who finds a message in a bottle.'


def t_spec():
    print('== spec (DSpark): greedy code / prose, spec on (server default) vs spec: false; tok/s from usage + wall '
          '(warm second run of each); then 2 concurrent spec requests', flush=True)
    body = lambda q, **kw: dict(max_tokens=400, temperature=0, chat_template_kwargs={'enable_thinking': False},
                                messages=[{'role': 'user', 'content': q}], **kw)

    def timed(b):
        post(b)                                                   # warm (prefix parked, graphs captured)
        t0 = time.time(); r = post(b); dt = time.time() - t0
        return r['choices'][0]['message']['content'], r['usage']['completion_tokens'] / dt
    res = {}
    for name, q in (('code', CODE), ('prose', PROSE)):
        (a, ta), (b, tb) = timed(body(q)), timed(body(q, spec=False))
        res[name] = a
        print(f'{name}: spec {ta:.1f} tok/s, plain {tb:.1f} tok/s (x{ta / tb:.2f}); outputs '
              + ('IDENTICAL' if a == b else f'differ at char {first_diff(a, b)}'), flush=True)
        check(a == b, f'{name}: greedy spec output == spec: false output')
        if name == 'code':
            check(ta >= 1.3 * tb, f'code: spec {ta:.1f} tok/s >= 1.3 x plain {tb:.1f}')
        else:
            check(ta >= 0.97 * tb, f'prose: spec {ta:.1f} tok/s >= 0.97 x plain {tb:.1f}')
    par, wall, peak = parallel([body(CODE), body(PROSE)])
    texts = [r['choices'][0]['message']['content'] if r else None for r in par]
    print(f'2 concurrent spec requests: {wall:.1f}s, peak batch {peak}', flush=True)
    check(all(par) and peak >= 2, '2 concurrent spec requests completed, decoded together')
    check(texts == [res['code'], res['prose']], 'concurrent spec outputs == solo (verify rows are batch-invariant)')


def t_specproc():
    print('== spec with logit processors (11-23: penalties / logit_bias / min_tokens / min_p now speculate): greedy '
          'spec == spec: false, fewer decode steps; sampled + min_p completes', flush=True)
    lb = {str(t): -1.0 for t in gate_ids(' foo')}
    b = dict(max_tokens=300, temperature=0, chat_template_kwargs={'enable_thinking': False}, frequency_penalty=0.1,
             presence_penalty=0.1, repetition_penalty=1.02, logit_bias=lb, min_tokens=30,
             messages=[{'role': 'user', 'content': CODE}])
    res = {}
    for on in (True, False):
        post(dict(b, spec=on))                                    # warm
        m0, t0 = metric('decode_steps'), time.time()
        r = post(dict(b, spec=on))
        dt, steps = time.time() - t0, metric('decode_steps') - m0
        res[on] = (r['choices'][0]['message']['content'], r['usage']['completion_tokens'], dt, steps)
    (a, na, ta, sa), (c, nc, tc, sc) = res[True], res[False]
    print(f'processors: spec {na / ta:.1f} tok/s in {sa:.0f} steps, plain {nc / tc:.1f} tok/s in {sc:.0f} steps', flush=True)
    check(a == c, 'greedy spec with processors == spec: false' + ('' if a == c else f' (differ at char {first_diff(a, c)})'))
    check(sa < 0.8 * sc, f'spec with processors drafts: {sa:.0f} decode steps vs {sc:.0f} plain')
    r = post(dict(b, temperature=0.8, top_p=0.95, min_p=0.05, seed=3))
    check(r['usage']['completion_tokens'] >= 30 and r['choices'][0]['message']['content'],
          f"sampled spec + processors + min_p: {r['usage']['completion_tokens']} tokens")


def sse(body):
    """Streamed request -> list of chunk dicts."""
    req = urllib.request.Request(BASE + '/v1/chat/completions', json.dumps(dict(model='x', stream=True, **body)).encode(),
                                 {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=600) as resp:
        return [json.loads(ln[6:]) for ln in (x.decode().strip() for x in resp) if ln.startswith('data: {')]


def t_logprobs():
    print('== logprobs: greedy chosen token == top-1, entries rebuild the text, streamed == plain, spec on == off',
          flush=True)
    b = dict(max_tokens=60, temperature=0, chat_template_kwargs={'enable_thinking': False}, logprobs=True,
             top_logprobs=5, messages=[{'role': 'user', 'content': CODE}])
    r = post(b)
    ch = r['choices'][0]
    ents = ch['logprobs']['content']
    text = bytes(x for e in ents for x in e['bytes']).decode('utf-8', 'replace')
    check(len(ents) >= 10 and text == ch['message']['content'], f'{len(ents)} entries rebuild the content')
    check(all(e['token'] == e['top_logprobs'][0]['token'] and abs(e['logprob'] - e['top_logprobs'][0]['logprob']) < 1e-4
              and len(e['top_logprobs']) == 5 and e['logprob'] <= 0 for e in ents), 'greedy: chosen == top-1, 5 tops')
    off = post(dict(b, spec=False))['choices'][0]['logprobs']['content']
    dl = max((abs(a['logprob'] - c['logprob']) for a, c in zip(ents, off)), default=9)
    check([e['token'] for e in off] == [e['token'] for e in ents] and dl < 0.05,
          f'spec on == spec off (tokens; max |dlogprob| {dl:.4f})')
    st = [e for c in sse(b) for c0 in c['choices'] for e in ((c0.get('logprobs') or {}).get('content') or [])]
    check([e['token'] for e in st] == [e['token'] for e in ents], f'streamed entries == plain ({len(st)})')
    t07 = post(dict(b, temperature=0.7, seed=1))['choices'][0]['logprobs']['content']
    check(all(e['logprob'] <= 0 for e in t07) and len(t07) > 5, 'T=0.7: entries present')
    m0 = metric('decode_steps')
    t0 = time.time(); post(dict(b, logprobs=False, top_logprobs=None)); a = time.time() - t0
    print(f'no-logprobs request {a:.2f}s (decode_steps +{metric("decode_steps") - m0:.0f})', flush=True)


def t_n():
    print('== n > 1: 3 choices, prompt counted once, completions summed, prefix shared (cached tokens)', flush=True)
    b = dict(max_tokens=40, temperature=0.9, seed=5, n=3, chat_template_kwargs={'enable_thinking': False},
             messages=[{'role': 'user', 'content': gate.filler(f'n{time.time():.0f}', 3000)[0] + '\nWrite one sentence about a fox.'}])
    c0 = metric('cached_tokens')
    r = post(b)
    texts = [c['message']['content'] for c in r['choices']]
    u = r['usage']
    check(len(texts) == 3 and len(set(texts)) >= 2, f'3 choices, distinct: {[t[:40] for t in texts]}')
    check(u['completion_tokens'] <= 120 and u['total_tokens'] == u['prompt_tokens'] + u['completion_tokens'], f'usage {u}')
    dc = metric('cached_tokens') - c0
    check(dc >= 1.5 * u['prompt_tokens'], f'choices 2-3 restored the prompt (cached +{dc:.0f}, prompt {u["prompt_tokens"]})')
    check([c['message']['content'] for c in post(b)['choices']] == texts, 'same seed -> same 3 choices')
    ch = sse(dict(b, stream_options={'include_usage': True}))
    check({c['choices'][0]['index'] for c in ch if c['choices']} == {0, 1, 2} and ch[-1].get('usage'), 'streamed n=3')


TOOLS = [{'type': 'function', 'function': {'name': 'get_weather', 'description': 'Current weather for a city',
                                           'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}},
                                                          'required': ['city']}}},
         {'type': 'function', 'function': {'name': 'get_time', 'description': 'Current time in a timezone',
                                           'parameters': {'type': 'object', 'properties': {'tz': {'type': 'string'}}}}}]


def t_tools():
    print('== tool_choice: required / named force a valid call even for a chit-chat prompt; none -> text', flush=True)
    for think in (True, False):
        base = dict(max_tokens=600, temperature=0, tools=TOOLS, messages=[{'role': 'user', 'content': 'Hi! How are you?'}],
                    **({} if think else {'chat_template_kwargs': {'enable_thinking': False}}))
        for choice, want in (('required', None), ({'type': 'function', 'function': {'name': 'get_time'}}, 'get_time')):
            r = post(dict(base, tool_choice=choice))
            c = r['choices'][0]
            tc = c['message'].get('tool_calls') or []
            ok = len(tc) >= 1 and c['finish_reason'] == 'tool_calls' and (want is None or tc[0]['function']['name'] == want)
            try:
                json.loads(tc[0]['function']['arguments'])
            except Exception:
                ok = False
            check(ok, f'think={think} {choice if isinstance(choice, str) else want}: {tc[:1]}')
        r = post(dict(base, tool_choice='none'))
        check(not r['choices'][0]['message'].get('tool_calls'), f'think={think} none: text only')
    r = post(dict(base, tool_choice='required', parallel_tool_calls=False,
                  messages=[{'role': 'user', 'content': 'Weather in Oslo, Paris and Rome?'}]))
    check(len(r['choices'][0]['message'].get('tool_calls') or []) == 1, 'parallel_tool_calls false: exactly one call')


def t_params():
    print('== sampling params: seed reproducible, penalties / logit_bias / min_p / top_k / min_tokens act, 400s', flush=True)
    b = dict(max_tokens=60, chat_template_kwargs={'enable_thinking': False},
             messages=[{'role': 'user', 'content': 'Write a short poem about rain.'}])
    a1, a2 = (post(dict(b, temperature=1.0, seed=42))['choices'][0]['message']['content'] for _ in range(2))
    check(a1 == a2, 'seed 42 twice -> identical')
    rep = dict(b, temperature=0, messages=[{'role': 'user', 'content': 'Say "la" 50 times.'}])
    plain = post(rep)['choices'][0]['message']['content']
    pen = post(dict(rep, frequency_penalty=1.5, presence_penalty=1.0))['choices'][0]['message']['content']
    check(pen.count('la') < plain.count('la'), f'penalties: la x{plain.count("la")} -> x{pen.count("la")}')
    yes = post(dict(rep, logit_bias={str(t): -100 for t in gate_ids('la') + gate_ids(' la')}))
    check(' la' not in yes['choices'][0]['message']['content'], 'logit_bias -100 bans the token')
    r = post(dict(b, temperature=1.0, min_p=0.1, top_k=20, repetition_penalty=1.1, seed=1, user='u', store=False,
                  stream_options={'include_usage': True}, max_completion_tokens=30))
    check(r['usage']['completion_tokens'] <= 30, f"min_p/top_k/repetition_penalty/max_completion_tokens: {r['usage']}")
    r = post(dict(b, temperature=0, min_tokens=20, max_tokens=40,
                  messages=[{'role': 'user', 'content': 'Reply with just: ok'}]))
    check(r['usage']['completion_tokens'] >= 20, f"min_tokens 20: {r['usage']['completion_tokens']}")
    # (window g2 11-23: this inherited b's max_tokens=60, too short for a poem as a JSON object: cut, finish 'length')
    r = post(dict(b, response_format={'type': 'json_object'}, temperature=0, max_tokens=400))
    c = r['choices'][0]
    try:
        ok = isinstance(json.loads(c['message']['content']), dict) and c['finish_reason'] == 'stop'
    except Exception:
        ok = False
    check(ok, f"json_object ({c['finish_reason']}, {r['usage']['completion_tokens']} tokens): "
              f"{c['message']['content'][:80]!r}")
    for bad in (dict(ignore_eos=True), dict(n=9), dict(top_logprobs=3)):
        try:
            post(dict(b, **bad)); ok = False
        except urllib.error.HTTPError as e:
            ok = e.code == 400
        check(ok, f'{bad} -> 400')


def t_ckpt():
    print('== mid-prompt checkpoints (server needs INK_CKPT_EVERY=32768): ~150k-token agent chat cold, then the same chat '
          'with one record edited at ~100k: restore from the 98304 (96k) checkpoint, not a cold prefill', flush=True)
    T0, CK = time.time(), 32768
    tag = 'ck' + str(int(T0))
    # gate.filler's 44.5 tokens/line is another tokenizer's: size it with ours (window 11-23: asked 150k, got 91k)
    tpl = len(gate_ids(gate.filler(tag, 44500, needles=0)[0])) / 1000
    doc, codes = gate.filler(tag, int(150000 * 44.5 / tpl), needles=2)
    name, code = next(iter(codes.items()))  # needle 0 sits at ~25%, before the edit
    lines = doc.split('\n')
    q = f'What is the access code for {name}? Answer with the code only.'

    def chat_(text, h=len(doc) // 3):  # an earlier assistant turn: a continued chat, so the prompt is disk-worthy
        # (checkpoints on); split at the same char for both versions (the edit changes the length: window 11-23, the
        # split moved and the first message differed at ~1/3)
        return [{'role': 'user', 'content': text[:h]}, {'role': 'assistant', 'content': 'Noted.'},
                {'role': 'user', 'content': text[h:] + '\n\n' + q}]
    full = prompt_ids(chat_(doc))
    i = int(len(lines) * 100000 / len(full))
    while 'status nominal.' not in lines[i]:
        i += 1
    edited = lines[:i] + [lines[i].replace('status nominal.', 'status degraded.')] + lines[i + 1:]
    ed = prompt_ids(chat_('\n'.join(edited)))
    e = next(j for j, (x, y) in enumerate(zip(full, ed)) if x != y)  # first token the edit changes
    want_ck, want_cached = (len(full) - 1) // CK, e // CK * CK      # (the prompt's end is parked, not checkpointed)
    print(f'prompt {len(full)} tokens (our tokenizer), edit at token {e}: expect {want_ck} checkpoints, restore from '
          f'{want_cached}', flush=True)
    body = dict(max_tokens=16, temperature=0, chat_template_kwargs={'enable_thinking': False})
    files = lambda: sum(f.startswith('c-') for _, _, fs in os.walk(os.path.expanduser('~/llm/inkcache')) for f in fs)
    f0 = files()
    t0 = time.time(); cold = post(dict(body, messages=chat_(doc))); tc = time.time() - t0
    time.sleep(2)  # (checkpoint files are written in the background)
    nf = files() - f0
    t0 = time.time(); warm = post(dict(body, messages=chat_('\n'.join(edited)))); tw = time.time() - t0
    cw = warm['usage'].get('prompt_tokens_details', {}).get('cached_tokens', 0)
    a, b = cold['choices'][0]['message']['content'], warm['choices'][0]['message']['content']
    print(f'cold {tc:.1f}s ({cold["usage"]["prompt_tokens"]} tokens, {nf} new checkpoint files): {a!r}\n'
          f'edited at line {i}/{len(lines)}: {tw:.1f}s cached {cw}/{warm["usage"]["prompt_tokens"]}: {b!r}', flush=True)
    check(cold['usage']['prompt_tokens'] == len(full), f"prompt size as rendered here ({cold['usage']['prompt_tokens']} "
          f"vs {len(full)})")
    check(nf >= want_ck >= 4, f'cold prompt wrote >= {want_ck} checkpoints ({nf}; 0 = INK_CKPT_EVERY not set on the server)')
    check(cw == want_cached >= 3 * CK, f'edited prompt restored from the {want_cached} checkpoint (cached {cw})')
    check(tw < 0.5 * tc, f'edited prompt at least 2x faster than cold ({tw:.1f}s vs {tc:.1f}s)')
    check(code in a and a == b, f'same answer as the cold run, needle correct ({code}: {a!r} / {b!r})')
    print(f'  ckpt took {time.time() - T0:.0f} s', flush=True)


_TOK = []


def ink_tok():
    if not _TOK:
        from tokenizers import Tokenizer
        _TOK.append(Tokenizer.from_file(os.path.expanduser('~/llm/models/Inkling-2-Mini-Uncensored/tokenizer.json')))
    return _TOK[0]


def gate_ids(s):
    return ink_tok().encode(s, add_special_tokens=False).ids


def prompt_ids(messages):
    """The prompt as the server renders it with thinking off (effort 0 + the pre-filled MODEL TEXT)."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    import chat
    return list(chat.render(ink_tok(), messages, None, 0)) + [chat.MODEL, chat.TEXT]


for name, fn in (('json', t_json), ('priority', t_priority), ('prefix', t_prefix), ('imgcache', t_imgcache),
                 ('stop', t_stop), ('concurrency', t_concurrency), ('spec', t_spec), ('logprobs', t_logprobs), ('n', t_n),
                 ('tools', t_tools), ('params', t_params), ('specproc', t_specproc)):
    if what in (name, 'all'):
        fn()
if what == 'ckpt':  # (not in all: needs a server started with INK_CKPT_EVERY=32768)
    t_ckpt()
print('FAILED:' if gate.FAILS else 'ALL PASSED', *gate.FAILS, sep='\n  ')
raise SystemExit(1 if gate.FAILS else 0)
