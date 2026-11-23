"""Offline self-test of the eval logic (CPU only, no model): a tiny local HTTP server returns canned OpenAI
responses with known right/wrong answers, and each suite's scoring must match.  python tests/evals/test_offline.py"""
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gsm8k  # noqa: E402
import haystack  # noqa: E402
import refusal  # noqa: E402
import run_all  # noqa: E402
import toolcall  # noqa: E402
from common import Client  # noqa: E402

GOLD = {q: g for q, g in gsm8k.load(1319)}
SEEN = {}


def reply(body):
    msgs, last = body['messages'], body['messages'][-1]
    text = last.get('content') or ''
    effort = body.get('reasoning_effort')
    if effort == 'minimal':
        return 400, {'error': 'unsupported'}
    msg, finish = {'role': 'assistant', 'content': ''}, 'stop'
    if body.get('tools') and last['role'] == 'user' and 'weather' in text:
        msg['tool_calls'] = [{'id': 'call_1', 'type': 'function',
                              'function': {'name': 'get_weather', 'arguments': '{"city": "Reykjavik"}'}}]
        finish = 'tool_calls'
    elif last['role'] == 'tool':
        t = json.loads(text)
        msg['content'] = f"It is {t['temp_c']} C in Reykjavik, a {t['condition']}."
    elif 'Fahrenheit' in text:
        msg['content'] = '73.4'
    elif 'Python codebase' in text:
        names = [n for n in re.findall(r'`(\w+)`', text.split('Answer from the codebase')[-1]) if n != 'retries']
        vals = [re.search(rf'def {n}\(.*?return "([^"]+)"', text, re.S).group(1) for n in names[:2]]
        vals.append(re.search(rf'def {names[2]}\(journal, retries=(\d+)', text).group(1))
        msg['content'] = '\n'.join(f'{i + 1}. {v}' for i, v in enumerate(vals))
    elif text.startswith('Solve this math'):
        q = text.split('\n\n', 1)[1]
        g = GOLD[q]
        msg['content'] = f'Work...\n**Answer: {int(float(g)) + 1000:,}**' if len(q) % 2 else f'So $\\boxed{{{g}}}$.'
    elif text in refusal.PROMPTS:
        msg['content'] = "I'm sorry, but I can't help with that." if 'lock' in text else 'Sure. Here is how it works...'
    elif text == toolcall.SWEEP_Q:
        msg['content'] = 'Answer: 14:35'
    if effort and effort != 'none':
        msg['reasoning_content'] = 'x' * {'low': 50, 'medium': 300, 'high': 900}[effort]
    return 200, {'choices': [{'message': msg, 'finish_reason': finish}],
                 'usage': {'prompt_tokens': sum(len(m.get('content') or '') for m in msgs) // 3, 'completion_tokens': 7}}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        self.send(200, {'data': [{'id': 'mock-model'}]})

    def do_POST(self):
        SEEN['auth'] = self.headers.get('Authorization')
        self.send(*reply(json.loads(self.rfile.read(int(self.headers['Content-Length'])))))


def main():
    # extraction unit cases
    assert gsm8k.extract('blah 3 then\nAnswer: 1,234') == '1234'
    assert gsm8k.extract('Answer: $18.00.') == '18'
    assert gsm8k.extract(r'\boxed{-7.5} done') == '-7.5'
    assert gsm8k.extract('it is 12 or maybe 15') == '15'
    assert gsm8k.extract('no number') is None
    assert refusal.refused("I can't help with that.") and not refusal.refused('Sure! ' + 'x' * 400 + " I can't")
    m1, n1, v1 = haystack.build(50000, seed=3)
    m2, _, _ = haystack.build(50000, seed=3)
    assert m1 == m2 and m1 != haystack.build(50000, seed=4)[0], 'haystack must be deterministic per seed'
    assert len(set(n1)) == 3 and all(f'def {n}(' in m1[0]['content'] for n in n1)
    assert haystack.size_of('1m') == 1000000 and haystack.size_of('32k') == 32000

    srv = ThreadingHTTPServer(('127.0.0.1', 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f'http://127.0.0.1:{srv.server_address[1]}'
    os.environ['EVAL_TEST_KEY'] = 'testkey'
    c = Client(base + '/v1', key_env='EVAL_TEST_KEY')
    assert c.model == 'mock-model' and c.base == base
    assert c.chat([{'role': 'user', 'content': 'hi'}], effort='minimal')['error'].startswith('HTTP 400')
    assert SEEN['auth'] == 'Bearer testkey'

    g = gsm8k.run(c, quick=True, parallel=4)['summary']
    want = round(100 * sum(len(q) % 2 == 0 for q, _ in gsm8k.load(20)) / 20, 1)
    assert g['gsm8k_off_acc'] == g['gsm8k_low_acc'] == want and g['gsm8k_n'] == 20, (g, want)
    r = refusal.run(c)['summary']
    want = round(100 * sum('lock' in p for p in refusal.PROMPTS) / len(refusal.PROMPTS), 1)
    assert r['refusal_off_rate'] == r['refusal_low_rate'] == want, r
    t = toolcall.run(c)['summary']
    assert t['tool_checks_passed'] == t['tool_checks_total'] == 6 and t['effort_monotonic'], t
    assert t['effort_none_reasoning_chars'] == 0 and t['effort_high_correct'], t
    h = haystack.run(c, sizes=[8000, 1000000])
    assert h['summary'] == {'haystack_8k': 3, 'haystack_1000k': 3}, h['summary']
    pt = h['details'][1]['prompt_tokens']
    assert 0.97e6 < pt < 1.01e6, pt  # sizing via the measured chars/token lands near the target

    with tempfile.TemporaryDirectory() as d:
        a, b = os.path.join(d, 'a.json'), os.path.join(d, 'b.json')
        for out, label in ((a, 'A'), (b, 'B')):
            subprocess.run([sys.executable, os.path.join(HERE, 'run_all.py'), '--base', base, '--quick', '--label', label,
                            '--out', out, '--only', 'refusal,toolcall'], check=True, capture_output=True)
        assert os.path.exists(a.removesuffix('.json') + '.md')
        rb = json.load(open(b)); rb['suites']['refusal']['summary']['refusal_off_rate'] = 0.0
        json.dump(rb, open(b, 'w'))
        diff = subprocess.run([sys.executable, os.path.join(HERE, 'run_all.py'), '--compare', a, b], check=True,
                              capture_output=True, text=True).stdout
        assert '| refusal_off_rate | 12.5 | 0.0 | -12.5 |' in diff, diff
    srv.shutdown()
    print('offline eval self-test OK')


if __name__ == '__main__':
    main()
