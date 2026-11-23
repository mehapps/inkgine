"""GSM8K subset: the first N (default 100) questions of the public test split, greedy, exact-number match.
Variants: thinking off and reasoning_effort=low. Context only (not a gate): maya reports 73-77% on 96 questions,
greedy, reasoning off, for the same base model.
Usage: python tests/evals/gsm8k.py [--base URL] [--quick] [--n 100] [--variants off,low] [--parallel 1]"""
import concurrent.futures as cf
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import client_from, parser  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'gsm8k_test.jsonl')
# openai/grade-school-math grade_school_math/data/test.jsonl (1319 lines), fetched 2026-11-22
SHA256 = '3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14'
PROMPT = ('Solve this math word problem. Think step by step briefly, then give the final answer on its own last line '
          'as "Answer: <number>" (digits only, no units).\n\n{q}')
NUM = r'-?\d[\d,]*(?:\.\d+)?'


def load(n):
    raw = open(DATA, 'rb').read()
    assert hashlib.sha256(raw).hexdigest() == SHA256, f'{DATA} checksum mismatch'
    rows = [json.loads(line) for line in raw.decode().splitlines()][:n]
    return [(r['question'], norm(r['answer'].split('####')[-1])) for r in rows]


def norm(s):
    s = s.strip().replace(',', '').replace('$', '').rstrip('.')
    try:
        f = float(s)
    except ValueError:
        return None
    return str(int(f)) if f == int(f) else str(f)


def extract(text):
    """'Answer: N' (last one) > \\boxed{N} > last number in the text."""
    text = text.replace('**', '')
    for pat in (r'Answer\s*[:：]\s*\$?\s*(' + NUM + ')', r'\\boxed\{\s*\$?(' + NUM + r')'):
        m = re.findall(pat, text, re.I)
        if m:
            return norm(m[-1])
    m = re.findall(NUM, text)
    return norm(m[-1]) if m else None


def run(client, quick=False, n=100, variants=('off', 'low'), parallel=1):
    qs = load(20 if quick else n)
    summary, details = {}, []
    for v in variants:
        def one(i):
            q, gold = qs[i]
            r = client.chat([{'role': 'user', 'content': PROMPT.format(q=q)}], effort=v,
                            max_tokens=1024 if v == 'off' else 6144)
            got = extract(r['content'])
            return dict(variant=v, i=i, gold=gold, got=got, ok=got == gold, finish=r['finish'], error=r['error'],
                        completion_tokens=r['usage'].get('completion_tokens'), secs=round(r['secs'], 1),
                        reasoning_chars=len(r['reasoning']), tail=r['content'][-160:])
        with cf.ThreadPoolExecutor(parallel) as ex:
            rows = list(ex.map(one, range(len(qs))))
        details += rows
        ok = sum(r['ok'] for r in rows)
        summary[f'gsm8k_{v}_acc'] = round(100 * ok / len(rows), 1)
        summary[f'gsm8k_{v}_errors'] = sum(bool(r['error']) for r in rows)
        summary[f'gsm8k_{v}_truncated'] = sum(r['finish'] == 'length' for r in rows)
        summary[f'gsm8k_{v}_secs'] = round(sum(r['secs'] for r in rows), 1)
        print(f'gsm8k {v}: {ok}/{len(rows)} ({summary[f"gsm8k_{v}_acc"]}%)', flush=True)
    summary['gsm8k_n'] = len(qs)
    return dict(summary=summary, details=details)


if __name__ == '__main__':
    p = parser(__doc__)
    p.add_argument('--n', type=int, default=100)
    p.add_argument('--variants', default='off,low')
    p.add_argument('--parallel', type=int, default=1)
    a = p.parse_args()
    res = run(client_from(a), a.quick, a.n, a.variants.split(','), a.parallel)
    print(json.dumps(res['summary'], indent=1))
    for r in res['details']:
        if not r['ok']:
            print(f"MISS {r['variant']} #{r['i']}: gold {r['gold']} got {r['got']} {r['error'] or ''} ...{r['tail'][-80:]!r}")
    if a.out:
        json.dump(res, open(a.out, 'w'), indent=1)
