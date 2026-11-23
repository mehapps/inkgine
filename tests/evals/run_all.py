"""Run every eval against one OpenAI-compatible endpoint; write a JSON results file and a markdown summary.
  python tests/evals/run_all.py --base http://localhost:30000 [--quick] [--only gsm8k,refusal,haystack,toolcall]
                                [--sizes 32k,128k] [--label ours] [--out results.json]
  python tests/evals/run_all.py --compare a.json b.json     # markdown diff of two results files
LiteLLM: --base https://<gateway> --key-env NAME_OF_ENV_VAR_WITH_KEY (the key itself is never written anywhere)."""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gsm8k  # noqa: E402
import haystack  # noqa: E402
import refusal  # noqa: E402
import toolcall  # noqa: E402
from common import client_from, parser  # noqa: E402

SUITES = {'gsm8k': gsm8k.run, 'refusal': refusal.run, 'toolcall': toolcall.run, 'haystack': haystack.run}


def table(rows, head):
    out = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    return '\n'.join(out + ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows])


def flat(res):
    return {k: v for s in res['suites'].values() for k, v in s['summary'].items()}


def markdown(res):
    return (f"## Evals: {res['label']} ({res['model']} @ {res['base']}, {res['started']}, "
            f"{'quick' if res['quick'] else 'full'})\n\n" + table(sorted(flat(res).items()), ['metric', 'value']))


def compare(a, b):
    fa, fb = flat(a), flat(b)
    rows = []
    for k in sorted(set(fa) | set(fb)):
        x, y = fa.get(k, '-'), fb.get(k, '-')
        num = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (x, y))
        rows.append([k, x, y, round(y - x, 2) if num else ('' if x == y else 'changed')])
    return table(rows, ['metric', a['label'], b['label'], 'delta'])


def main():
    p = parser(__doc__)
    p.add_argument('--only', help='comma list of suites (default all)')
    p.add_argument('--sizes', help='haystack sizes, e.g. 32k,128k,512k,1m')
    p.add_argument('--label', help='name for this run (default: model name)')
    p.add_argument('--compare', nargs=2, metavar=('A', 'B'))
    a = p.parse_args()
    if a.compare:
        ra, rb = (json.load(open(f)) for f in a.compare)
        print(compare(ra, rb))
        return
    c = client_from(a)
    res = dict(label=a.label or c.model, model=c.model, base=c.base, quick=a.quick,
               started=time.strftime('%Y-%m-%d %H:%M:%S %Z'), suites={})
    for name in (a.only.split(',') if a.only else SUITES):
        t = time.time()
        print(f'=== {name}', flush=True)
        kw = {'sizes': [haystack.size_of(s) for s in a.sizes.split(',')]} if name == 'haystack' and a.sizes else {}
        res['suites'][name] = SUITES[name](c, quick=a.quick, **kw)
        res['suites'][name]['secs'] = round(time.time() - t, 1)
    out = a.out or f"evals-{res['label'].replace('/', '_')}-{time.strftime('%Y%m%d-%H%M')}.json"
    json.dump(res, open(out, 'w'), indent=1)
    md = markdown(res)
    open(out.removesuffix('.json') + '.md', 'w').write(md + '\n')
    print(md + f'\n\nwrote {out} and {out.removesuffix(".json")}.md')


if __name__ == '__main__':
    main()
