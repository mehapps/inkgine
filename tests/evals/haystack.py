"""Code-haystack needle: a deterministic synthetic Python codebase of ~N tokens with 3 needle facts at depths 15/50/85%
(two functions returning a unique code, one default-argument value), surrounded by near-miss decoys with similar
names. One request per size, thinking off, greedy; score = exact needle values found in the answer (0-3).
Sizes are prompt tokens; the chars/token ratio is measured once with a small request (usage.prompt_tokens).
Usage: python tests/evals/haystack.py [--base URL] [--sizes 32k,128k,512k,1m] [--quick]"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import client_from, parser  # noqa: E402

VERBS = 'load save parse merge resolve compute validate render fetch flush encode decode sync index score'.split()
NOUNS = ('ledger tenant invoice cursor shard bucket session vault token config schema widget report queue '
         'profile batch cache route payload channel').split()
FIELDS = 'items rows values weights counts sizes offsets ids'.split()
QUESTION = ('Answer from the codebase above, exactly and briefly, one line each:\n'
            '1. What string does `{a}` return?\n2. What string does `{b}` return?\n'
            '3. What is the default value of the `retries` parameter of `{c}`?')


def code(rng):
    L = 'ABCDEFGHJKLMNPQRSTUVWXYZ'
    return f'{rng.choice(L)}{rng.choice(L)}-{rng.randrange(1000, 10000)}-{rng.choice(L)}{rng.choice(L)}'


def needles(seed):
    rng = random.Random(f'needle{seed}')
    sfx = rng.sample(range(100, 1000), 6)  # distinct: decoys share the stem, so the model must match the full name
    a, b, c = (f'{v}_{n}_{s}' for (v, n), s in
               zip((('resolve', 'vault_token'), ('derive', 'shard_secret'), ('replay', 'journal_batch')), sfx))
    va, vb, vc = code(rng), code(rng), str(rng.randrange(11, 98))
    decoys = sfx[3:]
    blocks = {0.15: f'def {a}(ctx):\n    """Return the vault token alias for ctx."""\n    return "{va}"\n',
              0.50: f'def {b}(ctx, salt=None):\n    """Derive the shard secret."""\n    return "{vb}"\n',
              0.85: f'def {c}(journal, retries={vc}, backoff=0.5):\n    """Replay a journal batch."""\n'
                    f'    for _ in range(retries):\n        if journal.flush():\n            return True\n'
                    f'    return False\n'}
    rngd = random.Random(f'decoy{seed}')
    decoy = [f'def resolve_vault_token_{decoys[0]}(ctx):\n    return "{code(rngd)}"\n',
             f'def derive_shard_secret_{decoys[1]}(ctx, salt=None):\n    return "{code(rngd)}"\n',
             f'def replay_journal_batch_{decoys[2]}(journal, retries={rngd.randrange(11, 98)}, backoff=0.5):\n'
             f'    return journal.flush()\n']
    return blocks, decoy, (a, b, c), (va, vb, vc)


def module(rng, i):
    lines = [f'# ---- file: pkg/{rng.choice(NOUNS)}_{i}.py ----', 'import math', '']
    for _ in range(rng.randrange(6, 12)):
        v, n, f = rng.choice(VERBS), rng.choice(NOUNS), rng.choice(FIELDS)
        a, b, k = rng.randrange(2, 50), rng.randrange(2, 9), rng.randrange(100, 10000)
        lines += [f'def {v}_{n}_{rng.randrange(1000, 100000)}(record, limit={k}):',
                  f'    """{v.capitalize()} the {n} {f} of a record, capped at limit."""',
                  '    total = 0', f'    for item in record.get("{f}", []):',
                  f'        if item > {a}:', f'            total += item * {b}',
                  f'        elif item % {b} == 0:', '            total -= 1', '    return min(total, limit)', '']
    return '\n'.join(lines) + '\n'


def build(n_chars, seed=0):
    rng = random.Random(f'hay{seed}')
    mods, size, i = [], 0, 0
    while True:  # stop under n_chars: the target is a ceiling (1M must stay inside the context limit)
        m = module(rng, i); size += len(m); i += 1
        if size > n_chars - 600:  # leave room for the needle and decoy blocks
            break
        mods.append(m)
    blocks, decoy, names, vals = needles(seed)
    for depth, d in zip((0.35, 0.7, 0.05), decoy):  # each decoy far from its needle
        mods.insert(int(len(mods) * depth), d)
    for depth in sorted(blocks, reverse=True):  # insert from the back so earlier depths stay put
        mods.insert(int(len(mods) * depth), blocks[depth])
    doc = ''.join(mods)
    msgs = [{'role': 'user', 'content': 'Here is a Python codebase.\n\n' + doc + '\n\n' +
             QUESTION.format(a=names[0], b=names[1], c=names[2])}]
    return msgs, names, vals


def size_of(s):
    s = s.strip().lower()
    mult = 1_000_000 if s.endswith('m') else 1000 if s.endswith('k') else 1
    return int(float(s.rstrip('km')) * mult)


def run(client, quick=False, sizes=None):
    sizes = sizes or ([8000, 32000] if quick else [32000, 128000, 512000, 1000000])
    cal_msgs, _, _ = build(12000)
    r = client.chat(cal_msgs, max_tokens=1, effort='off')
    pt = r['usage'].get('prompt_tokens')
    cpt = len(cal_msgs[0]['content']) / pt if pt else 3.0
    print(f'chars/token {cpt:.2f} (from {"server" if pt else "default"})', flush=True)
    summary, details = {}, []
    for n in sizes:
        msgs, names, vals = build(int((n - 120) * cpt), seed=n)
        r = client.chat(msgs, max_tokens=200, effort='off')
        hits = [v in r['content'] for v in vals]
        key = f'haystack_{n // 1000}k'
        summary[key] = sum(hits)
        details.append(dict(size=n, prompt_tokens=r['usage'].get('prompt_tokens'), hits=hits, expected=vals,
                            names=names, secs=round(r['secs'], 1), error=r['error'], answer=r['content'][:400]))
        print(f"{key}: {sum(hits)}/3, prompt_tokens {r['usage'].get('prompt_tokens')}, {r['secs']:.0f}s "
              f"{r['error'] or ''}", flush=True)
    return dict(summary=summary, details=details)


if __name__ == '__main__':
    import json
    p = parser(__doc__)
    p.add_argument('--sizes', help='comma list, e.g. 32k,128k,512k,1m')
    a = p.parse_args()
    res = run(client_from(a), a.quick, [size_of(s) for s in a.sizes.split(',')] if a.sizes else None)
    print(json.dumps(res['summary'], indent=1))
    if a.out:
        json.dump(res, open(a.out, 'w'), indent=1)
