"""Compare our engine against tests/oracle.json (teacher-forced top-1 agreement + greedy continuation).
Gate is relative to the model's own noise: per prompt we also run ourselves with the input embeddings nudged by +-1
bf16 ulp; our-vs-oracle top-1 must be >= (ours-vs-nudged top-1) - 0.03, and at most 3% of positions may be
disagreements where neither side is a near-tie (top-2 prob gap < 0.1). Those are all listed.
Usage (GPU window, nothing else on the GPU): python tests/compare.py      CPU logic check: python tests/compare.py --selftest"""
import json
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

SLACK, HARD_MAX, TIE = 0.03, 0.03, 0.1


def top1(entry):
    """SGLang top-logprob entry: list of [logprob, token_id, text?] -> (token_id, logprob)."""
    best = max(entry, key=lambda x: x[0])
    return int(best[1]), float(best[0])


def near_tie(entry, ours_v):
    po = sorted((math.exp(x[0]) for x in entry), reverse=True)
    pu = [math.exp(v) for v in ours_v]
    return (len(po) > 1 and po[0] - po[1] < TIE) or (len(pu) > 1 and pu[0] - pu[1] < TIE)


def show_disagreement(tok, ids, pos, entry, ours_i, ours_v, tie):
    """Print one top-1 disagreement: context tail, both top-5 with probs, near-tie flag."""
    orc = sorted(entry, key=lambda x: -x[0])[:5]
    fmt = lambda ts, ps: ', '.join(f'{tok.decode([int(t)])!r} {p:.3f}' for t, p in zip(ts, ps))
    print(f'  pos {pos}{" (near-tie)" if tie else " (NOT a near-tie)"}: ...{tok.decode(ids[max(0, pos - 24):pos])!r}')
    print(f'    oracle: {fmt([x[1] for x in orc], [math.exp(x[0]) for x in orc])}')
    print(f'    ours  : {fmt(ours_i, [math.exp(v) for v in ours_v])}')


def score(o, rows, nudged, show=None):
    """Teacher-forced stats over the oracle's scored positions. rows / nudged: {row r: (top5 ids, top5 logprobs)}, row r
    predicts token r+1. -> dict(n, agree, base, ties, hard=[positions], dlp)."""
    s = dict(n=0, agree=0, base=0, ties=0, hard=[], dlp=[])
    for j, entry in enumerate(o['input_top_logprobs'] or []):
        pos = o['start'] + j
        if not entry or pos - 1 not in rows:
            continue
        t_or, lp_or = top1(entry)
        ours_i, ours_v = rows[pos - 1]
        s['n'] += 1
        s['agree'] += ours_i[0] == t_or
        s['base'] += ours_i[0] == nudged[pos - 1][0][0]
        if ours_i[0] != t_or:
            tie = near_tie(entry, ours_v)
            s['ties'] += tie
            if not tie:
                s['hard'].append(pos)
            if show:
                show(pos, entry, ours_i, ours_v, tie)
        if t_or in ours_i:
            s['dlp'].append(abs(ours_v[ours_i.index(t_or)] - lp_or))
    return s


def judge(s):
    """-> (ok, why) for one prompt's score dict."""
    if s['n'] == 0:
        return True, 'no scored positions'
    rate, base, hard = s['agree'] / s['n'], s['base'] / s['n'], len(s['hard']) / s['n']
    why = []
    if rate < base - SLACK:
        why.append(f'top-1 {rate:.3f} < nudge baseline {base:.3f} - {SLACK}')
    if hard > HARD_MAX:
        why.append(f'{len(s["hard"])} non-near-tie disagreements = {hard:.1%} > {HARD_MAX:.0%}')
    return not why, '; '.join(why) or 'ok'


def teacher(m, ids, start, chunk, nudge):
    """Chunked prefill of ids; -> (rows {r: (top5 ids, top5 lp)} from start-1 on, last hidden, seq)."""
    from gpu_check import ulp_nudge  # same +-1 ulp nudge as gpu_check's noise baseline
    seq, rows = m.new_seq(), {}
    for a in range(0, len(ids), chunk):
        b = min(len(ids), a + chunk)
        x = m.embed(ids[a:b])
        f = m.forward(ulp_nudge(x, seed=1 + a) if nudge else x, seq)
        lo = max(a, (start if start is not None else len(ids)) - 1)
        for r0 in range(lo, b, 512):
            r1 = min(b, r0 + 512)
            v, i = torch.log_softmax(m.logits(f[r0 - a:r1 - a]), -1).topk(5, -1)
            for k in range(r1 - r0):
                rows[r0 + k] = (i[k].tolist(), v[k].tolist())
    return rows, f, seq


def main():
    from engine import Engine
    orc = json.load(open(os.path.join(HERE, 'oracle.json')))
    eng = Engine('cuda', max_ctx=1 << 17, vision=False)
    m = eng.model
    ok_all = True
    for name, o in orc['teacher'].items():
        ids, start = o['ids'], o['start']
        rows, f, seq = teacher(m, ids, start, eng.chunk, False)
        gen, f_last = [], f[-1:]                               # greedy continuation
        for _ in range(len(o.get('output_ids') or [])):
            t = int(m.logits(f_last)[0].argmax())
            gen.append(t)
            f_last = m.forward(m.embed([t]), seq)
        del seq, f, f_last                                     # pages back before the nudged run
        nudged = teacher(m, ids, start, eng.chunk, True)[0] if start is not None else {}
        show = lambda pos, e, oi, ov, tie: show_disagreement(eng.tok, ids, pos, e, oi, ov, tie)
        s = score(o, rows, nudged, show)
        ok, why = judge(s)
        same = next((k for k, (x, y) in enumerate(zip(gen, o['output_ids'])) if x != y), len(gen))
        n, agree = s['n'], s['agree']
        print(f'{name}: {len(ids)} tokens, top-1 agreement {agree}/{n} = {agree / max(n, 1):.3f} (nudge baseline '
              f'{s["base"] / max(n, 1):.3f}), mean |dlogprob| of oracle top-1 {sum(s["dlp"]) / max(len(s["dlp"]), 1):.3f} '
              f'({s["ties"]}/{n - agree} disagreements are near-ties; non-near-tie at {s["hard"]}); greedy matches '
              f'oracle for {same}/{len(gen)} tokens -> {why}')
        print('   ours  :', repr(eng.tok.decode(gen)), '\n   oracle:', repr(o.get('text')))
        ok_all &= ok and same >= min(8, len(gen))
    print('PASS' if ok_all else 'FAIL', f'(want top-1 >= nudge baseline - {SLACK}, <= {HARD_MAX:.0%} non-near-tie '
          'disagreements, greedy >= 8 tokens, on every prompt)')
    sys.exit(0 if ok_all else 1)


def selftest():
    """CPU check of score/judge with fake oracle entries and rows (no weights)."""
    lp = math.log
    sure = lambda t: [[lp(0.9), t], [lp(0.05), t + 1]]          # confident: gap 0.85
    tie = lambda t: [[lp(0.45), t], [lp(0.40), t + 1]]          # near-tie: gap 0.05
    row = lambda t, p=0.9: ([t, t + 1, 0, 0, 0], [lp(p), lp(1 - p - 0.01), -9, -9, -9])
    N = 100
    o = {'start': 1, 'input_top_logprobs': [sure(10 + j) if j % 10 else tie(10 + j) for j in range(N)]}
    good = {j: row(10 + j) for j in range(N)}                   # agrees everywhere
    s = score(o, good, good)
    assert (s['n'], s['agree'], s['base'], s['hard']) == (N, N, N, []) and judge(s)[0], s
    # 5 near-tie disagreements (j % 10 == 0 -> oracle near-tie), nudge baseline 0.95: ok
    ours = {**good, **{j: row(11 + j) for j in range(0, 50, 10)}}
    nudged = {**good, **{j: row(99) for j in range(5)}}
    s = score(o, ours, nudged)
    assert s['agree'] == 95 and s['ties'] == 5 and not s['hard'], s
    assert s['base'] == 91 and judge(s)[0], (s, judge(s))       # nudged differs at rows 0-4, ours also at 10-40
    # 4 confident disagreements (oracle sure, ours sure): 4% > 3% -> fail, positions listed
    ours2 = {**good, **{j: row(11 + j) for j in (1, 2, 3, 4)}}
    s = score(o, ours2, good)
    assert s['hard'] == [2, 3, 4, 5] and not judge(s)[0] and 'non-near-tie' in judge(s)[1], s
    # ours a near-tie at a confident oracle position counts as a tie
    ours3 = {**good, **{j: row(11 + j, p=0.47) for j in (1, 2, 3, 4)}}
    s = score(o, ours3, ours3)                                  # our nudged run never moves: baseline 1.0
    assert s['ties'] == 4 and not s['hard'] and judge(s)[0] is False, s  # 0.96 < baseline 1.0 - 0.03
    s = score(o, ours3, {**ours3, **{j: row(99) for j in (5, 6, 7)}})
    assert judge(s)[0], (s, judge(s))                           # baseline 0.97 -> 0.96 >= 0.94
    # rate below baseline - 0.03 fails even when all are near-ties
    s = dict(n=100, agree=90, base=95, ties=10, hard=[], dlp=[])
    assert not judge(s)[0] and 'baseline' in judge(s)[1]
    assert judge(dict(n=0, agree=0, base=0, ties=0, hard=[], dlp=[]))[0]
    print('selftest OK')


if __name__ == '__main__':
    selftest() if '--selftest' in sys.argv else main()
