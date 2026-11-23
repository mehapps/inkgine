"""DSpark shortlist head (dspark.py INK_DSPARK_SHORT): build the static token list offline and measure its coverage. CPU.

--build: token frequencies of local open-source text, tokenized with the Inkling tokenizer, in three domains:
  code  = Python stdlib + venv site-packages *.py, Rust crates (~/.cargo/registry), JS (~/.nvm), C headers;
  prose = *.md / *.rst docs of those packages, /usr/share/doc READMEs/changelogs, package METADATA, gsm8k questions;
  json  = *.json files of those packages (no tokenizer/vocab/merges files).
  Corpus score = weighted mean of per-domain relative frequency (code .35, prose .45, json .2). The local corpus has
  little narrative English, so the rank also uses a model-side frequency proxy: the row norm of the drafter's
  markov_w1 embedding (mtp.safetensors; rows of tokens the drafter rarely saw in training stay small). Alone it covers
  0.90 of held-out next tokens at 32K vs 0.85 for the corpus. Rank = the CORPUS_TOP best corpus ids, then ids by W1 norm,
  then the remaining corpus ids. Writes data/dspark_short.pt: {'rank': int32 ids (specials >= 200000 excluded: the
  drafter always adds them), 'bytes': per domain}.
--cover: held-out coverage (texts never in the corpus): recorded drafter-eval texts (tests/dspark_ablate.py taps),
  tests/oracle.json outputs: P(next token in S0) and in S0 + tokens seen earlier in the same sequence (= the dynamic
  part, INK_DSPARK_DYN slots), for several static sizes.
--eval: the real drafter on the recorded taps (tests/dspark_ablate.load, ~1-2 min per size): greedy acceptance per draft
  position and 1 - TV at T=1 with q restricted to the shortlist (S0 + context tokens) vs the full vocab, the full-vocab
  q mass outside the shortlist, and acceptance by drafter confidence (max prob, T=1): spec.Gate's prior.
--sim: the 32K eval only + a greedy round-cost simulation on its per-anchor traces (fixed widths vs spec.Gate).
Paths are env-free defaults of this host; nothing here runs in the engine.
"""
import glob
import gzip
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import torch  # noqa: E402

OUT = os.path.join(os.path.dirname(HERE), 'data', 'dspark_short.pt')
SPECIAL = 200000
HOME = os.path.expanduser('~')
VENV = glob.glob(os.path.join(os.path.dirname(HERE), '.venv/lib/python3*/site-packages'))
SKIP = ('sglang', 'vllm', 'llama', 'ingine', 'tokenspeed', 'tokenizer', 'vocab', 'merges', '/llm/src', '/llm/upstream',
        '/research-')   # other inference engines (rules) and vocabulary dumps (would flatten the counts)
WEIGHT = {'code': 0.35, 'prose': 0.45, 'json': 0.2}
CAP = {'code': 24 << 20, 'prose': 20 << 20, 'json': 5 << 20}
CORPUS_TOP = 8192


def files(roots, pats, cap_files=4000):
    out = []
    for r in roots:
        for p in pats:
            out += glob.glob(os.path.join(r, '**', p), recursive=True)
    out = sorted(f for f in set(out) if os.path.isfile(f) and not any(s in f.lower() for s in SKIP))
    step = max(1, len(out) // cap_files)
    return out[::step]


def read(f, lim=200_000):
    try:
        if f.endswith('.gz'):
            with gzip.open(f, 'rt', errors='ignore') as fh:
                return fh.read(lim)
        with open(f, errors='ignore') as fh:
            return fh.read(lim)
    except OSError:
        return ''


def corpus():
    py = ['/usr/lib/python3.14'] + VENV
    pkg = VENV + [f'{HOME}/.cargo/registry', f'{HOME}/.nvm/versions']
    src = {'code': (files(py, ['*.py'], 6000) + files([f'{HOME}/.cargo/registry'], ['*.rs'], 1500)
                    + files([f'{HOME}/.nvm/versions'], ['*.js', '*.ts'], 1000) + files(['/usr/include'], ['*.h'], 800)),
           'prose': (files(pkg, ['*.md', '*.rst'], 5000) + files(['/usr/share/doc'], ['README*', 'changelog*', '*.txt'], 800)
                     + files(VENV, ['METADATA'], 2000)),
           'json': files(pkg, ['*.json'], 3000)}
    for dom, fs in src.items():
        texts, n = [], 0
        for f in fs:
            t = read(f)
            if t and '\x00' not in t:
                texts.append(t)
                n += len(t)
            if n >= CAP[dom]:
                break
        if dom == 'prose':
            import json
            for line in open(os.path.join(HERE, 'evals', 'data', 'gsm8k_test.jsonl')):
                texts.append(json.loads(line)['question'])
        yield dom, texts


def tokenizer():
    from tokenizers import Tokenizer
    from engine import MODEL_DIR
    return Tokenizer.from_file(os.path.join(MODEL_DIR, 'tokenizer.json'))


def build():
    tok = tokenizer()
    V = tok.get_vocab_size()
    score, nbytes = torch.zeros(V, dtype=torch.float64), {}
    for dom, texts in corpus():
        t0 = time.time()
        c = torch.zeros(V, dtype=torch.float64)
        for a in range(0, len(texts), 256):
            for e in tok.encode_batch(texts[a:a + 256], add_special_tokens=False):
                c += torch.bincount(torch.tensor(e.ids, dtype=torch.long), minlength=V).double()
        nbytes[dom] = sum(map(len, texts))
        print(f'{dom}: {len(texts)} texts, {nbytes[dom] / 1e6:.1f} MB, {int(c.sum())} tokens, '
              f'{int((c > 0).sum())} distinct, {time.time() - t0:.0f} s', flush=True)
        score += WEIGHT[dom] * c / c.sum()
    score[SPECIAL:] = -1
    corp = score.argsort(descending=True)
    corp = corp[score[corp] > 0].tolist()
    from safetensors import safe_open
    import dspark
    with safe_open(dspark.MTP, 'pt') as f:
        w1 = f.get_tensor('markov_head.markov_w1.weight')[:SPECIAL].float().norm(dim=1)
    rank, seen = [], set()
    for t in corp[:CORPUS_TOP] + w1.argsort(descending=True).tolist() + corp[CORPUS_TOP:]:
        if t not in seen:
            seen.add(t)
            rank.append(t)
    rank = torch.tensor(rank[:65536], dtype=torch.int32)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    torch.save({'rank': rank, 'bytes': nbytes, 'weight': WEIGHT, 'vocab': V}, OUT)
    print(f'wrote {OUT}: {len(rank)} ranked ids')


def heldout():
    """[(name, ids)] of texts not in the corpus."""
    import json
    tok = tokenizer()
    out = []
    import dspark_ablate as A
    if os.path.exists(A.OUT):
        for name, d in torch.load(A.OUT, map_location='cpu').items():
            out.append((name, d['ids'].tolist()))
    o = json.load(open(os.path.join(HERE, 'oracle.json')))
    for k, v in o['teacher'].items():
        out.append((f'oracle-{k}', list(v['ids']) + list(v['output_ids'])))
    for k, v in o['chat'].items():
        out.append((f'chat-{k}', tok.encode((v.get('reasoning_content') or '') + (v['content'] or ''),
                                            add_special_tokens=False).ids))
    return out


def short_set(m0, rank=None):
    """Static shortlist of size m0: every special + the top-ranked ids (dspark.static_short does the same)."""
    import dspark
    return set(dspark.static_short(m0, rank).tolist())


def cover(sizes=(8192, 16384, 24576, 32768, 49152)):
    texts = heldout()
    print(f'held-out: {[(n, len(i)) for n, i in texts]}')
    for m0 in sizes:
        S = short_set(m0)
        row = []
        tot = [0, 0, 0]
        for name, ids in texts:
            seen, a, b = set(), 0, 0
            for i, t in enumerate(ids):
                if i:
                    a += t in S
                    b += t in S or t in seen
                seen.add(t)
            row.append(f'{name} {a / (len(ids) - 1):.3f}/{b / (len(ids) - 1):.3f}')
            tot = [tot[0] + a, tot[1] + b, tot[2] + len(ids) - 1]
        print(f'M0 {m0:6d}: all {tot[0] / tot[2]:.4f} static, {tot[1] / tot[2]:.4f} with context | ' + ', '.join(row),
              flush=True)


@torch.no_grad()
def evaluate(sizes=(32768, 49152), stride=4, rows=7, chunk=16, full=True):
    import dspark_ablate as A
    data_all, ds = A.load()
    torch.set_num_threads(int(os.environ.get('INK_THREADS', 6)))
    V = ds.U.shape[0]
    bins = torch.tensor([0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.01])
    for gname, names, m0 in [(g, n, m) for g, n in (('code+json', ('code1', 'code2', 'json1')),
                                                  ('chat+prose', ('chat1', 'chat2', 'prose1')))
                             for m in (0,) * full + tuple(sizes)]:
        data = {k: data_all[k] for k in names}
        t0 = time.time()
        stat = torch.zeros(rows, 4, dtype=torch.float64)      # per row: chain den, chain num, 1-TV sum, q mass outside
        cb = torch.zeros(len(bins) - 1, 2)                     # confidence bin: rows, argmax == target argmax
        base = torch.zeros(V, dtype=torch.bool)
        if m0:
            base[dspark_static(m0)] = True
        n_all, trace = 0, []
        for d in data.values():
            ids, T = d['ids'], len(d['ids'])
            taps = torch.stack([A.tap(d, k) for k in ds.cfg.taps])
            st = ds.new_state()
            ds.extend(st, taps)
            e = ds.head_ctx(taps)
            P = torch.arange(A.START, T - rows, stride)
            for a in range(0, len(P), chunk):
                p = P[a:a + chunk]
                ar = p[:, None] + torch.arange(rows)
                lg = ds.rows_tf(st, ids[p], p, ids[ar], e[p - 1])                       # [N, rows, V]
                qf = lg.softmax(-1)
                if m0:                                       # shortlist = static + tokens up to the anchor
                    mk = base.expand(len(p), V).clone()
                    for j, pj in enumerate(p.tolist()):
                        mk[j, ids[:pj + 1]] = True
                    out = (qf * ~mk[:, None]).sum(-1)
                    lg = lg.masked_fill(~mk[:, None], float('-inf'))
                else:
                    out = torch.zeros(lg.shape[:2])
                q = lg.softmax(-1)
                g = d['g'][ar]
                m = lg.argmax(-1) == g
                agree = ids[ar + 1] == g
                ok = torch.ones(len(p), dtype=torch.bool)
                conf = q.max(-1).values
                for r in range(rows):
                    stat[r, 0] += ok.sum()
                    stat[r, 1] += (ok & m[:, r]).sum()
                    sel = ok
                    bi = torch.bucketize(conf[sel, r], bins) - 1
                    cb.index_add_(0, bi, torch.stack((torch.ones(int(sel.sum())), m[sel, r].float()), 1))
                    ok = ok & m[:, r] & agree[:, r]
                for j in range(len(p)):                      # accepted drafts (a draft whose text disagreed ends it)
                    a = 0
                    while a < rows and m[j, a]:
                        a += 1
                        if not agree[j, a - 1]:
                            break
                    trace.append((conf[j].tolist(), a))
                pv, pi = d['pv'][ar], d['pi'][ar]
                stat[:, 2] += torch.minimum(pv, q.gather(-1, pi)).sum(-1).sum(0).double()
                stat[:, 3] += out.sum(0).double()
                n_all += len(p)
        acc = stat[:, 1] / stat[:, 0].clamp(min=1)
        tv = stat[:, 2] / n_all
        tau, tau_s = 1 + float(acc.cumprod(0).sum()), 1 + float(tv.cumprod(0).sum())
        print(f'{gname} M0 {m0 or V:6d} ({time.time() - t0:.0f} s): tau {tau:.2f} tau_s {tau_s:.2f} | a '
              + ' '.join(f'{x:.3f}' for x in acc) + ' | 1-TV ' + ' '.join(f'{x:.3f}' for x in tv)
              + ' | full-q mass outside ' + ' '.join(f'{x:.4f}' for x in stat[:, 3] / n_all), flush=True)
        print('   confidence (reached rows): ' + ', '.join(
            f'[{bins[i]:.2f},{min(1, bins[i + 1]):.2f}) n {int(cb[i, 0])} acc {cb[i, 1] / max(1, cb[i, 0]):.2f}'
            for i in range(len(cb)) if cb[i, 0]), flush=True)
        if m0 == 32768:
            simulate(trace)


def simulate(trace):
    """Greedy round-cost model on the per-anchor traces: tokens/ms vs plain for fixed widths (what Width converges to)
    and for spec.Gate's pick (in-sample prior, so slightly optimistic), at drafter costs of today (3.2 ms), and after
    the shortlist (~1.2 ms est.) / everything (~0.8 ms est.). Verify costs: spec.Width.STEP_MS (GPU 11-22)."""
    import spec
    T = spec.Width.STEP_MS
    for dms in (3.2, 1.2, 0.8):
        fixed = []
        for k in range(1, 8):
            tok = sum(min(a, k) + 1 for _, a in trace)
            fixed.append(tok / (len(trace) * (T[k] + dms)) * T[0])
        g = spec.Gate(draft_ms=dms)
        tok = ms = 0
        for c, a in trace:
            m = g.pick(c, 7)
            tok, ms = tok + min(a, m) + 1, ms + T[m] + dms
        print(f'   sim drafter {dms} ms: fixed k=1..7 x' + ' '.join(f'{x:.2f}' for x in fixed)
              + f' | gate x{tok / ms * T[0]:.2f} (controller turns drafting off below x1.0)', flush=True)


def dspark_static(m0):
    import dspark
    return dspark.static_short(m0).long()


if __name__ == '__main__':
    if '--build' in sys.argv:
        build()
    if '--cover' in sys.argv:
        cover()
    if '--eval' in sys.argv:
        evaluate()
    if '--sim' in sys.argv:          # only the 32K shortlist + the round-cost simulation
        evaluate(sizes=(32768,), full=False)
