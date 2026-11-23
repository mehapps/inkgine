"""DSpark drafter wiring ablation (SPEC-dspark.md sec. 10), CPU, real weights, no engine edits.

--record: teacher-forced CPU prefill of a few fixed prompts (tests/expert_stats.py's prose/code/chat/json), with
Model.layer wrapped to keep the bf16 `h` entering every one of the 48 visits, the output of visit 47 and the final
normed state. Saved to $INK_LOGS/ink-dspark-taps.pt (~30 min detached, see NOTES.md). Target logits are not stored:
they are softcap(final @ unembed), recomputed offline.
default: offline grid over the UNCONFIRMED drafter choices on the recorded taps (dspark.Cfg fields), acceptance per
draft position (greedy prefix match, and 1 - TV at temperature 1).
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import torch  # noqa: E402

import model as M  # noqa: E402

OUT = os.path.join(os.environ.get('INK_LOGS', os.path.expanduser('~/llm/logs')), 'ink-dspark-taps.pt')
PROMPTS = ('code1', 'chat1', 'prose1', 'json1', 'code2', 'chat2')


def record():
    import chat
    import expert_stats
    from tokenizers import Tokenizer
    from engine import MODEL_DIR
    tok = Tokenizer.from_file(os.path.join(MODEL_DIR, 'tokenizer.json'))
    torch.set_num_threads(int(os.environ.get('INK_THREADS', 8)))
    m = M.Model(MODEL_DIR, 'cpu', max_ctx=4096)
    rec, orig = {}, M.Model.layer

    def layer(self, L, h, blocks, msg, seq, v, pos):
        rec[v] = h.clone()
        out = orig(self, L, h, blocks, msg, seq, v, pos)
        if v == 47:
            rec[48] = out[0].clone()
        return out
    M.Model.layer = layer
    data = torch.load(OUT) if os.path.exists(OUT) else {}
    src = expert_stats.prompts()
    for name in PROMPTS:
        if name in data:
            continue
        ids = chat.render(tok, src[name], effort=0.9)
        t0 = time.time()
        rec.clear()
        with torch.no_grad():
            f = m.forward(m.embed(ids), m.new_seq())
        assert len(rec) == 49, len(rec)
        # vis [49, T, H]: 0..47 = input of visit v, 48 = output of visit 47 (pre final attn-res); final = normed state
        data[name] = {'ids': torch.tensor(ids), 'vis': torch.stack([rec[v] for v in range(49)]), 'final': f.clone()}
        print(f'{name}: {len(ids)} tokens, prefill {time.time() - t0:.0f} s', flush=True)
        torch.save(data, OUT)
    return data


def tap(d, k):
    """hs index k (dspark.Cfg.taps): < 48 = h entering visit k, 48 = final normed, 49 = output of visit 47."""
    return d['final'] if k == 48 else d['vis'][48 if k == 49 else k]


def load(dtype=torch.float32, topn=512):
    """Recorded taps + target per-position argmax / top-n probs (T = 1) + the drafter with the target's
    embed/unembed. Target targets are cached next to the taps."""
    from safetensors import safe_open
    import dspark
    from engine import MODEL_DIR
    data = torch.load(OUT)
    f = safe_open(os.path.join(MODEL_DIR, 'model-00001-of-00001.safetensors'), 'pt', device='cpu')
    U = f.get_tensor('model.llm.unembed.weight')[:M.VOCAB].to(dtype)
    ds = dspark.DSpark(embed=f.get_tensor('model.llm.embed.weight'), unembed=U,
                       embed_norm=f.get_tensor('model.llm.embed_norm.weight').float(), dtype=dtype)
    cache = OUT.replace('.pt', '-target.pt')
    tg = torch.load(cache) if os.path.exists(cache) else {}
    for name, d in data.items():
        if name not in tg:
            g, pv, pi = [], [], []
            for a in range(0, len(d['ids']), 128):
                lg = d['final'][a:a + 128].to(dtype) @ U.t()
                lg = 30 * torch.tanh(lg / 30)                                   # Model.logits
                pr = lg.softmax(-1)
                g.append(lg.argmax(-1))
                v, i = pr.topk(topn, -1)
                pv.append(v), pi.append(i)
            tg[name] = {'g': torch.cat(g), 'pv': torch.cat(pv), 'pi': torch.cat(pi)}
            torch.save(tg, cache)
        d.update(tg[name])
    return data, ds


START = 32  # leading template tokens are not anchors


@torch.no_grad()
def evaluate(ds, data, rows=8, stride=4, chunk=32):
    """Acceptance per draft row r (draft r+1, target logits at P+r) over anchors P = START, START+stride, ...
    greedy: m = drafter argmax (head fed the text tokens) == target argmax; a_r = P(m_r | m_<r and the text agreed
    with the target on the prefix, so text prev == real draft prev): a chained estimator of greedy acceptance.
    sampled (T = 1): 1 - TV = sum min(p, q) over the target's top-n tokens (lower bound; residual p mass reported)."""
    c = ds.cfg
    rows = min(rows, 8 if c.row0 else 7)
    num, den = torch.zeros(rows), torch.zeros(rows)
    raw, tv, n_all, resid = torch.zeros(rows), torch.zeros(rows), 0, 0.0
    for d in data.values():
        ids, T = d['ids'], len(d['ids'])
        taps = torch.stack([tap(d, k) for k in c.taps])
        st = ds.new_state()
        ds.extend(st, taps)
        e = ds.head_ctx(taps)
        P = torch.arange(START, T - rows, stride)
        for a in range(0, len(P), chunk):
            p = P[a:a + chunk]
            ar = p[:, None] + torch.arange(rows)                             # target logits rows P+r
            lg = ds.rows_tf(st, ids[p], p, ids[ar], e[p - 1])                 # [N, rows, V]
            g = d['g'][ar]
            m = lg.argmax(-1) == g
            agree = ids[ar + 1] == g
            ok = torch.ones(len(p), dtype=torch.bool)
            for r in range(rows):
                den[r] += ok.sum()
                num[r] += (ok & m[:, r]).sum()
                ok = ok & m[:, r] & agree[:, r]
            raw += m.float().sum(0)
            q = lg.softmax(-1)
            pv, pi = d['pv'][ar], d['pi'][ar]
            tv += torch.minimum(pv, q.gather(-1, pi)).sum(-1).sum(0)
            resid = max(resid, float(1 - pv.sum(-1).min()))
            n_all += len(p)
    acc = num / den.clamp(min=1)
    tvm = tv / n_all
    return {'a': acc.tolist(), 'raw': (raw / n_all).tolist(), 'tv': tvm.tolist(), 'n': n_all, 'den': den.tolist(),
            'tau': 1 + float(acc.cumprod(0).sum()), 'tau_s': 1 + float(tvm.cumprod(0).sum()), 'resid': resid}


def fmt(c, r, base=None):
    from dataclasses import asdict
    diff = {k: v for k, v in asdict(c).items() if base is None or getattr(base, k) != v}
    return (f'a1 {r["a"][0]:.3f} tv1 {r["tv"][0]:.3f} tau {r["tau"]:.2f} tau_s {r["tau_s"]:.2f} | '
            f'a {" ".join(f"{x:.2f}" for x in r["a"])} | tv {" ".join(f"{x:.2f}" for x in r["tv"])} | n {r["n"]} | {diff}')


def grid(stride):
    """Stage A: backbone only (head 'none', row 1): full product over the riskiest backbone choices, then
    coordinate descent on the rest. Stage B: head choices on the stage-A winner, scored by tau over 8 rows."""
    import itertools
    import json
    from dataclasses import asdict, replace
    import dspark
    torch.set_num_threads(int(os.environ.get('INK_THREADS', 6)))
    data, ds = load()
    log, base = [], dspark.Cfg()

    def run(c, rows, key):
        ds.cfg = c
        t0 = time.time()
        r = evaluate(ds, data, rows, stride)
        log.append({'cfg': asdict(c), 'rows': rows, **r})
        print(f'[{time.time() - t0:4.0f}s] {fmt(c, r, base)}', flush=True)
        json.dump(log, open(OUT.replace('.pt', '-grid.json'), 'w'))
        return r[key]

    best, bs = None, -1
    print('== stage A: product (head none, 1 row)', flush=True)
    for taps, fuse, tn, sm, anc, r0 in itertools.product(
            [(5, 23, 41, 48), (5, 23, 41, 49)], ['row', 'col'], [True, False], [False, True], ['raw', 'normed'],
            [True, False]):
        c = replace(base, head='none', taps=taps, fuse=fuse, tap_norm=tn, fuse_softmax=sm, anchor=anc, row0=r0)
        s = run(c, 1, 'a')[0]
        if s > bs:
            best, bs = c, s

    def descend(c, s, space, rows, key, pick=lambda r: r):
        changed = True
        while changed:
            changed = False
            for f, vals in space.items():
                for v in vals:
                    if getattr(c, f) == v:
                        continue
                    n = replace(c, **{f: v})
                    sn = pick(run(n, rows, key))
                    if sn > s + 1e-4:
                        c, s, changed = n, sn, True
        return c, s
    print(f'== stage A: descent from {fmt(best, {"a": [bs], "tv": [0], "tau": 0, "tau_s": 0, "n": 0})}', flush=True)
    best, bs = descend(best, bs, {
        'taps': [(6, 24, 42, 48), (4, 22, 40, 48)],
        'consumers': [(0, 1, 2, 3), (3, 2, 1, 0), (1, 2, 3, 0), (3, 0, 1, 2), (0, 1, 2, 0)],
        'rope': ['neox', 'gptj'], 'ctx_rope': [True, False], 'ctx_pos': [0, -1, 1], 'mask_scale': [1.0, 0.5, 2.0]},
        1, 'a', lambda r: r[0])
    print(f'== stage B: head (8 rows, tau) on {best}', flush=True)
    rows = 8 if best.row0 else 7
    base_tau = run(best, rows, 'tau')
    hb, hs = None, -1
    for head in ('a', 'b', 'c'):
        c = replace(best, head=head)
        s = run(c, rows, 'tau')
        if s > hs:
            hb, hs = c, s
    perms = [''.join(p) for p in itertools.permutations('swhe')]
    hb, hs = descend(hb, hs, {'zorder': perms, 'gco': [''.join(p) for p in itertools.permutations('gco')],
                              'bias': ['post', 'pre'], 'head': ['a', 'b', 'c']}, rows, 'tau')
    print(f'== backbone-only tau {base_tau:.2f}; with head {hs:.2f}', flush=True)
    hb2, _ = descend(hb, run(hb, rows, 'tau_s'), {'softcap': [True, False]}, rows, 'tau_s')
    print(f'== winner {hb2}', flush=True)
    return hb2


def final(c, stride):
    """Per-prompt table of the chosen config at a fine anchor stride."""
    data, ds = load()
    torch.set_num_threads(int(os.environ.get('INK_THREADS', 6)))
    ds.cfg = c
    for name in data:
        print(f'{name:7s} {len(data[name]["ids"])} tok: {fmt(c, evaluate(ds, {name: data[name]}, 8, stride), c)}',
              flush=True)
    print(f'all: {fmt(c, evaluate(ds, data, 8, stride), c)}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--record', action='store_true')
    ap.add_argument('--final', action='store_true', help='per-prompt table of dspark.Cfg() defaults')
    ap.add_argument('--stride', type=int, default=8)
    a = ap.parse_args()
    if a.record:
        record()
    elif a.final:
        import dspark
        final(dspark.Cfg(), a.stride)
    else:
        grid(a.stride)
