"""CPU: sequential round-cost simulation of the spec controllers on the recorded drafter taps (tests/dspark_ablate.py
--record: 6 teacher-forced prompts code1/code2/json1/chat1/chat2/prose1). Per anchor position p (stride 1): the real
drafter's 7 greedy drafts (full head) -> confidences and accepted count (drafts == target argmax while the text agrees
with the target), and the prompt-lookup copy after ids[:p+1] -> its accepted count. Then each controller walks the text
round by round (pos += emitted) with the FP8-window costs (verify T=1..8 6.41..11.16 ms, drafter 1.83 ms, verify16
beyond): Width alone (default), the 11-22 gate (its own on/off rule: realized tokens/ms EMA, probes 8/16/32), the fixed
gate (Width on/off + Gate.pick), hybrid (Width + copies), hybrid + fixed gate. Teacher-forced human text, so chat/prose
acceptance is pessimistic; json1 is a JSON transcript. Greedy only.
  python tests/spec_sim.py --trace    # ~5-15 min CPU (drafter over 2.5k anchors), -> ~/llm/logs/ink-spec-sim-trace.pt
  python tests/spec_sim.py            # simulation only (seconds)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
import torch  # noqa: E402

import spec  # noqa: E402

OUT = os.path.expanduser('~/llm/logs/ink-spec-sim-trace.pt')
VT = (6.41, 7.56, 8.29, 9.07, 9.99, 10.37, 10.69, 11.16)     # verify ms T=1..8, FP8 decode (window all3 11-22)
DMS = 1.83
EDGE = 1.0                                                   # spec_runner.COPY_EDGE


def trace(rows=7, chunk=16):
    import dspark_ablate as A
    data, ds = A.load()
    torch.set_num_threads(int(os.environ.get('INK_THREADS', 4)))
    out = {}
    for name, d in data.items():
        ids, g, T = d['ids'], d['g'], len(d['ids'])
        taps = torch.stack([A.tap(d, k) for k in ds.cfg.taps])
        st = ds.new_state()
        ds.extend(st, taps)
        e = ds.head_ctx(taps)
        P = torch.arange(A.START, T - rows)
        conf, acc = [], []
        for a in range(0, len(P), chunk):
            p = P[a:a + chunk]
            ar = p[:, None] + torch.arange(rows)
            lg = ds.rows_tf(st, ids[p], p, ids[ar], e[p - 1])
            m, agree = lg.argmax(-1) == g[ar], ids[ar + 1] == g[ar]
            conf += lg.softmax(-1).max(-1).values.tolist()
            for j in range(len(p)):
                k = 0
                while k < rows and m[j, k]:
                    k += 1
                    if not agree[j, k - 1]:
                        break
                acc.append(k)
        out[name] = dict(ids=ids.tolist(), g=g.tolist(), start=A.START, conf=conf, acc=acc)
        print(f'{name}: {len(acc)} anchors', flush=True)
    torch.save(out, OUT)


class OldGate(spec.Gate):
    """The 11-22 gate's own on/off rule (removed 11-23), for comparison."""
    def __init__(self, dms=DMS, rema=0.1, probe=8, pmax=32, margin=0.02, **kw):
        super().__init__(**kw)
        self.dms, self.rema, self.pmax, self.margin, self.rate, self.gap, self.idle = dms, rema, pmax, margin, None, \
            probe, 0

    def want(self):
        self.idle += 1
        if self.rate is None or self.rate > (1 + self.margin) / self.t[0]:
            self.gap = self.probe
            return True
        if self.idle >= self.gap:
            self.idle, self.gap = 0, min(2 * self.gap, self.pmax)
            return True
        return False

    def update(self, conf, m, accepted, ctl=None):
        super().update(conf, m, accepted)
        r = (accepted + 1) / (self.t[m] + self.dms)
        self.rate = r if self.rate is None else self.rate + self.rema * (r - self.rate)


def copy_acc(ids, g, p, j, m):
    """Copied tokens kept after anchor p: draft ids[j:j+m] vs the target argmax while the text agrees."""
    a = 0
    while a < m and j + a < p + 1 and ids[j + a] == g[p + a]:
        a += 1
        if ids[p + a] != g[p + a - 1]:
            break
    return a


def sim(d, mode):
    ids, g, s0, conf, acc = d['ids'], d['g'], d['start'], d['conf'], d['acc']
    tw = tuple(t + (DMS if k else 0) for k, t in enumerate(VT))
    ctl, gate, old = spec.Width(kmax=7, step_ms=tw), spec.Gate(verify=VT), OldGate(verify=VT)
    cp, vt = spec.Copy(), spec.verify16(VT)
    lk = spec.Lookup(ids[:s0 + 1]) if 'hybrid' in mode else None
    pos, tok, ms, dr, cr = s0, 0, 0.0, 0, 0
    end = s0 + len(acc)
    while pos < end:
        i = pos - s0
        c, a7 = conf[i], acc[i]
        if lk is not None:                                     # (lookup ids = text up to the anchor)
            lk.extend(ids[len(lk):pos + 1])
            j, n = lk.match()
            k = ctl.width()
            if j is not None:
                b = cp.bucket(n)
                m, r = cp.best(b, 15, vt)
                if m and r > max(EDGE * ctl.rates()[k] if k else 0, 1.02 / tw[0]):
                    a = copy_acc(ids, g, pos, j, m)
                    cp.update(b, m, a)
                    pos, tok, ms, cr = pos + a + 1, tok + a + 1, ms + vt[m], cr + 1
                    continue
        elif mode == 'gate_old':
            if not old.want():
                pos, tok, ms = pos + 1, tok + 1, ms + VT[0]
                continue
            m = old.pick(c, 7)
            a = min(a7, m)
            old.update(c, m, a)
            pos, tok, ms, dr = pos + a + 1, tok + a + 1, ms + VT[m] + DMS, dr + 1
            continue
        else:
            k = ctl.width()
        if not k:
            pos, tok, ms = pos + 1, tok + 1, ms + VT[0]
            continue
        dr += 1
        if 'gate' in mode:
            m = gate.pick(c, 7)
            a = min(a7, m)
            gate.update(c, m, a, ctl)
            pos, tok, ms = pos + a + 1, tok + a + 1, ms + VT[m] + DMS
        else:
            a = min(a7, k)
            ctl.update(k, a)
            pos, tok, ms = pos + a + 1, tok + a + 1, ms + tw[k]
    return tok / ms * VT[0], dr, cr


if __name__ == '__main__':
    if '--trace' in sys.argv:
        trace()
    data = torch.load(OUT)
    modes = ('width', 'gate_old', 'gate', 'hybrid', 'hybrid_gate')
    print('x of plain (sequential, greedy; drafting rounds / copy rounds):  ' + '  '.join(f'{m:>16s}' for m in modes))
    for name, d in data.items():
        r = [sim(d, m) for m in modes]
        print(f'{name:7s} ({len(d["acc"])} pos)  ' + '  '.join(f'x{x:.2f} ({dr:3d}/{cr:3d})' for x, dr, cr in r))
