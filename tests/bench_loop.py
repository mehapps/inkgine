"""Host cost of the decode loop, no model: the real Engine.decode_step / sampling / parser / stop / grammar (and with
--server the real Scheduler, asyncio emit and SSE through uvicorn, clients = curl processes) driven by a fake batch
decoder whose step returns ready logits (a spike on the next token of a multilingual script: text with CJK, emoji,
accents, so the parser sees multi-byte tokens). B requests decode together; a finished one is replaced at once.
CPU by default (sampling is compute there); INK_DEVICE=cuda puts the logits on the GPU (launches + the step's sync).
    python tests/bench_loop.py [--steps 400] [--B 1,2,4,8] [--tokens 400] [--temp 0] [--top-p 1] [--top-k 0]
                               [--stop] [--json] [--profile] [--server]
Prints ms per step (wall) and with --profile the cProfile time per component per step."""
import argparse
import cProfile
import os
import pstats
import socket
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
from tokenizers import Tokenizer

import chat
import grammar
from engine import MODEL_DIR, Engine, Request
from model import VOCAB

TEXT = ('The quick brown fox jumps over the lazy dog. Café naïve résumé — “quotes” … 日本語のテキストと中文混合。'
        '한국어 문장도 있습니다. Emoji: 🚀🔥👍🏽 👨‍👩‍👧 and math ∑∫√π≈3.14159. Code: `def f(x): return x**2`\n')


class FakeBD:
    """BatchDecoder stand-in: step() -> [R, VOCAB] fp32 static buffer; row i peaks on its slot's next script token."""

    def __init__(self, dev, script, slots=16, step_ms=0.0):
        self.step_s = step_ms / 1e3  # sleep per step (releases the GIL, like the host waiting on a CUDA graph)
        self.n, self.dec, self.decs, self.script = slots, False, [None] * slots, script
        g = torch.Generator().manual_seed(0)
        self.base = torch.randn(slots, VOCAB, generator=g).to(dev)
        self.out = self.base.clone()
        self.k = [0] * slots
        self.rows = torch.arange(slots, device=dev)

    def busy(self):
        return True

    def load(self, slot, seq):
        self.k[slot] = 0

    def free(self, slot):
        pass

    def drop(self, keep=0):
        return 0

    def row(self, slot):  # next script token for this slot
        t = self.script[self.k[slot] % len(self.script)]
        self.k[slot] += 1
        return t

    def step(self, rows):
        R = len(rows)
        if self.step_s:
            time.sleep(self.step_s)
        self.out[:R].copy_(self.base[:R])  # (the engine masks this buffer in place: json)
        self.out[self.rows[:R], torch.tensor([self.row(s) for s, _, _ in rows]).to(self.out.device)] = 30.0
        return self.out[:R]


class BenchEngine(Engine):
    """Engine with the decode loop only: prefill/start/finish are stubs, decode_step/_take/_feed are the real ones."""

    def __init__(self, dev, script, step_ms=0.0):
        self.tok = Tokenizer.from_file(os.path.join(MODEL_DIR, 'tokenizer.json'))
        self.model = type('M', (), {'dev': torch.device(dev)})()
        self.bd, self.chunk, self.parked, self.ooms, self.pool = FakeBD(dev, script, step_ms=step_ms), 4096, [], 0, None
        self.max_ctx, self.vision = 1 << 20, None
        grammar.compiler(self.tok)

    def can_start(self, r):
        return True

    def start(self, r):
        r.seq = object()

    def prefill_chunk(self, r, n=None):
        r.stats.update(prompt_tokens=len(r.ids), cached_tokens=0, prefill_s=0.0)
        return True

    def to_decode(self, r, slot):
        r.slot, r.pos = slot, len(r.ids)
        self.bd.load(slot, None)
        r.t1 = time.time()
        x = self.bd.base[slot].clone()
        x[self.bd.row(slot)] = 30.0
        self._take([r], x[None])

    def finish(self, r):
        r.slot = r.seq = None


def script_ids(tok):
    body = tok.encode(TEXT * 40, add_special_tokens=False).ids
    return [chat.MODEL, chat.THINK] + body[:150] + [chat.END, chat.MODEL, chat.TEXT] + body[150:]


def mk(eng, a, emit=None):
    rf = grammar.compile_format(eng.tok, {'type': 'json_object'}) if a.json else None
    return Request(eng, [1] * 16, [], max_tokens=a.tokens, temperature=a.temp, top_p=a.top_p, top_k=a.top_k, seed=1,
                   response_format=rf, stop=['NEVER-MATCHES', '\n\n'] if a.stop else None, emit=emit,
                   prefill_text=a.json)  # json: thinking off, every step masked


def direct(eng, a, B):
    """B requests in slots 0..B-1 through Engine.decode_step, refilled when done -> (ms/step, profile or None)."""
    reqs = []
    for s in range(B):
        r = mk(eng, a)
        eng.start(r); eng.to_decode(r, s)
        reqs.append(r)

    def run(n):
        for _ in range(n):
            eng.decode_step(reqs)
            for i, r in enumerate(reqs):
                if r.done:
                    eng.finish(r)
                    reqs[i] = mk(eng, a)
                    eng.start(reqs[i]); eng.to_decode(reqs[i], i)
        if eng.model.dev.type == 'cuda':
            torch.cuda.synchronize()
    run(20)  # warm
    t0 = time.perf_counter()
    run(a.steps)
    ms = (time.perf_counter() - t0) / a.steps * 1e3
    prof = None
    if a.profile:
        prof = cProfile.Profile()
        prof.enable(); run(a.steps); prof.disable()
    return ms, prof


COMP = {  # component -> function names whose cumulative time counts (first match per frame)
    'fake decoder step': [('bench_loop.py', 'step')],
    'grammar mask+accept': [('grammar.py', 'mask'), ('grammar.py', 'accept')],
    'sampling + sync': [('engine.py', 'sample_rows'), ('decode.py', 'sample'), ("~", "<method 'tolist'")],
    'parser feed': [('chat.py', 'feed')],
    'stop match': [('chat.py', 'feed@Stop')],
    'emit': [('engine.py', '<lambda>')],
    'decode_step total': [('engine.py', 'decode_step')],
}


def report(prof, steps):
    st = pstats.Stats(prof)
    rows = {}
    for (fn, line, name), (cc, nc, tt, ct, callers) in st.stats.items():
        base = os.path.basename(fn)
        rows[(base, name, line)] = ct
    out = {}
    for comp, keys in COMP.items():
        t = 0.0
        for base, name in keys:
            if name == 'feed@Stop':  # Stop.feed: the second 'feed' in chat.py (highest line number)
                feeds = sorted((ln, ct) for (b, n, ln), ct in rows.items() if b == 'chat.py' and n == 'feed')
                t += feeds[-1][1] if len(feeds) > 1 else 0
                continue
            if base == 'chat.py' and name == 'feed':
                feeds = sorted((ln, ct) for (b, n, ln), ct in rows.items() if b == 'chat.py' and n == 'feed')
                t += feeds[0][1] if feeds else 0
                continue
            t += sum(ct for (b, n, ln), ct in rows.items() if b == base and n.startswith(name))
        out[comp] = t / steps * 1e3
    return out


def server_mode(a, B):
    """Real server (uvicorn thread, Scheduler, SSE) on the fake engine; B curl streams of a.tokens tokens, each repeated
    until a.steps steps have run -> ms per step (wall / steps)."""
    import uvicorn
    import server
    eng = BenchEngine(os.environ.get('INK_DEVICE', 'cpu'), script_ids(Tokenizer.from_file(os.path.join(MODEL_DIR, 'tokenizer.json'))),
                      a.step_ms)
    if server.eng is None:
        server.eng, server.sched = eng, server.Scheduler(eng)
        with socket.socket() as s:
            s.bind(('127.0.0.1', 0)); server.PORT = s.getsockname()[1]
        threading.Thread(target=uvicorn.run, args=(server.app,), kwargs=dict(port=server.PORT, log_level='warning'),
                         daemon=True).start()
        time.sleep(1.5)
    body = ('{"messages":[{"role":"user","content":"hi"}],"stream":true,"max_tokens":%d,"temperature":%g,"top_p":%g}'
            % (a.tokens, a.temp, a.top_p))
    reps = max(1, a.steps // a.tokens)
    cmd = ['sh', '-c', f'for i in $(seq {reps}); do curl -sN -o /dev/null -H "Content-Type: application/json" '
                       f"-d '{body}' http://127.0.0.1:{server.PORT}/v1/chat/completions; done"]
    s0 = server.M['decode_steps']
    t0 = time.perf_counter()
    ps = [subprocess.Popen(cmd) for _ in range(B)]
    for p in ps:
        p.wait()
    dt = time.perf_counter() - t0
    steps = server.M['decode_steps'] - s0
    return dt / max(steps, 1) * 1e3, B * reps * a.tokens / dt


def check(dev):
    """engine.sample_rows == decode.sample row by row (bit-identical tokens, same generator states after) and
    grammar.mask_rows == the old per-row host mask, on random logits with ties and -inf rows. Exit 1 on a mismatch."""
    from decode import sample
    from engine import sample_rows
    g = torch.Generator().manual_seed(0)
    settings = [(0, 1, 0), (1.0, 1.0, 0), (0.7, 0.9, 0), (1.3, 0.5, 0), (0.7, 1.0, 40), (1.0, 0.95, 5), (-1, 0.9, 3),
                (0.8, 1.0, VOCAB + 5)]
    bad = 0
    for trial in range(60):
        R = 1 + trial % 8
        lg = torch.randn(R, VOCAB, generator=g) * (1 + trial % 4)
        if trial % 3 == 0:  # ties: bf16-rounded values
            lg = lg.bfloat16().float()
        if trial % 5 == 0:  # grammar-style -inf masks
            lg[:, torch.rand(VOCAB, generator=g) < 0.99] = float('-inf')
        lg = lg.to(dev)
        rows = [settings[(trial + i * 3) % len(settings)] for i in range(R)]
        if trial % 4 == 1:  # all rows one setting (one batched group)
            rows = [rows[0]] * R
        seeds = [trial * 100 + i for i in range(R)]
        mk = lambda: [torch.Generator(device=dev).manual_seed(x) for x in seeds]
        ga, gb = mk(), mk()
        got = sample_rows(lg.clone(), [r + (x,) for r, x in zip(rows, ga)]).tolist()
        want = [int(sample(lg[i].clone(), *rows[i], gb[i])) for i in range(R)]
        same_gen = all(torch.equal(a.get_state(), b.get_state()) for a, b in zip(ga, gb))
        if got != want or not same_gen:
            bad += 1
            print(f'sample_rows MISMATCH trial {trial}: {rows} {got} {want} gen states equal {same_gen}', flush=True)
    rf = grammar.compile_format(Tokenizer.from_file(os.path.join(MODEL_DIR, 'tokenizer.json')), {'type': 'json_object'})
    shifts = torch.arange(32, dtype=torch.int32)

    def old_mask(c, x):  # the per-row host mask grammar.Json.mask had before 2026-11-22 (json state)
        c.m.fill_next_token_bitmask(c.bits)
        ok = ((c.bits[0][:, None] >> shifts) & 1).bool().view(-1)[:VOCAB]
        x.masked_fill_(~ok.to(x.device), float('-inf'))
    for trial in range(10):
        R = 1 + trial % 6
        cons = [grammar.Json(rf, True) if (i + trial) % 3 else None for i in range(R)]
        refs = [grammar.Json(rf, True) if c is not None else None for c in cons]
        for c, r in zip(cons, refs):  # move the matchers into different grammar states
            if c is not None:
                for t in tok_ids[:trial % 3]:
                    c.accept(t); r.accept(t)
        lg = torch.randn(R, VOCAB, generator=g).to(dev)
        a, b = lg.clone(), lg.clone()
        grammar.mask_rows(a, cons)
        for i, r in enumerate(refs):
            if r is not None:
                old_mask(r, b[i])
        if not torch.equal(a, b):
            bad += 1
            print(f'mask_rows MISMATCH trial {trial}', flush=True)
    print(f'check on {dev}: ' + ('OK' if not bad else f'{bad} FAILED'), flush=True)
    return bad


tok_ids = [90, 1]  # filled in main(): '{' then '"' token ids (valid json_object prefixes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--tokens', type=int, default=400)
    ap.add_argument('--B', default='1,2,4,8')
    ap.add_argument('--temp', type=float, default=0.0)
    ap.add_argument('--top-p', type=float, default=1.0)
    ap.add_argument('--top-k', type=int, default=0)
    ap.add_argument('--stop', action='store_true')
    ap.add_argument('--json', action='store_true')
    ap.add_argument('--profile', action='store_true')
    ap.add_argument('--server', action='store_true')
    ap.add_argument('--step-ms', type=float, default=0.0, help='fake step time (sleep); host overhead = ms/step - this')
    ap.add_argument('--check', action='store_true', help='sample_rows / mask_rows equivalence on cpu (+ cuda with INK_DEVICE=cuda: GPU window only)')
    a = ap.parse_args()
    if a.check:
        t = Tokenizer.from_file(os.path.join(MODEL_DIR, 'tokenizer.json'))
        tok_ids[:] = [t.encode('{', add_special_tokens=False).ids[0], t.encode('"', add_special_tokens=False).ids[0]]
        sys.exit(sum(check(d) for d in ['cpu'] + (['cuda'] if os.environ.get('INK_DEVICE') == 'cuda' else [])) > 0)
    dev = os.environ.get('INK_DEVICE', 'cpu')
    torch.set_num_threads(int(os.environ.get('INK_THREADS', 4)))
    print(f'device {dev}, {a.steps} steps, replies {a.tokens} tokens, temp {a.temp} top_p {a.top_p} top_k {a.top_k}'
          f'{" stop" if a.stop else ""}{" json" if a.json else ""}{" server" if a.server else ""}', flush=True)
    if a.server:
        for B in map(int, a.B.split(',')):
            ms, tps = server_mode(a, B)
            print(f'B={B}: {ms:.3f} ms/step wall (server, fake step {a.step_ms:g} ms -> host {ms - a.step_ms:.3f} ms), '
                  f'{tps:.0f} tok/s aggregate', flush=True)
        return
    eng = BenchEngine(dev, script_ids(Tokenizer.from_file(os.path.join(MODEL_DIR, 'tokenizer.json'))), a.step_ms)
    for B in map(int, a.B.split(',')):
        ms, prof = direct(eng, a, B)
        line = f'B={B}: {ms:.3f} ms/step wall'
        if prof is not None:
            line += ' | ' + ', '.join(f'{k} {v:.3f}' for k, v in report(prof, a.steps).items())
        print(line, flush=True)


if __name__ == '__main__':
    main()
