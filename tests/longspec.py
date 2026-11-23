"""Long-context speculative decoding, against a running server (window_longspec.sh): one agent-style session per size.
The context = real source files (engine/spec.py first, so the copied spans sit ~SIZE tokens back) + gate.py filler up
to ~SIZE tokens, with the agent tools. Prefill once (cold), then 2 turns that copy from the context:
  tool: an edit_file call whose old_string is a whole function of spec.py, verbatim;
  edit: the whole class Width of spec.py rewritten with one rename (a code block copying a file span).
Each turn greedy, thinking off, spec on (server default) then `spec: false`, both restoring the session's parked prefix.
Prints decode tok/s ((completion - 1) / (wall - ttft)), the ratio, and whether the outputs are identical.
Exit 1 on an HTTP error, a non-identical greedy pair, or a missing tool call.
Usage: python tests/longspec.py SIZE [SIZE ...]      e.g. python tests/longspec.py 130000 512000 1030000"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.expanduser('~/llm/gate'))
from tokenizers import Tokenizer  # noqa: E402

import chat  # noqa: E402
import gate  # noqa: E402
from engine import MODEL_DIR  # noqa: E402

BASE = os.environ.get('INK_URL', 'http://localhost:30000')
TOOLS = [{'type': 'function', 'function': {
    'name': 'edit_file', 'description': 'Replace the exact text old_string in a file with new_string.',
    'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}, 'old_string': {'type': 'string'},
                                                    'new_string': {'type': 'string'}},
                   'required': ['path', 'old_string', 'new_string']}}},
    {'type': 'function', 'function': {
        'name': 'read_file', 'description': 'Read a file.',
        'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}}, 'required': ['path']}}}]
SYS = 'You are a coding agent working in a Python repository. Use the tools to make the requested change.'
FILES = ('spec.py', 'kvpool.py', 'chat.py')
TURNS = (
    ('tool', 'In engine/spec.py, add a type hint `t: tuple` to the parameter of the function verify16. Make exactly one '
             'edit_file call; old_string must be the whole current verify16 function copied verbatim from the file '
             'above, new_string the same function with the hint.', 3000),
    ('edit', 'Print the complete class Width from engine/spec.py exactly as in the file above, with one change only: '
             'rename the attribute `margin` to `edge` everywhere in the class. Output only one python code block, no '
             'tool call.', 4000),
)


def post(body, timeout=3 * 3600):
    r = urllib.request.Request(BASE + '/v1/chat/completions', json.dumps(body).encode(),
                               {'Content-Type': 'application/json'})
    t0 = time.time()
    try:
        out = json.load(urllib.request.urlopen(r, timeout=timeout))
    except urllib.error.HTTPError as e:
        raise SystemExit(f'HTTP {e.code}: {e.read()[:400]!r}')
    return out, time.time() - t0


def context(size, tok):
    """-> messages [system, user] of ~size tokens (with the tools rendered)."""
    src = ''.join(f'\n===== engine/{f} =====\n' + open(os.path.join(os.path.dirname(HERE), f)).read() for f in FILES)
    tag, n = 'ls' + str(int(time.time())), size
    for _ in range(4):  # gate's tokens/line is for another tokenizer: rescale with ours
        doc, _ = gate.filler(tag, max(n, 1000), needles=0)
        user = 'Repository files and logs follow.\n' + src + '\n===== logs/station.txt =====\n' + doc
        msgs = [{'role': 'system', 'content': SYS}, {'role': 'user', 'content': user + '\n\nReply with OK.'}]
        got = len(chat.render(tok, msgs, tools=TOOLS, effort=0)) + 2
        if abs(got - size) < size * 0.005:
            break
        n = int(n * size / got)
    return msgs, got


def answer(out):
    m = out['choices'][0]['message']
    return (m.get('content') or '') + json.dumps([c['function'] for c in m.get('tool_calls') or []])


def run(size, tok):
    msgs, n = context(size, tok)
    print(f'== session ~{n} tokens', flush=True)
    base = dict(model='x', tools=TOOLS, temperature=0, chat_template_kwargs={'enable_thinking': False}, timings=True)
    out, dt = post(dict(base, messages=msgs, max_tokens=4))
    print(f'prefill (cold): {out["usage"]["prompt_tokens"]} tokens, {dt:.0f} s, reply {answer(out)[:40]!r}', flush=True)
    ok = True
    hist = msgs + [{'role': 'assistant', 'content': out['choices'][0]['message'].get('content') or 'OK'}]
    for name, q, mt in TURNS:
        res = {}
        for spec in (True, False):
            body = dict(base, messages=hist + [{'role': 'user', 'content': q}], max_tokens=mt)
            if not spec:
                body['spec'] = False
            out, dt = post(body)
            tm = (out.get('timings') or [{}])[0]
            c = out['usage']['completion_tokens']
            dec = dt - tm.get('ttft_s', 0.0)
            res[spec] = (answer(out), c, (c - 1) / max(dec, 1e-6), out['usage']['prompt_tokens_details']['cached_tokens'],
                         tm.get('ttft_s', 0.0), out['choices'][0]['finish_reason'])
        (a, ca, ta, pa, fa, ra), (b, cb, tb, pb, fb, rb) = res[True], res[False]
        same = a == b
        ok &= same
        if name == 'tool':
            ok &= ra == 'tool_calls' and '"edit_file"' in a
        print(f'{name} @{n}: spec {ta:.1f} tok/s ({ca} tok, ttft {fa:.1f} s, cached {pa}, {ra}), plain {tb:.1f} tok/s '
              f'({cb} tok, ttft {fb:.1f} s, cached {pb}, {rb}): x{ta / tb:.2f}; greedy outputs '
              + ('IDENTICAL' if same else f'DIFFER at char {next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))}'),
              flush=True)
        if not same or (name == 'tool' and ra != 'tool_calls'):
            print(f'  spec : {a[:300]!r}\n  plain: {b[:300]!r}', flush=True)
    return ok


def main():
    tok = Tokenizer.from_file(os.path.join(MODEL_DIR, 'tokenizer.json'))
    ok = True
    for s in sys.argv[1:]:
        ok &= run(int(s), tok)
    print('LONGSPEC ' + ('PASS' if ok else 'FAIL'), flush=True)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
