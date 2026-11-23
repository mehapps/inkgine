"""Needle test at a chosen prompt size against a running server (reuses gate.py's filler + scoring).
Usage: python tests/long_ctx.py TOKENS [BASE]     e.g. python tests/long_ctx.py 1040000
Sizes the filler with our tokenizer so the prompt lands near TOKENS."""
import json
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.expanduser('~/llm/gate'))
from tokenizers import Tokenizer  # noqa: E402

import chat  # noqa: E402
import gate  # noqa: E402
from engine import MODEL_DIR  # noqa: E402


def build(target, tok):
    tag = 'ink' + str(int(time.time()))
    n = target
    for _ in range(4):  # gate's tokens/line is for Qwen's tokenizer; rescale with ours
        doc, codes = gate.filler(tag, n, needles=8)
        msgs = gate.ask_codes(doc, codes)
        got = len(chat.render(tok, msgs, effort=0)) + 2
        if abs(got - target) < target * 0.003:
            break
        n = int(n * target / got)
    return msgs, codes, got


def main():
    target = int(sys.argv[1])
    base = sys.argv[2] if len(sys.argv) > 2 else 'http://localhost:30000'
    tok = Tokenizer.from_file(os.path.join(MODEL_DIR, 'tokenizer.json'))
    msgs, codes, n = build(target, tok)
    print(f'prompt ~{n} tokens, {len(codes)} needles', flush=True)
    body = dict(model='x', messages=msgs, max_tokens=200, temperature=0,
                chat_template_kwargs={'enable_thinking': False})
    t = time.time()
    r = urllib.request.Request(base + '/v1/chat/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'})
    out = json.load(urllib.request.urlopen(r, timeout=6 * 3600))
    text = out['choices'][0]['message']['content']
    s = gate.score(text, codes)
    print(f'{s}/{len(codes)} codes, prompt_tokens {out["usage"]["prompt_tokens"]}, {time.time() - t:.0f} s')
    print(text[:600])
    sys.exit(0 if s == len(codes) else 1)


if __name__ == '__main__':
    main()
