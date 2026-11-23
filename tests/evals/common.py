"""Shared client for the evals: plain OpenAI-compatible /v1/chat/completions, stdlib only.
Works against our engine, Qwen/SGLang, the official SGLang Inkling image, or LiteLLM (key from an env var)."""
import argparse
import json
import os
import time
import urllib.error
import urllib.request


class Client:
    def __init__(self, base='http://localhost:30000', model=None, key_env='EVAL_API_KEY', timeout=3600):
        self.base, self.timeout = base.rstrip('/').removesuffix('/v1'), timeout
        self.key = os.environ.get(key_env) if key_env else None  # never hardcoded, never printed
        self.model = model or self._first_model()

    def _req(self, path, body=None, timeout=None):
        h = {'Content-Type': 'application/json'}
        if self.key:
            h['Authorization'] = 'Bearer ' + self.key
        r = urllib.request.Request(self.base + path, json.dumps(body).encode() if body is not None else None, h)
        with urllib.request.urlopen(r, timeout=timeout or self.timeout) as f:
            return json.loads(f.read())

    def _first_model(self):
        try:
            return self._req('/v1/models', timeout=15)['data'][0]['id']
        except Exception:
            return 'x'  # our engine ignores the name

    def chat(self, messages, max_tokens=512, effort=None, **extra):
        """effort: None = server default, 'off' = thinking disabled, else sent as reasoning_effort.
        Returns dict(content, reasoning, tool_calls, finish, usage, secs, error)."""
        body = dict(model=self.model, messages=messages, max_tokens=max_tokens, temperature=0, **extra)
        if effort == 'off':
            body['chat_template_kwargs'] = {'enable_thinking': False}  # understood by our engine and SGLang
        elif effort:
            body['reasoning_effort'] = effort
        t = time.time()
        try:
            r = self._req('/v1/chat/completions', body)
        except urllib.error.HTTPError as e:
            return dict(content='', reasoning='', tool_calls=[], finish='error', usage={}, secs=time.time() - t,
                        error=f'HTTP {e.code}: {e.read()[:300].decode(errors="replace")}')
        except Exception as e:
            return dict(content='', reasoning='', tool_calls=[], finish='error', usage={}, secs=time.time() - t,
                        error=repr(e)[:300])
        c = r['choices'][0]
        m = c.get('message') or {}
        return dict(content=m.get('content') or '', reasoning=m.get('reasoning_content') or m.get('reasoning') or '',
                    tool_calls=m.get('tool_calls') or [], finish=c.get('finish_reason'), usage=r.get('usage') or {},
                    secs=time.time() - t, error=None)


def parser(doc):
    p = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--base', default=os.environ.get('EVAL_BASE', 'http://localhost:30000'))
    p.add_argument('--model', help='model name (default: first id from /v1/models)')
    p.add_argument('--key-env', default='EVAL_API_KEY', help='env var holding a Bearer key (LiteLLM); unset = none')
    p.add_argument('--quick', action='store_true', help='small sizes')
    p.add_argument('--out', help='write this suite\'s result JSON here')
    return p


def client_from(args):
    return Client(args.base, args.model, args.key_env)


def main_for(run, doc):
    """Standalone entry point for one suite: python tests/evals/<suite>.py [--base ...]."""
    p = parser(doc)
    a = p.parse_args()
    res = run(client_from(a), quick=a.quick)
    print(json.dumps(res['summary'], indent=1))
    if a.out:
        with open(a.out, 'w') as f:
            json.dump(res, f, indent=1)
