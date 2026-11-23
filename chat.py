"""Prompt rendering and output parsing for Inkling 2 (SPEC.md §1, §10). Works on token ids."""
import hashlib
import json
import os
import threading
from array import array
from collections import OrderedDict

USER, MODEL, SYSTEM, TOOL = 200000, 200001, 200002, 200003
TEXT, IMAGE, THINK, END = 200004, 200005, 200008, 200010
EOS, TOOL_ERR, XML, INVOKE_JSON, INVOKE_TEXT, IMG_TOK = 200006, 200022, 200024, 200049, 200057, 200054
EFFORT = {'none': 0, 'minimal': 0.1, 'low': 0.2, 'medium': 0.7, 'high': 0.9, 'xhigh': 0.99, 'max': 0.99}
CONTENT_KIND = {THINK: 'reasoning', TEXT: 'content', TOOL_ERR: 'content', INVOKE_TEXT: 'content', INVOKE_JSON: 'tool'}


def fmt_effort(e):
    e = EFFORT.get(e, e) if isinstance(e, str) else e
    e = min(max(float(e), 0.0), 0.99)
    return '0' if e == 0 else repr(round(e, 6))


ENC_CACHE_MB = int(os.environ.get('INK_TOK_CACHE_MB', 64))  # render's encode memo (0 = off)


class EncCache:
    """tok.encode memo for render: blake2b(text) -> int32 ids, LRU bounded by bytes. Exact: render encodes every text
    piece on its own (special tokens are placed around it, never through the tokenizer), so a piece's ids depend on its
    text alone. An agent turn re-sends the whole history: only the new parts get tokenized."""
    MIN = 64  # shorter pieces: encoding is cheaper than the bookkeeping

    def __init__(self, tok, cap):
        self.tok, self.cap, self.size, self.d, self.lock = tok, cap, 0, OrderedDict(), threading.Lock()

    def __call__(self, s):
        if len(s) < self.MIN:
            return self.tok.encode(s, add_special_tokens=False).ids
        k = hashlib.blake2b(s.encode('utf-8', 'surrogatepass'), digest_size=16).digest()
        with self.lock:
            v = self.d.get(k)
            if v is not None:
                self.d.move_to_end(k)
                return v
        v = array('i', self.tok.encode(s, add_special_tokens=False).ids)
        n = len(v) * 4 + 100  # (+ key and dict entry)
        if n <= self.cap:
            with self.lock:
                if k not in self.d:
                    self.d[k], self.size = v, self.size + n
                    while self.size > self.cap:
                        self.size -= len(self.d.popitem(last=False)[1]) * 4 + 100
        return v


_ENC = {}  # id(tok) -> (tok, EncCache) (the tok reference keeps the id from being reused)


def encoder(tok):
    if not ENC_CACHE_MB:
        return lambda s: tok.encode(s, add_special_tokens=False).ids
    e = _ENC.get(id(tok))
    if e is None:
        e = _ENC[id(tok)] = (tok, EncCache(tok, ENC_CACHE_MB << 20))
    return e[1]


def render(tok, messages, tools=None, effort=0.9, image_tokens=None):
    """-> list of ids. image_tokens(part) -> number of tiles for an image part (placeholder expansion)."""
    enc = encoder(tok)
    ids = []

    def put(*xs):  # (enc returns an array: extend, not list +)
        for x in xs:
            ids.extend(x)
    if tools:
        decl = [{'description': t.get('function', t).get('description', ''), 'name': t.get('function', t)['name'],
                 'parameters': t.get('function', t).get('parameters', {}), 'type': t.get('type', 'function')}
                for t in tools]
        put([SYSTEM], enc('tool_declare'), [XML], enc(json.dumps(decl, sort_keys=True, separators=(',', ':'),
                                                                  ensure_ascii=False)), [END])
    effort_ids = [SYSTEM, TEXT] + list(enc(f'Thinking effort level: {fmt_effort(effort)}')) + [END]
    call_names = {}
    placed = False
    for m in messages:
        role = m['role']
        if role != 'system' and not placed:
            ids += effort_ids; placed = True
        parts = m.get('content')
        if parts is None:
            parts = []
        elif isinstance(parts, str):
            parts = [{'type': 'text', 'text': parts}]
        if role in ('system', 'developer', 'user'):
            r = SYSTEM if role != 'user' else USER
            for p in parts:
                if p.get('type') == 'text':
                    put([r, TEXT], enc(p['text']), [END])
                elif p.get('type') in ('image_url', 'image') and role == 'user':
                    ids += [r, IMAGE] + [IMG_TOK] * image_tokens(p) + [END]
        elif role == 'tool':
            name = m.get('name') or call_names.get(m.get('tool_call_id'), '')
            for p in parts:
                if p.get('type') == 'text':
                    put([TOOL], enc(name), [TEXT], enc(p['text']), [END])
                elif p.get('type') in ('image_url', 'image'):
                    put([TOOL], enc(name), [IMAGE], [IMG_TOK] * image_tokens(p), [END])
        elif role == 'assistant':
            text = ''.join(p.get('text', '') for p in parts if p.get('type') == 'text')
            rc = m.get('reasoning_content') or m.get('reasoning')
            calls = m.get('tool_calls') or []
            if rc:
                put([MODEL, THINK], enc(rc), [END])
            if text or not (rc or calls):
                put([MODEL, TEXT], enc(text), [END])
            for c in calls:
                fn = c.get('function', c)
                args = fn.get('arguments', {})
                args = json.loads(args) if isinstance(args, str) and args else (args or {})
                call_names[c.get('id')] = fn['name']
                body = call_body(fn['name'], args)
                put([MODEL], enc(fn['name']), [INVOKE_JSON], enc(body), [END])
            ids += [EOS]
    if not placed:
        ids += effort_ids
    return ids


def call_body(name, args):
    """A call's JSON as the model's chat template writes it: name first, then args with only the top-level keys sorted
    (nested values keep their order), compact, non-ASCII kept."""
    d = lambda v: json.dumps(v, separators=(',', ':'), ensure_ascii=False)
    return '{"name":' + d(name) + ',"args":{' + ','.join(d(k) + ':' + d(v) for k, v in sorted(args.items())) + '}}'


class Parser:
    """Incremental output parser. feed(id) -> list of (kind, text) deltas; kind in reasoning|content.
    Tool calls are collected in .tool_calls: the name is the header's (MODEL name INVOKE_JSON), else the body's "name";
    the args are the body's "args" (as rendered), else "arguments", else the whole body; no name or no object args ->
    the body goes out as content. Text is decoded incrementally: only the body tokens after the last emitted
    delta (byte-level BPE: a decode that doesn't end in U+FFFD ends on a whole character, so decode(a + b) ==
    decode(a) + decode(b) there), not the whole message each token."""

    def __init__(self, tok):
        self.tok, self.kind, self.body, self.sent, self.tool_calls = tok, None, [], 0, []  # sent: body tokens emitted
        self.name = []                                    # header name tokens of the current message

    def feed(self, t):
        if t == MODEL:
            self.kind, self.name = 'name', []; return []
        if t in CONTENT_KIND and self.kind in ('name', None):
            self.kind, self.body, self.sent = CONTENT_KIND[t], [], 0; return []
        if t == END:
            out = self._flush(final=True)
            if self.kind == 'tool':
                text = self.tok.decode(self.body, skip_special_tokens=False)
                call = self._call(text)
                if call is not None:
                    self.tool_calls.append(call)
                else:
                    out.append(('content', text))
            self.kind = None; return out
        if self.kind in ('reasoning', 'content', 'tool'):
            self.body.append(t)
            return self._flush()
        if self.kind == 'name' and t < USER:
            self.name.append(t)
        return []

    def _call(self, text):
        """Tool-call body -> {name, arguments} or None (not JSON, no name, args not an object)."""
        try:
            d = json.loads(text)
        except ValueError:
            return None
        if not isinstance(d, dict):
            return None
        name = self.tok.decode(self.name, skip_special_tokens=False).strip()
        name = name or (d['name'] if isinstance(d.get('name'), str) else '')
        a = d['args'] if 'args' in d else d['arguments'] if 'arguments' in d else {} if set(d) == {'name'} else d
        if isinstance(a, str):  # (OpenAI-style arguments string)
            try:
                a = json.loads(a)
            except ValueError:
                return None
        if not name or not isinstance(a, dict):
            return None
        return {'name': name, 'arguments': json.dumps(a)}

    def _flush(self, final=False):
        if self.kind not in ('reasoning', 'content'):
            return []
        d = self.tok.decode(self.body[self.sent:], skip_special_tokens=False)
        if not final and d.endswith('�'):  # wait for the rest of a multi-byte char
            return []
        self.sent = len(self.body)
        return [(self.kind, d)] if d else []


class Stop:
    """OpenAI `stop` on content text. feed(delta) -> (text safe to emit, hit): on a hit the text ends just before the
    earliest stop string; otherwise a tail that could still start a stop string is held back (flush() at the end)."""

    def __init__(self, stops):
        self.stops, self.held = stops, ''

    def feed(self, d):
        s = self.held + d
        hits = [i for i in (s.find(x) for x in self.stops) if i >= 0]
        if hits:
            self.held = ''
            return s[:min(hits)], True
        k = max((j for x in self.stops for j in range(1, min(len(x), len(s) + 1)) if s.endswith(x[:j])), default=0)
        self.held = s[len(s) - k:]
        return s[:len(s) - k], False

    def flush(self):
        out, self.held = self.held, ''
        return out
