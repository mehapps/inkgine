"""CPU, tokenizer only: tool-call round trip (render tools + a tool result, parse a fabricated model reply with a call)
and the server's thinking-off + tools rule (no text prefill), the stop matcher and the n / logprobs / stop /
tool_choice parameter checks. Run: python tests/test_chat.py"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from fastapi.testclient import TestClient
from tokenizers import Tokenizer

import chat
import server
from engine import MODEL_DIR

tok = Tokenizer.from_file(os.path.join(MODEL_DIR, 'tokenizer.json'))
enc = lambda s: tok.encode(s, add_special_tokens=False).ids
tools = [{'type': 'function', 'function': {'name': 'get_weather', 'description': 'Weather for a city',
                                           'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}}}}}]
msgs = [{'role': 'user', 'content': 'Weather in Oslo and Paris?'},
        {'role': 'assistant', 'content': None, 'tool_calls': [
            {'id': 'call_1', 'type': 'function', 'function': {'name': 'get_weather', 'arguments': '{"city": "Oslo"}'}}]},
        {'role': 'tool', 'tool_call_id': 'call_1', 'content': '-3 C, snow'}]
ids = chat.render(tok, msgs, tools, effort=0.7)
text = tok.decode(ids, skip_special_tokens=False)
assert ids[0] == chat.SYSTEM and '"name":"get_weather"' in text and 'Thinking effort level: 0.7' in text
call = [chat.MODEL] + enc('get_weather') + [chat.INVOKE_JSON] + enc('{"name":"get_weather","args":{"city":"Oslo"}}') + [chat.END, chat.EOS]
res = [chat.TOOL] + enc('get_weather') + [chat.TEXT] + enc('-3 C, snow') + [chat.END]
assert ids[-len(res) - len(call):] == call + res, tok.decode(ids[-40:], skip_special_tokens=False)
print('render ok:', repr(text[-160:]))

# fabricated reply: reasoning, a short text, then a second tool call, then EOS
out = ([chat.MODEL, chat.THINK] + enc('Need Paris too.') + [chat.END]
       + [chat.MODEL, chat.TEXT] + enc('Checking Paris.') + [chat.END]
       + [chat.MODEL] + enc('get_weather') + [chat.INVOKE_JSON] + enc('{"name":"get_weather","args":{"city":"Paris"}}')
       + [chat.END])
p, deltas = chat.Parser(tok), []
for t in out:
    deltas += p.feed(t)
reasoning = ''.join(d[1] for d in deltas if d[0] == 'reasoning')
content = ''.join(d[1] for d in deltas if d[0] == 'content')
assert reasoning == 'Need Paris too.' and content == 'Checking Paris.', deltas
# the call is written as rendered: its args stream (call_start, call_args...) and its arguments are exactly that text
assert p.tool_calls == [{'name': 'get_weather', 'arguments': '{"city":"Paris"}'}], p.tool_calls
calls = [d for d in deltas if d[0].startswith('call_')]
assert calls[0] == ('call_start', 'get_weather') and calls[-1] == ('call_end', p.tool_calls[0])
assert ''.join(d[1] for d in calls if d[0] == 'call_args') == '{"city":"Paris"}', calls
print('parse ok:', p.tool_calls)

# server: thinking off + tools -> no MODEL TEXT prefill (the model may answer with a tool call)
seen = []


from test_batch_sched import FakeEngine


class Fake(FakeEngine):
    def request(self, messages, **kw):  # records the engine kwargs the server derived; every request is 'ok:1:1'
        seen.append({k: v for k, v in kw.items() if k not in ('stats', 'emit', 'prio', 'cancel')})
        return super().request([{'content': 'ok:1:1'}], **{k: kw[k] for k in ('stats', 'emit', 'prio', 'cancel')})


server.eng = Fake(chunk_s=0, step_s=0)
server.eng.tok = tok  # (tool_choice auto compiles the tools' call grammars)
server.sched = server.Scheduler(server.eng)
c = TestClient(server.app)
base = dict(messages=[{'role': 'user', 'content': 'hi'}], chat_template_kwargs={'enable_thinking': False})
assert c.post('/v1/chat/completions', json=dict(base, tools=tools)).status_code == 200
assert c.post('/v1/chat/completions', json=base).status_code == 200
assert seen[0]['prefill_text'] is False and seen[1]['prefill_text'] is True and seen[0]['effort'] == 0, seen
print('thinking off: prefill without tools, none with tools')

# stop sequences: every split of the text into deltas gives the same cut, nothing past a stop string is emitted early
import random
for text, stops, want, hit in [('Hello END world', ['END'], 'Hello ', True), ('abcabd', ['abd', 'xyz'], 'abc', True),
                               ('no stop here', ['zz'], 'no stop here', False), ('a<|x', ['<|x|>'], 'a<|x', False),
                               ('xx STOP', ['STOP', 'x S'], 'x', True), ('', ['a'], '', False)]:
    for trial in range(30):
        cuts = sorted(random.sample(range(1, len(text)), min(len(text) - 1, random.randint(0, 4)))) if len(text) > 1 else []
        parts = [text[a:b] for a, b in zip([0] + cuts, cuts + [len(text)])]
        st, out, h = chat.Stop(stops), '', False
        for d in parts:
            e, h = st.feed(d)
            out += e
            assert not any(x in out for x in stops), (out, stops)
            if h:
                break
        if not h:
            out += st.flush()
        assert (out, h) == (want, hit), (text, parts, out, h)
print('stop matcher ok')


class WholeParser(chat.Parser):  # the reference: the old parser, which re-decoded the whole body every token
    def _flush(self, final=False):
        if self.kind not in ('reasoning', 'content'):
            return []
        s = self.tok.decode(self.body, skip_special_tokens=False)
        if not final and s.endswith('�'):
            return []
        d, self.sent = s[self.sent:], len(s)
        return [(self.kind, d)] if d else []


# incremental parser == whole-body decode, delta for delta, on random outputs: real multilingual/emoji text, random ids
# (byte fragments of multi-byte chars), lone byte tokens, literal U+FFFD, specials and message switches
rng = random.Random(0)
vocab = tok.get_vocab()
bytes1 = [i for s, i in vocab.items() if len(s) == 1 and i < chat.USER]
texts = ['Café naïve 日本語のテキスト 中文混合。한국어 🚀🔥👍🏽 👨‍👩‍👧 ∑∫√π≈3.14', 'plain ascii text, with code `x**2`\n',
         '� replacement � chars �', 'Ελληνικά и русский текст, עברית, العربية', '🇺🇸🇯🇵 ZWJ: 🧑‍💻 skin: 👋🏿']
pieces = [enc(t) for t in texts]
specials = [chat.MODEL, chat.THINK, chat.TEXT, chat.END, chat.INVOKE_JSON, chat.TOOL_ERR, chat.IMG_TOK, 200030]
for trial in range(3000):
    seq = [rng.choice([chat.THINK, chat.TEXT])] if rng.random() < 0.5 else [chat.MODEL, rng.choice([chat.THINK, chat.TEXT])]
    for _ in range(rng.randint(1, 60)):
        r = rng.random()
        seq += (rng.choice(pieces)[rng.randrange(8):][:rng.randint(1, 30)] if r < 0.35 else
                [rng.randrange(chat.USER)] if r < 0.6 else [rng.choice(bytes1)] * rng.randint(1, 3) if r < 0.85 else
                [rng.choice(specials)])
    a, b = chat.Parser(tok), WholeParser(tok)
    for t in seq:
        da, db = a.feed(t), b.feed(t)
        assert da == db, (trial, seq, da, db)
    assert a.tool_calls == b.tool_calls
print('incremental parser == whole-body decode (3000 random outputs)')

# server: bad n / logprobs / stop rejected (n > 1 and logprobs themselves: tests/test_api.py), stop normalized,
# tool_choice none drops tools
seen.clear()
bad = [dict(base, n=99), dict(base, logprobs='yes'), dict(base, stop=['a'] * 5), dict(base, stop=3)]
assert [c.post('/v1/chat/completions', json=b).status_code for b in bad] == [400] * 4
for b in (dict(base, stop='###', n=1), dict(base, stop=['a', ''], tools=tools, tool_choice='none'),
          dict(base, tools=tools, tool_choice='auto')):
    assert c.post('/v1/chat/completions', json=b).status_code == 200
assert seen[0]['stop'] == ['###'] and seen[1]['stop'] == ['a'] and seen[1]['tools'] is None, seen
assert seen[1]['prefill_text'] is True and seen[2]['tools'] == tools and seen[2]['stop'] is None, seen
print('server params ok (n, logprobs, stop, tool_choice)')


def render_ref(tok, messages, tools=None, effort=0.9, image_tokens=None):  # chat.render before the encode memo (T1)
    enc = lambda s: tok.encode(s, add_special_tokens=False).ids
    ids = []
    if tools:
        decl = [{'description': t.get('function', t).get('description', ''), 'name': t.get('function', t)['name'],
                 'parameters': t.get('function', t).get('parameters', {}), 'type': t.get('type', 'function')}
                for t in tools]
        ids += [chat.SYSTEM] + enc('tool_declare') + [chat.XML] + enc(json.dumps(decl, sort_keys=True, separators=(',', ':'),
                                                                                 ensure_ascii=False)) + [chat.END]
    effort_ids = [chat.SYSTEM, chat.TEXT] + enc(f'Thinking effort level: {chat.fmt_effort(effort)}') + [chat.END]
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
            r = chat.SYSTEM if role != 'user' else chat.USER
            for p in parts:
                if p.get('type') == 'text':
                    ids += [r, chat.TEXT] + enc(p['text']) + [chat.END]
                elif p.get('type') in ('image_url', 'image') and role == 'user':
                    ids += [r, chat.IMAGE] + [chat.IMG_TOK] * image_tokens(p) + [chat.END]
        elif role == 'tool':
            name = m.get('name') or call_names.get(m.get('tool_call_id'), '')
            for p in parts:
                if p.get('type') == 'text':
                    ids += [chat.TOOL] + enc(name) + [chat.TEXT] + enc(p['text']) + [chat.END]
                elif p.get('type') in ('image_url', 'image'):
                    ids += [chat.TOOL] + enc(name) + [chat.IMAGE] + [chat.IMG_TOK] * image_tokens(p) + [chat.END]
        elif role == 'assistant':
            text = ''.join(p.get('text', '') for p in parts if p.get('type') == 'text')
            rc = m.get('reasoning_content') or m.get('reasoning')
            calls = m.get('tool_calls') or []
            if rc:
                ids += [chat.MODEL, chat.THINK] + enc(rc) + [chat.END]
            if text or not (rc or calls):
                ids += [chat.MODEL, chat.TEXT] + enc(text) + [chat.END]
            for c_ in calls:
                fn = c_.get('function', c_)
                args = fn.get('arguments', {})
                args = json.loads(args) if isinstance(args, str) and args else (args or {})
                call_names[c_.get('id')] = fn['name']
                body = ('{"name":' + json.dumps(fn['name'], ensure_ascii=False) + ',"args":{' +  # chat_template.jinja
                        ','.join(json.dumps(k, ensure_ascii=False) + ':' + json.dumps(v, separators=(',', ':'), ensure_ascii=False)
                                 for k, v in sorted(args.items())) + '}}')
                ids += [chat.MODEL] + enc(fn['name']) + [chat.INVOKE_JSON] + enc(body) + [chat.END]
            ids += [chat.EOS]
    if not placed:
        ids += effort_ids
    return ids


# T1: render with the encode memo == the old render, on random agent conversations replayed turn by turn (every turn
# re-renders the whole history, so later turns mostly hit the memo): system/developer/user/assistant/tool, text +
# image parts, reasoning, tool calls (string/dict/empty args), tools, efforts, texts at the memo's MIN boundary,
# whitespace/unicode/special-looking text, repeated texts in different roles. Then the memo's byte bound (LRU).
rng = random.Random(1)
words = ['the', ' ', '\n', '\n\n', '  ', 'def f(x):\n    return x', '日本語', '🚀', '👨‍👩‍👧', 'é', '<|im_start|>', '</s>',
         '{"a": 1}', '\t', 'Ω', 'ab' * 40, ' trailing ', 'ünïcødé', '\r\n', '0123456789', '<tool_call>', '� ']


def rtext():
    t = ''.join(rng.choice(words) for _ in range(rng.choice([0, 1, 3, 10, 40, 200])))
    return t if rng.random() < 0.9 else t[:chat.EncCache.MIN + rng.randint(-2, 2)]


def rimg():
    return {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{rng.randrange(10 ** 6)}'}}


ntiles = lambda part: 1 + int(part['image_url']['url'].rsplit(',', 1)[1]) % 7
TOOLS2 = tools + [{'type': 'function', 'function': {'name': 'run', 'description': rtext(), 'parameters': {}}}]
cache = chat.encoder(tok)
total = 0
for conv in range(60):
    msgs, shared = [], [rtext() for _ in range(3)]
    if rng.random() < 0.5:
        msgs.append({'role': rng.choice(['system', 'developer']), 'content': rtext()})
    tl, eff = rng.choice([None, tools, TOOLS2]), rng.choice([0, 0.2, 0.9, 'high', 'none'])
    for turn in range(rng.randint(1, 12)):
        k = rng.random()
        if k < 0.35:
            parts = [rng.choice([{'type': 'text', 'text': rng.choice(shared + [rtext()])}, rimg()])
                     for _ in range(rng.randint(1, 3))]
            msgs.append({'role': 'user', 'content': rng.choice([parts, rtext()])})
        elif k < 0.7:
            m = {'role': 'assistant', 'content': rng.choice([None, '', rtext(), [{'type': 'text', 'text': rtext()}]])}
            if rng.random() < 0.5:
                m[rng.choice(['reasoning_content', 'reasoning'])] = rng.choice(shared + [rtext()])
            if rng.random() < 0.5:
                m['tool_calls'] = [{'id': f'c{conv}_{turn}_{j}', 'type': 'function', 'function': {
                    'name': rng.choice(['get_weather', 'run']),
                    'arguments': rng.choice(['', '{}', json.dumps({'city': rtext()}), {'q': rtext(), 'n': 3}])}}
                    for j in range(rng.randint(1, 2))]
            msgs.append(m)
        elif k < 0.9:
            calls = [c_['id'] for m in msgs for c_ in m.get('tool_calls') or []]
            msgs.append({'role': 'tool', 'tool_call_id': rng.choice(calls or ['none']),
                         'content': rng.choice([rtext(), [{'type': 'text', 'text': rtext()}, rimg()]])})
        else:
            msgs.append({'role': rng.choice(['system', 'developer', 'tool']), 'name': rng.choice([None, 'x']),
                         'content': rng.choice(shared)})
        a_, b_ = chat.render(tok, msgs, tl, eff, ntiles), render_ref(tok, msgs, tl, eff, ntiles)
        assert a_ == b_ and type(a_) is list, (conv, turn, msgs)
        total += 1
print(f'T1 render with the encode memo == old render ({total} renders of 60 random conversations, memo '
      f'{len(cache.d)} pieces, {cache.size >> 10} KiB)')
small = chat.EncCache(tok, 4096)
pieces = [f'piece {i} ' * 20 for i in range(40)]
for p_ in pieces * 2:
    assert list(small(p_)) == enc(p_)
assert small.size <= 4096 and 0 < len(small.d) < 40 and small.size == sum(len(v) * 4 + 100 for v in small.d.values())
small(pieces[-5]); small(pieces[0])               # LRU: a hit moves to the end; the oldest goes first
assert list(small.d.values())[-1] == small(pieces[0]) and len(small(pieces[0]))
print(f'T1 memo bounded by bytes: {small.size} B <= 4096, {len(small.d)} of 40 kept (LRU)')

# T2: images in the history (before the last assistant message) get rows/cols from the header only and keep their raw
# bytes; Engine._pixels decodes exactly those past the restored prefix. ids, keys and (rows, cols, hash) == the old
# prepare (decode everything), decoded tiles identical; also with INK_IMG_MAX_SIDE; bad images still BadRequest.
import base64
import hashlib
import io
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

import engine
import vision
from engine import BadRequest, Engine


def img_url(mode, size, fmt='PNG', seed=0):
    a = np.random.default_rng(seed).integers(0, 255, (size[1], size[0], 4), dtype=np.uint8)
    im = Image.fromarray(a, 'RGBA')
    im = im.convert(mode) if mode != 'I;16' else Image.fromarray(a[..., 0].astype(np.uint16) * 257)
    b = io.BytesIO()
    im.save(b, fmt)
    return {'type': 'image_url', 'image_url': {'url': 'data:image/x;base64,' + base64.b64encode(b.getvalue()).decode()}}


def old_prepare(E, msgs):  # Engine.prepare before T2: every image decoded + tiled
    pre = []

    def count(part):
        raw = engine.image_bytes(part)
        t, r, c = vision.preprocess(vision.load_image(raw))
        pre.append((t, r, c, int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), 'little')))
        return r * c
    return chat.render(tok, msgs, None, 0.9, count), pre


E = Engine.__new__(Engine)
E.tok, E.vision = tok, object()
imgs = [img_url('RGB', (100, 57)), img_url('RGBA', (41, 80), seed=1), img_url('L', (64, 40), 'JPEG', 2),
        img_url('P', (90, 33), seed=3), img_url('I;16', (50, 50), seed=4), img_url('RGB', (130, 20), 'JPEG', 5)]
msgs = [{'role': 'system', 'content': 'sys'}, {'role': 'user', 'content': [{'type': 'text', 'text': 'look'}, imgs[0], imgs[1]]},
        {'role': 'assistant', 'content': 'ok', 'tool_calls': [{'id': 'a', 'function': {'name': 'shot', 'arguments': '{}'}}]},
        {'role': 'tool', 'tool_call_id': 'a', 'content': [imgs[2], {'type': 'text', 'text': 'shot'}, imgs[3]]},
        {'role': 'assistant', 'content': 'seen'}, {'role': 'user', 'content': [imgs[4], imgs[5], {'type': 'text', 'text': '?'}]}]
for side in (0, 60):
    engine.MAX_SIDE = vision.MAX_SIDE = side
    ids, pre = E.prepare(msgs)
    rids, rpre = old_prepare(E, msgs)
    assert ids == rids and [x[1:] for x in pre] == [x[1:] for x in rpre]
    assert torch.equal(Engine.keys(ids, pre), Engine.keys(rids, rpre))
    assert [isinstance(x[0], bytes) for x in pre] == [True] * 4 + [False] * 2  # history raw, the new turn decoded
    pos = (Engine.keys(ids, pre) < 0).nonzero().flatten().tolist()
    ends = np.cumsum([r * c for _, r, c, _ in pre]) - 1
    for P, n_raw in ((int(pos[ends[1]]) + 1, 2), (int(pos[ends[1]]), 1), (0, 0), (len(ids), 4)):
        req = SimpleNamespace(pre=list(pre), keys=Engine.keys(ids, pre), P=P)
        Engine._pixels(req)
        assert sum(isinstance(x[0], bytes) for x in req.pre) == n_raw, (P, n_raw)
        for x, y in zip(req.pre, rpre):
            assert isinstance(x[0], bytes) or (torch.equal(x[0], y[0]) and x[1:] == y[1:])
engine.MAX_SIDE = vision.MAX_SIDE = 0
junk = {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(b'not an image').decode()}}
for bad in ([{'role': 'user', 'content': [junk]}, {'role': 'assistant', 'content': 'x'}, {'role': 'user', 'content': 'y'}],
            [{'role': 'user', 'content': [junk]}]):
    try:
        E.prepare(bad); raise AssertionError('bad image accepted')
    except BadRequest:
        pass
good = base64.b64decode(imgs[0]['image_url']['url'].split(',', 1)[1])
cut = {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(good[:len(good) // 2]).decode()}}
hist = [{'role': 'user', 'content': [cut]}, {'role': 'assistant', 'content': 'x'}, {'role': 'user', 'content': 'y'}]
ids, pre = E.prepare(hist)  # valid header, truncated pixels: found when (and only if) it has to be decoded
try:
    Engine._pixels(SimpleNamespace(pre=pre, keys=Engine.keys(ids, pre), P=0)); raise AssertionError('truncated accepted')
except BadRequest:
    pass
print('T2 lazy images: ids/keys/(rows, cols, hash) == old prepare (6 formats, MAX_SIDE 0/60), only images past P '
      'decoded (tiles identical), bad images -> BadRequest')
print('ALL OK')
