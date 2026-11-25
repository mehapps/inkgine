"""response_format (json_schema / json_object) as token masks via xgrammar.

The grammar covers the reply's final text message only: with thinking on, reasoning is free and the model is held to
MODEL -> (THINK ... END -> MODEL)* -> TEXT -> grammar -> END; with thinking off (prompt already ends in MODEL TEXT) the
grammar starts at the first token. Generation stops when the grammar accepts END.
"""
import json

import torch
import xgrammar as xgr

import chat

VOCAB = 200058  # unpadded logits (model.VOCAB); ids >= chat.USER are special tokens
_compiler = None
_SHIFTS = {}  # device -> arange(32) int32 (bit positions of a bitmask word)


def compiler(tok):
    global _compiler
    if _compiler is None:  # ~3 s once: 200k-token vocab analysis
        enc = [''] * VOCAB
        for s, i in tok.get_vocab(with_added_tokens=True).items():
            if i < chat.USER:  # specials stay '' so xgrammar never allows them inside the JSON
                enc[i] = s
        info = xgr.TokenizerInfo(enc, xgr.VocabType.BYTE_LEVEL, vocab_size=VOCAB, stop_token_ids=[chat.END])
        _compiler = xgr.GrammarCompiler(info)
    return _compiler


def compile_format(tok, rf):
    """response_format dict -> CompiledGrammar, or None for plain text. ValueError on a bad request."""
    t = (rf or {}).get('type', 'text')
    if t == 'text':
        return None
    if t == 'json_object':
        schema = {'type': 'object'}
    elif t == 'json_schema':
        js = rf.get('json_schema')
        if not isinstance(js, dict) or not isinstance(js.get('schema', {}), dict):
            raise ValueError('response_format.json_schema.schema must be an object')
        schema = js.get('schema') or {'type': 'object'}
    else:
        raise ValueError(f'unknown response_format type {t!r}')
    try:  # no free whitespace: a model can't loop on spaces/newlines inside the JSON
        return compiler(tok).compile_json_schema(json.dumps(schema), any_whitespace=False)
    except Exception as e:
        raise ValueError(f'bad json schema: {e}')


def mask_rows(lg, cons):
    """lg [R, VOCAB] float, cons [Json or None] per row: every constrained row masked in place. The json rows'
    bitmasks go to the device in one copy and are applied there in one masked_fill (no [VOCAB] bool per row on the host)."""
    js = []
    for i, c in enumerate(cons):
        if c is None:
            continue
        if c.state == 'json':
            c.m.fill_next_token_bitmask(c.bits)
            js.append(i)
        else:
            c.mask(lg[i])
    if js:
        dev = lg.device
        ok = unpack(torch.cat([cons[i].bits for i in js]), dev)
        if len(js) == lg.shape[0]:
            lg.masked_fill_(~ok, float('-inf'))
        else:
            g = torch.tensor(js, device=dev)
            lg[g] = lg.index_select(0, g).masked_fill_(~ok, float('-inf'))


def unpack(bits, dev, non_blocking=False):
    """Host bitmask [R, words] int32 -> allowed [R, VOCAB] bool on dev (one copy; non_blocking needs pinned bits)."""
    sh = _SHIFTS.get(dev)
    if sh is None:
        sh = _SHIFTS[dev] = torch.arange(32, dtype=torch.int32, device=dev)
    bits = bits.to(dev, non_blocking=non_blocking)
    return ((bits[:, :, None] >> sh) & 1).bool().view(len(bits), -1)[:, :VOCAB]


class Tools:
    """tool_choice 'required' / a named function (forced) or 'auto' (lazy: a call is constrained only once it starts),
    compiled once per request: the canonical token ids of each allowed name (as chat.render writes a call: MODEL name
    INVOKE_JSON {"name":..,"args":..} END) and the call-JSON grammar per name. ToolCall (per Request) is the matcher."""

    def __init__(self, tok, tools, names, think, parallel, lazy=False):
        by = {t.get('function', t)['name']: t.get('function', t) for t in tools}
        self.think, self.parallel, self.lazy, self.names, self.compiled, self.next = think, parallel, lazy, {}, {}, {}
        for n in names:
            ids = tuple(tok.encode(n, add_special_tokens=False).ids)
            self.names[ids] = n
            for i in range(len(ids)):
                self.next.setdefault(ids[:i], set()).add(ids[i])
            self.next.setdefault(ids, set()).add(chat.INVOKE_JSON)
            params = by[n].get('parameters') or {'type': 'object'}
            if isinstance(params.get('properties'), dict):  # args keys in sorted order, as chat.call_body writes them
                params = dict(params, properties=dict(sorted(params['properties'].items())))
            schema = {'type': 'object', 'properties': {'name': {'enum': [n]}, 'args': params},  # name first (template)
                      'required': ['name', 'args'], 'additionalProperties': False}
            try:
                # compact separators: calls are rendered (and were trained) as json.dumps(separators=(',', ':'))
                self.compiled[n] = compiler(tok).compile_json_schema(json.dumps(schema), any_whitespace=False,
                                                                     separators=(',', ':'))
            except Exception as e:
                raise ValueError(f'bad parameters schema for tool {n!r}: {e}')


def compile_tools(tok, tools, choice, think, parallel=True):
    """tool_choice -> Tools for 'required' / a named function, lazy Tools for 'auto' with tools, None for 'none' / no
    tools. ValueError on a bad request."""
    names = [t.get('function', t).get('name') for t in tools or []]
    if choice == 'none' or (choice in (None, 'auto') and not names):
        return None
    if choice in (None, 'auto'):
        return Tools(tok, tools, names, think, parallel, lazy=True)
    if choice == 'required':
        if not names:
            raise ValueError("tool_choice 'required' needs tools")
        want = names
    elif isinstance(choice, dict) and choice.get('type') == 'function' and isinstance(choice.get('function'), dict):
        n = choice['function'].get('name')
        if n not in names:
            raise ValueError(f'tool_choice names {n!r}, which is not in tools')
        # one call, as OpenAI: with parallel calls the model went on to the other tools it wanted, and every one was
        # forced into this name (copies of the call, or its args holding the other tools' names)
        return Tools(tok, tools, [n], think, parallel=False)
    else:
        raise ValueError(f'unknown tool_choice {choice!r}')
    return Tools(tok, tools, want, think, parallel)


class ToolCall:
    """Forces the reply to contain a tool call: MODEL [THINK ... END MODEL] name INVOKE_JSON call-JSON END, then
    (parallel) free generation (more calls or EOS) or done. Free generation (where a lazy Tools, tool_choice 'auto',
    starts) is unmasked; a MODEL there leads to 'lhead': a block header (any special token but a nameless
    INVOKE_JSON: SPEC §1, a name precedes a call) or the first token of a declared name, which starts a call matched
    like a forced one (name, INVOKE_JSON, call-JSON END, then free again, or done without parallel calls). Forced: at
    most one reasoning block (thinking on), of at most think_max tokens (then END is forced): window g2 11-23, with
    thinking on the model wanted a text answer, TEXT is masked, so it opened reasoning block after reasoning block
    until max_tokens and never called. Same interface as Json
    (state / mask / accept / done / m / bits) so mask_rows and spec_runner's draft masks work; every non-matcher state
    lives in `state` (immutable: ('think', n) carries the count), since spec_runner rolls back by restoring state + done
    and m.rollback(json tokens). m is a fresh matcher on each entry to 'json', so a draft that crossed into it and was
    rolled back leaves nothing behind (spec_runner also restores m: a draft may end one call and start the next)."""

    def __init__(self, tools, think_max=None):
        self.t, self.state, self.done, self.m = tools, 'free' if tools.lazy else 'start', False, None
        self.think_max = think_max
        self.bits = xgr.allocate_token_bitmask(1, VOCAB)

    def _keep(self, logits, keep):
        keep = list(keep)
        v = logits[keep].clone()
        logits.fill_(float('-inf'))
        logits[keep] = v

    def mask(self, logits):
        s = self.state
        if s == 'free':  # a call starts only through MODEL + a declared name (lhead): else, after MODEL MODEL name or
            logits[chat.INVOKE_JSON] = float('-inf')  # right after an END, it would call any tool the model made up
        elif s == 'json':
            mask_rows(logits[None], [self])
        elif s == 'lhead':
            keep = list(self.t.next[()])
            v, w = logits[keep].clone(), logits[chat.USER:VOCAB].clone()
            logits.fill_(float('-inf'))
            logits[chat.USER:VOCAB] = w
            logits[keep] = v
            logits[chat.INVOKE_JSON] = float('-inf')
        elif s[0] == 'think':
            if self.think_max is not None and s[1] >= self.think_max:
                self._keep(logits, [chat.END])  # reasoning budget spent: the call needs the rest
            else:
                logits[chat.EOS] = float('-inf')  # the call must still come
        elif s in ('start', 'start1'):
            self._keep(logits, [chat.MODEL])
        elif s in ('header', 'header1'):  # (header1: after the reasoning block, a name only)
            self._keep(logits, self.t.next[()] | ({chat.THINK} if self.t.think and s == 'header' else set()))
        else:  # ('name', prefix)
            self._keep(logits, self.t.next[s[1]])
        return logits

    def allows(self, t):
        """Whether mask() leaves token t finite, outside 'json' (host only, no [VOCAB] row)."""
        s, nx = self.state, self.t.next
        if s == 'free':
            return t != chat.INVOKE_JSON
        if s == 'lhead':
            return t != chat.INVOKE_JSON and (t in nx[()] or chat.USER <= t < VOCAB)
        if s[0] == 'think':
            return t == chat.END if self.think_max is not None and s[1] >= self.think_max else t != chat.EOS
        if s in ('start', 'start1'):
            return t == chat.MODEL
        if s in ('header', 'header1'):
            return t in nx[()] or (t == chat.THINK and self.t.think and s == 'header')
        return t in nx[s[1]]

    def accept(self, t, check=False):
        """Advance by token t. check: a token mask() bans is refused (-> False, nothing changes) instead of
        raising (json) or being taken unchecked (the other states)."""
        s = self.state
        if s == 'json':
            # (specials but END are never in the bitmask; accept_token would reject them too, logging a warning each)
            if check and t >= chat.USER and t != chat.END or not self.m.accept_token(t):
                if check:
                    return False
                raise RuntimeError(f'grammar rejected token {t}')
            if self.m.is_terminated():
                self.state, self.done = 'free', not self.t.parallel
            return True
        if check and not self.allows(t):
            return False
        if s == 'free':
            if t == chat.MODEL:
                self.state = 'lhead'
        elif s == 'lhead':
            self.state = 'free' if t >= chat.USER else ('name', (t,))
        elif s in ('start', 'start1'):
            self.state = 'header' if s == 'start' else 'header1'
        elif s in ('header', 'header1'):
            self.state = ('think', 0) if t == chat.THINK else ('name', (t,))
        elif s[0] == 'think':
            self.state = 'start1' if t == chat.END else ('think', s[1] + 1)
        else:  # ('name', prefix)
            if t == chat.INVOKE_JSON:
                self.m, self.state = xgr.GrammarMatcher(self.t.compiled[self.t.names[s[1]]]), 'json'
            else:
                self.state = ('name', s[1] + (t,))
        return True


class Json:
    def __init__(self, compiled, in_text):
        self.m = xgr.GrammarMatcher(compiled)
        self.state = 'json' if in_text else 'start'
        self.bits = xgr.allocate_token_bitmask(1, VOCAB)
        self.done = False

    def mask(self, logits):
        """logits [VOCAB] float, masked in place."""
        if self.state == 'json':
            mask_rows(logits[None], [self])
        elif self.state == 'think':
            logits[chat.EOS] = float('-inf')  # the reply must still get its JSON message
        else:
            keep = [chat.MODEL] if self.state == 'start' else [chat.THINK, chat.TEXT]
            v = logits[keep].clone()
            logits.fill_(float('-inf'))
            logits[keep] = v
        return logits

    def allows(self, t):
        """Whether mask() leaves token t finite, outside 'json' (host only)."""
        if self.state == 'think':
            return t != chat.EOS
        return t == chat.MODEL if self.state == 'start' else t in (chat.THINK, chat.TEXT)

    def accept(self, t, check=False):
        """As ToolCall.accept."""
        s = self.state
        if s == 'json':
            # (specials but END are never in the bitmask; accept_token would reject them too, logging a warning each)
            if check and t >= chat.USER and t != chat.END or not self.m.accept_token(t):
                if check:
                    return False
                raise RuntimeError(f'grammar rejected token {t}')
            self.done = self.m.is_terminated()
            return True
        if check and not self.allows(t):
            return False
        if s == 'start':
            self.state = 'header'
        elif s == 'header':
            self.state = 'json' if t == chat.TEXT else 'think'
        elif t == chat.END:
            self.state = 'start'
        return True
