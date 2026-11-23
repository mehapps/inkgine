"""Capture oracle outputs from the official SGLang Inkling server (black box) for fixed prompts.
Usage (inside a test window, oracle on :30001): python tests/oracle.py [http://localhost:30001]
Writes tests/oracle.json: per prompt the rendered ids, SGLang's top-5 prompt logprobs, and greedy chat outputs."""
import base64
import io
import json
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from tokenizers import Tokenizer  # noqa: E402

import chat  # noqa: E402
from engine import MODEL_DIR  # noqa: E402

BASE = sys.argv[1] if len(sys.argv) > 1 else 'http://localhost:30001'


def post(path, body, timeout=1800):
    r = urllib.request.Request(BASE + path, json.dumps(body).encode(), {'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(r, timeout=timeout))


NAMES = ('Mara', 'Tobias', 'Ilse', 'Rafael', 'Nadia', 'Old Keller', 'the miller', 'Ines', 'the schoolteacher',
         'Oskar', 'the ferryman', 'Lucia', 'Bram', 'the widow Hart', 'Yusuf', 'Petra', 'the young clerk', 'Anselm')
PLACES = ('the harbour', 'the orchard behind the chapel', 'the salt marsh', 'the upper meadow', 'the bakery on Vane Street',
          'the lighthouse stairs', 'the railway cutting', 'the old customs house', 'the reed beds', 'the market square',
          'the ridge above the quarry', 'the library reading room', 'the narrow bridge', 'the boatyard')
THINGS = ('a ledger with a cracked spine', 'two letters tied with string', 'a brass compass', 'a basket of late pears',
          'a map drawn on sailcloth', 'a jar of honey', 'a broken lantern', 'a coil of tarred rope', 'a pocket watch',
          'a sketchbook full of birds', 'a loaf still warm from the oven', 'a tin of buttons', 'a wooden flute')
WEATHER = ('A cold wind came off the water', 'Rain had fallen since dawn', 'The fog lifted slowly',
           'It was the first warm day of spring', 'Snow lay thin on the rooftops', 'The evening light turned copper',
           'Thunder rolled somewhere far inland', 'The tide was unusually low', 'A dry heat settled over the fields')
FEEL = ('uneasy', 'quietly pleased', 'tired but stubborn', 'curious', 'homesick', 'amused', 'anxious', 'wary',
        'unexpectedly hopeful', 'irritated', 'grateful', 'restless', 'ashamed', 'oddly calm')
TIME = ('Later that week', 'Before the bells rang', 'In the grey hour before breakfast', 'Some years earlier',
        'On the morning of the fair', 'After the funeral', 'Long after midnight', 'When the boats came in',
        'Two days before the wedding', 'During the long dry summer', 'On the last Sunday of the month')
TAIL = (' Nobody said anything more about it.', ' The dog followed at a distance.', ' It would matter later.',
        ' Somewhere a door banged in the wind.', ' The church clock was, as usual, ten minutes slow.',
        ' Afterwards nobody could agree on what had been said.', ' It was not the first time.',
        ' The smell of tar and wet rope hung over everything.')
SENT = (
    '{W}, and {a} walked down to {p} carrying {t}.',
    '{a} had not spoken to {b} since the argument about {t}, and still felt {f}.',
    'At {p}, {a} found {t} that nobody would admit to owning.',
    '{b} said that {p} used to flood every autumn before the new wall was built.',
    'Nobody in the village could remember who first brought {t} to {p}.',
    '{a} asked {b} whether the road past {p} would be open by Thursday.',
    'For a long while {a} sat at {p}, {f}, watching the gulls argue over scraps.',
    '{W}. {b} lit the stove and began, without hurry, to mend {t}.',
    'It was {b} who noticed that {t} had gone missing from {p}.',
    '"We could sell {t}," {a} suggested, but {b} only shook their head.',
    '{a} wrote in the margin of the ledger that prices at {p} had risen again.',
    'By noon the talk at {p} had turned, as it often did, to {b} and the unpaid debt.',
    '{b} remembered a winter when {t} had been worth more than a week of wages.',
    '{a} counted the coins twice, felt {f}, and set off toward {p} before dark.',
    'Children ran shouting through {p}, and {a}, {f}, let them pass.',
    '{T}, {a} and {b} met at {p} to settle the matter of {t}.',
    '{T}, a stranger arrived at {p} asking for {a} by name.',
    '{T}, {b} sold {t} to a trader from the south for far less than it was worth.',
    '{T}, the council voted to close {p}, and {a} was {f} for days.',
    '{a} believed {b} was lying about {t}, but could not prove it.',
    'The path to {p} was steep, and {a} stopped twice to rest.',
    'Whenever {b} passed {p}, they touched the old stone by the gate for luck.',
    '{a} kept {t} wrapped in a cloth on the top shelf, out of reach of the cat.',
    'There was a rumour that {b} had once been engaged to the ferryman\'s brother.',
    '{T}, {a} climbed to {p} and looked out over the grey roofs, {f}.',
)


def prose(target, tok, seed):
    """Deterministic, varied (non-repeating) story text of about `target` tokens."""
    import random
    rng, paras, n = random.Random(seed), [], 0
    while n < target:
        para = []
        for _ in range(rng.randint(3, 7)):
            a, b = rng.sample(NAMES, 2)
            s = rng.choice(SENT).format(a=a, b=b, p=rng.choice(PLACES), t=rng.choice(THINGS), W=rng.choice(WEATHER),
                                        f=rng.choice(FEEL), T=rng.choice(TIME))
            para.append(s[0].upper() + s[1:] + (rng.choice(TAIL) if rng.random() < 0.25 else ''))
        paras.append(' '.join(para))
        n += len(tok.encode(paras[-1], add_special_tokens=False).ids) + 2
    return '\n\n'.join(paras)


def text_prompts(tok):
    """Teacher-forced prompts: short chat, ~3k (exercises distance >= 1024), ~40k (far keys, several chunks)."""
    mid = prose(3000, tok, 1)
    long = prose(40000, tok, 2).split('\n\n')
    long.insert(len(long) // 2, 'That same evening Petra hid the spare key to the boathouse inside the tin of buttons.')
    return {
        'short': [{'role': 'user', 'content': 'Explain in two sentences why the sky is blue.'}],
        'mid': [{'role': 'user', 'content': mid + '\n\nSummarize the story above in one sentence.'}],
        'long': [{'role': 'user', 'content': '\n\n'.join(long) + '\n\nWhere did Petra hide the spare key to the boathouse?'}],
    }


def image_url(path):
    return 'data:image/jpeg;base64,' + base64.b64encode(open(path, 'rb').read()).decode()


def gen(ids, start):
    body = {'input_ids': ids, 'sampling_params': {'max_new_tokens': 16, 'temperature': 0}}
    if start is not None:
        body.update(return_logprob=True, logprob_start_len=start, top_logprobs_num=5)
    return post('/generate', body)


def main():
    tok = Tokenizer.from_file(os.path.join(MODEL_DIR, 'tokenizer.json'))
    path = os.path.join(HERE, 'oracle.json')
    out = {'teacher': {}, 'chat': {}}
    save = lambda: json.dump(out, open(path, 'w'))   # after every prompt, so a later failure keeps earlier data
    for name, msgs in text_prompts(tok).items():
        ids = chat.render(tok, msgs, effort=0.9)
        start = max(0, len(ids) - 512)   # last 512 positions: a 3k-position request timed out inside SGLang
        try:
            r = gen(ids, start)
        except Exception as e:
            print(name, 'logprob request failed:', e, '- retrying greedy only', flush=True)
            try:
                r, start = gen(ids, None), None
            except Exception as e2:
                print(name, 'FAILED:', e2, flush=True); continue
        mi = r['meta_info']
        out['teacher'][name] = {'ids': ids, 'start': start, 'input_top_logprobs': mi.get('input_top_logprobs'),
                                'output_ids': r.get('output_ids'), 'text': r.get('text')}
        save()
        print(name, len(ids), 'tokens; greedy:', repr(r.get('text', ''))[:80], flush=True)
    gate_img = os.path.expanduser('~/llm/gate/img16mp.jpg')
    for name, content, mt in (
        ('gateway', [{'type': 'image_url', 'image_url': {'url': image_url(gate_img)}},
                     {'type': 'text', 'text': 'What word is written in large letters in this image? Answer with the word only.'}], 20),
        ('capital', 'What is the capital of France? Answer in one word.', 10),
    ):
        try:
            r = post('/v1/chat/completions', {'model': 'x', 'messages': [{'role': 'user', 'content': content}],
                                              'max_tokens': mt, 'temperature': 0,
                                              'chat_template_kwargs': {'enable_thinking': False}})
        except Exception as e:
            print(name, 'FAILED:', e, flush=True); continue
        out['chat'][name] = r['choices'][0]['message']
        save()
        print(name, '->', r['choices'][0]['message'], flush=True)
    print('wrote tests/oracle.json')


if __name__ == '__main__':
    main()
