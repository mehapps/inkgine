"""Expert sharing measurement (DESIGN-batching.md sec. 4.2 step 0 / sec. 5), CPU, real weights, no engine edits.

Prefills several different natural prompts (2 each of prose, code, chat, json task) and records the top-8 expert ids
of every token at every one of the 48 layer visits (Model.moe wrapped; same routing formula as model.py). Then:
- batch: B tokens from B *different* prompts at random positions -> mean distinct experts U per visit
- verify: T consecutive tokens of one prompt -> mean U per visit
and converts U to bytes per decode step and tok/s with the design's byte model (sec. 4.3).
Raw routing is saved to $INK_LOGS/ink-expert-routing.pt; INK_STATS_ONLY=1 re-runs only the analysis from it.
Run detached (~10-30 min on CPU), see NOTES.md.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import torch  # noqa: E402

import model as M  # noqa: E402

OUT = os.path.join(os.environ.get('INK_LOGS', os.path.expanduser('~/llm/logs')), 'ink-expert-routing.pt')
SKIP = 24          # leading template tokens (identical effort header in every prompt) are not sampled
NVFP4_B = 2304 * 1152 + 2304 * 144 + 2304 * 576 + 2304 * 72     # w13 + scale + w2 + scale, uint8/e4m3: 4.48 MB
MXFP8_B = 2304 * 2304 + 2304 * 72 + 2304 * 1152 + 2304 * 36     # e4m3 + e8m0 scales: 8.21 MB
VBYTES = torch.tensor([NVFP4_B if L < 20 else MXFP8_B for L in M.VISITS], dtype=torch.float64)
DENSE, RING, BW = 3.68e9, 0.19e9, 1.4e12                         # design sec. 4.3
FIXED = {'today': 5.9e-3, 'tuned': 1.0e-3}

PROSE1 = """The lighthouse keeper had not spoken to another person in eleven days when the supply boat finally rounded
the point. She watched it from the gallery, one hand on the cold iron rail, and tried to remember how conversation
worked. You said something about the weather, she thought, and then the other person said something about the
weather, and after that it got easier. The boat was smaller than usual and rode low in the water. The man at the
tiller was not Tomas, who had brought her flour and kerosene every second Tuesday for six years, but someone younger,
with a red knitted cap and a way of looking up at the tower as if he expected it to fall on him. She went down the
spiral stairs slowly, counting them out of habit, one hundred and twelve, and was standing on the jetty when he threw
the rope. He missed the bollard twice. On the third try she caught the line herself and made it fast, and he
apologized in a voice that cracked halfway through the word. Tomas had broken his hip, he explained. Slipped on the
fish market steps. He would be fine, the doctors said, but not for a while, and in the meantime there was nobody
else, so here he was. She asked his name. Daniel. She asked whether he had ever handled a boat in a westerly, and he
admitted that he had not, and she decided she liked him for not pretending."""

PROSE2 = """Soil is not dirt. Dirt is what ends up on your boots; soil is a living system, and a single teaspoon of
healthy farmland can hold more microorganisms than there are people on Earth. Bacteria break down organic matter and
release nitrogen in forms that plants can absorb. Fungi extend thread-like hyphae for meters through the ground,
trading water and phosphorus to plant roots in exchange for sugars. Earthworms pull leaves into their burrows,
mixing the layers and leaving channels that let rain soak in instead of running off. When this system works, a field
can absorb a heavy storm the way a sponge absorbs a spill. When it breaks down, the same storm strips away the top
few centimeters, which took centuries to form. Modern agriculture has often treated soil as an inert medium for
holding up plants while fertilizer does the real work. Repeated tillage breaks up fungal networks, exposes carbon to
the air, and compacts the layer just below the plow. Yields stay high for a while because synthetic nitrogen masks
the decline, but the underlying structure erodes. Farmers who have switched to cover crops and reduced tillage
report that it can take five to ten years for the biology to recover, and that the first few seasons are the
hardest, because weeds and pests adjust before the beneficial organisms do. The economics are tricky too: the
benefits arrive slowly, while the costs of new equipment and lower early yields arrive at once."""

CODE1 = '''Here is the refactored module. I replaced the global cache with an LRU class and added type hints.

```python
from collections import OrderedDict
from typing import Callable, Generic, Hashable, TypeVar

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


class LRUCache(Generic[K, V]):
    """Least-recently-used cache with a fixed capacity."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._data: "OrderedDict[K, V]" = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: K, default: V | None = None) -> V | None:
        if key in self._data:
            self._data.move_to_end(key)
            self.hits += 1
            return self._data[key]
        self.misses += 1
        return default

    def put(self, key: K, value: V) -> None:
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = value
        while len(self._data) > self.capacity:
            self._data.popitem(last=False)

    def get_or_compute(self, key: K, fn: Callable[[K], V]) -> V:
        value = self.get(key)
        if value is None:
            value = fn(key)
            self.put(key, value)
        return value

    def __len__(self) -> int:
        return len(self._data)


def fib_cached(n: int, cache: LRUCache[int, int]) -> int:
    if n < 2:
        return n
    return cache.get_or_compute(n, lambda k: fib_cached(k - 1, cache) + fib_cached(k - 2, cache))
```

Note that `get_or_compute` treats a stored `None` as a miss; if you need to cache `None`, use a sentinel object.'''

CODE2 = '''I fixed the race in the worker pool. The bug was that `results` was appended from several goroutines without a
lock. Here is the corrected version:

```go
package pool

import (
	"context"
	"errors"
	"sync"
)

type Job func(ctx context.Context) (int, error)

type Result struct {
	Index int
	Value int
	Err   error
}

// Run executes jobs with at most n concurrent workers and returns results in job order.
func Run(ctx context.Context, n int, jobs []Job) ([]Result, error) {
	if n <= 0 {
		return nil, errors.New("pool: n must be positive")
	}
	results := make([]Result, len(jobs))
	idx := make(chan int)
	var wg sync.WaitGroup
	for w := 0; w < n; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range idx {
				v, err := jobs[i](ctx)
				results[i] = Result{Index: i, Value: v, Err: err}
			}
		}()
	}
	for i := range jobs {
		select {
		case idx <- i:
		case <-ctx.Done():
			close(idx)
			wg.Wait()
			return results, ctx.Err()
		}
	}
	close(idx)
	wg.Wait()
	return results, nil
}
```

Each worker now writes only to its own slot `results[i]`, so no mutex is needed, and cancellation closes the channel
before waiting so no worker blocks forever.'''

CHAT1 = [
    {'role': 'user', 'content': "hey, I'm planning a 4 day trip to Lisbon in March with my partner. we like food, "
     "walking, and not too many museums. any tips?"},
    {'role': 'assistant', 'content': "March is a nice time for Lisbon: mild, fewer crowds, and the jacaranda-free "
     "streets are easier to walk. A rough plan: Day 1, Alfama and Graca. Wander the alleys, take the 28 tram one way "
     "if the line is short, and catch sunset at the Miradouro da Senhora do Monte. Day 2, Belem in the morning for "
     "pasteis de nata at the original bakery, then walk back along the river toward LX Factory for lunch. Day 3, a "
     "day trip to Sintra; go early and pick one palace, Pena or Quinta da Regaleira, not both. Day 4, Principe Real "
     "and Bairro Alto, with dinner at a small tasca. For food, try bifanas, grilled sardines (more of a summer thing "
     "but some places have them), bacalhau a bras, and ginjinha. Wear shoes with grip: the stone pavements are "
     "slippery when wet."},
    {'role': 'user', 'content': "nice. is the lisboa card worth it if we skip most museums? and how do we get from "
     "the airport"},
    {'role': 'assistant', 'content': "Probably not. The card pays off if you visit several paid sites a day; for "
     "walking and food you are better off with a rechargeable Viva Viagem card loaded with zapping credit, which "
     "covers metro, trams, buses and the Sintra train. From the airport, the red metro line gets you to the center "
     "in about twenty minutes with one change, or a ride-hailing car costs roughly fifteen euros outside rush hour."},
]

CHAT2 = [
    {'role': 'system', 'content': 'You are a patient tutor for a high school student.'},
    {'role': 'user', 'content': "i don't get why the derivative of x squared is 2x. my teacher just said use the "
     "power rule but that feels like magic"},
    {'role': 'assistant', 'content': "Fair, let's make it less magic. The derivative asks: if I nudge x by a tiny "
     "amount h, how much does x squared change, relative to h? Start with (x + h) squared, which expands to x "
     "squared plus 2xh plus h squared. Subtract the original x squared and you are left with 2xh plus h squared. "
     "Divide by h to get the rate: 2x plus h. Now let h shrink toward zero. The h term vanishes and 2x is what "
     "remains. So the power rule is a shortcut for this exact calculation. You can try the same thing with x "
     "cubed: expand (x + h) cubed, subtract x cubed, divide by h, and see what survives when h goes to zero."},
    {'role': 'user', 'content': "ok i tried it and got 3x^2 + 3xh + h^2 and then 3x^2. so thats the power rule "
     "pattern right, the exponent comes down and goes one lower"},
    {'role': 'assistant', 'content': "Exactly right. The exponent comes down as a factor because when you expand "
     "(x + h) to the n, the term with one h always has coefficient n, and every other term has h squared or more, "
     "so it disappears in the limit. Want to try one where the rule seems to break, like the square root of x?"},
]

JSON1 = [
    {'role': 'user', 'content': 'Extract every order from this email as JSON with fields customer, items (name, '
     'qty, unit_price), and ship_by (ISO date). Email: "Hi, this is Maria Gomez from Bayside Cafe. Please send 12 '
     'bags of house blend at 14.50 each and 3 boxes of paper cups at 22 each, we need them by March 3rd 2027. Also '
     'my colleague Jun Park at the Elm St location wants 5 bags of decaf at 15.25 and a single grinder burr set at '
     '89.99, no rush, end of March is fine."'},
    {'role': 'assistant', 'content': '{"orders":[{"customer":"Maria Gomez (Bayside Cafe)","items":[{"name":"house '
     'blend coffee, bag","qty":12,"unit_price":14.5},{"name":"paper cups, box","qty":3,"unit_price":22.0}],'
     '"ship_by":"2027-03-03"},{"customer":"Jun Park (Bayside Cafe, Elm St)","items":[{"name":"decaf coffee, bag",'
     '"qty":5,"unit_price":15.25},{"name":"grinder burr set","qty":1,"unit_price":89.99}],"ship_by":"2027-03-31"}],'
     '"currency":"unspecified","notes":["Jun Park said no rush; end of March taken as 2027-03-31"]}'},
]

JSON2 = [
    {'role': 'user', 'content': 'Return a JSON object describing three test cases for a function '
     'parse_duration(s) that turns strings like "1h30m" into seconds. Use keys: name, input, expected, and '
     'error (null or the exception name). Include one invalid input.'},
    {'role': 'assistant', 'content': '{\n  "function": "parse_duration",\n  "cases": [\n    {\n      "name": '
     '"hours and minutes",\n      "input": "1h30m",\n      "expected": 5400,\n      "error": null\n    },\n    {\n'
     '      "name": "seconds only",\n      "input": "45s",\n      "expected": 45,\n      "error": null\n    },\n'
     '    {\n      "name": "days hours minutes seconds",\n      "input": "2d3h4m5s",\n      "expected": 183845,\n'
     '      "error": null\n    },\n    {\n      "name": "unknown unit",\n      "input": "10x",\n      "expected": '
     'null,\n      "error": "ValueError"\n    },\n    {\n      "name": "empty string",\n      "input": "",\n'
     '      "expected": null,\n      "error": "ValueError"\n    }\n  ]\n}'},
]


def prompts():
    def task(text):
        return [{'role': 'user', 'content': text.split('\n', 1)[0][:60] + '...'}, {'role': 'assistant', 'content': text}]
    return {'prose1': [{'role': 'user', 'content': 'Write the opening of a short story about a lighthouse keeper.'},
                       {'role': 'assistant', 'content': PROSE1}],
            'prose2': [{'role': 'user', 'content': 'Explain why soil health matters for farming.'},
                       {'role': 'assistant', 'content': PROSE2}],
            'code1': task(CODE1), 'code2': task(CODE2),
            'chat1': CHAT1, 'chat2': CHAT2, 'json1': JSON1, 'json2': JSON2}


def record():
    import chat
    from tokenizers import Tokenizer
    from engine import MODEL_DIR
    tok = Tokenizer.from_file(os.path.join(MODEL_DIR, 'tokenizer.json'))
    torch.set_num_threads(int(os.environ.get('INK_THREADS', 8)))
    m = M.Model(MODEL_DIR, 'cpu', max_ctx=4096)
    rec, orig = [], M.Model.moe

    def moe(self, L, u, impl=None):  # same routing as Model.moe (sec. 5), recorded, then the real MoE
        P = f'layers.{L}.mlp.'
        s = torch.sigmoid(u.float() @ self.W[P + 'gate.weight'].float().t())
        rec.append((s + self.W[P + 'gate.bias']).topk(8, -1).indices.to(torch.uint8))
        return orig(self, L, u, impl)
    M.Model.moe = moe
    data = {}
    for name, msgs in prompts().items():
        ids = chat.render(tok, msgs, effort=0.9)
        t0, rec[:] = time.time(), []
        with torch.no_grad():
            m.forward(m.embed(ids), m.new_seq())
        assert len(rec) == 48, len(rec)
        data[name] = torch.stack(rec)                                   # [48 visits, T, 8]
        print(f'{name}: {len(ids)} tokens, prefill {time.time() - t0:.0f} s', flush=True)
        torch.save(data, OUT)
    return data


def union(rows):
    """rows [48, n, 8] uint8 -> distinct experts per visit [48] (float)."""
    oh = torch.zeros(48, 128, dtype=torch.bool)
    oh.scatter_(1, rows.long().flatten(1), True)
    return oh.sum(1).double()


def analyze(data, trials=3000, seed=0):
    g = torch.Generator().manual_seed(seed)
    names = list(data)
    nv = torch.tensor([L < 20 for L in M.VISITS])
    print(f'\nprompts: {", ".join(f"{n} {data[n].shape[1]}" for n in names)} tokens; first {SKIP} not sampled; '
          f'{trials} trials each; uniform = 128 * (1 - (15/16)^n)')
    print(f'bytes per distinct expert: NVFP4 {NVFP4_B / 1e6:.2f} MB (44 visits), MXFP8 {MXFP8_B / 1e6:.2f} MB '
          f'(4 visits); one expert at every visit = {VBYTES.sum() / 1e9:.3f} GB')

    def batch(B):  # B different prompts, one random position each
        out = []
        for i in torch.randperm(len(names), generator=g)[:B].tolist():
            r = data[names[i]]
            out.append(r[:, SKIP + int(torch.randint(r.shape[1] - SKIP, (1,), generator=g))])
        return torch.stack(out, 1)

    def verify(T):  # one prompt, T consecutive tokens at a random start
        r = data[names[int(torch.randint(len(names), (1,), generator=g))]]
        a = SKIP + int(torch.randint(r.shape[1] - SKIP - T + 1, (1,), generator=g))
        return r[:, a:a + T]

    res, base = {}, None
    for kind, ns in (('batch', (1, 2, 4, 8)), ('verify', (2, 4, 8))):
        for n in ns:
            U = torch.stack([union(batch(n) if kind == 'batch' else verify(n)) for _ in range(trials)]).mean(0)
            ebytes = (U * VBYTES).sum().item()
            gb = DENSE + ebytes + RING * (n if kind == 'batch' else 1)
            base = base or gb  # batch 1 runs first
            line = (f'{kind} {n}: U {U.mean():5.1f} (NVFP4 {U[nv].mean():5.1f}, MXFP8 {U[~nv].mean():5.1f}; uniform '
                    f'{128 * (1 - (15 / 16) ** n):5.1f}) | experts {ebytes / 1e9:5.2f} GB | step {gb / 1e9:5.2f} GB')
            for k, fx in FIXED.items():
                t = fx + gb / BW
                line += f' | {k} {t * 1e3:5.1f} ms ' + (f'{n / t:4.0f} tok/s ({1 / t:3.0f} each)' if kind == 'batch'
                                                        else f'x{t / (fx + base / BW):.2f} of one token')
            print(line)
            res[(kind, n)] = (U.mean().item(), gb)
    return res


if __name__ == '__main__':
    data = torch.load(OUT) if os.environ.get('INK_STATS_ONLY') == '1' else record()
    analyze(data)
