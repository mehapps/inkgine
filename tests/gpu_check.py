"""GPU window only: is INK_MOE=triton as exact as the torch path? Three-way check on real weights.
1. Per layer (NVFP4 layer 0, MXFP8 layer 20), random 4096-token activations: torch path and triton path against an
   exact reference (INK_MOE=exact: same dequant and routing, fp32 GEMMs without TF32, bf16 only at the §5 rounding
   points). Also shows the torch path with PyTorch's default bf16 reduced-precision reduction on.
2. End to end on natural text (+ tests/oracle.json teacher prompts, first 1024 tokens, if present): mean KL and top-1
   vs the torch path (exact flags). Noise baselines: torch with the reduced-precision flag on, torch with the
   embeddings moved by +-1 bf16 ulp, triton vs itself (atomic-add order).
3. 4096-token chunk: time + peak memory per path (torch, torch with the old flag, triton).
Gate: triton per-layer mean rel error vs exact <= min(2.5x torch's (+1e-5), torch-default's), and triton's mean KL <= 2x the largest
noise baseline. Exits 1 on failure. Usage (Qwen down, nothing else on the GPU):
  ~/llm/engine/swap_test.sh 25 python tests/gpu_check.py"""
import json
import os
import sys
import time

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import torch  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

import chat  # noqa: E402
import model as M  # noqa: E402
from engine import MODEL_DIR  # noqa: E402

G, T = 2 ** 30, 4096
MM = torch.backends.cuda.matmul
PROMPTS = [
    'My grandmother kept bees for forty years behind her farmhouse in Vermont. Every August she would pull the frames, '
    'scrape the wax caps into a bowl, and spin the honey out in a hand-cranked extractor that squealed like a gate. '
    'She never wore gloves. When I asked her why the bees did not sting her, she said they knew her smell, and that '
    'panic is what gets you stung, not the bees. Write a short reflection on what she might have meant, and whether '
    'it applies to people as well.',
    'I am planning a week-long trip to Lisbon in early March with my partner. We like walking, seafood, small museums '
    'and live music, but we do not enjoy crowded tourist spots or nightclubs. We will stay near Alfama. Suggest a '
    'rough day-by-day plan, including one day trip outside the city, and tell us what the weather is usually like.',
    'Explain, for a smart high-school student, why the sky is blue during the day but red and orange at sunset. Then '
    'explain why clouds are white rather than blue, and what would change about the colour of the sky on a planet '
    'whose atmosphere was much thicker than ours.',
    'Here is a Python function:\n\ndef mean(xs):\n    total = 0\n    for x in xs:\n        total += x\n    return total / '
    'len(xs)\n\nWhat happens when it is called with an empty list? Rewrite it so it handles that case sensibly, and '
    'explain the trade-off between raising an exception and returning None.',
]


def timed(fn):
    fn(); torch.cuda.synchronize()                    # warm-up (Triton compile)
    t = time.time()
    y = fn(); torch.cuda.synchronize()
    return y, time.time() - t


def run(impl, fn, reduced=False):
    """fn() with INK_MOE=impl and the bf16 reduced-precision reduction flag set to `reduced`."""
    M.MOE_IMPL, MM.allow_bf16_reduced_precision_reduction = impl, reduced
    try:
        return fn()
    finally:
        M.MOE_IMPL, MM.allow_bf16_reduced_precision_reduction = 'torch', False


def ulp_nudge(x, seed=1):
    """Move each nonzero bf16 element by -1, 0 or +1 ulp (same size as one extra rounding)."""
    g = torch.Generator(device=x.device).manual_seed(seed)
    d = torch.randint(-1, 2, x.shape, generator=g, device=x.device, dtype=torch.int16)
    return (x.view(torch.int16) + torch.where(x != 0, d, 0)).view(torch.bfloat16)


def prompts(tok):
    out = [(f'natural {i}', chat.render(tok, [{'role': 'user', 'content': p}], effort=0.9)) for i, p in enumerate(PROMPTS)]
    path = os.path.join(HERE, 'oracle.json')
    if os.path.exists(path):
        for name, o in json.load(open(path))['teacher'].items():
            out.append((f'oracle {name}', o['ids'][:1024]))
    return out


def main():
    torch.manual_seed(0)
    m = M.Model(MODEL_DIR, 'cuda', max_ctx=2 * T)
    print(f'loaded: {torch.cuda.memory_allocated() / G:.2f} GiB', flush=True)
    ok = True

    print('\n== 1. per-layer MoE vs exact reference (fp32 GEMMs, bf16 at §5 points)')
    for L in (0, 20):
        u = M.rms(torch.randn(T, M.H, device='cuda')).bfloat16()
        ref = run('exact', lambda: m.moe(L, u)).float()
        res = {}
        for name, impl, red in (('torch', 'torch', False), ('torch+bf16red', 'torch', True), ('triton', 'triton', False)):
            got, t = run(impl, lambda: timed(lambda: m.moe(L, u)), red)
            d = (got.float() - ref).abs()
            res[name] = (d.mean() / ref.abs().mean()).item()
            print(f'layer {L:2d} {name:14s}: max abs {d.max():.4g}, mean rel {res[name]:.2e}, '
                  f'exact-equal {(d == 0).float().mean():.3f}, {t * 1e3:.1f} ms', flush=True)
            ok &= bool(torch.isfinite(got).all())
        # limit set from the 2026-11-21 window: triton was 1.3x/1.8x exact-torch (L0/L20) yet 6-9x better than torch's
        # default bf16-reduction path, and end-to-end KL sat inside the 1-ulp noise baseline
        good = res['triton'] <= min(2.5 * res['torch'] + 1e-5, res['torch+bf16red'])
        print(f'layer {L:2d}: triton {"no worse than" if good else "WORSE than"} torch vs exact', flush=True)
        ok &= good

    print('\n== 2. end to end: mean KL(torch-exact || path) and top-1 agreement')
    tok = Tokenizer.from_file(os.path.join(MODEL_DIR, 'tokenizer.json'))
    paths = {'torch+bf16red': ('torch', True, False), 'torch ulp-nudged emb': ('torch', False, True),
             'triton': ('triton', False, False), 'triton rerun': ('triton', False, False), 'exact MoE': ('exact', False, False)}
    tot = {k: [0., 0., 0] for k in paths}
    for pname, ids in prompts(tok):
        x0 = m.embed(ids)
        lp = lambda x, impl, red: run(impl, lambda: torch.log_softmax(m.logits(m.forward(x, m.new_seq())), -1), red)
        ref = lp(x0, 'torch', False)
        ref_top, p_ref = ref.argmax(-1), ref.exp()
        line, prev = [], None
        for k, (impl, red, nudge) in paths.items():
            got = lp(ulp_nudge(x0) if nudge else x0, impl, red)
            if k == 'triton rerun':   # vs the first triton run, not vs torch: pure run-to-run noise
                kl = (prev.exp() * (prev - got)).sum(-1)
                agree = (prev.argmax(-1) == got.argmax(-1)).float()
            else:
                kl = (p_ref * (ref - got)).sum(-1)
                agree = (ref_top == got.argmax(-1)).float()
            prev = got if k == 'triton' else prev
            tot[k][0] += kl.sum().item(); tot[k][1] += agree.sum().item(); tot[k][2] += len(ids)
            line.append(f'{k} KL {kl.mean():.2e} top1 {agree.mean():.3f}')
        print(f'{pname} ({len(ids)} tok): ' + '; '.join(line), flush=True)
        del ref, p_ref, prev
    kl = {k: v[0] / v[2] for k, v in tot.items()}
    for k, v in tot.items():
        print(f'  overall {k:22s}: mean KL {kl[k]:.3e}, top-1 {v[1] / v[2]:.4f}')
    noise = max(kl['torch+bf16red'], kl['torch ulp-nudged emb'], kl['triton rerun'])
    good = kl['triton'] <= 2 * noise
    print(f'triton KL {kl["triton"]:.3e} vs noise baseline {noise:.3e} (limit 2x): {"OK" if good else "TOO BIG"}', flush=True)
    ok &= good

    print('\n== 3. 4096-token chunk')
    x = m.embed(torch.randint(0, M.VOCAB, (T,)).tolist())
    for name, impl, red in (('torch', 'torch', False), ('torch+bf16red', 'torch', True), ('triton', 'triton', False)):
        torch.cuda.reset_peak_memory_stats()
        _, t = run(impl, lambda: timed(lambda: m.forward(x, m.new_seq())), red)
        print(f'{name:14s}: {t:.2f} s, peak {torch.cuda.max_memory_allocated() / G:.2f} GiB', flush=True)
    print('PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
