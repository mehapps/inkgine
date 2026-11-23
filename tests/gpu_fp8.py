"""GPU window: INK_FP8_DENSE and INK_GLUE gates. One model, fp8 copies built once; m.d8 / M.GLUE toggled.
1. _scaled_mm row-wise == fp32 emulation (per GEMM shape of the model), and GEMM ms bf16 vs fp8 at T=4096.
2. End to end on gpu_check's prompts (+ oracle teacher prompts, first 1024 tokens): mean KL(ref || path), top-1, with
   ref = bf16 dense, eager glue. Noise baselines: ref rerun, embeddings +-1 ulp. Gate: each path's KL <= 2x the
   1-ulp baseline. Paths: fp8 dense, glue compile, glue triton, fp8 + glue triton, glue2 (INK_GLUE2 kernels), glue
   triton + glue2, fp8 + glue triton + glue2, + INK_GATEFUSE / INK_MOETAIL / both (on glue triton + glue2).
3. Long prompt (real text, 2 chunks of 3072: the second reaches far attention, so the GLUE2 4-bit KV write and the
   FP8-PV far kernel (prefill_attn.FP8) are exercised; the short prompts never leave the near window): KL of the
   second chunk vs ref, same 2x 1-ulp gate (+ gatefuse + moetail, INK_FARSKIP 0 / 16 / 12; exact mode should give
   KL 0: ref uses the same Triton far kernel under auto). `--short-only` skips it.
Other INK_* flags pass through (e.g. INK_MOE_FP4ACT=1). Exit 1 on a kernel mismatch or KL outside the gate.
Usage: ~/llm/engine/swap_test.sh 20 ~/llm/engine/.venv/bin/python ~/llm/engine/tests/gpu_fp8.py"""
import os
import sys

for k, v in dict(INK_MOE='triton', INK_MOE_V2='1', INK_PREFILL_ATTN='auto', INK_FP8_DENSE='1',
                 PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True').items():
    os.environ.setdefault(k, v)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import torch  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

import fp8_dense as F  # noqa: E402
import model as M  # noqa: E402
import prefill_attn as PA  # noqa: E402
from engine import MODEL_DIR  # noqa: E402
from gpu_check import prompts, ulp_nudge  # noqa: E402
from gpu_moe import cuda_ms  # noqa: E402


def main():
    assert M.FP8_DENSE, 'needs INK_FP8_DENSE=1'
    print('flags:', ' '.join(f'{k}={v}' for k, v in sorted(os.environ.items()) if k.startswith('INK_')), flush=True)
    m = M.Model(MODEL_DIR, 'cuda', max_ctx=1 << 16)
    d8, ok = m.d8, True
    print(f'allocated with fp8 copies: {torch.cuda.memory_allocated() / 2**30:.2f} GiB', flush=True)

    print('\n== 1. kernels (T=4096)', flush=True)
    x = (torch.randn(4096, 3072, device='cuda') * 0.5).bfloat16()
    for name, (q, s), xx, wb in (('proj L5', d8.proj[5][0], x[:, :2304], m.W.proj[5][0]),
                                 ('wo L0', d8.wo[0], x, m.W['layers.0.attn.wo_ud.weight'])):
        got = F.mm(xx, q, s)
        ref = F.emulate(*F.quant_x(xx), q, s)
        e = ((got.float() - ref.float()).abs().max() / ref.float().abs().max()).item()
        good = e < 1e-2 and torch.isfinite(got).all().item() and F._SCALED_MM.get(got.device, True)
        ok &= good
        tb, t8 = cuda_ms(lambda: xx @ wb.t(), 20), cuda_ms(lambda: F.mm(xx, q, s), 20)
        tq = cuda_ms(lambda: F.quant_x(xx), 20)
        fl = 2 * xx.shape[0] * xx.shape[1] * q.shape[0] / 1e12
        print(f'{name}: scaled_mm vs emulate max rel {e:.2e} {"OK" if good else "BAD"}; bf16 {tb:.3f} ms '
              f'({fl / tb * 1e3:.0f} TFLOP/s), fp8 {t8:.3f} ms incl. act quant {tq:.3f} ms', flush=True)

    print('\n== 2. end to end KL vs bf16 dense', flush=True)
    tok = Tokenizer.from_file(os.path.join(MODEL_DIR, 'tokenizer.json'))

    def lp(x0, fp8, glue='eager', g2=False, pa8=False, gf=False, mt=False, fs=None, chunk=None, rows=None):
        """log-probs of the last `rows` rows (all) of the last chunk. gf/mt/fs: INK_GATEFUSE / INK_MOETAIL / INK_FARSKIP."""
        m.d8, M.GLUE[0], M.GLUE2[0], PA.FP8 = (d8 if fp8 else None), glue, g2, pa8
        M.GATEFUSE[0], M.MOETAIL[0], PA.SKIP[0] = gf, mt, fs
        try:
            seq = m.new_seq()
            for a in range(0, x0.shape[0], chunk or x0.shape[0]):
                f = m.forward(x0[a:a + (chunk or x0.shape[0])], seq)
            return torch.log_softmax(m.logits(f[-(rows or f.shape[0]):]), -1)
        finally:
            m.d8, M.GLUE[0], M.GLUE2[0], PA.FP8 = d8, 'eager', False, False
            M.GATEFUSE[0], M.MOETAIL[0], PA.SKIP[0] = False, False, None
    runs = {'rerun': (False, False, 'eager'), 'ulp-nudged emb': (False, True, 'eager'),
            'fp8 dense': (True, False, 'eager'), 'glue compile': (False, False, 'compile'),
            'glue triton': (False, False, 'triton'), 'fp8 + glue triton': (True, False, 'triton'),
            'glue2': (False, False, 'eager', True), 'glue triton + glue2': (False, False, 'triton', True),
            'fp8 + glue triton + glue2': (True, False, 'triton', True),
            'glue triton + glue2 + gatefuse': (False, False, 'triton', True, False, True),
            'glue triton + glue2 + moetail': (False, False, 'triton', True, False, False, True),
            'glue triton + glue2 + gatefuse + moetail': (False, False, 'triton', True, False, True, True)}
    tot = {k: [0., 0., 0] for k in runs}
    for pname, ids in prompts(tok):
        x0 = m.embed(ids)
        ref = lp(x0, False)
        p, top = ref.exp(), ref.argmax(-1)
        line = []
        for k, (fp8, nudge, glue, *g2) in runs.items():
            got = lp(ulp_nudge(x0) if nudge else x0, fp8, glue, *g2)
            kl = (p * (ref - got)).sum(-1)
            ag = (top == got.argmax(-1)).float()
            tot[k][0] += kl.sum().item(); tot[k][1] += ag.sum().item(); tot[k][2] += len(ids)
            line.append(f'{k} KL {kl.mean():.2e} top1 {ag.mean():.3f}')
        print(f'{pname} ({len(ids)} tok): ' + '; '.join(line), flush=True)
    kl = {k: v[0] / v[2] for k, v in tot.items()}
    for k, v in tot.items():
        print(f'  overall {k:16s}: mean KL {kl[k]:.3e}, top-1 {v[1] / v[2]:.4f}')
    for k in runs:
        if k in ('rerun', 'ulp-nudged emb'):
            continue
        good = kl[k] <= 2 * kl['ulp-nudged emb']
        print(f'{k} KL {kl[k]:.3e} vs 1-ulp baseline {kl["ulp-nudged emb"]:.3e} (limit 2x): '
              f'{"WITHIN noise" if good else "OUTSIDE noise"}')
        ok &= good
    if '--short-only' not in sys.argv:
        print('\n== 3. long prompt: 2 chunks of 3072 (second chunk reaches far attention), KL on its last 1024 rows',
              flush=True)
        from prof_prefill import text_ids
        x0 = m.embed(text_ids(6144))
        ref = lp(x0, False, chunk=3072, rows=1024)                  # last 1024 rows (VRAM: 0.8 GB per log-prob set)
        p, top = ref.exp(), ref.argmax(-1)
        lruns = {'rerun': (False, False, 'eager'), 'ulp-nudged emb': (False, True, 'eager'),
                 'glue triton + glue2': (False, False, 'triton', True), 'fp8-PV far': (False, False, 'eager', False, True),
                 'all: fp8 + glue triton + glue2 + fp8-PV': (True, False, 'triton', True, True),
                 'glue triton + glue2 + gatefuse + moetail': (False, False, 'triton', True, False, True, True),
                 'farskip exact (0)': (False, False, 'eager', False, False, False, False, 0.),
                 'farskip 16': (False, False, 'eager', False, False, False, False, 16.),
                 'farskip 12': (False, False, 'eager', False, False, False, False, 12.)}
        res = {}
        for k, (fp8, nudge, glue, *rest) in lruns.items():
            got = lp(ulp_nudge(x0) if nudge else x0, fp8, glue, *rest, chunk=3072, rows=1024)
            res[k] = (p * (ref - got)).sum(-1).mean().item()
            print(f'  {k:40s}: mean KL {res[k]:.3e}, top-1 {(top == got.argmax(-1)).float().mean():.4f}', flush=True)
        for k in list(lruns)[2:]:
            good = res[k] <= 2 * res['ulp-nudged emb']
            print(f'long {k} KL {res[k]:.3e} vs 1-ulp {res["ulp-nudged emb"]:.3e} (limit 2x): '
                  f'{"WITHIN noise" if good else "OUTSIDE noise"}')
            ok &= good
    print('PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
