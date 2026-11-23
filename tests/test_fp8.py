"""CPU: fp8_dense (INK_FP8_DENSE) on fake weights with the real shapes: shared-input == per-group GEMMs, pad rows
don't leak, error vs bf16 is fp8-sized, quant round trip. ~10 s. (GPU: tests/prof_prefill.py --fp8 checks _scaled_mm
== emulate and the end-to-end KL.)"""
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402

import fp8_dense as F  # noqa: E402


class FakeW:
    def __init__(self, g):
        self.proj, self.t = {}, {}
        for L in range(24):
            ns = [6552, 1056 if L in (5, 11, 17, 23) else 1024, 1056 if L in (5, 11, 17, 23) else 1024]
            self.proj[L] = (torch.randn(sum(ns), 2304, generator=g).bfloat16() * 0.02, ns)
            self.t[f'layers.{L}.attn.wo_ud.weight'] = torch.randn(2304, 3072, generator=g).bfloat16() * 0.02

    def __getitem__(self, n):
        return self.t[n]


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def main():
    g = torch.Generator().manual_seed(0)
    W = FakeW(g)
    d = F.Dense8(W)
    x = torch.randn(80, 2304, generator=g).bfloat16()
    for L in (0, 5):
        cat, ns = W.proj[L]
        ref = (x @ cat.t()).split(ns, -1)
        shared = d.qkv(L, x, x, x)
        sep = d.qkv(L, x, x.clone(), x.clone())
        for a, b, r, n in zip(shared, sep, ref, ns):
            assert a.shape == (80, n) and b.shape == (80, n)
            assert torch.equal(a, b), 'shared vs separate GEMM differ'
            e = rel(a, r)
            assert e < 0.06, e
        print(f'L{L} qkv rel err vs bf16: ' + ' '.join(f'{rel(a, r):.4f}' for a, r in zip(shared, ref)))
    o = torch.randn(80, 3072, generator=g).bfloat16()
    e = rel(d.out(0, o), o @ W['layers.0.attn.wo_ud.weight'].t())
    assert e < 0.06, e
    q, s = F.quant_w(W.proj[0][0])
    assert q.dtype == torch.float8_e4m3fn and s.shape == (W.proj[0][0].shape[0],)
    assert (q.float().abs().amax(1) == 448).all(), 'per-row scale must map amax to 448'
    z = F.quant_w(torch.zeros(16, 32).bfloat16())[0]
    assert (z.float() == 0).all()
    print(f'wo rel err {e:.4f}\nFP8 ALL OK')


if __name__ == '__main__':
    main()
