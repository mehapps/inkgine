"""CPU (Triton interpreter, ~3 min): INK_FP8_DECODE kernels on random weights with the real shapes vs a dequantized
reference: y = bf16(sum_k x_k * float(q[n, k]) * s[n]) summed in fp64. Covers decode_kernels._proj_kernel (bf16 input,
and NORM: fused input RMSNorm), _wo_kernel (+ attn conv + residual), decode_batch._bproj_kernel (3 inputs, rows
straddling group boundaries, 3 live rows of 16) and _bwo_kernel, fp8_dense._logits8_kernel (+ softcap) and its CPU
torch path, Dense8 weight-only qkv/out (mmw) and pad rows. Gate: >= 99% of outputs bit-equal to the reference in bf16
and max error <= 1 bf16 ulp of the row (only fp32 summation order differs). Also prints the fp8-vs-bf16 weight
rounding error (the real precision change, judged on the GPU by KL).
Run: cd engine && PATH=$PWD/.venv/bin:$PATH python tests/test_fp8_decode.py"""
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TRITON_INTERPRET'] = '1'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402

import decode_batch as DB  # noqa: E402
import decode_kernels as K  # noqa: E402
import fp8_dense as F8  # noqa: E402

H, g = 2304, torch.Generator().manual_seed(0)
rn = lambda *s: torch.randn(*s, generator=g)


def check(name, got, ref):
    """got: kernel output; ref: fp64 reference before the bf16 rounding."""
    gb, rb = got.float(), ref.bfloat16().float()
    eq = (gb == rb).float().mean().item()
    err = ((gb.double() - ref).abs().max() / ref.abs().max()).item()
    print(f'{name:34s} bit-equal to the dequantized reference {eq * 100:6.2f}%, max rel err {err:.1e}', flush=True)
    assert eq >= 0.99 and err < 2 ** -7, (name, eq, err)


def wq(n, k):
    w = (rn(n, k) * 0.02 * torch.exp(rn(n, 1) * 0.5)).bfloat16()   # rows of different scale
    q, s = F8.quant_w(w)
    return w, q, s, q.double() * s.double()[:, None]


def test_proj():
    ns = [6552, 1056, 1056]                                         # full visit: q|r|sink|gate, k + taps, v + taps
    N = sum(ns)
    w, q, s, deq = wq(N, H)
    qp = torch.cat((q, torch.zeros(-N % 16, H, dtype=q.dtype)))     # Dense8 layout: pad rows appended
    x = rn(3, H).bfloat16()
    rows = [(0, ns[0]), (ns[0], ns[0] + ns[1]), (ns[0] + ns[1], N)]
    for norm in (False, True):
        nws = [rn(H).abs() + 0.5 for _ in range(3)]
        nwt = torch.tensor([t.data_ptr() for t in nws])
        xin = (x.float() * 3) if norm else x
        xx = [(xin[j] * torch.rsqrt(xin[j].pow(2).mean() + 1e-6) * nws[j]).bfloat16() if norm else x[j]
              for j in range(3)]
        y = torch.zeros(8664, dtype=torch.bfloat16)
        K._proj_kernel[(N // 8,)](qp, ns[0], ns[1], xin, H, nwt, y, s, NORM=norm, F32=True, H=H, CH=K.CH, BN=8, W8=True)
        ref = torch.cat([deq[a:b] @ xx[j].double() for j, (a, b) in enumerate(rows)])
        check(f'proj W8 norm={norm}', y[:N], ref)
        refb = torch.cat([w[a:b].double() @ xx[j].double() for j, (a, b) in enumerate(rows)])
        print(f'  fp8 vs bf16 weights: rel err {((ref - refb).norm() / refb.norm()).item():.4f}')


def conv_ref(o, st, cw, h, bnd):
    """_wo_kernel's tail in fp64 on the bf16-rounded o: static conv (state rows s0, s1, s2) + residual."""
    ob = o.bfloat16().double()
    y = (st[0].double() * cw[:, 0] + st[1].double() * cw[:, 1] + st[2].double() * cw[:, 2] + ob * cw[:, 3] + ob)
    y = (y / 5 ** 0.5).float().bfloat16().double()
    return y if bnd else h.double() + y


def test_wo():
    _, q, s, deq = wq(H, 3072)
    x = rn(3072).bfloat16()
    cw = rn(H, 4).double() * 0.3
    st0, h = rn(3, H).bfloat16(), rn(H).bfloat16()
    for bnd in (False, True):
        st, r = st0.clone(), torch.zeros(H)
        K._wo_kernel[(H // 128,)](x, q, st, cw.float(), h, r, s, BND=bnd, F32=True, H=H, CH=K.CH, BN=128, W8=True)
        o = deq @ x.double()
        eq = (st[2] == o.bfloat16()).float().mean().item()          # the new conv state row = bf16(o)
        print(f'wo W8 bnd={bnd}: conv state == bf16(dequantized o) {eq * 100:.2f}%')
        assert eq >= 0.99
        ref = conv_ref(o, st0, cw, h, bnd)
        err = ((r.double() - ref).abs().max() / ref.abs().max()).item()
        print(f'  r (conv + residual) max rel err {err:.1e}')
        assert err < 2 ** -7


def test_bproj_bwo():
    ns = [6552, 1056, 1056]
    N = sum(ns)
    _, q, s, deq = wq(N, H)
    qp = torch.cat((q, torch.zeros(-N % 16, H, dtype=q.dtype)))
    RM, nr = DB.RM, 3
    X = rn(RM, 3, H).bfloat16()
    Y = torch.zeros(RM, 8664, dtype=torch.bfloat16)
    DB._bproj_kernel[(-(-N // 64),)](qp, ns[0], ns[1], N, X, H, 3 * H, Y, nr, s, F32=True, H=H, CH=K.CH, BN=64, RM=RM,
                                     W8=True)
    rows = [(0, ns[0]), (ns[0], ns[0] + ns[1]), (ns[0] + ns[1], N)]
    ref = torch.stack([torch.cat([deq[a:b] @ X[r, j].double() for j, (a, b) in enumerate(rows)]) for r in range(nr)])
    check('bproj W8 (3 rows, BN 64 straddles)', Y[:nr, :N], ref)
    assert (Y[nr:] == 0).all(), 'masked rows written'
    _, q, s, deq = wq(H, 3072)
    att = rn(RM, 3072).bfloat16()
    CC = DB.CC
    cst = rn(1, 3, CC).bfloat16()
    xh = torch.zeros(RM, CC, dtype=torch.bfloat16)
    cw = rn(H, 4) * 0.3
    hin, Rr = rn(RM, H).bfloat16(), torch.zeros(RM, H)
    meta = torch.zeros(8, RM, dtype=torch.int64)
    meta[3, :nr] = 1                     # 3 live rows, slot 0, group index 0 each: every conv reads the slot state
    DB._bwo_kernel[(H // 128,)](att, q, cst, 3 * CC, xh, cw, hin, Rr, meta, nr, s, BND=False, F32=True, H=H, CH=K.CH,
                                BN=128, RM=RM, CC=CC, W8=True)
    o = att[:nr].double() @ deq.t()
    check('bwo W8 o -> xh (3 rows)', xh[:nr, 2048:2048 + H], o)
    st = cst[0, :, 2048:2048 + H]
    ref = torch.stack([conv_ref(o[r], st, cw.double(), hin[r], False) for r in range(nr)])
    err = ((Rr[:nr].double() - ref).abs().max() / ref.abs().max()).item()
    print(f'bwo W8 r (conv + residual) max rel err {err:.1e}')
    assert err < 2 ** -7


def test_logits8():
    N = 1000                                                        # vocab rows (real: 200058)
    w, q, s, deq = wq(N, H)
    f = rn(5, H).bfloat16()
    got = F8.logits8(f, q, s, triton_cpu=True)
    lin = (f.double() @ deq.t())
    ref = 30 * torch.tanh(lin.bfloat16().float().double() / 30)     # softcap after the bf16 rounding
    err = (got.double() - ref).abs().max().item()
    print(f'logits8 kernel (5 rows) vs softcap(bf16(dequantized)) max abs err {err:.1e}')
    assert err < 1e-3
    cpu = F8.logits8(f, q, s)
    assert (cpu - got).abs().max().item() < 1e-3, 'torch path != kernel'
    f17 = rn(17, H).bfloat16()                                      # > 16 rows: two launches
    assert (F8.logits8(f17, q, s, triton_cpu=True) - F8.logits8(f17, q, s)).abs().max().item() < 1e-3
    print('logits8 torch CPU path == kernel, 17 rows (2 launches) OK')
    pre = F8.logits8(f, q, s, triton_cpu=True, cap=False)          # gpu_fp8dec's check: bf16(exact) bit-equal share
    eq = (pre == lin.bfloat16().float()).float().mean().item()
    print(f'logits8 pre-softcap vs bf16(exact): bit-equal {eq:.4f}')
    assert eq >= 0.99, eq


class FakeW:
    def __init__(self):
        self.proj, self.t = {}, {}
        for L in range(24):
            ns = [6552, 1056 if L in (5, 11, 17, 23) else 1024, 1056 if L in (5, 11, 17, 23) else 1024]
            self.proj[L] = ((rn(sum(ns), H) * 0.02).bfloat16(), ns)
            self.t[f'layers.{L}.attn.wo_ud.weight'] = (rn(H, 3072) * 0.02).bfloat16()

    def __getitem__(self, n):
        return self.t[n]


def test_dense8_weight_only():
    W = FakeW()
    d = F8.Dense8(W, a8_min=64)
    for L in (0, 5):
        (q, s), ns = d.proj[L]
        assert q.shape[0] % 16 == 0 and q.shape[0] - sum(ns) < 16 and (q[sum(ns):].float() == 0).all()
        x = rn(2, H).bfloat16()
        ys, ysep = d.qkv(L, x, x, x), d.qkv(L, x, x.clone(), x.clone())
        deq = q.double()[:sum(ns)] * s.double()[:sum(ns), None]
        ref = (x.double() @ deq.t()).split(ns, -1)
        for a, b, r in zip(ys, ysep, ref):
            assert torch.equal(a, b)
            check(f'Dense8 L{L} weight-only group', a, r)
    o = rn(2, 3072).bfloat16()
    q, s = d.wo[0]
    check('Dense8 wo weight-only', d.out(0, o), o.double() @ (q.double() * s.double()[:, None]).t())


def real_kl():
    """--real (~5 min, real weights, CPU, informational): whole forward of 2 natural chat prompts with fp8 weight-only
    dense + fp8 unembed vs bf16, all positions, next to a +-1-ulp embedding-nudge baseline (the GPU window gates it)."""
    import chat
    import model as M
    from engine import MODEL_DIR
    from tokenizers import Tokenizer
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from gpu_check import PROMPTS, ulp_nudge
    M.FP8_KEEP_BF16[0] = True
    M.FP8_DECODE = True
    m = M.Model(MODEL_DIR, 'cpu', max_ctx=4096)
    tok = Tokenizer.from_file(os.path.join(MODEL_DIR, 'tokenizer.json'))
    for p in PROMPTS[:2]:
        ids = chat.render(tok, [{'role': 'user', 'content': p}], effort=0.9)
        x0 = m.embed(ids)

        def lp(x, dec):
            m.d8.dec = dec
            return torch.log_softmax(m.logits(m.forward(x, m.new_seq())), -1)
        ref = lp(x0, False)
        pr = ref.exp()
        for name, (x, dec) in (('1-ulp nudge (bf16)', (ulp_nudge(x0), False)), ('fp8 weight-only', (x0, True))):
            got = lp(x, dec)
            k = (pr * (ref - got)).sum(-1)
            print(f'{len(ids)} tokens, {name:18s}: KL mean {k.mean():.2e} median {k.median():.2e} max {k.max():.2e}, '
                  f'top-1 {(ref.argmax(-1) == got.argmax(-1)).float().mean():.3f}', flush=True)


if __name__ == '__main__':
    if '--real' in sys.argv:
        real_kl()
        sys.exit()
    for n, fn in list(globals().items()):
        if n.startswith('test_'):
            fn()
    print('FP8 DECODE OK')
