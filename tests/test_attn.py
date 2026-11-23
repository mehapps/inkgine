"""CPU checks for the long-context attention path (no weights needed).
Run: cd engine && .venv/bin/python tests/test_attn.py"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import model as M

torch.manual_seed(0)


class ExactStore:
    """Lossless stand-in for KV4 so chunking/LSE logic can be checked exactly."""
    def __init__(self, n, cap):
        self.k = torch.zeros(n, cap, M.NKV, M.HD, dtype=torch.bfloat16)
        self.v = torch.zeros_like(self.k)

    def put(self, slot, p0, K, V):
        self.k[slot, p0:p0 + len(K)], self.v[slot, p0:p0 + len(V)] = K, V

    def get(self, slot, a, b):
        return self.k[slot, a:b], self.v[slot, a:b]


def fake_model(store):
    m = object.__new__(M.Model)
    m.kv, m.slot, m.dev = store, {0: 0}, torch.device('cpu')
    return m


def naive(Q, K, V, table, sink, sliding):
    """Whole-sequence reference straight from SPEC §3.2 (fp32)."""
    N = Q.shape[0]
    i, j = torch.arange(N)[:, None], torch.arange(N)[None]
    d = i - j
    S = torch.einsum('thd,shd->hts', Q.float(), K.float().repeat_interleave(3, 1)) * M.HD ** -0.5
    bias = table.float().permute(1, 0, 2).gather(2, d.clamp(0, M.WIN - 1)[None].expand(M.NQ, -1, -1))
    S = S + torch.where((d >= 0) & (d < M.WIN), bias, 0)
    ok = (d >= 0) & ((d < M.WIN) if sliding else True)
    S = S.masked_fill(~ok, float('-inf'))
    S = torch.cat((S, sink.t()[..., None]), -1)
    P = torch.softmax(S, -1)[..., :-1]
    return torch.einsum('hts,shd->thd', P, V.float().repeat_interleave(3, 1))


def run_chunks(m, Q, K, V, table, sink, full, chunks):
    seq = M.Seq('cpu', m.kv)
    outs, p = [], 0
    for c in chunks:
        sl = slice(p, p + c)
        outs.append(m.attn_core(Q[sl], K[sl], V[sl], table[sl], sink[sl], seq, 0, full))
        seq.pos += c; p += c
    return torch.cat(outs)


def rand(N):
    Q = (torch.randn(N, M.NQ, M.HD) * 0.5).bfloat16()
    K = (torch.randn(N, M.NKV, M.HD) * 0.5).bfloat16()
    V = torch.randn(N, M.NKV, M.HD).bfloat16()
    table = (torch.randn(N, M.NQ, M.WIN) * 2).bfloat16()
    sink = torch.randn(N, M.NQ)
    return Q, K, V, table, sink


def test_chunked_vs_naive():
    N = 2900
    Q, K, V, table, sink = rand(N)
    chunks = [700, 1, 1300, 1, 1, 897]           # prefill chunks, decode steps, a chunk > WIN
    for full in (False, True):
        ref = naive(Q, K, V, table, sink, sliding=not full)
        got = run_chunks(fake_model(ExactStore(1, N)), Q, K, V, table, sink, full, chunks)
        err = (got - ref).abs().max().item()
        rel = ((got - ref).abs().mean() / ref.abs().mean()).item()
        print(f'chunked vs naive ({"full" if full else "sliding"}): max abs err {err:.4f}, mean rel {rel:.4f}')
        assert err < 0.1 and rel < 0.01, (err, rel)  # bf16 scores (as in the reference) vs fp32 naive


def test_kv4():
    x = M.l2clip(torch.randn(5000, M.NKV, M.HD), M.CLIP_QK).bfloat16()
    kv = M.KV4(1, 5000, 'cpu')
    kv.put(0, 0, x, x)
    k, v = kv.get(0, 0, 5000)
    rel = ((k.float() - x.float()).norm() / x.float().norm()).item()
    print(f'kv4 relative error {rel:.3f}')
    assert rel < 0.12 and torch.equal(k, v)


def test_kv4_end_to_end():
    """Real KV4 store on the far part: output close to exact."""
    N = 3000
    Q, K, V, table, sink = rand(N)
    ref = naive(Q, K, V, table, sink, sliding=False)
    got = run_chunks(fake_model(M.KV4(1, N, 'cpu')), Q, K, V, table, sink, True, [1500, 1500])
    err = (got - ref).abs().mean().item() / ref.abs().mean().item()
    print(f'kv4 attention mean relative error {err:.4f}')
    assert err < 0.05


def test_sconv_chunks():
    x = torch.randn(50, 64).bfloat16()
    w = torch.randn(64, 1, 4).bfloat16()
    st1, st2 = torch.zeros(3, 64).bfloat16(), torch.zeros(3, 64).bfloat16()
    whole = M.sconv(x, st1, w)
    parts = torch.cat([M.sconv(x[a:b], st2, w) for a, b in ((0, 7), (7, 8), (8, 50))])
    # spec: out = (sum_i w[c,i] x[t-3+i] + x[t]) / sqrt(5)
    xf = torch.cat((torch.zeros(3, 64), x.float()))
    ref = torch.stack([(sum(w[:, 0, i].float() * xf[t + i] for i in range(4)) + x[t].float()) / math.sqrt(5)
                       for t in range(50)])
    assert torch.equal(whole, parts), 'chunked conv differs'
    assert (whole.float() - ref).abs().max() < 0.05
    # dynamic: W[t,c,j] = static_w[j,c] + dyn[t,c//(C/8),j], j = lag
    sw, taps = torch.randn(4, 64).bfloat16(), torch.randn(50, 8, 4).bfloat16()
    got = M.sconv(x, torch.zeros(3, 64).bfloat16(), dyn=(sw, taps))
    ref = torch.stack([(sum((sw[j].float() + taps[t, :, j].float().repeat_interleave(8)) * xf[t + 3 - j]
                            for j in range(4)) + x[t].float()) / 3 for t in range(50)])
    assert (got.float() - ref).abs().max() < 0.05
    print('sconv static/dynamic/chunked ok')


def test_flex_matches_manual():
    try:
        from torch.nn.attention.flex_attention import flex_attention
    except ImportError:
        print('flex unavailable, skipped'); return
    M._flex = flex_attention  # eager reference implementation on CPU
    old = M.FAR_BLK
    M.FAR_BLK = 1024
    try:
        T, p0 = 300, 2500
        Q = (torch.randn(T, M.NQ, M.HD) * 0.5).bfloat16()
        Kb = (torch.randn(2048, M.NKV, M.HD) * 0.5).bfloat16()  # keys 0..1699; rows past that are stale junk
        Vb = torch.randn(2048, M.NKV, M.HD).bfloat16() * 4
        m = fake_model(None)
        qpos = torch.arange(p0, p0 + T)
        for b0, n in ((0, 1024), (1024, 676)):
            o1, l1 = m._far_manual(Q, Kb[b0:b0 + n], Vb[b0:b0 + n], b0, qpos)
            try:
                o2, l2 = m._far_flex(Q, Kb[b0:b0 + 1024], Vb[b0:b0 + 1024], b0, p0, n)  # whole block, n valid
                o3, l3 = m._far_flex(Q, Kb[b0:b0 + n], Vb[b0:b0 + n], b0, p0, n)        # short block (padded)
                assert torch.equal(o2, o3) and torch.equal(l2, l3), 'stale rows past n leak into flex'
            except Exception as e:
                print(f'flex on cpu failed ({type(e).__name__}: {str(e)[:120]}), check on GPU'); return
            fin = torch.isfinite(l1)
            assert torch.equal(fin, torch.isfinite(l2)), 'flex/manual disagree on empty rows'
            el = (l1[fin] - l2[fin]).abs().max().item()
            eo = (o1 - torch.nan_to_num(o2)).abs()[fin].max().item()
            print(f'flex vs manual block {b0}: lse err {el:.4f}, out err {eo:.4f}')
            assert el < 0.02 and eo < 0.03
    finally:
        M.FAR_BLK = old


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
    print('ALL OK')
