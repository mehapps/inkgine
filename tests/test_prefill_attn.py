"""CPU checks for prefill_attn (Triton CPU interpreter) against the torch path and an fp32 reference.
Run: cd engine && PATH=$PWD/.venv/bin:$PATH .venv/bin/python tests/test_prefill_attn.py   (~a few minutes)
Pass: triton's error vs fp32 is within the torch path's own (bf16) error, and triton vs torch is within bf16 noise."""
import os
import sys

os.environ['TRITON_INTERPRET'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402

import model as M  # noqa: E402
import prefill_attn as PA  # noqa: E402
from kvpool import KVPool  # noqa: E402

torch.manual_seed(0)


def fake_model():
    m = object.__new__(M.Model)
    m.slot, m.dev = {0: 0, 1: 1}, torch.device('cpu')
    return m


def rel(a, b):
    """(max abs err / max |b|, mean abs err / mean |b|)"""
    d = (a - b).abs()
    return (d.max() / b.abs().max()).item(), (d.mean() / b.abs().mean()).item()


def kv_rand(n):
    return M.l2clip(torch.randn(n, M.NKV, M.HD), M.CLIP_QK).bfloat16(), M.rms(torch.randn(n, M.NKV, M.HD)).bfloat16()


def q_rand(T):
    return M.l2clip(torch.randn(T, M.NQ, M.HD) * 3, M.CLIP_QK).bfloat16()


def far_ref(Q, kv, slot, p0, kend):
    """fp32 attention over the dequantized far keys (exact K/V values of the store)."""
    T = Q.shape[0]
    K, V = (x.float() for x in kv.get(slot, 0, kend))
    S = torch.einsum('tkgd,skd->tkgs', Q.float().view(T, M.NKV, 3, M.HD), K) * M.HD ** -0.5
    ok = torch.arange(kend)[None] <= torch.arange(p0, p0 + T)[:, None] - M.WIN
    S = S.masked_fill(~ok[:, None, None], float('-inf'))
    lse = torch.logsumexp(S, -1)
    o = torch.einsum('tkgs,skd->tkgd', torch.nan_to_num(torch.exp(S - lse[..., None])), V)
    return o.reshape(T, M.NQ, M.HD), lse.reshape(T, M.NQ)


def store(kind, n):
    if kind == 'kv4':
        return M.KV4(2, n, 'cpu')
    pool = KVPool(n + 3 * 256, 'cpu', 'test', cache_dir=None)
    tbl = pool.new_table()
    pool.reserve(tbl, n)
    perm = torch.randperm(len(tbl.pages)).tolist()                     # non-contiguous page table
    tbl.pages = [tbl.pages[i] for i in perm]
    tbl.t[:len(perm)] = torch.tensor(tbl.pages)
    return tbl


def check_far(name, kind, p0, T, split=None):
    n = p0 + T
    kv = store(kind, n)
    K, V = kv_rand(n)
    kv.put(1, 0, K, V)
    Q = q_rand(T)
    kend = p0 + T - M.WIN
    M.PREFILL_ATTN = 'torch'
    o1, l1 = fake_model().far(Q, kv, 1, torch.arange(p0, p0 + T), kend, p0)
    o2, l2 = PA.far(Q, kv, 1, p0, kend, split)
    o0, l0 = far_ref(Q, kv, 1, p0, kend)
    fin = torch.isfinite(l0)
    assert torch.equal(fin, torch.isfinite(l2)) and torch.equal(fin, torch.isfinite(l1)), 'empty rows differ'
    assert (o2[~fin] == 0).all()
    e_t, e_x, e_tx = rel(o1[fin], o0[fin]), rel(o2[fin], o0[fin]), rel(o2[fin], o1[fin])
    le = (l2[fin] - l0[fin]).abs().max().item()
    print(f'far {name:34s} torch-vs-fp32 max {e_t[0]:.2e} mean {e_t[1]:.2e} | triton-vs-fp32 max {e_x[0]:.2e} '
          f'mean {e_x[1]:.2e} | triton-vs-torch max {e_tx[0]:.2e} mean {e_tx[1]:.2e} | lse err {le:.1e}')
    assert e_x[1] <= 1.5 * e_t[1] + 1e-4 and e_x[0] <= 1.5 * e_t[0] + 1e-3 and le < 1e-3, (e_t, e_x, le)


def naive_mixed(Q, K, V, Kq, Vq, table, sink, sliding):
    """fp32 reference, SPEC §3.2: exact K/V at distance < 1024 (bias), 4-bit (Kq/Vq) beyond, sink in the denominator."""
    N = Q.shape[0]
    d = torch.arange(N)[:, None] - torch.arange(N)[None]
    near = (d >= 0) & (d < M.WIN)
    rep = lambda x: x.float().repeat_interleave(3, 1)
    S = torch.where(near, torch.einsum('thd,shd->hts', Q.float(), rep(K)),
                    torch.einsum('thd,shd->hts', Q.float(), rep(Kq))) * M.HD ** -0.5
    S = S + torch.where(near, table.float().permute(1, 0, 2).gather(2, d.clamp(0, M.WIN - 1)[None].expand(M.NQ, -1, -1)), 0)
    S = S.masked_fill(~(near if sliding else d >= 0), float('-inf'))
    P = torch.softmax(torch.cat((S, sink.t()[..., None]), -1), -1)[..., :-1]
    return torch.einsum('hts,shd->thd', P * near, rep(V)) + torch.einsum('hts,shd->thd', P * ~near, rep(Vq))


def run_chunks(impl, kind, Q, K, V, table, sink, full, chunks):
    M.PREFILL_ATTN = impl
    m = fake_model()
    kv = store(kind, Q.shape[0])
    seq, outs, p = M.Seq('cpu', kv), [], 0
    for c in chunks:
        sl = slice(p, p + c)
        tb = (table[0][sl], table[1]) if isinstance(table, tuple) else table[sl]
        outs.append(m.attn_core(Q[sl], K[sl], V[sl], tb, sink[sl], seq, 0, full))
        seq.pos += c; p += c
    M.PREFILL_ATTN = 'torch'
    return torch.cat(outs)


def check_near(chunks, tuple_table, kind, auto=False):
    N = sum(chunks)
    Q, (K, V) = q_rand(N), kv_rand(N)
    sink = torch.randn(N, M.NQ) * 3
    R = M.l2clip(torch.randn(N, M.NQ, 16), M.CLIP_REL).bfloat16()
    proj = (torch.randn(16, M.WIN) * 0.5).bfloat16()
    table_t = R @ proj                                                  # bf16, as attn_core builds it
    table = (R, proj) if tuple_table else table_t
    kv4 = M.KV4(1, N, 'cpu'); kv4.put(0, 0, K, V)
    Kq, Vq = kv4.get(0, 0, N)
    for full in (False, True):
        ref = naive_mixed(Q, K, V, Kq, Vq, table_t, sink, not full)
        a = run_chunks('torch', kind, Q, K, V, table, sink, full, chunks)
        b = run_chunks('triton', kind, Q, K, V, table, sink, full, chunks)
        e_t, e_x, e_tx = rel(a, ref), rel(b, ref), rel(b, a)
        print(f'near+far {"full" if full else "sliding"} chunks {chunks} {"R x proj" if tuple_table else "table"} '
              f'{kind}: torch-vs-fp32 max {e_t[0]:.2e} mean {e_t[1]:.2e} | triton-vs-fp32 max {e_x[0]:.2e} '
              f'mean {e_x[1]:.2e} | triton-vs-torch max {e_tx[0]:.2e} mean {e_tx[1]:.2e}')
        assert e_x[1] <= 1.5 * e_t[1] + 1e-4 and e_x[0] <= 1.5 * e_t[0] + 1e-3, (e_t, e_x)
        if auto and full:  # INK_PREFILL_ATTN=auto: triton far for the chunks with < PA_FAR_MAX far keys, torch after
            old, M.PA_FAR_MAX = M.PA_FAR_MAX, 500
            c = run_chunks('auto', kind, Q, K, V, table, sink, full, chunks)
            M.PA_FAR_MAX = old
            e_a = rel(c, ref)
            print(f'  auto (far max 500): vs fp32 max {e_a[0]:.2e} mean {e_a[1]:.2e}')
            assert e_a[1] <= 1.5 * e_t[1] + 1e-4 and e_a[0] <= 1.5 * e_t[0] + 1e-3, (e_t, e_a)


def test_far():
    check_far('chunk crossing 1024 (kv4)', 'kv4', 900, 300)
    check_far('pool, partial last page', 'pool', 3000, 200)
    check_far('pool, long, many pages', 'pool', 20000, 100)
    check_far('pool, long, forced split 1000', 'pool', 9000, 64, split=1000)
    check_far('pool, one query (decode shape)', 'pool', 7777, 1)
    check_far('pool, full far tiles + diagonal', 'pool', 4096, 700)


def test_near():
    check_near([300, 1100, 1, 600], True, 'pool', auto=True)  # inside the first 1024, crossing it, a decode step, later chunk
    check_near([1500, 37], False, 'kv4')             # one chunk > WIN from 0, precomputed table


def test_farskip():
    """INK_FARSKIP: exact mode (0) bit-identical to no skipping (and actually skips); delta > 0 within its provable
    bound (dropped mass < keys * e^-delta of the far sum) vs no skipping; tile_meta == torch min/max."""
    p0, T = 3000, 48
    kend = p0 + T - M.WIN
    u = torch.randn(M.HD)
    for kind, qs, delta, split in (('pool', 2.5, 0, None), ('pool', 1.0, 8, None), ('kv4', 1.0, 12, 1024), ('pool', 0.15, 8, None)):
        kv = store(kind, p0 + T)
        K = torch.randn(p0 + T, M.NKV, M.HD) * 0.3 - u                  # cold keys: anti-aligned with the queries
        K[:PA.BN] += 2 * u                                              # first tile hot: running max set early
        K = M.l2clip(3 * K, M.CLIP_QK).bfloat16()
        V = M.rms(torch.randn(p0 + T, M.NKV, M.HD)).bfloat16()
        kv.put(1, 0, K, V)
        Q = (M.l2clip(3 * (u + 0.3 * torch.randn(T, M.NQ, M.HD)), M.CLIP_QK) * qs).bfloat16()  # qs > 1: like tau > 1
        c, s_, tbl = PA.store_of(kv, 1)
        meta = PA.tile_meta(c[0], s_[0], tbl, kend)
        Kd = kv.get(1, 0, -(-kend // PA.BN) * PA.BN)[0].float().view(-1, PA.BN, M.NKV, M.HD)
        assert torch.equal(meta[:, :, 0].float(), Kd.amax(1)) and torch.equal(meta[:, :, 1].float(), Kd.amin(1)), 'meta'
        o0, l0 = PA.far(Q, kv, 1, p0, kend, split, skip=-1)
        PA.SKIP_STATS[0] = st = torch.zeros(2, dtype=torch.int64)
        try:
            o1, l1 = PA.far(Q, kv, 1, p0, kend, split, skip=delta)
        finally:
            PA.SKIP_STATS[0] = None
        fr = st[0].item() / st[1].item()
        assert st[0] > 0, 'nothing skipped'
        fin = torch.isfinite(l0)
        assert torch.equal(fin, torch.isfinite(l1))
        if delta == 0:
            assert torch.equal(o0, o1) and torch.equal(l0, l1), 'exact mode not bit-identical'
            print(f'farskip exact ({kind}): bit-identical, skipped {fr:.1%} of (program, tile) work')
            continue
        eps = kend * float(torch.exp(torch.tensor(-delta)))             # bound on the dropped fraction of a row's mass
        dl = (l1[fin] - l0[fin]).abs().max().item()
        do = (o1[fin] - o0[fin]).abs().max().item()
        assert dl <= -torch.log1p(torch.tensor(-eps)).item() + 1e-5 and do <= 2 * eps * V.float().abs().max() + 1e-5, \
            (dl, do, eps)
        assert qs > 0.5 or dl > 0, 'mild case: skipped tiles should carry visible mass'
        print(f'farskip delta {delta} ({kind}, split {split}, q x{qs}): skipped {fr:.1%}; lse diff {dl:.1e}, o diff {do:.1e} '
              f'(bound {eps:.1e})')


if __name__ == '__main__':
    if '--farskip' in sys.argv:
        test_farskip()
        sys.exit(print('FARSKIP OK'))
    test_far()
    test_near()
    test_farskip()
    print('ALL OK')
