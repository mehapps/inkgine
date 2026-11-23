"""ViT attention as one Triton flash kernel (`INK_VIT_ATTN=triton`, off by default; vision.py).

Full bidirectional attention over one image (the reference has no window/mask; IMAGES.md §3), 16 heads x 64,
scale 64^-0.5. bf16 Q K^T on the tensor cores with fp32 accumulate, fp32 online softmax (exp2), bf16 P @ V with fp32
accumulate, one bf16 rounding of the output: the same rounding points as a flash sdpa kernel, so it is checked by a
noise gate (tests/test_vision.py, tests/gpu_vision.py), not for bit equality.
Inputs are the [H, S, 64] transposed views vision.py makes (strides (64, 1024, 1)): no contiguous copies; the output
is written in [S, H, 64] order, so `out.transpose(0, 1).reshape(S, 1024)` is free.
INK_VIT_TRITON_CFG="BM,BN,warps,stages" (default 128,64,8,3: no spills in the sm_120 compile; 4 warps spill
~0.5-2 KB); tests/gpu_vision.py sweeps it on the GPU.
"""
import os

import torch
import triton
import triton.language as tl
from moe_triton import INTERP, _bf16, _mma

CFG = tuple(int(v) for v in os.environ.get('INK_VIT_TRITON_CFG', '128,64,8,3').split(','))


@triton.jit
def _block(acc, m_i, l_i, q, k_ptrs, v_ptrs, n0, S, ss, offs_n, MASK: tl.constexpr, F32: tl.constexpr):
    if MASK:
        ok = (n0 + offs_n) < S
        k = tl.load(k_ptrs + n0 * ss, mask=ok[:, None], other=0.0)
        v = tl.load(v_ptrs + n0 * ss, mask=ok[:, None], other=0.0)
    else:
        k = tl.load(k_ptrs + n0 * ss)
        v = tl.load(v_ptrs + n0 * ss)
    s = _mma(q, tl.trans(k), tl.zeros([q.shape[0], k.shape[0]], tl.float32), F32) * 0.18033688011112042   # 64^-0.5 * log2(e): softmax in base 2
    if MASK:
        s = tl.where(ok[None, :], s, float('-inf'))
    m_new = tl.maximum(m_i, tl.max(s, 1))
    p = tl.exp2(s - m_new[:, None])
    alpha = tl.exp2(m_i - m_new)
    l_i = l_i * alpha + tl.sum(p, 1)
    acc = _mma(_bf16(p, F32), v, acc * alpha[:, None], F32)
    return acc, m_new, l_i


@triton.jit
def _vit_attn_kernel(q_ptr, k_ptr, v_ptr, o_ptr, S, sh, ss, osh, oss,
                     BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr, F32: tl.constexpr):
    pm, h = tl.program_id(0), tl.program_id(1)
    offs_m = pm * BM + tl.arange(0, BM)
    offs_n = tl.arange(0, BN)
    offs_d = tl.arange(0, D)
    qm = offs_m < S
    q = tl.load(q_ptr + h * sh + offs_m[:, None] * ss + offs_d[None, :], mask=qm[:, None], other=0.0)
    k_ptrs = k_ptr + h * sh + offs_n[:, None] * ss + offs_d[None, :]
    v_ptrs = v_ptr + h * sh + offs_n[:, None] * ss + offs_d[None, :]
    m_i = tl.full([BM], float('-inf'), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    full = (S // BN) * BN
    for n0 in range(0, full, BN):            # whole key blocks: no masks
        acc, m_i, l_i = _block(acc, m_i, l_i, q, k_ptrs, v_ptrs, n0, S, ss, offs_n, False, F32)
    if full < S:                             # tail (S = 4 * tiles + 1 is always odd)
        acc, m_i, l_i = _block(acc, m_i, l_i, q, k_ptrs, v_ptrs, full, S, ss, offs_n, True, F32)
    o = acc / l_i[:, None]
    tl.store(o_ptr + h * osh + offs_m[:, None] * oss + offs_d[None, :], o.to(o_ptr.dtype.element_ty), mask=qm[:, None])


def attn(q, k, v, cfg=None):
    """q, k, v [H, S, 64] (same strides, last dim contiguous) -> [H, S, 64] view of an [S, H, 64] buffer."""
    H, S, D = q.shape
    assert k.stride() == q.stride() == v.stride() and q.stride(2) == 1 and D == 64
    BM, BN, nw, ns = cfg or CFG
    if INTERP:  # the CPU interpreter mishandles bf16 memory: fp32 copies (same values), bf16 rounding emulated (F32)
        q, k, v = (x.float() for x in (q, k, v))
    out = torch.empty((S, H, D), dtype=q.dtype, device=q.device).transpose(0, 1)
    _vit_attn_kernel[(triton.cdiv(S, BM), H)](q, k, v, out, S, q.stride(0), q.stride(1), out.stride(0), out.stride(1),
                                              BM=BM, BN=BN, D=D, F32=INTERP, num_warps=nw, num_stages=ns)
    return out.bfloat16() if INTERP else out
