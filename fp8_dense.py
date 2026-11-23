"""FP8 (e4m3) dense weights: prefill GEMMs (`INK_FP8_DENSE=1`) and decode (`INK_FP8_DECODE=1`); both off by default
(numerics change). One fp8 copy serves both.

Weights: per output channel scale s_w = amax(row) / 448, q = rne(w / s_w) in e4m3 (made once at load from the bf16
checkpoint values).
- `mm` (INK_FP8_DENSE, calls with T >= INK_FP8_MIN rows: prefill chunks): activations too, per token (row) scale
  s_x = amax / 448, e4m3. y = (q_x @ q_w^T) * s_x * s_w, fp32 accumulate, bf16 out: torch._scaled_mm with row-wise
  scales on cuda, an fp32 emulation elsewhere (CPU tests).
- `mmw` (weight-only, every other call once INK_FP8_DECODE drops the bf16 copies): y = bf16((x @ q_w^T) * s_w) with
  bf16 activations, fp32 products and sums; the decode kernels (decode_kernels / decode_batch, `W8`) do the same math
  with the dequant in registers, so this is their reference.
- `logits8`: the unembed in fp8 (INK_FP8_DECODE): softcap(bf16((f @ q_u^T) * s_u)), one Triton small-GEMM kernel.
"""
import torch

import triton
import triton.language as tl
from moe_triton import INTERP, _bf16, _mma

E4M3_MAX = 448.0
_SCALED_MM = {}  # device -> bool (row-wise _scaled_mm works there)
QROWS = 16384    # rows per quantization chunk (bounds the fp32 temp: the unembed is 200k rows)


def quant_w(w):
    """[N, K] bf16 -> (q [N, K] e4m3, s [N] fp32)."""
    qs, ss = [], []
    for a in range(0, w.shape[0], QROWS):
        x = w[a:a + QROWS].float()
        s = (x.abs().amax(1) / E4M3_MAX).clamp_min(1e-12)
        qs.append((x / s[:, None]).to(torch.float8_e4m3fn))
        ss.append(s)
    return torch.cat(qs), torch.cat(ss)


def quant_x(x):
    """[T, K] bf16 -> (q e4m3, s [T, 1] fp32): one per-token scale."""
    xf = x.float()
    s = (xf.abs().amax(1, keepdim=True) / E4M3_MAX).clamp_min(1e-12)
    return (xf / s).to(torch.float8_e4m3fn), s


def emulate(qx, sx, qw, sw):
    """Same math in fp32 (exact products of e4m3 values; only the summation order differs from the tensor cores)."""
    return ((qx.float() @ qw.float().t()) * sx * sw[None]).bfloat16()


def mm(x, qw, sw):
    """bf16 [T, K] @ (fp8 weight [N, K], scales [N])^T -> bf16 [T, N] (fp8 activations; N % 16 == 0 on cuda)."""
    qx, sx = quant_x(x)
    dev = x.device
    if dev.type == 'cuda' and _SCALED_MM.get(dev, True):
        try:
            return torch._scaled_mm(qx, qw.t(), scale_a=sx, scale_b=sw[None], out_dtype=torch.bfloat16)
        except (RuntimeError, NotImplementedError) as e:  # no row-wise fp8 GEMM for this arch/build
            _SCALED_MM[dev] = False
            print(f'fp8_dense: torch._scaled_mm row-wise unavailable ({str(e).splitlines()[0]}); emulating', flush=True)
    return emulate(qx, sx, qw, sw)


def mmw(x, qw, sw):
    """Weight-only: bf16 [T, K] @ (fp8 [N, K] * s [N])^T -> bf16 [T, N]; the scale multiplies the fp32 sum."""
    if x.is_cuda:  # e4m3 -> bf16 is exact; bf16 tensor cores, fp32 out (no bf16 rounding before the scale)
        y = torch.mm(x, qw.to(torch.bfloat16).t(), out_dtype=torch.float32)
    else:
        y = x.float() @ qw.float().t()
    return (y * sw).bfloat16()


@triton.jit
def _logits8_kernel(x_ptr, nr, q_ptr, s_ptr, out_ptr, N, F32: tl.constexpr, H: tl.constexpr, CH: tl.constexpr,
                    BN: tl.constexpr, RM: tl.constexpr):
    """out[r, n] = bf16((x_r @ q_n) * s_n) as fp32, rows r < nr of x [*, H] bf16, vocab rows n < N (fp8 q [N, H])."""
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    nm = n < N
    rr = tl.arange(0, RM)
    rm = (rr < nr)[:, None]
    c = tl.arange(0, CH)
    acc = tl.zeros([RM, BN], tl.float32)
    for k0 in range(0, H, CH):
        x = tl.load(x_ptr + rr[:, None] * H + k0 + c[None, :], mask=rm, other=0.)
        w = tl.load(q_ptr + n[:, None] * H + k0 + c[None, :], mask=nm[:, None], other=0.).to(tl.bfloat16)
        acc = _mma(x, tl.trans(w), acc, F32)
    y = _bf16(acc * tl.load(s_ptr + n, mask=nm, other=0.)[None, :], F32).to(tl.float32)
    tl.store(out_ptr + rr[:, None] * N + n[None, :], y, mask=rm & nm[None, :])


LBN = 64  # vocab rows per logits8 program (GPU: tests/gpu_fp8dec.py sweeps it)
LNW = 4


def logits8(f, q, s, triton_cpu=False, cap=True):
    """Softcapped logits [T, N] fp32 from final normed f [T, H] bf16 and the fp8 unembed (q [N, H], s [N]). cuda (or
    triton_cpu: interpreter tests): the Triton kernel, 16 rows per launch; cpu: torch in vocab chunks (same math).
    cap=False: the bf16-valued pre-softcap logits (kernel checks)."""
    T, N = f.shape[0], q.shape[0]
    if f.is_cuda or triton_cpu:
        f = f.contiguous()
        out = torch.empty(T, N, dtype=torch.float32, device=f.device)
        for r0 in range(0, T, 16):
            _logits8_kernel[(triton.cdiv(N, LBN),)](f[r0:], min(16, T - r0), q, s, out[r0:], N, F32=INTERP, H=f.shape[1],
                                                    CH=256, BN=LBN, RM=16, num_warps=LNW)
    else:
        out = torch.cat([mmw(f, q[a:a + QROWS], s[a:a + QROWS]).float() for a in range(0, N, QROWS)], 1)
    return 30 * torch.tanh(out / 30) if cap else out


class Dense8:
    """FP8 copies of the dense attention weights: per layer the fused attention matrix in Weights.proj[L]'s row order
    (groups q|r|sink|gate, k[|k taps], v[|v taps]) with zero rows appended to a multiple of 16 (_scaled_mm needs
    N % 16 == 0), and wo; with `unembed` also the unembed (decode). a8_min: calls with at least that many rows use fp8
    activations (`mm`), smaller ones are weight-only (`mmw`). `dec`: the decode paths read these copies (Model sets it
    from INK_FP8_DECODE and then drops the bf16 projections + wo)."""

    def __init__(self, W, a8_min=64, unembed=None):
        self.proj, self.wo, self.a8_min, self.dec = {}, {}, a8_min, False
        for L in range(24):
            cat, ns = W.proj[L]
            self.proj[L] = quant_w(torch.cat((cat, cat.new_zeros(-cat.shape[0] % 16, cat.shape[1])))), ns
            self.wo[L] = quant_w(W[f'layers.{L}.attn.wo_ud.weight'])
        self.un = None if unembed is None else quant_w(unembed)

    def qkv(self, L, xq, xk, xv):
        """-> (yq, yk, yv) = the rows of each group times its input, like Weights.proj's split."""
        (q, s), ns = self.proj[L]
        a8 = xq.shape[0] >= self.a8_min
        f = mm if a8 else mmw
        offs = (0, ns[0], ns[0] + ns[1])
        if xk is xq and xv is xq:
            y = f(xq, q, s)
            return [y[:, a:a + n] for a, n in zip(offs, ns)]
        # fp8 activations: N % 16 rows per call (group 0 = 6552 rows also computes 8 rows of group 1, dropped)
        return [f(x, q[a:a + (-(-n // 16) * 16 if a8 else n)], s[a:a + (-(-n // 16) * 16 if a8 else n)])[:, :n]
                for x, a, n in zip((xq, xk, xv), offs, ns)]

    def out(self, L, x):
        q, s = self.wo[L]
        return (mm if x.shape[0] >= self.a8_min else mmw)(x, q, s)

    def logits(self, f):
        return logits8(f, *self.un)
