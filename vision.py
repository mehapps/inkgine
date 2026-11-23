"""Inkling 2 vision: preprocessing + ViT + projector (SPEC.md §8, IMAGES.md). Each 40x40 tile -> one LM token.

Matches the HF reference (IMAGES.md §3): no resize, pixel values bf16 before the patch matmul, pos-emb resampled
in fp32 then added in bf16, RoPE in fp32 then back to bf16, sdpa attention by default (eager = the reference's bf16 scores / fp32 softmax, same answers in the GPU grid).
Env knobs (all default to reference behaviour):
  INK_IMG_MAX_SIDE  opt-in downscale (bicubic) so the longest side is <= this many px. Default 0 = never resize.
  INK_VIT_ATTN      'sdpa' (default, fused kernel: 0.2 s vs 21 s vision at 16.7 MP, identical answers on the 2026-11-21
                    image grid) or 'eager' (reference math, chunked over queries; for reference checks).
                    Speed options (off; noise-gated by tests/gpu_vision.py): 'flash' / 'cudnn' / 'efficient' = sdpa
                    pinned to one backend; 'triton' = vit_attn.py (INK_VIT_TRITON_CFG tiles).
  INK_VIT_GPU_PRE   1 = exact host speedup (off by default): preprocess returns the raw uint8 [h,w,3] pixels (no host
                    bf16 canvas, ~2 s of CPU at 16.7 MP), Vision pads/normalizes/tiles them on the device with a
                    256-entry table made by the host formula (bit-identical tiles), 1/2 the bytes copied to the GPU.
  INK_VIT_FP8       1 = ViT + projector linears as e4m3 GEMMs (fp8_dense.mm: per-row weight + per-token activation
                    scales), calls with >= 64 rows; numerics change (noise-gated). Off by default.
  INK_VIT_ATTN_MB   memory budget for one eager query chunk (default 1024 MiB). Smaller = less memory, more steps.
"""
import io
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

TILE, PATCH, D, HEADS = 40, 20, 1024, 16
MAX_SIDE = int(os.environ.get('INK_IMG_MAX_SIDE', 0))
GPU_PRE = os.environ.get('INK_VIT_GPU_PRE', '0') == '1'
PAD = -1.0  # pad value on the 0-255 scale (SPEC §8)
SDPA = {'flash': 'FLASH_ATTENTION', 'cudnn': 'CUDNN_ATTENTION', 'efficient': 'EFFICIENT_ATTENTION'}
FP8_ROWS = 8192  # INK_VIT_FP8: rows per quantize + GEMM call (bounds the fp32 activation temp)


def norm_lut():
    """uint8 value -> normalized bf16, by the same torch ops preprocess applies to a band (GPU_PRE's table)."""
    return (torch.arange(256, dtype=torch.uint8).float() / 255 - 0.5) / 0.5


def load_image(data: bytes) -> Image.Image:
    try:
        from pillow_heif import register_heif_opener  # optional iPhone HEIC support
        register_heif_opener()
    except ImportError:
        pass
    return Image.open(io.BytesIO(data))


def to_rgb(im: Image.Image) -> np.ndarray:
    """-> uint8 [H,W,3]. Like the reference: RGBA path only if the image has transparency, else convert('RGB')."""
    if im.mode.startswith('I;16'):
        # Not in the reference: PIL's convert would clip 16-bit gray to 0/255 (all white); scale to 8 bit instead.
        im = Image.fromarray((np.asarray(im).astype(np.uint16) >> 8).astype(np.uint8))
    if not im.has_transparency_data:
        return np.asarray(im if GPU_PRE and im.mode == 'RGB' else im.convert('RGB'))  # convert('RGB') of RGB = a copy
    a = np.asarray(im.convert('RGBA')).astype(np.uint16)  # c*a + bg*(255-a) + 127 <= 65152 fits uint16
    h, w = a.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    bg = np.where(((yy // 8) + (xx // 8)) % 2 == 0, 204, 153).astype(np.uint16)[..., None]
    al = a[..., 3:4]
    return ((a[..., :3] * al + bg * (255 - al) + 127) // 255).astype(np.uint8)


def preprocess(im: Image.Image):
    """-> (tiles [n,40,40,3] bf16 normalized to [-1,1], rows, cols). Padding bottom/right with -1 (0-255 scale)."""
    a = to_rgb(im)
    if MAX_SIDE and max(a.shape[:2]) > MAX_SIDE:  # opt-in only; the reference never resizes
        k = MAX_SIDE / max(a.shape[:2])
        size = (max(1, round(a.shape[1] * k)), max(1, round(a.shape[0] * k)))
        a = np.asarray(Image.fromarray(a).resize(size, Image.BICUBIC))
    h, w = a.shape[:2]
    rows, cols = math.ceil(h / TILE), math.ceil(w / TILE)
    if GPU_PRE:  # raw pixels; Vision._patches pads/normalizes/tiles on the device
        # shortcut: pageable memory (pinning in a handler thread could hit a CUDA graph capture); ~5 ms at 16.7 MP
        return torch.from_numpy(np.require(a, requirements=['C', 'W'])), rows, cols
    x = torch.full((rows * TILE, cols * TILE, 3), (PAD / 255 - 0.5) / 0.5, dtype=torch.bfloat16)
    for y in range(0, h, 1024):  # row bands: fp32 temporaries stay small for huge images
        band = torch.tensor(a[y:y + 1024])
        x[y:y + band.shape[0], :w] = (band.float() / 255 - 0.5) / 0.5
    tiles = x.view(rows, TILE, cols, TILE, 3).permute(0, 2, 1, 3, 4).reshape(rows * cols, TILE, TILE, 3)
    return tiles, rows, cols


def pos_taps(idx, size):
    """Bilinear (align_corners=False, border clamp) taps over the 32-wide table -> [(i0,w0),(i1,w1)], fp32 weights."""
    src = (idx.float() + 0.5) * 32 / size - 0.5
    f = src.floor()
    return [((f + o).long().clamp(0, 31), (1 - (src - (f + o)).abs()).clamp(min=0)) for o in (0, 1)]


class Vision:
    def __init__(self, f, dev):
        """f: an open safetensors handle of the main checkpoint."""
        self.dev = dev
        self.t = {k[len('model.visual.'):]: f.get_tensor(k).to(dev) for k in f.keys() if k.startswith('model.visual.')}
        inv = 10000 ** (-torch.arange(0, 32, 2, dtype=torch.float32) / 32)          # 16 freqs
        self.inv = inv.to(dev)
        self.attn = os.environ.get('INK_VIT_ATTN', 'sdpa')
        self.attn_bytes = int(os.environ.get('INK_VIT_ATTN_MB', 1024)) << 20
        self.fp8 = os.environ.get('INK_VIT_FP8', '0') == '1'
        self.q8 = {}  # INK_VIT_FP8: weight name -> (e4m3, row scales), made on first use
        self.lut = norm_lut().bfloat16().to(dev)

    def lin(self, x, p):
        """x @ W(p).T + bias(p): bf16 F.linear, or e4m3 GEMMs in FP8_ROWS row blocks (INK_VIT_FP8, >= 64 rows)."""
        w, b = self.t[p + '.weight'], self.t.get(p + '.bias')
        if not self.fp8 or x.shape[0] < 64:
            return F.linear(x, w, b)
        import fp8_dense
        if p not in self.q8:
            self.q8[p] = fp8_dense.quant_w(w)
        qw, sw = self.q8[p]
        y = x.new_empty((x.shape[0], w.shape[0]))
        for a in range(0, x.shape[0], FP8_ROWS):
            y[a:a + FP8_ROWS] = fp8_dense.mm(x[a:a + FP8_ROWS], qw, sw)
        return y if b is None else y.add_(b)

    def _patches(self, tiles, rows, cols):
        """-> [4n, 1200] bf16 patch vectors on the device, in sequence order (tile raster, 2x2 inside, (y, x, c))."""
        n = rows * cols
        if tiles.dtype != torch.uint8:  # preprocess's tiles [n,40,40,3] bf16
            x = tiles.to(self.dev).bfloat16().view(n, 2, PATCH, 2, PATCH, 3).permute(0, 1, 3, 2, 4, 5)
            return x.reshape(n * 4, PATCH * PATCH * 3)
        u = tiles.to(self.dev)  # GPU_PRE: raw pixels [h,w,3]: pad with PAD, normalize by the table, tile
        h, w = u.shape[:2]
        c = torch.full((rows * TILE, cols * TILE, 3), (PAD / 255 - 0.5) / 0.5, dtype=torch.bfloat16, device=self.dev)
        for y in range(0, h, 1024):  # bands bound the int32 index temp
            band = u[y:y + 1024]
            c[y:y + band.shape[0], :w] = self.lut[band.int()]
        return c.view(rows, 2, PATCH, cols, 2, PATCH, 3).permute(0, 3, 1, 4, 2, 5, 6).reshape(n * 4, PATCH * PATCH * 3)

    def rope(self, r, c):
        ang = torch.cat((c[:, None] * self.inv, r[:, None] * self.inv), -1)        # [S,32]
        ang = ang.repeat_interleave(2, -1)                                          # [S,64]
        return ang.cos(), ang.sin()

    def pos_emb(self, r, c, hp, wp):
        """Per-patch position embedding [P,1024]: 4-tap fp32 sum of bf16 table rows, cast to bf16."""
        tab = self.t['backbone.embeddings.position_embedding']                     # [32*32,1024] bf16
        acc = None
        for ir, wr in pos_taps(r, hp):
            for ic, wc in pos_taps(c, wp):
                term = tab[ir * 32 + ic] * (wr * wc)[:, None]                       # bf16 * fp32 -> fp32
                acc = term if acc is None else acc + term
        return acc.bfloat16()

    def attention(self, q, k, v):
        """[H,S,64] bf16 -> [H,S,64]. Eager: bf16 scores, fp32 softmax, bf16 probs @ v, chunked over queries so
        the S x S matrix is never whole (S = 42k at 4096 px)."""
        if self.attn == 'sdpa':
            return F.scaled_dot_product_attention(q[None], k[None], v[None])[0]
        if self.attn in SDPA:
            from torch.nn.attention import SDPBackend, sdpa_kernel
            with sdpa_kernel(getattr(SDPBackend, SDPA[self.attn])):
                return F.scaled_dot_product_attention(q[None], k[None], v[None])[0]
        if self.attn == 'triton':
            import vit_attn
            return vit_attn.attn(q, k, v)
        S = q.shape[1]
        qc = max(1, self.attn_bytes // (HEADS * S * 12))  # ~12 B per score: bf16 s, fp32 softmax, bf16 p, slack
        out = torch.empty_like(q)
        kt = k.transpose(1, 2)
        for a in range(0, S, qc):
            s = (q[:, a:a + qc] @ kt) * 0.125                                       # 64^-0.5 (exact in bf16)
            out[:, a:a + qc] = torch.softmax(s, -1, dtype=torch.float32).to(q.dtype) @ v
        return out

    @torch.no_grad()
    def __call__(self, tiles, rows, cols):
        """tiles [n,40,40,3] -> [n,2304] bf16."""
        t, dev = self.t, self.dev
        n = rows * cols
        x = self._patches(tiles, rows, cols)
        e = x @ t['backbone.embeddings.patch_embedding.weight'].t()                    # [4n,1024] bf16
        del x
        tr = torch.arange(n, device=dev) // cols
        tc = torch.arange(n, device=dev) % cols
        py = torch.tensor([0, 0, 1, 1], device=dev)
        px = torch.tensor([0, 1, 0, 1], device=dev)
        r = (2 * tr[:, None] + py).flatten()
        c = (2 * tc[:, None] + px).flatten()
        e = e + self.pos_emb(r, c, 2 * rows, 2 * cols)                                  # bf16 add
        cls = (t['backbone.embeddings.class_embedding'] + t['backbone.embeddings.cls_position_embedding']).view(1, D)
        h = torch.cat((cls.to(e.dtype), e))                                            # [S,1024]
        del e
        h = F.layer_norm(h, (D,), t['backbone.ln_pre.weight'], t['backbone.ln_pre.bias'], 1e-5)
        cos, sin = self.rope(torch.cat((r.new_zeros(1), r + 1)).float(), torch.cat((c.new_zeros(1), c + 1)).float())
        S = h.shape[0]
        for i in range(23):
            p = f'backbone.layers.{i}.'
            y = F.layer_norm(h, (D,), t[p + 'ln_1.weight'], t[p + 'ln_1.bias'], 1e-5)
            q, k, v = (self.lin(y, p + f'self_attn.{n_}_proj').view(S, HEADS, 64) for n_ in 'qkv')
            q, k = (self._rot(z, cos, sin) for z in (q, k))
            a = self.attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1))
            del q, k, v
            a = a.transpose(0, 1).reshape(S, D)
            h = h + t[p + 'ls_1.gamma'] * self.lin(a, p + 'self_attn.out_proj')
            y = F.layer_norm(h, (D,), t[p + 'ln_2.weight'], t[p + 'ln_2.bias'], 1e-5)
            y = F.gelu(self.lin(y, p + 'mlp.c_fc'))
            h = h + t[p + 'ls_2.gamma'] * self.lin(y, p + 'mlp.c_proj')
        x = h[1:].reshape(n, 4 * D)
        if not self.fp8:
            return F.gelu(x @ t['proj_in.weight'].t()) @ t['proj_out.weight'].t()
        return self.lin(F.gelu(self.lin(x, 'proj_in')), 'proj_out')

    @staticmethod
    def _rot(z, cos, sin):
        zf = z.float()
        rot = torch.stack((-zf[..., 1::2], zf[..., 0::2]), -1).flatten(-2)
        return (zf * cos[:, None] + rot * sin[:, None]).bfloat16()
