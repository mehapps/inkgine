"""Why does the 4096 px gate image fail? Image-size grid for a GPU test window, plus CPU unit checks.

  python tests/image_scale.py          # GPU window: loads Engine('cuda') once, runs the grid (~5-10 min)
  python tests/image_scale.py --unit   # CPU only: preprocessing / pos-emb / attention checks vs independent code

Grid questions: does reading fail with pixel/token count (big canvas, small or normal letters still fine?) or with
letter size relative to the 40 px tiles (huge letters fail even on a small canvas, e.g. a crop of the gate image)?
Also: eager vs sdpa ViT attention on the failing image.
"""
import base64
import io
import math
import os
import random
import sys
import time

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import vision  # noqa: E402

GATE = os.path.expanduser('~/llm/gate/img16mp.jpg')
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
Q_WORD = 'What word is written in large letters in this image? Answer with the word only.'


def font(size):
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default(size=size)


# ---------------------------------------------------------------- images (deterministic)
def word_image(w, h, size, word='GATEWAY', bg='noise'):
    """Word centered; bg 'noise' = gate-style noise with a white box behind the word, 'white' = plain white.
    -> (image, ink height of the letters in px)."""
    random.seed(1)
    im = Image.effect_noise((w, h), 40).convert('RGB') if bg == 'noise' else Image.new('RGB', (w, h), 'white')
    d, f = ImageDraw.Draw(im), font(size)
    x0, y0, x1, y1 = d.textbbox((0, 0), word, font=f)
    ox, oy = (w - (x1 - x0)) // 2 - x0, (h - (y1 - y0)) // 2 - y0
    if bg == 'noise':
        m = size // 4
        d.rectangle([ox + x0 - m, oy + y0 - m, ox + x1 + m, oy + y1 + m], fill='white')
    d.text((ox, oy), word, fill='black', font=f)
    return im, y1 - y0


def document(w=4096, h=4096, size=48):
    """Many lines of normal-size text; -> (image, question, expected)."""
    rng = random.Random(0)
    im, f = Image.new('RGB', (w, h), 'white'), font(size)
    d, lines, y = ImageDraw.Draw(im), {}, 60
    i = 1
    while y + size * 1.5 < h:
        code = ''.join(rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ23456789') for _ in range(5))
        lines[i] = code
        d.text((80, y), f'Line {i:02d}: the code for locker {i:02d} is {code}. Keep this note safe.',
               fill='black', font=f)
        y += int(size * 1.45)
        i += 1
    k = 37 if 37 in lines else len(lines) // 2
    return im, f'What is the code for locker {k:02d}? Answer with the code only.', lines[k]


def resized(im, side):
    k = side / max(im.size)
    return im.resize((max(1, round(im.size[0] * k)), max(1, round(im.size[1] * k))), Image.BICUBIC)


def data_url(im, fmt):
    b = io.BytesIO()
    im.save(b, fmt, **({'quality': 90} if fmt == 'JPEG' else {}))
    return f'data:image/{fmt.lower()};base64,' + base64.b64encode(b.getvalue()).decode()


# ---------------------------------------------------------------- GPU grid
def cases():
    raw = open(GATE, 'rb').read()
    gate = Image.open(io.BytesIO(raw)).convert('RGB')
    yield 'gate 4096 (original JPEG)', 'data:image/jpeg;base64,' + base64.b64encode(raw).decode(), Q_WORD, 'GATEWAY', 'eager'
    yield 'gate 4096 sdpa attention', 'data:image/jpeg;base64,' + base64.b64encode(raw).decode(), Q_WORD, 'GATEWAY', 'sdpa'
    yield 'gate 3072 (control, old cap)', data_url(resized(gate, 3072), 'PNG'), Q_WORD, 'GATEWAY', 'eager'
    for s in (3584, 2048):
        yield f'gate {s}', data_url(resized(gate, s), 'PNG'), Q_WORD, 'GATEWAY', 'eager'
    # same letter pixels as the failing image, much smaller canvas (token count / letter size separation)
    yield 'gate crop 4096x1400 (same letters)', data_url(gate.crop((0, 1300, 4096, 2700)), 'PNG'), Q_WORD, 'GATEWAY', 'eager'
    yield 'gate crop 3600x1100 (same letters)', data_url(gate.crop((250, 1450, 3850, 2550)), 'PNG'), Q_WORD, 'GATEWAY', 'eager'
    for bg in ('noise', 'white'):
        for size in (150, 300, 620, 740):  # 620 = the gate image's font (ink ~470 px); 740 is the widest that fits
            im, ink = word_image(4096, 4096, size, bg=bg)
            yield (f'4096^2 {bg} font {size} (ink {ink}px)', data_url(im, 'JPEG' if bg == 'noise' else 'PNG'),
                   Q_WORD, 'GATEWAY', 'eager')
    im, ink = word_image(4096, 4096, 620, word='HARBOR', bg='white')
    yield f'4096^2 white HARBOR font 620 (ink {ink}px)', data_url(im, 'PNG'), Q_WORD, 'HARBOR', 'eager'
    im, ink = word_image(4096, 2048, 620, bg='noise')
    yield f'4096x2048 noise font 620 (ink {ink}px)', data_url(im, 'JPEG'), Q_WORD, 'GATEWAY', 'eager'
    im, q, exp = document()
    yield 'document 4096^2, 48px lines', data_url(im, 'PNG'), q, exp, 'eager'
    yield 'document 2048^2 (downscaled)', data_url(resized(im, 2048), 'PNG'), q, exp, 'eager'


def gpu_grid():
    from engine import Engine
    t0 = time.time()
    eng = Engine('cuda', max_ctx=int(os.environ.get('INK_MAX_CTX', 32768)))
    print(f'engine loaded in {time.time() - t0:.0f} s', flush=True)
    rows = []
    for name, url, q, exp, attn in cases():
        eng.vision.attn = attn
        msgs = [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': url}}, {'type': 'text', 'text': q}]}]
        _, pre = eng.prepare(msgs)
        _, r, c = pre[0]
        st = {}
        torch.cuda.reset_peak_memory_stats()
        out = ''.join(t for _, t in eng.generate(msgs, max_tokens=16, temperature=0, prefill_text=True, stats=st)).strip()
        ok = exp.upper() in out.upper()
        rows.append((name, ok))
        print(f'{"PASS" if ok else "FAIL"}  {name:42s} {c * 40}x{r * 40} px-padded  {r * c:>6} img tok  '
              f'vision {st["vision_s"]:.1f}s prefill {st["prefill_s"]:.1f}s  peak '
              f'{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB  expect {exp!r} got {out!r}', flush=True)
    eng.vision.attn = os.environ.get('INK_VIT_ATTN', 'sdpa')
    print(f'\n{sum(ok for _, ok in rows)}/{len(rows)} pass, {time.time() - t0:.0f} s total')


# ---------------------------------------------------------------- CPU unit checks
def naive_tiles(arr):
    """Independent: per-tile crop of a uint8 [H,W,3] array, fp64 normalization, then bf16."""
    h, w = arr.shape[:2]
    rows, cols = -(-h // 40), -(-w // 40)
    out = np.empty((rows * cols, 40, 40, 3))
    for ty in range(rows):
        for tx in range(cols):
            t = np.full((40, 40, 3), -1.0)
            blk = arr[ty * 40:ty * 40 + 40, tx * 40:tx * 40 + 40]
            t[:blk.shape[0], :blk.shape[1]] = blk
            out[ty * cols + tx] = (t / 255 - 0.5) / 0.5
    return torch.from_numpy(out).bfloat16(), rows, cols


def unit():
    rng = np.random.default_rng(0)
    for w, h in ((41, 97), (97, 41), (40, 40), (1, 1), (4096, 4096), (4095, 2050)):
        arr = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        tiles, r, c = vision.preprocess(Image.fromarray(arr))
        ref, rr, rc = naive_tiles(arr)
        assert (r, c) == (rr, rc) == (math.ceil(h / 40), math.ceil(w / 40)), (w, h, r, c)
        assert tiles.dtype == torch.bfloat16 and torch.equal(tiles, ref), (w, h)
        print(f'preprocess {w}x{h}: {r}x{c}={r * c} tiles, exact match; last value {tiles.flatten()[-1].item()}')
    # transparency rules
    rgba = rng.integers(0, 256, (37, 53, 4), dtype=np.uint8)
    out = vision.to_rgb(Image.fromarray(rgba, 'RGBA'))
    for y, x in ((0, 0), (8, 0), (36, 52), (17, 9)):
        bg = 204 if ((y // 8) + (x // 8)) % 2 == 0 else 153
        a = int(rgba[y, x, 3])
        assert list(out[y, x]) == [(int(v) * a + bg * (255 - a) + 127) // 255 for v in rgba[y, x, :3]]
    p = Image.fromarray(rng.integers(0, 256, (20, 30, 3), dtype=np.uint8)).convert('P')
    assert not p.has_transparency_data and np.array_equal(vision.to_rgb(p), np.asarray(p.convert('RGB')))
    p.info['transparency'] = 0
    assert p.has_transparency_data and np.array_equal(vision.to_rgb(p), vision.to_rgb(p.convert('RGBA')))
    g16 = Image.fromarray(np.array([[0, 1000, 30000, 65535]], dtype=np.uint16))
    assert vision.to_rgb(g16)[0, :, 0].tolist() == [0, 3, 117, 255], vision.to_rgb(g16)
    cmyk = Image.new('CMYK', (5, 5), (0, 255, 255, 0))
    assert vision.to_rgb(cmyk)[0, 0].tolist() == [255, 0, 0]
    print('to_rgb: RGBA checkerboard composite, P with/without transparency, I;16 scaling, CMYK ok')
    # opt-in downscale
    old, vision.MAX_SIDE = vision.MAX_SIDE, 3072
    assert vision.preprocess(Image.new('RGB', (4096, 2048)))[1:] == (39, 77)
    vision.MAX_SIDE = old
    assert vision.preprocess(Image.new('RGB', (4096, 2048)))[1:] == (52, 103)
    print('INK_IMG_MAX_SIDE: opt-in downscale works, default off')
    # pos-emb taps vs F.interpolate (independent bilinear)
    v = object.__new__(vision.Vision)
    v.t = {'backbone.embeddings.position_embedding': torch.randn(1024, 1024).bfloat16()}
    for hp, wp in ((2, 2), (6, 14), (64, 64), (206, 206)):
        r = torch.arange(hp).repeat_interleave(wp)
        c = torch.arange(wp).repeat(hp)
        got = v.pos_emb(r, c, hp, wp).float()
        grid = v.t['backbone.embeddings.position_embedding'].float().t().reshape(1, 1024, 32, 32)
        ref = F.interpolate(grid, size=(hp, wp), mode='bilinear', align_corners=False)[0][:, r, c].t()
        err = (got - ref).abs().max().item()
        same = (got == ref.bfloat16().float()).float().mean().item()
        assert err < 0.04, err
        print(f'pos-emb {hp}x{wp}: max |diff| vs F.interpolate {err:.4f} (bf16 rounding), {same:.4%} identical in bf16')
    # attention: eager chunked vs sdpa, and chunk-size invariance
    q, k, val = (torch.randn(16, 301, 64).bfloat16() for _ in range(3))
    v.attn, v.attn_bytes = 'eager', 1 << 30
    a1 = v.attention(q, k, val)
    v.attn_bytes = 16 * 301 * 12 * 7  # 7 queries per chunk
    a2 = v.attention(q, k, val)
    v.attn = 'sdpa'
    a3 = v.attention(q, k, val)
    assert torch.equal(a1, a2)
    print(f'attention: chunked == unchunked exactly; eager vs sdpa max |diff| {(a1.float() - a3.float()).abs().max():.4f}')
    print('unit checks: all passed')


if __name__ == '__main__':
    unit() if '--unit' in sys.argv else gpu_grid()
