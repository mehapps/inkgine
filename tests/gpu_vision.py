"""GPU window: vision speed flags (vision.py INK_VIT_*), 1 / 4 / 16.7 MP. Run by tests/window_vision.sh.
  python tests/gpu_vision.py bench   Vision only (no text model): attention kernel sweep at the 16.7 MP length,
                                     per-component cost model (h2d, patch embed, qkv/out, attention, MLP, projector, glue)
                                     per size, then per config and size: host (decode + preprocess) and GPU vision
                                     time (synced), embedding diff vs the current default (exact flags must be
                                     bit-identical; others within 2x the eager-vs-sdpa noise, else "LM gate decides")
  python tests/gpu_vision.py read    full Engine: gate document (Locker 37 = 7Q4K-2938) per config: next-token KL vs
                                     the default (noise = eager attention), greedy read check, synced wall time of
                                     prepare + prefill (the turn-1 TTFT part images add)
Env as in the window: INK_VIT_TRITON_CFG is picked by `bench` (printed: BEST_TRITON_CFG=...).
INK_VISION_SMOKE=1 python tests/gpu_vision.py bench: CPU smoke of the bench code (tiny images, Triton interpreter).
"""
import io
import os
import statistics
import sys
import time

SMOKE = os.environ.get('INK_VISION_SMOKE') == '1'
if SMOKE:
    os.environ['TRITON_INTERPRET'] = '1'
DEV = 'cpu' if SMOKE else 'cuda'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

import vision  # noqa: E402

DOC = os.path.expanduser('~/llm/gate/img16mp_doc.jpg')
Q = 'What is the access code for Locker 37 in this document? Answer with the code only.'
ATTN_CFGS = ((128, 64, 8, 3), (128, 64, 4, 3), (128, 64, 8, 2), (128, 128, 8, 2), (128, 128, 4, 2), (64, 64, 4, 3),
             (128, 32, 4, 3))


def images():
    """(label, jpeg bytes): the gate document at 16.7 MP and downscaled to 4 / 1 MP (realistic content)."""
    raw = open(DOC, 'rb').read()
    im = Image.open(io.BytesIO(raw)).convert('RGB')
    out = []
    if SMOKE:
        return [(f'{s}px', enc(im.resize((s, s)))) for s in (80, 120)]
    for side in (1024, 2048):
        b = io.BytesIO()
        im.resize((side, side), Image.LANCZOS).save(b, 'JPEG', quality=92)
        out.append((f'{side * side / 1e6:.1f}MP', b.getvalue()))
    return out + [('16.7MP', raw)]


def enc(im):
    b = io.BytesIO()
    im.save(b, 'JPEG', quality=92)
    return b.getvalue()


def sync():
    if DEV == 'cuda':
        torch.cuda.synchronize()


def sync_time(fn, reps):
    """-> (median s, last result); synced before and after each call."""
    ts, r = [], None
    for _ in range(1 if SMOKE else reps):
        sync()
        t0 = time.perf_counter()
        r = fn()
        sync()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), r


def ev_time(fn, reps=5):
    fn()
    if DEV == 'cpu':
        return sync_time(fn, 1)[0]
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / reps / 1e3


def load_vision():
    from safetensors import safe_open
    path = os.path.join(os.environ.get('INK_MODEL', os.path.expanduser('~/llm/models/Inkling-2-Mini-Uncensored')),
                        'model-00001-of-00001.safetensors')
    with safe_open(path, 'pt', device='cpu') as f:
        return vision.Vision(f, DEV)


def attn_sweep(v, S):
    """Attention alone at length S (16 heads x 64, vision.py's strided views): ms per call and TFLOP/s per kernel."""
    import vit_attn
    base = torch.randn(S, 3, 16, 64, device=DEV, dtype=torch.bfloat16)
    q, k, val = (base[:, i].transpose(0, 1) for i in range(3))
    flop = 4 * S * S * 1024
    ref = F.scaled_dot_product_attention(q[None], k[None], val[None])[0].float()
    best = None
    for name in ('sdpa', 'flash', 'cudnn', 'efficient'):
        v.attn = name
        try:
            t = ev_time(lambda: v.attention(q, k, val))
            print(f'attn S={S} {name:10s} {t * 1e3:8.2f} ms  {flop / t / 1e12:6.1f} TFLOP/s', flush=True)
        except Exception as e:  # noqa: BLE001  (backend not available for this arch/shape)
            print(f'attn S={S} {name:10s} unavailable: {str(e).splitlines()[0][:100]}', flush=True)
    for cfg in ATTN_CFGS:
        try:
            out = vit_attn.attn(q, k, val, cfg)
            err = (out.float() - ref).abs().max().item()
            t = ev_time(lambda: vit_attn.attn(q, k, val, cfg))
            print(f'attn S={S} triton {cfg} {t * 1e3:8.2f} ms  {flop / t / 1e12:6.1f} TFLOP/s  max|diff| vs sdpa {err:.4f}',
                  flush=True)
            if best is None or t < best[0]:
                best = (t, cfg)
        except Exception as e:  # noqa: BLE001
            print(f'attn S={S} triton {cfg} FAILED: {str(e).splitlines()[0][:100]}', flush=True)
    v.attn = 'sdpa'
    return best[1] if best else vit_attn.CFG


def components(v, label, raw):
    """Cost model of the default path at one size: ms per component (one layer's op x 23), glue = the rest."""
    im = vision.load_image(raw)
    t0 = time.perf_counter()
    im.load()
    t_dec = time.perf_counter() - t0
    t0 = time.perf_counter()
    tiles, rows, cols = vision.preprocess(im)
    t_pre = time.perf_counter() - t0
    n = rows * cols
    S = 4 * n + 1
    t, p = v.t, 'backbone.layers.0.'
    dev = DEV
    c = {'host decode': t_dec, 'host preprocess': t_pre}
    c['h2d (bf16 tiles)'] = ev_time(lambda: tiles.to(dev), 3)
    x = torch.randn(4 * n, 1200, device=dev, dtype=torch.bfloat16)
    c['patch embed'] = ev_time(lambda: x @ t['backbone.embeddings.patch_embedding.weight'].t())
    y = torch.randn(S, 1024, device=dev, dtype=torch.bfloat16)
    c['qkv + out proj'] = 23 * ev_time(lambda: [F.linear(y, t[p + f'self_attn.{m}_proj.weight'], t[p + f'self_attn.{m}_proj.bias'])
                                                for m in ('q', 'k', 'v', 'out')])
    c['mlp (fc, gelu, proj)'] = 23 * ev_time(lambda: F.linear(F.gelu(F.linear(y, t[p + 'mlp.c_fc.weight'], t[p + 'mlp.c_fc.bias'])),
                                                              t[p + 'mlp.c_proj.weight'], t[p + 'mlp.c_proj.bias']))
    base = torch.randn(S, 3, 16, 64, device=dev, dtype=torch.bfloat16)
    q, k, val = (base[:, i].transpose(0, 1) for i in range(3))
    c['attention (sdpa)'] = 23 * ev_time(lambda: v.attention(q, k, val), 3)
    xp = torch.randn(n, 4096, device=dev, dtype=torch.bfloat16)
    c['projector'] = ev_time(lambda: F.gelu(xp @ t['proj_in.weight'].t()) @ t['proj_out.weight'].t())
    total, _ = sync_time(lambda: v(tiles, rows, cols), 3)
    gpu = sum(val for key, val in c.items() if not key.startswith('host'))
    c['glue (norms, rope, pos-emb, residuals; = rest)'] = total - gpu
    lin_tf = 23 * 2 * S * (4 * 1024 ** 2 + 2 * 1024 * 4096) / 1e12
    att_tf = 23 * 4 * S * S * 1024 / 1e12
    print(f'\n== cost model {label}: {rows}x{cols} tiles, ViT length {S}; FLOPs: linears {lin_tf:.2f} T, attention '
          f'{att_tf:.2f} T; GPU vision total {total * 1e3:.1f} ms (synced, incl. h2d)')
    for key, val in c.items():
        print(f'   {key:48s} {val * 1e3:9.1f} ms')
    return total


def configs(best_attn):
    """(name, attn, gpu_pre, fp8, exact): each flag alone, then stacked (fastest attention + exact host path)."""
    out = [('default (sdpa)', 'sdpa', False, False, True), ('INK_VIT_GPU_PRE=1', 'sdpa', True, False, True)]
    out += [(f'INK_VIT_ATTN={a}', a, False, False, False) for a in ('flash', 'cudnn', 'efficient', 'triton')]
    out += [('INK_VIT_FP8=1', 'sdpa', False, True, False),
            (f'stack GPU_PRE + ATTN={best_attn}', best_attn, True, False, best_attn == 'sdpa'),
            (f'stack GPU_PRE + ATTN={best_attn} + FP8', best_attn, True, True, False)]
    return out


def apply(v, attn, gpu_pre, fp8):
    v.attn, v.fp8 = attn, fp8
    vision.GPU_PRE = gpu_pre


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def bench():
    import vit_attn
    v = load_vision()
    imgs = images()
    best_cfg = attn_sweep(v, 65 if SMOKE else 4 * 103 * 103 + 1)
    vit_attn.CFG = best_cfg
    print(f'BEST_TRITON_CFG={",".join(map(str, best_cfg))}', flush=True)
    for label, raw in imgs:
        components(v, label, raw)
    # fastest attention at 16.7 MP, measured inside the ViT below; the stack uses the sweep's winner among backends
    times, embs, noise = {}, {}, {}
    for label, raw in imgs:
        apply(v, 'eager', False, False)
        tiles = vision.preprocess(vision.load_image(raw))
        eager = v(*tiles)
        apply(v, 'sdpa', False, False)
        base = v(*tiles)
        noise[label] = rel(eager, base)
        embs[label] = base
        print(f'noise {label}: eager vs sdpa rel {noise[label]:.2e}', flush=True)
        del eager
    attn_t = {}
    for a in ('sdpa', 'flash', 'cudnn', 'efficient', 'triton'):
        apply(v, a, False, False)
        try:
            tiles = vision.preprocess(vision.load_image(imgs[-1][1]))
            attn_t[a], _ = sync_time(lambda: v(*tiles), 2)
        except Exception:  # noqa: BLE001
            pass
    best_attn = min(attn_t, key=attn_t.get)
    print(f'fastest attention inside the ViT at 16.7 MP: {best_attn} '
          f'({", ".join(f"{k} {t:.3f}s" for k, t in attn_t.items())})', flush=True)
    fails = 0
    print(f'\n{"config":40s} {"size":7s} {"host s":>7s} {"gpu s":>7s} {"total s":>8s} {"vs default":>10s}  embedding gate')
    for name, attn, gpu_pre, fp8, exact in configs(best_attn):
        for label, raw in imgs:
            apply(v, attn, gpu_pre, fp8)
            try:
                def host():
                    im = vision.load_image(raw)
                    return vision.preprocess(im)
                th = statistics.median(timeit(host) for _ in range(3))
                tiles = host()
                v(*tiles)  # warm (compiles, fp8 weight quantization)
                tg, out = sync_time(lambda: v(*tiles), 3 if label == '16.7MP' else 5)
            except Exception as e:  # noqa: BLE001
                print(f'{name:40s} {label:7s} unavailable / failed: {str(e).splitlines()[0][:120]}', flush=True)
                fails += exact  # an sdpa backend missing on this arch is a finding, not a failure
                continue
            times[name, label] = th + tg
            r = rel(out, embs[label])
            if exact:
                ok = torch.equal(out, embs[label])
                gate = 'bit-identical' if ok else f'NOT IDENTICAL (rel {r:.2e}) FAIL'
            else:
                ok = r <= 2 * noise[label]
                gate = f'rel {r:.2e} vs noise {noise[label]:.2e}: ' + ('PASS' if ok else 'over 2x noise (LM KL gate decides)')
            fails += exact and not ok
            d = times.get(('default (sdpa)', label))
            print(f'{name:40s} {label:7s} {th:7.3f} {tg:7.3f} {th + tg:8.3f} '
                  f'{(d - th - tg) * 1e3 if d else 0:+8.0f}ms  {gate}', flush=True)
    apply(v, 'sdpa', False, False)
    print(f'BEST_ATTN={best_attn}')
    print('BENCH ' + ('PASS' if not fails else f'{fails} FAIL'))
    return fails


def timeit(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def read():
    import chat
    import random
    import vit_attn
    from engine import Engine
    if os.environ.get('INK_VIT_TRITON_CFG'):
        vit_attn.CFG = tuple(int(x) for x in os.environ['INK_VIT_TRITON_CFG'].split(','))
    eng = Engine('cuda', max_ctx=32768)
    m, v = eng.model, eng.vision
    raw = open(DOC, 'rb').read()
    import base64
    msgs = [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,'
                                                                                  + base64.b64encode(raw).decode()}},
                                         {'type': 'text', 'text': Q}]}]
    best = os.environ.get('BEST_ATTN', 'triton')
    cfgs = [('default (sdpa)', 'sdpa', False, False), ('eager (noise baseline)', 'eager', False, False)]
    cfgs += [(n, a, g, f) for n, a, g, f, _ in configs(best)[1:]]
    ref, fails, noise_kl = None, 0, None
    for name, attn, gpu_pre, fp8 in cfgs:
        apply(v, attn, gpu_pre, fp8)
        try:
            for warm in (True, False):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                ids, pre = eng.prepare(msgs)
                ids = ids + [chat.MODEL, chat.TEXT]  # thinking off: answer text right away
                pre = [(t, r, c, random.getrandbits(62)) for t, r, c, _ in pre]  # fresh keys: no prefix-cache hit
                seq, f = eng.prefill(ids, pre, {}, max_tokens=24)
                lp = torch.log_softmax(m.logits(f[-1:]).float(), -1)[0]
                torch.cuda.synchronize()
                wall = time.perf_counter() - t0
                if warm:
                    eng.pool.release(seq.kv)
            toks = [int(lp.argmax())]
            for _ in range(20):
                if toks[-1] in (chat.EOS, chat.END):
                    break
                f = m.forward(m.embed([toks[-1]]), seq)
                toks.append(int(m.logits(f[-1:])[0].argmax()))
            eng.pool.release(seq.kv)
        except Exception as e:  # noqa: BLE001
            print(f'{name:40s} FAILED {str(e).splitlines()[0][:150]}', flush=True)
            fails += 1
            continue
        text = eng.tok.decode([t for t in toks if t not in (chat.EOS, chat.END)])
        ok = '7Q4K-2938' in text.upper()
        if ref is None:
            ref = lp
        kl = float((ref.exp() * (ref - lp)).sum())
        top1 = int(lp.argmax()) == int(ref.argmax())
        fails += not ok
        if attn == 'eager' and not gpu_pre and not fp8:
            noise_kl = kl
        verdict = '' if noise_kl is None or name.startswith('eager') else \
            ('  within noise' if kl <= max(2 * noise_kl, 1e-3) else '  OVER 2x noise KL')
        print(f'{name:40s} prepare+prefill {wall:6.2f} s  KL vs default {kl:.2e}  top-1 {"same" if top1 else "DIFF"}  '
              f'read {"PASS" if ok else "FAIL"} {text.strip()[:30]!r}{verdict}', flush=True)
    apply(v, 'sdpa', False, False)
    print('READ ' + ('PASS' if not fails else f'{fails} FAIL'))
    return fails


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'bench'
    sys.exit(1 if {'bench': bench, 'read': read}[mode]() else 0)
