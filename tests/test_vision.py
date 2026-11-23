"""CPU checks for the vision speed flags (vision.py, vit_attn.py). No GPU.
  python tests/test_vision.py            INK_VIT_GPU_PRE tiles == the host path (bit-exact, many modes/sizes);
                                         vit_attn kernel (Triton interpreter) vs fp64, within sdpa's own error
  python tests/test_vision.py --real     + real ViT weights on small images: GPU_PRE embeddings bit-exact, triton / fp8
                                         embedding error vs the default sdpa path next to the eager-vs-sdpa noise
  python tests/test_vision.py --compile  offline sm_120 compile of vit_attn configs (shared memory, spills); no interpreter
"""
import os
import sys

COMPILE = '--compile' in sys.argv
if COMPILE:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ.pop('TRITON_INTERPRET', None)
else:
    os.environ['TRITON_INTERPRET'] = '1'
os.environ.setdefault('TRITON_DEFAULT_FP_FUSION', '1')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

import vision  # noqa: E402

torch.set_num_threads(2)
CFGS = ((128, 64, 8, 3), (128, 64, 4, 3), (128, 64, 4, 2), (128, 128, 8, 2), (128, 128, 4, 2), (64, 64, 4, 3),
        (128, 64, 8, 2), (64, 128, 4, 2), (128, 32, 4, 3))


def images():
    rng = np.random.default_rng(0)
    every = np.tile(np.arange(256, dtype=np.uint8), 3 * 7).reshape(56, 96, 1).repeat(3, 2)  # every value, all channels
    yield 'all 256 values 96x56', Image.fromarray(every)
    for w, h in ((1, 1), (40, 40), (41, 97), (97, 41), (203, 155), (130, 2050)):  # 2050 rows: 1024-row bands + tail
        yield f'rgb {w}x{h}', Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))
    rgba = rng.integers(0, 256, (77, 61, 4), dtype=np.uint8)
    yield 'rgba (composited)', Image.fromarray(rgba, 'RGBA')
    yield 'rgba opaque', Image.fromarray(np.dstack([rgba[..., :3], np.full((77, 61), 255, np.uint8)]), 'RGBA')
    yield 'L', Image.fromarray(rng.integers(0, 256, (45, 83), dtype=np.uint8))
    p = Image.fromarray(rng.integers(0, 256, (50, 70, 3), dtype=np.uint8)).convert('P')
    p.info['transparency'] = 3
    yield 'P + transparency', p
    yield 'I;16', Image.fromarray(rng.integers(0, 65536, (42, 81), dtype=np.uint16))
    yield 'CMYK', Image.new('CMYK', (90, 50), (0, 255, 255, 0))


def stub(dev='cpu'):
    v = object.__new__(vision.Vision)
    v.dev, v.lut = torch.device(dev), vision.norm_lut().bfloat16().to(dev)
    return v


def test_gpu_pre():
    v = stub()
    for name, im in images():
        vision.GPU_PRE = False
        t0, r0, c0 = vision.preprocess(im)
        vision.GPU_PRE = True
        t1, r1, c1 = vision.preprocess(im)
        vision.GPU_PRE = False
        assert t1.dtype == torch.uint8 and (r0, c0) == (r1, c1), name
        a, b = v._patches(t0, r0, c0), v._patches(t1, r1, c1)
        assert a.dtype == b.dtype == torch.bfloat16 and torch.equal(a, b), name
        print(f'GPU_PRE {name:22s} {r0}x{c0} tiles: patches bit-identical ({a.shape[0]} x {a.shape[1]})')
    lut = vision.norm_lut()
    x = torch.arange(256, dtype=torch.uint8)[None].repeat(3, 1)
    assert torch.equal(lut.bfloat16()[x.int()], ((x.float() / 255 - 0.5) / 0.5).bfloat16())


def ref64(q, k, v):
    s = (q.double() @ k.double().transpose(1, 2)) * 0.125
    return torch.softmax(s, -1) @ v.double()


def test_triton_attn():
    import vit_attn
    g = torch.Generator().manual_seed(1)
    for S, cfg in ((1, None), (65, None), (301, None), (200, (64, 32, 4, 2)), (130, (128, 128, 8, 2))):
        base = (torch.randn(S, 3, 16, 64, generator=g) * 2).bfloat16()  # [S, qkv, H, 64]: vision.py's strided views
        q, k, v = (base[:, i].transpose(0, 1) for i in range(3))
        ref = ref64(q, k, v)
        got = vit_attn.attn(q, k, v, cfg)
        sd = F.scaled_dot_product_attention(q[None], k[None], v[None])[0]
        e_t, e_s = ((x.double() - ref).abs().max().item() for x in (got, sd))
        assert got.shape == q.shape and got.dtype == torch.bfloat16
        assert torch.equal(got.transpose(0, 1).reshape(S, 1024), got.transpose(0, 1).contiguous().view(S, 1024))
        assert e_t <= 2 * e_s + 1e-2, (S, e_t, e_s)
        print(f'vit_attn S={S:4d} cfg={cfg or vit_attn.CFG}: max |err| vs fp64 {e_t:.5f} (sdpa {e_s:.5f})')


def load_vision():
    from safetensors import safe_open
    path = os.path.join(os.environ.get('INK_MODEL', os.path.expanduser('~/llm/models/Inkling-2-Mini-Uncensored')),
                        'model-00001-of-00001.safetensors')
    with safe_open(path, 'pt', device='cpu') as f:
        return vision.Vision(f, 'cpu')


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def test_real():
    v = load_vision()
    rng = np.random.default_rng(2)
    for w, h in ((120, 80), (400, 280)):
        arr = (np.linspace(0, 255, w * h * 3).reshape(h, w, 3) + rng.integers(0, 40, (h, w, 3))).clip(0, 255)
        im = Image.fromarray(arr.astype(np.uint8))
        v.attn, v.fp8 = 'sdpa', False
        base = v(*vision.preprocess(im))
        vision.GPU_PRE = True
        pre = v(*vision.preprocess(im))
        vision.GPU_PRE = False
        assert torch.equal(base, pre), 'GPU_PRE embeddings differ'
        tiles = vision.preprocess(im)
        v.attn = 'eager'
        noise = rel(v(*tiles), base)
        v.attn = 'triton'
        e_tr = rel(v(*tiles), base)
        v.attn, v.fp8 = 'sdpa', True
        e_8 = rel(v(*tiles), base)
        v.fp8 = False
        print(f'real {w}x{h} ({tiles[1] * tiles[2]} tiles): GPU_PRE bit-identical; rel err vs sdpa: eager {noise:.2e} '
              f'(noise), triton {e_tr:.2e}, fp8 {e_8:.2e}')
        assert e_tr <= 2 * noise + 1e-3, (e_tr, noise)


def compile_sm120():
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    import vit_attn
    from compile_sm120 import ptxas_info  # tests/ is on sys.path (script dir)
    fn = vit_attn._vit_attn_kernel
    types = dict(q_ptr='*bf16', k_ptr='*bf16', v_ptr='*bf16', o_ptr='*bf16', S='i32', sh='i32', ss='i32', osh='i32',
                 oss='i32')
    bad = 0
    for BM, BN, nw, ns in CFGS:
        sig = {n: types.get(n, 'constexpr') for n in fn.arg_names}
        try:
            k = triton.compile(ASTSource(fn=fn, signature=sig, constexprs=dict(BM=BM, BN=BN, D=64, F32=False)),
                               target=GPUTarget('cuda', 120, 32), options=dict(num_warps=nw, num_stages=ns))
            md = k.metadata
            over = md.shared > 101376
            bad += over
            print(f'vit_attn BM{BM} BN{BN} w{nw} s{ns}: {"SHARED > 101376" if over else "ok"} shared {md.shared}  '
                  f'{ptxas_info(k)}')
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f'vit_attn BM{BM} BN{BN} w{nw} s{ns}: FAILED {str(e).splitlines()[0] if str(e) else type(e).__name__}')
    print('COMPILE OK' if not bad else f'{bad} FAILED')
    return bad


if __name__ == '__main__':
    if COMPILE:
        sys.exit(1 if compile_sm120() else 0)
    test_gpu_pre()
    test_triton_attn()
    if '--real' in sys.argv:
        test_real()
    print('test_vision: all passed')
