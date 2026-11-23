"""CPU, real weights: count the aten ops of ONE decode step (Decoder._run, T=1, far kernel path) per model component.
Each op is either a view (output aliases an input, no GPU kernel) or a launch (compute, copy, in-place: one kernel on
cuda, roughly). Triton kernels (far, MoE, fused decode kernels) are counted as one launch each by name.
Run: cd engine && CUDA_VISIBLE_DEVICES= TRITON_INTERPRET=1 .venv/bin/python tests/decode_ops.py   (~1-2 min)
INK_DEC_FUSE=1 counts the fused path instead."""
import collections
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('TRITON_INTERPRET', '1')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402
from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402
from torch.utils._pytree import tree_flatten  # noqa: E402

import decode as D  # noqa: E402
import model as M  # noqa: E402

stack = ['step glue']
GLUE = ('step glue', 'layer glue', 'attn glue')


def label(name, only_from=None):
    def wrap(fn):
        def f(*a, **k):
            push = only_from is None or stack[-1] in only_from
            if push:
                stack.append(name)
            try:
                return fn(*a, **k)
            finally:
                if push:
                    stack.pop()
        return f
    return wrap


class Count(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.c = collections.Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        ins = {a.untyped_storage().data_ptr() for a in tree_flatten((args, kwargs))[0] if isinstance(a, torch.Tensor)}
        outs = [o for o in tree_flatten(out)[0] if isinstance(o, torch.Tensor)]
        name = func.__name__
        view = (not name.split('.')[0].endswith('_') and outs and
                all(o.untyped_storage().data_ptr() in ins for o in outs))
        comp = stack[-1]
        if comp == 'attn glue' and name.split('.')[0] in ('mm', 'addmm'):
            comp = 'projections (q|r|sink|gate,k,v, wo)'
        self.c[(comp, "view" if view else "launch")] += 1
        return out


def main():
    from engine import MODEL_DIR
    m = M.Model(MODEL_DIR, 'cpu', max_ctx=4096)
    # labels (innermost wins; norms only when called from glue)
    M.attn_res = label('attn-res mixes')(M.attn_res)
    M.rms = label('norms', GLUE)(M.rms)
    M.sconv = label('short convs')(M.sconv)
    M.l2clip = label('q/k/v prologue (clip, tau, scale)')(M.l2clip)
    for cls, meth, name in ((M.Model, 'layer', 'layer glue'), (M.Model, 'gate', 'loop gates (MG)'),
                            (M.Model, 'attention', 'attn glue'), (D.DecodeState, 'attend', 'attention near'),
                            (M.Model, 'moe', 'moe routing'),
                            (M.Model, 'moe_grouped', 'moe experts'), (M.Model, 'logits', 'logits')):
        setattr(cls, meth, label(name)(getattr(cls, meth)))
    M.Model.near_merge = staticmethod(label('attention near')(M.Model.near_merge))
    D.far_attn = label('attention far (kernel + merge)')(D.far_attn)
    kern = collections.Counter()
    real_grouped = __import__('moe_triton').grouped

    def fake_grouped(u, *a, **k):  # kernels can't run on CPU at real size in reasonable time: count 2 launches
        kern[('moe experts', 'launch')] += 2
        return u.clone()
    __import__('moe_triton').grouped = fake_grouped
    far_k = D._far_kernel

    class FK:
        def __getitem__(self, grid):
            def run(*a, **k):
                kern[(stack[-1], 'launch')] += 1
            return run
    D._far_kernel = FK()
    fuse = os.environ.get('INK_DEC_FUSE') == '1'
    if fuse:  # every fused kernel: one launch, labelled by kernel name
        import decode_kernels as K

        class KK:
            def __init__(self, n):
                self.n = 'fused: ' + n.strip('_').replace('_kernel', '')

            def __getitem__(self, grid):
                return lambda *a, **k: kern.__setitem__((self.n, 'launch'), kern[(self.n, 'launch')] + 1)
        for n, k in list(vars(K).items()):
            if n.endswith('_kernel') and hasattr(k, 'run'):
                setattr(K, n, KK(n))
        D._far_kernel = D._far_dot_kernel = KK('_far_kernel')
    seq = m.new_seq()
    assert m.pool.reserve(seq.kv, 2048)
    seq.pos = 1100
    dec = D.Decoder(m, graph=False, fuse=fuse)
    dec.load(seq)
    dec.st.moe_impl, dec.st.kern = 'triton', True
    dec.x_raw.copy_(m.W.embed[11:12])
    dec.st.pos.fill_(1100)
    if fuse:  # build the static buffers (one-time ops) outside the count
        dec.fused = K.Fused(m, dec.st, dec.x_raw, dec.out)
    cnt = Count()
    with torch.no_grad(), cnt:
        dec._run()
    D._far_kernel = far_k
    __import__('moe_triton').grouped = real_grouped
    tot = cnt.c + kern
    comps = sorted({c for c, _ in tot}, key=lambda c: -tot[(c, 'launch')])
    print(f'{"component":42s} {"launches":>9s} {"views":>7s}')
    for c in comps:
        print(f'{c:42s} {tot[(c, "launch")]:9d} {tot[(c, "view")]:7d}')
    L = sum(v for (c, k), v in tot.items() if k == 'launch')
    V = sum(v for (c, k), v in tot.items() if k == 'view')
    print(f'{"TOTAL":42s} {L:9d} {V:7d}   (triton launches incl.: {sum(kern.values())})')


if __name__ == '__main__':
    main()
