"""CPU only, no GPU: compile every decode Triton kernel for the RTX 5090 (sm_120) with Triton's offline compiler
(ptxas ships with Triton) and print shared memory / warps. Catches GPU-only compile errors the CPU
interpreter accepts (constexpr/runtime branch mixing, dot shapes, layouts) and shared memory over the 5090's
101376 B/block (fails only at launch) before a window, with the num_warps the engine uses (decode_kernels.NW).
Run: cd engine && PATH=$PWD/.venv/bin:$PATH .venv/bin/python tests/compile_sm120.py"""
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.pop('TRITON_INTERPRET', None)
os.environ.setdefault('TRITON_DEFAULT_FP_FUSION', '1')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import triton  # noqa: E402
from triton.backends.compiler import GPUTarget  # noqa: E402
from triton.compiler import ASTSource  # noqa: E402

import decode as D  # noqa: E402
import decode_kernels as K  # noqa: E402

F, B, I32, I64, U8, F8 = '*fp32', '*bf16', '*i32', '*i64', '*u8', '*fp8e4nv'
CH, H = K.CH, 2304
ALIGNED = {'_far3_kernel', '_far_dot_kernel'}   # without the attrs: 1 B loads and spills a real launch doesn't have
SMEM = 101376   # RTX 5090: max shared memory per block (a launch above this fails, the compile doesn't)


def sig(fn, types, consts):
    """Parameters missing from types and consts keep their default as a constexpr (INK_DEC_* options off); with the
    flags on (env INK_DEC_UNROLL / PDL / L2PF, as decode_kernels.kopts()) the option constexprs are set and the
    prefetch pointer args typed."""
    opt = K.kopts()[0]
    consts = dict(consts, **{k: v for k, v in opt.items() if k in fn.arg_names})
    if opt['PF']:
        types = dict(types, **{k: t for k, t in (('pfa', U8), ('pfn', 'i32'), ('pfb', U8), ('pfm', 'i32'))
                               if k in fn.arg_names})
    s = {}
    for i, n in enumerate(fn.arg_names):
        if n not in consts and n not in types:
            consts[n] = fn.params[i].default
        s[n] = 'constexpr' if n in consts else types[n]
    # pointers as a real launch sees them (16-B aligned torch storage), for kernels whose loads vectorize on it
    attrs = {(i,): [['tt.divisibility', 16]] for i, n in enumerate(fn.arg_names)
             if fn.__name__ in ALIGNED and str(s[n]).startswith('*')}
    return ASTSource(fn=fn, signature=s, constexprs=consts, attrs=attrs)


def cases():
    b = K.BLK
    c = dict(F32=False, H=H, CH=CH)
    mix = dict(q_ptr=F, B=F, eidx=I32, n='i32', x_ptr=B, bslot='i32', a_ptr=F, xo_ptr=B, nwt=I64, qm_ptr=F,
               midx=I32, nm='i32', xm_ptr=B, m_ptr=F, nq='i32')
    for first, norm, sa in ((True, True, False), (False, True, False), (False, False, True)):
        yield f'mix first={first} norm={norm}', K._mix_kernel, mix, dict(c, FIRST=first, NORM=norm, STORE_A=sa)
    yield 'mix (mlp, fp32 x)', K._mix_kernel, dict(mix, x_ptr=F, xm_ptr=F), dict(c, FIRST=False, NORM=True, STORE_A=False)
    yield 'gate_z', K._gate_z_kernel, dict(a_ptr=F, xs='i32', m_ptr=F, dtab=I64, z_ptr=F), dict(c, ZB=b['gate']), \
        K.NW['gate']
    yield 'gate_out', K._gate_out_kernel, dict(a_ptr=F, xs='i32', m_ptr=F, gtab=I64, z_ptr=F, o_ptr=F), \
        dict(H=H, BH=b['gout']), K.NW['gout']
    for w8 in (False, True):
        for norm in (False, True):
            yield f'proj norm={norm} w8={w8}', K._proj_kernel, dict(w_ptr=F8 if w8 else B, n0='i32', n1='i32',
                                                                     x_ptr=F if norm else B, xs='i32', nwt=I64, y_ptr=B,
                                                                     s_ptr=F), dict(c, NORM=norm, BN=b['proj'], W8=w8)
        yield f'wo w8={w8}', K._wo_kernel, dict(x_ptr=B, w_ptr=F8 if w8 else B, st=B, cw=F, h_ptr=B, r_ptr=F, s_ptr=F), \
            dict(c, BND=False, BN=b['wo'], W8=w8), K.NW['wo']
    import fp8_dense
    yield 'logits8 (fp8_dense.py)', fp8_dense._logits8_kernel, dict(x_ptr=B, nr='i32', q_ptr=F8, s_ptr=F, out_ptr=F,
                                                                    N='i32'), \
        dict(F32=False, H=H, CH=CH, BN=fp8_dense.LBN, RM=16), fp8_dense.LNW
    pre = dict(y_ptr=B, koff='i32', voff='i32', tkoff='i32', tvoff='i32', qsc=F, rsc=F, kw=F, vw=F, stk=B, stv=B,
               rk=B, rv=B, rpos=I64, pos_ptr=I64, c0=U8, c1=U8, s0=U8, s1=U8, tbl=I64, qo=B, ro=B)
    for full in (False, True):
        yield f'pre full={full}', K._pre_kernel, pre, dict(FULL=full, F32=False, PG=256)
    nt = 1024 // b['near']
    yield 'near_part', K._near_part_kernel, dict(qb=B, rb=B, proj=B, rk=B, rpos=I64, pos_ptr=I64, pm=F, pl=F), \
        dict(F32=False, BLK=b['near'])
    for far in (False, True):
        yield f'near_out far={far}', K._near_out_kernel, dict(qb=B, rb=B, proj=B, y_ptr=B, rk=B, rv=B, rpos=I64,
                                                               pos_ptr=I64, pm=F, pl=F, fl=F, ns='i32', pacc=F, plse=F), \
            dict(FAR=far, F32=False, BLK=b['near'], NS=nt)
        yield f'near_fin far={far}', K._near_fin_kernel, dict(pacc=F, plse=F, fo=F, fl=F, ns='i32', y_ptr=B, out=B), \
            dict(FAR=far, F32=False, NS=nt)
    yield 'router', K._router_kernel, dict(x_ptr=F, nw=F, u_ptr=B, gw=F, s_ptr=F), dict(c, NORM=True, RB=b['router']), K.NW['router']
    moe = dict(lwe='i64', lwr='i32', lse='i64', lsr='i32')
    for fp4 in (True, False):
        w, s = (U8, F8) if fp4 else (F8, U8)
        yield f'w13 fp4={fp4}', K._w13_kernel, dict(moe, s_ptr=F, bias=F, u_ptr=B, w_ptr=w, sc_ptr=s, gs_ptr=F, act=B,
                                                     idx_ptr=I32, wt_ptr=F), \
            dict(FP4=fp4, F32=False, H=H, NI=K.NI, BN=b['w13'], BK=b['kbk']), K.NW['w13']
        yield f'w2 fp4={fp4}', K._w2_kernel, dict(moe, act=B, idx_ptr=I32, wt_ptr=F, w_ptr=w, sc_ptr=s, gs_ptr=F,
                                                   part=F), \
            dict(FP4=fp4, F32=False, H=H, NI=K.NI, BN=b['w2'], BK=b['kbk2']), K.NW['w2']
    for fp4 in (True, False):
        w, s = (U8, F8) if fp4 else (F8, U8)
        yield f'w2all fp4={fp4}', K._w2all_kernel, dict(moe, act=B, idx_ptr=I32, wt_ptr=F, w_ptr=w, sc_ptr=s,
                                                         gs_ptr=F, st=B, cw=F, r_ptr=F, h_out=B), \
            dict(FP4=fp4, F32=False, H=H, NI=K.NI, BN=b['w2a'], BK=b['kbk2']), K.NW['w2a']
    yield 'w2_fin', K._w2_fin_kernel, dict(part=F, st=B, cw=F, r_ptr=F, h_out=B), dict(F32=False, H=H, BN=b['w2f'])
    yield 'far (decode.py)', D._far_kernel, dict(q_ptr=B, c0=U8, c1=U8, s0=F8, s1=F8, tbl=I64, pos_ptr=I64, o_ptr=F,
                                                  l_ptr=F, scale='fp32'), dict(SPLIT=D.DEC_SPLIT, PG=256, BLK=D.FBLK)
    yield 'far dot (decode.py)', D._far_dot_kernel, dict(q_ptr=B, c0=U8, c1=U8, s0=F8, s1=F8, tbl=I64, pos_ptr=I64,
                                                          o_ptr=F, l_ptr=F, scale='fp32'), \
        dict(SPLIT=D.DEC_SPLIT, PG=256, BLK=D.DOT_BLK, F32=False)
    yield f'far dot FAR2 nsp={D.FAR_NSP}', D._far_dot_kernel, dict(q_ptr=B, c0=U8, c1=U8, s0=F8, s1=F8, tbl=I64,
                                                                  pos_ptr=I64, o_ptr=F, l_ptr=F, scale='fp32'), \
        dict(SPLIT=D.DEC_SPLIT, PG=256, BLK=D.DOT_BLK, F32=False, NSP=D.FAR_NSP), D.FAR_KW['num_warps']
    for blk, st, nw in ((D.FAR3_KW['BLK'], D.FAR3_KW['ST'], D.FAR3_KW['num_warps']), (32, 3, 4), (64, 2, 4), (64, 4, 4),
                        (128, 3, 4), (128, 3, 8)):
        yield f'far3 BLK={blk} ST={st}', D._far3_kernel, dict(q_ptr=B, c0='*i32', c1='*i32', s0=F8, s1=F8, tbl=I64,
                                                               pos_ptr=I64, o_ptr=F, l_ptr=F, scale='fp32'), \
            dict(PG=256, BLK=blk, F32=False, NSP=D.FAR_NSP, ST=st), nw
    for far in (False, True):
        yield f'near_out FAR2 far={far}', K._near_out_kernel, dict(qb=B, rb=B, proj=B, y_ptr=B, rk=B, rv=B, rpos=I64,
                                                                    pos_ptr=I64, pm=F, pl=F, fl=F, ns='i32', pacc=F,
                                                                    plse=F), \
            dict(FAR=far, F32=False, BLK=K.BLK['near'], NS=1024 // K.BLK['near'], NSF=64)
        yield f'near_fin FAR2 far={far}', K._near_fin_kernel, dict(pacc=F, plse=F, fo=F, fl=F, ns='i32', y_ptr=B, out=B), \
            dict(FAR=far, F32=False, NS=1024 // K.BLK['near'], NSF=64)
    for name in getattr(K, 'EXTRA_COMPILE', ()):
        yield from getattr(K, name)()


def ptxas_info(k):
    """ptxas -v registers / spill line of a compiled kernel (spills = local memory traffic)."""
    import subprocess
    import tempfile
    ptxas = os.path.join(os.path.dirname(triton.__file__), 'backends/nvidia/bin/ptxas-blackwell')
    with tempfile.NamedTemporaryFile('w', suffix='.ptx') as f:
        f.write(k.asm['ptx'])
        f.flush()
        r = subprocess.run([ptxas, '-v', '-arch=sm_120a', f.name, '-o', '/dev/null'], capture_output=True, text=True)
    return ' '.join(x.split('info    : ')[-1].strip() for x in r.stderr.splitlines() if 'spill' in x or 'registers' in x)


def main():
    tgt, bad = GPUTarget('cuda', 120, 32), 0
    print('INK_DEC options', K.kopts()[0], flush=True)
    for name, fn, types, consts, *nw in cases():
        try:
            k = triton.compile(sig(fn, types, consts), target=tgt, options=dict(num_warps=nw[0]) if nw else None)
            md = k.metadata
            over = md.shared > SMEM
            bad += over
            print(f'{name:28s} {"SHARED MEMORY > %d" % SMEM if over else "ok"}  shared {md.shared}  warps {md.num_warps}'
                  f'  {ptxas_info(k)}')
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f'{name:28s} FAILED: {str(e).splitlines()[0] if str(e) else type(e).__name__}')
            if os.environ.get('V'):
                import traceback
                traceback.print_exc()
    print('COMPILE OK' if not bad else f'{bad} FAILED')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
