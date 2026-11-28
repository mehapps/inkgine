"""CPU tests for Light Mode (/admin/lite):
1. Weights.offload_experts / reload_experts (all layers, partial keep_layers, safetensors slices & CPU fallback,
   OOM rollback) and streamed Model.moe numerical equivalence vs resident (T=1 & T=64, NVFP4 & MXFP8, torch / exact /
   emulate / triton interpreter, with & without tail).
2. Engine.enter_lite / exit_lite: slots=1, SeqBatch, spec='lookup', smaller chunk & margin, drafter/decoder drop,
   GPU-parked states flushed to disk, lite-mode park writes directly to disk (0 GPU-parked), prefix cache restore
   across normal -> lite -> normal, lookup speculation end-to-end on SeqBatch, OOM rollback on exit_lite.
3. Scheduler + HTTP GET/POST /admin/lite + watchdog + /metrics: input validation, non-blocking /health while draining,
   drains active decode without dropping it, preempts active prefill and resumes in lite mode, 1-at-a-time concurrency,
   503 on exit OOM, watchdog auto-exit + OOM retry, Prometheus metrics.

Run: cd engine-dev && PATH=$PWD/.venv/bin:$PATH .venv/bin/python tests/test_lite.py"""
import ctypes
import gc
import os
import shutil
import sys
import tempfile
import threading
import time
import weakref
from types import SimpleNamespace

os.environ['CUDA_VISIBLE_DEVICES'] = ''  # strictly CPU-only
os.environ.setdefault('TRITON_INTERPRET', '1')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from fastapi.testclient import TestClient
from safetensors import safe_open
from safetensors.torch import save_file

import chat
import decode
import decode_kernels as DK
import engine
import kvpool
import model as M
import moe_triton
import server
import spec_runner
from engine import Engine, Request
from loopbatch import LoopBatch
from spec_runner import SeqBatch
from test_batch_sched import FakeEngine


# ===================================================================== 1. Weights offload + streamed Model.moe
def _make_moe_tensors(layers, E, H=M.H, NI=1152):
    """Build synthetic packed NVFP4 (L < 20) and MXFP8 (L >= 20) MoE tensors for `layers`."""
    t = {}
    for L in layers:
        p = f'layers.{L}.mlp.'
        t[p + 'gate.weight'] = (torch.randn(E, H) * 0.05).float()
        t[p + 'gate.bias'] = torch.randn(E) * 0.1
        for w, (rows, cols) in (('w13', (2 * NI, H)), ('w2', (H, NI))):
            q = p + 'experts.' + w + '_weight'
            if L < 20:
                t[q] = torch.randint(0, 256, (E, rows, cols // 2), dtype=torch.uint8)
                t[q + '_scale'] = (torch.rand(E, rows, cols // 16) * 48 + 16).to(torch.float8_e4m3fn)
                t[q + '_amax'] = torch.rand(E) + 0.5
                t[q + '_outer_scale_denominator'] = torch.tensor([2688.0])
            else:
                t[q] = torch.randn(E, rows, cols).to(torch.float8_e4m3fn)
                t[q + '_scale'] = torch.randint(118, 124, (E, rows, cols // 32), dtype=torch.uint8)
    return t


def test_weights_offload_and_reload():
    torch.manual_seed(1)
    tmp = tempfile.mkdtemp(prefix='ink-lite-w-')
    try:
        # 24 tiny layers saved in a real safetensors file to verify PySafeSlice lifecycle + OOM rollback
        st_path = os.path.join(tmp, 'model-00001-of-00001.safetensors')
        tensors = {}
        for L in range(24):
            p = f'model.llm.layers.{L}.mlp.experts.'
            tensors[p + 'w13_weight'] = torch.randint(0, 255, (4, 8, 8), dtype=torch.uint8)
            tensors[p + 'w13_weight_scale'] = torch.ones((4, 8, 1), dtype=torch.float8_e4m3fn)
            tensors[p + 'w2_weight'] = torch.randint(0, 255, (4, 8, 4), dtype=torch.uint8)
            tensors[p + 'w2_weight_scale'] = torch.ones((4, 8, 1), dtype=torch.float8_e4m3fn)
            if L < 20:
                tensors[p + 'w13_weight_amax'] = torch.ones(4)
                tensors[p + 'w13_weight_outer_scale_denominator'] = torch.tensor([1.0])
                tensors[p + 'w2_weight_amax'] = torch.ones(4)
                tensors[p + 'w2_weight_outer_scale_denominator'] = torch.tensor([1.0])
        save_file(tensors, st_path)

        W = object.__new__(M.Weights)
        W.dev, W.e2m1 = torch.device('cpu'), torch.tensor(M.E2M1)
        W.f_path = st_path
        W.f = safe_open(st_path, 'pt', device='cpu')
        W.t = {k[len('model.llm.'):]: W.f.get_tensor(k) for k in W.f.keys()}
        W.offloaded, W.ram_layers, W._stage, W._stage_ev, W._stage_slot, W._mm = set(), set(), {}, [None, None], 0, None

        # Offload all 24 layers -> zero-copy read-only mmap views
        orig_refs = [weakref.ref(W.t[f'layers.{L}.mlp.experts.{suf}']) for L in range(24) for suf in M.Weights.LAZY]
        W.offload_experts()
        gc.collect()
        assert W.offloaded == set(range(24)), W.offloaded
        assert W._mm is not None, 'expected read-only mmap handle'
        assert all(r() is None for r in orig_refs), 'offload_experts leaked original expert tensors'
        for L in range(24):
            for suf in M.Weights.LAZY:
                k = f'layers.{L}.mlp.experts.{suf}'
                assert isinstance(W.t[k], torch.Tensor)
                assert torch.equal(W.t[k], tensors['model.llm.' + k])
            if L < 20:
                assert isinstance(W.t[f'layers.{L}.mlp.experts.w13_weight_amax'], torch.Tensor)
                assert isinstance(W.t[f'layers.{L}.mlp.experts.w2_weight_outer_scale_denominator'], torch.Tensor)

        # Partial keep_layers (first 4 core layers: 6..9)
        keep4 = set(range(6, 10))
        W.offload_experts(keep_layers=keep4)
        assert W.offloaded == set(range(24)) - keep4
        for L in keep4:
            for suf in M.Weights.LAZY:
                k = f'layers.{L}.mlp.experts.{suf}'
                assert isinstance(W.t[k], torch.Tensor)
                assert torch.equal(W.t[k], tensors['model.llm.' + k])

        # Shrink keep_layers to 2 (layers 8, 9 re-offloaded)
        r8 = weakref.ref(W.t['layers.8.mlp.experts.w13_weight'])
        W.offload_experts(keep_layers={6, 7})
        gc.collect()
        assert W.offloaded == set(range(24)) - {6, 7}
        assert r8() is None, 'shrinking keep_layers leaked layer 8 tensor'

        # OOM mid-reload rolls back cleanly
        real_load = W._load_expert_tensor

        def fail_on_layer_10(n):
            if n.startswith('layers.10.'):
                raise torch.OutOfMemoryError('simulated OOM on layer 10')
            return real_load(n)

        W._load_expert_tensor = fail_on_layer_10
        off_before = set(W.offloaded)
        try:
            W.reload_experts()
            raise AssertionError('expected OutOfMemoryError')
        except torch.OutOfMemoryError:
            pass
        finally:
            W._load_expert_tensor = real_load
        assert W.offloaded == off_before, (W.offloaded, off_before)

        # Clean reload succeeds
        W.reload_experts()
        assert W.offloaded == set()
        for L in range(24):
            for suf in M.Weights.LAZY:
                k = f'layers.{L}.mlp.experts.{suf}'
                assert isinstance(W.t[k], torch.Tensor) and torch.equal(W.t[k], tensors['model.llm.' + k])

        # Fallback when f_path is absent (PySafeSlice) and when f is None (synthetic CPU tensors)
        W.f_path = None
        W.offload_experts(keep_layers={0})
        assert W.offloaded == set(range(1, 24))
        assert not isinstance(W.t['layers.1.mlp.experts.w13_weight'], torch.Tensor)
        W.reload_experts()
        assert W.offloaded == set()
        W.f = None
        W.offload_experts(keep_layers={0})
        assert W.offloaded == set(range(1, 24))
        assert isinstance(W.t['layers.1.mlp.experts.w13_weight'], torch.Tensor)
        W.reload_experts()
        assert W.offloaded == set()
        print('PASS 1a: Weights.offload_experts (mmap + slice + cpu) / reload_experts / partial keep / OOM rollback')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_weights_ram_experts():
    torch.manual_seed(7)
    tmp = tempfile.mkdtemp(prefix='ink-lite-ram-')
    try:
        st_path = os.path.join(tmp, 'model-00001-of-00001.safetensors')
        tensors = {}
        for L in range(24):
            p = f'model.llm.layers.{L}.mlp.experts.'
            if L < 20:
                tensors[p + 'w13_weight'] = torch.randint(0, 255, (4, 8, 8), dtype=torch.uint8)
                tensors[p + 'w13_weight_scale'] = torch.ones((4, 8, 1), dtype=torch.float8_e4m3fn)
                tensors[p + 'w2_weight'] = torch.randint(0, 255, (4, 8, 4), dtype=torch.uint8)
                tensors[p + 'w2_weight_scale'] = torch.ones((4, 8, 1), dtype=torch.float8_e4m3fn)
                tensors[p + 'w13_weight_amax'] = torch.ones(4)
                tensors[p + 'w13_weight_outer_scale_denominator'] = torch.tensor([1.0])
                tensors[p + 'w2_weight_amax'] = torch.ones(4)
                tensors[p + 'w2_weight_outer_scale_denominator'] = torch.tensor([1.0])
            else:
                tensors[p + 'w13_weight'] = torch.randn((4, 8, 16)).to(torch.float8_e4m3fn)
                tensors[p + 'w13_weight_scale'] = torch.randint(118, 124, (4, 8, 1), dtype=torch.uint8)
                tensors[p + 'w2_weight'] = torch.randn((4, 8, 8)).to(torch.float8_e4m3fn)
                tensors[p + 'w2_weight_scale'] = torch.randint(118, 124, (4, 8, 1), dtype=torch.uint8)
        save_file(tensors, st_path)

        W = object.__new__(M.Weights)
        W.dev, W.e2m1, W.f_path = torch.device('cpu'), torch.tensor(M.E2M1), st_path
        W.f = safe_open(st_path, 'pt', device='cpu')
        W.t = {k[len('model.llm.'):]: W.f.get_tensor(k) for k in W.f.keys()}
        W.offloaded, W.ram_layers, W._stage, W._stage_ev, W._stage_slot, W._mm = set(), set(), {}, [None, None], 0, None

        def is_mmap_view(k):
            t = W.t[k]
            mv = W._mmap_view(k)
            return isinstance(t, torch.Tensor) and t.data_ptr() == mv.data_ptr()

        # 1) Direct offload with ram_layers=core (6..17) and keep_layers={6, 7} -> keep wins over ram_layers
        keep2 = {6, 7}
        core = set(range(6, 18))
        W.offload_experts(keep_layers=keep2, ram_layers=core)
        assert W.offloaded == set(range(24)) - keep2
        assert W.ram_layers == core - keep2
        for L in range(24):
            for suf in M.Weights.LAZY:
                k = f'layers.{L}.mlp.experts.{suf}'
                t = W.t[k]
                assert isinstance(t, torch.Tensor) and t.device.type == 'cpu'
                assert torch.equal(t, tensors['model.llm.' + k])
                if L in W.ram_layers:
                    assert not is_mmap_view(k), f'layer {L} {suf} still shares mmap storage'
                    assert t.untyped_storage().data_ptr() != W._mmap_view(k).untyped_storage().data_ptr()
                elif L in W.offloaded:
                    assert is_mmap_view(k), f'layer {L} {suf} should be a zero-copy mmap view'

        # 2) Transition mmap -> all RAM (from offloaded mmap views)
        W.offload_experts(keep_layers=(), ram_layers=range(24))
        assert W.offloaded == set(range(24)) and W.ram_layers == set(range(24))
        for L in range(24):
            for suf in M.Weights.LAZY:
                k = f'layers.{L}.mlp.experts.{suf}'
                assert not is_mmap_view(k), f'layer {L} {suf} not detached from mmap'
                assert torch.equal(W.t[k], tensors['model.llm.' + k])

        # 3) Transition RAM -> mmap frees the owned RAM tensors
        ram_refs = [weakref.ref(W.t[f'layers.{L}.mlp.experts.{suf}']) for L in range(24) for suf in M.Weights.LAZY]
        W.offload_experts(keep_layers=(), ram_layers=())
        gc.collect()
        assert W.offloaded == set(range(24)) and W.ram_layers == set()
        assert all(r() is None for r in ram_refs), 'switching RAM -> mmap leaked owned RAM tensors'
        for L in range(24):
            for suf in M.Weights.LAZY:
                assert is_mmap_view(f'layers.{L}.mlp.experts.{suf}')

        # 4) OOM rollback inside _reload_layers preserves ram_layers vs mmap backing
        W.offload_experts(keep_layers=(), ram_layers=core)
        assert W.ram_layers == core
        real_load = W._load_expert_tensor

        def fail_on_layer_10(n):
            if n.startswith('layers.10.'):
                raise torch.OutOfMemoryError('simulated OOM on layer 10')
            return real_load(n)

        W._load_expert_tensor = fail_on_layer_10
        try:
            W.reload_experts()
            raise AssertionError('expected OutOfMemoryError')
        except torch.OutOfMemoryError:
            pass
        finally:
            W._load_expert_tensor = real_load
        assert W.offloaded == set(range(24)) and W.ram_layers == core
        for L in range(24):
            k = f'layers.{L}.mlp.experts.w13_weight'
            if L in core:
                assert not is_mmap_view(k), f'rollback turned RAM layer {L} into mmap'
            else:
                assert is_mmap_view(k), f'rollback lost mmap view on layer {L}'
            assert torch.equal(W.t[k], tensors['model.llm.' + k])

        # 5) Clean reload clears ram_layers
        W.reload_experts()
        assert W.offloaded == set() and W.ram_layers == set()

        # 6) Fallback without f_path (PySafeSlice): ram_layers materialize owned CPU tensors, others stay slices
        W.f_path = None
        W.offload_experts(keep_layers=(), ram_layers={6, 7})
        assert W.offloaded == set(range(24)) and W.ram_layers == {6, 7}
        assert isinstance(W.t['layers.6.mlp.experts.w13_weight'], torch.Tensor)
        assert not isinstance(W.t['layers.0.mlp.experts.w13_weight'], torch.Tensor)
        W.reload_experts()
        assert W.offloaded == set() and W.ram_layers == set()
        print('PASS 1c: Weights ram_layers (owned CPU detachment, mmap<->RAM transitions, OOM rollback, slice fallback)')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_streamed_moe_numerics():
    torch.manual_seed(42)
    tmp = tempfile.mkdtemp(prefix='ink-lite-moe-')
    old_batch = M.LITE_EXPERT_BATCH
    old_v2 = moe_triton.V2
    try:
        E = 24
        M.LITE_EXPERT_BATCH = 5  # forces 5 batches (5, 5, 5, 5, 4) across 24 experts
        raw = _make_moe_tensors((0, 6, 20), E)
        st_path = os.path.join(tmp, 'model-00001-of-00001.safetensors')
        save_file({'model.llm.' + k: v for k, v in raw.items()}, st_path)

        m_res = object.__new__(M.Model)
        m_res.W = object.__new__(M.Weights)
        m_res.W.dev, m_res.W.e2m1, m_res.W.f = torch.device('cpu'), torch.tensor(M.E2M1), None
        m_res.W.t = {k: v.clone() for k, v in raw.items()}
        m_res.W.offloaded, m_res.W.ram_layers, m_res.W._stage, m_res.W._stage_ev = set(), set(), {}, None

        m_off = object.__new__(M.Model)
        m_off.W = object.__new__(M.Weights)
        m_off.W.dev, m_off.W.e2m1 = torch.device('cpu'), torch.tensor(M.E2M1)
        m_off.W.f_path = st_path
        m_off.W.f = safe_open(st_path, 'pt', device='cpu')
        m_off.W.t = {k: v.clone() for k, v in raw.items()}
        m_off.W.offloaded, m_off.W.ram_layers, m_off.W._stage, m_off.W._stage_ev, m_off.W._stage_slot, m_off.W._mm = (
            set(), set(), {}, [None, None], 0, None)
        m_off.W.offload_experts()
        assert m_off.W.offloaded == {0, 6, 20} and m_off.W.ram_layers == set() and m_off.W._mm is not None

        # Compare mmap, core (L=6 in RAM, L=0/20 mmap), and all (0, 6, 20 in RAM) against resident
        modes = (('mmap', ()), ('core', range(6, 18)), ('all', range(24)))
        inputs = {(L, T): M.rms(torch.randn(T, M.H)).bfloat16() for L in (0, 6, 20) for T in (1, 64)}
        refs = {(L, T, impl): m_res.moe(L, inputs[(L, T)], impl=impl)
                for L in (0, 6, 20) for T in (1, 64) for impl in ('torch', 'exact', 'emulate')}
        triton_u1 = {L: M.rms(torch.randn(1, M.H)).bfloat16() for L in (0, 6, 20)}
        triton_refs = {}
        for L in (0, 6, 20):
            for v2 in (False, True):
                moe_triton.V2 = v2
                triton_refs[(L, v2)] = m_res.moe(L, triton_u1[L], impl='triton')

        mode_outs = {}
        for mode, rl in modes:
            m_off.W.offload_experts(ram_layers=rl)
            exp_ram = ({0, 6, 20} & set(rl))
            assert m_off.W.ram_layers == exp_ram, (mode, m_off.W.ram_layers, exp_ram)
            for L in (0, 6, 20):
                for T in (1, 64):
                    u = inputs[(L, T)]
                    for impl in ('torch', 'exact', 'emulate'):
                        ref = refs[(L, T, impl)]
                        got = m_off.moe(L, u, impl=impl)
                        assert torch.equal(ref, got), (
                            f'mode={mode} L={L} T={T} impl={impl}: max diff {(ref.float() - got.float()).abs().max()}')
                        if impl == 'torch':
                            mode_outs[(mode, L, T)] = got

                    if mode == 'mmap' and L in (0, 20):
                        # Check tail=(st, conv_w, r): res_out applied once at the end
                        st1 = torch.randn(3, M.H, dtype=torch.bfloat16)
                        st2 = st1.clone()
                        cw = torch.randn(M.H, 1, 4, dtype=torch.float32) * 0.1
                        r = torch.randn(T, M.H, dtype=torch.float32)
                        ref_h, ref_hn = M.res_out(refs[(L, T, 'torch')], st1, cw, r)
                        got_h, got_hn = m_off.moe(L, u, impl='torch', tail=(st2, cw, r))
                        assert torch.equal(ref_h, got_h) and torch.equal(ref_hn, got_hn) and torch.equal(st1, st2), (
                            f'tail mismatch L={L} T={T}')

                # Check Triton interpreter path (v1 atomic & v2 fin callback) for mmap (0, 20) and RAM (6, 20)
                if (mode == 'mmap' and L in (0, 20)) or (mode == 'core' and L == 6) or (mode == 'all' and L == 20):
                    for v2 in (False, True):
                        moe_triton.V2 = v2
                        ref_t = triton_refs[(L, v2)]
                        got_t = m_off.moe(L, triton_u1[L], impl='triton')
                        du = (ref_t.view(torch.int16).int() - got_t.view(torch.int16).int()).abs()
                        assert du.max().item() <= 1, f'mode={mode} triton V2={v2} L={L} T=1 max ulp {du.max().item()}'

        for L in (0, 6, 20):
            for T in (1, 64):
                assert torch.equal(mode_outs[('mmap', L, T)], mode_outs[('core', L, T)])
                assert torch.equal(mode_outs[('mmap', L, T)], mode_outs[('all', L, T)])

        print('PASS 1b: streamed Model.moe == resident across mmap/core/all '
              '(T=1 & T=64, NVFP4 & MXFP8, torch/exact/emulate/triton, tail)')
    finally:
        M.LITE_EXPERT_BATCH = old_batch
        moe_triton.V2 = old_v2
        shutil.rmtree(tmp, ignore_errors=True)


# ===================================================================== 2. Engine enter_lite / exit_lite / disk cache / lookup
class MiniWeights:
    def __init__(self):
        self.dev = torch.device('cpu')
        self.offloaded = set()
        self.ram_layers = set()
        self.pinned_layers = set()
        self.fail_reload = False

    def offload_experts(self, keep_layers=(), ram_layers=()):
        keep = set(keep_layers)
        self.offloaded = set(range(24)) - keep
        self.ram_layers = set(ram_layers) - keep
        self.pinned_layers = set(self.ram_layers) if self.dev.type == 'cuda' else set()

    def reload_experts(self):
        if self.fail_reload:
            raise torch.OutOfMemoryError('CUDA out of memory (simulated)')
        self.offloaded.clear()
        self.ram_layers.clear()
        self.pinned_layers.clear()


class MiniModel:
    """Fast deterministic model that exercises real KVPool pages, Seq rings/conv, and SeqBatch + Spec.Lookup."""
    def __init__(self, pool, vocab=256):
        self.dev = torch.device('cpu')
        self.pool = pool
        self.W = MiniWeights()
        self.vocab = vocab
        self.cycle = [10, 11, 12, 13, 14, 15, 16, 17]
        self.last_id = None

    def new_seq(self, tbl=None):
        seq = SimpleNamespace(
            pos=0,
            kv=tbl if tbl is not None else self.pool.new_table(),
            conv=[{'k': torch.zeros(3, 4, dtype=torch.bfloat16),
                   'v': torch.zeros(3, 4, dtype=torch.bfloat16),
                   'attn': torch.zeros(3, 4, dtype=torch.bfloat16),
                   'mlp': torch.zeros(3, 4, dtype=torch.bfloat16)}],
            ring=[(torch.zeros(0, M.NKV, M.HD, dtype=torch.bfloat16),
                   torch.zeros(0, M.NKV, M.HD, dtype=torch.bfloat16))],
            dev=self.dev,
        )
        return seq

    def embed(self, ids):
        # Encode token id in column 0 so forward/logits can recover it cleanly
        x = torch.zeros(len(ids), 4, dtype=torch.bfloat16)
        for i, tid in enumerate(ids):
            x[i, 0] = float(tid)
        return x

    def layer(self, L, h, blocks, msg, seq, v, pos):
        return h, blocks

    def forward(self, x, seq):
        T = x.shape[0]
        # Write synthetic K/V into full-visit slot 0 so KVPool pages carry real data
        K = torch.zeros(T, M.NKV, M.HD, dtype=torch.bfloat16)
        V = torch.zeros(T, M.NKV, M.HD, dtype=torch.bfloat16)
        K[:, 0, 0] = x[:, 0]
        V[:, 0, 0] = x[:, 0]
        seq.kv.put(0, seq.pos, K, V)
        # Update conv state & ring in-place like real Model.forward
        for t in range(T):
            seq.conv[0]['mlp'] = torch.cat((seq.conv[0]['mlp'][1:], x[t:t + 1]))
        rk, rv = seq.ring[0]
        seq.ring[0] = (torch.cat((rk, K))[-64:], torch.cat((rv, V))[-64:])
        # Call self.layer at TAP_VISITS so forward_taps captures taps cleanly
        for v in spec_runner.TAP_VISITS:
            x, _ = self.layer(0, x, [], None, seq, v, None)
        seq.pos += T
        return x

    def logits(self, f):
        # Deterministic next-token rule: continue `self.cycle` so Spec.Lookup gets high acceptance!
        out = torch.full((f.shape[0], self.vocab), -10.0, dtype=torch.float32)
        for i in range(f.shape[0]):
            tid = int(round(f[i, 0].item()))
            nxt = self.cycle[(self.cycle.index(tid) + 1) % len(self.cycle)] if tid in self.cycle else self.cycle[0]
            out[i, nxt] = 10.0
        return out


class TrackedBD(LoopBatch):
    def __init__(self, model, slots=4):
        super().__init__(model, slots, dec=False)
        self.drops = 0

    def drop(self, keep=0):
        self.drops += 1
        return super().drop(keep=keep)


class FakeTok:
    def id_to_token(self, tid):
        return f't{tid}'

    def decode(self, ids, skip_special_tokens=False):
        return ' '.join(f't{i}' for i in ids)


def _make_mini_engine(cache_dir):
    pool = kvpool.KVPool(512, 'cpu', 'mini-wid', cache_dir=cache_dir, cap_gb=1.0, min_free_gb=0.0)
    m = MiniModel(pool)
    E = Engine.__new__(Engine)
    E.tok = FakeTok()
    E.pool = pool
    E.model = m
    E.graph = False
    E.bd = TrackedBD(m, slots=4)
    E.spec = 'dspark'
    E.ds = SimpleNamespace(live=0)
    E.ds_free = [SimpleNamespace(graphs={})]
    E.spec_live = set()
    E.vision = None
    E.chunk = 4096
    E.max_ctx = 512
    E.parked = []
    E.mem_free = None
    E.ooms = 0
    E.vision_images = 0
    E._margin = None
    E.lite, E.lite_keep_core, E.lite_ram_experts, E._full_state = False, 0, 'mmap', None
    return E


def test_engine_lite_lifecycle():
    tmp = tempfile.mkdtemp(prefix='ink-lite-eng-')
    old_page, old_snap, old_disk = kvpool.PAGE, engine.SNAP_MIN, engine.DISK_MIN
    old_vms = spec_runner.VERIFY_MS
    try:
        kvpool.PAGE = 16
        engine.SNAP_MIN = 8
        engine.DISK_MIN = 8
        spec_runner.VERIFY_MS = (1.0,) * 8  # encourage lookup drafting on CPU

        eng = _make_mini_engine(tmp)
        full_bd = eng.bd

        # Run turn 1 in normal mode (keep=True): parked on GPU
        t1 = [10, 11, 12, 13, 14, 15, 16, 17] * 3  # 24 tokens (1 full page + 8 tail)
        st1 = {}
        seq1, _ = eng.prefill(t1, [], stats=st1, max_tokens=4, keep=True)
        eng.pool.release(seq1.kv)
        assert len(eng.parked) == 1 and not eng.lite

        # Enter light mode with keep_core_layers=4
        res = eng.enter_lite(keep_core_layers=4, chunk=512, margin_mb=256)
        assert res['lite'] is True
        assert res['keep_core_layers'] == 4
        assert res['offloaded_layers'] == 20
        assert res['ram_experts'] == 'mmap'
        assert res['ram_layers'] == 0
        assert res['pinned_layers'] == 0
        assert res['decoder'] == 'seq'
        assert res['slots'] == 1
        assert res['chunk'] == 512
        assert res['spec'] == 'lookup'
        assert res['vram_allocated_mb'] is None and res['vram_free_mb'] is None
        assert eng.model.W.offloaded == set(range(24)) - {6, 7, 8, 9}
        assert eng.ds is None and eng.ds_free == []
        assert full_bd.drops >= 1
        assert eng.parked == []
        assert isinstance(eng.bd, SeqBatch) and eng.bd.n == 1
        assert eng._margin == (256 << 20)
        eng.pool.flush()
        assert eng.pool.states, 'turn 1 state should have been flushed to disk on enter_lite'

        # Reconfigure while in lite mode (keep_core_layers=2, chunk=256)
        res2 = eng.enter_lite(keep_core_layers=2, chunk=256)
        assert res2['keep_core_layers'] == 2 and res2['offloaded_layers'] == 22 and res2['chunk'] == 256
        assert eng._full_state[0][1] is TrackedBD and eng._full_state[1] == 4096 and eng._full_state[2] == 'dspark'

        # Turn 2 in lite mode restores turn 1 from disk AND writes turn 2 straight to disk without GPU-parking
        t2 = t1 + [10, 11, 12, 13, 14, 15, 16, 17]  # 32 tokens
        st2 = {}
        seq2, _ = eng.prefill(t2, [], stats=st2, max_tokens=4, keep=True)
        eng.pool.release(seq2.kv)
        assert st2['cached_tokens'] == len(t1), st2
        assert eng.parked == [], 'lite mode must not accumulate GPU-parked states'
        eng.pool.flush()
        assert len(t2) in eng.pool.states, 'turn 2 should be saved directly to disk in lite mode'

        # Turn 3 in lite mode restores turn 2 from disk
        t3 = t2 + [10, 11, 12, 13]
        st3 = {}
        seq3, _ = eng.prefill(t3, [], stats=st3, max_tokens=4, keep=True)
        eng.pool.release(seq3.kv)
        assert st3['cached_tokens'] == len(t2), st3
        assert eng.parked == []

        # Lookup speculation end-to-end on SeqBatch(1) in lite mode == plain decode, with drafts accepted!
        def run_gen(use_spec):
            req = Request(eng, list(t1), [], keep=False, max_tokens=20, temperature=0.0, prefill_text=True,
                          spec=use_spec)
            eng.start(req)
            while not eng.prefill_chunk(req):
                pass
            eng.to_decode(req, 0)
            while not req.done:
                eng.decode_step([req])
            sp_stats = dict(req.sp.stats) if req.sp is not None else None
            eng.finish(req)
            return req.out, sp_stats

        out_plain, _ = run_gen(False)
        out_spec, sp_stats = run_gen(True)
        assert out_spec == out_plain, (out_spec, out_plain)
        assert sp_stats is not None and sp_stats['accepted'] > 0, sp_stats
        assert len(eng.bd.th[0]) == 0  # cleared by free()

        # OOM during exit_lite leaves lite mode completely intact
        eng.model.W.fail_reload = True
        try:
            eng.exit_lite()
            raise AssertionError('expected OutOfMemoryError')
        except torch.OutOfMemoryError:
            pass
        assert eng.lite is True and eng.slots == 1 and eng.spec == 'lookup' and eng._full_state is not None
        assert len(eng.model.W.offloaded) == 22

        # Clean exit_lite restores full mode with a freshly rebuilt decoder
        eng.model.W.fail_reload = False
        res_out = eng.exit_lite()
        assert res_out['lite'] is False and res_out['offloaded_layers'] == 0
        assert res_out['slots'] == 4 and res_out['chunk'] == 4096 and res_out['spec'] == 'dspark'
        assert isinstance(eng.bd, TrackedBD) and eng.bd is not full_bd and eng._margin is None and eng._full_state is None

        # Turn 4 back in normal mode still restores turn 3 from disk!
        t4 = t3 + [14, 15, 16, 17]
        st4 = {}
        seq4, _ = eng.prefill(t4, [], stats=st4, max_tokens=4, keep=True)
        eng.pool.release(seq4.kv)
        assert st4['cached_tokens'] == len(t3), st4

        print(f'PASS 2: Engine lite lifecycle, disk cache across normal->lite->normal, '
              f'lookup spec accepted={sp_stats["accepted"]}/{sp_stats["drafted"]}, OOM rollback')
    finally:
        kvpool.PAGE = old_page
        engine.SNAP_MIN = old_snap
        engine.DISK_MIN = old_disk
        spec_runner.VERIFY_MS = old_vms
        shutil.rmtree(tmp, ignore_errors=True)


def test_no_expert_ref_leak():
    """Regression test: BatchDecoder.F caches direct references to every layer's expert tensors (decode_kernels.py:834)
    and server._instrument wraps bd._alloc (reference cycle). Entering light mode must drop the old decoder completely
    so gc.collect() frees every offloaded expert tensor, and exit_lite() must rebuild a fresh decoder bound to the
    newly reloaded expert tensors."""
    tmp = tempfile.mkdtemp(prefix='ink-lite-leak-')
    try:
        st_path = os.path.join(tmp, 'model-00001-of-00001.safetensors')
        tensors = {}
        for L in range(24):
            p = f'model.llm.layers.{L}.mlp.experts.'
            tensors[p + 'w13_weight'] = torch.randint(0, 255, (4, 8, 8), dtype=torch.uint8)
            tensors[p + 'w13_weight_scale'] = torch.ones((4, 8, 1), dtype=torch.float8_e4m3fn)
            tensors[p + 'w2_weight'] = torch.randint(0, 255, (4, 8, 4), dtype=torch.uint8)
            tensors[p + 'w2_weight_scale'] = torch.ones((4, 8, 1), dtype=torch.float8_e4m3fn)
        save_file(tensors, st_path)

        W = object.__new__(M.Weights)
        W.dev, W.e2m1, W.f_path = torch.device('cpu'), torch.tensor(M.E2M1), st_path
        W.f = safe_open(st_path, 'pt', device='cpu')
        W.t = {k[len('model.llm.'):]: W.f.get_tensor(k) for k in W.f.keys()}
        W.offloaded, W.ram_layers, W._stage, W._stage_ev, W._stage_slot, W._mm = set(), set(), {}, [None, None], 0, None

        class FakeFused:
            def __init__(self, vis):
                self.vis = vis

        class FakeBatchDecoder:
            dec = True

            def __init__(self, model, slots=4, graph=False):
                self.m, self.n, self.graph = model, slots, graph
                self.seqs, self.decs = [None] * slots, [None] * slots
                # Mirrors BatchDecoder.F.vis[v]['ex'] holding direct refs to every layer's expert tensors
                self.F = FakeFused([
                    {'ex': [model.W.t[f'layers.{L}.mlp.experts.{suf}'] for suf in M.Weights.LAZY]}
                    for L in range(24)
                ])

            def busy(self):
                return any(s is not None for s in self.seqs)

            def drop(self, keep=0):
                return 0

            def _alloc(self, n):
                return n

        eng = _make_mini_engine(tmp)
        eng.model.W = W
        eng.bd = FakeBatchDecoder(eng.model, slots=4, graph=False)
        server._instrument(eng)

        bd_ref = weakref.ref(eng.bd)
        f_ref = weakref.ref(eng.bd.F)
        ex_refs = [weakref.ref(W.t[f'layers.{L}.mlp.experts.{suf}']) for L in range(24) for suf in M.Weights.LAZY]

        eng.enter_lite(keep_core_layers=0)
        gc.collect()
        assert bd_ref() is None, 'old BatchDecoder still alive after enter_lite'
        assert f_ref() is None, 'old BatchDecoder.F still alive after enter_lite'
        alive = sum(r() is not None for r in ex_refs)
        assert alive == 0, f'{alive}/{len(ex_refs)} offloaded expert tensors still alive (VRAM ref leak!)'
        assert isinstance(eng.bd, SeqBatch) and eng.bd.use_taps is False

        eng.exit_lite()
        assert isinstance(eng.bd, FakeBatchDecoder) and eng.bd.n == 4
        for L in range(24):
            for j, suf in enumerate(M.Weights.LAZY):
                assert eng.bd.F.vis[L]['ex'][j] is W.t[f'layers.{L}.mlp.experts.{suf}']
        print('PASS 2b: weakref check confirms 0/96 offloaded expert tensors leaked (BatchDecoder.F dropped & rebuilt)')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_engine_ram_experts_transitions():
    tmp = tempfile.mkdtemp(prefix='ink-lite-eng-ram-')
    try:
        st_path = os.path.join(tmp, 'model-00001-of-00001.safetensors')
        tensors = {}
        for L in range(24):
            p = f'model.llm.layers.{L}.mlp.experts.'
            tensors[p + 'w13_weight'] = torch.randint(0, 255, (4, 8, 8), dtype=torch.uint8)
            tensors[p + 'w13_weight_scale'] = torch.ones((4, 8, 1), dtype=torch.float8_e4m3fn)
            tensors[p + 'w2_weight'] = torch.randint(0, 255, (4, 8, 4), dtype=torch.uint8)
            tensors[p + 'w2_weight_scale'] = torch.ones((4, 8, 1), dtype=torch.float8_e4m3fn)
        save_file(tensors, st_path)

        W = object.__new__(M.Weights)
        W.dev, W.e2m1, W.f_path = torch.device('cpu'), torch.tensor(M.E2M1), st_path
        W.f = safe_open(st_path, 'pt', device='cpu')
        W.t = {k[len('model.llm.'):]: W.f.get_tensor(k) for k in W.f.keys()}
        W.offloaded, W.ram_layers, W._stage, W._stage_ev, W._stage_slot, W._mm = set(), set(), {}, [None, None], 0, None

        eng = _make_mini_engine(tmp)
        eng.model.W = W

        # Invalid ram_experts raises ValueError without changing state
        try:
            eng.enter_lite(ram_experts='bogus')
            raise AssertionError('expected ValueError')
        except ValueError:
            pass
        assert eng.lite is False and W.offloaded == set() and W.ram_layers == set()

        # Enter lite: mmap -> core -> core+keep -> all+keep -> all -> mmap
        r = eng.enter_lite(keep_core_layers=0, ram_experts='mmap')
        assert r['ram_experts'] == 'mmap' and r['ram_layers'] == 0 and r['offloaded_layers'] == 24
        assert W.ram_layers == set()

        r = eng.enter_lite(keep_core_layers=0, ram_experts='core')
        assert r['ram_experts'] == 'core' and r['ram_layers'] == 12 and r['offloaded_layers'] == 24
        assert W.ram_layers == set(range(6, 18))

        r = eng.enter_lite(keep_core_layers=4, ram_experts='core')
        assert r['ram_experts'] == 'core' and r['ram_layers'] == 8 and r['offloaded_layers'] == 20
        assert W.ram_layers == set(range(10, 18)) and W.offloaded == set(range(24)) - {6, 7, 8, 9}

        r = eng.enter_lite(keep_core_layers=4, ram_experts='all')
        assert r['ram_experts'] == 'all' and r['ram_layers'] == 20 and r['offloaded_layers'] == 20
        assert W.ram_layers == set(range(24)) - {6, 7, 8, 9}

        r = eng.enter_lite(keep_core_layers=0, ram_experts='all')
        assert r['ram_experts'] == 'all' and r['ram_layers'] == 24 and r['offloaded_layers'] == 24
        assert W.ram_layers == set(range(24))

        r = eng.enter_lite(keep_core_layers=0, ram_experts='mmap')
        assert r['ram_experts'] == 'mmap' and r['ram_layers'] == 0 and r['offloaded_layers'] == 24
        assert W.ram_layers == set()

        # OOM rollback on exit_lite (both during reload_experts AND during _build_bd) preserves ram_layers
        r = eng.enter_lite(keep_core_layers=2, ram_experts='core')
        assert r['ram_experts'] == 'core' and r['ram_layers'] == 10 and W.ram_layers == set(range(8, 18))

        # Case A: OOM during reload_experts
        real_load = W._load_expert_tensor
        W._load_expert_tensor = lambda n: (_ for _ in ()).throw(torch.OutOfMemoryError('reload OOM')) \
            if n.startswith('layers.12.') else real_load(n)
        try:
            eng.exit_lite()
            raise AssertionError('expected OutOfMemoryError')
        except torch.OutOfMemoryError:
            pass
        finally:
            W._load_expert_tensor = real_load
        assert eng.lite is True and eng.lite_ram_experts == 'core'
        assert W.offloaded == set(range(24)) - {6, 7} and W.ram_layers == set(range(8, 18))
        st = eng.lite_status()
        assert st['ram_experts'] == 'core' and st['ram_layers'] == 10

        # Case B: OOM during _build_bd (after reload_experts succeeded)
        real_build = eng._build_bd
        eng._build_bd = lambda spec: (_ for _ in ()).throw(torch.OutOfMemoryError('build_bd OOM'))
        try:
            eng.exit_lite()
            raise AssertionError('expected OutOfMemoryError')
        except torch.OutOfMemoryError:
            pass
        finally:
            eng._build_bd = real_build
        assert eng.lite is True and eng.lite_ram_experts == 'core'
        assert W.offloaded == set(range(24)) - {6, 7} and W.ram_layers == set(range(8, 18))
        for L in range(8, 18):
            k = f'layers.{L}.mlp.experts.w13_weight'
            assert W.t[k].data_ptr() != W._mmap_view(k).data_ptr()

        # Clean exit clears ram_layers and resets ram_experts to 'mmap'
        r_out = eng.exit_lite()
        assert r_out['lite'] is False and r_out['offloaded_layers'] == 0
        assert r_out['ram_experts'] == 'mmap' and r_out['ram_layers'] == 0
        assert W.offloaded == set() and W.ram_layers == set()
        print('PASS 2c: Engine ram_experts transitions (mmap/core/all + keep overlap) & exit_lite OOM rollback')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ===================================================================== 3. Scheduler + /admin/lite + watchdog + /metrics
class LiteFakeEngine(FakeEngine):
    def __init__(self, slots=4, chunk_s=0.04, step_s=0.02):
        super().__init__(slots=slots, chunk_s=chunk_s, step_s=step_s)
        self.full_slots = slots
        self.full_chunk = self.chunk
        self.lite = False
        self.lite_keep_core = 0
        self.lite_ram_experts = 'mmap'
        self.spec = 'dspark'
        self.model = SimpleNamespace(W=SimpleNamespace(offloaded=set(), ram_layers=set(), pinned_layers=set()))
        self.fail_exit_count = 0

    def enter_lite(self, keep_core_layers=0, chunk=None, margin_mb=None, ram_experts='mmap'):
        if ram_experts not in ('mmap', 'core', 'all'):
            raise ValueError(f"ram_experts must be 'mmap', 'core', or 'all', got {ram_experts!r}")
        keep = max(0, min(12, int(keep_core_layers)))
        keep_layers = set(range(6, 6 + keep))
        self.lite = True
        self.lite_keep_core = keep
        self.lite_ram_experts = ram_experts
        self.slots = 1
        self.chunk = min(int(chunk or 1024), self.full_chunk)
        self.spec = 'lookup'
        self.model.W.offloaded = set(range(24)) - keep_layers
        self.model.W.ram_layers = (set() if ram_experts == 'mmap' else
                                   (set(range(6, 18)) if ram_experts == 'core' else set(range(24))) - keep_layers)
        self.L('enter_lite', keep, ram_experts)
        return self.lite_status()

    def exit_lite(self):
        if not self.lite:
            return self.lite_status()
        if self.fail_exit_count > 0:
            self.fail_exit_count -= 1
            self.L('exit_lite_oom')
            raise torch.OutOfMemoryError('simulated GPU OOM on exit_lite')
        self.lite = False
        self.lite_keep_core = 0
        self.lite_ram_experts = 'mmap'
        self.slots = self.full_slots
        self.chunk = self.full_chunk
        self.spec = 'dspark'
        self.model.W.offloaded = set()
        self.model.W.ram_layers = set()
        self.model.W.pinned_layers = set()
        self.L('exit_lite')
        return self.lite_status()

    def lite_status(self):
        return {
            'lite': self.lite,
            'keep_core_layers': self.lite_keep_core,
            'offloaded_layers': len(self.model.W.offloaded),
            'ram_experts': self.lite_ram_experts,
            'ram_layers': len(self.model.W.ram_layers),
            'pinned_layers': len(self.model.W.pinned_layers),
            'decoder': 'seq' if self.lite else 'batch',
            'slots': self.slots,
            'chunk': self.chunk,
            'spec': self.spec,
            'vram_allocated_mb': None,
            'vram_reserved_mb': None,
            'vram_free_mb': None,
        }


def test_scheduler_and_http_lite():
    eng = LiteFakeEngine(slots=4, chunk_s=0.04, step_s=0.025)
    old_eng, old_sched, old_retry = server.eng, server.sched, server.LITE_RETRY_S
    server.eng = eng
    server.sched = server.Scheduler(eng)
    client = TestClient(server.app)

    def ask(name, chunks, toks, box, delay=0.0):
        if delay:
            time.sleep(delay)
        r = client.post('/v1/chat/completions', json={'messages': [{'role': 'user', 'content': f'{name}:{chunks}:{toks}'}]})
        assert r.status_code == 200, r.text
        box[name] = r.json()['choices'][0]['message']['content']

    try:
        # 3a. Initial status & validation
        r = client.get('/admin/lite')
        assert r.status_code == 200
        d = r.json()
        assert d['lite'] is False and d['slots'] == 4 and d['transitioning'] is False and d['timeout_at'] is None
        assert d['ram_experts'] == 'mmap' and d['ram_layers'] == 0

        for bad in (
            {}, {'enabled': 'yes'}, {'enabled': 1}, {'enabled': True, 'keep_core_layers': -1},
            {'enabled': True, 'keep_core_layers': 13}, {'enabled': True, 'keep_core_layers': True},
            {'enabled': True, 'timeout_min': -5}, {'enabled': True, 'timeout_min': True},
            {'enabled': True, 'chunk': 0}, {'enabled': True, 'margin_mb': -1},
            {'enabled': True, 'ram_experts': 'invalid'}, {'enabled': True, 'ram_experts': 123},
            {'enabled': True, 'ram_experts': None}, {'enabled': True, 'ram_experts': True},
            {'enabled': True, 'ram_experts': []},
        ):
            rb = client.post('/admin/lite', json=bad)
            assert rb.status_code == 400, (bad, rb.status_code, rb.text)

        # 3a2. Valid ram_experts values via POST/GET /admin/lite
        for req_body, exp_mode, exp_ram, exp_off in (
            ({'enabled': True, 'ram_experts': 'core', 'timeout_min': 0}, 'core', 12, 24),
            ({'enabled': True, 'keep_core_layers': 4, 'ram_experts': 'core', 'timeout_min': 0}, 'core', 8, 20),
            ({'enabled': True, 'keep_core_layers': 4, 'ram_experts': 'all', 'timeout_min': 0}, 'all', 20, 20),
            ({'enabled': True, 'ram_experts': 'mmap', 'timeout_min': 0}, 'mmap', 0, 24),
        ):
            rr = client.post('/admin/lite', json=req_body)
            assert rr.status_code == 200, (req_body, rr.text)
            rd = rr.json()
            assert rd['ram_experts'] == exp_mode and rd['ram_layers'] == exp_ram and rd['offloaded_layers'] == exp_off, rd
            rg = client.get('/admin/lite').json()
            assert rg['ram_experts'] == exp_mode and rg['ram_layers'] == exp_ram, rg
        rx = client.post('/admin/lite', json={'enabled': False})
        assert rx.status_code == 200 and rx.json()['ram_experts'] == 'mmap' and rx.json()['ram_layers'] == 0

        # 3b. Active decode drains cleanly, queued waiter waits until after enter_lite, /health stays responsive
        out = {}
        lite_res = {}
        health_ok = []
        t_dec = threading.Thread(target=ask, args=('dec1', 1, 16, out))
        t_dec.start()
        time.sleep(0.10)  # dec1 is now decoding

        def do_enter():
            lite_res['r'] = client.post('/admin/lite', json={'enabled': True, 'keep_core_layers': 3, 'timeout_min': 0})

        t_lite = threading.Thread(target=do_enter)
        t_lite.start()
        time.sleep(0.05)  # transition is pending while dec1 drains
        t_wait = threading.Thread(target=ask, args=('wait1', 1, 3, out))
        t_wait.start()
        rh = client.get('/health')
        health_ok.append(rh.status_code == 200 and rh.json()['ok'])

        t_dec.join(); t_lite.join(); t_wait.join()
        assert health_ok == [True]
        assert lite_res['r'].status_code == 200 and lite_res['r'].json()['lite'] is True
        assert out['dec1'] == ''.join(f'dec1{i} ' for i in range(16)), out
        assert out['wait1'] == ''.join(f'wait1{i} ' for i in range(3)), out
        # wait1 must have started AFTER enter_lite
        idx_enter = next(i for i, e in enumerate(eng.log) if e[0] == 'enter_lite')
        idx_wait = next(i for i, e in enumerate(eng.log) if e[:2] == ('start', 'wait1'))
        assert idx_wait > idx_enter, eng.log

        # 3c. Enforces 1-at-a-time concurrency in lite mode
        eng.steps.clear()
        out2 = {}
        ts = [threading.Thread(target=ask, args=(f'c{i}', 1, 6, out2, i * 0.01)) for i in range(3)]
        for t in ts: t.start()
        for t in ts: t.join()
        assert all(len(s) == 1 for s in eng.steps), f'lite mode ran >1 slot: {eng.steps}'

        # 3d. /metrics reflects lite_mode=1 and offloaded_layers=21
        mt = client.get('/metrics').text
        assert 'inkling_lite_mode 1' in mt and 'inkling_offloaded_layers 21' in mt, mt

        # 3e. Exit OOM -> 503, stays in lite mode; retry succeeds -> 200 & metrics back to 0
        eng.fail_exit_count = 1
        ro = client.post('/admin/lite', json={'enabled': False})
        assert ro.status_code == 503 and eng.lite is True, (ro.status_code, ro.text)
        ro2 = client.post('/admin/lite', json={'enabled': False})
        assert ro2.status_code == 200 and ro2.json()['lite'] is False and ro2.json()['slots'] == 4
        mt2 = client.get('/metrics').text
        assert 'inkling_lite_mode 0' in mt2 and 'inkling_offloaded_layers 0' in mt2, mt2

        # 3f. Active prefill is preempted cleanly on enter_lite and resumes in lite mode
        eng.log.clear()
        out3 = {}
        t_pf = threading.Thread(target=ask, args=('pf1', 12, 2, out3))
        t_pf.start()
        time.sleep(0.10)  # pf1 is mid-prefill
        re = client.post('/admin/lite', json={'enabled': True, 'timeout_min': 0})
        assert re.status_code == 200 and re.json()['lite'] is True
        t_pf.join()
        assert out3['pf1'] == 'pf10 pf11 ', out3
        assert eng.has('preempted', 'pf1'), eng.log
        idx_pre = next(i for i, e in enumerate(eng.log) if e[:2] == ('preempted', 'pf1'))
        idx_ent = next(i for i, e in enumerate(eng.log) if e[0] == 'enter_lite')
        assert idx_pre < idx_ent, eng.log
        assert client.post('/admin/lite', json={'enabled': False}).status_code == 200

        # 3f2. Active decode in lite mode is paused on exit_lite and resumes seamlessly in full mode (incl. OOM retry)
        assert client.post('/admin/lite', json={'enabled': True, 'timeout_min': 0}).status_code == 200
        eng.log.clear()
        out3b = {}
        t_ld = threading.Thread(target=ask, args=('ldec', 1, 24, out3b))
        t_ld.start()
        time.sleep(0.10)  # ldec is mid-decode in lite mode
        eng.fail_exit_count = 1
        r_oom = client.post('/admin/lite', json={'enabled': False})
        assert r_oom.status_code == 503 and eng.lite is True
        time.sleep(0.08)  # ldec re-admitted in lite mode after OOM rollback
        r_ok = client.post('/admin/lite', json={'enabled': False})
        assert r_ok.status_code == 200 and r_ok.json()['lite'] is False
        t_ld.join()
        assert out3b['ldec'] == ''.join(f'ldec{i} ' for i in range(24)), out3b
        idx_park = next(i for i, e in enumerate(eng.log) if e[:2] == ('parked', 'ldec'))
        idx_exit = next(i for i, e in enumerate(eng.log) if e[0] == 'exit_lite')
        idx_resume = next(i for i in range(idx_exit + 1, len(eng.log)) if eng.log[i][:2] == ('start', 'ldec'))
        assert idx_park < idx_exit < idx_resume, eng.log

        # 3f3. No-op exit_lite when already in full mode does not preempt an active decode
        eng.log.clear()
        out3c = {}
        t_fd = threading.Thread(target=ask, args=('fdec', 1, 12, out3c))
        t_fd.start()
        time.sleep(0.08)
        r_noop = client.post('/admin/lite', json={'enabled': False})
        assert r_noop.status_code == 200 and r_noop.json()['lite'] is False
        t_fd.join()
        assert out3c['fdec'] == ''.join(f'fdec{i} ' for i in range(12)), out3c
        assert not eng.has('parked', 'fdec'), eng.log

        # 3g. Watchdog auto-exit (including OOM retry)
        server.LITE_RETRY_S = 0.15
        eng.fail_exit_count = 1
        rw = client.post('/admin/lite', json={'enabled': True, 'timeout_min': 0.004})  # ~0.24s
        assert rw.status_code == 200 and rw.json()['timeout_at'] is not None
        deadline = time.time() + 3.0
        while time.time() < deadline and eng.lite:
            time.sleep(0.05)
        assert not eng.lite, 'watchdog did not auto-exit after OOM retry'
        assert eng.has('exit_lite_oom') and eng.has('exit_lite')
        assert client.get('/admin/lite').json()['timeout_at'] is None

        print('PASS 3: Scheduler + /admin/lite drain/preempt/1-slot/OOM-503/watchdog/metrics')
    finally:
        server.sched._cancel_watchdog()
        server.eng, server.sched, server.LITE_RETRY_S = old_eng, old_sched, old_retry


# ===================================================================== 4. Fused Lite Mode (resident / dma / stage)
def test_route_top8_kernel():
    """_route_top8_kernel matches _w13_kernel's top-8 selection bit-for-bit, including tie breaking (lowest index)."""
    torch.manual_seed(101)
    H, NI = 64, 32
    u = torch.randn(H, dtype=torch.bfloat16)
    w13 = torch.randint(0, 256, (128, 2 * NI, H // 2), dtype=torch.uint8)
    s13 = torch.ones((128, 2 * NI, H // 16), dtype=torch.float8_e4m3fn)
    gs13 = torch.ones(128, dtype=torch.float32)

    # Case 1: random scores + biases
    s_rand = torch.sigmoid(torch.randn(128))
    b_rand = torch.randn(128) * 0.2

    # Case 2: deliberate ties right across the top-8 cutoff (slots 4..11 all tied at score 2.0)
    s_tie = torch.linspace(0.1, 0.9, 128)
    b_tie = torch.zeros(128)
    for e in (5, 19, 40, 77):
        b_tie[e] = 5.0 - s_tie[e]  # top 4 all tied at 5.0 -> expect [5, 19, 40, 77]
    for e in (100, 3, 64, 12, 88, 25, 1, 99):
        b_tie[e] = 2.0 - s_tie[e]  # next 8 all tied at 2.0 -> lowest 4 win: [1, 3, 12, 25]

    # Case 3: all 128 scores identical -> expect [0, 1, 2, 3, 4, 5, 6, 7]
    s_all = torch.full((128,), 0.5)
    b_all = torch.zeros(128)

    for label, s, b, exp_ids in (
        ('random', s_rand, b_rand, None),
        ('ties', s_tie, b_tie, [5, 19, 40, 77, 1, 3, 12, 25]),
        ('all_equal', s_all, b_all, list(range(8))),
    ):
        idx_r, wt_r = torch.zeros(8, dtype=torch.int32), torch.zeros(8)
        idx_w, wt_w = torch.zeros(8, dtype=torch.int32), torch.zeros(8)
        act = torch.zeros(8, NI, dtype=torch.bfloat16)
        DK._route_top8_kernel[(1,)](s, b, idx_r, wt_r)
        DK._w13_kernel[(1, 8)](s, b, u, w13, s13, gs13, w13.stride(0), w13.stride(1), s13.stride(0), s13.stride(1),
                               act, idx_w, wt_w, FP4=True, F32=True, H=H, NI=NI, BN=NI, BK=H)
        assert torch.equal(idx_r, idx_w), (label, idx_r, idx_w)
        assert torch.equal(wt_r, wt_w), (label, wt_r, wt_w)
        if exp_ids is not None:
            assert idx_r.tolist() == exp_ids, (label, idx_r.tolist(), exp_ids)
    print('PASS 4a: _route_top8_kernel == _w13_kernel bit-exact (random, boundary ties, all-equal)')


class _FakeHostAllocRT:
    """CPU test mock for cudaHostAlloc(..., 0) / cudaFreeHost backed by real ctypes buffers."""
    def __init__(self):
        self.active = {}       # ptr -> (buf, nbytes)
        self.alloc_calls = 0
        self.freed = []
        self.fail_on_call = None

    def cudaHostAlloc(self, p_ref, nbytes_v, flags_v):
        self.alloc_calls += 1
        nb, fl = int(nbytes_v.value), int(flags_v.value)
        assert fl == 0 and nb > 0, (nb, fl)
        if self.fail_on_call is not None and self.alloc_calls == self.fail_on_call:
            return 2  # cudaErrorMemoryAllocation
        buf = ctypes.create_string_buffer(nb)
        ptr = ctypes.addressof(buf)
        assert ptr not in self.active, f'duplicate alloc addr {ptr:#x}'
        self.active[ptr] = (buf, nb)
        p_ref._obj.value = ptr
        return 0

    def cudaFreeHost(self, ptr_v):
        ptr = int(ptr_v.value)
        assert ptr in self.active, f'double/invalid cudaFreeHost of {ptr:#x}'
        del self.active[ptr]
        self.freed.append(ptr)
        return 0


def test_weights_pinned_lifecycle_and_dma():
    """Weights pinned_layers lifecycle via cudaHostAlloc(..., 0) / cudaFreeHost (exact-size allocation, view/slice
    lifetime, partial allocation failure rollback, OOM rollback, 0 leaked allocations) and stage_experts_dma."""
    torch.manual_seed(13)
    tmp = tempfile.mkdtemp(prefix='ink-lite-pin-')
    try:
        st_path = os.path.join(tmp, 'model-00001-of-00001.safetensors')
        raw = _make_moe_tensors((0, 6, 7, 20), E=16, H=64, NI=32)
        save_file({'model.llm.' + k: v for k, v in raw.items()}, st_path)

        W = object.__new__(M.Weights)
        W.dev, W.e2m1, W.f_path = torch.device('cpu'), torch.tensor(M.E2M1), st_path
        W.f = safe_open(st_path, 'pt', device='cpu')
        W.t = {k[len('model.llm.'):]: W.f.get_tensor(k) for k in W.f.keys()}
        W.offloaded, W.ram_layers, W.pinned_layers = set(), set(), set()
        W._stage, W._stage_ev, W._stage_slot, W._mm = {}, [None, None], 0, None

        # On CPU, _offload_layer_ram leaves pinned_layers empty
        W.offload_experts(ram_layers={6, 7, 20})
        assert W.ram_layers == {6, 7, 20} and W.pinned_layers == set()

        # stage_experts_dma == stage_experts on both NVFP4 (L=6) and MXFP8 (L=20)
        for L in (6, 20):
            sub_e = [15, 2, 11, 0, 7, 4, 9, 13]
            st_a = W.stage_experts(L, sub_e, cap=8)
            st_b = W.stage_experts_dma(L, sub_e, cap=8)
            assert len(st_a) == len(st_b) == 4
            for a, b in zip(st_a, st_b):
                assert torch.equal(a, b), (L, a.shape, b.shape)
        W.reload_experts()
        assert W.offloaded == set() and W.ram_layers == set() and W.pinned_layers == set()

        # Simulated CUDA cudaHostAlloc / cudaFreeHost lifecycle (no GPU needed)
        fake_rt = _FakeHostAllocRT()
        orig_cudart = M._CUDART
        orig_load = W._load_expert_tensor
        old_cuda_empty = torch.cuda.empty_cache
        cache_releases = [0]

        W.dev = SimpleNamespace(type='cuda')
        W._load_expert_tensor = lambda n: W.t[n].clone() if isinstance(W.t[n], torch.Tensor) else W.f.get_tensor('model.llm.' + n)
        W._release_pinned_cache = lambda: (gc.collect(), cache_releases.__setitem__(0, cache_releases[0] + 1))
        M._CUDART = fake_rt
        torch.cuda.empty_cache = lambda: None
        try:
            # 1) Direct _alloc_host_pinned slice/view lifetime: buffer stays alive while a slice or view exists,
            #    and cudaFreeHost fires immediately once the last view is dropped.
            t_test = M._alloc_host_pinned((4, 8, 8), torch.float8_e4m3fn, fake_rt)
            p_test = int(t_test.data_ptr())
            assert p_test in fake_rt.active and fake_rt.active[p_test][1] == 4 * 8 * 8
            sl = t_test[1]
            vw = sl.view(torch.uint8)
            del t_test
            gc.collect()
            assert p_test in fake_rt.active, 'cudaFreeHost fired while slice/view still alive!'
            del sl
            gc.collect()
            assert p_test in fake_rt.active, 'cudaFreeHost fired while view still alive!'
            del vw
            gc.collect()
            assert p_test not in fake_rt.active and p_test in fake_rt.freed, 'cudaFreeHost did not fire after last view'
            fake_rt.alloc_calls = 0
            fake_rt.freed.clear()

            # 2) Pin layers 6, 7, 20: layer 7 fails on its 2nd tensor (call 6) -> 1st tensor of layer 7 MUST be
            #    freed immediately via cudaFreeHost, layer 7 falls back to unpinned RAM, layers 6 & 20 are pinned!
            fake_rt.fail_on_call = 6
            W.offload_experts(ram_layers={6, 7, 20})
            assert W.offloaded == {0, 6, 7, 20}
            assert W.ram_layers == {6, 7, 20}
            assert W.pinned_layers == {6, 20}, W.pinned_layers
            assert len(fake_rt.freed) == 1, 'partial alloc failure must immediately free earlier tensor of L=7'
            assert len(fake_rt.active) == 8  # 4 exact-size tensors each for layers 6 and 20
            for L in (6, 20):
                for suf in M.Weights.LAZY:
                    k = f'layers.{L}.mlp.experts.{suf}'
                    t = W.t[k]
                    assert torch.equal(t, raw[k])
                    assert fake_rt.active[int(t.data_ptr())][1] == t.numel() * t.element_size()
            del t
            for suf in M.Weights.LAZY:
                k = f'layers.7.mlp.experts.{suf}'
                assert torch.equal(W.t[k], raw[k]) and int(W.t[k].data_ptr()) not in fake_rt.active

            # 3) OOM mid-reload preserves pinned_layers (6, 20 pinned; 7 unpinned RAM; 0 mmap) with 0 leaked allocs
            fake_rt.fail_on_call = None
            real_l = W._load_expert_tensor

            def fail_on_20(n):
                if n.startswith('layers.20.'):
                    raise torch.OutOfMemoryError('simulated reload OOM on layer 20')
                return real_l(n)

            W._load_expert_tensor = fail_on_20
            try:
                W.reload_experts()
                raise AssertionError('expected OutOfMemoryError')
            except torch.OutOfMemoryError:
                pass
            finally:
                W._load_expert_tensor = real_l
            gc.collect()
            assert W.offloaded == {0, 6, 7, 20} and W.ram_layers == {6, 7, 20} and W.pinned_layers == {6, 20}
            assert len(fake_rt.active) == 8

            # 4) Switching pinned layers back to mmap frees all cudaHostAlloc buffers immediately
            before = cache_releases[0]
            W.offload_experts(ram_layers=())
            assert W.ram_layers == set() and W.pinned_layers == set()
            assert fake_rt.active == {}, f'leaked cudaHostAlloc buffers: {list(fake_rt.active)}'
            assert cache_releases[0] > before

            # 5) Re-pin, stage a fake DMA buffer in _stage, verify reload_experts frees all allocs & clears _stage
            W.offload_experts(ram_layers={6, 20})
            assert W.pinned_layers == {6, 20} and len(fake_rt.active) == 8
            W._stage[('dma', 'w13_weight', (64, 32), torch.uint8, 8)] = (None, torch.empty(8, 64, 32, dtype=torch.uint8))
            before = cache_releases[0]
            W.reload_experts()
            assert W.offloaded == set() and W.ram_layers == set() and W.pinned_layers == set()
            assert fake_rt.active == {} and W._stage == {} and cache_releases[0] > before
        finally:
            M._CUDART = orig_cudart
            torch.cuda.empty_cache = old_cuda_empty
            W.dev = torch.device('cpu')
            W._load_expert_tensor = orig_load
        print('PASS 4b: Weights cudaHostAlloc/cudaFreeHost lifecycle (exact size/slice lifetime/partial fallback/OOM rollback) + stage_experts_dma')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _make_synthetic_fused_model(layers_keep=(), layers_pinned=(), H=None, NI=None):
    """Build a synthetic Model + DecodeState + Fused with dimensions (H, NI) and E=128 experts."""
    torch.manual_seed(202)
    H = M.H if H is None else H
    NI = DK.NI if NI is None else NI
    W = object.__new__(M.Weights)
    W.dev, W.e2m1, W.f, W.f_path = torch.device('cpu'), torch.tensor(M.E2M1), None, None
    W.offloaded = set(range(24)) - set(layers_keep)
    W.ram_layers = set(layers_pinned) - set(layers_keep)
    W.pinned_layers = set(layers_pinned) - set(layers_keep)
    W._stage, W._stage_ev, W._stage_slot, W._mm = {}, [None, None], 0, None

    W.t = {}
    W.proj, W.gcat = {}, {}
    cat_slide = torch.zeros(6552 + 2048, H, dtype=torch.bfloat16)
    cat_full = torch.zeros(6552 + 2112, H, dtype=torch.bfloat16)
    z_h = torch.ones(H, dtype=torch.float32)
    z_q = torch.zeros(H, dtype=torch.float32)
    z_q3 = torch.zeros(3, H, dtype=torch.float32)
    z_wo = torch.zeros(H, 3072, dtype=torch.bfloat16)
    z_sc_h = torch.randn(H, 1, 4, dtype=torch.float32) * 0.1
    z_sc_kv = torch.zeros(1024, 1, 4, dtype=torch.float32)
    z_sc_dyn = torch.zeros(4, 1024, dtype=torch.float32)
    z_qsc = torch.ones(M.NQ, M.HD, dtype=torch.float32)
    z_rsc = torch.ones(M.NQ, 16, dtype=torch.float32)
    z_rel = torch.zeros(16, M.WIN, dtype=torch.bfloat16)
    z_down = torch.zeros(64, 2 * H, dtype=torch.bfloat16)
    z_gcat = torch.zeros(2 * H, 64, dtype=torch.float32)
    W.t['norm.weight'] = z_h
    W.t['embed_norm.weight'] = z_h
    W.t['attn_res.output_query'] = z_q

    fp4_sh = (
        torch.randint(0, 256, (128, 2 * NI, H // 2), dtype=torch.uint8),
        (torch.rand(128, 2 * NI, H // 16) * 48 + 16).to(torch.float8_e4m3fn),
        torch.randint(0, 256, (128, H, NI // 2), dtype=torch.uint8),
        (torch.rand(128, H, NI // 16) * 48 + 16).to(torch.float8_e4m3fn),
        torch.rand(128) + 0.5,
        torch.tensor([2688.0]),
        torch.rand(128) + 0.5,
        torch.tensor([2688.0]),
    )
    mx_sh = (
        torch.randn(128, 2 * NI, H).to(torch.float8_e4m3fn),
        torch.randint(118, 124, (128, 2 * NI, H // 32), dtype=torch.uint8),
        torch.randn(128, H, NI).to(torch.float8_e4m3fn),
        torch.randint(118, 124, (128, H, NI // 32), dtype=torch.uint8),
    )

    for L in range(24):
        P, A = f'layers.{L}.', f'layers.{L}.attn.'
        full = L in M.FULL
        if L:
            W.t[f'attn_res.attn_queries.{L - 1}'] = z_q3 if full else z_q
        W.t[f'attn_res.mlp_queries.{L}'] = z_q
        if full:
            for c in 'qkv':
                W.t[P + f'{c}_input_norm.weight'] = z_h
            W.proj[L] = (cat_full, [6552, 1056, 1056])
            W.t[A + 'k_sconv.static_w'] = z_sc_dyn
            W.t[A + 'v_sconv.static_w'] = z_sc_dyn
        else:
            W.t[P + 'attn_norm.weight'] = z_h
            W.proj[L] = (cat_slide, [6552, 1024, 1024])
            W.t[A + 'k_sconv.weight'] = z_sc_kv
            W.t[A + 'v_sconv.weight'] = z_sc_kv
        if L in M.CORE:
            for c in ('attn', 'k', 'v', 'mlp'):
                cp = f'layers.{L}.controllers.{c}.'
                W.t[cp + 'down_weight'] = z_down
                W.gcat[cp] = z_gcat
        W.t[A + 'q_l2_clip.scale'] = z_qsc
        W.t[A + 'rel_l2_clip.scale'] = z_rsc
        W.t[A + 'rel_logits_proj.proj'] = z_rel
        W.t[A + 'wo_ud.weight'] = z_wo
        W.t[P + 'attn_sconv.weight'] = z_sc_h
        W.t[P + 'mlp_sconv.weight'] = z_sc_h
        W.t[P + 'mlp_norm.weight'] = z_h
        W.t[P + 'mlp.gate.weight'] = torch.zeros(128, H, dtype=torch.float32)
        W.t[P + 'mlp.gate.bias'] = torch.randn(128, dtype=torch.float32) * 0.2
        E = P + 'mlp.experts.'
        if L < 20:
            W.t[E + 'w13_weight'] = fp4_sh[0]
            W.t[E + 'w13_weight_scale'] = fp4_sh[1]
            W.t[E + 'w2_weight'] = fp4_sh[2]
            W.t[E + 'w2_weight_scale'] = fp4_sh[3]
            W.t[E + 'w13_weight_amax'] = fp4_sh[4]
            W.t[E + 'w13_weight_outer_scale_denominator'] = fp4_sh[5]
            W.t[E + 'w2_weight_amax'] = fp4_sh[6]
            W.t[E + 'w2_weight_outer_scale_denominator'] = fp4_sh[7]
        else:
            W.t[E + 'w13_weight'] = mx_sh[0]
            W.t[E + 'w13_weight_scale'] = mx_sh[1]
            W.t[E + 'w2_weight'] = mx_sh[2]
            W.t[E + 'w2_weight_scale'] = mx_sh[3]

    m = object.__new__(M.Model)
    m.W, m.dev, m.d8 = W, torch.device('cpu'), None
    m.slot = {v: i for i, v in enumerate(M.FULL_VISITS)}
    m.mq = [torch.zeros(4, H, dtype=torch.float32) for _ in range(12)]
    kv = M.KV4(len(M.FULL_VISITS), 256, 'cpu')
    st = decode.DecodeState(torch.device('cpu'), kv)
    if H != M.H:
        for c in st.conv:
            c['attn'] = torch.zeros(3, H, dtype=torch.bfloat16)
            c['mlp'] = torch.zeros(3, H, dtype=torch.bfloat16)
    x_raw = torch.zeros(1, H, dtype=torch.bfloat16)
    out = torch.zeros(1, 16, dtype=torch.float32)
    return m, st, DK.Fused(m, st, x_raw, out)


def test_fused_moe_modes_and_mixed_visit():
    """Fused MoE execution across resident / dma / stage on both NVFP4 (L=7/0) and MXFP8 (L=20/21) layers,
    with both w2all=False and w2all=True, is bit-exact (max_abs_diff == 0). Also verifies full-size (H=2304, NI=1152)
    mixed-layer Fused initialization, PF disabling on offloaded layers, and exact staging method dispatch."""
    old_blk, old_h, old_ni = dict(DK.BLK), DK.H, DK.NI
    try:
        # 1) Full-size (H=2304, NI=1152) Fused.__init__ check: per-visit moe_mode, d['ex'], d['pf']['w13'], buffers
        _, _, f_full_res = _make_synthetic_fused_model(layers_keep=range(24), layers_pinned=())
        _, _, f_full_mix = _make_synthetic_fused_model(layers_keep=(6,), layers_pinned=(7, 20))
        for d in f_full_res.vis:
            assert d['moe_mode'] == 'resident' and d['ex'] is not None
        assert not hasattr(f_full_res, 'idx07')
        assert hasattr(f_full_mix, 'idx07') and f_full_mix.idx07.tolist() == list(range(8))
        assert f_full_mix.gs13_8.shape == (8,) and f_full_mix.gs2_8.shape == (8,)
        for d in f_full_mix.vis:
            L = d['L']
            if L == 6:
                assert d['moe_mode'] == 'resident' and d['ex'] is not None
            elif L in (7, 20):
                assert d['moe_mode'] == 'dma' and d['ex'] is None and d['pf']['w13'] == {}
            else:
                assert d['moe_mode'] == 'stage' and d['ex'] is None and d['pf']['w13'] == {}
        del f_full_res, f_full_mix

        # 2) Compact-dim (H=256, NI=128) real Fused.visit() MoE execution via Triton CPU interpreter
        DK.H, DK.NI = 256, 128
        DK.BLK.update(w13=128, w2=256, w2a=256, w2f=256, kbk=256, kbk2=128)

        m_res, st_res, f_res = _make_synthetic_fused_model(layers_keep=range(24), layers_pinned=(), H=256, NI=128)
        m_mix, st_mix, f_mix = _make_synthetic_fused_model(layers_keep=(6,), layers_pinned=(7, 20), H=256, NI=128)

        g = torch.Generator().manual_seed(303)
        S_in = torch.sigmoid(torch.randn(128, generator=g))
        U_in = M.rms(torch.randn(256, generator=g)).bfloat16()
        R_in = torch.randn(256, generator=g) * 0.1
        conv0 = torch.randn(3, 256, generator=g).bfloat16()

        calls = {'stage': [], 'dma': []}
        orig_st, orig_dm = m_mix.W.stage_experts, m_mix.W.stage_experts_dma
        m_mix.W.stage_experts = lambda L, ids, cap=None: (calls['stage'].append(L), orig_st(L, ids, cap))[1]
        m_mix.W.stage_experts_dma = lambda L, ids, cap=None: (calls['dma'].append(L), orig_dm(L, ids, cap))[1]

        def run_moe_only(f, v, w2all):
            f.w2all = w2all
            f.S.copy_(S_in)
            f.U.copy_(U_in)
            f.R.copy_(R_in)
            f.st.conv[v]['mlp'].copy_(conv0)
            f.hbuf[v].zero_()
            d = f.vis[v]
            saved = {}
            noop = lambda *a, **k: None

            class NoopGrid:
                def __getitem__(self, _):
                    return noop

            for name in ('_mix_kernel', '_gate_z_kernel', '_gate_out_kernel', '_proj_kernel', '_pre_kernel',
                         '_wo_kernel', '_router_kernel'):
                saved[name] = getattr(DK, name)
                setattr(DK, name, NoopGrid())
            orig_near = f.near
            f.near = noop
            try:
                f.visit(v, d)
            finally:
                for name, fn in saved.items():
                    setattr(DK, name, fn)
                f.near = orig_near
            return f.hbuf[v].clone(), f.st.conv[v]['mlp'].clone(), f.idx.clone(), f.wt.clone()

        # Visit 6 (L=6, resident NVFP4), 7 (L=7, dma NVFP4), 44 (L=20, dma MXFP8),
        # 0 (L=0, stage NVFP4), 45 (L=21, stage MXFP8)
        for w2all in (False, True):
            for v, label in ((6, 'NVFP4_L6_res'), (7, 'NVFP4_L7_dma'), (44, 'MXFP8_L20_dma'),
                             (0, 'NVFP4_L0_stage'), (45, 'MXFP8_L21_stage')):
                h_ref, c_ref, i_ref, w_ref = run_moe_only(f_res, v, w2all)
                h_got, c_got, i_got, w_got = run_moe_only(f_mix, v, w2all)
                assert torch.equal(h_ref, h_got) and torch.equal(c_ref, c_got), (w2all, label, 'h/conv')
                assert torch.equal(i_ref, i_got) and torch.equal(w_ref, w_got), (w2all, label, 'idx/wt')

        assert calls['dma'] == [7, 20, 7, 20], calls
        assert calls['stage'] == [0, 21, 0, 21], calls
        print('PASS 4c: Fused.visit MoE bit-exact (resident == dma == stage, NVFP4 & MXFP8, w2all False/True)')
    finally:
        DK.H, DK.NI = old_h, old_ni
        DK.BLK.clear()
        DK.BLK.update(old_blk)


def test_engine_fused_lite_reconfig_and_leaks():
    """Engine.enter_lite on CUDA uses fused LoopBatch(dec=True, graph=False, fuse=True) + spec='off', drops old
    Fused/Decoder refs BEFORE offloading on every reconfiguration (core <-> all <-> mmap) and on exit_lite
    (0 weakref / cudaHostAlloc leaks), and rolls back cleanly on exit_lite OOM."""
    tmp = tempfile.mkdtemp(prefix='ink-lite-fused-eng-')
    old_cuda_empty = torch.cuda.empty_cache
    old_cudart = M._CUDART
    try:
        st_path = os.path.join(tmp, 'model-00001-of-00001.safetensors')
        tensors = {}
        for L in range(24):
            p = f'model.llm.layers.{L}.mlp.experts.'
            tensors[p + 'w13_weight'] = torch.randint(0, 255, (4, 8, 8), dtype=torch.uint8)
            tensors[p + 'w13_weight_scale'] = torch.ones((4, 8, 1), dtype=torch.float8_e4m3fn)
            tensors[p + 'w2_weight'] = torch.randint(0, 255, (4, 8, 4), dtype=torch.uint8)
            tensors[p + 'w2_weight_scale'] = torch.ones((4, 8, 1), dtype=torch.float8_e4m3fn)
        save_file(tensors, st_path)

        fake_rt = _FakeHostAllocRT()

        W = object.__new__(M.Weights)
        W.dev, W.e2m1, W.f_path = SimpleNamespace(type='cuda'), torch.tensor(M.E2M1), st_path
        W.f = safe_open(st_path, 'pt', device='cpu')
        W.t = {k[len('model.llm.'):]: W.f.get_tensor(k) for k in W.f.keys()}
        W.offloaded, W.ram_layers, W.pinned_layers = set(), set(), set()
        W._stage, W._stage_ev, W._stage_slot, W._mm = {}, [None, None], 0, None
        W._load_expert_tensor = lambda n: W.t[n].clone() if isinstance(W.t[n], torch.Tensor) else W.f.get_tensor('model.llm.' + n)
        W._release_pinned_cache = lambda: gc.collect()

        M._CUDART = fake_rt
        torch.cuda.empty_cache = lambda: None

        eng = _make_mini_engine(tmp)
        eng.model.dev = W.dev
        eng.model.W = W

        r = eng.enter_lite(ram_experts='core')
        assert r['lite'] is True and r['decoder'] == 'fused' and r['spec'] == 'off'
        assert r['pinned_layers'] == 12 and r['ram_layers'] == 12 and len(fake_rt.active) == 48
        assert isinstance(eng.bd, LoopBatch) and eng.bd.dec is True and eng.bd.graph is False and eng.bd.fuse is True

        class _Obj:
            def __init__(self, **kw):
                self.__dict__.update(kw)

        # Attach a fake warm Decoder + Fused holding refs to pinned/resident experts
        fake_fused = _Obj(vis=[
            {'ex': [W.t[f'layers.{L}.mlp.experts.{suf}'] for suf in M.Weights.LAZY]}
            for L in range(6, 18)
        ])
        fake_dec = _Obj(fused=fake_fused, seq=None)
        eng.bd.decs[0] = fake_dec
        dec_ref = weakref.ref(fake_dec)
        fused_ref = weakref.ref(fake_fused)
        pin_refs = [weakref.ref(W.t[f'layers.{L}.mlp.experts.{suf}']) for L in range(6, 18) for suf in M.Weights.LAZY]
        del fake_dec, fake_fused

        # Reconfigure core -> mmap: old Decoder/Fused & pinned cudaHostAlloc buffers MUST be freed (0 leaks)!
        r = eng.enter_lite(ram_experts='mmap')
        gc.collect()
        assert r['pinned_layers'] == 0 and r['ram_layers'] == 0 and r['decoder'] == 'fused'
        assert fake_rt.active == {}, f'reconfigure leaked cudaHostAlloc buffers: {list(fake_rt.active)}'
        assert dec_ref() is None and fused_ref() is None, 'reconfigure leaked old lite Decoder/Fused'
        assert all(pr() is None for pr in pin_refs), 'reconfigure leaked pinned host expert tensors'

        # Reconfigure to all, attach fake warm Decoder/Fused, test exit_lite OOM rollback then clean exit
        r = eng.enter_lite(ram_experts='all')
        assert r['pinned_layers'] == 24 and len(fake_rt.active) == 96
        fake_fused2 = _Obj(vis=[
            {'ex': [W.t[f'layers.{L}.mlp.experts.{suf}'] for suf in M.Weights.LAZY]}
            for L in range(24)
        ])
        fake_dec2 = _Obj(fused=fake_fused2, seq=None)
        eng.bd.decs[0] = fake_dec2
        dec2_ref, fused2_ref = weakref.ref(fake_dec2), weakref.ref(fake_fused2)
        del fake_dec2, fake_fused2

        # OOM during _build_bd rolls back to fused LoopBatch with 24 pinned layers
        real_build = eng._build_bd
        eng._build_bd = lambda spec: (_ for _ in ()).throw(torch.OutOfMemoryError('build OOM'))
        try:
            eng.exit_lite()
            raise AssertionError('expected OutOfMemoryError')
        except torch.OutOfMemoryError:
            pass
        finally:
            eng._build_bd = real_build
        gc.collect()
        assert dec2_ref() is None and fused2_ref() is None, 'exit_lite did not drop old lite Decoder/Fused before reload'
        assert eng.lite is True and len(W.pinned_layers) == 24 and len(fake_rt.active) == 96
        assert isinstance(eng.bd, LoopBatch) and eng.bd.dec is True and eng.bd.fuse is True

        # Clean exit frees all pinned refs & cudaHostAlloc buffers and restores full mode
        pin_refs2 = [weakref.ref(W.t[f'layers.{L}.mlp.experts.{suf}']) for L in range(24) for suf in M.Weights.LAZY]
        ro = eng.exit_lite()
        gc.collect()
        assert ro['lite'] is False and ro['pinned_layers'] == 0
        assert ro['decoder'] == 'batch' and ro['slots'] == 4 and ro['spec'] == 'dspark'
        assert fake_rt.active == {}, f'exit_lite leaked cudaHostAlloc buffers: {list(fake_rt.active)}'
        assert all(pr() is None for pr in pin_refs2), 'exit_lite leaked pinned host expert tensors'
        print('PASS 4d: Engine fused lite reconfiguration (core/all/mmap), 0 weakref/alloc leaks, OOM rollback')
    finally:
        M._CUDART = old_cudart
        torch.cuda.empty_cache = old_cuda_empty
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    test_weights_offload_and_reload()
    test_weights_ram_experts()
    test_streamed_moe_numerics()
    test_engine_lite_lifecycle()
    test_no_expert_ref_leak()
    test_engine_ram_experts_transitions()
    test_scheduler_and_http_lite()
    test_route_top8_kernel()
    test_weights_pinned_lifecycle_and_dma()
    test_fused_moe_modes_and_mixed_visit()
    test_engine_fused_lite_reconfig_and_leaks()
    print('ALL LITE TESTS OK')
