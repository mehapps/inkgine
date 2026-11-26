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
        W.offloaded, W._stage, W._stage_ev, W._stage_slot, W._mm = set(), {}, [None, None], 0, None

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


def test_streamed_moe_numerics():
    torch.manual_seed(42)
    tmp = tempfile.mkdtemp(prefix='ink-lite-moe-')
    old_batch = M.LITE_EXPERT_BATCH
    old_v2 = moe_triton.V2
    try:
        E = 24
        M.LITE_EXPERT_BATCH = 5  # forces 5 batches (5, 5, 5, 5, 4) across 24 experts
        raw = _make_moe_tensors((0, 20), E)
        st_path = os.path.join(tmp, 'model-00001-of-00001.safetensors')
        save_file({'model.llm.' + k: v for k, v in raw.items()}, st_path)

        m_res = object.__new__(M.Model)
        m_res.W = object.__new__(M.Weights)
        m_res.W.dev, m_res.W.e2m1, m_res.W.f = torch.device('cpu'), torch.tensor(M.E2M1), None
        m_res.W.t = {k: v.clone() for k, v in raw.items()}
        m_res.W.offloaded, m_res.W._stage, m_res.W._stage_ev = set(), {}, None

        m_off = object.__new__(M.Model)
        m_off.W = object.__new__(M.Weights)
        m_off.W.dev, m_off.W.e2m1 = torch.device('cpu'), torch.tensor(M.E2M1)
        m_off.W.f_path = st_path
        m_off.W.f = safe_open(st_path, 'pt', device='cpu')
        m_off.W.t = {k: v.clone() for k, v in raw.items()}
        m_off.W.offloaded, m_off.W._stage, m_off.W._stage_ev, m_off.W._stage_slot, m_off.W._mm = (
            set(), {}, [None, None], 0, None)
        m_off.W.offload_experts()
        assert m_off.W.offloaded == {0, 20} and m_off.W._mm is not None

        for L in (0, 20):
            for T in (1, 64):
                u = M.rms(torch.randn(T, M.H)).bfloat16()
                for impl in ('torch', 'exact', 'emulate'):
                    ref = m_res.moe(L, u, impl=impl)
                    got = m_off.moe(L, u, impl=impl)
                    assert torch.equal(ref, got), f'L={L} T={T} impl={impl}: max diff {(ref.float() - got.float()).abs().max()}'

                # Check tail=(st, conv_w, r): res_out applied once at the end
                st1 = torch.randn(3, M.H, dtype=torch.bfloat16)
                st2 = st1.clone()
                cw = torch.randn(M.H, 1, 4, dtype=torch.float32) * 0.1
                r = torch.randn(T, M.H, dtype=torch.float32)
                ref_y = m_res.moe(L, u, impl='torch')
                ref_h, ref_hn = M.res_out(ref_y, st1, cw, r)
                got_h, got_hn = m_off.moe(L, u, impl='torch', tail=(st2, cw, r))
                assert torch.equal(ref_h, got_h) and torch.equal(ref_hn, got_hn) and torch.equal(st1, st2), (
                    f'tail mismatch L={L} T={T}')

            # Check Triton interpreter path (v1 atomic & v2 fin callback) at T=1 for both NVFP4 and MXFP8
            u1 = M.rms(torch.randn(1, M.H)).bfloat16()
            for v2 in (False, True):
                moe_triton.V2 = v2
                ref_t = m_res.moe(L, u1, impl='triton')
                got_t = m_off.moe(L, u1, impl='triton')
                du = (ref_t.view(torch.int16).int() - got_t.view(torch.int16).int()).abs()
                assert du.max().item() <= 1, f'triton V2={v2} L={L} T=1 max ulp {du.max().item()}'

        print('PASS 1b: streamed Model.moe == resident (T=1 & T=64, NVFP4 & MXFP8, torch/exact/emulate/triton, tail)')
    finally:
        M.LITE_EXPERT_BATCH = old_batch
        moe_triton.V2 = old_v2
        shutil.rmtree(tmp, ignore_errors=True)


# ===================================================================== 2. Engine enter_lite / exit_lite / disk cache / lookup
class MiniWeights:
    def __init__(self):
        self.dev = torch.device('cpu')
        self.offloaded = set()
        self.fail_reload = False

    def offload_experts(self, keep_layers=()):
        keep = set(keep_layers)
        self.offloaded = set(range(24)) - keep

    def reload_experts(self):
        if self.fail_reload:
            raise torch.OutOfMemoryError('CUDA out of memory (simulated)')
        self.offloaded.clear()


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
    E.lite, E.lite_keep_core, E._full_state = False, 0, None
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
        W.offloaded, W._stage, W._stage_ev, W._stage_slot, W._mm = set(), {}, [None, None], 0, None

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


# ===================================================================== 3. Scheduler + /admin/lite + watchdog + /metrics
class LiteFakeEngine(FakeEngine):
    def __init__(self, slots=4, chunk_s=0.04, step_s=0.02):
        super().__init__(slots=slots, chunk_s=chunk_s, step_s=step_s)
        self.full_slots = slots
        self.full_chunk = self.chunk
        self.lite = False
        self.lite_keep_core = 0
        self.spec = 'dspark'
        self.model = SimpleNamespace(W=SimpleNamespace(offloaded=set()))
        self.fail_exit_count = 0

    def enter_lite(self, keep_core_layers=0, chunk=None, margin_mb=None):
        keep = max(0, min(12, int(keep_core_layers)))
        self.lite = True
        self.lite_keep_core = keep
        self.slots = 1
        self.chunk = min(int(chunk or 1024), self.full_chunk)
        self.spec = 'lookup'
        self.model.W.offloaded = set(range(24)) - set(range(6, 6 + keep))
        self.L('enter_lite', keep)
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
        self.slots = self.full_slots
        self.chunk = self.full_chunk
        self.spec = 'dspark'
        self.model.W.offloaded = set()
        self.L('exit_lite')
        return self.lite_status()

    def lite_status(self):
        return {
            'lite': self.lite,
            'keep_core_layers': self.lite_keep_core,
            'offloaded_layers': len(self.model.W.offloaded),
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

        for bad in (
            {}, {'enabled': 'yes'}, {'enabled': 1}, {'enabled': True, 'keep_core_layers': -1},
            {'enabled': True, 'keep_core_layers': 13}, {'enabled': True, 'keep_core_layers': True},
            {'enabled': True, 'timeout_min': -5}, {'enabled': True, 'timeout_min': True},
            {'enabled': True, 'chunk': 0}, {'enabled': True, 'margin_mb': -1},
        ):
            rb = client.post('/admin/lite', json=bad)
            assert rb.status_code == 400, (bad, rb.status_code, rb.text)

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


if __name__ == '__main__':
    test_weights_offload_and_reload()
    test_streamed_moe_numerics()
    test_engine_lite_lifecycle()
    test_no_expert_ref_leak()
    test_scheduler_and_http_lite()
    print('ALL LITE TESTS OK')
