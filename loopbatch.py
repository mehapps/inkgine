"""LoopBatch: slow but correct stand-in for decode_batch.BatchDecoder (DESIGN-batching.md 1.4), same API, so the
scheduler runs several decodes now. A step loops over its rows one sequence at a time: per slot either a
decode.Decoder (dec=True, INK_GRAPH: today's fixed-shape/graph path, created lazily on a slot's first use, ~0.2 GB
static state each; all of them share one capture stream and one CUDA graph pool, so a slot adds no graph memory of its
own; the loaded Seq's prefill rings (~0.2 GB) are let go once copied) or the plain Seq path (Model.forward on the loaded Seq). Rows are independent, so a
row's logits equal a solo run's bit for bit. Aggregate tok/s is the single-stream speed; the real win comes with
BatchDecoder (one step reads the weights once for all rows).

API (the contract BatchDecoder implements):
    lb.load(slot, seq)      take over a prefilled Seq at seq.pos (its table holds pages reserved for prompt + max_tokens)
    lb.free(slot)
    lg = lb.step(rows)      rows [(slot, pos, token)], pos = that slot's next position; -> logits [R, VOCAB] fp32
    lb.commit({slot: n})    accepted rows per slot (spec verify); plain decode commits inside step
    st = lb.state(slot)     clone of the slot's small state at its pos: {'conv.{v}.{k|v|attn|mlp}', 'ring.{v}.k|v'}
                            (rings time-ordered, last min(pos, 1023) positions), Engine.state_of's format without 'f'
    lb.drop(keep=0)         free the Decoders of empty slots except the first `keep` -> number freed (VRAM pressure,
                            solo mode, idle trim)
"""
import torch

from decode import Decoder
from model import WIN


class LoopBatch:
    def __init__(self, model, slots=4, dec=False, graph=None, fuse=None):
        self.m, self.n, self.dec = model, slots, dec
        self.graph, self.fuse = graph, fuse
        self.seqs = [None] * slots     # loaded Seq per slot
        self.decs = [None] * slots     # dec=True: Decoder per slot (lazy)
        self.side = self.gpool = None  # shared by the slot Decoders' graphs (one cuBLAS workspace, one graph pool)

    def busy(self):
        return any(s is not None for s in self.seqs)

    def idle(self):
        return sum(d is not None and s is None for d, s in zip(self.decs, self.seqs))

    @torch.no_grad()
    def load(self, slot, seq):
        assert self.seqs[slot] is None, f'slot {slot} in use'
        if self.dec:
            d = self.decs[slot]
            if d is None:
                kw = {}
                if self.graph is not None:
                    kw['graph'] = self.graph
                if self.fuse is not None:
                    kw['fuse'] = self.fuse
                d = self.decs[slot] = Decoder(self.m, embed_dev=False, **kw)  # (no 0.9 GB table per slot; stream() unused)
                if d.graph:  # Decoder._capture makes its own stream + pool only while these are None
                    if self.gpool is None:
                        self.side, self.gpool = torch.cuda.Stream(), torch.cuda.graph_pool_handle()
                    d.side, d.pool = self.side, self.gpool
            d.load(seq)  # seq.forward must not run on seq afterwards
            e = seq.ring[0][0].new_zeros(0, *seq.ring[0][0].shape[1:])
            seq.ring = [(e, e)] * len(seq.ring)  # copied into the Decoder: free the prefill rings (state() reads d.st)
        self.seqs[slot] = seq

    def free(self, slot):
        self.seqs[slot] = None
        if self.decs[slot] is not None:
            self.decs[slot].seq = None  # (it would keep the finished Seq alive)

    @torch.no_grad()
    def step(self, rows):
        out = []
        for slot, pos, tok in rows:
            seq = self.seqs[slot]
            assert seq is not None and pos == seq.pos, (slot, pos, seq and seq.pos)
            if self.dec:
                out.append(self.decs[slot].step(tok))  # each Decoder has its own static output buffer
            else:
                out.append(self.m.logits(self.m.forward(self.m.embed([tok]), seq)))
        return torch.cat(out)

    def commit(self, acc):
        assert all(n == 1 for n in acc.values()), 'LoopBatch has no verify rows'

    @torch.no_grad()
    def state(self, slot):
        seq = self.seqs[slot]
        if self.dec:
            d = self.decs[slot].st
            n = min(seq.pos, WIN - 1)
            idx = torch.arange(seq.pos - n, seq.pos, device=d.rpos.device) % WIN
            conv, ring = d.conv, [(k[idx], v[idx]) for k, v in d.ring]
        else:
            conv, ring = seq.conv, seq.ring
        st = {}
        for v, (c, (rk, rv)) in enumerate(zip(conv, ring)):
            st.update({f'conv.{v}.{k}': x.clone() for k, x in c.items()})
            st[f'ring.{v}.k'], st[f'ring.{v}.v'] = rk.clone(), rv.clone()
        return st

    def drop(self, keep=0):
        n = 0
        for i, (d, s) in enumerate(zip(self.decs, self.seqs)):
            if d is not None and s is None:
                if keep:
                    keep -= 1
                else:
                    self.decs[i], n = None, n + 1
        if all(d is None for d in self.decs):  # graphs gone: a fresh pool next time (a pool whose last graph died may be freed)
            self.side = self.gpool = None
        if n and self.m.dev.type == 'cuda':
            torch.cuda.empty_cache()
        return n


class SoloFused:
    """INK_SOLO_FUSED=1 (off by default): a BatchDecoder (`bd`) whose lone decode runs on the single-row fused Decoder
    (faster at B=1 until BatchDecoder's B=1 tuning closes the gap; +~0.2 GB for that Decoder). A request moves to the
    batch decoder when a second one loads, and back when it is the only one left. A move is state(slot) -> load, the
    same handoff as a parked decode resuming (time-ordered rings + conv states; pages stay in the pool): exact state,
    but the two backends' bf16 numerics differ, so a greedy reply may depend on whether it ran alone or batched."""

    def __init__(self, bd, model):
        self.bd, self.one, self.at = bd, LoopBatch(model, 1, dec=True), None  # at: the slot decoding on `one`
        self.n, self.dec = bd.n, True

    def busy(self):
        return self.at is not None or self.bd.busy()

    def idle(self):
        return self.one.idle() + self.bd.idle()

    @property
    def decs(self):  # the batch decoder's (to_decode's margin check is about its buffers)
        return self.bd.decs

    @torch.no_grad()
    def _move(self, src, i, dst, j):
        st, seq = src.state(i), src.seqs[i]
        seq.conv = [{k: st[f'conv.{v}.{k}'] for k in c} for v, c in enumerate(seq.conv)]
        seq.ring = [(st[f'ring.{v}.k'], st[f'ring.{v}.v']) for v in range(len(seq.ring))]
        dst.load(j, seq)  # before src.free: an OOM here leaves the request where it was
        src.free(i)

    def load(self, slot, seq):
        if not self.busy():
            self.one.load(0, seq)
            self.at = slot
            return
        if self.at is not None:
            self._move(self.one, 0, self.bd, self.at)
            self.at = None
        self.bd.load(slot, seq)

    def free(self, slot):
        if slot == self.at:
            self.one.free(0)
            self.at = None
            return
        self.bd.free(slot)
        live = [s for s, q in enumerate(self.bd.seqs) if q is not None]
        if len(live) == 1 and not getattr(self.bd, 'pend', None):
            try:
                self._move(self.bd, live[0], self.one, 0)
                self.at = live[0]
            except torch.OutOfMemoryError:  # (the fused Decoder's first build): stays on the batch decoder
                pass

    def step(self, rows):
        if self.at is None:
            return self.bd.step(rows)
        assert len(rows) == 1 and rows[0][0] == self.at, rows
        return self.one.step([(0,) + tuple(rows[0][1:])])

    def commit(self, acc):
        (self.bd if self.at is None else self.one).commit(acc)

    def state(self, slot):
        return self.one.state(0) if slot == self.at else self.bd.state(slot)

    def drop(self, keep=0):
        return self.one.drop(keep) + self.bd.drop(keep)
