# Design: batched decode, continuous batching, speculative verify (2026-11-22, design only)

Goal: several concurrent requests (Qwen does 2; we want up to 8) without hurting the single-stream path, and a
multi-token verify path for prompt-lookup / DSpark drafts. Fits ARCHITECTURE.md decision 5 (one chunk of prefill at a
time, batched decode, preempt between chunks) and roadmap phases 3 and 6. Numbers marked *est.* are arithmetic, not
measurements.

## 0. One idea underneath everything: a step = R rows, each row = (slot, pos, token)
- A **slot** is a fixed-address per-sequence decode state (rings, conv states, page table, ring positions).
- A step runs R rows (R <= 16 = one tensor-core M tile). Rows of the same slot are consecutive with ascending pos.
  Batched decode = R slots with 1 row each. Spec verify = 1 slot with T rows. Mixed (4 seqs, one verifying 4
  drafts) = 7 rows. **One kernel set, one graph per R**, so the verify path costs nothing extra to build once
  batching exists.
- Rows of one slot are causal through masks that already exist (near: `d = pos - rpos >= 0`; far: key < pos-1023),
  and convs read earlier rows of the same slot from a per-step input history (sec. 1.4). State (conv) is changed
  only by a separate **commit** after the step, given accepted rows per slot. Rings and pages need no rollback
  (sec. 3.3).

## 1. Batched decode (new file `decode_batch.py`)

### 1.1 Static layout (allocated once, graph-safe; `Bcap = INK_MAX_BATCH` slots, `Rmax = 16` rows)
| buffer | shape | note |
|---|---|---|
| ring K/V | per visit `[Bcap, WR, 8, 128]` bf16 x2 | **WR = 1088** (1024 + 64): slot index `pos % WR`; needed for verify (sec. 3.3); mask stays `d < 1024`, so the math is unchanged |
| rpos | `[Bcap, WR]` int64 | position held by each ring slot, -1 empty |
| conv | per visit `[Bcap, 3, C]` bf16 for k, v, attn, mlp | read-only during a step; written by commit |
| tbl | `[Bcap, pool.n]` int64 | page tables (fixed-address copies of each `kvpool.Table.t`, like `Decoder.load` today) |
| row inputs | `rowslot[Rmax]`, `rowpos[Rmax]`, `rowk[Rmax]` (index within its slot's group), `live[Rmax]`, `tok[Rmax]`, `x_raw[Rmax,H]` | host fills with `copy_` before replay; nothing else changes between replays |
| groups | `gslot[Rmax]`, `grow0[Rmax]`, `gn[Rmax]`, `ng` | slot groups for the attention kernels (sec. 1.3) |
| activations | Fused's buffers with a leading row dim: `hbuf[48,Rmax,H]`, `B[Rmax,8,H]`, `Y[Rmax,8664]`, `A/X/M/G/Z`, `att`, `U`, `S[Rmax,128]` | `hbuf` also holds the DSpark taps for free (inputs of visits 5/23/41 = `hbuf[4]/[22]/[40]`) |
| conv input history | `xh[48, Rmax, 6656]` bf16 (k 1024, v 1024, attn 2304, mlp 2304) | ~10 MB; enables in-step causality and rollback |
| MoE plan | `ids[Rmax,8]`, `wts[Rmax,8]`, `ulist[128]`, `umask[128]` (row bitmask), `uslot[128,Rmax]` (slot k of the expert in that row, -1), `ucnt` | built on device, fixed size |
| far partials | `fo[Rmax/1.., ns, 24, 128]` per group | ns = cap / split as today (3 MB per group at 1M) |
| logits | `[Rmax, VOCAB]` fp32 | 13 MB |

Pad rows (R bucket > live rows): `live = 0`, and every kernel skips writes for them (ring, rpos, page row, xh) and
the MoE plan gives them no experts (so padding adds no weight bytes). No dummy slot, no zero-page writes.

### 1.2 Graphs
- One graph per R, captured lazily on first use, for R in 1..8 (and 12, 16 for verify widths). With all scratch
  preallocated the graph pool only holds Triton/cuBLAS temporaries, so 8-10 graphs should be cheap; measure the pool
  size per graph in the first window, fall back to {1,2,4,8,16} with padding if it isn't.
- Everything that varies is a device tensor read by the kernels: positions, slots, groups, live mask, tokens. Grids
  depend only on R (and on cap for the far kernel), never on positions or on which slots are used.
- Sampling and acceptance stay outside the graph in v1 (one host sync per step for R tokens; same as today's plain
  loop). Device-side sampling with a counter-based RNG keyed (seed, position) is v2 (makes seeded outputs
  independent of batch composition; needed for the pipelined batched loop).
- Long-prompt solo mode: reuse `INK_DEC_DROP_MIN`: if a >= 64k prompt is admitted while no other sequence decodes,
  drop the BatchDecoder (slots + graphs) and rebuild it with the needed Bcap before decode (1-2 s per graph).

### 1.3 Kernels, generalized from `decode_kernels.py` (import its jit helpers `_r _conv4 _quant_put _e4m3 _deq_dot _e2m1 _mma _bf16`, `visit_plan`, `kv_ptrs`; edit nothing there)
- **GEMV -> small GEMM** (proj 8664x2304, wo 2304x3072, router 128x2304 fp32, loop gate down/out, unembed via
  cuBLAS): weight tile `[BN, CH]` loaded once, activations `[16, CH]` (rows padded to 16, masked), `tl.dot` with fp32
  accumulate, same bf16 rounding points per row. Bytes are the same as T=1; the question is only whether small-M
  `tl.dot` reaches bandwidth (measure first, sec. 5). Input norm / rms prologues become per row.
- **mix** (attn-res): grid `(nq, R)`; depth entries `B[row]`. Activation-only, cheap.
- **pre**: grid `(8 kv heads, R)`: per row its own pos, tau, ring slot `pos % WR`, page row via `tbl[rowslot]`.
  K/V conv inputs go to `xh`; the conv for row r uses `xh` rows r-1..r-3 when `rowk >= 1..3`, else the slot's
  conv state (state rows `3-(i-rowk)`). No state write.
- **near** (part / out / fin): grid `(8, WR/64, ng)`: program packs the group's rows x 3 q-heads into the 16-row M
  tile (up to 5 rows per group in one tile; T=8 needs M=32), reads the ring tile once for all of them. Per-row
  mask uses each row's pos. For plain batched decode a group = 1 row (3 of 16 M rows used; bandwidth-bound anyway).
- **far**: same grouping: grid `(8, splits, ng)`, one K/V page read serves every row of the group, per-row
  `kend = pos_r - 1023`. Essential for verify at long context (otherwise T x 1.2 GB at 130k). Distinct sequences
  read their own pages: unavoidable, ~9 KB per context token per row.
- **MoE** (per-row routing, union of experts): (1) router GEMM -> `S[R,128]`; (2) `plan` kernel (1 program): top-8
  per live row (same tie rule as today), weights, then the union: `ulist` (distinct experts, ascending), `umask`,
  `uslot`, `ucnt`; (3) **w13**: grid `(NI/BN, 8*R capped at 128)`; program u >= ucnt exits; else dequantizes its
  expert's gate/up tile once into registers and `tl.dot`s it with the rows in `umask` (16-row M tile), writes
  `act[row, k]`; (4) **w2**: grid `(H/BN, same)`, writes `part[row, k, H] = bf16(act @ w2^T) * w`; (5) **w2_fin**:
  per row sums the 8 slots in slot order (as today) + mlp conv (input to `xh`) + residual. Every expert is read once
  per step no matter how many rows chose it.
- **wo**: GEMM + attn conv (input to `xh`, previous rows from `xh`/state) + residual.
- **commit** (1 launch, after acceptance): per slot with `n_acc` accepted rows, new conv state = last 3 of
  `[state ; xh rows of the group[:n_acc]]` for all 48 visits x 4 convs. Plain decode: `n_acc = 1`, inside the graph.
- **Batch invariance as a goal**: each row's result must not depend on R or on its row index (same K order per
  output element, slot-order MoE sums, per-row split counts that depend on the row's own length). Then greedy spec
  == greedy plain *bit-exactly* when plain decode also uses these kernels at R=1. Tested, not assumed.
- B=1 keeps today's `Fused` path (being tuned now) until the batched kernels at R=1 measure as fast; then delete
  Fused so there is one decode path (and spec == plain holds exactly).

### 1.4 Host API (the contract the scheduler and spec code program against)
```
bd = BatchDecoder(model, slots=INK_MAX_BATCH)
bd.load(slot, seq)          # copy a prefilled Seq (rings time-ordered -> slot pos % WR, conv, table) into a slot
bd.free(slot)
lg = bd.step(rows)          # rows: [(slot, pos, token)], slot groups consecutive; -> logits [R, VOCAB] (static)
bd.commit({slot: n_acc})    # default: every row accepted; plain decode commits inside the graph
st = bd.state(slot)         # Seq-like clone at the slot's pos (park / preempt), same format as Engine.state_of
```
A slow `LoopBatch` stub with the same API (B independent `Seq`s through `Model.forward`) lets the scheduler and spec
work proceed on CPU before the kernels exist.

### 1.5 Memory budget (*est.*; 32 GiB card, idle 26.3 GiB with the 1M pool, 1M decode min free 2.06 GB measured)
- Per slot: rings 48 x 2 x 1088 x 8 x 128 x 2 B = **214 MB**; conv 1.9 MB; rest negligible. Shared scratch
  (R=16): xh 10 MB, hbuf 3.5 MB, far partials ~3 MB per group, logits 13 MB.
- Bcap=4: 0.86 GB; Bcap=8: 1.7 GB. Plus the prefilling Seq's own rings (<= 0.2 GB) and parked states (0.2 GB each).
- 1M alone with 8 slots allocated would leave ~0.3 GB free during its prefill: too tight. Hence: **default
  INK_MAX_BATCH=4**, solo mode (1.2) frees the slots for long prompts, and raise to 8 after `gpu_mem.py` with the
  new buffers shows >= 0.7 GB min free at 1M.
- If 8 slots must coexist with large contexts: **fp8 rings** (e4m3, no scale: K is l2-clipped to 22.6 per head,
  V is RMS-normed so |v| <= 11.3; basalt idea) halve rings to 107 MB/slot. That breaks decision 2's "exact near
  window", so it ships only behind a KL/needle gate. Not in v1.
- KV pages: unchanged rule: reserve prompt + max_tokens at admission (no overcommit, no mid-decode NoRoom). 8 x
  (20k + 4k) = 192k tokens of 1M: pages are not the binding constraint for chat traffic; slots/VRAM are.

## 2. Scheduler: continuous batching (`server.Scheduler` + `engine.py` split into steps)

### 2.1 Engine split (owner: scheduler agent)
`generate()` becomes a `Request` object plus steps, and the old `generate()` stays as a wrapper (tests, tools):
`eng.start(req)` (prepare, keys, restore parked/disk prefix, reserve pages, raises NoRoom) -> `eng.prefill_chunk(req)`
-> done? -> `eng.to_decode(req, slot)` (logits from `f`, first token, `bd.load`) -> `eng.decode_step(reqs)` (one
`bd.step`, per-row grammar mask + sample with each request's generator, one `.tolist()`, parser/stop/delta callbacks)
-> `eng.finish(req)` (release pages, free slot). Parser, stopper, grammar matcher and RNG live in the Request, so
parking and resuming a decode keeps the stream continuous.

### 2.2 The loop (one worker thread, single CUDA stream)
```
loop:
  admit: walk the heap in (priority desc, arrival) order; admit while a slot is free and pool.can_admit(need)
         (need = prompt + max_tokens - cached; parked states are pushed out first, as engine.admit does today).
         No backfilling past a blocked head of the same or higher priority (protects the 1M request from
         starvation by a stream of small ones), except requests that fit in pages left after the head's need.
  pick:  if prefills and decodes both exist, choose by time share: decodes get DECODE_SHARE of wall time
         (default 0.5; 0.8 when a VIP is decoding and the prefill is non-VIP; 0.3 when the prefill is VIP and the
         decodes are not). Track measured chunk / step times as a deficit counter: whoever is behind runs next.
  prefill: one chunk of the highest-priority prefilling request. Chunk = INK_CHUNK (4096) when nothing decodes,
         INK_MIX_CHUNK (2048) while decodes run (caps a decode stall at ~0.5 s short ctx / ~1.8 s at 1M, est.).
  decode: one batched step over all decoding slots (+ spec drafts, sec. 3), emit deltas, finish/free.
  preempt: a waiting higher-priority request that cannot be admitted (no slot or no pages) preempts the
         lowest-priority, newest running request: a prefill is parked between chunks (as today); a decode is
         parked at its current pos: keys = prompt + generated ids, state = bd.state(slot), pages forked into the
         prefix tree, keep=True so pages + state go to disk if the pool needs the pages; the Request (parser,
         RNG, grammar) goes back to the heap and resumes through restore with no recompute.
```
- Cancel (client gone): checked every iteration; frees slot and pages, no park.
- NoRoom handling stays (requeue; error after INK_NOROOM_S), but with batching NoRoom means "wait for a running
  request to finish", and the head-of-line rule decides who gets the freed pages.
- Fairness: within a level every decoding request gets a row every step (inherently fair); prefills FIFO within a
  level. VIP starvation of low priority is by design; an optional aging knob (`INK_AGE_S`) is the escape hatch,
  default off.
- 1M request: needs ~all pages; it runs alone because nothing else fits; with head-of-line blocking it waits only
  until running requests drain (or preempts them if it is VIP).

## 3. Speculative decoding on the verify path

### 3.1 Prompt-lookup drafts (new file `spec.py`, no model)
- Per request: a hash index over n-grams of prompt + generated ids (n = 3..1, longest match first, most recent
  occurrence); draft = the following up to K tokens (K by budget: 4 default, <= 15 so a group fits one tile).
- Only propose when a match exists; adaptive controller (maya): keep a per-request acceptance estimate `a`; draft
  width k maximizing `(1 - a^(k+1)) / (1 - a) / t_step(k)`, using measured step times per R; probe after 8 misses.
- In a batch, draft rows count against Rmax; drafting yields to new sequences when rows are short (plain rows first).

### 3.2 Acceptance
- Greedy: accept draft i while `argmax(logits[i-1]) == d_i`; then the bonus token = argmax of the first rejected
  (or last) row. Output identical to plain greedy if the kernels are batch-invariant (1.3).
- Sampling: apply temperature/top-k/top-p to each row first (the real target p). Prompt-lookup drafts are point
  masses (q = 1 at d): accept with probability `p(d)`; on rejection sample from p with d removed, renormalized.
  DSpark drafts (distribution q): accept with `min(1, p/q)`, residual `max(p - q, 0)` normalized. Both are exact.
- Draft length is clipped to remaining max_tokens and stops at EOS/END/stop strings; grammar requests (json) draft
  only tokens the matcher accepts (check on host; reject the rest of the draft at the first invalid one).

### 3.3 Rollback after accepting a of T rows
- **Conv states**: never written during the step; `commit({slot: a+1})` builds them from `xh` (sec. 1.3). This is
  the per-draft-position checkpoint, for free.
- **Rings**: WR = 1024 + 64 means rows p..p+T-1 overwrite slots of positions p+j-1088 < p-1023, which no current or
  future query needs. Rejected positions sit in slots with `rpos > pos` (masked by `d < 0`) until that position is
  decoded again, and a step writes its own K/V before reading. No ring rollback, no backup copy.
- **KV pages**: position-addressed; rejected rows are rewritten later and the far kernel only reads keys < pos-1023.
  Pages for pos + T must be reserved (they are: prompt + max_tokens, drafts clipped to it).
- Position bookkeeping: `pos += a + 1` on the host. Parking publishes pages up to the accepted pos only.

### 3.4 DSpark (later, after prompt-lookup works)
- SPEC §9 is high-level; the math (fuse_weight over taps [5,23,41,48], block-of-8 mask embeds, 3 layers incl. one
  global over drafter context K/V, markov head) is UNCONFIRMED. Per the rules: a **subagent** re-checks official
  sources only (model card, config `mtp_config`, mtp.safetensors names/shapes, any Thinking Machines blog/paper)
  and extends SPEC §9; no SGLang/vLLM code, and maya/basalt drafter notes are SGLang/vLLM-derived (ideas only).
- Key property: **a wrong drafter costs speed, not correctness** (verification is exact). So the gate is
  acceptance rate on fixed prompts (published references: basalt 68% conversation / 51% prose / 97% structured).
- Costs to plan: 0.6 GB weights + drafter context KV (one global layer; cap it, e.g. 64k fp8 ~ 0.25 GB) next to
  the 1M pool: DSpark is off for > ~512k contexts or when Bcap slots are in use, unless gpu_mem says otherwise.
  Taps come from `hbuf` (no extra kernel). Drafter forward runs eager torch first, graph later.

## 4. Order, exit checks, ownership, expected speed

### 4.1 File ownership (no two agents edit the same file)
| agent | owns (creates/edits) | reads only |
|---|---|---|
| A kernels | `decode_batch.py`, `tests/test_decode_batch.py`, `tests/gpu_batch.py` | decode_kernels.py, decode.py (other agent is tuning them: import, never edit; keep helper names stable, note in NOTES) |
| B scheduler | `engine.py`, `server.py`, `tests/test_sched.py`, `tests/test_batch_sched.py`, `tests/gpu_features.py` | decode_batch.py API (uses `LoopBatch` until A lands) |
| C spec | `spec.py`, `tests/test_spec.py` | BatchDecoder API |
| main | NOTES.md, ARCHITECTURE.md, window.sh, kvpool.py if needed | - |

### 4.2 Steps (each: CPU exit, then GPU exit in a window)
0. **Measure first** (CPU, no GPU; A or main): record top-8 ids per visit from real-weight prefills of a few
   natural 2k prompts; compute mean distinct experts per visit for B random tokens of different prompts (batch)
   and for T consecutive tokens of one prompt (verify). Also: prompt-lookup hit/accept rates by replaying stored
   greedy outputs (tests/oracle.json, gate transcripts) through spec.py's matcher. These two numbers set Bcap, K
   and the expected speeds below. GPU (first window): one small-M `tl.dot` GEMV prototype on the 8664x2304 proj
   at M = 1..16 vs today's `_proj_kernel` (bandwidth reached?) and graph-pool bytes per R.
1. A: `LoopBatch` stub + API (1.4). CPU: stub B-step == B single Seq steps exactly.
2. A: batched kernels + plan + commit, eager. CPU (interpreter, real weights like INK_REAL tests): (a) R=B distinct
   slots == B separate `Fused` steps within bf16 noise (KL/max-abs gates as test_decode); (b) T rows of one slot ==
   T teacher-forced single steps; (c) commit(a) state == state after a single steps; (d) **batch invariance**: a
   row's logits bit-identical across R, row index and pad rows; (e) pad rows write nothing (hash of rings/pages/conv
   before/after); `compile_sm120.py`-style offline compile. GPU (`gpu_batch.py`): graph == eager lockstep per R,
   KL vs noise against B x Fused, tok/s per R at 1k/40k/130k, VRAM per slot, graph pool size.
3. B (parallel with 2, on LoopBatch): Request split + loop + admission + time share + decode preemption. CPU:
   test_batch_sched (fake engine: priority admission order, no backfill past blocked head, time share within 10%,
   VIP preempts a decode and the resumed output == uninterrupted output (greedy, real weights, small), cancel frees
   pages, no page leak); existing test_sched/test_engine_cpu/test_chat pass. Switch to BatchDecoder after step 2.
   GPU: gpu_features all, 8 concurrent short + 1 long request (every one correct, ITL p99 logged), the gate queue
   test (178k + 6x20k), and the SGLang change gate (max image + json_schema + queued long prompt).
4. C (parallel from the start): spec.py matcher + acceptance. CPU: rejection sampling matches the target
   distribution (chi-square, fake logits), greedy spec == greedy plain on a fake deterministic model, controller
   picks k=0 when acceptance is low. After 2: verify on real weights, greedy spec == plain (bit-exact if 1.3 holds,
   else tie-margin path test). GPU: tok/s on prose / code edit / copy prompts, spec on vs off.
5. Default on: INK_MAX_BATCH=4, spec lookup on; raise Bcap after gpu_mem at 1M. Then DSpark research subagent.

### 4.3 Expected throughput (*est.*, short context)
Bytes per step = dense 3.68 GB (fused attn proj 1.95 + wo 0.68 + unembed 0.92 + loop gates 0.13, read once per
step) + 0.23 GB per distinct expert per visit (4.8 MB x 48) x U(B) + 0.19 GB rings per row. U(B) with uniform
routing (an upper bound; real router skew means more sharing): 8, 15.5, 29.1, 51.6 for B = 1, 2, 4, 8.
| B | GB/step | today's overhead (5.9 ms fixed + 1.4 TB/s) | tuned (1.0 ms fixed + 1.4 TB/s) |
|---|---|---|---|
| 1 | 5.7 | 10.0 ms: 100 tok/s | 5.1 ms: 197 tok/s |
| 2 | 7.6 | 11.4 ms: 176 total, 88 each | 6.5 ms: 310 total, 155 each |
| 4 | 11.1 | 13.9 ms: 288 total, 72 each | 9.0 ms: 446 total, 112 each |
| 8 | 17.1 | 18.1 ms: 442 total, 55 each | 13.2 ms: 606 total, 76 each |
- Per-request speed **does drop** with B: expert bytes grow with the union (8 experts x B mostly different
  sequences), unlike verify rows of one sequence. Dense weights are the shared part. Measuring U (step 0) replaces
  the uniform bound; if real U(8) is ~35, B=8 tuned gives ~10.5 ms: ~760 total / 95 each.
- At 130k each row adds 1.17 GB of far KV per step: B=4 at 130k is ~15.8 GB/step (~12 ms tuned, ~83 tok/s each).
- Spec verify (T=4 of one seq; maya measured 5 tokens = 2.7x one token's expert bytes): ~8.1 GB, ~6.8 ms tuned vs
  5.1 ms for one token: break-even at 1.34 tokens per round. Expect ~+50-90% on edits/copying, ~0 on prose (no
  matches -> no drafts). Single stream >= 200 tok/s on code edits is reachable before the kernel tuning finishes.

## 5. Risks and what to measure first
- **Capture safety with variable R**: everything variable must be a device tensor; host writes only via `copy_`
  into captured buffers; no `.item()`, no allocation outside the graph pool, no Python branch on device values.
  Test: op/launch trace identical across replays with different slots/positions/live masks (as test_decode does
  per far bucket), and a slot swap test (load a different seq into a slot between replays).
- **Memory**: 214 MB/slot + graphs + prefill transient near the 1M pool. Measure gpu_mem at 1M with Bcap 4/8;
  solo mode and fp8 rings are the levers. Graph pool growth per R is unknown: measure.
- **Small-M kernel efficiency**: if `tl.dot` with M=16 padding loses bandwidth vs the tuned GEMVs, batched steps
  cost more than the table says. Prototype on one matrix first (step 0).
- **MoE plan overflow / skew**: union capped at 128 (fixed grid 8R <= 128); measure U; a hot expert chosen by all
  rows is fine (one read, 16-row tile).
- **Numerics**: batched rows differ from Fused by reduction order (bf16-noise level); gate by KL vs noise, not
  exact match, until Fused is replaced. Batch invariance is what keeps spec == plain exact.
- **Priority starvation**: head-of-line blocking protects big requests but stalls small ones behind a big head of
  the same priority; time share protects decodes from long prefills; VIP floods starve low priority by design.
  Log per-request queue wait and ITL p99 in /metrics.
- **Decode preemption cost**: parking a 130k decode to disk writes ~1.2 GB pages + 0.2 GB state (~1 s on NVMe) and
  is SSD wear; only when pages are really needed (slot-only shortage parks on the GPU).
- **Concurrent tuning**: decode_kernels.py helper signatures may change under the tuning agent; decode_batch.py
  pins what it imports, test_decode_batch catches drift.
- **Measured (2026-11-22, `tests/expert_stats.py`, CPU real-weights prefills of 8 natural prompts: 2 each prose /
  code / chat / json, 348-613 tokens; top-8 ids at all 48 visits; 3000 random draws).** Distinct experts per visit U,
  mean over visits (NVFP4 / MXFP8 visits similar: e.g. B=8 46.9 / 49.2): **batch (different sequences) B=2/4/8:
  15.1 / 27.5 / 47.1** (uniform 15.5 / 29.1 / 51.6): almost no cross-sequence sharing, the uniform bound in 4.3 is
  ~right (B=8 tuned ~12.5 ms, ~640 total / 80 each, not the hoped 760 / 95). **verify (consecutive tokens of one
  sequence) T=2/4/8: 12.7 / 19.9 / 30.0**: strong sharing; step cost vs one token (tuned) x1.15 / x1.38 / x1.71
  (today x1.08 / x1.20 / x1.36), so spec breaks even at 1.38 accepted+bonus tokens per T=4 round. Bytes per distinct
  expert: NVFP4 4.48 MB, MXFP8 8.21 MB (0.230 GB across 48 visits). Implication: batching buys aggregate throughput
  at a steady per-request cost (B=4 ~115 each tuned); spec verify is the cheaper way to add rows; spec.Width uses
  these numbers as its default step-time model.
