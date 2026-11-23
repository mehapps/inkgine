# Pre-window review (2026-11-21, read-only, CPU only)

Scope: model.py, decode.py, kvpool.py, engine.py, server.py, grammar.py, moe_triton.py, vision.py, chat.py,
tests/window.sh (phase e), gpu_check.py, gpu_decode.py, gpu_features.py, long_ctx.py, compare.py, swap_test.sh.
Findings are ranked by likelihood x cost in the window. Confidence: H / M / L.

## Summary
The engine code itself looks sound. I found no page-table off-by-one, refcount leak or graph-capture violation in the
decode path. The biggest risks are in how the window is run: step order, total time, the fail rules, and restoring
Qwen. After that come one memory squeeze at 1M and two test checks that can fail when nothing is broken.

## 1. Window script / procedure

**W1. Phase e can end the window, or hit the timeout, before it tests the new code (M-H).** `tests/window.sh:114-136`
- Order today: gpu_check (~15 min in window d) -> gpu_decode (~8-12 min: about 1.1k Seq-path steps at ~0.1-0.3 s)
  -> oracle start (wait up to 20 min; oracle.py failed in the 14:48 window) -> oracle.py -> compare -> server ->
  gate -> features -> 130k -> 1M (~10-16 min). Estimate: 60-80 min plus the Qwen stop/restore. That doesn't fit a 75 min
  cap, and the line 8 header asks for 110. With a 75 min cap, `timeout` kills the 1M needle.
- The oracle docker and `compare.py` are the most fragile steps. compare is gated per prompt on top-1 >= 0.95 **and**
  greedy == oracle for >= 8 tokens (`compare.py:77`). They run before anything exercises the pool, the parked states
  or the graphs. One oracle hiccup or one early near-tie gives a window with no result for the integration work.
- Fix: put the server steps first: gpu_check -> gpu_decode -> server: gate image, gpu_features, 130k, 1M. Run
  oracle + compare last or in a separate window. Consider dropping gpu_check part 2/3: window d already decided the
  kernel; only the limit changed.

**W2. The script breaks the user's "stop at first failure" rule in two places (H).**
- `window.sh:117`: if gpu_decode fails, the window keeps going for about 50 more minutes with GRAPH=0. The memory
  rule says to restore Qwen on the first failure. Either `fail` there, or get the user's explicit OK for this exception.
- `window.sh:132`: `grep -v json_schema` still hides the json_schema FAIL, although json is implemented now. The
  change gate (16.7 MP image, then json_schema right after it) is a hard user requirement. Drop the `grep -v`.

**W3. Restoring Qwen can race our still-running server (M, expensive).** `swap_test.sh:26`, `window.sh:18`
- Restore does `pkill` (TERM), `sleep 5`, then `docker start`, and never checks that VRAM was freed. On TERM, uvicorn
  waits for in-flight request tasks. If a handler is still awaiting the worker (a client timed out mid-prefill; a 1M
  prefill runs ~15 min), the server keeps ~27 GB. Qwen then OOMs at start, the restore loop waits 15 min and warns, and
  only the failsafe at max+5 brings Qwen back.
- Fix (both places): `for i in $(seq 30); do pgrep -f 'llm/engine/[s]erver.py' >/dev/null || break; sleep 1; done;
  pkill -9 -f 'llm/engine/[s]erver.py'`. Then wait until `nvidia-smi --query-compute-apps=pid --format=csv,noheader`
  is empty before `docker start`.

**W4. No per-step timeouts (M).** gpu_features `post(timeout=1800)` and long_ctx (6 h) can each burn the rest of the
window on a hung server, and gpu_decode can hang with no limit at all. Fix: wrap the steps, e.g.
`timeout 20m python tests/gpu_decode.py`, `timeout 15m ... gpu_features.py all`, `timeout 25m ... long_ctx.py 1040000`.

## 2. Tests that can fail when nothing is broken

**T1. The gpu_features prefix speed checks are borderline when GRAPH=0 (M; H if gpu_decode fails).**
`gpu_features.py:111-126`
- Cold turn 1 = a 20k prefill (~5-6 s with triton) + 8 decode steps. Warm and disk = a tiny prefill + **24** decode
  steps. On the Seq path (~0.1-0.15 s/token) that gives warm/cold of about 0.45-0.55 against the `< 0.5` limit. A
  failure here ends the window before the needles.
- Fix: use `max_tokens=8` for warm and disk as well, or time prefill only (stream it and use TTFT).

**T2. `a == b` (warm vs disk answer) depends on the triton atomic-add order (L).** fp32 atomics in `_w2_kernel` make
decode nondeterministic below one bf16 ulp. A near-tie within 24 greedy tokens would flip the answer. Fix: also accept
a mismatch when both answers contain the code.

**T3. `gpu_decode.py --long` runs out of pool pages (M; not in phase e).** `graph.seq` / `eager.seq` (decode.py:113)
keep the previous prompt's clone, which shares its Table. The 40k table (~157 pages) stays alive while the 130k
prompt (~508 pages) prefills in a 547-page pool, so kvpool.put hits `AssertionError: KV pool full`.
Fix: after each prompt, `graph.seq = eager.seq = None` (before `del seq`).

## 3. Memory (1M, 32 GB)

**M1. 1M margin with graphs is about 0.5 GB, and a failed park throws away a 15 min prefill (M).**
- The baseline is 1.4 GB min free at 1M (dev window 2: torch MoE, no pool, no graphs). New in phase e:
  - Decoder static state: 0.2 GB.
  - The graph private pool. It is shared by all buckets and holds about one far block's temporaries: K+V bf16 at
    128 MB each + dequant temps, ~0.5 GB. The normal allocator never gets this memory back, so the 1M prefill can't
    use it.
  - The end-of-prompt `state_of` clone: 0.2 GB.
- Parked states and their pages are pushed out at admit (engine.py:153). That part is fine.
- An OOM inside `park()` (engine.py:197 -> state_of / fork) fails the whole request after the prefill.
- Fix: wrap the end-of-prompt `self.park(...)` in `try/except torch.cuda.OutOfMemoryError` (log, skip). Optionally skip
  GPU parking when `len(ids)` is huge and `keep` is False: a 1M one-off is never reused.
- In the window: if the 1M needle OOMs, rerun it once with `INK_GRAPH=0`. The `needle()` helper turns the wrong knob:
  it retries with 8k chunks, which is larger than 4096.
- The gate image VRAM check (>= 300 MiB) should pass. Image prefill peaked at 19.6 GiB on a 32k pool, so on the 1M
  pool it is about 28.3 GiB + 0.6-0.8 GB of new state, leaving roughly 2 GB free.

## 4. Scheduler / server robustness

**S1. NoRoom retries never stop (L likelihood, high cost).** `server.py:77-80`
- The job is requeued with its old sequence number, so it pops again at once.
- Each retry re-renders the prompt, re-preprocesses its images, and `pool.restore` may read a 200 MB state file.
- Everything at or below its priority waits forever. With one active sequence this needs a page leak, but then the
  server is wedged.
- Fix: count retries, and after ~30 s with nothing else active, `put('error', ...)` (503) and log free_pages / parked /
  len(nodes).

**S2. The skip for a cancelled queued job leaves `self.active` set (L).** `server.py:68-69`: `continue` jumps past the
`finally`, so /metrics shows `running 1` while idle and no `notify_all` happens. Fix: set `self.active = None`
before `continue`.

**S3. A client disconnect mid-prefill acts as a preemption (L).**
- engine.py:182-186 parks the progress with `keep=a >= DISK_MIN`, so an abandoned one-off prompt gets a disk write
  later (unwanted SSD wear).
- server.py:73 counts the disconnect as a preemption. That inflates the metric gpu_features checks.
- Fix: pass the reason (cancel vs higher priority). On cancel, release without parking and don't count it.

**S4. Minor, cheap fixes.**
- engine.py:243 runs one extra decode step after the last allowed token when `n == max_tokens`. Fix: break before the
  step.
- engine.py:248 reports finish_reason `length` when the JSON finished exactly at max_tokens. Use `stop` if `cons.done`.

## 5. Checked and looks OK (no action needed)
- **Graph capture in the decode path.**
  - Nothing in the captured path syncs with the host: no `.item()`, no `.tolist()`, no data-dependent Python
    branches (DecodeState forces triton MoE; `repeat_interleave` takes int repeats).
  - The embedding H2D copy and `pos.fill_` run outside the graph. Sampling runs outside (no RNG in the graph).
  - Triton kernels are compiled by the eager warm-up, so the capture sees the same specialization.
  - The page table sits at a fixed address (`st.kv.t`, copied in at load and on growth), and the far bucket is a
    Python constant per graph.
  - Sharing one pool across buckets is safe because graphs never replay concurrently. DecodeState
    `slot/ok/dist` live in the graph pool and are only touched inside each graph's own replay.
- **Residual capture risks.**
  - `capture_error_mode` is the default `global`: any CUDA call from another thread during capture breaks it. The
    uvicorn thread makes none today (keep /metrics and the handlers torch-free).
  - The first time triton MoE runs at T=1 on the GPU is gpu_decode's ref path (so far it ran only in the CPU
    interpreter).
- **Pages.** fork copies the partial tail page into a private page. publish adds only full pages; the live sequence
  never writes below P. restore writes the tail rows into a freshly reserved page.
  - The reservation covers len(ids) + max_tokens, including the extra final step, so the Decoder's grow branch never
    fires.
  - Every exit path releases its pages: NoRoom / Preempted explicitly, a decode error in generate's finally, and
    other prefill errors through `Table.__del__` once the traceback is dropped.
- **Image prefix keys.** I tested `h + arange` with h >= 2^63: it wraps and does not raise. Restore skips vision for
  covered images, and the row/position selection stays consistent for an image that straddles P.
- **Grammar with the graph decoder.** The mask is applied in place to the static logits buffer before `sample`, and
  the next replay overwrites it. EOS is masked in every grammar state, and END is the xgrammar stop token.
- **Env.** INK_MOE / INK_CHUNK / INK_GRAPH are exported before `start_server` and reach the server.
  `PYTORCH_CUDA_ALLOC_CONF` is set in server.py and in the test scripts.
- **Disk floor.** 182 GB free now, above the 100 GB `INK_CACHE_MIN_FREE_GB`, so the gpu_features disk check can write.
  NOTES.md still says 94 GB.
