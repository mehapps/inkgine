# Inkling engine: target architecture and roadmap

The big picture every change should fit into. NOTES.md is the log and status; this file is where we're going and why.
Update it when a decision changes. Inkling 2 Mini only, one RTX 5090 (32 GB), one host (30 GB RAM, NVMe).

## End state (what "done" means)
Replaces Qwen behind LiteLLM as the live model, with:
- 1M-token context: one 1M request runs alone; otherwise several smaller requests share the same memory pool.
- Text + full-resolution images, tools, reasoning effort, json_schema, streaming, OpenAI API.
- VIP priority (LiteLLM's `priority` field; VIPs jump the queue and can pause lower-priority work).
- Prefix cache with a disk tier (multi-turn chats don't re-read their history).
- Fast enough that switching isn't a downgrade (targets below), and it passes the SGLang change gate in memory.

## Layers (each only talks to the one below)
```
server.py      HTTP/OpenAI API, streaming, parsing, json_schema masks       (per request)
scheduler      queue by priority, admission by KV pages, chunked prefill    (NEW, replaces the global lock)
               interleaved with batched decode, preemption
state          paged KV pool + per-seq conv/ring state, page tables,        (KV4/Seq today -> pages)
               prefix tree, GPU-parked states + disk tier (no host-RAM tier)
model.py       fixed-shape Inkling forward over a batch of seq chunks        (single seq today)
kernels        packed NVFP4/MXFP8 grouped MoE, 4-bit-KV attention,          (torch/flex today)
               vision, sampling; prefill extras (flags, off): fused fp32 glue (prefill_glue.py, INK_GLUE; INK_GLUE2: attention
               prologue + 4-bit KV write, output gate in the near kernel, residual + norm, attn_res + norm),
               FP8 dense projections (fp8_dense.py, INK_FP8_DENSE), FP8-PV far attention (INK_PA_FP8);
               decode (flag, off): one fp8 dense copy, weight-only, read by the decode kernels + unembed (INK_FP8_DECODE);
               decode speed flags (off, same math): unrolled K loops, PDL, L2 weight prefetch, fused w2 epilogue, FAR2 far
               attention (fixed even splits of the far range + vector merge: cost follows context, not pool size);
               FAR3 (flag, off): leaner far kernel (int32 code words, exponent-bit int4 dequant, 3 CTAs/SM)
```

## Decisions that shape later work
1. **KV memory is the shared budget.** Only the 8 full-attention visits keep long KV (9 KB/token at 4-bit, so ~1M
   tokens = 9 GiB). Sliding visits only need the last 1023 tokens per sequence, plus 3 conv inputs per conv per visit.
   So the pool counts tokens across all active sequences: 1M for one, or e.g. 8 x 128k.
   -> KV4 must become **paged** (fixed pages, e.g. 256 tokens, page table per sequence, refcounts) before multi-request
   or prefix sharing. The current "one slot per visit, positions 0..N" layout is single-sequence by design.
2. **Exact near window, approximate far.** bf16 for distance < 1024 (the relative bias lives there, exact like the
   reference), 4-bit beyond. Any new kernel keeps this split and is checked against the torch path.
3. **Per-sequence small state** (rings: ~4 MB/visit/seq in bf16 = ~200 MB per seq for 48 visits; conv states: tiny).
   With many sequences, rings dominate small requests. Option: keep rings 4-bit too, or cap concurrency (~8).
4. **Prefix restore = far pages + rebuild the near state.** Saving rings/conv per prefix is ~200 MB, so instead store
   far KV pages, then re-run the last ~1024 prompt tokens to rebuild rings/conv states (attending to the restored
   pages). Cheap (one chunk) but approximate where 4-bit far keys feed the rebuild; measure vs a cold prefill before
   relying on it. Pages are keyed by (weights hash, KV format, token prefix): see the 2026-11-07 "Muffins" incident.
   **As built (2026-11-21):** exact restore only: pages + the saved ~200 MB state (parked on the GPU, or a disk file
   for continued chats). The approximate rebuild stays an option for page-only hits (not implemented).
5. **One process, one GPU, simple batching.** Prefill one chunk at a time (big chunks are compute-bound), decode all
   running sequences in one batched step, interleaved so a 1M prefill doesn't stall VIP decodes (preempt between
   chunks). No tensor parallelism, no multi-GPU, no model abstraction.
6. **Correctness gates every speed change:** torch reference path stays in the code (env switch) as the oracle for
   new kernels; oracle compare vs official SGLang for end-to-end; needle + gate tests in windows.

## Roadmap (dependencies in order)
| # | Phase | Needs | Exit check |
|---|---|---|---|
| 1 | Single request, 1M, text+image, correct | - | **done 2026-11-21** (1M needle 8/8, gate image, oracle) |
| 2 | Full-res images; packed-weight MoE kernel; 4-bit KV read in-kernel; natural-text oracle | 1 | image grid in window; gpu_check; 1M prefill well under 10 min |
| 3 | Paged KV pool + batched forward over several seqs + scheduler with priority + preemption | 2 (batched MoE kernel) | gate/priority.py passes; gate queue test (178k + 6x20k) passes. **Partly done 2026-11-21:** priority queue + preemption between prefill chunks, one active seq (server.Scheduler; CPU-tested, GPU check pending); **paging done on CPU 2026-11-21** (kvpool.py: 256-token pages, one table per seq, refcounts; preempted prompts park and resume at any length; NoRoom requeue); batching open |
| 4 | Prefix tree + disk tier (user 2026-11-21: GPU + disk only, no host-RAM tier; host RAM is shared with video renders) | 3 (pages) | gate turns/restore tests; multi-turn reuse speedup. **Done on CPU 2026-11-21:** chain-hashed page tree keyed by (weights hash, KV format, tokens/image hashes), end-of-prompt states parked on the GPU (INK_PARK), disk tier in ~/llm/inkcache (checksummed atomic files, LRU, size cap + free-space floor, written only on eviction and only for continued chats / preempted prompts), bit-exact vs cold; GPU check: window e |
| 5 | json_schema (xgrammar masks), tool-call polish, /metrics, served name, LiteLLM cutover + rollback | 3 | full SGLang change gate in memory passes; user OK. **Done on CPU 2026-11-21:** json_schema/json_object, /metrics, tool round-trip test; cutover open |
| 6 | Decode speed: CUDA graphs, fused small ops, prompt-lookup then DSpark speculation | 2-3 | tok/s targets. **Started 2026-11-21:** fixed-shape decode + CUDA graph per far bucket (decode.py, engine behind INK_GRAPH=1; gpu_decode in window e); one graph for all positions (far kernel over 4-bit pages); fused T=1 step (decode_kernels.py, INK_DEC_FUSE: 10.5k -> ~550 launches), tensor-core far kernel (INK_DEC_FAR_DOT), pipelined device-side token loop (INK_DEC_EMBED_GPU + Decoder.stream) CPU-verified, GPU pending |

## Targets (revise with data)
Reference points from other single-GPU Inkling implementations (ingine: 5090, FP8 dense, FP8 KV, max 262k context;
maya: 3090): decode 490-570 tok/s short / ~260 at 254k (ingine), 128 tok/s (maya, 3090); prefill
~10.9k tok/s at 20k (ingine), ~2.5k (maya). A decode token reads **~5.55 GB** of weights as we store them (counted
2026-11-21 from the checkpoint: fused attention projections 1.95 GB (bf16, loop core read 3x), 8 routed experts 1.84,
unembed 0.92, wo 0.68, loop gates 0.13), so the floor is **~3.1 ms (~320 tok/s)**; the earlier "~3 GB / 1.7 ms" was
wrong. 400+ needs fewer bytes: FP8-per-row dense weights (takeaway 7) would cut ~1.8 GB (-> ~2.1 ms floor).
- Decode: >= 200 tok/s single stream at short context (stretch 300; 400 needs FP8 dense weights), >= 100 tok/s at 1M
  (4-bit KV is ~9 GB at 1M -> ~5.5 ms of far reads per token on top of the ~3.1 ms weight floor). Measured 2026-11-21:
  44 tok/s short, 34.7 at 40k, ~12 at 1M (INK_GRAPH=1, ~10.5k torch launches per step); fused path (INK_DEC_FUSE,
  ~550 launches) awaiting its GPU window.
- Prefill: >= 8k tok/s at short context; 1M prompt in < 5 min. Today: ~1.3k tok/s, 1M in ~16 min.
- TTFT for a short chat: < 0.3 s.

## Design takeaways (ranked)
1. Decode step as **one CUDA graph** per batch width, routing/top-k on the GPU (no host sync), on-device sampling,
   launch step n+1 before reading token n.
2. **Fused MoE decode kernel** reading packed NVFP4/MXFP8 experts and dequantizing in registers.
3. **Grouped-GEMM MoE prefill** (sort token-expert pairs into tiles, dequant inside the GEMM loop).
4. **Prefix reuse by end-of-prompt snapshot** (rings + conv states + length, ~300 MB; full-attention KV is
   position-addressed so it needs no rollback). Supersedes the "rebuild near state" idea in decision 4 for recent
   prefixes; keep a few snapshots in host RAM, far pages on disk.
5. Attention kernel packing the 3 query heads of a KV head into one block; sink via `o * sigmoid(lse - sink)`.
6. Cheap speculation first (prompt-lookup drafts), DSpark later from our SPEC only (maya's drafter follows SGLang
   code, so use ideas only).
7. Concatenated per-layer projections; FP8-per-row dense weights (memory/speed; check accuracy).
8. (basalt, same 5090, tested to 400k not 1M) Attention kernel computing the relative bias in-kernel
   (query rel-state x bias-table slice) with 3 q-heads per KV head and split-K merged with the sink; prefill MoE on
   FP4 tensor cores with activations as two FP4 terms (value + residual, ~8-bit accuracy, KLD 0.0026 vs 0.0041 one
   term); L2 prefetch of next layer's likely experts (-3.6% decode). Their 1M plan offloads 12 GiB of experts to host
   RAM (121 GiB box) because fp8 KV is ~20 GiB at 1M; our 4-bit KV keeps all weights on the GPU. Speeds there:
   ~280 tok/s plain decode short, 108 at 400k; prefill 12.3k tok/s at 16k, ~4.1k at 400k.
**Test gates to adopt:** golden top-256-logit compare (mean KL <= 0.01, per stage <= 0.25, top-1 >= 95%) against
our torch reference path; GSM8K subset (maya: 96 questions, greedy, reasoning off ~73-77); code-haystack needle;
long-generation stability with a tie margin. Judge 4-bit-KV drift by KL/accuracy, not exact token match (router has
many near-ties).

## Things deliberately not planned
Other models, multi-GPU, LoRA, fine-tuning, video input, audio.
