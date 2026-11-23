# Benchmarks

Measured 2026-11-23 on one RTX 5090 (32 GB) with a 6-core host, 30 GB RAM and NVMe, using an NVFP4/MXFP8
Inkling-2-Mini checkpoint and the server defaults in `server.py` (FP8 decode weights, CUDA graphs, fused decode,
FAR2 far attention, DSpark speculation, batch of up to 4). Greedy decoding unless noted. Numbers come from the GPU
test scripts in `tests/` and are not averaged over many runs, so expect a few percent of noise.

## Decode, one request (tok/s)

Kernel-level, from `tests/gpu_decode2.py` (batched decoder at B=1, which is what the server uses for a lone request,
and the single-row fused decoder):

| context | batched B=1 | fused single-row |
|---|---|---|
| ~1k | 154 | 190 |
| 40k | 145 | 179 |
| 131k | 135 | 162 |
| 524k | 96 | 112 |
| 1M | 73 | 80 |

Through the HTTP server (`tests/gpu_features.py`, short context, end-to-end tok/s from usage + wall time):

| workload | plain | with DSpark | speedup | output |
|---|---|---|---|---|
| code | 142 | 296 | 2.09x | identical |
| prose | 143 | 141 | 0.99x | identical |

## Speculative decoding at long context

From `tests/longspec.py` against the live server. Tool = an `edit_file` call that copies a function from early in
the context, edit = a class rewrite. Greedy output was identical with and without speculation in every case.

| context | tool call | edit |
|---|---|---|
| 130k | 3.03x | |
| 512k | 3.48x | 4.98x |
| 1M | 2.21x (66 -> 146 tok/s) | 5.19x (75 -> 389 tok/s) |

## Concurrency

Kernel-level aggregate decode at B=4 (`gpu_decode2.py`): 401 tok/s at ~1k, 351 at 40k, 296 at 131k, 178 at 524k,
109 at 1M. Through the server, 4 short requests at once: 178 tok/s aggregate vs 114 tok/s for one at a time.

## Prefill and time to first token

| prompt | result |
|---|---|
| 130k-token needle test, 8 needles | 8/8 found, 14 s end to end |
| 1.04M-token needle test, 8 needles | 8/8 found, 384 s end to end |
| 1M cold prefill on the live server | 379 s |
| 16.7-megapixel image | TTFT 1.9-2.0 s |

A warm turn that reuses a cached prefix (GPU or disk tier) only prefills the new suffix.
