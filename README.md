# inkgine

A single-GPU inference server for [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini), written in
PyTorch + Triton for one RTX 5090 (sm_120, 32 GB). It serves an OpenAI-compatible API.

- Text + full-resolution images, tool calls, reasoning effort, `json_schema` (xgrammar), streaming
- Up to 1M tokens of context (paged 4-bit KV)
- Prefix cache with a disk tier, so multi-turn agent sessions don't re-read their history
- Continuous batching with priority queueing and preemption
- CUDA-graph decode, fused decode kernels, DSpark / prompt-lookup speculative decoding

Status: runs in production on one machine. Tuned for one GPU model; expect work to run elsewhere.

## Run

Needs an NVFP4/MXFP8 Inkling-2-Mini checkpoint (weights are not in this repo).

```bash
uv venv .venv && uv pip install --python .venv -r deploy/requirements.lock \
    --extra-index-url https://download.pytorch.org/whl/cu132
INK_MODEL=/path/to/model PATH=$PWD/.venv/bin:$PATH python server.py   # http://127.0.0.1:30000/v1
```

Or in Docker: build `deploy/Dockerfile`, then see `deploy/compose.example.yaml`.

Main settings (env): `INK_MODEL`, `INK_HOST`, `INK_PORT`, `INK_MAX_CTX`, `INK_SERVED` (model id), `INK_CACHE_DIR`,
`INK_CACHE_GB`. The full list of `INK_*` flags and their defaults is at the top of `server.py` and `engine.py`.

## Layout

`server.py` (API, scheduler) → `engine.py` → `model.py` → kernels (`moe_*`, `prefill_*`, `decode_*`, `vit_attn.py`).
`kvpool.py` holds the paged KV and prefix cache, `dspark.py` / `spec*.py` the speculation. Design docs:
`ARCHITECTURE.md`, `SPEC.md`, `DESIGN-batching.md`, `IMAGES.md`. Tests: `tests/test_*.py` run on CPU, and
`tests/gpu_*.py` / `window_*.sh` need the GPU.

## License

Apache-2.0. Inkling 2 Mini is by Thinking Machines (Apache-2.0).
