#!/usr/bin/env bash
# Vision window: speed flags of vision.py (all off by default): INK_VIT_GPU_PRE (exact host path), INK_VIT_ATTN=
# flash|cudnn|efficient|triton (noise-gated), INK_VIT_FP8 (noise-gated), alone and stacked, at 1 / 4 / 16.7 MP.
#   ~/llm/engine/swap_test.sh 25 ~/llm/engine/tests/window_vision.sh
# 1. bench (Vision only): attention sweep at the 16.7 MP length (picks the Triton tiles), cost model per size, time per
#    flag + embedding gate (exact flags bit-identical; others vs the eager-vs-sdpa noise)
# 2. read (full Engine, server defaults): gate document Locker 37 = 7Q4K-2938 per config, next-token KL vs default
#    next to the eager-attention noise KL, synced prepare + prefill wall time.
# Measurements; both steps run even if the first fails. Expected ~10-12 min + Qwen restart; caps sum to 24.
set -u
cd "$(dirname "$0")/.." && export PATH=$PWD/.venv/bin:$PATH
stamp=$(date +%Y%m%d-%H%M)
log=~/llm/logs/ink-window-vision-$stamp.log
exec > >(tee -a "$log") 2>&1
echo "log: $log"
trap '"$PWD/stop_gpu.sh"' EXIT
run() { local t=$1; shift; echo "== $* (limit $t)"; timeout --foreground -k 30 "$t" "$@"; echo "rc=$?"; }

export TRITON_DEFAULT_FP_FUSION=1
# text path as served (server.py CUDA_DEFAULTS); the vision flags are toggled inside the tests
export INK_MOE=triton INK_GRAPH=1 INK_CHUNK=4096 INK_PREFILL_ATTN=auto INK_DEC_FUSE=1 INK_DEC_FAR_DOT=1 INK_MOE_V2=1 \
  INK_GLUE=triton INK_GLUE2=1 INK_MOE_FP4ACT=1 INK_FP8_DECODE=1

run 12m python tests/gpu_vision.py bench
cfg=$(grep -o 'BEST_TRITON_CFG=[0-9,]*' "$log" | tail -1 | cut -d= -f2)
best=$(grep -o 'BEST_ATTN=[a-z]*' "$log" | tail -1 | cut -d= -f2)
echo "triton tiles: ${cfg:-default}  fastest attention: ${best:-triton}"
run 12m env ${cfg:+INK_VIT_TRITON_CFG=$cfg} BEST_ATTN=${best:-triton} python tests/gpu_vision.py read
nvidia-smi --query-gpu=memory.used --format=csv,noheader
