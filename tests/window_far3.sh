#!/usr/bin/env bash
# Far decode attention window (2026-11-23, INK_DEC_FAR3, off by default): measurements + exactness gates only, nothing
# is turned on. Stops at the first failure (user rule); swap_test then restores Qwen.
#   ~/llm/engine/swap_test.sh 60 ~/llm/engine/tests/window_far3.sh
# Expected ~35-40 min: gpu_far3 sweep on a synthetic 1M store (~8-12), tune_decode at 131k FAR2 vs FAR3 (~6),
# gpu_decode2 base,FAR2,FAR3 at 1k/40k/131k/512k/1M with the sweep's best FAR3 config (~18-22, 1M prefill ~6 of it).
set -u
cd "$(dirname "$0")/.." && export PATH=$PWD/.venv/bin:$PATH
log=~/llm/logs/ink-window-far3-$(date +%Y%m%d-%H%M).log
exec > >(tee -a "$log") 2>&1
echo "log: $log"
cleanup() { "$PWD/stop_gpu.sh"; kill "$WATCH" 2>/dev/null; }
trap cleanup EXIT
trap 'exit 1' INT TERM
( low=99999; while :; do f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits); \
  [ "$f" -lt "$low" ] && low=$f && echo "$low" > /tmp/ink-vram-low; sleep 1; done ) & WATCH=$!
fail() { echo "STOPPING WINDOW: $1 failed"; echo "min free VRAM: $(cat /tmp/ink-vram-low) MiB"; exit 1; }
step() { local t=$1 name=$2; shift 2; echo "== $name (limit $t)"; timeout --foreground -k 30 "$t" "$@" || fail "$name"; }

# the server's decode defaults (server.py CUDA_DEFAULTS)
export INK_MOE=triton INK_DEC_FAR_DOT=1 INK_FP8_DECODE=1 INK_MOE_V2=1 INK_GLUE=triton INK_GLUE2=1 INK_MOE_FP4ACT=1 \
  INK_CHUNK=4096 INK_PREFILL_ATTN=auto TRITON_DEFAULT_FP_FUSION=1

# 1. far kernel alone: ms + GB/s per visit (8 launches in a graph) at 131k/524k/1040k far keys, FAR2 vs the FAR3
#    BLK x stages x warps x splits grid, streaming-read ceiling, FAR3 vs FAR2 vs fp32 exactness; best config -> env file
best=/tmp/ink-far3-best.env
rm -f "$best"
step 20m "gpu_far3 sweep" python tests/gpu_far3.py --out "$best"
FAR3_ENV=$(cat "$best")
echo "FAR3 config for the rest of the window: $FAR3_ENV"

# 2. whole step per kernel at 131k (far kernel's share and GB/s inside a real step)
step 8m "tune_decode 131072 [FAR2]" env INK_DEC_FAR2=1 python tests/tune_decode.py 131072
step 8m "tune_decode 131072 [FAR3]" env INK_DEC_FAR3=1 $FAR3_ENV python tests/tune_decode.py 131072

# 3. fused tok/s + lockstep KL gates vs base (per-row 1-ulp noise) at ~1k/40k/131k/512k/1M; FAR3 fused only (the
#    batch path is FAR2's kernel); split-count sweep around the best at the long marks
step 40m "gpu_decode2 base,FAR2,FAR3" env $FAR3_ENV python tests/gpu_decode2.py --configs base,FAR2,FAR3 \
  --long --512k --1m --nsp 42,84,126
echo "WINDOW PASSED; min free VRAM: $(cat /tmp/ink-vram-low) MiB"
