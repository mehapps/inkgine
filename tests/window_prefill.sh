#!/usr/bin/env bash
# Prefill-speed window 1: per-component breakdown (current defaults + FP4 MoE), FP8 dense + glue (compile / triton) speed and KL
# gates, far-attention bf16 vs FP8-PV vs flex timings.
#   ~/llm/engine/swap_test.sh 55 ~/llm/engine/tests/window_prefill.sh
# Steps continue past a failure (all are measurements; none changes state). ~40 min expected.
set -u
cd "$(dirname "$0")/.." && export PATH=$PWD/.venv/bin:$PATH
log=~/llm/logs/ink-window-prefill-$(date +%Y%m%d-%H%M).log
exec > >(tee -a "$log") 2>&1
echo "log: $log"
trap '"$PWD/stop_gpu.sh"' EXIT
run() { local t=$1; shift; echo "== $* (limit $t)"; timeout --foreground -k 30 "$t" "$@"; echo "rc=$?"; }

export INK_MOE=triton INK_MOE_V2=1 INK_PREFILL_ATTN=auto
run 12m env INK_MOE_FP4ACT=1 python tests/prof_prefill.py --kernels
run 6m  env INK_MOE_FP4ACT=1 INK_FP8_DENSE=1 python tests/prof_prefill.py --pos 0,32768,262144 --skip-seq
run 6m  env INK_MOE_FP4ACT=1 INK_GLUE=compile python tests/prof_prefill.py --pos 0,32768,262144 --skip-seq
run 6m  env INK_MOE_FP4ACT=1 INK_GLUE=triton python tests/prof_prefill.py --pos 0,32768,262144 --skip-seq
run 6m  env INK_MOE_FP4ACT=1 INK_GLUE=triton INK_FP8_DENSE=1 python tests/prof_prefill.py --pos 0,8192,32768
run 10m env INK_MOE_FP4ACT=1 python tests/gpu_fp8.py
run 8m  python tests/gpu_prefill.py --ctx 4096,65536,262144,1040000 --fp8
