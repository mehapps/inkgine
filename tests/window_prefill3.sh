#!/usr/bin/env bash
# Prefill window 3 (~12-15 min): the "cliff" past 32k in window 2's real sequential prefill. Hypothesis: flex far
# attention compiles on first use (engine warm-up doesn't warm it) and, before model.FLEX_TQ, once more per new
# 1024-row query bucket (186368 ends on a 2048-row chunk). Now one compiled shape (pad rows masked).
#   ~/llm/engine/swap_test.sh 25 ~/llm/engine/tests/window_prefill3.sh
# Measurements only; steps continue past a failure.
set -u
cd "$(dirname "$0")/.." && export PATH=$PWD/.venv/bin:$PATH
log=~/llm/logs/ink-window-prefill3-$(date +%Y%m%d-%H%M).log
exec > >(tee -a "$log") 2>&1
echo "log: $log"
trap '"$PWD/stop_gpu.sh"' EXIT
run() { local t=$1; shift; echo "== $* (limit $t)"; timeout --foreground -k 30 "$t" "$@"; echo "rc=$?"; }

export INK_MOE=triton INK_MOE_V2=1 INK_PREFILL_ATTN=auto INK_MOE_FP4ACT=1 TRITON_DEFAULT_FP_FUSION=1
export INK_GLUE=triton INK_GLUE2=1
P=tests/prof_prefill.py
# 1. flex tails: one compiled shape, flex == triton far within noise at T = 4096 / 2048 / 777 / 64 (first-call ms)
run 5m  python tests/gpu_flextail.py
# 2. cold sequential to 186k (window 2: 7.06 s at 65k, 8.37 s at 77k, 24.88 s at 186k); expect ~1.5 s less at 186k
run 6m  python $P --pos 0 --seq 186368
# 3. same with flex already compiled by the 64k curve point (= what Model.warm_flex in engine warm-up would give);
#    if 77k is still well above the curve's 6.6 s, the cliff is not compilation
run 6m  python $P --pos 0,65536 --seq 186368
