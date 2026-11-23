#!/usr/bin/env bash
# DSpark round-cost window: the current DSpark path vs the new flags, one gpu_spec.py run per combination (greedy,
# T=0.7, T=1.0 on code / json / chat / prose; tok/s, acceptance per position, round-cost breakdown, drafter ms at
# 0.3k/8k/30k context, verify step ms at T=1..8), plus the sampler fidelity test on cuda.
#   ~/llm/engine/swap_test.sh 70 ~/llm/engine/tests/window_spec2.sh
# ~40 min expected (6 gpu_spec runs ~5-6 min each incl. load, sampler ~2 min). Measurements only, no state changes:
# steps continue past a failure; exit status = 1 if any step failed (a greedy-equality FAIL in gpu_spec is a bug:
# report it, don't ship that flag).
set -u
cd "$(dirname "$0")/.." && export PATH=$PWD/.venv/bin:$PATH
log=~/llm/logs/ink-window-spec2-$(date +%Y%m%d-%H%M).log
exec > >(tee -a "$log") 2>&1
echo "log: $log"
trap '"$PWD/stop_gpu.sh"' EXIT
rc=0
run() { local t=$1; shift; echo "== $* (limit $t)"; timeout --foreground -k 30 "$t" "$@"; local r=$?; echo "rc=$r"; [ $r -eq 0 ] || rc=1; }

export INK_MOE=triton INK_MOE_V2=1 INK_DEC_FAR_DOT=1 INK_GRAPH=1
run 5m  env INK_DEVICE=cuda INK_TAIL_N=4000 python tests/test_sampler.py
run 10m python tests/gpu_spec.py                                                       # current DSpark path
run 10m env INK_DSPARK_SHORT=32768 python tests/gpu_spec.py                            # (a) shortlist head
run 10m env INK_DSPARK_SHORT=32768 INK_DSPARK_CTXB=1 python tests/gpu_spec.py          # + layer-1 ctx buckets
run 10m env INK_DSPARK_SHORT=32768 INK_DSPARK_CTXB=1 INK_SPEC_GATE=1 python tests/gpu_spec.py   # (d) + conf gate
run 10m env INK_DSPARK_SHORT=32768 INK_DSPARK_CTXB=1 INK_SPEC_GATE=1 INK_SPEC_ACCEPT_DEV=1 python tests/gpu_spec.py
run 10m env INK_FP8_DECODE=1 INK_DSPARK_SHORT=32768 INK_DSPARK_CTXB=1 INK_SPEC_GATE=1 INK_SPEC_ACCEPT_DEV=1 python tests/gpu_spec.py  # + FP8 decode, drafter on the FP8 unembed
echo "== summaries"; grep -A13 '== summary' "$log" | grep -v '^--$'
exit $rc
