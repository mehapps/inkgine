#!/usr/bin/env bash
# Prefill-speed window 2 (after window 1: INK_GLUE=triton is the default candidate; FP8 dense / FP8-PV dropped):
# INK_GLUE2 (fused attention prologue + 4-bit KV write, output gate in the near kernel, residual + norms, attn_res +
# norm) on top of INK_GLUE=triton; round 3 on top of that: INK_GATEFUSE (loop gate: norms + down GEMM, fp32 up GEMM +
# tanh/blend + norm, 2 kernels), INK_MOETAIL (expert sum + post-MLP conv + residual + norm, 1 kernel), INK_FARSKIP
# (far-attention key-tile skipping with a provable box bound); FP4 MoE with one activation term; weighted to real
# traffic (median prompt 77k, p90 186k): chunk curves 0-192k with 77k / 186k totals, real cold prefills to 186k, KL
# gates (short prompts + a 2-chunk long prompt reaching far attention), host syncs + GPU-busy check, 8k vs 4k chunks at
# 1M (VRAM), the far tile-skip measurement on a 196k cold prefill of source code + the INK_FARSKIP kernel on the same
# captured data (time vs no skip / flex, error vs the flex-vs-triton noise), and a 196k source-code cold prefill A/B.
#   ~/llm/engine/swap_test.sh 130 ~/llm/engine/tests/window_prefill2.sh
# Measurements only; steps continue past a failure. ~75-85 min expected (window 1: a prof_prefill curve ~1-2 min;
# gpu_moe and gpu_fp8 dominate); caps sum to ~125.
set -u
cd "$(dirname "$0")/.." && export PATH=$PWD/.venv/bin:$PATH
log=~/llm/logs/ink-window-prefill2-$(date +%Y%m%d-%H%M).log
exec > >(tee -a "$log") 2>&1
echo "log: $log"
trap '"$PWD/stop_gpu.sh"' EXIT
run() { local t=$1; shift; echo "== $* (limit $t)"; timeout --foreground -k 30 "$t" "$@"; echo "rc=$?"; }

export INK_MOE=triton INK_MOE_V2=1 INK_PREFILL_ATTN=auto INK_MOE_FP4ACT=1 TRITON_DEFAULT_FP_FUSION=1
P=tests/prof_prefill.py
CURVE="--pos 0,16384,32768,65536,131072,196608 --skip-seq"
G1="INK_GLUE=triton"; G2="INK_GLUE=triton INK_GLUE2=1"; G3="$G2 INK_GATEFUSE=1 INK_MOETAIL=1"

# 1. correctness first: short-prompt KL (glue triton, glue2, both, fp8 paths, + gatefuse / moetail / both) + long
#    2-chunk prompt (glue2, fp8-PV, gatefuse + moetail, farskip 0 (must be KL 0) / 16 / 12)
run 15m python tests/gpu_fp8.py
# 2. chunk curves -> per-component ms at each P + 77k/186k totals (integrated); each new flag alone on G2, then stacked
run 4m  env $G1 python $P $CURVE
run 4m  env $G2 python $P $CURVE
run 4m  env $G2 INK_GATEFUSE=1 python $P $CURVE
run 4m  env $G2 INK_MOETAIL=1 python $P $CURVE
run 4m  env $G3 python $P $CURVE
run 4m  env $G3 INK_MOE_FP4_TERMS=1 python $P $CURVE
# 3. host syncs + top kernels (GPU busy: 'Self CUDA time total' vs chunk ms) with the new kernels
run 4m  env $G3 python $P --pos 0,65536 --syncs --kernels --skip-seq
# 4. real cold prefills (fresh seq, repeated 4k text block) to 186k, marks at 77824 / 186368 (G1 alone: window 1 curve)
run 8m  env $G2 python $P --pos 0 --seq 186368
run 8m  env $G3 python $P --pos 0 --seq 186368
# 5. 1M: 4k vs 8k chunks, peak VRAM (server free VRAM at 1M is ~1.3-1.8 GB)
run 6m  env $G3 python $P --pos 0,1040384 --skip-seq
run 8m  env $G3 python $P --T 8192 --pos 0,1040384 --skip-seq
# 6. far attention on source code: how much (row, tile) work holds negligible mass at 64k/128k/192k (oracle / box /
#    ball bounds), then the INK_FARSKIP kernel on the captured calls (BN 64/32, delta 0/12/16/20/24: ms vs no skip and
#    vs flex, skipped fraction, error vs noise; exit 1 if a delta is outside noise or exact mode isn't bit-identical)
run 18m env $G3 python tests/gpu_farskip.py --at 65536,131072,196608
#    end to end on the same source text, no capture: G3 vs G3 + INK_FARSKIP=16 (auto then uses the Triton far kernel
#    at every range, which is ~10% slower than flex beyond 64k without skipping)
run 5m  env $G3 python tests/gpu_farskip.py --at 77824,131072,196608 --time-only
run 5m  env $G3 INK_FARSKIP=16 python tests/gpu_farskip.py --at 77824,131072,196608 --time-only
# 7. FP4 MoE with one activation term: per-visit error/time + end-to-end KL vs noise (verdict printed)
run 25m env -u INK_MOE_FP4ACT INK_MOE_FP4_TERMS=1 python tests/gpu_moe.py   # it times old / v2 / fp4 itself
