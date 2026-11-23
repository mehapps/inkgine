#!/usr/bin/env bash
# Decode speed flags window (2026-11-23, all flags off by default): INK_DEC_UNROLL, INK_DEC_PDL, INK_DEC_L2PF,
# INK_DEC_W2ALL, INK_DEC_FAR2 (+ INK_DEC_FAR_NSP), each alone and stacked. Measurements + exactness gates only; nothing
# is turned on. Stops at the first failure (user rule); swap_test then restores Qwen.
#   ~/llm/engine/swap_test.sh 120 ~/llm/engine/tests/window_decode2.sh
# Expected ~80-95 min: tune_decode x7 at 1.1k (~12) + far at 131k x2 + sweeps (~8), gpu_decode2 1k/40k/130k/512k/1M x 7
# configs (fused + batched B=1/2/4[/8], gates vs base; ~45-55 incl. the 1M prefill), gpu_spec with all flags (~10).
set -u
cd "$(dirname "$0")/.." && export PATH=$PWD/.venv/bin:$PATH
log=~/llm/logs/ink-window-decode2-$(date +%Y%m%d-%H%M).log
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
  INK_CHUNK=4096 INK_PREFILL_ATTN=auto
ALL="INK_DEC_UNROLL=1 INK_DEC_PDL=1 INK_DEC_L2PF=1 INK_DEC_W2ALL=1 INK_DEC_FAR2=1"

# 1. per-kernel breakdown (tune_decode: each kernel's launches replayed in a graph, then the whole step as one graph;
#    far kernel now included) at 1.1k for each flag alone and stacked
for f in "" INK_DEC_UNROLL=1 INK_DEC_PDL=1 INK_DEC_L2PF=1 INK_DEC_W2ALL=1 INK_DEC_FAR2=1 "$ALL"; do
  step 6m "tune_decode 1100 [${f:-base}]" env $f python tests/tune_decode.py 1100
done
# far attention per kernel at 131k (base split kernel vs FAR2), FAR2 tile/warps/stages sweep, W2ALL block sweep
step 8m "tune_decode 131072 [base]" python tests/tune_decode.py 131072
step 10m "tune_decode 131072 [FAR2] + far_dot sweep" env INK_DEC_FAR2=1 python tests/tune_decode.py 131072 --sweep --only far_dot
step 8m "tune_decode 1100 [W2ALL UNROLL] w2all sweep" env INK_DEC_W2ALL=1 INK_DEC_UNROLL=1 python tests/tune_decode.py 1100 --sweep --only w2all

# 2. fused + batched tok/s at ~1k / 40k / 130k / 512k / 1M for every config, exactness gates vs base, FAR_NSP sweep
step 75m "gpu_decode2 (all configs, 1k..1M)" python tests/gpu_decode2.py --long --512k --1m --nsp 32,64,84

# 3. speculative decoding with all flags: greedy spec == plain (DSpark verify rows go through decode_batch)
step 15m "gpu_spec with all decode2 flags" env $ALL INK_GRAPH=1 python tests/gpu_spec.py
echo "WINDOW PASSED; min free VRAM: $(cat /tmp/ink-vram-low) MiB"
