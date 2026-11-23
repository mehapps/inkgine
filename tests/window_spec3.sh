#!/usr/bin/env bash
# Spec-in-the-server window (11-23 engine wiring): the server with the DSpark round-cost flags, device draft ids and
# spec for requests with logit processors. Three server configs, each: startup log shows the spec flags, then
# gpu_features spec / specproc / logprobs / params / json / tools against it.
#   A: INK_DSPARK_SHORT=32768 INK_DSPARK_CTXB=1 INK_SPEC_GATE=1 INK_SPEC_ACCEPT_DEV=1
#   B: INK_DSPARK_SHORT=32768 INK_DSPARK_CTXB=1 INK_SPEC_ACCEPT_DEV=1 INK_DEC_EMBED_GPU=1 INK_SAMPLE_F64=1
#      (no gate: device draft ids via bd.step(rows, tok=...); embedding table on the GPU, +0.9 GB)
#   C: A + INK_FP8_DECODE=1 (drafter on the FP8 unembed, shortlist)
#   ~/llm/engine/swap_test.sh 60 ~/llm/engine/tests/window_spec3.sh
# ~40 min expected (3 x (server start + warm-up ~2-3, features ~8-10)). Stops at the first failure (user rule);
# swap_test then restores Qwen. No needles here: B holds +0.9 GB (not for 1M until gpu_mem says so).
set -u
cd "$(dirname "$0")/.." && export PATH=$PWD/.venv/bin:$PATH
log=~/llm/logs/ink-window-spec3-$(date +%Y%m%d-%H%M).log
exec > >(tee -a "$log") 2>&1
echo "log: $log"
cleanup() { "$PWD/stop_gpu.sh"; kill "$WATCH" 2>/dev/null; }
trap cleanup EXIT
trap 'exit 1' INT TERM
( low=99999; while :; do f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits); \
  [ "$f" -lt "$low" ] && low=$f && echo "$low" > /tmp/ink-vram-low; sleep 1; done ) & WATCH=$!
fail() { echo "STOPPING WINDOW: $1 failed"; echo "min free VRAM: $(cat /tmp/ink-vram-low) MiB"; exit 1; }
step() { local t=$1 name=$2; shift 2; echo "== $name (limit $t)"; timeout --foreground -k 30 "$t" "$@" || fail "$name"; }
wait_http() { for _ in $(seq "$2"); do curl -sf -o /dev/null "$1" && return 0; sleep 1; done; return 1; }

config() {  # name, env assignments...
  local name=$1; shift
  "$PWD/stop_gpu.sh" >/dev/null 2>&1
  SLOG=~/llm/logs/ink-server-spec3-$name-$(date +%H%M%S).log
  env "$@" python "$PWD/server.py" > "$SLOG" 2>&1 &
  wait_http http://localhost:30000/health 600 || fail "server start $name (log $SLOG)"
  grep -m1 '^spec ' "$SLOG" || fail "server $name: no spec flags line (spec off?)"
  echo "server $name up; GPU: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
  for t in spec specproc logprobs params json tools; do
    step 15m "$name: gpu_features $t" python tests/gpu_features.py "$t"
  done
  curl -s localhost:30000/metrics | grep -E 'inkling_(ooms|decode_steps|decode_rows)' || true
}

R="INK_DSPARK_SHORT=32768 INK_DSPARK_CTXB=1 INK_SPEC_ACCEPT_DEV=1"
config A $R INK_SPEC_GATE=1
config B $R INK_DEC_EMBED_GPU=1 INK_SAMPLE_F64=1
config C $R INK_SPEC_GATE=1 INK_FP8_DECODE=1
echo "WINDOW PASSED; min free VRAM: $(cat /tmp/ink-vram-low) MiB"
