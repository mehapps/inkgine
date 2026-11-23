#!/usr/bin/env bash
# Everything one GPU test window runs. Always launch through swap_test.sh (it stops and restores Qwen), e.g.
#   ~/llm/engine/swap_test.sh 75 ~/llm/engine/tests/window.sh e   # engine + server features; ~40-45 min expected
#   ~/llm/engine/swap_test.sh 50 ~/llm/engine/tests/window.sh f   # oracle capture + compare; ~20-30 min expected
# e: gpu_check ~3 (window d: 2 min), gpu_decode ~10, server start ~1, gate image ~2, gpu_features ~6, needle 130k ~3,
#    needle 1M ~12-16 (986 s with torch MoE) => ~40-45 min; the one allowed retry (1M OOM -> INK_GRAPH=0) adds ~17.
# g: prefill measurements ~40-55, gpu_spec ~10-25, server+gate+features ~15, needles ~10, quick evals ~10 => ~90-115 min
#   ~/llm/engine/swap_test.sh 130 ~/llm/engine/tests/window.sh g
# g2: re-verify of g's server failures, then g's server part again (no prefill measurements / gpu_spec): ~60-75 min
#   ~/llm/engine/swap_test.sh 90 ~/llm/engine/tests/window.sh g2
# f: oracle start <= 20 (wait limit), oracle.py ~5-10, compare ~5.
# Both stop at the first failure (user rule): swap_test then restores Qwen; fix offline, wait for the user's OK.
# Each step has its own timeout. Phases a-d (windows of 2026-11-21) are superseded by e + f (see NOTES.md log).
set -u
cd "$(dirname "$0")/.." && export PATH=$PWD/.venv/bin:$PATH
phase=${1:-e}
log=~/llm/logs/ink-window-$(date +%Y%m%d-%H%M)-$phase.log
exec > >(tee -a "$log") 2>&1
echo "log: $log"

cleanup() {
  "$PWD/stop_gpu.sh"
  kill "$WATCH" 2>/dev/null
}
trap cleanup EXIT
trap 'exit 1' INT TERM
# min free VRAM over the whole window
( low=99999; while :; do f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits); \
  [ "$f" -lt "$low" ] && low=$f && echo "$low" > /tmp/ink-vram-low; sleep 1; done ) & WATCH=$!

fail() { echo "STOPPING WINDOW: $1 failed"; echo "min free VRAM: $(cat /tmp/ink-vram-low) MiB"; exit 1; }

# --foreground keeps the step in our process group, so swap_test's outer timeout reaches it too
step() {  # limit, name, command...
  local t=$1 name=$2; shift 2
  echo "== $name (limit $t)"
  timeout --foreground -k 30 "$t" "$@" || fail "$name"
}

wait_http() {  # url, seconds
  for _ in $(seq "$2"); do curl -sf -o /dev/null "$1" && return 0; sleep 1; done; return 1
}

start_server() {
  SLOG=~/llm/logs/ink-server-$(date +%H%M%S).log
  python "$PWD/server.py" > "$SLOG" 2>&1 &
  wait_http http://localhost:30000/health 600 || { echo "server did not start"; return 1; }
  echo "server up (INK_MOE=$INK_MOE INK_CHUNK=$INK_CHUNK INK_GRAPH=$INK_GRAPH; other fast paths: server.py cuda defaults unless set); GPU: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
}

# CUDA OOM with graphs (their private pool holds ~0.5 GB): restart once with INK_GRAPH=0 and rerun
needle() {  # prompt tokens, limit
  echo "== needle $1 (limit $2)"
  timeout --foreground -k 30 "$2" python tests/long_ctx.py "$1" && return 0
  grep -q "OutOfMemory\|out of memory" "$SLOG" && [ "$INK_GRAPH" = 1 ] || fail "needle $1"
  echo "CUDA OOM with INK_GRAPH=1; retrying once with INK_GRAPH=0"
  "$PWD/stop_gpu.sh"; export INK_GRAPH=0
  start_server || fail "server restart (INK_GRAPH=0)"
  timeout --foreground -k 30 "$2" python tests/long_ctx.py "$1" || fail "needle $1 (INK_GRAPH=0)"
}

if [ "$phase" = e ]; then
  step 10m "gpu_check.py (MoE torch / triton / exact)" python tests/gpu_check.py
  step 20m "gpu_decode.py (graph decode vs current)" python tests/gpu_decode.py
  export INK_MOE=triton INK_CHUNK=4096 INK_GRAPH=1   # 4k = verified peak 28.3 GiB at 1M
  start_server || fail "server start"
  step 10m "gate image (16.7 MP image, then json_schema)" env SGL=http://localhost:30000 python ~/llm/gate/gate.py image
  step 15m "gpu_features.py (json, priority + preemption resume, prefix reuse GPU + disk)" python tests/gpu_features.py all
  needle 130000 10m
  needle 1040000 25m
elif [ "$phase" = f ]; then
  export INK_MOE=triton INK_CHUNK=4096
  echo "== oracle start (official SGLang image, black box) on :30001"
  docker run -d --name ink-oracle --gpus all --ipc=host -p 127.0.0.1:30001:30000 \
    -v "$HOME/llm/models:/models:ro" lmsysorg/sglang:inkling-2-mini \
    sglang serve --model-path /models/Inkling-2-Mini-Uncensored --host 0.0.0.0 --port 30000 --tp 1 \
    --attention-backend fa4 --dtype bfloat16 --page-size 128 --enable-multimodal \
    --reasoning-parser inkling --tool-call-parser inkling2 --context-length 65536 >/dev/null || fail "oracle docker run"
  wait_http http://localhost:30001/health 1200 || fail "oracle start"
  step 20m "oracle.py (capture)" python tests/oracle.py http://localhost:30001
  docker logs ink-oracle > ~/llm/logs/ink-oracle-$(date +%H%M%S).log 2>&1
  "$PWD/stop_gpu.sh"   # removes the oracle, waits for its VRAM
  step 20m "compare.py (INK_MOE=triton vs oracle)" python tests/compare.py
elif [ "$phase" = g ]; then
  # 1) prefill measurements (no state change, never stops the window); own log ink-window-prefill-*.log
  echo "== window_prefill.sh (measurements, limit 55m)"
  timeout --foreground -k 30 55m bash "$PWD/tests/window_prefill.sh"; echo "window_prefill rc=$?"
  "$PWD/stop_gpu.sh"
  # 2) DSpark in the server + FP4 MoE (explicit; default not flipped yet). Stops at the first failure.
  step 25m "gpu_spec.py (greedy identical, kmax=0 ~x1.00)" env INK_MOE=triton INK_DEC_FAR_DOT=1 INK_GRAPH=1 python tests/gpu_spec.py
  export INK_MOE=triton INK_CHUNK=4096 INK_GRAPH=1 INK_MOE_FP4ACT=1
  start_server || fail "server start"
  step 10m "gate image (16.7 MP image, then json_schema)" env SGL=http://localhost:30000 python ~/llm/gate/gate.py image
  step 10m "gpu_features.py spec" python tests/gpu_features.py spec
  step 20m "gpu_features.py all (spec on)" python tests/gpu_features.py all
  needle 130000 10m
  needle 1040000 25m
  step 20m "quick evals" python tests/evals/run_all.py --base http://localhost:30000 --quick
  curl -s localhost:30000/metrics | grep -E 'inkling_(ooms|last_prefill|last_decode|slot)' || true
elif [ "$phase" = g2 ]; then
  # re-verify of window g server failures (11-23 fixes: drafter graph pool -> 500s, json cut with spec, priority
  # test with spec): spec first (fresh server, as in g, so tok/s compare), then the parts that failed, then the rest
  # of g's server part. Same server env as g. ~60-75 min.
  export INK_MOE=triton INK_CHUNK=4096 INK_GRAPH=1 INK_MOE_FP4ACT=1
  start_server || fail "server start"
  for t in spec specproc json priority prefix; do step 15m "gpu_features.py $t (spec on)" python tests/gpu_features.py $t; done
  ! grep -q "Internal Server Error" "$SLOG" || fail "a 500 in the server log ($SLOG)"
  grep "spec off for a request" "$SLOG" || echo "no spec-off fallbacks"
  echo "min free VRAM so far: $(cat /tmp/ink-vram-low) MiB"
  step 10m "gate image (16.7 MP image, then json_schema)" env SGL=http://localhost:30000 python ~/llm/gate/gate.py image
  step 25m "gpu_features.py all (spec on)" python tests/gpu_features.py all
  needle 130000 10m
  needle 1040000 25m
  step 20m "quick evals" python tests/evals/run_all.py --base http://localhost:30000 --quick
  ! grep -q "Internal Server Error" "$SLOG" || fail "a 500 in the server log ($SLOG)"
  curl -s localhost:30000/metrics | grep -E 'inkling_(ooms|last_prefill|last_decode|slot)' || true
else
  echo "unknown phase $phase (e, f, g or g2)"; exit 2
fi
echo "WINDOW PASSED; min free VRAM: $(cat /tmp/ink-vram-low) MiB"
