#!/usr/bin/env bash
# Long-context speculative decoding window (11-23 long-spec agent): DSpark + copy hybrid (server defaults) at 130k /
# 512k / 1M. Launch through swap_test.sh (stops and restores Qwen):
#   ~/llm/engine/swap_test.sh 45 ~/llm/engine/tests/window_longspec.sh
# Steps (expected ~20 min of GPU time, ~25-30 min with the Qwen stop/restore):
#   1. far kernel on the GPU (~2 min, Triton compiles): a verify group of <= 10 rows reads the far KV once and each
#      row's far partials == its 1-row step BIT FOR BIT (greedy spec == plain at long context); stop if not
#   2. server with defaults (~2 min load + warm-up)
#   3. tests/longspec.py: per size one agent session (cold prefill once: 130k ~15 s, 512k ~2.5 min, 1M ~6.5 min), then
#      a tool-call turn (edit_file copying a function ~SIZE tokens back) and a code-edit turn (a class copied with a
#      rename), each spec on vs spec: false: decode tok/s, ratio, greedy outputs identical (~1-2 min per size)
#   4. server log: drafter / VRAM decisions, OOMs, 500s, tracebacks; min free VRAM over the window
# Stops at the first failure (user rule: swap_test then restores Qwen; fix offline, wait for the user's OK).
set -u
cd "$(dirname "$0")/.." && export PATH=$PWD/.venv/bin:$PATH
log=~/llm/logs/ink-window-longspec-$(date +%Y%m%d-%H%M).log
exec > >(tee -a "$log") 2>&1
echo "log: $log"
cleanup() { "$PWD/stop_gpu.sh"; kill "$WATCH" 2>/dev/null; }
trap cleanup EXIT
trap 'exit 1' INT TERM
( low=99999; while :; do f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits); \
  [ "$f" -lt "$low" ] && low=$f && echo "$low" > /tmp/ink-vram-low; sleep 1; done ) & WATCH=$!
SLOG=~/llm/logs/ink-server-longspec-$(date +%H%M%S).log
report() {
  echo "== server log: drafter / VRAM / errors ($SLOG)"
  grep -E "spec (dspark|lookup-only|off)|VRAM|OOM|out of memory|Traceback|Error|\" 5[0-9][0-9] " "$SLOG" | sort | uniq -c | head -40
  curl -s http://localhost:30000/metrics | grep -E "^inkling_(ooms_total|slot_decoders|last_decode_tps)"
  echo "min free VRAM: $(cat /tmp/ink-vram-low) MiB"
}
fail() { echo "STOPPING WINDOW: $1 failed"; [ -f "$SLOG" ] && report; exit 1; }
step() { local t=$1 name=$2; shift 2; echo "== $name (limit $t)"; timeout --foreground -k 30 "$t" "$@" || fail "$name"; }

step 10m "far kernel: verify group == 1-row steps bit for bit (GPU)" \
  env TRITON_INTERPRET=0 python tests/test_decode2.py test_bfar_verify
echo "== server (defaults)"
python "$PWD/server.py" > "$SLOG" 2>&1 &
for _ in $(seq 600); do curl -sf -o /dev/null http://localhost:30000/health && break; sleep 1; done
curl -sf -o /dev/null http://localhost:30000/health || fail "server start"
echo "server up; GPU: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
step 40m "long-context spec sessions 130k / 512k / 1M" python tests/longspec.py 130000 512000 1030000
report
grep -qE "Traceback|\" 5[0-9][0-9] |out of memory" "$SLOG" && fail "server log clean (no 500 / OOM / traceback)"
echo "WINDOW PASSED; min free VRAM: $(cat /tmp/ink-vram-low) MiB"
