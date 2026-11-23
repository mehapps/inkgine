#!/usr/bin/env bash
# Soak window: start the engine (container by default) with the candidate flags, run tests/soak.py for HOURS, collect
# a report. Run it only inside a GPU window (user OK):
#   ~/llm/engine/swap_test.sh $((HOURS*60+40)) ~/llm/engine/tests/window_soak.sh [HOURS]
#   default 2 h: ~/llm/engine/swap_test.sh 160 ~/llm/engine/tests/window_soak.sh 2
# Env knobs:
#   SOAK_FLAGS   engine flags for this run (default: every pending candidate flip, below); SOAK_FLAGS= = current defaults
#   SOAK_MODE    container (default; image inkling-engine:local, built from deploy/Dockerfile) | venv
#   SOAK_CONCURRENCY (6), SOAK_ARGS (extra tests/soak.py args), SOAK_PORT (30000)
# Budget (minutes): engine start <= 20, smoke 3, soak HOURS*60, final drain + idle <= 6. The server process gets its own
# `timeout` of HOURS*60+35 min, so it is gone before swap_test's limit even if this shell is SIGKILLed (stop_gpu.sh
# does not see a process inside a container).
# Output: ~/llm/logs/ink-window-soak-TS.log (this log), ink-soak-TS.json (soak report), ink-soak-TS-server.log.
set -u
H=${1:-2}
cd "$(dirname "$0")/.." && export PATH=$PWD/.venv/bin:$PATH
ENGINE=$PWD
ts=$(date +%Y%m%d-%H%M); L=~/llm/logs; mkdir -p "$L"
log=$L/ink-window-soak-$ts.log; rep=$L/ink-soak-$ts.json; slog=$L/ink-soak-$ts-server.log
exec > >(tee -a "$log") 2>&1
echo "log: $log"
MODE=${SOAK_MODE:-container}
FLAGS=${SOAK_FLAGS-INK_MOE_FP4ACT=1 INK_FP8_DECODE=1 INK_GLUE=triton INK_DSPARK_SHORT=32768 INK_DSPARK_CTXB=1 INK_SPEC_GATE=1 INK_SPEC_ACCEPT_DEV=1}
LIFE=$((H * 60 + 35))m
NAME=ink-soak
P=${SOAK_PORT:-30000}; B=http://127.0.0.1:$P
MODELS=${INK_MODEL:-$HOME/llm/models/Inkling-2-Mini-Uncensored}
CACHE=${INK_CACHE_DIR:-$HOME/llm/inkcache}
echo "mode $MODE, $H h, flags: ${FLAGS:-<server defaults>}"

cleanup() {
  if [ "$MODE" = container ]; then
    docker logs "$NAME" > "$slog" 2>&1
    docker rm -f "$NAME" >/dev/null 2>&1
  else
    "$ENGINE/stop_gpu.sh"
  fi
  echo "server log: $slog"
}
trap cleanup EXIT

if [ "$MODE" = container ]; then
  docker image inspect inkling-engine:local >/dev/null 2>&1 || { echo "image inkling-engine:local not built (deploy/Dockerfile)"; exit 1; }
  docker rm -f "$NAME" >/dev/null 2>&1
  mkdir -p ~/.cache/inkling-engine/home ~/.cache/inkling-engine/triton ~/.cache/inkling-engine/inductor "$CACHE"
  envs=(); for f in $FLAGS; do envs+=(-e "$f"); done
  # Same env/mounts as deploy/compose.inkling.yaml, but a standalone container (not part of the live compose project).
  docker run -d --name "$NAME" --gpus all -u "$(id -u):$(id -g)" -p 127.0.0.1:$P:30000 \
    -e TZ=America/New_York -e INK_HOST=0.0.0.0 -e INK_PORT=30000 -e INK_MODEL=/models/inkling -e INK_CACHE_DIR=/inkcache \
    -e HOME=/cache/home -e XDG_CACHE_HOME=/cache -e TRITON_CACHE_DIR=/cache/triton -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor \
    "${envs[@]}" \
    -v "$ENGINE:/app:ro" -v "$MODELS:/models/inkling:ro" -v "$CACHE:/inkcache" -v ~/.cache/inkling-engine:/cache \
    inkling-engine:local timeout -k 30 "$LIFE" python server.py >/dev/null || exit 1
  alive() { [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" = true ]; }
else
  env $FLAGS INK_PORT=$P timeout -k 30 "$LIFE" python server.py > "$slog" 2>&1 &
  spid=$!
  alive() { kill -0 "$spid" 2>/dev/null; }
fi

echo "[$(date +%T)] waiting for /health (max 20 min)"
for _ in $(seq 240); do
  curl -sf $B/health | grep -q true && break
  alive || { echo "server died during start"; exit 1; }
  sleep 5
done
curl -sf $B/health | grep -q true || { echo "server not healthy after 20 min"; exit 1; }
echo "[$(date +%T)] healthy; $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)"
curl -sf $B/metrics | grep -E 'warmup_seconds|slots|batch_decoder'

echo "[$(date +%T)] smoke (3 min): stop here if the basic path fails"
timeout -k 30 6m python tests/soak.py --base $B --smoke --out "$L/ink-soak-$ts-smoke.json" || { echo "SMOKE FAILED: not soaking"; exit 1; }

echo "[$(date +%T)] soak $H h"
timeout -k 30 $((H * 60 + 8))m python tests/soak.py --base $B --duration "${H}h" --concurrency "${SOAK_CONCURRENCY:-6}" \
  --drain-s 300 --out "$rep" ${SOAK_ARGS:-}
rc=$?
echo "[$(date +%T)] soak rc=$rc"
curl -sf $B/metrics > "$L/ink-soak-$ts-metrics.txt" && echo "final metrics: $L/ink-soak-$ts-metrics.txt"
du -sh "$CACHE"
echo "report: $rep"
exit $rc
