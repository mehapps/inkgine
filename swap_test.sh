#!/usr/bin/env bash
# Test window for our engine: production down -> run CMD on the 5090 -> production back, health + chat checked.
# Production is the Inkling container since the 2026-11-23 cutover (PROD=llm-inkling-1); before it, Qwen (llm-sglang-1).
# Usage: engine/swap_test.sh MAX_MINUTES CMD [ARGS...]
#   e.g. engine/swap_test.sh 30 python -m engine.test_logits
# Failsafe: a systemd timer starts production at MAX_MINUTES+5 even if this shell is killed.
set -u
max=${1:?MAX_MINUTES}; shift; [ $# -gt 0 ] || { echo "need a command" >&2; exit 2; }
PROD=${PROD:-llm-inkling-1}
ts=$(date +%Y%m%d-%H%M%S); unit="prod-failsafe-$ts"
cd "$(dirname "$0")/.."
stop="$PWD/engine/stop_gpu.sh"   # kills our test server (TERM, then -9 after 30 s) + oracle, waits up to 60 s for free VRAM

exec 9>/tmp/gpu.lock
flock -n 9 || { echo "GPU lock held (another test/render)" >&2; exit 1; }
# shortcut: H3 scripts don't take /tmp/gpu.lock yet; this check catches a render that is already on the GPU.
prodpids=" $(docker top "$PROD" -eo pid 2>/dev/null | tail -n +2 | tr '\n' ' ') "
others=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader |
         while IFS=', ' read -r pid name; do case "$prodpids" in *" $pid "*) ;; *) echo "$pid $name";; esac; done | grep -v sglang)
[ -z "$others" ] || { echo "GPU busy with: $others" >&2; exit 1; }

docker logs --tail 2000 "$PROD" > "logs/prod-pre-swap-$ts.log" 2>&1
systemd-run --user --quiet --unit="$unit" --on-active="$((max + 5))m" \
  sh -c "'$stop'; docker start $PROD" \
  || { echo "failsafe timer not created, aborting before stopping production" >&2; exit 1; }

restore() {
  trap - EXIT INT TERM
  echo "[restore] starting production ($PROD)"
  "$stop"                                           # leftovers from a killed window would hold the GPU
  docker start "$PROD" >/dev/null
  for _ in $(seq 1 90); do curl -sf -o /dev/null http://127.0.0.1:30000/health && break; sleep 10; done
  if curl -sf -m 300 http://127.0.0.1:30000/v1/chat/completions -H 'Content-Type: application/json' \
       -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Say OK."}],"max_tokens":256}' >/dev/null; then
    systemctl --user stop "$unit.timer" 2>/dev/null
    echo "production back: health + chat OK"
  else
    echo "WARNING: production not answering; failsafe timer $unit left armed. docker logs --tail 30 $PROD"
  fi
}
trap restore EXIT INT TERM

echo "stopping production $PROD (window up to ${max} min)"
docker stop "$PROD" >/dev/null
timeout --kill-after=30 "${max}m" "$@"
echo "test exited rc=$?"
