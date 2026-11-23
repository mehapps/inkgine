#!/usr/bin/env bash
# Correctness rerun with all 11-23 defaults (g2) -> ckpt at a real 150k -> decode FAR2/UNR/PDL at 130k/512k/1M.
#   ~/llm/engine-win/swap_test.sh 150 ~/llm/engine-win/tests/window_final.sh
set -u
E="$(cd "$(dirname "$0")/.." && pwd)"; cd "$E"; export PATH=$E/.venv/bin:$PATH
st() { "$E/stop_gpu.sh"; }
echo "== window g2 $(date +%T)"; bash tests/window.sh g2 || { echo "WINDOW_FINAL: g2 failed"; exit 1; }
st
echo "== ckpt $(date +%T)"; L=~/llm/logs/ink-server-ckpt-$(date +%H%M).log
INK_CKPT_EVERY=32768 python server.py > "$L" 2>&1 &
timeout 900 bash -c 'until curl -sf localhost:30000/health >/dev/null; do sleep 2; done' && \
  for t in ckpt prefix; do timeout 15m python tests/gpu_features.py $t; echo "ckpt-$t rc=$?"; done
echo "500s: $(grep -c 'Internal Server Error' "$L")"; st
echo "== decode $(date +%T)"; timeout 45m env INK_MOE=triton INK_DEC_FAR_DOT=1 INK_FP8_DECODE=1 python tests/gpu_decode2.py --configs base,FAR2,BEST --long --512k --1m; echo "decode rc=$?"; st
echo "WINDOW_FINAL DONE $(date +%T)"
