#!/usr/bin/env bash
# FP8 dense decode window (INK_FP8_DECODE): VRAM, KL gates + speed A/B vs bf16, kernel tuning data, DSpark spec with FP8,
# then a server with FP8 decode: gate image, needles 130k + 1M, quick evals (compared with the last bf16 quick run).
#   ~/llm/engine/swap_test.sh 100 ~/llm/engine/tests/window_fp8dec.sh
# Expected ~70-85 min: vram x2 ~5, gpu_fp8dec --long ~25, tune_decode ~6, gpu_spec ~15-25, server + gate ~5,
# needles ~12, quick evals ~10. Stops at the first failure (user rule); swap_test then restores Qwen.
set -u
cd "$(dirname "$0")/.." && export PATH=$PWD/.venv/bin:$PATH
log=~/llm/logs/ink-window-fp8dec-$(date +%Y%m%d-%H%M).log
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

export INK_MOE=triton INK_DEC_FAR_DOT=1
step 5m  "VRAM at 1M, bf16 dense"  env INK_FP8_DECODE=0 python tests/gpu_fp8dec.py --vram
step 5m  "VRAM at 1M, fp8 dense (bf16 copies dropped)" env INK_FP8_DECODE=1 python tests/gpu_fp8dec.py --vram
step 35m "gpu_fp8dec.py --long (kernels, KL gates, speed A/B)" python tests/gpu_fp8dec.py --long
step 10m "tune_decode.py fp8 (per-kernel %peak, graph gap; proj/wo sweep)" env INK_FP8_DECODE=1 python tests/tune_decode.py 1100 --sweep --only proj,wo
step 30m "gpu_spec.py with FP8 decode (greedy spec == plain)" env INK_FP8_DECODE=1 INK_GRAPH=1 python tests/gpu_spec.py

export INK_FP8_DECODE=1 INK_CHUNK=4096 INK_GRAPH=1
SLOG=~/llm/logs/ink-server-fp8dec-$(date +%H%M%S).log
python "$PWD/server.py" > "$SLOG" 2>&1 &
wait_http http://localhost:30000/health 600 || fail "server start (log $SLOG)"
echo "server up (INK_FP8_DECODE=1); GPU: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
step 10m "gate image (16.7 MP image, then json_schema)" env SGL=http://localhost:30000 python ~/llm/gate/gate.py image
step 10m "needle 130k" python tests/long_ctx.py 130000
step 25m "needle 1M" python tests/long_ctx.py 1040000
out=~/llm/logs/evals-ink-fp8dec-quick.json
step 20m "quick evals" python tests/evals/run_all.py --base http://localhost:30000 --quick --label fp8dec --out "$out"
[ -f ~/llm/logs/evals-ink-quick.json ] && python tests/evals/run_all.py --compare ~/llm/logs/evals-ink-quick.json "$out"
curl -s localhost:30000/metrics | grep -E 'inkling_(ooms|last_prefill|last_decode)' || true
echo "WINDOW PASSED; min free VRAM: $(cat /tmp/ink-vram-low) MiB"
