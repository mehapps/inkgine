#!/usr/bin/env bash
# Prefill window 4: exact MoE savings over the current defaults (INK_GLUE=triton INK_GLUE2=1 INK_MOE_FP4ACT=1, 2 terms).
# Flags (all off by default, all meant bit-identical): INK_MOE_ROUTE (routed-row sort + block tables in 3 launches),
# INK_MOE_AMAX (act row amax from the w13 epilogue, one-pass u amax), tile / block-size winners of the sweep
# (INK_MOE_FP4_CFG / _FP4_BB / _FP4_BS, INK_MOE_MX_CFG / _BB / _BS; gpu_moe4 writes them to $best).
#   ~/llm/engine/swap_test.sh 60 ~/llm/engine/tests/window_prefill4.sh
# Measurements only; steps continue past a failure. ~35-45 min expected; caps sum to ~57.
set -u
cd "$(dirname "$0")/.." && export PATH=$PWD/.venv/bin:$PATH
stamp=$(date +%Y%m%d-%H%M)
log=~/llm/logs/ink-window-prefill4-$stamp.log
bestf=~/llm/logs/ink-moe4-best-$stamp.env
exec > >(tee -a "$log") 2>&1
echo "log: $log"
trap '"$PWD/stop_gpu.sh"' EXIT
run() { local t=$1; shift; echo "== $* (limit $t)"; timeout --foreground -k 30 "$t" "$@"; echo "rc=$?"; }

export INK_MOE=triton INK_MOE_V2=1 INK_PREFILL_ATTN=auto INK_MOE_FP4ACT=1 INK_GLUE=triton INK_GLUE2=1
export TRITON_DEFAULT_FP_FUSION=1
P=tests/prof_prefill.py
CURVE="--pos 0,16384,32768,65536,131072,196608 --skip-seq"

# 1. MoE: per-visit torch.equal vs the default path + ms (routing included) for each flag; tile sweep -> $bestf;
#    end-to-end KL (short prompts + 2x4096 long prompt) for each flag, stacked, and the tile winners
: > "$bestf"
run 20m python tests/gpu_moe4.py --best "$bestf"
best=$(cat "$bestf" 2>/dev/null)
echo "tile winners: ${best:-none}"
S="INK_MOE_ROUTE=1 INK_MOE_AMAX=1 $best"
# 2. chunk curves 0-192k (77k / 186k totals integrated): defaults, each flag alone, tile winners alone, stacked
run 4m  python $P $CURVE
run 4m  env INK_MOE_ROUTE=1 python $P $CURVE
run 4m  env INK_MOE_AMAX=1 python $P $CURVE
[ -n "$best" ] && run 4m env $best python $P $CURVE
run 4m  env $S python $P $CURVE
# 3. top kernels + host syncs with the stack (route must add no host sync)
run 4m  env $S python $P --pos 0,65536 --syncs --kernels --skip-seq
# 4. real cold prefill to 186k (marks at 77824 / 186368): defaults vs stack (window 3 defaults: 8.4 s / 24.9 s)
run 7m  python $P --pos 0 --seq 186368
run 7m  env $S python $P --pos 0 --seq 186368
