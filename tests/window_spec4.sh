#!/usr/bin/env bash
# Spec window 4 (11-23, CPU-built): hybrid drafting (DSpark + prompt lookup, INK_SPEC_HYBRID) and the fixed confidence
# gate (INK_SPEC_GATE: Width decides whether to draft), each alone and stacked, on top of the current server defaults.
# One gpu_spec.py --agent run per config: code / json / chat / prose + agent cases (tool: edit_file call copying from a
# read_file result; edit: rewritten class copying a provided file; reason: thinking text about it), greedy and T=0.7;
# tok/s plain vs spec, greedy equality gate, acceptance, copy / mixed round counts, verify step ms at T = 1..16.
#   ~/llm/engine/swap_test.sh 45 ~/llm/engine/tests/window_spec4.sh
# ~30 min expected (6 runs x ~3-4 min incl. load; agent prompts ~3k tokens). Measurements only: steps continue past a
# failure; exit 1 if any step failed (a greedy-equality FAIL = bug in that flag: don't ship it).
set -u
cd "$(dirname "$0")/.." && export PATH=$PWD/.venv/bin:$PATH
log=~/llm/logs/ink-window-spec4-$(date +%Y%m%d-%H%M).log
exec > >(tee -a "$log") 2>&1
echo "log: $log"
trap '"$PWD/stop_gpu.sh"' EXIT
rc=0
run() { local t=$1; shift; echo "== $* (limit $t)"; timeout --foreground -k 30 "$t" "$@"; local r=$?; echo "rc=$r"; [ $r -eq 0 ] || rc=1; }

# server.py CUDA_DEFAULTS as of 11-22 23:26 (tests importing engine directly don't get them)
export INK_MOE=triton INK_GRAPH=1 INK_CHUNK=4096 INK_PREFILL_ATTN=auto INK_DEC_FUSE=1 INK_DEC_FAR_DOT=1 \
  INK_GLUE=triton INK_GLUE2=1 INK_MOE_FP4ACT=1 INK_FP8_DECODE=1 INK_DSPARK_SHORT=32768 INK_DSPARK_CTXB=1 \
  INK_SPEC_ACCEPT_DEV=1
A="tests/gpu_spec.py --agent --temps 0,0.7"
run 8m python $A                                                    # baseline (current defaults)
run 8m env INK_SPEC_HYBRID=1 python $A                              # hybrid alone
run 8m env INK_SPEC_GATE=1 python $A                                # fixed gate alone
run 8m env INK_SPEC_HYBRID=1 INK_SPEC_GATE=1 python $A              # stacked
run 8m env INK_SPEC_HYBRID=1 INK_SPEC_SLEEP=0 python $A             # hybrid, never dormant (does sleep cost copies?)
run 8m env INK_SPEC_HYBRID=1 INK_SPEC_GATE=1 INK_SPEC_COPY_EDGE=1.5 python $A   # stacked, copies only when clearly better than DSpark
echo "== summaries"; grep -A16 '== summary' "$log" | grep -v '^--$'
exit $rc
