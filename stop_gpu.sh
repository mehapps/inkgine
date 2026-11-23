#!/usr/bin/env bash
# Stop what our GPU tests can leave on the GPU (oracle container, our server.py, test scripts) and wait until its VRAM is
# free, so Qwen (or the next server) doesn't start into an OOM. Used by swap_test.sh (restore + failsafe) and tests/window.sh.
# Exit 0 = GPU clear of non-sglang compute processes; 1 = still busy after the bounded wait (caller goes on anyway).
# Targets only processes that hold the GPU (nvidia-smi) AND belong to llm/engine (cmdline or cwd): a shell whose command
# text mentions server.py never holds the GPU (a text-matching pkill once killed swap_test's own restore), and an H3
# render (ComfyUI under ~/h3) is never touched.
ours() {
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    { tr '\0' ' ' < /proc/"$p"/cmdline; readlink /proc/"$p"/cwd; } 2>/dev/null | grep -q 'llm/engine' && echo "$p"
  done
}
docker rm -f ink-oracle ink-soak >/dev/null 2>&1   # ink-soak: containerised engine, invisible to the pid scan below
p=$(ours); [ -n "$p" ] && kill $p 2>/dev/null
for _ in $(seq "${STOP_TERM_S:-30}"); do [ -z "$(ours)" ] && break; sleep 1; done
p=$(ours)
if [ -n "$p" ]; then
  echo "[stop_gpu] still holding the GPU after TERM: $p; kill -9"
  kill -9 $p 2>/dev/null
  for _ in $(seq 10); do [ -z "$(ours)" ] && break; sleep 1; done
fi
for _ in $(seq "${STOP_GPU_S:-60}"); do
  busy=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | grep -v sglang)
  [ -z "$busy" ] && exit 0
  sleep 1
done
echo "[stop_gpu] WARNING: GPU still has compute processes: $busy"
exit 1
