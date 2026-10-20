#!/usr/bin/env bash
# Stop the API server, benchmark and model download left running by run-wsl.sh, run-mac.sh or
# benchmark.bat (e.g. after closing the terminal window), so they release GPU and RAM.
#
#   bash scripts/stop.sh
#
# Asks each process to exit, then kills whatever is still running after 10 seconds.
set -uo pipefail

pattern='uvicorn main:app|scripts/benchmark\.py|omnivoice_api\.download'

if ! pgrep -f "$pattern" >/dev/null; then
  echo "Nothing to stop."
else
  pgrep -af "$pattern"
  pkill -TERM -f "$pattern"
  for _ in {1..10}; do
    pgrep -f "$pattern" >/dev/null || break
    sleep 1
  done
  if pgrep -f "$pattern" >/dev/null; then
    echo "Still running after 10 seconds; killing."
    pkill -KILL -f "$pattern"
  fi
  echo "Stopped."
fi

[[ -x /usr/lib/wsl/lib/nvidia-smi ]] && PATH="/usr/lib/wsl/lib:$PATH"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=memory.used,memory.total --format=csv
exit 0
