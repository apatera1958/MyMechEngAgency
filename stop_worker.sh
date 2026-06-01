#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIDFILE="$ROOT/logs/worker.pid"
if [ ! -f "$PIDFILE" ]; then
  echo "No worker.pid found."
  exit 0
fi
PID="$(cat "$PIDFILE")"
if kill -0 "$PID" >/dev/null 2>&1; then
  kill "$PID"
  echo "Stopped worker PID=$PID"
else
  echo "Worker PID=$PID is not running."
fi
rm -f "$PIDFILE"
