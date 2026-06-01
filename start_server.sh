#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="myagency-server"
HOST="${MYAGENCY_HOST:-127.0.0.1}"
PORT="${MYAGENCY_PORT:-5000}"

MINIFORGE_DIR="${MYAGENCY_MINIFORGE_DIR:-$ROOT/miniforge3}"
if [ ! -f "$MINIFORGE_DIR/etc/profile.d/conda.sh" ]; then
  echo "ERROR: Miniforge not found at: $MINIFORGE_DIR" >&2
  echo "Run: bash create_python_environment.sh" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$MINIFORGE_DIR/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

if [ -f "$ROOT/secrets/openai.env" ]; then
  # shellcheck disable=SC1090
  source "$ROOT/secrets/openai.env"
fi

export MYAGENCY_ROOT="$ROOT"
export PATH_AGENCY_V1="$ROOT/core"
export AGENCY_CONDA_ENV="$ENV_NAME"
export AGENCY_MINIFORGE_DIR="$MINIFORGE_DIR"
export AGENCY_PY="$MINIFORGE_DIR/envs/$ENV_NAME/bin/python"
export PYTHONPATH="$ROOT/web:${PYTHONPATH:-}"

mkdir -p "$ROOT/logs" "$ROOT/jobs" "$ROOT/results" "$ROOT/uploads" "$ROOT/secrets"

# Stop a stale worker from a previous local run, if one is still alive.
if [ -f "$ROOT/logs/worker.pid" ]; then
  OLD_PID="$(cat "$ROOT/logs/worker.pid" 2>/dev/null || true)"
  if [ -n "${OLD_PID:-}" ] && kill -0 "$OLD_PID" >/dev/null 2>&1; then
    echo "Existing worker appears to be running with PID=$OLD_PID."
  else
    rm -f "$ROOT/logs/worker.pid"
  fi
fi

if [ ! -f "$ROOT/logs/worker.pid" ]; then
  echo "Starting background worker..."
  "$AGENCY_PY" "$ROOT/web/worker.py" >> "$ROOT/logs/worker.log" 2>&1 &
  WORKER_PID=$!
  echo "$WORKER_PID" > "$ROOT/logs/worker.pid"
  echo "Worker PID=$WORKER_PID"
fi

echo "Starting Flask at http://$HOST:$PORT"
cd "$ROOT/web"
python -m flask --app app run --host "$HOST" --port "$PORT"
