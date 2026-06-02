#!/usr/bin/env bash
set -euo pipefail

export MYAGENCY_ROOT="${MYAGENCY_ROOT:-$(pwd)}"
export PATH_AGENCY_V1="$MYAGENCY_ROOT/core"
export PYTHONPATH="$MYAGENCY_ROOT/web:$MYAGENCY_ROOT:${PYTHONPATH:-}"

mkdir -p "$MYAGENCY_ROOT/logs" "$MYAGENCY_ROOT/jobs" "$MYAGENCY_ROOT/results" "$MYAGENCY_ROOT/uploads" "$MYAGENCY_ROOT/secrets"

echo "Starting MyMechEngAgency worker..."
python web/worker.py >> "$MYAGENCY_ROOT/logs/worker.log" 2>&1 &

echo "Starting Gunicorn web service..."
exec gunicorn web.app:app --bind 0.0.0.0:${PORT:-10000}