#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DAYS="${1:-7}"
find "$ROOT/results" -mindepth 1 -maxdepth 1 -type d -mtime +"$DAYS" -print -exec rm -rf {} +
find "$ROOT/uploads" -mindepth 1 -maxdepth 1 -mtime +"$DAYS" -print -exec rm -rf {} +
find "$ROOT/logs" -type f -name '*.log' -mtime +"$DAYS" -print -delete
