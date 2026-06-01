#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$ROOT/create_python_environment.sh"
echo "Local install complete. Next: bash start_server.sh"
