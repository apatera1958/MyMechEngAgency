#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y git curl wget bzip2 ca-certificates build-essential tmux
fi

bash "$ROOT/create_python_environment.sh"

echo "AWS install complete. Edit $ROOT/secrets/openai.env, then run: bash start_server.sh"
