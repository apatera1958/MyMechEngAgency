#!/usr/bin/env bash
set -euo pipefail

# create_python_environment.sh
# Installs Miniforge if needed and creates/updates the MyAgency conda environment.
# Works on macOS Apple Silicon, macOS Intel, and Linux x86_64/aarch64.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$ROOT/environment.yml"
ENV_NAME="myagency-server"
MINIFORGE_DIR="${MYAGENCY_MINIFORGE_DIR:-$ROOT/miniforge3}"

if [ ! -f "$ENV_FILE" ]; then
  echo "ERROR: environment.yml not found at: $ENV_FILE" >&2
  exit 1
fi

install_miniforge() {
  local os arch installer url
  os="$(uname -s)"
  arch="$(uname -m)"

  case "$os:$arch" in
    Darwin:arm64)  installer="Miniforge3-MacOSX-arm64.sh" ;;
    Darwin:x86_64) installer="Miniforge3-MacOSX-x86_64.sh" ;;
    Linux:x86_64)  installer="Miniforge3-Linux-x86_64.sh" ;;
    Linux:aarch64) installer="Miniforge3-Linux-aarch64.sh" ;;
    *)
      echo "ERROR: unsupported OS/architecture: $os $arch" >&2
      exit 1
      ;;
  esac

  url="https://github.com/conda-forge/miniforge/releases/latest/download/$installer"
  echo "Installing Miniforge into: $MINIFORGE_DIR"
  echo "Downloading: $url"

  if command -v curl >/dev/null 2>&1; then
    curl -L -o "/tmp/$installer" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "/tmp/$installer" "$url"
  else
    echo "ERROR: need curl or wget to download Miniforge." >&2
    exit 1
  fi

  bash "/tmp/$installer" -b -p "$MINIFORGE_DIR"
}

if [ ! -f "$MINIFORGE_DIR/etc/profile.d/conda.sh" ]; then
  install_miniforge
else
  echo "Found existing Miniforge at: $MINIFORGE_DIR"
fi

# shellcheck disable=SC1091
source "$MINIFORGE_DIR/etc/profile.d/conda.sh"
conda config --set auto_activate_base false || true

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Updating existing environment: $ENV_NAME"
  conda env update -n "$ENV_NAME" -f "$ENV_FILE" --prune
else
  echo "Creating environment: $ENV_NAME"
  conda env create -f "$ENV_FILE"
fi

mkdir -p "$ROOT/uploads" "$ROOT/results" "$ROOT/jobs" "$ROOT/logs" "$ROOT/secrets"

if [ ! -f "$ROOT/secrets/openai.env" ]; then
  cat > "$ROOT/secrets/openai.env" <<'KEYEOF'
# Fill this in by hand on the private server.
# Do not commit or share this file if it contains a real key.
export OPENAI_API_KEY=" "
KEYEOF
  chmod 600 "$ROOT/secrets/openai.env"
fi

cat <<DONE

Done.

To activate manually later:
  source "$MINIFORGE_DIR/etc/profile.d/conda.sh"
  conda activate "$ENV_NAME"

Next test step:
  bash start_server.sh

DONE
