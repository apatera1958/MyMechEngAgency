#!/usr/bin/env bash
set -euo pipefail

# Run_Agency.sh - launcher for Agency client
#
# Usage (New background):
#   ./Run_Agency.sh <N_requested> /abs/path/to/problem_folder
#   (problem_folder must contain PROB/ProblemStatement.pdf)
#
# Usage (Continuation interactive):
#   ./Run_Agency.sh -DATESTAMP <datestamp> 0 /abs/path/to/problem_folder
#
# Notes:
# - Exports PROB_DIR (absolute path to problemfolder/PROB).
# - New runs:
#     * Mac / linux: background via nohup, logs in PROB.
#     * Engaging: background via Slurm using templates/slurm.sbatch.template.
# - Continuation runs in foreground to support interactive chat.
#
# Environment knobs (can override before calling):
: "${AGENCY_CONDA_ENV:=myagency-server}"
: "${AGENCY_PY:=}"                 # if empty, will be set to current sys.executable
: "${AGENCY_MINIFORGE_DIR:=}"          # optional self-contained Miniforge path
: "${AGENCY_TIMEOUT_INCR:=1500}"   # kept for potential future use
: "${AGENCY_INITIAL_TIMEOUT:=1500}"

LOG_PREFIX="background"

die()  { echo "ERROR: $*" >&2; exit 2; }
info() { echo "[info] $*"; }

# -------------------------
# Agency root resolution
# -------------------------
resolve_agency_root() {
    if [ -n "${PATH_AGENCY_V1:-}" ]; then
        ROOT="$PATH_AGENCY_V1"
        if [ ! -f "$ROOT/bin/Run_Agency.sh" ]; then
            die "PATH_AGENCY_V1 is set to '$ROOT' but bin/Run_Agency.sh not found there."
        fi
        printf "%s\n" "$ROOT"
        return
    fi
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
    CAND="$(cd "$SCRIPT_DIR/.." && pwd)"
    if [ -f "$CAND/bin/Run_Agency.sh" ]; then
        printf "%s\n" "$CAND"
        return
    fi
    die "Could not determine Agency root. Set PATH_AGENCY_V1 to the v1/Agency folder."
}

get_platform() {
    local root="$1"
    local cfg="$root/config.yaml"
    if [ ! -f "$cfg" ]; then
        echo "Mac"
        return
    fi
    python3 - <<PY
import yaml, pathlib
cfg_path = pathlib.Path("$cfg")
try:
    cfg = yaml.safe_load(cfg_path.read_text())
    print((cfg or {}).get("platform", "Mac"))
except Exception:
    print("Mac")
PY
}

ensure_problemstatement_pdf() {
    local pdir="$1"
    local ps="$pdir/ProblemStatement.pdf"
    if [ ! -f "$ps" ]; then
        die "Expected ProblemStatement.pdf in $pdir but did not find it."
    fi
}

# -------------------------
# Arg parsing
# -------------------------
if [ "$#" -lt 2 ]; then
    die "Insufficient arguments. Usage: ./Run_Agency.sh <N> /path/to/problem  OR ./Run_Agency.sh -DATESTAMP <datestamp> 0 /path/to/problem"
fi

FIRST="$1"
if [ "$FIRST" = "-DATESTAMP" ]; then
    if [ "$#" -lt 4 ]; then
        die "Continuation usage: $0 -DATESTAMP <datestamp> 0 /abs/path/to/problem_folder"
    fi
    MODE="continuation"
    DATESTAMP="$2"
    shift 3
    PROBLEM_ARG="$1"
else
    MODE="new"
    N_REQUESTED="$1"
    PROBLEM_ARG="$2"
fi

# -------------------------
# Canonicalize PROBLEM_ARG -> PROBLEM_FOLDER (absolute)
# -------------------------
PROBLEM_FOLDER="$(python3 -c 'import os,sys; p=sys.argv[1] if len(sys.argv)>1 else ""; print(os.path.abspath(os.path.expanduser(p)))' "$PROBLEM_ARG")"

if [ ! -d "$PROBLEM_FOLDER" ]; then
    die "Problem folder does not exist: $PROBLEM_FOLDER"
fi

PROB_DIR="$PROBLEM_FOLDER/PROB"
if [ ! -d "$PROB_DIR" ]; then
    die "PROB subfolder not found under problem folder: $PROBLEM_FOLDER (expected: $PROB_DIR)"
fi

# absolute paths
PROBLEM_FOLDER="$(cd "$PROBLEM_FOLDER" && pwd)"
PROB_DIR="$(cd "$PROB_DIR" && pwd)"

AGENCY_ROOT="$(resolve_agency_root)"
if [ -z "$AGENCY_MINIFORGE_DIR" ]; then
    # In MyAgency_Server layout, AGENCY_ROOT is ROOT/core, so Miniforge is one level up.
    AGENCY_MINIFORGE_DIR="$(cd "$AGENCY_ROOT/.." && pwd)/miniforge3"
fi
CLIENT_PY="$AGENCY_ROOT/client/agency_client.py"

if [ ! -f "$CLIENT_PY" ]; then
    die "Client not found at: $CLIENT_PY"
fi

# Determine Python executable from current environment if not set
if [ -z "$AGENCY_PY" ]; then
    AGENCY_PY="$(python3 -c 'import sys; print(sys.executable)')"
fi

# Export PROB_DIR so the client has a fallback
export PROB_DIR

# Export timeouts so client can optionally read them
export AGENCY_INITIAL_TIMEOUT
export AGENCY_TIMEOUT_INCR

if [ "$MODE" = "new" ]; then
    STAMP="$(date +%Y%m%d-%H%M%S)"
else
    STAMP="$DATESTAMP"
fi

LOGPATH="$PROB_DIR/${LOG_PREFIX}_${STAMP}.log"

PLATFORM="$(get_platform "$AGENCY_ROOT")"
info "Detected platform from config.yaml: $PLATFORM"

# -------------------------
# New mode: background run
# -------------------------
if [ "$MODE" = "new" ]; then
    info "Preparing NEW run (background): N_requested=$N_REQUESTED"
    info "Problem folder: $PROBLEM_FOLDER"
    info "PROB folder: $PROB_DIR"
    info "Log will be: $LOGPATH"

    ensure_problemstatement_pdf "$PROB_DIR"

    cd "$AGENCY_ROOT"

    # --- Engaging: submit via Slurm ---
    if [ "$PLATFORM" = "Engaging" ]; then
        TEMPLATE="$AGENCY_ROOT/templates/slurm.sbatch.template"
        if [ ! -f "$TEMPLATE" ]; then
            die "Slurm template not found at: $TEMPLATE"
        fi

        JOB_SCRIPT="$(mktemp -t agency_slurm_XXXXXX).sbatch"
        sed \
          -e "s|%LOGPATH%|$LOGPATH|g" \
          -e "s|%PYTHON%|$AGENCY_PY|g" \
          -e "s|%CLIENT_PY%|$CLIENT_PY|g" \
          -e "s|%N%|$N_REQUESTED|g" \
          -e "s|%STAMP%|$STAMP|g" \
          -e "s|%PROBLEM_FOLDER%|$PROBLEM_FOLDER|g" \
          "$TEMPLATE" > "$JOB_SCRIPT"

        info "Submitting Slurm job with script: $JOB_SCRIPT"
        SUBMIT_OUT="$(sbatch "$JOB_SCRIPT")" || die "sbatch failed"
        JOB_ID="$(echo "$SUBMIT_OUT" | awk '{print $NF}')"
        info "Slurm job submitted: JOB_ID=$JOB_ID"
        info "Logs will be in: $LOGPATH"
        info "Transcript and summaries will be written into: $PROB_DIR (by client)."
        exit 0
    fi

    # --- Mac / linux: nohup background ---
    RUN_SHIM="$(mktemp -t agency_runner_XXXXXX).sh"
    cat > "$RUN_SHIM" <<SH
#!/usr/bin/env bash
set -euo pipefail
if [ -n "$AGENCY_MINIFORGE_DIR" ] && [ -f "$AGENCY_MINIFORGE_DIR/etc/profile.d/conda.sh" ]; then
    source "$AGENCY_MINIFORGE_DIR/etc/profile.d/conda.sh" || true
    conda activate "$AGENCY_CONDA_ENV" >/dev/null 2>&1 || true
fi
exec "$AGENCY_PY" "$CLIENT_PY" "$N_REQUESTED" --problem "$PROBLEM_FOLDER" --stamp "$STAMP" --nochat
SH
    chmod +x "$RUN_SHIM"
    nohup "$RUN_SHIM" > "$LOGPATH" 2>&1 &
    PID=$!
    info "Background job started (nohup). PID=$PID"
    info "To monitor progress: tail -f $LOGPATH"
    ( sleep 5; rm -f "$RUN_SHIM" ) >/dev/null 2>&1 &

    info "Transcript and summaries will be written into: $PROB_DIR (by client)."
    exit 0
fi

# -------------------------
# Continuation (interactive) - run in foreground
# -------------------------
info "Launching CONTINUATION (interactive). datestamp=$STAMP"
cd "$AGENCY_ROOT"

if [ -n "$AGENCY_MINIFORGE_DIR" ] && [ -f "$AGENCY_MINIFORGE_DIR/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$AGENCY_MINIFORGE_DIR/etc/profile.d/conda.sh" || true
    conda activate "$AGENCY_CONDA_ENV" >/dev/null 2>&1 || true
fi

"$AGENCY_PY" "$CLIENT_PY" 0 --problem "$PROBLEM_FOLDER" --chat-only --resume-stamp "$STAMP"
exit $?
