#!/usr/bin/env python3
"""
agency_script_CLI.py (v1, cleaned)

- Uses PATH_AGENCY_V1 (preferred) or script location to find Agency root.
- Stores per-user last_dir in ~/MyAgency/last_dir.txt (problem folder path).
- Invokes bin/Run_Agency.sh (which handles background vs continuation).
- Validates PROB/ProblemStatement.pdf for New runs.
"""

from __future__ import annotations
import os
import sys
import subprocess
import argparse
from pathlib import Path
from typing import Optional

# -----------------------
# Per-user last_dir (MyAgency)
# -----------------------
def get_user_last_dir_file() -> Path:
    p = Path.home() / "MyAgency"
    p.mkdir(parents=True, exist_ok=True)
    return p / "last_dir.txt"

LAST_DIR_FILE: Path = get_user_last_dir_file()

def read_last_dir() -> str:
    try:
        if LAST_DIR_FILE.exists():
            txt = LAST_DIR_FILE.read_text(encoding="utf-8").strip()
            if txt:
                return txt
    except Exception as e:
        print(f"Warning: could not read {LAST_DIR_FILE}: {e}", file=sys.stderr)
    return str(Path.home())

def write_last_dir(p: Path) -> None:
    try:
        LAST_DIR_FILE.write_text(str(p), encoding="utf-8")
    except Exception as e:
        print(f"Warning: could not write {LAST_DIR_FILE}: {e}", file=sys.stderr)

# -----------------------
# Agency root resolution
# -----------------------
def resolve_agency_root() -> Path:
    # 1) prefer explicit env var (admin/user sets PATH_AGENCY_V1)
    env = os.environ.get("PATH_AGENCY_V1")
    if env:
        root = Path(env).expanduser().resolve()
        if (root / "bin" / "Run_Agency.sh").exists():
            return root
        raise FileNotFoundError(f"PATH_AGENCY_V1 is set to '{root}' but bin/Run_Agency.sh not found there.")
    # 2) fallback: assume this file lives in v1/Agency/
    here = Path(__file__).resolve()
    cand = here.parent  # Agency/
    if (cand / "bin" / "Run_Agency.sh").exists():
        return cand
    cand2 = here.parent.parent
    if (cand2 / "bin" / "Run_Agency.sh").exists():
        return cand2
    raise FileNotFoundError("Could not determine Agency root. Set PATH_AGENCY_V1 to the v1/Agency folder.")

# -----------------------
# Helpers
# -----------------------
def prompt_folder_interactive() -> Path:
    default = read_last_dir()
    print("Select Problems/PROBLEMNAME folder (the one that contains the PROB subfolder).")
    print(f"Press Enter to use last path: {default}")
    while True:
        user = input("Folder path: ").strip()
        if not user:
            user = default
        p = Path(user).expanduser().resolve()
        if p.is_dir():
            # store the actual problem folder (not the parent)
            write_last_dir(p)
            return p
        print(f"Path does not exist or is not a directory: {p}")

def run_run_agency(agency_root: Path, args_list: list, cwd: Optional[Path] = None) -> int:
    run_agency = agency_root / "bin" / "Run_Agency.sh"
    if not run_agency.exists():
        print(f"Error: Run_Agency.sh not found in {agency_root / 'bin'}.", file=sys.stderr)
        return 2
    cmd = [str(run_agency)] + args_list
    print(f"Running: {' '.join(cmd)} (cwd={cwd or agency_root})")
    try:
        subprocess.run(cmd, check=True, cwd=str(cwd or agency_root), env=os.environ)
    except subprocess.CalledProcessError as e:
        print(f"Error: Run_Agency.sh failed: {e}", file=sys.stderr)
        return e.returncode if isinstance(e.returncode, int) else 1
    return 0

# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser(prog="agency_script_CLI.py", description="Agency CLI")
    parser.add_argument("problem", nargs="?", help="path to problem folder (the folder that contains PROB/)")
    parser.add_argument("--mode", choices=("new", "continuation"), help="New (background) or Continuation (interactive)")
    parser.add_argument("--n", type=int, default=None, help="N_requested for New runs")
    parser.add_argument("--datestamp", type=str, default=None, help="datestamp for Continuation")
    args = parser.parse_args()

    # If problem provided on CLI use it, else prompt interactively
    if args.problem:
        problem_folder = Path(args.problem).expanduser().resolve()
        if not problem_folder.is_dir():
            print(f"Error: specified problem path does not exist or is not a directory: {problem_folder}", file=sys.stderr)
            sys.exit(2)
        # update last_dir to this problem folder
        write_last_dir(problem_folder)
    else:
        problem_folder = prompt_folder_interactive()

    # PROB subfolder validation
    prob_sub = problem_folder / "PROB"
    if not prob_sub.is_dir():
        print(f"Error: PROB subfolder not found in: {problem_folder}", file=sys.stderr)
        sys.exit(2)

    # Determine mode: CLI option overrides; else prompt
    #mode = args.mode
    #if not mode:
    #    while True:
    #        m = input("Enter mode (New or Continuation): ").strip().lower()
    #        if m in ("new", "continuation"):
    #            mode = m
    #            break
    #        print("Invalid input. Please enter 'New' or 'Continuation'.")
    mode = "new" #3/26

    # Show API env check for convenience
    for k in ("OPENAI_API_KEY",):
        v = os.environ.get(k, "")
        print(f"ENV {k}: {'SET' if v else 'NOT SET'}")

    # Resolve Agency root
    try:
        agency_root = resolve_agency_root()
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)

    if mode == "new":
        # Determine N_requested
        N = args.n
        if N is None:
            while True:
                try:
                    n_in = input("Enter number of solve agents (N_requested, positive integer): ").strip()
                    N = int(n_in)
                    if N >= 1:
                        break
                except Exception:
                    pass
                print("Invalid input. Please enter a positive integer.")

        # Enforce ProblemStatement.pdf exists
        ps_pdf = prob_sub / "ProblemStatement.pdf"
        if not ps_pdf.is_file():
            print(f"Error: Expected ProblemStatement.pdf in {prob_sub}.", file=sys.stderr)
            sys.exit(2)

        rc = run_run_agency(agency_root, [str(N), str(problem_folder)])
        if rc != 0:
            sys.exit(rc)
        print("Run_Agency launched (New mode). Backgrounding handled by Run_Agency.sh.")
    else:
        # Continuation: need datestamp
        stamp = args.datestamp
        if not stamp:
            datestamps = sorted(
                [f.stem.replace("transcript_", "") for f in prob_sub.glob("transcript_*.txt")],
                reverse=True,
            )
            if not datestamps:
                print(f"Error: No transcripts found in {prob_sub}.", file=sys.stderr)
                sys.exit(2)
            print("Available datestamps:", ", ".join(datestamps))
            print("Enter 'latest' for the most recent transcript.")
            while True:
                d = input("Enter datestamp (or 'latest'): ").strip()
                if d.lower() == "latest":
                    stamp = datestamps[0]
                    break
                if d in datestamps:
                    stamp = d
                    break
                print(f"Invalid datestamp. Choose from {', '.join(datestamps)} or 'latest'.")

        transcript_file = prob_sub / f"transcript_{stamp}.txt"
        if not transcript_file.exists():
            print(f"Error: Transcript for datestamp {stamp} not found in {prob_sub}.", file=sys.stderr)
            sys.exit(2)

        # Continuation: -DATESTAMP <stamp> 0 /abs/path/to/problem_folder
        rc = run_run_agency(agency_root, ["-DATESTAMP", str(stamp), "0", str(problem_folder)])
        if rc != 0:
            sys.exit(rc)
        print("Run_Agency launched (Continuation). Interactive chat should run in foreground.")

if __name__ == "__main__":
    main()
