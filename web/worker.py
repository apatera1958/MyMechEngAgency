from __future__ import annotations
import os
import subprocess
import sys
import time
import re
import zipfile
from pathlib import Path
from datetime import datetime
import db
from continuation_helper import run_continuation

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
SECRETS = ROOT / "secrets" / "openai.env"

def load_openai_env() -> None:
    if not SECRETS.exists():
        return
    for line in SECRETS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        if "=" in line:
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip().strip('"').strip("'")

def _discover_stamp(prob_dir: Path, fallback: str | None = None) -> tuple[str | None, Path | None, Path | None]:
    htmls = sorted(prob_dir.glob("transcript_*.html"), key=lambda p: p.stat().st_mtime, reverse=True)
    if htmls:
        stamp = htmls[0].stem.replace("transcript_", "")
        return stamp, prob_dir / f"transcript_{stamp}.txt", htmls[0]
    txts = sorted(prob_dir.glob("transcript_*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if txts:
        stamp = txts[0].stem.replace("transcript_", "")
        return stamp, txts[0], prob_dir / f"transcript_{stamp}.html"
    return fallback, None, None


def _safe_name(text: str | None, fallback: str = "Problem") -> str:
    text = (text or fallback).strip()
    text = re.sub(r"\.[Pp][Dd][Ff]$", "", text)
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
    return (text or fallback)[:60]


def _write_bundle(prob_dir: Path, stamp: str | None, html_path: Path | None, tx_path: Path | None, label: str | None) -> Path | None:
    """Create a user-facing Bundle_<problem-name>_<stamp>.zip without modifying core code."""
    if not stamp:
        return None
    safe = _safe_name(label)
    bundle_path = prob_dir / f"Bundle_{safe}_{stamp}.zip"
    candidates = [
        (html_path, html_path.name if html_path else None),
        (tx_path, tx_path.name if tx_path else None),
        (prob_dir / "ProblemStatement.pdf", "ProblemStatement.pdf"),
        (prob_dir / "PROBhints.txt", "PROBhints.txt"),
        (prob_dir / "PROBgpt_models.yaml", "PROBgpt_models.yaml"),
    ]
    for pattern in ("continuation_resp_*.json", "continuation_ci_debug_*.json"):
        for q in sorted(prob_dir.glob(pattern)):
            candidates.append((q, q.name))
    try:
        with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for p, arc in candidates:
                if p and arc and Path(p).exists():
                    z.write(str(p), arcname=arc)
        return bundle_path
    except Exception:
        return None

def run_analysis(job) -> None:
    load_openai_env()
    jid = job["id"]
    problem_dir = Path(job["problem_dir"])
    prob_dir = Path(job["prob_dir"])
    log_path = Path(job["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    env = os.environ.copy()
    # If the server has no configured key, the web app may store a per-job user key.
    # This is intended only as a fallback/self-host mode; public demo deployments should use a server key.
    if job["user_openai_key"]:
        env["OPENAI_API_KEY"] = job["user_openai_key"]
    env["PATH_AGENCY_V1"] = str(CORE)
    env["PROB_DIR"] = str(prob_dir)

    cmd = [sys.executable, str(CORE / "client" / "agency_client.py"), str(job["n_agents"] or 2), "--problem", str(problem_dir), "--stamp", stamp, "--nochat"]
    db.update_job(jid, status="running", stamp=stamp)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("Running: " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=str(CORE), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        db.update_job(jid, status="failed", error=f"MyAgency exited with code {proc.returncode}")
        return
    stamp2, tx, html = _discover_stamp(prob_dir, stamp)
    if not html or not html.exists():
        db.update_job(jid, status="failed", error="Run completed but transcript HTML was not found")
        return
    _write_bundle(prob_dir, stamp2 or stamp, html, tx, job["job_name"] or job["original_filename"])
    db.update_job(jid, status="complete", stamp=stamp2, transcript_txt=str(tx) if tx else None, result_html=str(html))

def run_cont(job) -> None:
    load_openai_env()
    jid = job["id"]
    log_path = Path(job["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if job["user_openai_key"]:
        os.environ["OPENAI_API_KEY"] = job["user_openai_key"]
    db.update_job(jid, status="running")
    try:
        with log_path.open("w", encoding="utf-8") as log:
            log.write(f"Continuation for parent={job['parent_id']} stamp={job['stamp']}\n")
            answer, tx, html = run_continuation(Path(job["prob_dir"]), job["stamp"], job["question"])
            log.write("\nAnswer:\n" + answer + "\n")
        db.update_job(jid, status="complete", transcript_txt=str(tx), result_html=str(html))
    except Exception as e:
        db.update_job(jid, status="failed", error=str(e))

def main() -> None:
    db.init_db()
    print("MyAgency worker started. Press Ctrl-C to stop.", flush=True)
    while True:
        job = db.queued_one()
        if not job:
            time.sleep(2)
            continue
        if job["kind"] == "continuation":
            run_cont(job)
        else:
            run_analysis(job)

if __name__ == "__main__":
    main()
