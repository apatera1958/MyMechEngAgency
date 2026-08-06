from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import time
import traceback
import zipfile
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Mapping

import db
from continuation_helper import run_continuation
from explore_helper import run_explore

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
SECRETS = ROOT / "secrets" / "openai.env"

# A per-job user key is a legacy/self-host fallback.  continuation_helper and
# explore_helper read OPENAI_API_KEY from the process environment, so any such
# job must run without other jobs in this process.  The dispatcher enforces
# that rule; this lock is an additional safeguard.
_USER_KEY_LOCK = threading.Lock()


@dataclass(frozen=True)
class ActiveJob:
    job_id: str
    kind: str
    uses_user_key: bool


def _env_int(name: str, default: int, minimum: int = 1, maximum: int = 64) -> int:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError:
        print(f"Invalid {name}={raw!r}; using {default}.", flush=True)
        value = default
    return max(minimum, min(maximum, value))


def _env_float(name: str, default: float, minimum: float) -> float:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError:
        print(f"Invalid {name}={raw!r}; using {default}.", flush=True)
        value = default
    return max(minimum, value)


def load_openai_env() -> None:
    if not SECRETS.exists():
        return
    for line in SECRETS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        if "=" in line:
            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip().strip('"').strip("'")


@contextmanager
def _temporary_user_openai_key(user_key: str | None) -> Iterator[None]:
    """Temporarily install a per-job key, then restore the server key.

    The dispatcher runs a per-key job exclusively, so no ordinary concurrent
    job can observe this temporary environment value.
    """
    if not user_key:
        yield
        return

    with _USER_KEY_LOCK:
        previous = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = user_key
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop("OPENAI_API_KEY", None)
            else:
                os.environ["OPENAI_API_KEY"] = previous


def _discover_stamp(
    prob_dir: Path, fallback: str | None = None
) -> tuple[str | None, Path | None, Path | None]:
    htmls = sorted(
        prob_dir.glob("transcript_*.html"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if htmls:
        stamp = htmls[0].stem.replace("transcript_", "")
        return stamp, prob_dir / f"transcript_{stamp}.txt", htmls[0]

    txts = sorted(
        prob_dir.glob("transcript_*.txt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if txts:
        stamp = txts[0].stem.replace("transcript_", "")
        return stamp, txts[0], prob_dir / f"transcript_{stamp}.html"

    return fallback, None, None


def _safe_name(text: str | None, fallback: str = "Problem") -> str:
    text = (text or fallback).strip()
    text = re.sub(r"\.[Pp][Dd][Ff]$", "", text)
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
    return (text or fallback)[:60]


def _write_bundle(
    prob_dir: Path,
    stamp: str | None,
    html_path: Path | None,
    tx_path: Path | None,
    label: str | None,
) -> Path | None:
    """Create initial user-facing Bundle_<problem-name>-0_<stamp>.zip."""
    if not stamp:
        return None

    safe = _safe_name(label)
    for old in prob_dir.glob(f"Bundle_*_{stamp}.zip"):
        try:
            old.unlink()
        except Exception:
            pass

    bundle_path = prob_dir / f"Bundle_{safe}-0_{stamp}.zip"
    candidates = [
        (html_path, html_path.name if html_path else None),
        (tx_path, tx_path.name if tx_path else None),
        (prob_dir / "ProblemStatement.pdf", "ProblemStatement.pdf"),
        (prob_dir / "PROBhints.txt", "PROBhints.txt"),
        (prob_dir / "PROBgpt_models.yaml", "PROBgpt_models.yaml"),
    ]
    for pattern in ("continuation_resp_*.json", "continuation_ci_debug_*.json"):
        for path in sorted(prob_dir.glob(pattern)):
            candidates.append((path, path.name))

    try:
        with zipfile.ZipFile(
            bundle_path, "w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            for path, arcname in candidates:
                if path and arcname and Path(path).exists():
                    archive.write(str(path), arcname=arcname)
        return bundle_path
    except Exception:
        return None


def run_analysis(job: Mapping[str, Any]) -> None:
    load_openai_env()
    job_id = str(job["id"])
    problem_dir = Path(job["problem_dir"])
    prob_dir = Path(job["prob_dir"])
    log_path = Path(job["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    env = os.environ.copy()
    # Per-job keys are passed only to this child process, so concurrent analysis
    # jobs cannot overwrite one another's keys.
    if job.get("user_openai_key"):
        env["OPENAI_API_KEY"] = str(job["user_openai_key"])
    env["PATH_AGENCY_V1"] = str(CORE)
    env["PROB_DIR"] = str(prob_dir)

    cmd = [
        sys.executable,
        str(CORE / "client" / "agency_client.py"),
        str(job.get("n_agents") or 2),
        "--problem",
        str(problem_dir),
        "--stamp",
        stamp,
        "--nochat",
    ]

    # claim_queued_job() has already changed status to running.
    db.update_job(job_id, stamp=stamp, error=None)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("Running: " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(CORE),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )

    if proc.returncode != 0:
        db.update_job(
            job_id,
            status="failed",
            error=f"MyAgency exited with code {proc.returncode}",
        )
        return

    stamp2, tx_path, html_path = _discover_stamp(prob_dir, stamp)
    if not html_path or not html_path.exists():
        db.update_job(
            job_id,
            status="failed",
            error="Run completed but transcript HTML was not found",
        )
        return

    _write_bundle(
        prob_dir,
        stamp2 or stamp,
        html_path,
        tx_path,
        job.get("job_name") or job.get("original_filename"),
    )
    db.update_job(
        job_id,
        status="complete",
        stamp=stamp2,
        transcript_txt=str(tx_path) if tx_path else None,
        result_html=str(html_path),
        error=None,
    )


def run_cont(job: Mapping[str, Any]) -> None:
    load_openai_env()
    job_id = str(job["id"])
    log_path = Path(job["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with _temporary_user_openai_key(job.get("user_openai_key")):
            with log_path.open("w", encoding="utf-8") as log:
                log.write(f"Continuation for job={job_id} stamp={job.get('stamp')}\n")
                log.flush()
                answer, tx_path, html_path = run_continuation(
                    Path(job["prob_dir"]), job.get("stamp"), job.get("question")
                )
                log.write("\nAnswer:\n" + answer + "\n")

        # Return the visible row to analysis/complete after incorporation.
        db.update_job(
            job_id,
            kind="analysis",
            status="complete",
            transcript_txt=str(tx_path),
            result_html=str(html_path),
            question=None,
            error=None,
        )
    except Exception as exc:
        db.update_job(job_id, status="failed", error=str(exc))


def run_explore_job(job: Mapping[str, Any]) -> None:
    load_openai_env()
    job_id = str(job["id"])
    log_path = Path(job["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with _temporary_user_openai_key(job.get("user_openai_key")):
            with log_path.open("w", encoding="utf-8") as log:
                log.write(f"Explore for job={job_id} stamp={job.get('stamp')}\n")
                log.flush()
                notebook_path = run_explore(
                    Path(job["prob_dir"]), job.get("stamp")
                )
                log.write(f"Notebook written: {notebook_path}\n")

        # Return the visible row to its normal completed-analysis state.
        db.update_job(
            job_id,
            kind="analysis",
            status="complete",
            error=None,
        )
    except Exception as exc:
        db.update_job(job_id, status="failed", error=str(exc))


def _run_claimed_job(job: Mapping[str, Any]) -> None:
    """Run one already-claimed job and contain unexpected exceptions."""
    job_id = str(job["id"])
    kind = str(job.get("kind") or "analysis")
    print(f"Starting job {job_id} kind={kind}", flush=True)

    try:
        if kind == "continuation":
            run_cont(job)
        elif kind == "explore":
            run_explore_job(job)
        else:
            run_analysis(job)
    except Exception as exc:
        message = f"Unexpected worker error: {type(exc).__name__}: {exc}"
        db.fail_job_if_running(job_id, message)
        print(message, file=sys.stderr, flush=True)
        traceback.print_exc()
    finally:
        row = db.one(job_id)
        final_status = row["status"] if row else "missing"
        print(
            f"Finished job {job_id} kind={kind} status={final_status}",
            flush=True,
        )


def _collect_finished(active: dict[Future[None], ActiveJob]) -> None:
    finished = [future for future in active if future.done()]
    for future in finished:
        state = active.pop(future)
        try:
            future.result()
        except Exception as exc:
            # _run_claimed_job normally contains all exceptions; this is a final
            # guard for executor-level failures.
            message = f"Worker future failed: {type(exc).__name__}: {exc}"
            db.fail_job_if_running(state.job_id, message)
            print(message, file=sys.stderr, flush=True)


def main() -> None:
    db.init_db()
    load_openai_env()

    concurrency = _env_int("BACKGROUND_JOB_CONCURRENCY", 1, minimum=1, maximum=32)
    poll_seconds = _env_float("BACKGROUND_JOB_POLL_SECONDS", 1.0, minimum=0.1)
    heartbeat_seconds = _env_float(
        "BACKGROUND_JOB_HEARTBEAT_SECONDS", 30.0, minimum=5.0
    )
    reclaim_seconds = _env_float(
        "BACKGROUND_JOB_RECLAIM_SECONDS", 60.0, minimum=10.0
    )
    stale_minutes = _env_int(
        "BACKGROUND_JOB_STALE_MINUTES", 20, minimum=1, maximum=24 * 60
    )

    reclaimed = db.requeue_stale_running_jobs(stale_minutes)
    if reclaimed:
        print(f"Requeued {reclaimed} stale running job(s).", flush=True)

    print(
        "MyAgency concurrent worker started: "
        f"concurrency={concurrency}, stale_after={stale_minutes} min. "
        "Press Ctrl-C to stop.",
        flush=True,
    )

    active: dict[Future[None], ActiveJob] = {}
    last_heartbeat = time.monotonic()
    last_reclaim = time.monotonic()

    with ThreadPoolExecutor(
        max_workers=concurrency,
        thread_name_prefix="myagency-job",
    ) as executor:
        try:
            while True:
                _collect_finished(active)
                current = time.monotonic()

                if active and current - last_heartbeat >= heartbeat_seconds:
                    try:
                        db.touch_jobs(state.job_id for state in active.values())
                    except Exception as exc:
                        print(f"Heartbeat database error: {exc}", file=sys.stderr, flush=True)
                    last_heartbeat = current

                if current - last_reclaim >= reclaim_seconds:
                    try:
                        reclaimed = db.requeue_stale_running_jobs(stale_minutes)
                        if reclaimed:
                            print(
                                f"Requeued {reclaimed} stale running job(s).",
                                flush=True,
                            )
                    except Exception as exc:
                        print(f"Stale-job scan error: {exc}", file=sys.stderr, flush=True)
                    last_reclaim = current

                dispatched_any = False
                while len(active) < concurrency:
                    # A per-job key changes process-wide OPENAI_API_KEY for
                    # continuation/explore. Preserve FIFO and let current jobs
                    # drain before claiming such a job.
                    if active and db.oldest_queued_uses_user_key():
                        break
                    if any(state.uses_user_key for state in active.values()):
                        break

                    try:
                        row = db.claim_queued_job()
                    except Exception as exc:
                        print(f"Queue claim error: {exc}", file=sys.stderr, flush=True)
                        break

                    if row is None:
                        break

                    job = dict(row)
                    job_id = str(job["id"])
                    uses_user_key = bool(job.get("user_openai_key"))

                    # Defensive race guard: if a user-key job appeared between
                    # the FIFO check and atomic claim, return it to the queue.
                    if uses_user_key and active:
                        db.requeue_job_if_running(job_id)
                        break

                    try:
                        future = executor.submit(_run_claimed_job, job)
                    except Exception as exc:
                        db.requeue_job_if_running(
                            job_id, f"Could not submit job to executor: {exc}"
                        )
                        raise

                    active[future] = ActiveJob(
                        job_id=job_id,
                        kind=str(job.get("kind") or "analysis"),
                        uses_user_key=uses_user_key,
                    )
                    dispatched_any = True
                    print(
                        f"Claimed job {job_id}; active={len(active)}/{concurrency}",
                        flush=True,
                    )

                    if uses_user_key:
                        break

                if not dispatched_any:
                    time.sleep(poll_seconds if not active else min(poll_seconds, 0.5))

        except KeyboardInterrupt:
            print(
                "Worker stopping: no new jobs will be claimed; waiting for active jobs.",
                flush=True,
            )


if __name__ == "__main__":
    main()
