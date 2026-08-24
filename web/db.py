from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]

# Keep the historical local path by default.  On Render, setting
# MYAGENCY_STORAGE_ROOT redirects the SQLite database beneath the persistent
# disk mount.
_STORAGE_ROOT_RAW = os.environ.get("MYAGENCY_STORAGE_ROOT", "").strip()
if _STORAGE_ROOT_RAW:
    DB_PATH = Path(_STORAGE_ROOT_RAW).expanduser() / "jobs" / "jobs.sqlite"
else:
    DB_PATH = ROOT / "jobs" / "jobs.sqlite"

SQLITE_TIMEOUT_SECONDS = 30.0
SQLITE_BUSY_TIMEOUT_MS = 30_000

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    handle TEXT UNIQUE NOT NULL,
    access_code_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_users_handle ON users(handle);

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    user_id INTEGER,
    kind TEXT NOT NULL DEFAULT 'analysis',
    parent_id TEXT,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    queued_at TEXT,
    updated_at TEXT NOT NULL,
    job_name TEXT,
    owner_email TEXT,
    access_code_hash TEXT,
    session_id TEXT,
    user_openai_key TEXT,
    requester_ip TEXT,
    original_filename TEXT,
    pdf_page_count INTEGER,
    hints_filename TEXT,
    models_filename TEXT,
    input_pdf TEXT,
    problem_dir TEXT,
    prob_dir TEXT,
    stamp TEXT,
    n_agents INTEGER,
    question TEXT,
    result_html TEXT,
    transcript_txt TEXT,
    log_path TEXT,
    error TEXT,
    FOREIGN KEY(user_id) REFERENCES users(id)
);
CREATE INDEX IF NOT EXISTS idx_jobs_status_created ON jobs(status, created_at);
CREATE INDEX IF NOT EXISTS idx_jobs_user_created ON jobs(user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_jobs_owner_created ON jobs(owner_email, created_at);
CREATE INDEX IF NOT EXISTS idx_jobs_ip_created ON jobs(requester_ip, created_at);
"""

MIGRATION_COLUMNS = {
    "jobs": {
        "user_id": "INTEGER",
        "job_name": "TEXT",
        "owner_email": "TEXT",
        "access_code_hash": "TEXT",
        "session_id": "TEXT",
        "user_openai_key": "TEXT",
        "requester_ip": "TEXT",
        "original_filename": "TEXT",
        "pdf_page_count": "INTEGER",
        "hints_filename": "TEXT",
        "models_filename": "TEXT",
        "queued_at": "TEXT",
    }
}


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def queue_now() -> str:
    # Microseconds preserve FIFO ordering when multiple queue events occur
    # within the same second.
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")


def cutoff(hours: int) -> str:
    return (datetime.now() - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")


def connect() -> sqlite3.Connection:
    """Open one short-lived SQLite connection.

    Every web request and worker thread gets its own connection.  WAL mode is
    initialized in init_db(); busy_timeout lets concurrent writers wait briefly
    rather than immediately failing with "database is locked".
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=SQLITE_TIMEOUT_SECONDS)
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _ensure_columns(con: sqlite3.Connection, table: str, cols: dict[str, str]) -> None:
    existing = {r[1] for r in con.execute(f"PRAGMA table_info({table})").fetchall()}
    for col, typ in cols.items():
        if col not in existing:
            con.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")


def init_db() -> None:
    with connect() as con:
        # WAL permits readers while a worker thread is writing and is the
        # appropriate SQLite mode for this single-instance Render deployment.
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
        con.executescript(SCHEMA)
        for table, cols in MIGRATION_COLUMNS.items():
            _ensure_columns(con, table, cols)

        # Existing jobs predate queued_at. Preserve their historical order.
        con.execute(
            "UPDATE jobs SET queued_at=created_at "
            "WHERE queued_at IS NULL OR queued_at=''"
        )
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_jobs_status_queued "
            "ON jobs(status, queued_at, id)"
        )


def one(job_id: str):
    with connect() as con:
        return con.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()


def user_by_id(user_id: int):
    with connect() as con:
        return con.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()


def user_by_handle(handle: str):
    with connect() as con:
        return con.execute("SELECT * FROM users WHERE handle=?", (handle,)).fetchone()


def create_user(handle: str, access_code_hash: str):
    ts = now()
    with connect() as con:
        con.execute(
            "INSERT INTO users (handle, access_code_hash, created_at, updated_at) VALUES (?,?,?,?)",
            (handle, access_code_hash, ts, ts),
        )
        return con.execute("SELECT * FROM users WHERE handle=?", (handle,)).fetchone()


def recent(limit: int = 20):
    with connect() as con:
        return con.execute(
            "SELECT jobs.*, users.handle FROM jobs LEFT JOIN users ON jobs.user_id=users.id "
            "ORDER BY jobs.created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()


def user_jobs(user_id: int, limit: int = 50):
    with connect() as con:
        return con.execute(
            """
            SELECT * FROM jobs
            WHERE user_id=?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()


def session_jobs(session_id: str, limit: int = 50):
    with connect() as con:
        return con.execute(
            """
            SELECT * FROM jobs
            WHERE session_id=?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()


def count_jobs_for_session_since(session_id: str, since: str) -> int:
    with connect() as con:
        row = con.execute(
            "SELECT COUNT(*) AS n FROM jobs WHERE session_id=? AND created_at>=?",
            (session_id, since),
        ).fetchone()
        return int(row["n"] or 0)


def queued_one():
    """Return the oldest queued job without claiming it.

    Retained for compatibility with any older code.  Concurrent workers must
    use claim_queued_job() instead.
    """
    with connect() as con:
        return con.execute(
            "SELECT * FROM jobs WHERE status='queued' ORDER BY COALESCE(queued_at, created_at) ASC, id ASC LIMIT 1"
        ).fetchone()


def claim_queued_job():
    """Atomically claim and return the oldest queued job.

    BEGIN IMMEDIATE obtains SQLite's write reservation before the SELECT.  The
    selected row is changed from queued to running in the same transaction, so
    a second thread/process cannot receive the same job.
    """
    con = connect()
    try:
        con.execute("BEGIN IMMEDIATE")
        row = con.execute(
            "SELECT id FROM jobs WHERE status='queued' "
            "ORDER BY COALESCE(queued_at, created_at) ASC, id ASC LIMIT 1"
        ).fetchone()
        if row is None:
            con.commit()
            return None

        job_id = row["id"]
        ts = now()
        changed = con.execute(
            """
            UPDATE jobs
            SET status='running', updated_at=?, error=NULL
            WHERE id=? AND status='queued'
            """,
            (ts, job_id),
        ).rowcount

        if changed != 1:
            # Defensive only: BEGIN IMMEDIATE should make this impossible.
            con.rollback()
            return None

        claimed = con.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        con.commit()
        return claimed
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def oldest_queued_uses_user_key() -> bool:
    """Whether the next FIFO job carries a per-job OpenAI key.

    Such jobs are run alone by worker.py because continuation/explore helpers
    obtain their key from process-wide environment variables.
    """
    with connect() as con:
        row = con.execute(
            "SELECT user_openai_key FROM jobs WHERE status='queued' "
            "ORDER BY COALESCE(queued_at, created_at) ASC, id ASC LIMIT 1"
        ).fetchone()
        return bool(row and row["user_openai_key"])


def touch_jobs(job_ids: Iterable[str]) -> int:
    """Refresh updated_at for currently running jobs (worker heartbeat)."""
    ids = list(dict.fromkeys(job_ids))
    if not ids:
        return 0
    qs = ",".join(["?"] * len(ids))
    ts = now()
    with connect() as con:
        cur = con.execute(
            f"UPDATE jobs SET updated_at=? WHERE status='running' AND id IN ({qs})",
            [ts, *ids],
        )
        return int(cur.rowcount)


def requeue_stale_running_jobs(stale_minutes: int) -> int:
    """Return abandoned running jobs to the queue.

    Active jobs are heartbeated by worker.py.  A running row whose updated_at
    is older than the configured threshold is therefore presumed to belong to
    a worker that was killed or redeployed.
    """
    if stale_minutes <= 0:
        return 0
    stale_before = (datetime.now() - timedelta(minutes=stale_minutes)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    ts = now()
    with connect() as con:
        cur = con.execute(
            """
            UPDATE jobs
            SET status='queued', queued_at=?, updated_at=?, error=NULL
            WHERE status='running' AND updated_at < ?
            """,
            (queue_now(), ts, stale_before),
        )
        return int(cur.rowcount)


def requeue_job_if_running(job_id: str, error: str | None = None) -> bool:
    """Put one claimed job back if it has not already finished."""
    with connect() as con:
        cur = con.execute(
            """
            UPDATE jobs
            SET status='queued', queued_at=?, updated_at=?, error=?
            WHERE id=? AND status='running'
            """,
            (queue_now(), now(), error, job_id),
        )
        return cur.rowcount == 1


def fail_job_if_running(job_id: str, error: str) -> bool:
    """Mark a job failed without overwriting an already-completed result."""
    with connect() as con:
        cur = con.execute(
            """
            UPDATE jobs
            SET status='failed', updated_at=?, error=?
            WHERE id=? AND status='running'
            """,
            (now(), error, job_id),
        )
        return cur.rowcount == 1


def count_jobs_for_user_since(user_id: int, since: str) -> int:
    """Count original job rows created for this user in the time window."""
    with connect() as con:
        row = con.execute(
            "SELECT COUNT(*) AS n FROM jobs WHERE user_id=? AND created_at>=?",
            (user_id, since),
        ).fetchone()
        return int(row["n"] or 0)


def count_jobs_for_ip_since(ip: str, since: str) -> int:
    """Count original job rows created from this network address in the time window."""
    with connect() as con:
        row = con.execute(
            "SELECT COUNT(*) AS n FROM jobs WHERE requester_ip=? AND created_at>=?",
            (ip, since),
        ).fetchone()
        return int(row["n"] or 0)


def count_active_jobs() -> int:
    with connect() as con:
        row = con.execute(
            "SELECT COUNT(*) AS n FROM jobs WHERE status IN ('queued','running')"
        ).fetchone()
        return int(row["n"])


def count_active_jobs_for_session(session_id: str) -> int:
    with connect() as con:
        row = con.execute(
            "SELECT COUNT(*) AS n FROM jobs "
            "WHERE session_id=? AND status IN ('queued','running')",
            (session_id,),
        ).fetchone()
        return int(row["n"] or 0)


def queue_snapshot(job_id: str):
    """Return FIFO position and current queue/running counts for one queued job."""
    with connect() as con:
        target = con.execute(
            "SELECT COALESCE(queued_at, created_at) AS queue_time "
            "FROM jobs WHERE id=? AND status='queued'",
            (job_id,),
        ).fetchone()
        if target is None:
            return None

        queue_time = target["queue_time"]
        ahead = con.execute(
            """
            SELECT COUNT(*) AS n
            FROM jobs
            WHERE status='queued'
              AND (
                    COALESCE(queued_at, created_at) < ?
                    OR (
                        COALESCE(queued_at, created_at) = ?
                        AND id < ?
                    )
                  )
            """,
            (queue_time, queue_time, job_id),
        ).fetchone()
        queued = con.execute(
            "SELECT COUNT(*) AS n FROM jobs WHERE status='queued'"
        ).fetchone()
        running = con.execute(
            "SELECT COUNT(*) AS n FROM jobs WHERE status='running'"
        ).fetchone()

        return {
            "position": int(ahead["n"] or 0) + 1,
            "queued_jobs": int(queued["n"] or 0),
            "running_jobs": int(running["n"] or 0),
        }


def old_jobs(hours: int = 48):
    with connect() as con:
        return con.execute("SELECT * FROM jobs WHERE created_at < ?", (cutoff(hours),)).fetchall()


def delete_jobs(job_ids: Iterable[str]) -> None:
    ids = list(job_ids)
    if not ids:
        return
    qs = ",".join(["?"] * len(ids))
    with connect() as con:
        con.execute(f"DELETE FROM jobs WHERE id IN ({qs})", ids)


def insert_job(**kw) -> None:
    ts = now()
    fields = {
        "id": kw["id"],
        "user_id": kw.get("user_id"),
        "kind": kw.get("kind", "analysis"),
        "parent_id": kw.get("parent_id"),
        "status": kw.get("status", "queued"),
        "created_at": ts,
        "queued_at": queue_now(),
        "updated_at": ts,
        "job_name": kw.get("job_name"),
        "owner_email": kw.get("owner_email"),
        "access_code_hash": kw.get("access_code_hash"),
        "session_id": kw.get("session_id"),
        "user_openai_key": kw.get("user_openai_key"),
        "requester_ip": kw.get("requester_ip"),
        "original_filename": kw.get("original_filename"),
        "pdf_page_count": kw.get("pdf_page_count"),
        "hints_filename": kw.get("hints_filename"),
        "models_filename": kw.get("models_filename"),
        "input_pdf": kw.get("input_pdf"),
        "problem_dir": kw.get("problem_dir"),
        "prob_dir": kw.get("prob_dir"),
        "stamp": kw.get("stamp"),
        "n_agents": kw.get("n_agents"),
        "question": kw.get("question"),
        "result_html": kw.get("result_html"),
        "transcript_txt": kw.get("transcript_txt"),
        "log_path": kw.get("log_path"),
        "error": kw.get("error"),
    }
    cols = ",".join(fields.keys())
    qs = ",".join(["?"] * len(fields))
    with connect() as con:
        con.execute(f"INSERT INTO jobs ({cols}) VALUES ({qs})", tuple(fields.values()))


def update_job(job_id: str, **kw) -> None:
    if kw.get("status") == "queued":
        kw["queued_at"] = queue_now()
    kw["updated_at"] = now()
    sets = ", ".join([f"{k}=?" for k in kw])
    vals = list(kw.values()) + [job_id]
    with connect() as con:
        con.execute(f"UPDATE jobs SET {sets} WHERE id=?", vals)
