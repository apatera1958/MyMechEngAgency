from __future__ import annotations
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "jobs" / "jobs.sqlite"

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
    }
}

def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def cutoff(hours: int) -> str:
    return (datetime.now() - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")

def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def _ensure_columns(con: sqlite3.Connection, table: str, cols: dict[str, str]) -> None:
    existing = {r[1] for r in con.execute(f"PRAGMA table_info({table})").fetchall()}
    for col, typ in cols.items():
        if col not in existing:
            con.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")

def init_db() -> None:
    with connect() as con:
        con.executescript(SCHEMA)
        for table, cols in MIGRATION_COLUMNS.items():
            _ensure_columns(con, table, cols)

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
            "SELECT jobs.*, users.handle FROM jobs LEFT JOIN users ON jobs.user_id=users.id ORDER BY jobs.created_at DESC LIMIT ?",
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
    with connect() as con:
        return con.execute(
            "SELECT * FROM jobs WHERE status='queued' ORDER BY created_at ASC LIMIT 1"
        ).fetchone()

def count_jobs_for_user_since(user_id: int, since: str) -> int:
    with connect() as con:
        row = con.execute(
            "SELECT COUNT(*) AS n FROM jobs WHERE user_id=? AND created_at>=? AND kind='analysis'",
            (user_id, since),
        ).fetchone()
        return int(row["n"])

def count_jobs_for_ip_since(ip: str, since: str) -> int:
    with connect() as con:
        row = con.execute(
            "SELECT COUNT(*) AS n FROM jobs WHERE requester_ip=? AND created_at>=? AND kind='analysis'",
            (ip, since),
        ).fetchone()
        return int(row["n"])

def count_active_jobs() -> int:
    with connect() as con:
        row = con.execute("SELECT COUNT(*) AS n FROM jobs WHERE status IN ('queued','running')").fetchone()
        return int(row["n"])

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
        "id": kw["id"], "user_id": kw.get("user_id"),
        "kind": kw.get("kind", "analysis"), "parent_id": kw.get("parent_id"),
        "status": kw.get("status", "queued"), "created_at": ts, "updated_at": ts,
        "job_name": kw.get("job_name"),
        "owner_email": kw.get("owner_email"), "access_code_hash": kw.get("access_code_hash"),
        "session_id": kw.get("session_id"),
        "user_openai_key": kw.get("user_openai_key"), "requester_ip": kw.get("requester_ip"),
        "original_filename": kw.get("original_filename"),
        "pdf_page_count": kw.get("pdf_page_count"),
        "hints_filename": kw.get("hints_filename"),
        "models_filename": kw.get("models_filename"),
        "input_pdf": kw.get("input_pdf"), "problem_dir": kw.get("problem_dir"), "prob_dir": kw.get("prob_dir"),
        "stamp": kw.get("stamp"), "n_agents": kw.get("n_agents"), "question": kw.get("question"),
        "result_html": kw.get("result_html"), "transcript_txt": kw.get("transcript_txt"),
        "log_path": kw.get("log_path"), "error": kw.get("error"),
    }
    cols = ",".join(fields.keys())
    qs = ",".join(["?"] * len(fields))
    with connect() as con:
        con.execute(f"INSERT INTO jobs ({cols}) VALUES ({qs})", tuple(fields.values()))

def update_job(job_id: str, **kw) -> None:
    kw["updated_at"] = now()
    sets = ", ".join([f"{k}=?" for k in kw])
    vals = list(kw.values()) + [job_id]
    with connect() as con:
        con.execute(f"UPDATE jobs SET {sets} WHERE id=?", vals)
