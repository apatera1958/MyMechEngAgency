from __future__ import annotations
import os
import shutil
import uuid
import re
import time
from pathlib import Path
from datetime import datetime, timedelta
from flask import Flask, render_template, request, redirect, url_for, send_file, abort, flash, session, Response
from werkzeug.utils import secure_filename
from . import db

ROOT = Path(__file__).resolve().parents[1]

# Runtime state can be redirected to a persistent Render disk while preserving
# the existing local-development layout when MYAGENCY_STORAGE_ROOT is unset.
_STORAGE_ROOT_RAW = os.environ.get("MYAGENCY_STORAGE_ROOT", "").strip()
STORAGE_ROOT = Path(_STORAGE_ROOT_RAW).expanduser() if _STORAGE_ROOT_RAW else ROOT
UPLOADS = STORAGE_ROOT / "uploads"
RESULTS = STORAGE_ROOT / "results"
LOGS = STORAGE_ROOT / "logs"

for _directory in (UPLOADS, RESULTS, LOGS):
    _directory.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD_MB = int(os.environ.get("MYAGENCY_MAX_UPLOAD_MB", "25"))
MAX_PDF_PAGES = int(os.environ.get("MYAGENCY_MAX_PDF_PAGES", "5"))
MAX_AGENT_SOLVES_PER_JOB = max(1, int(os.environ.get("MYAGENCY_MAX_AGENT_SOLVES_PER_JOB", "4")))
MAX_ACTIVE_JOBS = int(os.environ.get("MYAGENCY_MAX_ACTIVE_JOBS", "10"))
MAX_ACTIVE_JOBS_PER_SESSION = max(1, int(os.environ.get("MYAGENCY_MAX_ACTIVE_JOBS_PER_SESSION", "1")))
BACKGROUND_JOB_CONCURRENCY = max(1, int(os.environ.get("BACKGROUND_JOB_CONCURRENCY", "1")))
ESTIMATED_JOB_MINUTES = max(1, int(os.environ.get("MYAGENCY_ESTIMATED_JOB_MINUTES", "5")))
MAX_JOBS_PER_SESSION_DAY = int(os.environ.get("MYAGENCY_MAX_JOBS_PER_SESSION_DAY", "3"))
MAX_JOBS_PER_IP_DAY = int(os.environ.get("MYAGENCY_MAX_JOBS_PER_IP_DAY", "10"))
JOB_RETENTION_HOURS = int(os.environ.get("MYAGENCY_JOB_RETENTION_HOURS", "48"))
SESSION_LIFETIME_DAYS = int(os.environ.get("MYAGENCY_SESSION_LIFETIME_DAYS", "30"))
ADMIN_SESSION_MINUTES = int(os.environ.get("MYAGENCY_ADMIN_SESSION_MINUTES", "60"))
BACKGROUND_SITE_URL = os.environ.get("MYAGENCY_BACKGROUND_SITE_URL", "https://sites.mit.edu/mech-eng-analysis-ai/").strip()
SITE_PASSWORD = os.environ.get("MYAGENCY_SITE_PASSWORD", "").strip()

app = Flask(__name__)
app.secret_key = os.environ.get("MYAGENCY_FLASK_SECRET", "development-secret-change-me")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=SESSION_LIFETIME_DAYS)
app.config["SESSION_REFRESH_EACH_REQUEST"] = True


def _job_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]


def _server_key_available() -> bool:
    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    return bool(key)


def _ensure_session_id() -> str:
    # Keep the browser identity across Render deploys/restarts and ordinary
    # browser restarts.  The signed cookie remains protected by the fixed
    # MYAGENCY_FLASK_SECRET.
    session.permanent = True
    sid = session.get("session_id")
    if not sid:
        sid = uuid.uuid4().hex
        session["session_id"] = sid
    return sid


def _is_authenticated() -> bool:
    # If no site password is configured, the site is open.
    return (not SITE_PASSWORD) or bool(session.get("site_ok"))


def _require_auth():
    if not _is_authenticated():
        flash("Please enter the site password.")
        return redirect(url_for("enter"))
    _ensure_session_id()
    return None


def _pdf_page_count(pdf_path: Path) -> int:
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(pdf_path))
        return len(reader.pages)
    except Exception as e:
        raise ValueError(f"Could not read PDF page count: {e}") from e


def _optional_upload(field_name: str, target_path: Path, allowed_suffixes: tuple[str, ...]) -> str | None:
    f = request.files.get(field_name)
    if not f or not f.filename:
        return None
    original = secure_filename(f.filename)
    if not original:
        raise ValueError("Optional file has an invalid filename.")
    if not original.lower().endswith(allowed_suffixes):
        allowed = ", ".join(allowed_suffixes)
        raise ValueError(f"{original} must have one of these extensions: {allowed}")
    f.save(target_path)
    return original


def _client_ip() -> str:
    xff = (request.headers.get("X-Forwarded-For") or "").split(",")[0].strip()
    return xff or request.remote_addr or "unknown"


def _session_active_limit_message() -> str:
    if MAX_ACTIVE_JOBS_PER_SESSION == 1:
        return (
            "This browser already has an active job (queued or running). "
            "Please wait for it to finish before starting another."
        )
    return (
        f"This browser already has {MAX_ACTIVE_JOBS_PER_SESSION} active jobs "
        "(queued or running). Please wait for one to finish before starting another."
    )


def _cleanup_old_jobs() -> None:
    old = db.old_jobs(JOB_RETENTION_HOURS)
    ids = []
    for job in old:
        ids.append(job["id"])
        try:
            if job["problem_dir"]:
                p = Path(job["problem_dir"])
                job_root = p.parent if p.name == "Problem" else p
                if job_root.exists() and RESULTS in job_root.parents:
                    shutil.rmtree(job_root, ignore_errors=True)
        except Exception:
            pass
    db.delete_jobs(ids)


def _latest_bundle(prob_dir: Path | str | None) -> Path | None:
    if not prob_dir:
        return None
    p = Path(prob_dir)
    if not p.exists():
        return None
    bundles = sorted(p.glob("Bundle_*.zip"), key=lambda x: x.stat().st_mtime, reverse=True)
    return bundles[0] if bundles else None


def _exploration_notebook_path(job) -> Path | None:
    """Return the generated Explore notebook path when it exists."""
    try:
        if job and job["prob_dir"]:
            path = Path(job["prob_dir"]) / "Engineering_Exploration.ipynb"
            if path.exists():
                return path
    except Exception:
        pass
    return None


def _prepare_transcript_for_browser(html: str) -> str:
    """Prepare transcript.html for online viewing only.

    The saved transcript.html and the downloadable Bundle are left unchanged.
    For the online transcript view, remove links/buttons that point to the
    Bundle-local ProblemStatement.pdf, because the online session already
    identifies the problem and the Bundle remains the archival copy.
    """

    # Remove quoted anchor tags whose href ends in ProblemStatement.pdf.
    # This covers buttons implemented as anchors, e.g.
    # <a class="button" href="ProblemStatement.pdf">Problem Statement</a>.
    html = re.sub(
        r"<a\b(?=[^>]*\bhref\s*=\s*[\"'][^\"']*ProblemStatement\.pdf[\"'])(?:[^>]|\n)*?</a>",
        "",
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # Remove unquoted anchor tags whose href ends in ProblemStatement.pdf.
    html = re.sub(
        r"<a\b(?=[^>]*\bhref\s*=\s*[^\s>]*ProblemStatement\.pdf)(?:[^>]|\n)*?</a>",
        "",
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )

    return html


def _problem_statement_path_for_job(job) -> Path | None:
    """Return the best available path to the job's uploaded PDF."""
    candidates: list[Path] = []

    try:
        if job and job["input_pdf"]:
            candidates.append(Path(job["input_pdf"]))
    except Exception:
        pass

    try:
        if job and job["prob_dir"]:
            candidates.append(Path(job["prob_dir"]) / "ProblemStatement.pdf")
    except Exception:
        pass

    try:
        if job and job["problem_dir"]:
            candidates.append(Path(job["problem_dir"]) / "PROB" / "ProblemStatement.pdf")
    except Exception:
        pass

    for path in candidates:
        if path.exists():
            return path
    return None

def _next_follow_on_number_for_job(job) -> int:
    """Return the next Follow-On number for this job's transcript/html.

    This is used only for user-facing labels at queue time; the worker
    recomputes the number from the actual transcript when it appends.
    """
    try:
        html_path = Path(job["result_html"]) if job and job["result_html"] else None
        if html_path and html_path.exists():
            text = html_path.read_text(encoding="utf-8", errors="ignore")
            nums = [int(m.group(1)) for m in re.finditer(r">\s*Follow-On\s+(\d+)\s*<", text)]
            return (max(nums) + 1) if nums else 1
    except Exception:
        pass
    return 1


def _session_job_name_suggestions(session_id: str) -> list[str]:
    """Job-name suggestions limited to the current browser session."""
    seen: set[str] = set()
    out: list[str] = []
    try:
        for job in db.session_jobs(session_id, 100):
            name = (job["job_name"] or "").strip()
            if name and name not in seen:
                seen.add(name)
                out.append(name)
    except Exception:
        pass
    return out


@app.before_request
def _before_request():
    db.init_db()
    if request.endpoint not in {"static", "log"}:
        _cleanup_old_jobs()


@app.after_request
def _no_cache_for_dynamic_pages(response):
    """Avoid stale job/status pages in browsers and on Render/proxies."""
    try:
        if request.endpoint not in {"static"}:
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
    except Exception:
        pass
    return response


@app.route("/")
def home():
    return render_template(
        "home.html",
        background_site_url=BACKGROUND_SITE_URL,
        authenticated=_is_authenticated(),
        site_password_required=bool(SITE_PASSWORD),
    )


@app.route("/about")
def about():
    return render_template("about.html", background_site_url=BACKGROUND_SITE_URL)


@app.route("/enter", methods=["GET", "POST"])
def enter():
    if not SITE_PASSWORD:
        session["site_ok"] = True
        _ensure_session_id()
        return redirect(url_for("home"))
    if request.method == "POST":
        if (request.form.get("password") or "") == SITE_PASSWORD:
            session["site_ok"] = True
            _ensure_session_id()
            flash("Site access granted.")
            return redirect(url_for("home"))
        flash("Incorrect site password.")
    return render_template("enter.html")


@app.route("/reset-session")
def reset_session():
    session.clear()
    flash("Session cleared.")
    return redirect(url_for("home"))


@app.route("/submit", methods=["GET", "POST"])
def submit():
    auth = _require_auth()
    if auth:
        return auth
    sid = _ensure_session_id()

    server_key = _server_key_available()
    if request.method == "GET":
        return render_template(
            "submit.html",
            max_upload_mb=MAX_UPLOAD_MB,
            max_pdf_pages=MAX_PDF_PAGES,
            max_agent_solves_per_job=MAX_AGENT_SOLVES_PER_JOB,
            server_key_available=server_key,
            max_jobs_per_session_day=MAX_JOBS_PER_SESSION_DAY,
            job_name_suggestions=_session_job_name_suggestions(sid),
        )

    if db.count_active_jobs() >= MAX_ACTIVE_JOBS:
        flash(f"The queue is currently full. Please try again shortly. Maximum active jobs: {MAX_ACTIVE_JOBS}.")
        return redirect(url_for("jobs"))

    if db.count_active_jobs_for_session(sid) >= MAX_ACTIVE_JOBS_PER_SESSION:
        flash(_session_active_limit_message())
        return redirect(url_for("jobs"))

    since = db.cutoff(24)
    if db.count_jobs_for_session_since(sid, since) >= MAX_JOBS_PER_SESSION_DAY:
        flash(f"This browser session has reached the limit of {MAX_JOBS_PER_SESSION_DAY} jobs in 24 hours.")
        return redirect(url_for("jobs"))

    ip = _client_ip()
    if db.count_jobs_for_ip_since(ip, since) >= MAX_JOBS_PER_IP_DAY:
        flash("This network address has reached the daily demo limit. Please try again tomorrow.")
        return redirect(url_for("jobs"))

    user_key = None
    if not server_key:
        user_key = (request.form.get("openai_api_key") or "").strip()
        if not user_key:
            flash("This server has no OpenAI API key configured. Please enter your OpenAI API key for this job.")
            return redirect(url_for("submit"))

    job_name = (request.form.get("job_name") or "").strip()
    if not job_name:
        flash("Please provide a short job name, such as 'Beam warmup' or 'Heat exchanger test'.")
        return redirect(url_for("submit"))

    f = request.files.get("pdf")
    if not f or not f.filename:
        flash("Please choose a PDF file.")
        return redirect(url_for("submit"))
    name = secure_filename(f.filename)
    if not name.lower().endswith(".pdf"):
        flash("Please upload a .pdf file.")
        return redirect(url_for("submit"))

    jid = _job_id()
    job_root = RESULTS / jid
    problem_dir = job_root / "Problem"
    prob_dir = problem_dir / "PROB"
    prob_dir.mkdir(parents=True, exist_ok=True)
    upload_path = prob_dir / "ProblemStatement.pdf"
    f.save(upload_path)

    try:
        page_count = _pdf_page_count(upload_path)
    except ValueError as e:
        shutil.rmtree(job_root, ignore_errors=True)
        flash(str(e))
        return redirect(url_for("submit"))

    if page_count > MAX_PDF_PAGES:
        shutil.rmtree(job_root, ignore_errors=True)
        flash(f"Please upload a PDF with at most {MAX_PDF_PAGES} pages. This file has {page_count} pages.")
        return redirect(url_for("submit"))

    try:
        hints_filename = _optional_upload("hints_file", prob_dir / "PROBhints.txt", (".txt",))
        models_filename = _optional_upload("models_file", prob_dir / "PROBgpt_models.yaml", (".yaml", ".yml"))
    except ValueError as e:
        shutil.rmtree(job_root, ignore_errors=True)
        flash(str(e))
        return redirect(url_for("submit"))

    try:
        n_agents = int(request.form.get("n_agents") or "2")
    except (TypeError, ValueError):
        n_agents = 2
    n_agents = max(1, min(n_agents, MAX_AGENT_SOLVES_PER_JOB))

    db.insert_job(
        id=jid,
        session_id=sid,
        kind="analysis",
        status="queued",
        job_name=job_name,
        user_openai_key=user_key,
        requester_ip=ip,
        original_filename=name,
        pdf_page_count=page_count,
        hints_filename=hints_filename,
        models_filename=models_filename,
        input_pdf=str(upload_path),
        problem_dir=str(problem_dir),
        prob_dir=str(prob_dir),
        n_agents=n_agents,
        log_path=str(LOGS / f"{jid}.log"),
    )
    flash("Job queued.")
    return redirect(url_for("job_status", job_id=jid))


@app.route("/jobs")
def jobs():
    auth = _require_auth()
    if auth:
        return auth
    sid = _ensure_session_id()
    session_jobs = db.session_jobs(sid, 50)
    notebook_jobs = {job["id"] for job in session_jobs if _exploration_notebook_path(job)}
    return render_template(
        "jobs.html",
        jobs=session_jobs,
        notebook_jobs=notebook_jobs,
        retention_hours=JOB_RETENTION_HOURS,
    )


def _session_can_access(job) -> bool:
    if not job:
        return False
    return bool(job["session_id"]) and job["session_id"] == session.get("session_id")


@app.route("/job/<job_id>")
def job_status(job_id: str):
    auth = _require_auth()
    if auth:
        return auth
    job = db.one(job_id)
    if not _session_can_access(job):
        abort(404)
    parent = db.one(job["parent_id"]) if job["parent_id"] else None
    bundle = _latest_bundle(job["prob_dir"])

    queue_info = None
    if job["status"] == "queued":
        snapshot = db.queue_snapshot(job_id)
        if snapshot:
            jobs_ahead = snapshot["running_jobs"] + snapshot["position"] - 1
            batches_ahead = (
                jobs_ahead + BACKGROUND_JOB_CONCURRENCY - 1
            ) // BACKGROUND_JOB_CONCURRENCY
            queue_info = {
                **snapshot,
                "worker_concurrency": BACKGROUND_JOB_CONCURRENCY,
                "estimated_wait_minutes": batches_ahead * ESTIMATED_JOB_MINUTES,
            }

    return render_template(
        "job_status.html",
        job=job,
        parent=parent,
        bundle=bundle,
        exploration_notebook=_exploration_notebook_path(job),
        queue_info=queue_info,
    )


@app.route("/job/<job_id>/explore", methods=["POST"])
def explore_job(job_id: str):
    auth = _require_auth()
    if auth:
        return auth
    job = db.one(job_id)
    if not _session_can_access(job):
        abort(404)
    if job["status"] != "complete":
        flash("Explore is available only after the selected job is complete.")
        return redirect(url_for("job_status", job_id=job_id))
    if not job["stamp"] or not job["transcript_txt"] or not job["result_html"]:
        flash("Explore is unavailable because the transcript is not ready.")
        return redirect(url_for("job_status", job_id=job_id))
    if db.count_active_jobs() >= MAX_ACTIVE_JOBS:
        flash(f"The queue is currently full. Please try again shortly. Maximum active jobs: {MAX_ACTIVE_JOBS}.")
        return redirect(url_for("job_status", job_id=job_id))
    if db.count_active_jobs_for_session(job["session_id"]) >= MAX_ACTIVE_JOBS_PER_SESSION:
        flash(_session_active_limit_message())
        return redirect(url_for("job_status", job_id=job_id))

    db.update_job(
        job_id,
        kind="explore",
        status="queued",
        error=None,
    )
    flash("Engineering Exploration notebook queued.")
    return redirect(url_for("job_status", job_id=job_id))


@app.route("/job/<job_id>/continue", methods=["POST"])
def continue_job(job_id: str):
    auth = _require_auth()
    if auth:
        return auth
    job = db.one(job_id)
    if not _session_can_access(job):
        abort(404)
    if job["status"] != "complete":
        flash("Continuation is available only after the selected job is complete.")
        return redirect(url_for("job_status", job_id=job_id))
    question = (request.form.get("question") or "").strip()
    if not question:
        flash("Please enter a follow-up question.")
        return redirect(url_for("job_status", job_id=job_id))
    if not job["stamp"] or not job["transcript_txt"] or not job["result_html"]:
        flash("Continuation is unavailable because the transcript is not ready.")
        return redirect(url_for("job_status", job_id=job_id))
    if db.count_active_jobs() >= MAX_ACTIVE_JOBS:
        flash(f"The queue is currently full. Please try again shortly. Maximum active jobs: {MAX_ACTIVE_JOBS}.")
        return redirect(url_for("job_status", job_id=job_id))
    if db.count_active_jobs_for_session(job["session_id"]) >= MAX_ACTIVE_JOBS_PER_SESSION:
        flash(_session_active_limit_message())
        return redirect(url_for("job_status", job_id=job_id))

    # Reuse the same visible job row. The worker will append the follow-on
    # to the existing transcript and rebuild the single current Bundle.
    db.update_job(
        job_id,
        kind="continuation",
        status="queued",
        question=question,
        error=None,
    )
    flash("Follow-on queued for this problem.")
    return redirect(url_for("job_status", job_id=job_id))


@app.route("/result/<job_id>")
def result(job_id: str):
    auth = _require_auth()
    if auth:
        return auth
    job = db.one(job_id)
    if not _session_can_access(job) or not job["result_html"]:
        abort(404)
    path = Path(job["result_html"])
    if not path.exists():
        abort(404)
    return send_file(path)




@app.route("/transcript/<job_id>")
def transcript(job_id: str):
    auth = _require_auth()
    if auth:
        return auth
    job = db.one(job_id)
    if not _session_can_access(job) or not job["result_html"]:
        abort(404)
    path = Path(job["result_html"])
    if not path.exists():
        abort(404)
    session["current_transcript_job_id"] = job_id
    html = path.read_text(encoding="utf-8", errors="ignore")
    html = _prepare_transcript_for_browser(html)
    return Response(html, mimetype="text/html")


@app.route("/transcript/ProblemStatement.pdf")
def transcript_problem_statement_compat():
    """Compatibility route for old/unrewritten relative transcript links."""
    auth = _require_auth()
    if auth:
        return auth
    job_id = session.get("current_transcript_job_id")
    if not job_id:
        abort(404)
    job = db.one(job_id)
    if not _session_can_access(job):
        abort(404)
    path = _problem_statement_path_for_job(job)
    if not path:
        abort(404)
    return send_file(path, mimetype="application/pdf")


@app.route("/problem-statement/<job_id>")
def problem_statement(job_id: str):
    auth = _require_auth()
    if auth:
        return auth
    job = db.one(job_id)
    if not _session_can_access(job):
        abort(404)
    path = _problem_statement_path_for_job(job)
    if not path:
        abort(404)
    return send_file(path, mimetype="application/pdf")


@app.route("/download/<job_id>")
def download(job_id: str):
    auth = _require_auth()
    if auth:
        return auth
    job = db.one(job_id)
    if not _session_can_access(job):
        abort(404)
    bundle = _latest_bundle(job["prob_dir"])
    if not bundle or not bundle.exists():
        abort(404)
    return send_file(bundle, as_attachment=True, download_name=bundle.name)


@app.route("/notebook/<job_id>")
def download_notebook(job_id: str):
    auth = _require_auth()
    if auth:
        return auth
    job = db.one(job_id)
    if not _session_can_access(job):
        abort(404)
    path = _exploration_notebook_path(job)
    if not path:
        abort(404)
    return send_file(path, as_attachment=True, download_name="Engineering_Exploration.ipynb")


@app.route("/log/<job_id>")
def log(job_id: str):
    auth = _require_auth()
    if auth:
        return auth
    job = db.one(job_id)
    if not _session_can_access(job) or not job["log_path"]:
        abort(404)
    path = Path(job["log_path"])
    if not path.exists():
        return "Log file has not been created yet.", 200, {"Content-Type": "text/plain; charset=utf-8"}
    return send_file(path, mimetype="text/plain")


def _admin_authorized() -> bool:
    """Return True only while the short-lived admin authorization is valid."""
    try:
        expires_at = float(session.get("admin_ok_until", 0))
    except (TypeError, ValueError):
        expires_at = 0.0

    if expires_at <= time.time():
        session.pop("admin_ok_until", None)
        return False
    return True


@app.route("/admin/session-debug")
def admin_session_debug():
    """Small diagnostic for checking browser-session continuity after deploys."""
    password = os.environ.get("MYAGENCY_ADMIN_PASSWORD", "").strip()
    if not password:
        return "Admin access is not configured.", 503
    if not _admin_authorized():
        return redirect(url_for("admin"))

    sid = session.get("session_id")
    rows = db.recent(20)
    lines = [
        f"Current browser session_id: {sid or '(none)'}",
        "",
        "Recent jobs:",
    ]
    for row in rows:
        lines.append(
            f"{row['id']}  status={row['status']}  "
            f"job_name={row['job_name'] or ''!r}  "
            f"session_id={row['session_id'] or '(none)'}"
        )
    return Response("\n".join(lines) + "\n", mimetype="text/plain")


@app.route("/admin", methods=["GET", "POST"])
def admin():
    password = os.environ.get("MYAGENCY_ADMIN_PASSWORD", "").strip()

    # Fail closed: /admin is unavailable unless an admin password is configured.
    if not password:
        return "Admin access is not configured.", 503

    if request.method == "POST":
        if (request.form.get("password") or "") == password:
            session["admin_ok_until"] = time.time() + ADMIN_SESSION_MINUTES * 60
            return redirect(url_for("admin"))
        flash("Incorrect admin password.")

    if not _admin_authorized():
        return render_template("admin_login.html")
    return render_template(
        "admin.html",
        jobs=db.recent(100),
        active_jobs=db.count_active_jobs(),
        max_active_jobs=MAX_ACTIVE_JOBS,
        retention_hours=JOB_RETENTION_HOURS,
        server_key_available=_server_key_available(),
    )


if __name__ == "__main__":
    db.init_db()
    app.run(host="127.0.0.1", port=5000, debug=True)
