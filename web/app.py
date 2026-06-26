from __future__ import annotations
import os
import shutil
import uuid
import re
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_file, abort, flash, session, jsonify, Response
from werkzeug.utils import secure_filename
from . import db

ROOT = Path(__file__).resolve().parents[1]
UPLOADS = ROOT / "uploads"
RESULTS = ROOT / "results"
MAX_UPLOAD_MB = int(os.environ.get("MYAGENCY_MAX_UPLOAD_MB", "25"))
MAX_PDF_PAGES = int(os.environ.get("MYAGENCY_MAX_PDF_PAGES", "5"))
MAX_ACTIVE_JOBS = int(os.environ.get("MYAGENCY_MAX_ACTIVE_JOBS", "10"))
MAX_JOBS_PER_SESSION_DAY = int(os.environ.get("MYAGENCY_MAX_JOBS_PER_SESSION_DAY", "3"))
MAX_JOBS_PER_IP_DAY = int(os.environ.get("MYAGENCY_MAX_JOBS_PER_IP_DAY", "10"))
JOB_RETENTION_HOURS = int(os.environ.get("MYAGENCY_JOB_RETENTION_HOURS", "48"))
BACKGROUND_SITE_URL = os.environ.get("MYAGENCY_BACKGROUND_SITE_URL", "https://sites.mit.edu/mech-eng-analysis-ai/").strip()
SITE_PASSWORD = os.environ.get("MYAGENCY_SITE_PASSWORD", "").strip()

app = Flask(__name__)
app.secret_key = os.environ.get("MYAGENCY_FLASK_SECRET", "development-secret-change-me")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


def _job_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]


def _server_key_available() -> bool:
    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    return bool(key)


def _ensure_session_id() -> str:
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


def _transcript_version(job) -> float:
    """Return the mtime of the current transcript HTML, or 0.0 if unavailable."""
    try:
        if not job or not job["result_html"]:
            return 0.0
        p = Path(job["result_html"])
        if not p.exists():
            return 0.0
        return float(p.stat().st_mtime)
    except Exception:
        return 0.0


def _rewrite_transcript_links_for_live_view(html: str, job_id: str) -> str:
    """Rewrite Bundle-relative links so they work in Live Transcript.

    The saved transcript HTML is intentionally portable inside Bundle.zip,
    where links such as ProblemStatement.pdf are correct relative links.
    The Live Transcript view is served from Flask routes, so those same
    relative links would otherwise resolve incorrectly. This function only
    rewrites the browser/live copy; it does not modify the saved transcript
    or the Bundle.
    """
    ps_url = url_for("problem_statement", job_id=job_id)

    # Common exact forms. These also catch JavaScript/window.open uses that
    # quote the bare relative filename rather than using an href attribute.
    for rel in ("ProblemStatement.pdf", "./ProblemStatement.pdf", "PROB/ProblemStatement.pdf", "./PROB/ProblemStatement.pdf"):
        html = html.replace(f'"{rel}"', f'"{ps_url}"')
        html = html.replace(f"'{rel}'", f"'{ps_url}'")

    # More general href/src attributes ending in ProblemStatement.pdf,
    # including paths such as ../PROB/ProblemStatement.pdf.
    html = re.sub(
        r'(?i)(\b(?:href|src)=\s*["\'])([^"\']*/)?ProblemStatement\.pdf(["\'])',
        lambda m: f"{m.group(1)}{ps_url}{m.group(3)}",
        html,
    )
    return html


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
            server_key_available=server_key,
            max_jobs_per_session_day=MAX_JOBS_PER_SESSION_DAY,
            job_name_suggestions=_session_job_name_suggestions(sid),
        )

    if db.count_active_jobs() >= MAX_ACTIVE_JOBS:
        flash(f"The queue is currently full. Please try again shortly. Maximum active jobs: {MAX_ACTIVE_JOBS}.")
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

    n_agents = int(request.form.get("n_agents") or "2")
    n_agents = max(1, min(n_agents, 8))

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
        log_path=str(ROOT / "logs" / f"{jid}.log"),
    )
    flash("Job queued.")
    return redirect(url_for("job_status", job_id=jid))


@app.route("/jobs")
def jobs():
    auth = _require_auth()
    if auth:
        return auth
    sid = _ensure_session_id()
    return render_template("jobs.html", jobs=db.session_jobs(sid, 50), retention_hours=JOB_RETENTION_HOURS)


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

    return render_template(
        "job_status.html",
        job=job,
        parent=parent,
        bundle=bundle,
    )


@app.route("/live/<job_id>")
def live_transcript(job_id: str):
    """Wrapper page for a non-disruptive live transcript view."""
    auth = _require_auth()
    if auth:
        return auth
    job = db.one(job_id)
    if not _session_can_access(job):
        abort(404)
    return render_template(
        "live_transcript.html",
        job=job,
        transcript_version=_transcript_version(job),
    )


@app.route("/live-status/<job_id>")
def live_status(job_id: str):
    """Small polling endpoint used by the Live Transcript page.

    It reports whether the transcript file has changed without forcing
    the visible transcript iframe to reload. The user chooses when to
    load the update.
    """
    auth = _require_auth()
    if auth:
        return auth
    job = db.one(job_id)
    if not _session_can_access(job):
        abort(404)

    version = _transcript_version(job)
    return jsonify(
        {
            "job_id": job["id"],
            "status": job["status"],
            "updated_at": job["updated_at"],
            "has_transcript": bool(job["result_html"] and version > 0.0),
            "transcript_version": version,
            "result_url": url_for("live_result", job_id=job["id"]) if job["result_html"] and version > 0.0 else None,
            "bundle_url": url_for("download", job_id=job["id"]) if job["result_html"] and _latest_bundle(job["prob_dir"]) else None,
        }
    )


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


@app.route("/problem-statement/<job_id>")
def problem_statement(job_id: str):
    auth = _require_auth()
    if auth:
        return auth
    job = db.one(job_id)
    if not _session_can_access(job) or not job["input_pdf"]:
        abort(404)
    path = Path(job["input_pdf"])
    if not path.exists():
        abort(404)
    return send_file(path, mimetype="application/pdf")


@app.route("/live-result/<job_id>")
def live_result(job_id: str):
    """Serve transcript HTML for the Live Transcript iframe.

    This leaves the stored transcript untouched, but rewrites links that are
    valid inside Bundle.zip and invalid from a Flask route.
    """
    auth = _require_auth()
    if auth:
        return auth
    job = db.one(job_id)
    if not _session_can_access(job) or not job["result_html"]:
        abort(404)
    path = Path(job["result_html"])
    if not path.exists():
        abort(404)
    html = path.read_text(encoding="utf-8", errors="ignore")
    html = _rewrite_transcript_links_for_live_view(html, job_id)
    return Response(html, mimetype="text/html")


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


@app.route("/admin", methods=["GET", "POST"])
def admin():
    password = os.environ.get("MYAGENCY_ADMIN_PASSWORD", "")
    if password:
        if request.method == "POST":
            if (request.form.get("password") or "") == password:
                session["admin_ok"] = True
            else:
                flash("Incorrect admin password.")
        if not session.get("admin_ok"):
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
