from __future__ import annotations
import hashlib
import hmac
import os
import secrets
import shutil
import uuid
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_file, abort, flash, session
from werkzeug.utils import secure_filename
from . import db

ROOT = Path(__file__).resolve().parents[1]
UPLOADS = ROOT / "uploads"
RESULTS = ROOT / "results"
MAX_UPLOAD_MB = int(os.environ.get("MYAGENCY_MAX_UPLOAD_MB", "25"))
MAX_PDF_PAGES = int(os.environ.get("MYAGENCY_MAX_PDF_PAGES", "5"))
MAX_ACTIVE_JOBS = int(os.environ.get("MYAGENCY_MAX_ACTIVE_JOBS", "10"))
MAX_JOBS_PER_USER_DAY = int(os.environ.get("MYAGENCY_MAX_JOBS_PER_USER_DAY", "3"))
MAX_JOBS_PER_IP_DAY = int(os.environ.get("MYAGENCY_MAX_JOBS_PER_IP_DAY", "10"))
JOB_RETENTION_HOURS = int(os.environ.get("MYAGENCY_JOB_RETENTION_HOURS", "48"))
BACKGROUND_SITE_URL = os.environ.get("MYAGENCY_BACKGROUND_SITE_URL", "https://sites.mit.edu/mech-eng-analysis-ai/").strip()

app = Flask(__name__)
app.secret_key = os.environ.get("MYAGENCY_FLASK_SECRET", "development-secret-change-me")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


def _job_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]


def _hash_access_code(handle: str, code: str) -> str:
    secret = app.secret_key.encode("utf-8")
    msg = ((handle or "").upper() + "\n" + (code or "")).encode("utf-8")
    return hmac.new(secret, msg, hashlib.sha256).hexdigest()


def _generate_handle() -> str:
    # Avoid confusing characters like O/0 and I/1.
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    for _ in range(100):
        token = "".join(secrets.choice(alphabet) for _ in range(6))
        handle = f"MECH-{token}"
        if not db.user_by_handle(handle):
            return handle
    raise RuntimeError("Could not generate a unique handle.")


def _generate_access_code() -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    return "".join(secrets.choice(alphabet) for _ in range(6))


def _current_user():
    uid = session.get("user_id")
    if not uid:
        return None
    return db.user_by_id(int(uid))


def _server_key_available() -> bool:
    # start_server.sh sources secrets/openai.env; Render can set OPENAI_API_KEY directly.
    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    return bool(key)


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
    # For Render/proxies, X-Forwarded-For is typical. For local testing, remote_addr is fine.
    xff = (request.headers.get("X-Forwarded-For") or "").split(",")[0].strip()
    return xff or request.remote_addr or "unknown"


def _cleanup_old_jobs() -> None:
    old = db.old_jobs(JOB_RETENTION_HOURS)
    ids = []
    for job in old:
        ids.append(job["id"])
        try:
            if job["problem_dir"]:
                # problem_dir is .../results/<jobid>/Problem, so remove the parent job folder.
                p = Path(job["problem_dir"])
                job_root = p.parent if p.name == "Problem" else p
                if job_root.exists() and RESULTS in job_root.parents:
                    shutil.rmtree(job_root, ignore_errors=True)
        except Exception:
            pass
    db.delete_jobs(ids)


@app.before_request
def _before_request():
    db.init_db()
    # Lightweight cleanup; sufficient for the prototype. Later this can be a scheduled task.
    if request.endpoint not in {"static", "log"}:
        _cleanup_old_jobs()


@app.route("/")
def home():
    user = _current_user()
    return render_template("home.html", background_site_url=BACKGROUND_SITE_URL, user=user)


@app.route("/about")
def about():
    return render_template("about.html", background_site_url=BACKGROUND_SITE_URL)


@app.route("/new-user", methods=["GET", "POST"])
def new_user():
    if request.method == "GET":
        return render_template("new_user.html")
    handle = _generate_handle()
    access_code = _generate_access_code()
    user = db.create_user(handle, _hash_access_code(handle, access_code))
    session["user_id"] = int(user["id"])
    session["handle"] = handle
    return render_template("credentials.html", handle=handle, access_code=access_code)


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("login.html")
    handle = (request.form.get("handle") or "").strip().upper()
    access_code = (request.form.get("access_code") or "").strip().upper()
    user = db.user_by_handle(handle)
    if not user or user["access_code_hash"] != _hash_access_code(handle, access_code):
        flash("Unknown handle/access-code combination.")
        return redirect(url_for("login"))
    session["user_id"] = int(user["id"])
    session["handle"] = handle
    flash("Signed in.")
    return redirect(url_for("jobs"))


@app.route("/logout")
def logout():
    session.clear()
    flash("Signed out.")
    return redirect(url_for("home"))


@app.route("/submit", methods=["GET", "POST"])
def submit():
    user = _current_user()
    if not user:
        flash("Create a MyAgency ID or sign in before submitting a problem.")
        return redirect(url_for("home"))

    server_key = _server_key_available()
    if request.method == "GET":
        return render_template(
            "submit.html",
            user=user,
            max_upload_mb=MAX_UPLOAD_MB,
            max_pdf_pages=MAX_PDF_PAGES,
            server_key_available=server_key,
            max_jobs_per_user_day=MAX_JOBS_PER_USER_DAY,
        )

    if db.count_active_jobs() >= MAX_ACTIVE_JOBS:
        flash(f"The queue is currently full. Please try again shortly. Maximum active jobs: {MAX_ACTIVE_JOBS}.")
        return redirect(url_for("jobs"))

    since = db.cutoff(24)
    if db.count_jobs_for_user_since(int(user["id"]), since) >= MAX_JOBS_PER_USER_DAY:
        flash(f"This MyAgency ID has reached the limit of {MAX_JOBS_PER_USER_DAY} jobs in 24 hours.")
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
        user_id=int(user["id"]),
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
    user = _current_user()
    if not user:
        flash("Sign in with your MyAgency ID to view jobs.")
        return redirect(url_for("login"))
    return render_template("jobs.html", user=user, jobs=db.user_jobs(int(user["id"]), 50), retention_hours=JOB_RETENTION_HOURS)


@app.route("/job/<job_id>")
def job_status(job_id: str):
    job = db.one(job_id)
    if not job:
        abort(404)
    parent = db.one(job["parent_id"]) if job["parent_id"] else None
    return render_template("job_status.html", job=job, parent=parent)


@app.route("/job/<job_id>/continue", methods=["POST"])
def continue_job(job_id: str):
    user = _current_user()
    parent = db.one(job_id)
    if not parent:
        abort(404)
    if not user or int(parent["user_id"] or -1) != int(user["id"]):
        abort(403)
    if parent["status"] != "complete":
        flash("Continuation is available only after the selected job is complete.")
        return redirect(url_for("job_status", job_id=job_id))
    question = (request.form.get("question") or "").strip()
    if not question:
        flash("Please enter a follow-up question.")
        return redirect(url_for("job_status", job_id=job_id))

    jid = _job_id()
    db.insert_job(
        id=jid,
        user_id=int(user["id"]),
        kind="continuation",
        parent_id=job_id,
        status="queued",
        job_name=(parent["job_name"] or "Job") + " — follow-up",
        user_openai_key=parent["user_openai_key"],
        requester_ip=_client_ip(),
        original_filename=parent["original_filename"],
        problem_dir=parent["problem_dir"],
        prob_dir=parent["prob_dir"],
        stamp=parent["stamp"],
        question=question,
        result_html=parent["result_html"],
        transcript_txt=parent["transcript_txt"],
        log_path=str(ROOT / "logs" / f"{jid}.log"),
    )
    return redirect(url_for("job_status", job_id=jid))


@app.route("/result/<job_id>")
def result(job_id: str):
    job = db.one(job_id)
    if not job or not job["result_html"]:
        abort(404)
    path = Path(job["result_html"])
    if not path.exists():
        abort(404)
    return send_file(path)


@app.route("/download/<job_id>")
def download(job_id: str):
    job = db.one(job_id)
    if not job or not job["result_html"]:
        abort(404)
    path = Path(job["result_html"])
    if not path.exists():
        abort(404)
    return send_file(path, as_attachment=True, download_name=path.name)


@app.route("/log/<job_id>")
def log(job_id: str):
    job = db.one(job_id)
    if not job or not job["log_path"]:
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
