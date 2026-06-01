#!/usr/bin/env python3
"""
agency_client.py - Responses-first Agency client (with optional concurrency and chat)

Features:
- Uploads PDF in main process -> file_id
- Solve workers using OpenAI Responses API (optionally via ProcessPool)
- Per-worker diagnostics written into PROB: worker_start_*.json, worker_ok_*.json, worker_err_*.json
- Aggregated response summary resp_<stamp>.json with per-call datestamp
- Transcript transcript_<stamp>.txt + optional HTML transcript_<stamp>.html
- Chat-only continuation mode that reads an existing transcript and launches Agent Chat
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Tuple

try:
    from openai import OpenAI
except Exception as e:  # pragma: no cover - environment check
    print("ERROR: openai package with OpenAI client is required.", file=sys.stderr)
    raise


# ---------------- Helpers ----------------
def datestamp(fmt: str) -> str:
    return datetime.now().strftime(fmt)


def write_json_safe(p: Path, obj: Any) -> None:
    try:
        p.write_text(json.dumps(obj, default=str, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"[warn] could not write {p}: {e}", flush=True)


#NEW BLOCK below (3/14) which generalizes output (and pretties up code input/output)
def unroll_text(s: str) -> str:
    """
    Make tool code/log strings readable.

    - Normalizes Windows newlines (CRLF/CR) to '\n'
    - If the string seems to contain *literal* backslash escapes (e.g. '\\n')
      and has no real newlines, convert those to real newlines/tabs.
    """
    if s is None:
        return ""
    s = str(s)

    # Normalize actual newlines first
    s = s.replace("\r\n", "\n").replace("\r", "\n")

    # Conservative unescape of literal backslash sequences (only if it looks needed)
    if "\\n" in s and "\n" not in s:
        s = (
            s.replace("\\r\\n", "\n")
             .replace("\\n", "\n")
             .replace("\\t", "\t")
        )

    return s


def extract_text_from_resp(resp) -> str:
    """
    Extract a readable markdown transcript from a Responses API result.

    Includes:
      - assistant output_text (and summary_text/refusal if present)
      - code_interpreter_call code (optional; easy to comment out)
      - code_interpreter_call stdout/stderr logs (the printed results)

    Note: to actually receive logs, your request must include:
      include=["code_interpreter_call.outputs"]
    """
    # Toggle(s) you can comment out / flip later
    SHOW_CODE = True
    SHOW_LOGS = True

    try:
        # Convert resp -> dict robustly across SDK versions
        if isinstance(resp, dict):
            d = resp
        elif hasattr(resp, "to_dict"):
            d = resp.to_dict()
        elif hasattr(resp, "model_dump"):
            d = resp.model_dump()
        elif hasattr(resp, "dict"):
            d = resp.dict()
        else:
            return unroll_text(str(resp)).strip()

        out = d.get("output") or []
        md_parts = []
        ci_parts = [] #new
        ci_idx = 0
        msg_idx = 0

        for item in out:
            if not isinstance(item, dict):
                continue

            itype = item.get("type")

            # --- Code Interpreter tool call ---
            if itype == "code_interpreter_call":
                ci_idx += 1
                ci_parts.append(f"## Code Interpreter call #{ci_idx}\n")

                if SHOW_CODE:
                    code = unroll_text(item.get("code", ""))
                    if code.strip():
                        ci_parts.append("### Code\n")
                        ci_parts.append(f"```python\n{code}\n```\n")

                if SHOW_LOGS:
                    outputs = item.get("outputs") or []
                    saw_logs = False
                    for o in outputs:
                        if not isinstance(o, dict):
                            continue
                        if o.get("type") == "logs":
                            saw_logs = True
                            logs = unroll_text(o.get("logs", ""))
                            ci_parts.append("### Logs\n")
                            ci_parts.append(f"```text\n{logs}\n```\n")

                    if not outputs:
                        ci_parts.append(
                            "_No tool outputs attached (check you used "
                            "`include=['code_interpreter_call.outputs']`)._\n"
                        )
                    elif outputs and not saw_logs:
                        ci_parts.append("_Tool outputs present, but no `logs` entries found._\n")

                continue

            # --- Assistant message text ---
            if itype == "message" and item.get("role") == "assistant":
                msg_idx += 1
                content = item.get("content") or []
                text_pieces = []

                for c in content:
                    if not isinstance(c, dict):
                        continue
                    ctype = c.get("type")
                    if ctype in ("output_text", "summary_text", "refusal"):
                        t = c.get("text")
                        if isinstance(t, str) and t.strip():
                            text_pieces.append(unroll_text(t).strip())

                if text_pieces:
                    md_parts.append(f"## Assistant message #{msg_idx}\n")
                    md_parts.append("\n\n".join(text_pieces) + "\n")

                continue
            
        md_parts.extend(ci_parts)

        # If we built anything, return it; otherwise fall back
        if md_parts:
            return "\n".join(md_parts).strip()

        # Fallbacks (rare)
        if d.get("output_text"):
            return unroll_text(str(d["output_text"])).strip()

        return unroll_text(str(d)).strip()

    except Exception:
        try:
            return unroll_text(repr(resp)).strip()
        except Exception:
            return ""
            
# END NEW BLOCK (3/14)

def is_transient_exception(e: Exception) -> bool:
    """
    Very simple heuristic: treat timeouts and common transient HTTP issues as retryable.
    """
    s = repr(e).lower()
    if any(k in s for k in ("timeout", "timed out", "502", "503", "504", "rate", "connection")):
        return True
    return False


# ---------------- Worker (top-level, picklable) ----------------
def one_solve_process(
    index: int,
    file_id: str,
    model: str,
    instruction_text: str,
    problem_dir: str,
    stamp: str,
    timeout_seconds: int,
    max_output_tokens: int,
    reasoning_effort_hint: Optional[str],
    temperature: Optional[float],
    attempt_max_retries: int = 2,
) -> Tuple[int, Optional[str], dict]:
    """
    Worker that performs a single Solve call.

    Returns (index, text_or_None, meta_dict)
    where meta_dict always contains at least:
        tag, index, stamp, model_used, status, duration_s, usage, resp/err
    """
    prob_dir = Path(problem_dir)
    call_tag = uuid.uuid4().hex[:10]
    start_time = time.time()

    # Build canonical input blocks (Responses schema)
    input_blocks = []
    if reasoning_effort_hint:
        input_blocks.append(
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"Reasoning effort: {reasoning_effort_hint}",
                    }
                ],
            }
        )
    
    input_blocks = []

# next two blocks modified 3/14 to make instructions more authoritative (especially re latex formatting for html)
# System instructions (text only)
    input_blocks.append(
        {
            "role": "system",  # keep instructions authoritative
            "content": [
                {"type": "input_text", "text": instruction_text},
            ],
        }
    )

# 3/26
# Optional per-problem hints (PROB/PROBhints.txt)
    hints_text = ""
    hints_path = prob_dir / "PROBhints.txt"
    if hints_path.exists():
        try:
            hints_text = hints_path.read_text(encoding="utf-8").strip()
        except Exception:
            hints_text = ""
# 3/26

# User provides the actual problem via file
 #   input_blocks.append(
 #       {
 #           "role": "user",
 #           "content": [
 #               {"type": "input_file", "file_id": file_id},
 #               # optional, but often helpful:
 #               {"type": "input_text", "text": "Use the attached file as the problem statement."},
 #           ],
 #       }
 #   )
 
 #3/26
 # User provides the actual problem via file (+ optional PROB/PROBhints.txt)
    hints_text = ""
    hints_path = prob_dir / "PROBhints.txt"
    if hints_path.exists():
        try:
            hints_text = hints_path.read_text(encoding="utf-8").strip()
        except Exception:
            hints_text = ""

    user_content = [
        {"type": "input_file", "file_id": file_id},
        {"type": "input_text", "text": "Use the attached file as the problem statement."},
    ]

    if hints_text:
        user_content.append(
            {
                "type": "input_text",
                "text": (
                    "User-provided hints (treat as additional given information/assumptions; "
                    "flag conflicts with the PDF):\n\n" + hints_text
                ),
            }
        )

    input_blocks.append(
        {
            "role": "user",
            "content": user_content,
        }
    )
    #3/26


    # Diagnostics start
    write_json_safe(
        prob_dir / f"worker_start_{call_tag}.json",
        {
            "tag": call_tag,
            "index": index,
            "stamp": stamp,
            "model_requested": model,
            "file_id": file_id,
            "timeout_s": timeout_seconds,
            "max_output_tokens": max_output_tokens,
            "start_time": start_time,
        },
    )

    client = OpenAI()
    attempt = 0
    last_exc = None
    while attempt <= attempt_max_retries:
        attempt += 1
        try:
            resp = client.responses.create(
                model=model,
                input=input_blocks,
                tools=[
                    {"type": "code_interpreter", "container": {"type": "auto"}},
                    # {"type": "web_search"},  # add this
                ],
                include=["code_interpreter_call.outputs"],
                timeout=timeout_seconds,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
            )
            duration = time.time() - start_time
            try:
                resp_d = resp.to_dict() if hasattr(resp, "to_dict") else resp
            except Exception:
                resp_d = repr(resp)
            ok = {
                "tag": call_tag,
                "index": index,
                "stamp": stamp,
                "model_used": getattr(resp, "model", None),
                "status": getattr(resp, "status", None),
                "duration_s": duration,
                "usage": getattr(resp, "usage", None),
                "resp": resp_d,
            }
            write_json_safe(prob_dir / f"worker_ok_{call_tag}.json", ok)
            text = extract_text_from_resp(resp)
            return index, text, ok
        except Exception as e:
            last_exc = e
            tb = traceback.format_exc()
            err = {
                "tag": call_tag,
                "index": index,
                "stamp": stamp,
                "attempt": attempt,
                "exception": repr(e),
                "traceback": tb,
                "time": time.time(),
            }
            # capture HTTP-level details if present
            http_resp = getattr(e, "response", None)
            try:
                if http_resp is not None:
                    headers = dict(getattr(http_resp, "headers", {}) or {})
                    body = getattr(http_resp, "text", None) or getattr(
                        http_resp, "body", None
                    )
                    err["http_headers"] = headers
                    err["http_body_truncated"] = str(body)[:2000] if body else None
            except Exception:
                pass
            write_json_safe(prob_dir / f"worker_err_{call_tag}_attempt{attempt}.json", err)
            if is_transient_exception(e) and attempt <= attempt_max_retries:
                time.sleep(2 ** (attempt - 1))
                continue
            # Give up
            return index, None, err
    return index, None, {
        "tag": call_tag,
        "index": index,
        "stamp": stamp,
        "error": "max_retries_exceeded",
        "last_exception": repr(last_exc),
    }


# ---------------- Config / PROB helpers ----------------
def load_config(cfg_path: Path) -> dict:
    import yaml

    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_prob_dir(path_in: str) -> Path:
    """
    Resolve a user-provided path to the PROB folder.
    Accepts:
      - path already pointing at PROB
      - problem folder that contains PROB
      - any descendant of a problem folder; we climb until we find PROB
    """
    p = Path(path_in).expanduser().resolve()
    # Case 1: already PROB
    if p.is_dir() and p.name == "PROB":
        return p
    # Case 2: problem folder containing PROB
    if (p / "PROB").is_dir():
        return p / "PROB"
    # Case 3: climb upwards
    cur = p
    while True:
        cand = cur / "PROB"
        if cand.is_dir():
            return cand
        if cur == cur.parent:
            break
        cur = cur.parent
    raise FileNotFoundError(f"Could not locate a PROB folder for the given path: {p}")


# ---------------- Chat-only continuation ----------------
def run_chat_only(args: argparse.Namespace) -> None:
    """
    Chat-only continuation mode.
    Expects PROB_DIR env var to be set by Run_Agency.sh and --resume-stamp provided.
    """
    # PROB_DIR is exported by Run_Agency.sh
    prob_env = os.environ.get("PROB_DIR")
    if not prob_env:
        print(
            "[chat-only] ERROR: PROB_DIR env var not set. Run via Run_Agency.sh.",
            file=sys.stderr,
        )
        sys.exit(1)
    prob_dir = Path(prob_env).expanduser().resolve()

    if not args.resume_stamp:
        print(
            "[chat-only] ERROR: --resume-stamp is required in chat-only mode.",
            file=sys.stderr,
        )
        sys.exit(2)
    stamp = args.resume_stamp

    # Locate transcript
    tx_path = prob_dir / f"transcript_{stamp}.txt"
    if not tx_path.exists():
        print(f"[chat-only] ERROR: transcript not found: {tx_path}", file=sys.stderr)
        sys.exit(3)

    # Show transcript to user
    print(f"[chat-only] Showing transcript from {tx_path}:\n")
    try:
        transcript_text = tx_path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"[chat-only] ERROR reading transcript: {e}", file=sys.stderr)
        return 4
    print(transcript_text)
    print("\n[chat-only] --- End of transcript ---\n")

    # Collect chat turns to append back into the transcript at the end
    chat_turns: List[Tuple[str, str]] = []
    

    # Load config and choose model for chat
    here = Path(__file__).resolve().parents[1]
    cfg_path = here / "config.yaml"
    if not cfg_path.exists():
        print(f"[chat-only] ERROR: config.yaml not found at {cfg_path}", file=sys.stderr)
        sys.exit(5)
    cfg = load_config(cfg_path)

    # Model + reasoning for Chat: use core_model / core_reasoning_effort if provided
    model_core = cfg.get("core_model") or cfg.get("model", "gpt-5")
    reasoning_effort = cfg.get("core_reasoning_effort", cfg.get("reasoning_effort"))
    timeout_seconds = int(cfg.get("timeout_seconds", 1800))
    max_output_tokens = int(
        cfg.get("output", {}).get("default_max_output_tokens", 3000)
    )
    temperature = cfg.get("temperature", None)

    # Optional chat instruction
    agents_dir = here / "Agents"
    chat_instr_path = agents_dir / "Instruction_Chat.txt"
    if chat_instr_path.exists():
        chat_instr = chat_instr_path.read_text(encoding="utf-8")
    else:
        chat_instr = (
            "You are an engineering analysis assistant. "
            "The user is asking follow-up questions about the previous transcript. "
            "Answer clearly and concisely, referring back to that work when helpful."
        )

    # Ensure API key
    api_env = (cfg.get("auth") or {}).get("api_key_env", "OPENAI_API_KEY")
    key = os.environ.get(api_env)
    if not key:
        print(f"[chat-only] ERROR: API key env var {api_env} not set.", file=sys.stderr)
        sys.exit(6)
    os.environ["OPENAI_API_KEY"] = key

    client = OpenAI()

    print("[chat-only] You can now ask questions about this run.")
    print("[chat-only] Press Enter on an empty line to exit.\n")

    while True:
        try:
            user_q = input("User> ").strip()
        except EOFError:
            break
        if not user_q:
            break

        # Build a fresh prompt each time: transcript + question
        input_blocks = []
        if reasoning_effort:
            input_blocks.append(
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": f"Reasoning effort: {reasoning_effort}",
                        }
                    ],
                }
            )
        input_blocks.append(
            {
                "role": "system",
                "content": [{"type": "input_text", "text": chat_instr}],
            }
        )
        input_blocks.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Here is the previous transcript from the Agency run:\n\n"
                            + transcript_text
                            + "\n\nNow answer this follow-up question from the user:\n\n"
                            + user_q
                        ),
                    }
                ],
            }
        )

        try:
            resp = client.responses.create(
                model=model_core,
                input=input_blocks,
                timeout=timeout_seconds,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
            )
            answer = extract_text_from_resp(resp)
        except Exception as e:
            print(f"\n[chat-only] ERROR from model: {e}\n", file=sys.stderr)
            continue

        print("\nAgent> " + (answer or "(no answer)"))
        print("\n---\n")
        
        chat_turns.append((user_q, answer or ""))
        
    # Append chat transcript back into transcript_<stamp>.txt
    if chat_turns:
        from datetime import datetime  # already imported at top, but safe if repeated in function scope
        try:
            with tx_path.open("a", encoding="utf-8") as f:
                f.write("\n=== Chat continuation ({}) ===\n\n".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ))
                for uq, ans in chat_turns:
                    f.write("User> " + uq + "\n\n")
                    f.write("Agent> " + (ans or "(no answer)") + "\n\n")
                    f.write("---\n\n")
            print(f"[chat-only] Appended {len(chat_turns)} Q&A turns to {tx_path}")
        except Exception as e:
            print(f"[chat-only] WARNING: could not append chat to transcript: {e}", file=sys.stderr)

    print("[chat-only] Chat session ended.")
    sys.exit(0)


# ---------------- HTML helper ----------------

def render_html_from_transcript(transcript_text: str, cfg: dict) -> str:
    """Render transcript text to HTML, with optional MathJax based on config."""
    output_cfg = cfg.get("output") or {}
    html_cfg = output_cfg.get("html") or {}
    mode = (html_cfg.get("mode") or "pre").lower()

    # Basic escaping of HTML special characters, but leave backslashes and $ alone.
    escaped = (
        transcript_text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

    if mode == "mathjax":
        # MathJax wrapper that renders LaTeX in $...$ or \(...\), etc.
        return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Agency transcript</title>
  <script>
    window.MathJax = {{
      tex: {{
        inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
        displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
      }},
      svg: {{ fontCache: 'global' }}
    }};
  </script>
  <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js" async></script>
  <style>
    body {{
      font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      margin: 1.5rem;
      line-height: 1.5;
    }}
    #transcript {{
      white-space: pre-wrap;
      font-family: Menlo, Consolas, 'Courier New', monospace;
    }}
  </style>
</head>
<body>
<div id="transcript">
{escaped}
</div>
</body>
</html>
"""
    else:
        # Simple <pre> wrapper with monospaced text
        return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Agency transcript</title>
  <style>
    body {{
      font-family: Menlo, Consolas, 'Courier New', monospace;
      margin: 1.5rem;
      white-space: pre-wrap;
    }}
  </style>
</head>
<body>
<pre>
{escaped}
</pre>
</body>
</html>
"""



# ---------------- Main ----------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agency client (Responses-based, with optional concurrency and chat)"
    )

    # Positional N (optional) and positional problem (optional) for backward compatibility
    parser.add_argument(
        "N",
        nargs="?",
        type=int,
        default=3,
        help="Number of Solve agents requested (N_requested)",
    )
    parser.add_argument(
        "problem",
        nargs="?",
        default=".",
        help="Path to the problem folder (or the PROB folder)",
    )

    # Optional explicit problem flag (alternative syntax)
    parser.add_argument(
        "--problem",
        dest="problem_flag",
        type=str,
        default=None,
        help="(Optional) Problem folder path (alternative syntax)",
    )

    parser.add_argument(
        "--stamp", type=str, default=None, help="Datestamp for this run"
    )
    parser.add_argument(
        "--nochat",
        action="store_true",
        help="Skip interactive chat at the end (for future use)",
    )
    parser.add_argument(
        "--chat-only",
        action="store_true",
        help="Chat-only resume mode (Continuation)",
    )
    parser.add_argument(
        "--resume-stamp",
        type=str,
        default=None,
        help="Datestamp to resume in chat-only mode",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model override for Solve (advanced)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout seconds override (advanced)",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help="Max output tokens override (advanced)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Dump extra debug info to stdout (advanced)",
    )

    args = parser.parse_args()

    # Chat-only mode: handle and exit early
    if args.chat_only:
        run_chat_only(args)
        return

    # Normal run (New mode from Run_Agency.sh)
    problem_arg = args.problem_flag if args.problem_flag else args.problem

    # Enforce nochat if not a TTY (background)
    if not sys.stdin.isatty():
        args.nochat = True

    # Locate config.yaml next to v1/Agency root (client sits in v1/Agency/client)
    here = Path(__file__).resolve().parents[1]
    cfg_path = here / "config.yaml"
    if not cfg_path.exists():
        print(f"Error: config.yaml not found at {cfg_path}", file=sys.stderr)
        sys.exit(2)
    cfg = load_config(cfg_path)

    # Models and reasoning efforts
    solve_model = cfg.get("solve_model") or cfg.get("model") or "gpt-5"
    core_model = cfg.get("core_model") or solve_model
    # Allow command-line override for Solve model
    if args.model:
        solve_model = args.model

    solve_reasoning_effort = cfg.get(
        "solve_reasoning_effort", cfg.get("reasoning_effort")
    )
    core_reasoning_effort = cfg.get(
        "core_reasoning_effort", cfg.get("reasoning_effort")
    )

    timeout_seconds = int(args.timeout or cfg.get("timeout_seconds", 1800))
    max_output_tokens = int(
        args.max_output_tokens
        or cfg.get("output", {}).get("default_max_output_tokens", 3000)
    )
    temperature = cfg.get("temperature", None)

    # Ensure API key available
    api_env = (cfg.get("auth") or {}).get("api_key_env", "OPENAI_API_KEY")
    if not os.getenv(api_env):
        print(
            f"Error: API key env var {api_env} not set. Export it before running.",
            file=sys.stderr,
        )
        sys.exit(3)
    os.environ["OPENAI_API_KEY"] = os.getenv(api_env)

    # Resolve PROB folder robustly
    # Prefer PROB_DIR env (set by Run_Agency.sh) if present; else use problem_arg
    prob_env = os.environ.get("PROB_DIR")
    try:
        if prob_env:
            prob_dir = Path(prob_env).expanduser().resolve()
        else:
            prob_dir = resolve_prob_dir(problem_arg)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(4)

    if not prob_dir.is_dir():
        print(f"Error: PROB dir does not exist: {prob_dir}", file=sys.stderr)
        sys.exit(4)
        
    # 4/1
    # ---------------- PROB-local override: GPT models ----------------
    # Optional file in PROB/ that can override solve_model and/or core_model per problem.
    override_path = prob_dir / "PROBgpt_models.yaml"
    if override_path.exists():
        try:
            import yaml
            ov = yaml.safe_load(override_path.read_text(encoding="utf-8")) or {}
            if isinstance(ov, dict):
                sm = ov.get("solve_model")
                cm = ov.get("core_model")

                if isinstance(sm, str) and sm.strip():
                    solve_model = sm.strip()
                    print(f"[info] PROB override solve_model -> {solve_model} (from {override_path.name})", flush=True)

                if isinstance(cm, str) and cm.strip():
                    core_model = cm.strip()
                    print(f"[info] PROB override core_model -> {core_model} (from {override_path.name})", flush=True)

        except Exception as e:
            print(f"[warn] Could not apply model overrides from {override_path}: {e}", flush=True)
    #
        
    

    print(
        f"[info] Transcript and HTML will be written into: {prob_dir} (client is responsible)",
        flush=True,
    )

    # Find PDF
    pdfs = sorted(prob_dir.glob("*.pdf"))
    if not pdfs:
        print(f"Error: no .pdf found in {prob_dir}", file=sys.stderr)
        sys.exit(5)
    if len(pdfs) > 1:
        print(
            f"[warn] multiple PDFs found in {prob_dir}, using first: {pdfs[0].name}",
            flush=True,
        )
    pdf_path = pdfs[0]

    # Stamp
    stamp = args.stamp or datestamp(
        cfg.get("output", {}).get("datestamp_format", "%Y%m%d-%H%M%S")
    )
    tx_name = f"transcript_{stamp}.txt"
    html_name = f"transcript_{stamp}.html"
    resp_name = f"resp_{stamp}.json"

    # Upload canonical (main process)
    print(f"[info] uploading {pdf_path} ...", flush=True)
    client_up = OpenAI()
    try:
        up = client_up.files.create(file=open(str(pdf_path), "rb"), purpose="user_data")
        file_id = getattr(up, "id", None) or (
            up.get("id") if isinstance(up, dict) else None
        )
        if not file_id:
            print(f"[error] upload returned no file id: {repr(up)}", file=sys.stderr)
            sys.exit(6)
        print(f"[upload] uploaded {pdf_path} -> file_id={file_id}", flush=True)
    except Exception as e:
        print(f"[error] upload failed: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(7)

    # Load Agent instruction texts
    agents_dir = here / "Agents"
    solve_instr_path = agents_dir / "Instruction_Solve.txt"
    compare_instr_path = agents_dir / "Instruction_Compare.txt"
    rec_instr_path = agents_dir / "Instruction_Recommend.txt"
    report_instr_path = agents_dir / "Instruction_Report.txt"

    if solve_instr_path.exists():
        solve_instr = solve_instr_path.read_text(encoding="utf-8")
    else:
        solve_instr = (
            "Solve the engineering analysis problem described in the attached PDF. "
            "Provide a correct final answer and a short derivation."
        )

    compare_instr = (
        compare_instr_path.read_text(encoding="utf-8")
        if compare_instr_path.exists()
        else "Compare and summarize the candidate solutions."
    )
    rec_instr = (
        rec_instr_path.read_text(encoding="utf-8")
        if rec_instr_path.exists()
        else "Recommend the best solution with justification."
    )
    report_instr = (
        report_instr_path.read_text(encoding="utf-8")
        if report_instr_path.exists()
        else None
    )

    instruction_text = solve_instr

    # Solve loop parameters
    N_requested = int(args.N)
    concurrency = int(cfg.get("parallel", {}).get("solves_concurrency", 0))

    sols: List[str] = []
    solve_metas: List[dict] = []
    errs: List[dict] = []

    if concurrency <= 0:
        print(
            f"[loop] Starting {N_requested} Solve agents sequentially (concurrency=0).",
            flush=True,
        )
        for idx in range(1, N_requested + 1):
            i, sol, meta = one_solve_process(
                index=idx,
                file_id=file_id,
                model=solve_model,
                instruction_text=instruction_text,
                problem_dir=str(prob_dir),
                stamp=stamp,
                timeout_seconds=timeout_seconds,
                max_output_tokens=max_output_tokens,
                reasoning_effort_hint=solve_reasoning_effort,
                temperature=temperature,
            )
            if sol:
                sols.append(sol)
                solve_metas.append(meta)
                print(f"[solve-{idx}] completed", flush=True)
            else:
                errs.append(meta)
                print(f"[solve-{idx}] error: {meta}", flush=True)
    else:
        print(
            f"[loop] Starting {N_requested} Solve agents with ProcessPool concurrency={concurrency}.",
            flush=True,
        )
        results: List[Tuple[int, Optional[str], dict]] = []
        with ProcessPoolExecutor(max_workers=concurrency) as ex:
            futs = [
                ex.submit(
                    one_solve_process,
                    idx,
                    file_id,
                    solve_model,
                    instruction_text,
                    str(prob_dir),
                    stamp,
                    timeout_seconds,
                    max_output_tokens,
                    solve_reasoning_effort,
                    temperature,
                )
                for idx in range(1, N_requested + 1)
            ]
            for fut in as_completed(futs):
                try:
                    res = fut.result()
                    results.append(res)
                except Exception as e:
                    tb = traceback.format_exc()
                    results.append(
                        (
                            -1,
                            None,
                            {"exception": repr(e), "traceback": tb, "stamp": stamp},
                        )
                    )

        for (i, sol, meta) in results:
            if sol:
                sols.append(sol)
                solve_metas.append(meta)
                print(f"[solve-{i}] completed", flush=True)
            else:
                errs.append(meta)
                print(f"[solve-{i}] error: {meta}", flush=True)

    # ---------------- Comparison, Recommendation, Report ----------------
    comparison_text: Optional[str] = None
    recommendation_text: Optional[str] = None
    report_text: Optional[str] = None

    compare_meta: Optional[dict] = None
    recommend_meta: Optional[dict] = None
    report_meta: Optional[dict] = None

    nonempty_solutions = [s for s in sols if s]

    if len(nonempty_solutions) >= 2:
        combined = "\n\n---\n\n".join(
            [f"=== Solution {i+1} ===\n{s}" for i, s in enumerate(nonempty_solutions)]
        )

        # Compare
        compare_blocks = []
        if core_reasoning_effort:
            compare_blocks.append(
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": f"Reasoning effort: {core_reasoning_effort}",
                        }
                    ],
                }
            )
        compare_blocks.append(
            {
                "role": "system",
                "content": [{"type": "input_text", "text": compare_instr}],
            }
        )
        compare_blocks.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Compare and assess these candidate solutions:\n\n"
                        + combined,
                    }
                ],
            }
        )
        try:
            t0 = time.time()
            resp_cmp = OpenAI().responses.create(
                model=core_model,
                input=compare_blocks,
                timeout=timeout_seconds,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
            )
            comparison_text = extract_text_from_resp(resp_cmp)
            compare_meta = {
                "stage": "compare",
                "stamp": stamp,
                "requested_model": core_model,
                "model_used": getattr(resp_cmp, "model", None),
                "status": getattr(resp_cmp, "status", None),
                "duration_s": time.time() - t0,
                "usage": getattr(resp_cmp, "usage", None),
            }
        except Exception as e:
            comparison_text = f"(Comparison failed: {e})"
            compare_meta = {
                "stage": "compare",
                "stamp": stamp,
                "error": repr(e),
            }

        # Recommend
        rec_blocks = []
        if core_reasoning_effort:
            rec_blocks.append(
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": f"Reasoning effort: {core_reasoning_effort}",
                        }
                    ],
                }
            )
        rec_blocks.append(
            {
                "role": "system",
                "content": [{"type": "input_text", "text": rec_instr}],
            }
        )
        rec_blocks.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Using the comparison and assessment of the candidate solutions, "
                            "develop a recommended solution with justification.\n\n"
                            "Solutions:\n\n"
                            + combined
                            + "\n\nComparison (if any):\n\n"
                            + (comparison_text or "(comparison unavailable)")
                        ),
                    }
                ],
            }
        )
        try:
            t0 = time.time()
            resp_rec = OpenAI().responses.create(
                model=core_model,
                input=rec_blocks,
                timeout=timeout_seconds,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
            )
            recommendation_text = extract_text_from_resp(resp_rec)
            recommend_meta = {
                "stage": "recommend",
                "stamp": stamp,
                "requested_model": core_model,
                "model_used": getattr(resp_rec, "model", None),
                "status": getattr(resp_rec, "status", None),
                "duration_s": time.time() - t0,
                "usage": getattr(resp_rec, "usage", None),
            }
        except Exception as e:
            recommendation_text = f"(Recommendation failed: {e})"
            recommend_meta = {
                "stage": "recommend",
                "stamp": stamp,
                "error": repr(e),
            }

    # Report (human-facing)
    if report_instr:
        try:
            t0 = time.time()
            report_blocks = []
            if core_reasoning_effort:
                report_blocks.append(
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "input_text",
                                "text": f"Reasoning effort: {core_reasoning_effort}",
                            }
                        ],
                    }
                )
            report_blocks.append(
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": report_instr}],
                }
            )
            
                        # Only pass the recommendation (and comparison, if present) as context.
            body_parts: List[str] = []
            if recommendation_text:
                body_parts.append("=== Recommendation ===\n" + recommendation_text + "\n")
            if comparison_text:
                body_parts.append(
                    "=== Comparison (for context) ===\n" + comparison_text + "\n"
                )
            body = "\n".join(body_parts)

            report_blocks.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "Using the context below, write a clear, human-facing "
                                "engineering report that explains the recommended "
                                "solution, its reasoning, and the main verification/"
                                "validation checks.\n\n"
                                "Do NOT reproduce the full step-by-step solutions or "
                                "copy the comparison verbatim; the reader already has "
                                "those earlier in the transcript. Summarize them in "
                                "your own words instead.\n\n"
                                "Context:\n\n" + body
                            ),
                        }
                    ],
                }
            )           
            
            
            resp_rep = OpenAI().responses.create(
                model=core_model,
                input=report_blocks,
                timeout=timeout_seconds,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
            )
            report_text = extract_text_from_resp(resp_rep)
            report_meta = {
                "stage": "report",
                "stamp": stamp,
                "requested_model": core_model,
                "model_used": getattr(resp_rep, "model", None),
                "status": getattr(resp_rep, "status", None),
                "duration_s": time.time() - t0,
                "usage": getattr(resp_rep, "usage", None),
            }
        except Exception as e:
            report_text = f"(Report failed: {e})"
            report_meta = {
                "stage": "report",
                "stamp": stamp,
                "error": repr(e),
            }
    else:
        report_text = "(No Report instruction found.)"

    # ---------------- Build transcript text ----------------
    parts: List[str] = []

    if sols:
        for idx, s in enumerate(sols, start=1):
            parts.append(f"=== Solution {idx} ===\n{s}\n\n")
    else:
        parts.append("=== Solutions ===\n(no successful solutions)\n\n")

    if comparison_text is not None:
        parts.append("=== Comparison and Assessment ===\n" + (comparison_text or "(none)") + "\n\n")

    if recommendation_text is not None:
        parts.append("=== Recommendation ===\n" + (recommendation_text or "(none)") + "\n\n")

    if report_text is not None:
        parts.append("=== Report ===\n" + (report_text or "(none)") + "\n\n")

    # Model / usage summary
    models_start_idx = len(parts)
    parts.append("=== Models and Usage ===\n")
    parts.append(f"Solve model requested: {solve_model}\n")
    for idx, meta in enumerate(solve_metas, start=1):
        parts.append(
            f"  Solution {idx}: used={meta.get('model_used')} "
            f"status={meta.get('status')} "
            f"duration_s={meta.get('duration_s')}\n"
        )
    if compare_meta:
        parts.append(
            f"Compare: requested={compare_meta.get('requested_model')} "
            f"used={compare_meta.get('model_used')} "
            f"status={compare_meta.get('status')} "
            f"duration_s={compare_meta.get('duration_s')}\n"
        )
    if recommend_meta:
        parts.append(
            f"Recommend: requested={recommend_meta.get('requested_model')} "
            f"used={recommend_meta.get('model_used')} "
            f"status={recommend_meta.get('status')} "
            f"duration_s={recommend_meta.get('duration_s')}\n"
        )
    if report_meta:
        parts.append(
            f"Report: requested={report_meta.get('requested_model')} "
            f"used={report_meta.get('model_used')} "
            f"status={report_meta.get('status')} "
            f"duration_s={report_meta.get('duration_s')}\n"
        )

    # Also append raw error metadata (if any)
    if errs:
        parts.append("\n=== Errors ===\n")
        for e_meta in errs:
            try:
                parts.append(json.dumps(e_meta, indent=2, default=str) + "\n")
            except Exception:
                parts.append(repr(e_meta) + "\n")
    
    model_util_text = "".join(parts[models_start_idx:])
    transcript_text = "".join(parts)

    tx_path = prob_dir / tx_name
    tx_path.write_text(transcript_text, encoding="utf-8")
    print(f"✅ wrote transcript {tx_path}", flush=True)

    # ---------------- Aggregate worker_ok_*.json into resp_<stamp>.json ----------------
    ok_files = sorted(prob_dir.glob("worker_ok_*.json"))
    if ok_files:
        agg = []
        for f in ok_files:
            try:
                agg.append(json.loads(f.read_text(encoding="utf-8")))
            except Exception:
                pass
        if agg:
            resp_agg_path = prob_dir / resp_name
            write_json_safe(resp_agg_path, agg)
            print(f"✅ wrote aggregated resp summary {resp_agg_path}", flush=True)

    # ---------------- HTML ----------------
        # HTML
    #html_path = prob_dir / html_name
    #html_text = render_html_from_transcript(transcript_text, cfg)
    #html_path.write_text(html_text, encoding="utf-8")
    #print(f"✅ wrote HTML {html_path}", flush=True)

    # ---------------- HTML (button sections; keeps your helper unchanged) ----------------
    output_cfg = cfg.get("output") or {}
    html_cfg = output_cfg.get("html") or {}
    mode = (html_cfg.get("mode") or "pre").lower()

    import html as _html
    import re as _re

    html_path = prob_dir / html_name

    def _sec_id(title: str) -> str:
        return "sec_" + _re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")

    def _panel(title: str, body: str) -> str:
        sid = _sec_id(title)
        return f"""
        <button type="button" class="btn" onclick="toggle('{sid}')">{_html.escape(title)}</button>
        <div id="{sid}" class="panel" style="display:none;">
          <h2>{_html.escape(title)}</h2>
          <div class="content">{_html.escape(body or "")}</div>
        </div>
        """

    panels = []

    # Solutions (one button per solution)
    if sols:
        for idx, s in enumerate(sols, start=1):
            panels.append(_panel(f"Solution {idx}", s or "(none)"))
    else:
        panels.append(_panel("Solutions", "(no successful solutions)"))

    if comparison_text is not None:
        panels.append(_panel("Comparison and Assessment", comparison_text or "(none)"))

    if recommendation_text is not None:
        panels.append(_panel("Recommendation", recommendation_text or "(none)"))

    if report_text is not None:
        panels.append(_panel("Report", report_text or "(none)"))
        
    panels.append(_panel("GPT Model Utilization", model_util_text or "(none)"))

    mathjax_head = ""
    typeset_on_open_js = ""
    if mode == "mathjax":
        mathjax_head = """
      <script>
        window.MathJax = {
          tex: {
           inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
            displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
          },
          svg: { fontCache: 'global' }
        };
      </script>
      <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js" async></script>
    """
        typeset_on_open_js = """
        if (opening && window.MathJax && MathJax.typesetPromise) {
          MathJax.typesetPromise([el]);
        }
    """
    # 4/1
    pdf_button_html = f"""<button type="button" class="btn"
      onclick="window.open('{_html.escape(pdf_path.name)}','_blank')">Problem Statement (PDF)</button>"""
      
    html_text = f"""<!doctype html>
    <html>
    <head>
      <meta charset="utf-8" />
      <title>Transcript</title>
    {mathjax_head}
      <style>
        body {{ font-family: Arial, sans-serif; margin: 16px; }}
        .btn {{ margin: 6px 8px 6px 0; padding: 8px 10px; cursor: pointer; }}
        .panel {{ border: 1px solid #ddd; padding: 12px; margin: 10px 0 18px 0; }}
        .content {{
          white-space: pre-wrap;
          margin: 0;
          line-height: 1.5; #3/30
          font-family: Menlo, Consolas, 'Courier New', monospace;
        }}
      </style>
      <script>
        function toggle(id) {{
          const el = document.getElementById(id);
          if (!el) return;
          const opening = (el.style.display === "none" || el.style.display === "");
          el.style.display = opening ? "block" : "none";
    {typeset_on_open_js}
        }}
      </script>
    </head>
    <body>
      {pdf_button_html}
      {"".join(panels)}
    </body>
    </html>
    """

    html_path.write_text(html_text, encoding="utf-8")
    print(f"✅ wrote HTML {html_path}", flush=True)


    # ---------------- Student bundle ZIP ----------------
    import zipfile

    problem_name = prob_dir.parent.name
    bundle_path = prob_dir / f"Bundle_{problem_name}_{stamp}.zip"
    # bundle_path = prob_dir / f"Bundle_{stamp}.zip"

    bundle_candidates = [
        (html_path, html_path.name),
        (pdf_path, pdf_path.name),  # typically ProblemStatement.pdf
        (prob_dir / "PROBhints.txt", "PROBhints.txt"),
        (prob_dir / "PROBgpt_models.yaml", "PROBgpt_models.yaml"),
    ]

    try:
        with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for p, arc in bundle_candidates:
                if p and Path(p).exists():
                    z.write(str(p), arcname=arc)
        print(f"✅ wrote student bundle {bundle_path}", flush=True)
    except Exception as e:
        print(f"[warn] could not write bundle zip {bundle_path}: {e}", flush=True)

    print("[done] Agency client run complete.", flush=True)


if __name__ == "__main__":
    main()
