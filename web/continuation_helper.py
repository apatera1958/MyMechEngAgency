from __future__ import annotations
import html as _html
import os
import re
import sys
import json
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional
import yaml
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"

# Import only utility behavior from the unmodified core client.
try:
    sys.path.insert(0, str(CORE / "client"))
    from agency_client import extract_text_from_resp as _core_extract_text_from_resp  # type: ignore
except Exception:
    _core_extract_text_from_resp = None


def _unroll_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).replace("\r\n", "\n").replace("\r", "\n")
    if "\\n" in s and "\n" not in s:
        s = s.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")
    return s


def _resp_to_dict(resp):
    if isinstance(resp, dict):
        return resp
    for meth in ("to_dict", "model_dump", "dict"):
        if hasattr(resp, meth):
            try:
                return getattr(resp, meth)()
            except Exception:
                pass
    return None


def _extract_text(resp) -> str:
    """Fallback parser matching the core client's text/code/log extraction."""
    d = _resp_to_dict(resp)
    if d is None:
        return _unroll_text(str(resp)).strip()

    out = d.get("output") or []
    md_parts = []
    ci_parts = []
    ci_idx = 0
    msg_idx = 0

    for item in out:
        if not isinstance(item, dict):
            continue
        itype = item.get("type")

        if itype == "code_interpreter_call":
            ci_idx += 1
            ci_parts.append(f"## Code Interpreter call #{ci_idx}\n")

            code = _unroll_text(item.get("code", ""))
            if code.strip():
                ci_parts.append("### Code\n")
                ci_parts.append(f"```python\n{code}\n```\n")

            outputs = item.get("outputs") or []
            saw_logs = False
            for o in outputs:
                if not isinstance(o, dict):
                    continue
                if o.get("type") == "logs":
                    saw_logs = True
                    logs = _unroll_text(o.get("logs", ""))
                    ci_parts.append("### Logs\n")
                    ci_parts.append(f"```text\n{logs}\n```\n")

            if not outputs:
                ci_parts.append("_No tool outputs attached._\n")
            elif outputs and not saw_logs:
                ci_parts.append("_Tool outputs present, but no `logs` entries found._\n")
            continue

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
                        text_pieces.append(_unroll_text(t).strip())
            if text_pieces:
                md_parts.append(f"## Assistant message #{msg_idx}\n")
                md_parts.append("\n\n".join(text_pieces) + "\n")
            continue

    md_parts.extend(ci_parts)
    if md_parts:
        return "\n".join(md_parts).strip()
    if d.get("output_text"):
        return _unroll_text(str(d["output_text"])).strip()
    return _unroll_text(str(d)).strip()



def _iter_dicts(obj):
    """Yield every dict in a nested Responses API object."""
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _iter_dicts(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _iter_dicts(v)


def _extract_code_from_ci_item(item: dict) -> str:
    """Extract code source from several observed Responses API shapes."""
    candidates = []
    for key in ("code", "input", "source"):
        v = item.get(key)
        if isinstance(v, str):
            candidates.append(v)
    action = item.get("action")
    if isinstance(action, dict):
        for key in ("code", "input", "source"):
            v = action.get(key)
            if isinstance(v, str):
                candidates.append(v)
    for c in candidates:
        c = _unroll_text(c)
        if c.strip():
            return c
    return ""


def _extract_logs_from_outputs(outputs) -> list[str]:
    logs: list[str] = []
    if not isinstance(outputs, list):
        return logs
    for o in outputs:
        if not isinstance(o, dict):
            continue
        typ = o.get("type")
        if typ == "logs" and isinstance(o.get("logs"), str):
            logs.append(_unroll_text(o.get("logs", "")))
        # Some SDK/API variants put text-ish outputs in other fields.
        elif typ in {"output_text", "text"} and isinstance(o.get("text"), str):
            logs.append(_unroll_text(o.get("text", "")))
        elif isinstance(o.get("content"), str):
            logs.append(_unroll_text(o.get("content", "")))
    return [x for x in logs if x.strip()]


def _extract_code_interpreter_sections(resp) -> str:
    """
    Robustly extract code-interpreter code/log blocks, including nested variants.

    The core client handles the common top-level shape. This catches additional
    Responses API shapes seen in web continuation runs without changing core.
    """
    d = _resp_to_dict(resp)
    if d is None:
        return ""
    parts: list[str] = []
    seen: set[str] = set()
    idx = 0
    for item in _iter_dicts(d):
        if item.get("type") != "code_interpreter_call":
            continue
        code = _extract_code_from_ci_item(item)
        logs = _extract_logs_from_outputs(item.get("outputs") or [])
        if not code and not logs:
            continue
        sig = json.dumps({"code": code, "logs": logs}, sort_keys=True, default=str)
        if sig in seen:
            continue
        seen.add(sig)
        idx += 1
        parts.append(f"## Code Interpreter call #{idx}\n")
        if code.strip():
            parts.append("### Code\n")
            parts.append(f"```python\n{code}\n```\n")
        if logs:
            for log_no, logtxt in enumerate(logs, start=1):
                label = "### Logs" if len(logs) == 1 else f"### Logs {log_no}"
                parts.append(label + "\n")
                parts.append(f"```text\n{logtxt}\n```\n")
        else:
            parts.append("_No log outputs attached._\n")
    return "\n".join(parts).strip()


def _merge_answer_with_ci(resp) -> str:
    """Return assistant text plus any missing code-interpreter sections."""
    if _core_extract_text_from_resp is not None:
        base = _core_extract_text_from_resp(resp) or ""
    else:
        base = _extract_text(resp) or ""
    ci = _extract_code_interpreter_sections(resp)
    if ci and "Code Interpreter call" not in base:
        return (base.rstrip() + "\n\n" + ci).strip()
    return base.strip() or ci.strip() or "(no answer)"

def _load_cfg() -> dict:
    cfg_path = CORE / "config.yaml"
    if not cfg_path.exists():
        return {}
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}


def _sec_id(title: str) -> str:
    return "sec_" + re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")


def _next_follow_on_number(html_path: Path) -> int:
    if not html_path.exists():
        return 1
    text = html_path.read_text(encoding="utf-8", errors="ignore")
    nums = [int(m.group(1)) for m in re.finditer(r">\s*Follow-On\s+(\d+)\s*<", text)]
    return (max(nums) + 1) if nums else 1


def _append_panel_to_html(html_path: Path, follow_no: int, question: str, answer: str) -> None:
    when = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    title = f"Follow-On {follow_no}"
    panel_id = f"sec_follow_on_{follow_no}"
    body = f"Time: {when}\n\nUser> {question}\n\nAgent> {answer}"
    panel = f"""
      <button type="button" class="btn" title="{_html.escape(question, quote=True)}" onclick="toggle('{panel_id}')">{_html.escape(title)}</button>
      <div id="{panel_id}" class="panel" style="display:none;">
        <h2 title="{_html.escape(question, quote=True)}">{_html.escape(title)}</h2>
        <div class="content">{_html.escape(body)}</div>
      </div>
"""
    if not html_path.exists():
        html_path.write_text(f"<html><body><pre>{_html.escape(body)}</pre></body></html>", encoding="utf-8")
        return
    text = html_path.read_text(encoding="utf-8")
    if "</body>" in text:
        text = text.replace("</body>", panel + "\n</body>")
    else:
        text += panel
    html_path.write_text(text, encoding="utf-8")


def _rebuild_bundle(prob_dir: Path, stamp: str, html_path: Path, tx_path: Path) -> Path:
    problem_name = prob_dir.parent.name
    bundle_path = prob_dir / f"Bundle_{problem_name}_{stamp}.zip"
    candidates = [
        (html_path, html_path.name),
        (tx_path, tx_path.name),
        (prob_dir / "ProblemStatement.pdf", "ProblemStatement.pdf"),
        (prob_dir / "PROBhints.txt", "PROBhints.txt"),
        (prob_dir / "PROBgpt_models.yaml", "PROBgpt_models.yaml"),
    ]
    # Include raw continuation response JSON files for diagnosis and provenance.
    for p in sorted(prob_dir.glob("continuation_resp_*.json")):
        candidates.append((p, p.name))
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p, arc in candidates:
            if p and Path(p).exists():
                z.write(str(p), arcname=arc)
    return bundle_path


def run_continuation(prob_dir: Path, stamp: str, question: str) -> tuple[str, Path, Path]:
    cfg = _load_cfg()
    tx_path = prob_dir / f"transcript_{stamp}.txt"
    html_path = prob_dir / f"transcript_{stamp}.html"
    if not tx_path.exists():
        raise FileNotFoundError(f"Transcript not found: {tx_path}")

    # The full augmented transcript so far, including earlier Follow-On sections.
    transcript_text = tx_path.read_text(encoding="utf-8")

    api_env = (cfg.get("auth") or {}).get("api_key_env", "OPENAI_API_KEY")
    if not os.environ.get(api_env):
        raise RuntimeError(f"API key environment variable {api_env} is not set")

    # Use the same chat-agent instruction source as the original client.
    chat_instr_path = CORE / "Agents" / "Instruction_Chat.txt"
    if chat_instr_path.exists():
        chat_instr = chat_instr_path.read_text(encoding="utf-8")
    else:
        chat_instr = "You are an engineering analysis assistant answering follow-up questions about a previous transcript."

    model_core = cfg.get("core_model") or cfg.get("model", "gpt-5")
    reasoning_effort = cfg.get("core_reasoning_effort", cfg.get("reasoning_effort"))
    timeout_seconds = int(cfg.get("timeout_seconds", 1800))
    max_output_tokens = int((cfg.get("output") or {}).get("default_max_output_tokens", 3000))
    temperature = cfg.get("temperature", None)

    input_blocks = []
    if reasoning_effort:
        input_blocks.append({"role": "system", "content": [{"type": "input_text", "text": f"Reasoning effort: {reasoning_effort}"}]})
    input_blocks.append({"role": "system", "content": [{"type": "input_text", "text": chat_instr}]})
    input_blocks.append({"role": "user", "content": [{"type": "input_text", "text": "Here is the previous transcript from the Agency run, including all follow-ons so far:\n\n" + transcript_text + "\n\nNow answer this new follow-on question from the user:\n\n" + question}]})

    resp = OpenAI().responses.create(
        model=model_core,
        input=input_blocks,
        tools=[{"type": "code_interpreter", "container": {"type": "auto"}}],
        include=["code_interpreter_call.outputs"],
        timeout=timeout_seconds,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
    )

    raw_path = prob_dir / f"continuation_resp_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    try:
        d = _resp_to_dict(resp)
        raw_path.write_text(json.dumps(d if d is not None else str(resp), default=str, indent=2), encoding="utf-8")
    except Exception:
        pass

    answer = _merge_answer_with_ci(resp)

    follow_no = _next_follow_on_number(html_path)
    when = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with tx_path.open("a", encoding="utf-8") as f:
        f.write(f"\n=== Follow-On {follow_no} ({when}) ===\n\n")
        f.write("User> " + question + "\n\n")
        f.write("Agent> " + answer + "\n\n---\n")

    _append_panel_to_html(html_path, follow_no, question, answer)
    _rebuild_bundle(prob_dir, stamp, html_path, tx_path)
    return answer, tx_path, html_path
