from __future__ import annotations
import html as _html
import os
import re
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Optional
import yaml
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"

# Prefer the authoritative extraction routine from the core client so
# continuation output is treated the same way as initial Agency output,
# including code_interpreter_call source and logs when present.
try:
    sys.path.insert(0, str(CORE / "client"))
    from agency_client import extract_text_from_resp as _core_extract_text_from_resp  # type: ignore
except Exception:  # keep a local fallback below
    _core_extract_text_from_resp = None

def _unroll_text(s: str) -> str:
    """Normalize tool code/log strings for readable transcripts."""
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
    """
    Match the core client behavior for Responses API output extraction.

    In particular, include code_interpreter_call source code and logs when
    the response includes code_interpreter_call.outputs.
    """
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

def _load_cfg() -> dict:
    cfg_path = CORE / "config.yaml"
    if not cfg_path.exists():
        return {}
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

def _append_panel_to_html(html_path: Path, question: str, answer: str) -> None:
    when = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    panel_id = "sec_followup_" + re.sub(r"[^a-z0-9]+", "_", when.lower()).strip("_")
    body = f"Time: {when}\n\nUser> {question}\n\nAgent> {answer}"
    panel = f"""
      <button type="button" class="btn" onclick="toggle('{panel_id}')">Follow-up: {_html.escape(question[:60])}</button>
      <div id="{panel_id}" class="panel" style="display:none;">
        <h2>Follow-up continuation</h2>
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

def run_continuation(prob_dir: Path, stamp: str, question: str) -> tuple[str, Path, Path]:
    cfg = _load_cfg()
    tx_path = prob_dir / f"transcript_{stamp}.txt"
    html_path = prob_dir / f"transcript_{stamp}.html"
    if not tx_path.exists():
        raise FileNotFoundError(f"Transcript not found: {tx_path}")
    transcript_text = tx_path.read_text(encoding="utf-8")

    api_env = (cfg.get("auth") or {}).get("api_key_env", "OPENAI_API_KEY")
    if not os.environ.get(api_env):
        raise RuntimeError(f"API key environment variable {api_env} is not set")

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
    input_blocks.append({"role": "user", "content": [{"type": "input_text", "text": "Here is the previous transcript from the Agency run:\n\n" + transcript_text + "\n\nNow answer this follow-up question from the user:\n\n" + question}]})

    resp = OpenAI().responses.create(
        model=model_core,
        input=input_blocks,
        tools=[{"type": "code_interpreter", "container": {"type": "auto"}}],
        include=["code_interpreter_call.outputs"],
        timeout=timeout_seconds,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
    )

    # Save raw continuation response next to the transcript. This is useful for
    # checking whether a code_interpreter_call was actually returned by the API.
    raw_path = prob_dir / f"continuation_resp_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    try:
        d = _resp_to_dict(resp)
        raw_path.write_text(json.dumps(d if d is not None else str(resp), default=str, indent=2), encoding="utf-8")
    except Exception:
        pass

    if _core_extract_text_from_resp is not None:
        answer = _core_extract_text_from_resp(resp) or "(no answer)"
    else:
        answer = _extract_text(resp) or "(no answer)"

    when = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with tx_path.open("a", encoding="utf-8") as f:
        f.write(f"\n=== Web continuation ({when}) ===\n\n")
        f.write("User> " + question + "\n\n")
        f.write("Agent> " + answer + "\n\n---\n")
    _append_panel_to_html(html_path, question, answer)
    return answer, tx_path, html_path
