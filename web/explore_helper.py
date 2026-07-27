from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
NOTEBOOK_FILENAME = "Engineering_Exploration.ipynb"

# Reuse the core client's Responses API text extraction when available.
try:
    sys.path.insert(0, str(CORE / "client"))
    from agency_client import extract_text_from_resp as _core_extract_text_from_resp  # type: ignore
except Exception:
    _core_extract_text_from_resp = None


def _load_cfg() -> dict[str, Any]:
    cfg_path = CORE / "config.yaml"
    if not cfg_path.exists():
        return {}
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}


def _response_to_dict(resp: Any) -> Any:
    if isinstance(resp, dict):
        return resp
    for method_name in ("to_dict", "model_dump", "dict"):
        method = getattr(resp, method_name, None)
        if callable(method):
            try:
                return method()
            except Exception:
                pass
    return None


def _extract_response_text(resp: Any) -> str:
    if _core_extract_text_from_resp is not None:
        try:
            text = _core_extract_text_from_resp(resp)
            if text and str(text).strip():
                return str(text).strip()
        except Exception:
            pass

    data = _response_to_dict(resp)
    if isinstance(data, dict):
        output_text = data.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        pieces: list[str] = []
        for item in data.get("output") or []:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            for content in item.get("content") or []:
                if not isinstance(content, dict):
                    continue
                if content.get("type") in {"output_text", "summary_text"}:
                    text = content.get("text")
                    if isinstance(text, str) and text.strip():
                        pieces.append(text.strip())
        if pieces:
            return "\n\n".join(pieces)

    return str(resp).strip()


def _extract_report(transcript_text: str) -> str:
    """Extract the original Report section from a MyAgency text transcript."""
    match = re.search(
        r"(?ms)^===\s*Report\s*===\s*\n(.*?)(?=^===\s*[^\n=]+\s*===\s*$|\Z)",
        transcript_text,
    )
    if not match:
        raise ValueError("The transcript does not contain an '=== Report ===' section.")
    report = match.group(1).strip()
    if not report:
        raise ValueError("The transcript Report section is empty.")
    return report


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    return fenced.group(1).strip() if fenced else text


def _parse_notebook_json(text: str) -> dict[str, Any]:
    """Parse a model response that should contain one complete notebook object."""
    candidate = _strip_json_fence(text)
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        # Defensive fallback: locate the outermost JSON object if the model
        # accidentally added a short preface or suffix.
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Explore did not return a JSON object.")
        try:
            obj = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"Explore returned invalid notebook JSON: {exc}") from exc

    if not isinstance(obj, dict):
        raise ValueError("The generated notebook must be a JSON object.")
    return obj


def _normalize_source(source: Any) -> list[str]:
    if isinstance(source, str):
        return source.splitlines(keepends=True) or [source]
    if isinstance(source, list) and all(isinstance(line, str) for line in source):
        return source
    raise ValueError("Every notebook cell must have a string or list-of-strings 'source'.")


def _validate_and_normalize_notebook(notebook: dict[str, Any]) -> dict[str, Any]:
    """Validate the minimum nbformat contract and normalize cell fields."""
    cells = notebook.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("The generated notebook has no cells.")

    normalized_cells: list[dict[str, Any]] = []
    for index, cell in enumerate(cells, start=1):
        if not isinstance(cell, dict):
            raise ValueError(f"Notebook cell {index} is not an object.")
        cell_type = cell.get("cell_type")
        if cell_type not in {"markdown", "code", "raw"}:
            raise ValueError(f"Notebook cell {index} has unsupported cell_type={cell_type!r}.")

        normalized = dict(cell)
        normalized["metadata"] = normalized.get("metadata") if isinstance(normalized.get("metadata"), dict) else {}
        normalized["source"] = _normalize_source(normalized.get("source", []))

        if cell_type == "code":
            normalized["execution_count"] = None
            normalized["outputs"] = []
        else:
            normalized.pop("execution_count", None)
            normalized.pop("outputs", None)
        normalized_cells.append(normalized)

    metadata = notebook.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    metadata.setdefault(
        "kernelspec",
        {"display_name": "Python 3", "language": "python", "name": "python3"},
    )
    metadata.setdefault(
        "language_info",
        {"name": "python", "version": "3"},
    )

    return {
        "cells": normalized_cells,
        "metadata": metadata,
        "nbformat": 4,
        "nbformat_minor": int(notebook.get("nbformat_minor", 5)),
    }


def run_explore(prob_dir: Path, stamp: str) -> Path:
    """Generate or regenerate Engineering_Exploration.ipynb for a completed job."""
    cfg = _load_cfg()
    transcript_path = prob_dir / f"transcript_{stamp}.txt"
    if not transcript_path.exists():
        raise FileNotFoundError(f"Transcript not found: {transcript_path}")

    instruction_path = CORE / "Agents" / "Instruction_Explore.txt"
    if not instruction_path.exists():
        raise FileNotFoundError(f"Explore instruction file not found: {instruction_path}")

    transcript_text = transcript_path.read_text(encoding="utf-8")
    report_text = _extract_report(transcript_text)
    explore_instruction = instruction_path.read_text(encoding="utf-8").strip()
    if not explore_instruction:
        raise ValueError("Instruction_Explore.txt is empty.")

    api_key_env = (cfg.get("auth") or {}).get("api_key_env", "OPENAI_API_KEY")
    if not os.environ.get(api_key_env):
        raise RuntimeError(f"API key environment variable {api_key_env} is not set")

    model = cfg.get("explore_model") or cfg.get("core_model") or cfg.get("model", "gpt-5")
    reasoning_effort = cfg.get("explore_reasoning_effort", cfg.get("core_reasoning_effort", cfg.get("reasoning_effort")))
    timeout_seconds = int(cfg.get("timeout_seconds", 1800))
    max_output_tokens = int(
        cfg.get("explore_max_output_tokens")
        or (cfg.get("output") or {}).get("explore_max_output_tokens")
        or 12000
    )
    temperature = cfg.get("temperature", None)

    system_blocks: list[dict[str, Any]] = []
    if reasoning_effort:
        system_blocks.append(
            {
                "role": "system",
                "content": [{"type": "input_text", "text": f"Reasoning effort: {reasoning_effort}"}],
            }
        )
    system_blocks.append(
        {
            "role": "system",
            "content": [{"type": "input_text", "text": explore_instruction}],
        }
    )

    user_text = (
        "Create the complete Jupyter Notebook requested by the Explore instructions. "
        "Return only the notebook JSON object: no Markdown fence and no surrounding commentary.\n\n"
        "PRIMARY SOURCE — CURRENT REPORT\n"
        "================================\n"
        f"{report_text}\n\n"
        "ADDITIONAL CONTEXT — COMPLETE CURRENT TRANSCRIPT\n"
        "================================================\n"
        "Use this only when it supplies relevant details omitted from the Report. "
        "It includes any Follow-On questions added since the original run.\n\n"
        f"{transcript_text}"
    )
    input_blocks = system_blocks + [
        {"role": "user", "content": [{"type": "input_text", "text": user_text}]}
    ]

    response = OpenAI().responses.create(
        model=model,
        input=input_blocks,
        timeout=timeout_seconds,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
    )

    raw_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    raw_path = prob_dir / f"explore_resp_{raw_stamp}.json"
    try:
        raw = _response_to_dict(response)
        raw_path.write_text(
            json.dumps(raw if raw is not None else str(response), default=str, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass

    response_text = _extract_response_text(response)
    notebook = _validate_and_normalize_notebook(_parse_notebook_json(response_text))

    output_path = prob_dir / NOTEBOOK_FILENAME
    temporary_path = prob_dir / f".{NOTEBOOK_FILENAME}.tmp"
    temporary_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_path.replace(output_path)
    return output_path
