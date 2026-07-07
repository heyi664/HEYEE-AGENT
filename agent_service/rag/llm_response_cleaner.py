from __future__ import annotations

import json
from typing import Any, cast


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = _strip_code_fence(text.strip())
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("LLM response must contain a JSON object")
    parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("LLM response must be a JSON object")
    return cast(dict[str, Any], parsed)


def _strip_code_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()
