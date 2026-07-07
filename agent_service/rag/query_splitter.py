from __future__ import annotations

QUESTION_ENDINGS = {"?", "？"}
SEPARATORS = {";", "；", "\n"}


def split_query(question: str) -> list[str]:
    original = question.strip()
    if not original:
        return []

    parts: list[str] = []
    current: list[str] = []
    for char in original:
        if char in SEPARATORS:
            _append_part(parts, "".join(current), keep_empty=False)
            current = []
            continue
        current.append(char)
        if char in QUESTION_ENDINGS:
            _append_part(parts, "".join(current), keep_empty=False)
            current = []
    _append_part(parts, "".join(current), keep_empty=False)

    return parts if len(parts) > 1 else [original]


def _append_part(parts: list[str], value: str, *, keep_empty: bool) -> None:
    cleaned = value.strip()
    if cleaned or keep_empty:
        parts.append(cleaned)
