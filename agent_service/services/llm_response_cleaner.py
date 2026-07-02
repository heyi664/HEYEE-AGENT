from __future__ import annotations

import re


class LLMResponseCleaner:
    _leading_code_fence = re.compile(r"^```[\w-]*\s*\n?")
    _trailing_code_fence = re.compile(r"\n?```\s*$")

    @classmethod
    def strip_markdown_code_fence(cls, raw: str | None) -> str | None:
        if raw is None:
            return None
        cleaned = raw.strip()
        cleaned = cls._leading_code_fence.sub("", cleaned, count=1)
        cleaned = cls._trailing_code_fence.sub("", cleaned, count=1)
        return cleaned.strip()
