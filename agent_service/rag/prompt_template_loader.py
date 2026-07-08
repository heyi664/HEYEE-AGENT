from __future__ import annotations

import re
from pathlib import Path


class PromptTemplateLoader:
    def __init__(self, base_dir: Path | None = None) -> None:
        self._base_dir = base_dir or Path(__file__).resolve().parents[1] / "prompts"
        self._cache: dict[str, str] = {}

    def load(self, path: str) -> str:
        if path not in self._cache:
            self._cache[path] = (self._base_dir / path).read_text(encoding="utf-8")
        return self._cache[path]

    def render(self, path: str, slots: dict[str, str]) -> str:
        template = self.load(path)
        rendered = template
        for key, value in slots.items():
            rendered = rendered.replace("{" + key + "}", value)
        rendered = re.sub(r"\{[A-Za-z0-9_]+\}", "", rendered)
        return cleanup_prompt(rendered)


def cleanup_prompt(prompt: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", prompt).strip()
