from __future__ import annotations

from pydantic import BaseModel


class RetrievedSource(BaseModel):
    title: str
    content: str
    score: float | None = None
    source_type: str | None = None
    url: str | None = None

