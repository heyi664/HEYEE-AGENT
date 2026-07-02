from __future__ import annotations

from pydantic import BaseModel


class RetrievedChunk(BaseModel):
    id: str
    text: str
    score: float | None = None


class RetrievedSource(BaseModel):
    title: str
    content: str
    score: float | None = None
    source_type: str | None = None
    url: str | None = None
