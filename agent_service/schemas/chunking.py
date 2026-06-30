from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class KnowledgeChunkMessage:
    doc_id: str
    kb_id: str | None = None
    file_url: str | None = None
    chunk_strategy: str | None = None
    chunk_config: dict[str, Any] | None = None
    requested_by: str | None = None
    message_id: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> KnowledgeChunkMessage:
        doc_id = str(payload.get("docId") or payload.get("doc_id") or "").strip()
        if not doc_id:
            raise ValueError("docId is required")
        chunk_config = payload.get("chunkConfig") or payload.get("chunk_config")
        if chunk_config is not None and not isinstance(chunk_config, dict):
            raise ValueError("chunkConfig must be an object")
        return cls(
            doc_id=doc_id,
            kb_id=_optional_text(payload.get("kbId") or payload.get("kb_id")),
            file_url=_optional_text(payload.get("fileUrl") or payload.get("file_url")),
            chunk_strategy=_optional_text(
                payload.get("chunkStrategy") or payload.get("chunk_strategy")
            ),
            chunk_config=chunk_config,
            requested_by=_optional_text(payload.get("requestedBy") or payload.get("requested_by")),
            message_id=_optional_text(payload.get("messageId") or payload.get("message_id")),
        )


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    chunk_index: int
    content: str
    content_hash: str
    char_count: int
    token_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VectorChunk:
    chunk_id: str
    doc_id: str
    kb_id: str
    chunk_index: int
    content: str
    vector: list[float]
    metadata: dict[str, Any]


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
