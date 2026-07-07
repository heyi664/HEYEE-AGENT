from __future__ import annotations

import json
from typing import Any, Protocol, cast

from agent_service.core.config import get_settings


class QueryTermMappingRepository(Protocol):
    def list_mappings(self) -> dict[str, str]: ...


class QueryTermMappingService:
    def __init__(
        self,
        mappings: dict[str, str] | None = None,
        repository: QueryTermMappingRepository | None = None,
        redis_client: Any | None = None,
        cache_key: str = "heyee:rag:query-term-mappings",
        cache_ttl_seconds: int | None = None,
    ) -> None:
        settings = get_settings()
        self._static_mappings = mappings
        self._repository = repository
        self._redis_client = redis_client
        self._cache_key = cache_key
        self._cache_ttl_seconds = cache_ttl_seconds or settings.rag_term_mapping_cache_ttl_seconds
        self._loaded_mappings: dict[str, str] | None = None

    def normalize(self, question: str) -> str:
        normalized = question
        for source, target in self._ordered_mappings().items():
            normalized = self._apply_mapping(normalized, source, target)
        return normalized

    def _ordered_mappings(self) -> dict[str, str]:
        mappings = self._load_mappings()
        return dict(sorted(mappings.items(), key=lambda item: len(item[0]), reverse=True))

    def _load_mappings(self) -> dict[str, str]:
        if self._loaded_mappings is not None:
            return self._loaded_mappings
        if self._static_mappings is not None:
            self._loaded_mappings = self._static_mappings
            return self._loaded_mappings

        cached = self._read_cache()
        if cached is not None:
            self._loaded_mappings = cached
            return cached

        if self._repository is None:
            self._loaded_mappings = {}
            return self._loaded_mappings

        mappings = self._repository.list_mappings()
        self._write_cache(mappings)
        self._loaded_mappings = mappings
        return mappings

    def _read_cache(self) -> dict[str, str] | None:
        if self._redis_client is None:
            return None
        raw = self._redis_client.get(self._cache_key)
        if not raw:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        parsed = json.loads(str(raw))
        if not isinstance(parsed, dict):
            return None
        return {str(key): str(value) for key, value in cast(dict[object, object], parsed).items()}

    def _write_cache(self, mappings: dict[str, str]) -> None:
        if self._redis_client is None:
            return
        self._redis_client.setex(
            self._cache_key,
            self._cache_ttl_seconds,
            json.dumps(mappings, ensure_ascii=False),
        )

    def _apply_mapping(self, text: str, source: str, target: str) -> str:
        if not text or not source or not target:
            return text
        chunks: list[str] = []
        index = 0
        while index < len(text):
            hit = text.find(source, index)
            if hit < 0:
                chunks.append(text[index:])
                break
            chunks.append(text[index:hit])
            if text.startswith(target, hit):
                chunks.append(target)
                index = hit + len(target)
            else:
                chunks.append(target)
                index = hit + len(source)
        return "".join(chunks)
