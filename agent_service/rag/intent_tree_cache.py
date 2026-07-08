from __future__ import annotations

import json
from typing import Any

from agent_service.rag.intent_models import IntentNodeRecord


class IntentTreeCache:
    def __init__(
        self,
        redis_client: Any | None,
        *,
        key: str = "heyee:rag:intent:tree",
        ttl_seconds: int = 604800,
    ) -> None:
        self._redis_client = redis_client
        self._key = key
        self._ttl_seconds = ttl_seconds

    def get_records(self) -> list[IntentNodeRecord] | None:
        if self._redis_client is None:
            return None
        try:
            raw = self._redis_client.get(self._key)
            if not raw:
                return None
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            data = json.loads(str(raw))
            if not isinstance(data, list):
                return None
            return [IntentNodeRecord(**item) for item in data if isinstance(item, dict)]
        except Exception:
            return None

    def set_records(self, records: list[IntentNodeRecord]) -> None:
        if self._redis_client is None:
            return
        try:
            payload = json.dumps([record.__dict__ for record in records], ensure_ascii=False)
            setter = getattr(self._redis_client, "setex", None)
            if setter is not None:
                setter(self._key, self._ttl_seconds, payload)
            else:
                self._redis_client.set(self._key, payload)
        except Exception:
            return

    def clear(self) -> None:
        if self._redis_client is None:
            return
        try:
            self._redis_client.delete(self._key)
        except Exception:
            return
