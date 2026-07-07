from __future__ import annotations

import logging
from typing import Any, Protocol

from agent_service.core.config import get_settings

logger = logging.getLogger(__name__)


class SummaryCompressionLock(Protocol):
    def acquire(self, conversation_id: str, user_id: str) -> bool: ...

    def release(self, conversation_id: str, user_id: str) -> None: ...


class NoopSummaryCompressionLock:
    def acquire(self, conversation_id: str, user_id: str) -> bool:
        return True

    def release(self, conversation_id: str, user_id: str) -> None:
        return None


class RedisSummaryCompressionLock:
    def __init__(self, redis_url: str | None = None, ttl_seconds: int | None = None) -> None:
        settings = get_settings()
        self._redis_url = redis_url or settings.memory_redis_url
        self._ttl_seconds = ttl_seconds or settings.memory_lock_ttl_seconds
        self._client: object | None = None

    def acquire(self, conversation_id: str, user_id: str) -> bool:
        if not self._redis_url:
            return True
        try:
            client = self._get_client()
            return bool(
                client.set(
                    self._key(conversation_id, user_id),
                    "1",
                    nx=True,
                    ex=self._ttl_seconds,
                )
            )
        except Exception:
            logger.exception(
                "redis summary compression lock acquire failed conversationId=%s userId=%s",
                conversation_id,
                user_id,
            )
            return True

    def release(self, conversation_id: str, user_id: str) -> None:
        if not self._redis_url:
            return
        try:
            self._get_client().delete(self._key(conversation_id, user_id))
        except Exception:
            logger.exception(
                "redis summary compression lock release failed conversationId=%s userId=%s",
                conversation_id,
                user_id,
            )

    def _get_client(self) -> Any:
        if self._client is None:
            import redis

            redis_url = self._redis_url
            if redis_url is None:
                raise RuntimeError("MEMORY_REDIS_URL is not configured")
            self._client = redis.Redis.from_url(redis_url, decode_responses=True)
        return self._client

    def _key(self, conversation_id: str, user_id: str) -> str:
        return f"heyee:conversation-summary-lock:{user_id}:{conversation_id}"


