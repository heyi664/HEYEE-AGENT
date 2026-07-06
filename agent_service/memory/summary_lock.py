from __future__ import annotations

from typing import Protocol
from uuid import uuid4


class SummaryCompressionLock(Protocol):
    def acquire(self, conversation_id: str, user_id: str) -> str | None: ...

    def release(self, token: str) -> None: ...


class RedisSummaryCompressionLock:
    def __init__(
        self,
        redis_url: str,
        *,
        ttl_seconds: int = 120,
        key_prefix: str = "heyee:memory:summary:lock",
    ) -> None:
        from redis import Redis

        self._client = Redis.from_url(redis_url, decode_responses=True)
        self._ttl_seconds = ttl_seconds
        self._key_prefix = key_prefix.rstrip(":")
        self._tokens: dict[str, str] = {}

    def acquire(self, conversation_id: str, user_id: str) -> str | None:
        key = self._build_key(conversation_id, user_id)
        token = uuid4().hex
        acquired = self._client.set(key, token, nx=True, ex=self._ttl_seconds)
        if not acquired:
            return None
        self._tokens[token] = key
        return token

    def release(self, token: str) -> None:
        key = self._tokens.pop(token, None)
        if not key:
            return
        script = """
        if redis.call('get', KEYS[1]) == ARGV[1] then
            return redis.call('del', KEYS[1])
        end
        return 0
        """
        self._client.eval(script, 1, key, token)

    def _build_key(self, conversation_id: str, user_id: str) -> str:
        return f"{self._key_prefix}:{user_id}:{conversation_id}"
