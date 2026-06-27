from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from agent_service.core.config import Settings

UPLOAD_PATH = "/v1/knowledge-documents/upload"
logger = logging.getLogger(__name__)

ACQUIRE_SCRIPT = """
local key = KEYS[1]
local now_ms = tonumber(ARGV[1])
local lease_ms = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
local token = ARGV[4]
redis.call('ZREMRANGEBYSCORE', key, '-inf', now_ms)
local count = redis.call('ZCARD', key)
if count < limit then
    redis.call('ZADD', key, now_ms + lease_ms, token)
    redis.call('PEXPIRE', key, lease_ms)
    return token
end
return nil
"""


class UploadRateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, settings: Settings) -> None:  # type: ignore[no-untyped-def]
        super().__init__(app)
        self.settings = settings

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        if not self._should_limit(request):
            return await call_next(request)

        try:
            permit = await self._acquire_permit()
        except RuntimeError:
            logger.exception("upload rate limiter is unavailable")
            return JSONResponse(
                status_code=503,
                content={"detail": "Upload rate limiter is unavailable"},
            )

        if permit is None:
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many concurrent upload requests"},
            )

        try:
            return await call_next(request)
        finally:
            try:
                await self._release_permit(permit)
            except RuntimeError:
                logger.exception("failed to release upload rate limit permit")

    def _should_limit(self, request: Request) -> bool:
        return (
            self.settings.upload_rate_limit_enabled
            and request.method.upper() == "POST"
            and request.url.path == UPLOAD_PATH
        )

    async def _acquire_permit(self) -> str | None:
        redis = self._load_redis()
        token = uuid.uuid4().hex
        deadline = time.monotonic() + self.settings.upload_rate_limit_acquire_timeout_ms / 1000
        lease_ms = self.settings.upload_rate_limit_lease_seconds * 1000
        client = redis.Redis.from_url(
            self.settings.upload_rate_limit_redis_url,
            decode_responses=True,
        )
        try:
            while True:
                now_ms = int(time.time() * 1000)
                try:
                    acquired = await client.eval(
                        ACQUIRE_SCRIPT,
                        1,
                        self.settings.upload_rate_limit_key,
                        now_ms,
                        lease_ms,
                        self.settings.upload_rate_limit_permits,
                        token,
                    )
                except redis.RedisError as exc:
                    raise RuntimeError("redis upload rate limiter failed") from exc
                if acquired:
                    return token
                if self.settings.upload_rate_limit_acquire_timeout_ms == 0:
                    return None
                if time.monotonic() >= deadline:
                    return None
                await asyncio.sleep(0.05)
        finally:
            await client.aclose()

    async def _release_permit(self, token: str) -> None:
        redis = self._load_redis()
        client = redis.Redis.from_url(
            self.settings.upload_rate_limit_redis_url,
            decode_responses=True,
        )
        try:
            try:
                await client.zrem(self.settings.upload_rate_limit_key, token)
            except redis.RedisError as exc:
                raise RuntimeError("redis upload rate limiter release failed") from exc
        finally:
            await client.aclose()

    def _load_redis(self):  # type: ignore[no-untyped-def]
        try:
            import redis.asyncio as redis
        except ImportError as exc:  # pragma: no cover - depends on runtime environment
            raise RuntimeError("redis package is required for upload rate limiting") from exc
        return redis
