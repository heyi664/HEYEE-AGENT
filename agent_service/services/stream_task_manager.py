from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class CancellableStream(Protocol):
    def cancel(self) -> None: ...


class DistributedCancellationBackend(Protocol):
    """Persist cancellation facts and broadcast them to the other instances."""

    async def start(self, on_cancel: Callable[[str], None]) -> bool: ...

    async def mark_cancelled(self, task_id: str) -> bool: ...

    async def publish_cancel(self, task_id: str) -> bool: ...

    async def is_cancelled(self, task_id: str) -> bool: ...

    async def clear(self, task_id: str) -> None: ...

    async def close(self) -> None: ...


@dataclass
class StreamTaskInfo:
    cancelled: bool = False
    handle: CancellableStream | None = None
    expires_at: float | None = None


class RedisCancellationBackend:
    """Redis Key + Pub/Sub implementation used only when explicitly configured."""

    def __init__(
        self,
        redis_url: str,
        *,
        key_prefix: str,
        channel: str,
        ttl_seconds: int,
    ) -> None:
        self._redis_url = redis_url
        self._key_prefix = key_prefix.rstrip(":")
        self._channel = channel
        self._ttl_seconds = ttl_seconds
        self._client: Any | None = None
        self._pubsub: Any | None = None
        self._listener: asyncio.Task[None] | None = None
        self._running = False
        self._on_cancel: Callable[[str], None] | None = None

    async def start(self, on_cancel: Callable[[str], None]) -> bool:
        try:
            import redis.asyncio as redis

            self._client = redis.Redis.from_url(self._redis_url, decode_responses=True)
            await self._client.ping()
            self._pubsub = self._client.pubsub()
            await self._pubsub.subscribe(self._channel)
            self._on_cancel = on_cancel
            self._running = True
            self._listener = asyncio.create_task(self._listen(), name="stream-cancel-listener")
            logger.info("stream cancellation Redis coordination enabled channel=%s", self._channel)
            return True
        except Exception:
            logger.exception("stream cancellation Redis coordination is unavailable")
            await self.close()
            return False

    async def mark_cancelled(self, task_id: str) -> bool:
        client = self._client
        if client is None:
            return False
        try:
            await client.set(self._key(task_id), "1", ex=self._ttl_seconds)
            return True
        except Exception:
            logger.exception("failed to persist stream cancellation taskId=%s", task_id)
            return False

    async def publish_cancel(self, task_id: str) -> bool:
        client = self._client
        if client is None:
            return False
        try:
            await client.publish(self._channel, task_id)
            return True
        except Exception:
            logger.exception("failed to broadcast stream cancellation taskId=%s", task_id)
            return False

    async def is_cancelled(self, task_id: str) -> bool:
        client = self._client
        if client is None:
            return False
        try:
            return bool(await client.get(self._key(task_id)))
        except Exception:
            logger.exception("failed to read stream cancellation marker taskId=%s", task_id)
            return False

    async def clear(self, task_id: str) -> None:
        client = self._client
        if client is None:
            return
        try:
            await client.delete(self._key(task_id))
        except Exception:
            logger.warning("failed to clear stream cancellation marker taskId=%s", task_id)

    async def close(self) -> None:
        self._running = False
        listener = self._listener
        self._listener = None
        if listener is not None:
            listener.cancel()
            try:
                await listener
            except asyncio.CancelledError:
                pass
        pubsub = self._pubsub
        self._pubsub = None
        if pubsub is not None:
            try:
                await pubsub.unsubscribe(self._channel)
                await pubsub.aclose()
            except Exception:
                logger.warning("failed to close stream cancellation Redis subscription")
        client = self._client
        self._client = None
        if client is not None:
            try:
                await client.aclose()
            except Exception:
                logger.warning("failed to close stream cancellation Redis client")

    async def _listen(self) -> None:
        pubsub = self._pubsub
        if pubsub is None:
            return
        try:
            while self._running:
                message = await pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=1.0,
                )
                if not message or message.get("type") != "message":
                    continue
                task_id = str(message.get("data") or "").strip()
                if task_id and self._on_cancel is not None:
                    self._on_cancel(task_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            if self._running:
                logger.exception("stream cancellation Redis subscriber stopped unexpectedly")

    def _key(self, task_id: str) -> str:
        return f"{self._key_prefix}:{task_id}"


class StreamTaskManager:
    """Local task registry with optional Redis-backed cross-instance cancellation.

    Token callbacks only call :meth:`is_cancelled`, which is a local memory lookup.
    Redis is consulted when a task registers to close the cancel-before-register race.
    """

    def __init__(
        self,
        backend: DistributedCancellationBackend | None = None,
        *,
        cancelled_ttl_seconds: int = 1800,
        max_tasks: int = 10000,
    ) -> None:
        self._tasks: dict[str, StreamTaskInfo] = {}
        self._backend = backend
        self._cancelled_ttl_seconds = cancelled_ttl_seconds
        self._max_tasks = max_tasks
        self._distributed_enabled = False

    async def start(
        self,
        *,
        redis_url: str | None = None,
        key_prefix: str = "heyee:stream:cancel",
        channel: str = "heyee:stream:cancel",
        ttl_seconds: int = 1800,
        max_tasks: int | None = None,
    ) -> None:
        self._cancelled_ttl_seconds = ttl_seconds
        if max_tasks is not None:
            self._max_tasks = max_tasks
        if self._backend is None and redis_url:
            self._backend = RedisCancellationBackend(
                redis_url,
                key_prefix=key_prefix,
                channel=channel,
                ttl_seconds=ttl_seconds,
            )
        if self._backend is not None:
            self._distributed_enabled = await self._backend.start(self.cancel_local)

    async def close(self) -> None:
        if self._backend is not None:
            await self._backend.close()
        self._distributed_enabled = False

    async def register(self, task_id: str) -> bool:
        self._purge_expired()
        if task_id in self._tasks or len(self._tasks) >= self._max_tasks:
            return False
        self._tasks[task_id] = StreamTaskInfo()
        if self._distributed_enabled and self._backend is not None:
            if await self._backend.is_cancelled(task_id):
                self.cancel_local(task_id)
        return True

    def bind(self, task_id: str, handle: CancellableStream) -> bool:
        info = self._tasks.get(task_id)
        if info is None:
            return False
        info.handle = handle
        if info.cancelled:
            handle.cancel()
            return False
        return True

    async def cancel(self, task_id: str) -> bool:
        persisted = False
        published = False
        if self._distributed_enabled and self._backend is not None:
            # Persist before publishing so a late registering node cannot miss cancellation.
            persisted = await self._backend.mark_cancelled(task_id)
            # Still attempt the immediate notification if persistence has a transient failure.
            # The ordering remains marker first, then broadcast.
            published = await self._backend.publish_cancel(task_id)
        local = self.cancel_local(task_id)
        return local or persisted or published

    def cancel_local(self, task_id: str) -> bool:
        self._purge_expired()
        info = self._tasks.get(task_id)
        if info is None or info.cancelled:
            return False
        info.cancelled = True
        info.expires_at = time.monotonic() + self._cancelled_ttl_seconds
        if info.handle is not None:
            info.handle.cancel()
        return True

    async def finalize(self, task_id: str) -> None:
        self._purge_expired()
        info = self._tasks.get(task_id)
        if info is None:
            return
        if info.cancelled:
            # Keep the local tombstone for delayed callbacks; Redis keeps the same fact by TTL.
            info.handle = None
            info.expires_at = time.monotonic() + self._cancelled_ttl_seconds
            return
        self._tasks.pop(task_id, None)
        if self._distributed_enabled and self._backend is not None:
            await self._backend.clear(task_id)

    def contains(self, task_id: str) -> bool:
        self._purge_expired()
        return task_id in self._tasks

    def is_cancelled(self, task_id: str) -> bool:
        self._purge_expired()
        info = self._tasks.get(task_id)
        return bool(info and info.cancelled)

    def _purge_expired(self) -> None:
        now = time.monotonic()
        expired = [
            task_id
            for task_id, info in self._tasks.items()
            if info.expires_at is not None and info.expires_at <= now
        ]
        for task_id in expired:
            self._tasks.pop(task_id, None)


stream_task_manager = StreamTaskManager()
