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
    owner_id: str | None = None
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

    async def register_owner(self, task_id: str, owner_id: str) -> bool:
        """Record task ownership so a cancel request can be authorized on any worker."""

        client = self._client
        if client is None:
            return False
        try:
            created = await client.set(
                self._owner_key(task_id), owner_id, ex=self._ttl_seconds, nx=True
            )
            if created:
                return True
            return (await client.get(self._owner_key(task_id))) == owner_id
        except Exception:
            logger.exception("failed to persist stream task owner taskId=%s", task_id)
            return False

    async def get_owner(self, task_id: str) -> str | None:
        client = self._client
        if client is None:
            return None
        try:
            owner = await client.get(self._owner_key(task_id))
            return str(owner) if owner is not None else None
        except Exception:
            logger.exception("failed to read stream task owner taskId=%s", task_id)
            return None

    async def clear_owner(self, task_id: str) -> None:
        client = self._client
        if client is None:
            return
        try:
            await client.delete(self._owner_key(task_id))
        except Exception:
            logger.warning("failed to clear stream task owner taskId=%s", task_id)

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
        """Reconnect the non-durable Pub/Sub hint; Redis markers cover missed events."""

        retry_delay = 0.25
        while self._running:
            pubsub: Any | None = None
            try:
                client = self._client
                if client is None:
                    return
                pubsub = client.pubsub()
                self._pubsub = pubsub
                await pubsub.subscribe(self._channel)
                retry_delay = 0.25
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
                    logger.warning(
                        "stream cancellation subscriber disconnected; retrying in %.2fs",
                        retry_delay,
                        exc_info=True,
                    )
                    await asyncio.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, 5.0)
            finally:
                if self._pubsub is pubsub:
                    self._pubsub = None
                if pubsub is not None:
                    try:
                        await pubsub.unsubscribe(self._channel)
                        await pubsub.aclose()
                    except Exception:
                        pass

    def _key(self, task_id: str) -> str:
        return f"{self._key_prefix}:{task_id}"

    def _owner_key(self, task_id: str) -> str:
        return f"{self._key_prefix}:owner:{task_id}"


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

    @property
    def is_ready(self) -> bool:
        """A configured distributed cancellation backend must be connected at startup."""

        return self._backend is None or self._distributed_enabled

    async def register(self, task_id: str, *, owner_id: str | None = None) -> bool:
        self._purge_expired()
        if task_id in self._tasks or len(self._tasks) >= self._max_tasks:
            return False
        normalized_owner = owner_id.strip() if owner_id else None
        self._tasks[task_id] = StreamTaskInfo(owner_id=normalized_owner)
        if self._distributed_enabled and self._backend is not None:
            register_owner = getattr(self._backend, "register_owner", None)
            if normalized_owner is not None and callable(register_owner):
                if not await register_owner(task_id, normalized_owner):
                    self._tasks.pop(task_id, None)
                    return False
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

    async def cancel(self, task_id: str, *, owner_id: str | None = None) -> bool:
        if owner_id is not None and not await self.is_owned_by(task_id, owner_id):
            logger.warning("rejected stream cancellation for non-owner taskId=%s", task_id)
            return False
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

    async def is_owned_by(self, task_id: str, owner_id: str) -> bool:
        """Check the local registry first, then Redis for a stream owned by another worker."""

        self._purge_expired()
        normalized_owner = owner_id.strip()
        info = self._tasks.get(task_id)
        if info is not None:
            return info.owner_id is None or info.owner_id == normalized_owner
        if not self._distributed_enabled or self._backend is None:
            return False
        get_owner = getattr(self._backend, "get_owner", None)
        if not callable(get_owner):
            return False
        return await get_owner(task_id) == normalized_owner

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
            clear_owner = getattr(self._backend, "clear_owner", None)
            if callable(clear_owner):
                await clear_owner(task_id)

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
