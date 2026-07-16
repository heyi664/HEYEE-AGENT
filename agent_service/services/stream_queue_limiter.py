from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


class QueueAcquireStatus(StrEnum):
    GRANTED = "granted"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class StreamQueuePermit:
    task_id: str
    permit_id: str
    distributed: bool


@dataclass(frozen=True)
class QueueAcquireResult:
    status: QueueAcquireStatus
    permit: StreamQueuePermit | None = None


@dataclass
class _LocalWaiter:
    task_id: str
    deadline: float


class StreamQueueLimiter:
    """Fair stream admission control with a local fallback and Redis cluster mode.

    Redis mode uses an expiring permit ZSET, a FIFO waiting ZSET, per-entry TTL markers,
    a monotonic sequence and Pub/Sub wakeups.  The Lua claim operation performs the
    queue-window check and permit allocation atomically.
    """

    _CLAIM_SCRIPT = """
local queue_key = KEYS[1]
local permit_key = KEYS[2]
local entry_prefix = ARGV[1]
local task_id = ARGV[2]
local permit_id = ARGV[3]
local max_concurrent = tonumber(ARGV[4])
local now_ms = tonumber(ARGV[5])
local lease_until_ms = tonumber(ARGV[6])
local scan_limit = tonumber(ARGV[7])

redis.call('ZREMRANGEBYSCORE', permit_key, '-inf', now_ms)

local entries = redis.call('ZRANGE', queue_key, 0, scan_limit)
for _, member in ipairs(entries) do
  if redis.call('EXISTS', entry_prefix .. member) == 0 then
    redis.call('ZREM', queue_key, member)
  end
end

local rank = redis.call('ZRANK', queue_key, task_id)
if not rank or rank >= max_concurrent then
  return 0
end
if redis.call('ZCARD', permit_key) >= max_concurrent then
  return 0
end

redis.call('ZREM', queue_key, task_id)
redis.call('DEL', entry_prefix .. task_id)
redis.call('ZADD', permit_key, lease_until_ms, permit_id)
return 1
"""

    def __init__(
        self,
        *,
        enabled: bool = True,
        max_concurrent: int = 3,
        max_wait_seconds: float = 20,
        lease_seconds: float = 600,
        poll_interval_seconds: float = 0.2,
        redis_url: str | None = None,
        key_prefix: str = "heyee:stream:queue",
    ) -> None:
        self._enabled = enabled
        self._max_concurrent = max_concurrent
        self._max_wait_seconds = max_wait_seconds
        self._lease_seconds = lease_seconds
        self._poll_interval_seconds = poll_interval_seconds
        self._redis_url = redis_url
        self._key_prefix = key_prefix.rstrip(":")
        self._client: Any | None = None
        self._pubsub: Any | None = None
        self._listener: asyncio.Task[None] | None = None
        self._notification = asyncio.Event()
        self._distributed = False
        self._unavailable = False
        self._running = False
        self._closing = False
        self._distributed_waiting: set[str] = set()

        self._local_condition = asyncio.Condition()
        self._local_waiting: deque[_LocalWaiter] = deque()
        self._local_active: dict[str, StreamQueuePermit] = {}

    @property
    def lease_seconds(self) -> float:
        return self._lease_seconds

    async def start(
        self,
        *,
        enabled: bool,
        max_concurrent: int,
        max_wait_seconds: float,
        lease_seconds: float,
        poll_interval_ms: int,
        redis_url: str | None,
        key_prefix: str,
    ) -> None:
        self._enabled = enabled
        self._max_concurrent = max_concurrent
        self._max_wait_seconds = max_wait_seconds
        self._lease_seconds = lease_seconds
        self._poll_interval_seconds = poll_interval_ms / 1000
        self._redis_url = redis_url
        self._key_prefix = key_prefix.rstrip(":")
        self._unavailable = False
        self._closing = False
        self._notification.clear()
        if not enabled or not redis_url:
            return
        try:
            import redis.asyncio as redis

            self._client = redis.Redis.from_url(redis_url, decode_responses=True)
            await self._client.ping()
            self._pubsub = self._client.pubsub()
            await self._pubsub.subscribe(self._notify_channel)
            self._running = True
            self._listener = asyncio.create_task(
                self._listen_notifications(),
                name="stream-queue-notifications",
            )
            self._distributed = True
            logger.info("stream queue limiter uses Redis keyPrefix=%s", self._key_prefix)
        except Exception:
            self._unavailable = True
            logger.exception("stream queue Redis coordination is unavailable")
            await self.close()

    async def close(self) -> None:
        self._closing = True
        self._notification.set()
        async with self._local_condition:
            # Active streams keep their local permit until their normal finally block releases
            # it.  Only wake queued requests so they can leave with UNAVAILABLE immediately.
            self._local_condition.notify_all()

        client = self._client
        waiting_task_ids = tuple(self._distributed_waiting)
        self._distributed_waiting.clear()
        if client is not None and waiting_task_ids:
            try:
                await client.zrem(self._queue_key, *waiting_task_ids)
                await client.delete(*(self._entry_key(task_id) for task_id in waiting_task_ids))
                await client.publish(self._notify_channel, "shutdown")
            except Exception:
                logger.warning("failed to clear queued stream tasks during shutdown")
        self._running = False
        self._distributed = False
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
                await pubsub.unsubscribe(self._notify_channel)
                await pubsub.aclose()
            except Exception:
                logger.warning("failed to close stream queue Redis subscription")
        self._client = None
        if client is not None:
            try:
                await client.aclose()
            except Exception:
                logger.warning("failed to close stream queue Redis client")

    async def acquire(
        self,
        task_id: str,
        *,
        should_cancel: Callable[[], Awaitable[bool]],
    ) -> QueueAcquireResult:
        if not self._enabled:
            return QueueAcquireResult(
                QueueAcquireStatus.GRANTED,
                StreamQueuePermit(task_id, f"bypass:{task_id}", False),
            )
        if self._closing or self._unavailable:
            return QueueAcquireResult(QueueAcquireStatus.UNAVAILABLE)
        if self._distributed:
            return await self._acquire_distributed(task_id, should_cancel=should_cancel)
        return await self._acquire_local(task_id, should_cancel=should_cancel)

    async def cancel(self, task_id: str) -> None:
        """Remove a waiting request immediately; active permits release in ``release``."""
        if not self._enabled:
            return
        if self._distributed:
            await self._remove_distributed_waiter(task_id, notify=True)
            return
        async with self._local_condition:
            self._local_waiting = deque(
                item for item in self._local_waiting if item.task_id != task_id
            )
            self._local_condition.notify_all()

    async def release(self, permit: StreamQueuePermit | None) -> None:
        if permit is None or not self._enabled:
            return
        if permit.distributed:
            client = self._client
            if client is None:
                return
            try:
                await client.zrem(self._permit_key, permit.permit_id)
                await client.publish(self._notify_channel, "release")
            except Exception:
                logger.exception("failed to release stream permit taskId=%s", permit.task_id)
            return
        async with self._local_condition:
            self._local_active.pop(permit.permit_id, None)
            self._local_condition.notify_all()

    async def renew(self, permit: StreamQueuePermit | None) -> bool:
        if permit is None or not permit.distributed or not self._enabled:
            return True
        client = self._client
        if client is None:
            return False
        try:
            renewed = await client.zadd(
                self._permit_key,
                {permit.permit_id: self._lease_deadline_ms()},
                xx=True,
                ch=True,
            )
            return bool(renewed)
        except Exception:
            logger.exception("failed to renew stream permit taskId=%s", permit.task_id)
            return False

    async def _acquire_local(
        self,
        task_id: str,
        *,
        should_cancel: Callable[[], Awaitable[bool]],
    ) -> QueueAcquireResult:
        deadline = time.monotonic() + self._max_wait_seconds
        waiter = _LocalWaiter(task_id, deadline)
        async with self._local_condition:
            self._local_waiting.append(waiter)
            while True:
                if self._closing:
                    self._remove_local_waiter(task_id)
                    self._local_condition.notify_all()
                    return QueueAcquireResult(QueueAcquireStatus.UNAVAILABLE)
                if await should_cancel():
                    self._remove_local_waiter(task_id)
                    self._local_condition.notify_all()
                    return QueueAcquireResult(QueueAcquireStatus.CANCELLED)
                if self._local_waiting and self._local_waiting[0] is waiter:
                    if len(self._local_active) < self._max_concurrent:
                        self._local_waiting.popleft()
                        permit = StreamQueuePermit(task_id, uuid4().hex, False)
                        self._local_active[permit.permit_id] = permit
                        return QueueAcquireResult(QueueAcquireStatus.GRANTED, permit)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._remove_local_waiter(task_id)
                    self._local_condition.notify_all()
                    return QueueAcquireResult(QueueAcquireStatus.TIMED_OUT)
                try:
                    await asyncio.wait_for(
                        self._local_condition.wait(),
                        timeout=min(remaining, self._poll_interval_seconds),
                    )
                except TimeoutError:
                    continue

    async def _acquire_distributed(
        self,
        task_id: str,
        *,
        should_cancel: Callable[[], Awaitable[bool]],
    ) -> QueueAcquireResult:
        client = self._client
        if client is None:
            return QueueAcquireResult(QueueAcquireStatus.UNAVAILABLE)
        deadline = time.monotonic() + self._max_wait_seconds
        permit = StreamQueuePermit(task_id, uuid4().hex, True)
        self._distributed_waiting.add(task_id)
        try:
            await client.set(self._entry_key(task_id), "1", ex=self._entry_ttl_seconds)
            score = await client.incr(self._sequence_key)
            await client.zadd(self._queue_key, {task_id: score})
            await client.publish(self._notify_channel, "enqueue")
        except Exception:
            logger.exception("failed to enqueue stream taskId=%s", task_id)
            self._distributed_waiting.discard(task_id)
            return QueueAcquireResult(QueueAcquireStatus.UNAVAILABLE)

        try:
            while True:
                if self._closing:
                    await self._remove_distributed_waiter(task_id, notify=True)
                    return QueueAcquireResult(QueueAcquireStatus.UNAVAILABLE)
                if self._unavailable:
                    # The Pub/Sub listener can fail after this request has joined the queue.
                    # Do not continue issuing claims against a coordination backend that is no
                    # longer considered healthy; remove the entry and make the failure explicit.
                    await self._remove_distributed_waiter(task_id, notify=False)
                    return QueueAcquireResult(QueueAcquireStatus.UNAVAILABLE)
                if await should_cancel():
                    await self._remove_distributed_waiter(task_id, notify=True)
                    return QueueAcquireResult(QueueAcquireStatus.CANCELLED)
                if time.monotonic() >= deadline:
                    await self._remove_distributed_waiter(task_id, notify=True)
                    return QueueAcquireResult(QueueAcquireStatus.TIMED_OUT)
                try:
                    claimed = await client.eval(
                        self._CLAIM_SCRIPT,
                        2,
                        self._queue_key,
                        self._permit_key,
                        self._entry_prefix,
                        task_id,
                        permit.permit_id,
                        self._max_concurrent,
                        int(time.time() * 1000),
                        self._lease_deadline_ms(),
                        self._max_concurrent + 16,
                    )
                except Exception:
                    logger.exception("failed to claim stream queue position taskId=%s", task_id)
                    await self._remove_distributed_waiter(task_id, notify=False)
                    return QueueAcquireResult(QueueAcquireStatus.UNAVAILABLE)
                if int(claimed or 0) == 1:
                    return QueueAcquireResult(QueueAcquireStatus.GRANTED, permit)
                self._notification.clear()
                remaining = max(0.001, deadline - time.monotonic())
                try:
                    await asyncio.wait_for(
                        self._notification.wait(),
                        timeout=min(remaining, self._poll_interval_seconds),
                    )
                except TimeoutError:
                    continue
        except asyncio.CancelledError:
            await self._remove_distributed_waiter(task_id, notify=True)
            raise
        finally:
            self._distributed_waiting.discard(task_id)

    async def _remove_distributed_waiter(self, task_id: str, *, notify: bool) -> None:
        self._distributed_waiting.discard(task_id)
        client = self._client
        if client is None:
            return
        try:
            await client.zrem(self._queue_key, task_id)
            await client.delete(self._entry_key(task_id))
            if notify:
                await client.publish(self._notify_channel, "queue-change")
        except Exception:
            logger.warning("failed to remove queued stream taskId=%s", task_id)

    def _remove_local_waiter(self, task_id: str) -> None:
        self._local_waiting = deque(item for item in self._local_waiting if item.task_id != task_id)

    async def _listen_notifications(self) -> None:
        pubsub = self._pubsub
        if pubsub is None:
            return
        try:
            while self._running:
                message = await pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=max(1.0, self._poll_interval_seconds * 5),
                )
                if message and message.get("type") == "message":
                    self._notification.set()
        except asyncio.CancelledError:
            raise
        except Exception:
            if self._running:
                self._unavailable = True
                self._notification.set()
                logger.exception("stream queue Redis subscriber stopped unexpectedly")

    @property
    def _queue_key(self) -> str:
        return f"{self._key_prefix}:waiting"

    @property
    def _permit_key(self) -> str:
        return f"{self._key_prefix}:permits"

    @property
    def _sequence_key(self) -> str:
        return f"{self._key_prefix}:sequence"

    @property
    def _entry_prefix(self) -> str:
        return f"{self._key_prefix}:entry:"

    @property
    def _notify_channel(self) -> str:
        return f"{self._key_prefix}:notify"

    def _entry_key(self, task_id: str) -> str:
        return f"{self._entry_prefix}{task_id}"

    @property
    def _entry_ttl_seconds(self) -> int:
        return max(1, int(self._max_wait_seconds) + 5)

    def _lease_deadline_ms(self) -> int:
        return int((time.time() + self._lease_seconds) * 1000)


stream_queue_limiter = StreamQueueLimiter()
