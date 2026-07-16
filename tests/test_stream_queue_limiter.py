from __future__ import annotations

import asyncio

import pytest

from agent_service.services.stream_queue_limiter import (
    QueueAcquireStatus,
    StreamQueueLimiter,
)


async def _not_cancelled() -> bool:
    return False


@pytest.mark.asyncio
async def test_local_queue_grants_requests_in_fifo_order() -> None:
    limiter = StreamQueueLimiter(
        max_concurrent=1,
        max_wait_seconds=1,
        poll_interval_seconds=0.01,
    )
    first = await limiter.acquire("first", should_cancel=_not_cancelled)
    second_task = asyncio.create_task(limiter.acquire("second", should_cancel=_not_cancelled))
    third_task = asyncio.create_task(limiter.acquire("third", should_cancel=_not_cancelled))

    await asyncio.sleep(0.03)
    assert not second_task.done()
    assert not third_task.done()

    await limiter.release(first.permit)
    second = await asyncio.wait_for(second_task, timeout=0.3)
    assert second.status == QueueAcquireStatus.GRANTED
    assert not third_task.done()

    await limiter.release(second.permit)
    third = await asyncio.wait_for(third_task, timeout=0.3)
    assert third.status == QueueAcquireStatus.GRANTED
    await limiter.release(third.permit)


@pytest.mark.asyncio
async def test_local_queue_times_out_without_a_permit() -> None:
    limiter = StreamQueueLimiter(
        max_concurrent=1,
        max_wait_seconds=0.04,
        poll_interval_seconds=0.01,
    )
    first = await limiter.acquire("first", should_cancel=_not_cancelled)

    result = await limiter.acquire("waiting", should_cancel=_not_cancelled)

    assert result.status == QueueAcquireStatus.TIMED_OUT
    await limiter.release(first.permit)


@pytest.mark.asyncio
async def test_local_queue_removes_cancelled_waiter() -> None:
    limiter = StreamQueueLimiter(
        max_concurrent=1,
        max_wait_seconds=1,
        poll_interval_seconds=0.01,
    )
    first = await limiter.acquire("first", should_cancel=_not_cancelled)
    cancellation = asyncio.Event()

    async def should_cancel() -> bool:
        return cancellation.is_set()

    waiting = asyncio.create_task(limiter.acquire("waiting", should_cancel=should_cancel))
    await asyncio.sleep(0.03)
    cancellation.set()

    result = await asyncio.wait_for(waiting, timeout=0.3)
    assert result.status == QueueAcquireStatus.CANCELLED
    await limiter.release(first.permit)


@pytest.mark.asyncio
async def test_disabled_queue_bypasses_admission_control() -> None:
    limiter = StreamQueueLimiter(enabled=False)

    result = await limiter.acquire("bypass", should_cancel=_not_cancelled)

    assert result.status == QueueAcquireStatus.GRANTED
    assert result.permit is not None
    assert result.permit.permit_id == "bypass:bypass"
