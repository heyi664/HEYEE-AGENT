from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_conversation_id: ContextVar[str | None] = ContextVar("conversation_id", default=None)
_metrics_logger = logging.getLogger("agent_service.metrics")


@contextmanager
def bind_conversation(conversation_id: str) -> Iterator[None]:
    token = _conversation_id.set(conversation_id)
    try:
        yield
    finally:
        _conversation_id.reset(token)


def record_stage(stage: str, *, elapsed_ms: int, **metrics: object) -> None:
    fields = [
        f"stage={stage}",
        f"conversationId={_conversation_id.get() or '-'}",
        f"elapsedMs={max(elapsed_ms, 0)}",
    ]
    fields.extend(
        f"{key}={_metric_value(value)}" for key, value in sorted(metrics.items())
    )
    _metrics_logger.info("rag_metric %s", " ".join(fields))


def _metric_value(value: object) -> str:
    if value is None:
        return "-"
    return str(value).replace(" ", "_").replace("\n", "_")
