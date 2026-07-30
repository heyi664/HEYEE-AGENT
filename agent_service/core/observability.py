from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass

_conversation_id: ContextVar[str | None] = ContextVar("conversation_id", default=None)
_trace_id: ContextVar[str | None] = ContextVar("trace_id", default=None)
_task_id: ContextVar[str | None] = ContextVar("task_id", default=None)
_metrics_logger = logging.getLogger("agent_service.metrics")


@dataclass(frozen=True)
class TraceBinding:
    """Tokens required to restore the request trace context.

    A streaming request crosses preparation tasks, model callbacks and queue-admission
    stages.  Context variables give all synchronous stage logs in that request the same
    identifiers without passing a growing list of logging arguments through every layer.
    """

    trace_token: Token[str | None]
    task_token: Token[str | None]


def push_trace(trace_id: str, *, task_id: str | None = None) -> TraceBinding:
    """Bind a trace to the current async context until :func:`pop_trace` is called."""

    return TraceBinding(
        trace_token=_trace_id.set(trace_id),
        task_token=_task_id.set(task_id),
    )


def pop_trace(binding: TraceBinding) -> None:
    _task_id.reset(binding.task_token)
    _trace_id.reset(binding.trace_token)


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
        f"traceId={_trace_id.get() or '-'}",
        f"taskId={_task_id.get() or '-'}",
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
