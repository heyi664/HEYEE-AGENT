from __future__ import annotations

from typing import Protocol


class CancellableStream(Protocol):
    def cancel(self) -> None: ...


class StreamTaskManager:
    """In-process task registry used by the HTTP stream and explicit cancel endpoint."""

    def __init__(self) -> None:
        self._handles: dict[str, CancellableStream] = {}
        self._cancelled: set[str] = set()

    def bind(self, task_id: str, handle: CancellableStream) -> bool:
        if task_id in self._handles:
            return False
        self._handles[task_id] = handle
        self._cancelled.discard(task_id)
        return True

    def cancel(self, task_id: str) -> bool:
        handle = self._handles.get(task_id)
        if handle is None:
            return False
        self._cancelled.add(task_id)
        handle.cancel()
        return True

    def unbind(self, task_id: str) -> None:
        self._handles.pop(task_id, None)
        self._cancelled.discard(task_id)

    def contains(self, task_id: str) -> bool:
        return task_id in self._handles

    def is_cancelled(self, task_id: str) -> bool:
        return task_id in self._cancelled


stream_task_manager = StreamTaskManager()
