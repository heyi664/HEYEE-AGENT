from __future__ import annotations

from agent_service.schemas.chat import ChatHistoryItem


def trim_history(history: list[ChatHistoryItem], limit: int = 10) -> list[ChatHistoryItem]:
    return history[-limit:]

