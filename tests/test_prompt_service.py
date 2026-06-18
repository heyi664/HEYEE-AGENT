from __future__ import annotations

from agent_service.schemas.chat import ChatHistoryItem
from agent_service.services.prompt_service import build_messages


def test_build_messages_keeps_recent_history() -> None:
    history = [ChatHistoryItem(role="user", content=f"message-{index}") for index in range(12)]

    messages = build_messages(history, "current")

    assert messages[0]["role"] == "system"
    assert len(messages) == 12
    assert messages[1]["content"] == "message-2"
    assert messages[-1] == {"role": "user", "content": "current"}

