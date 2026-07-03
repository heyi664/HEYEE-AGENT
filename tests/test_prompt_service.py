from __future__ import annotations

from agent_service.memory.models import MemoryContext, MemoryMessage, MemorySummary
from agent_service.services.prompt_service import build_messages


def test_build_messages_uses_memory_context_and_current_message_once() -> None:
    context = MemoryContext(
        summary=MemorySummary(
            id="sum_1",
            conversation_id="conv_1",
            user_id="user_1",
            last_message_id="msg_1",
            content="用户预算 5000。",
        ),
        messages=[
            MemoryMessage(id="msg_2", role="user", content="历史问题"),
            MemoryMessage(id="msg_3", role="assistant", content="历史回答"),
        ],
    )

    messages = build_messages(context, "当前问题")

    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "system", "content": "对话摘要：用户预算 5000。"}
    assert messages[-1] == {"role": "user", "content": "当前问题"}
    assert [message["content"] for message in messages].count("当前问题") == 1
