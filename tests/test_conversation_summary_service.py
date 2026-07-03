from __future__ import annotations

import pytest

from agent_service.memory.conversation_summary_service import ConversationSummaryService
from agent_service.memory.models import MemoryMessage
from agent_service.services.llm_service import LLMResult


class FakeLLMService:
    def __init__(self) -> None:
        self.messages: list[dict[str, str]] = []

    async def complete(self, messages: list[dict[str, str]]) -> LLMResult:
        self.messages = messages
        return LLMResult(reply="用户咨询过预算限制，预算为 5000。")


@pytest.mark.asyncio
async def test_summarize_merges_existing_summary_and_pending_messages() -> None:
    llm = FakeLLMService()
    service = ConversationSummaryService(llm_service=llm, max_chars=300)

    result = await service.summarize(
        existing_summary="用户咨询过笔记本采购。",
        pending_messages=[
            MemoryMessage(id="msg_2", role="user", content="预算是 5000"),
            MemoryMessage(id="msg_3", role="assistant", content="已记录预算。"),
        ],
    )

    assert result == "用户咨询过预算限制，预算为 5000。"
    assert llm.messages[0]["role"] == "system"
    assert "不要记录具体答案" in llm.messages[0]["content"]
    assert "预算是 5000" in llm.messages[1]["content"]
