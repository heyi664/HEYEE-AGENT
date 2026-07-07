from __future__ import annotations

import pytest

from agent_service.memory.models import MemoryMessage
from agent_service.rag.query_rewrite_service import QueryRewriteService
from agent_service.services.llm_service import LLMResult


class FakeLLMService:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.messages: list[dict[str, str]] = []

    async def complete(self, messages: list[dict[str, str]]) -> LLMResult:
        self.messages = messages
        return LLMResult(reply=self.reply)


@pytest.mark.asyncio
async def test_rewrite_falls_back_to_normalized_rule_split_when_llm_disabled() -> None:
    service = QueryRewriteService(
        llm_service=None,
        term_mappings={"苹果手机": "iPhone"},
        enabled=False,
    )

    result = await service.rewrite("苹果手机怎么保修？换电池多少钱？")

    assert result.original_question == "苹果手机怎么保修？换电池多少钱？"
    assert result.rewritten_question == "iPhone怎么保修？换电池多少钱？"
    assert result.sub_questions == ["iPhone怎么保修？", "换电池多少钱？"]


@pytest.mark.asyncio
async def test_rewrite_uses_llm_json_result() -> None:
    llm = FakeLLMService(
        '{"rewritten_question": "iPhone 保修多久？", "sub_questions": ["iPhone 保修多久？"]}'
    )
    service = QueryRewriteService(llm_service=llm, enabled=True)

    result = await service.rewrite("那它保修多久？")

    assert result.rewritten_question == "iPhone 保修多久？"
    assert result.sub_questions == ["iPhone 保修多久？"]



@pytest.mark.asyncio
async def test_rewrite_accepts_ragent_main_rewrite_field() -> None:
    llm = FakeLLMService(
        '{"rewrite": "iPhone 保修多久？", "should_split": false, '
        '"sub_questions": ["iPhone 保修多久？"]}'
    )
    service = QueryRewriteService(llm_service=llm, enabled=True)

    result = await service.rewrite("那它保修多久？")

    assert result.rewritten_question == "iPhone 保修多久？"
    assert result.sub_questions == ["iPhone 保修多久？"]


@pytest.mark.asyncio
async def test_rewrite_forces_single_sub_question_when_should_split_is_false() -> None:
    llm = FakeLLMService(
        '{"rewrite": "iPhone 保修多久？", "should_split": false, '
        '"sub_questions": ["错误子问题 A", "错误子问题 B"]}'
    )
    service = QueryRewriteService(llm_service=llm, enabled=True)

    result = await service.rewrite("那它保修多久？")

    assert result.rewritten_question == "iPhone 保修多久？"
    assert result.sub_questions == ["iPhone 保修多久？"]
@pytest.mark.asyncio
async def test_rewrite_includes_recent_history_in_prompt() -> None:
    llm = FakeLLMService(
        '{"rewritten_question": "iPhone 保修多久？", "sub_questions": ["iPhone 保修多久？"]}'
    )
    service = QueryRewriteService(llm_service=llm, enabled=True, history_turns=1)

    await service.rewrite(
        "那它保修多久？",
        history=[
            MemoryMessage(id="m1", role="user", content="我想买 iPhone"),
            MemoryMessage(id="m2", role="assistant", content="可以看看 iPhone 15"),
            MemoryMessage(id="m3", role="user", content="它支持无线充电吗？"),
            MemoryMessage(id="m4", role="assistant", content="支持"),
        ],
    )

    prompt_text = "\n".join(message["content"] for message in llm.messages)
    assert "它支持无线充电吗？" in prompt_text
    assert "我想买 iPhone" not in prompt_text



@pytest.mark.asyncio
async def test_rewrite_trims_recent_history_by_message_and_total_budget() -> None:
    llm = FakeLLMService(
        '{"rewritten_question": "iPhone 保修多久？", "sub_questions": ["iPhone 保修多久？"]}'
    )
    service = QueryRewriteService(
        llm_service=llm,
        enabled=True,
        history_turns=2,
        history_message_max_chars=13,
        history_max_chars=40,
    )

    await service.rewrite(
        "那它保修多久？",
        history=[
            MemoryMessage(id="m1", role="user", content="第一轮很久以前的问题"),
            MemoryMessage(id="m2", role="assistant", content="第一轮很久以前的回答"),
            MemoryMessage(id="m3", role="user", content="iPhone 15 Pro Max 是否支持无线充电？"),
            MemoryMessage(
                id="m4",
                role="assistant",
                content="支持无线充电，而且回答内容非常非常长",
            ),
        ],
    )

    prompt_text = "\n".join(message["content"] for message in llm.messages)
    assert "第一轮很久以前" not in prompt_text
    assert "iPhone 15 Pro" in prompt_text
    assert "支持无线充电，而且回答内容非常非常长" not in prompt_text
@pytest.mark.asyncio
async def test_rewrite_falls_back_when_llm_returns_invalid_json() -> None:
    llm = FakeLLMService("not json")
    service = QueryRewriteService(
        llm_service=llm,
        term_mappings={"苹果手机": "iPhone"},
        enabled=True,
    )

    result = await service.rewrite("苹果手机怎么保修？换电池多少钱？")

    assert result.rewritten_question == "iPhone怎么保修？换电池多少钱？"
    assert result.sub_questions == ["iPhone怎么保修？", "换电池多少钱？"]


@pytest.mark.asyncio
async def test_rewrite_limits_sub_questions() -> None:
    llm = FakeLLMService(
        '{"rewritten_question": "A？B？C？", "sub_questions": ["A？", "B？", "C？"]}'
    )
    service = QueryRewriteService(llm_service=llm, enabled=True, max_sub_questions=2)

    result = await service.rewrite("A？B？C？")

    assert result.sub_questions == ["A？", "B？"]
