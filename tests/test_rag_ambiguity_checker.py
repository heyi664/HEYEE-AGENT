from __future__ import annotations

import pytest

from agent_service.rag.ambiguity_checker import AmbiguityLLMChecker
from agent_service.rag.intent_models import IntentLevel, IntentNode, NodeScore
from agent_service.services.llm_service import LLMResult


class FakeLLM:
    def __init__(self, reply: str | Exception) -> None:
        self.reply = reply

    async def complete(self, messages: list[dict[str, str]]) -> LLMResult:
        if isinstance(self.reply, Exception):
            raise self.reply
        return LLMResult(reply=self.reply)


def score(node_id: str, value: float) -> NodeScore:
    return NodeScore(
        node=IntentNode(id=node_id, name=node_id, level=IntentLevel.CATEGORY),
        score=value,
    )


@pytest.mark.asyncio
async def test_ambiguity_checker_parses_boolean_response() -> None:
    checker = AmbiguityLLMChecker(llm_service=FakeLLM('{"ambiguous": false}'))

    assert await checker.check_ambiguity("退换政策是什么？", [score("3c", 0.7)]) is False


@pytest.mark.asyncio
async def test_ambiguity_checker_fails_closed_to_prompt() -> None:
    checker = AmbiguityLLMChecker(llm_service=FakeLLM(RuntimeError("offline")))

    assert await checker.check_ambiguity("退换政策是什么？", [score("3c", 0.7)]) is True
