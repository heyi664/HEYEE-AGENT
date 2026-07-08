from __future__ import annotations

import pytest

from agent_service.rag.intent_guidance import IntentGuidanceService
from agent_service.rag.intent_models import (
    IntentKind,
    IntentLevel,
    IntentNode,
    NodeScore,
    SubQuestionIntent,
)


class FakeChecker:
    def __init__(self, result: bool) -> None:
        self.result = result
        self.called = False

    async def check_ambiguity(self, question: str, ranked: list[NodeScore]) -> bool:
        self.called = True
        return self.result


def category_with_topic(category_id: str, topic_id: str, score: float) -> NodeScore:
    domain = IntentNode(id="product", name="商品服务", level=IntentLevel.DOMAIN)
    category = IntentNode(id=category_id, name=category_id, level=IntentLevel.CATEGORY)
    topic = IntentNode(
        id=topic_id,
        name="退换政策",
        level=IntentLevel.TOPIC,
        kind=IntentKind.KB,
    )
    domain.children = [category]
    category.parent = domain
    category.children = [topic]
    topic.parent = category
    category.full_path = f"商品服务 > {category_id}"
    topic.full_path = f"商品服务 > {category_id} > 退换政策"
    return NodeScore(node=topic, score=score)


@pytest.mark.asyncio
async def test_guidance_prompts_when_same_topic_spans_categories() -> None:
    service = IntentGuidanceService(checker=FakeChecker(True), score_ratio=0.8, margin=0.15)
    sub_intents = [
        SubQuestionIntent(
            "退换政策是什么？",
            [
                category_with_topic("3C 数码", "3c-return", 0.88),
                category_with_topic("家电", "appliance-return", 0.76),
            ],
        )
    ]

    decision = await service.detect_ambiguity("退换政策是什么？", sub_intents)

    assert decision.is_prompt()
    assert "退换政策" in decision.prompt
    assert "3C 数码" in decision.prompt
    assert "家电" in decision.prompt


@pytest.mark.asyncio
async def test_guidance_skips_when_user_mentions_domain_name() -> None:
    checker = FakeChecker(True)
    service = IntentGuidanceService(checker=checker, score_ratio=0.8, margin=0.15)
    sub_intents = [
        SubQuestionIntent(
            "商品服务退换政策是什么？",
            [
                category_with_topic("3C 数码", "3c-return", 0.88),
                category_with_topic("家电", "appliance-return", 0.86),
            ],
        )
    ]

    decision = await service.detect_ambiguity("商品服务退换政策是什么？", sub_intents)

    assert not decision.is_prompt()
    assert checker.called is False


@pytest.mark.asyncio
async def test_guidance_skips_when_top_score_clearly_leads() -> None:
    checker = FakeChecker(True)
    service = IntentGuidanceService(checker=checker, score_ratio=0.8, margin=0.15)
    sub_intents = [
        SubQuestionIntent(
            "退换政策是什么？",
            [
                category_with_topic("3C 数码", "3c-return", 0.9),
                category_with_topic("家电", "appliance-return", 0.5),
            ],
        )
    ]

    decision = await service.detect_ambiguity("退换政策是什么？", sub_intents)

    assert not decision.is_prompt()
    assert checker.called is False


@pytest.mark.asyncio
async def test_guidance_uses_checker_in_gray_zone() -> None:
    checker = FakeChecker(False)
    service = IntentGuidanceService(checker=checker, score_ratio=0.8, margin=0.15)
    sub_intents = [
        SubQuestionIntent(
            "退换政策是什么？",
            [
                category_with_topic("3C 数码", "3c-return", 0.9),
                category_with_topic("家电", "appliance-return", 0.62),
            ],
        )
    ]

    decision = await service.detect_ambiguity("退换政策是什么？", sub_intents)

    assert checker.called is True
    assert not decision.is_prompt()
