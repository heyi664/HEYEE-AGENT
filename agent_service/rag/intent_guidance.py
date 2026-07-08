from __future__ import annotations

import re
from typing import Protocol

from agent_service.core.config import get_settings
from agent_service.rag.intent_models import (
    GuidanceDecision,
    IntentLevel,
    IntentNode,
    NodeScore,
    SubQuestionIntent,
)
from agent_service.rag.prompt_template_loader import PromptTemplateLoader

GUIDANCE_PROMPT = "guidance-prompt.st"


class AmbiguityCheckerProtocol(Protocol):
    async def check_ambiguity(self, question: str, ranked: list[NodeScore]) -> bool: ...


class IntentGuidanceService:
    def __init__(
        self,
        *,
        checker: AmbiguityCheckerProtocol,
        prompt_loader: PromptTemplateLoader | None = None,
        score_ratio: float | None = None,
        margin: float | None = None,
        max_options: int | None = None,
    ) -> None:
        settings = get_settings()
        self._checker = checker
        self._prompt_loader = prompt_loader or PromptTemplateLoader()
        self._score_ratio = (
            score_ratio if score_ratio is not None else settings.rag_guidance_score_ratio
        )
        self._margin = margin if margin is not None else settings.rag_guidance_margin
        self._max_options = (
            max_options if max_options is not None else settings.rag_guidance_max_options
        )

    async def detect_ambiguity(
        self,
        question: str,
        sub_intents: list[SubQuestionIntent],
    ) -> GuidanceDecision:
        if len(sub_intents) != 1:
            return GuidanceDecision.none()

        ranked = _rank_best_per_category(
            [score for score in sub_intents[0].node_scores if score.node.is_kb()]
        )
        if len(ranked) < 2:
            return GuidanceDecision.none()
        if _mentions_domain(question, ranked):
            return GuidanceDecision.none()

        top = ranked[0].score
        second = ranked[1].score
        if top <= 0:
            return GuidanceDecision.none()
        ratio = second / top
        if ratio < self._score_ratio - self._margin:
            return GuidanceDecision.none()
        if ratio < self._score_ratio:
            should_prompt = await self._checker.check_ambiguity(question, ranked)
            if not should_prompt:
                return GuidanceDecision.none()

        prompt = self._prompt_loader.render(
            GUIDANCE_PROMPT,
            {
                "topic_name": _topic_name(sub_intents[0]),
                "options": _render_options(ranked[: self._max_options]),
            },
        )
        return GuidanceDecision.prompt_user(prompt)


def _topic_name(sub_intent: SubQuestionIntent) -> str:
    for score in sub_intent.node_scores:
        if score.node.is_kb():
            return score.node.name
    return sub_intent.sub_question


def _rank_best_per_category(scores: list[NodeScore]) -> list[NodeScore]:
    best: dict[str, NodeScore] = {}
    for score in scores:
        category = _resolve_category_node(score.node)
        key = category.id
        current = best.get(key)
        if current is None or score.score > current.score:
            best[key] = NodeScore(category, score.score, score.reason)
    return sorted(best.values(), key=lambda item: item.score, reverse=True)


def _resolve_category_node(node: IntentNode) -> IntentNode:
    current = node
    while current.parent is not None:
        if current.level is IntentLevel.CATEGORY:
            return current
        current = current.parent
    return current


def _mentions_domain(question: str, ranked: list[NodeScore]) -> bool:
    normalized_question = _normalize(question)
    if not normalized_question:
        return False
    for score in ranked:
        domain = _resolve_domain_node(score.node)
        if domain is None:
            continue
        normalized_domain = _normalize(domain.name)
        if len(normalized_domain) >= 2 and normalized_domain in normalized_question:
            return True
    return False


def _resolve_domain_node(node: IntentNode) -> IntentNode | None:
    current: IntentNode | None = node
    while current is not None:
        if current.level is IntentLevel.DOMAIN:
            return current
        current = current.parent
    return None


def _normalize(value: str) -> str:
    return re.sub(r"[\W_]+", "", value.lower())


def _render_options(ranked: list[NodeScore]) -> str:
    lines: list[str] = []
    for index, score in enumerate(ranked, start=1):
        node = score.node
        display = node.full_path or node.name
        lines.append(f"{index}) {display}")
    return "\n".join(lines)
