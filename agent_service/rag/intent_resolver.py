from __future__ import annotations

import asyncio
from collections.abc import Iterable
from typing import Protocol

from agent_service.core.config import get_settings
from agent_service.rag.intent_models import (
    IntentCandidate,
    IntentGroup,
    NodeScore,
    SubQuestionIntent,
)
from agent_service.rag.schemas import RewriteResult


class IntentClassifierProtocol(Protocol):
    async def classify_targets(self, question: str) -> list[NodeScore]: ...


class IntentResolver:
    def __init__(
        self,
        classifier: IntentClassifierProtocol,
        *,
        min_score: float | None = None,
        max_intent_count: int | None = None,
    ) -> None:
        settings = get_settings()
        self._classifier = classifier
        self._min_score = (
            min_score if min_score is not None else settings.rag_intent_min_score
        )
        self._max_intent_count = (
            max_intent_count
            if max_intent_count is not None
            else settings.rag_max_intent_count
        )

    async def resolve(self, rewrite_result: RewriteResult) -> list[SubQuestionIntent]:
        sub_questions = rewrite_result.sub_questions or [rewrite_result.rewritten_question]
        tasks = [self._classify_intents(question) for question in sub_questions]
        results = await asyncio.gather(*tasks)
        sub_intents = [
            SubQuestionIntent(question, scores)
            for question, scores in zip(sub_questions, results, strict=True)
        ]
        return self.cap_total_intents(sub_intents)

    async def _classify_intents(self, question: str) -> list[NodeScore]:
        try:
            scores = await self._classifier.classify_targets(question)
        except Exception:
            return []
        return [
            score
            for score in sorted(scores, key=lambda item: item.score, reverse=True)
            if score.score >= self._min_score
        ][: self._max_intent_count]

    def cap_total_intents(self, sub_intents: list[SubQuestionIntent]) -> list[SubQuestionIntent]:
        total = sum(len(item.node_scores) for item in sub_intents)
        if total <= self._max_intent_count:
            return sub_intents

        all_candidates = _collect_candidates(sub_intents)
        guaranteed = _select_top_per_sub_question(all_candidates, len(sub_intents))
        remaining = self._max_intent_count - len(guaranteed)
        additional = _select_additional(all_candidates, guaranteed, remaining)
        selected = [*guaranteed, *additional]
        return _rebuild_sub_intents(sub_intents, selected)

    def merge_intent_group(
        self,
        sub_intents: Iterable[SubQuestionIntent | tuple[str, list[NodeScore]]],
    ) -> IntentGroup:
        kb_intents: list[NodeScore] = []
        mcp_intents: list[NodeScore] = []
        for item in sub_intents:
            scores = item.node_scores if isinstance(item, SubQuestionIntent) else item[1]
            for score in scores:
                if score.node.is_kb():
                    kb_intents.append(score)
                elif score.node.is_mcp():
                    mcp_intents.append(score)
        return IntentGroup(mcp_intents=mcp_intents, kb_intents=kb_intents)

    def is_system_only(
        self,
        sub_intents: Iterable[SubQuestionIntent | tuple[str, list[NodeScore]]],
    ) -> bool:
        seen = False
        for item in sub_intents:
            scores = item.node_scores if isinstance(item, SubQuestionIntent) else item[1]
            for score in scores:
                seen = True
                if not score.node.is_system():
                    return False
        return seen


def _collect_candidates(sub_intents: list[SubQuestionIntent]) -> list[IntentCandidate]:
    candidates: list[IntentCandidate] = []
    for index, sub_intent in enumerate(sub_intents):
        for node_score in sub_intent.node_scores:
            candidates.append(IntentCandidate(index, node_score))
    candidates.sort(key=lambda item: item.node_score.score, reverse=True)
    return candidates


def _select_top_per_sub_question(
    candidates: list[IntentCandidate],
    sub_question_count: int,
) -> list[IntentCandidate]:
    selected_flags = [False] * sub_question_count
    selected: list[IntentCandidate] = []
    for candidate in candidates:
        index = candidate.sub_question_index
        if selected_flags[index]:
            continue
        selected_flags[index] = True
        selected.append(candidate)
        if len(selected) == sub_question_count:
            break
    return selected


def _select_additional(
    candidates: list[IntentCandidate],
    selected: list[IntentCandidate],
    remaining: int,
) -> list[IntentCandidate]:
    if remaining <= 0:
        return []
    selected_ids = {
        (candidate.sub_question_index, candidate.node_score.node.id) for candidate in selected
    }
    additional: list[IntentCandidate] = []
    for candidate in candidates:
        key = (candidate.sub_question_index, candidate.node_score.node.id)
        if key in selected_ids:
            continue
        additional.append(candidate)
        if len(additional) >= remaining:
            break
    return additional


def _rebuild_sub_intents(
    original: list[SubQuestionIntent],
    selected: list[IntentCandidate],
) -> list[SubQuestionIntent]:
    grouped: dict[int, list[NodeScore]] = {}
    for candidate in selected:
        grouped.setdefault(candidate.sub_question_index, []).append(candidate.node_score)
    rebuilt: list[SubQuestionIntent] = []
    for index, sub_intent in enumerate(original):
        scores = sorted(grouped.get(index, []), key=lambda item: item.score, reverse=True)
        rebuilt.append(SubQuestionIntent(sub_intent.sub_question, scores))
    return rebuilt
