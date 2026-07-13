from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol

from agent_service.core.observability import record_stage
from agent_service.memory.models import MemoryMessage
from agent_service.rag.ambiguity_checker import AmbiguityLLMChecker
from agent_service.rag.intent_classifier import IntentClassifier
from agent_service.rag.intent_guidance import IntentGuidanceService
from agent_service.rag.intent_models import (
    GuidanceDecision,
    IntentGroup,
    IntentNodeRecord,
    IntentTreeData,
    NodeScore,
    SubQuestionIntent,
)
from agent_service.rag.intent_resolver import IntentResolver
from agent_service.rag.intent_tree import build_intent_tree
from agent_service.rag.query_rewrite_service import QueryRewriteService
from agent_service.rag.schemas import RewriteResult
from agent_service.repositories.intent_node_repository import IntentNodeRepository
from agent_service.services.llm_service import get_llm_service


class RewriteServiceProtocol(Protocol):
    async def rewrite(
        self,
        question: str,
        history: list[MemoryMessage] | None = None,
    ) -> RewriteResult: ...


class IntentResolverProtocol(Protocol):
    async def resolve(self, rewrite_result: RewriteResult) -> list[SubQuestionIntent]: ...

    def merge_intent_group(self, sub_intents: Any) -> IntentGroup: ...

    def is_system_only(self, sub_intents: Any) -> bool: ...


class GuidanceServiceProtocol(Protocol):
    async def detect_ambiguity(
        self,
        question: str,
        sub_intents: list[SubQuestionIntent],
    ) -> GuidanceDecision: ...


@dataclass(frozen=True)
class IntentRecognitionResult:
    rewrite_result: RewriteResult
    sub_intents: list[SubQuestionIntent]
    guidance: GuidanceDecision
    kb_intents: list[NodeScore]
    mcp_intents: list[NodeScore]
    is_system_only: bool

    @property
    def original_question(self) -> str:
        return self.rewrite_result.original_question

    @property
    def rewritten_question(self) -> str:
        return self.rewrite_result.rewritten_question

    @property
    def sub_questions(self) -> list[str]:
        return self.rewrite_result.sub_questions

    def to_response_dict(self) -> dict[str, Any]:
        return {
            "originalQuestion": self.original_question,
            "rewrittenQuestion": self.rewritten_question,
            "subQuestions": self.sub_questions,
            "subIntents": [_sub_intent_to_dict(item) for item in self.sub_intents],
            "kbIntents": [_score_to_dict(score) for score in self.kb_intents],
            "mcpIntents": [_score_to_dict(score) for score in self.mcp_intents],
            "isSystemOnly": self.is_system_only,
            "guidance": {
                "action": self.guidance.action.name,
                "prompt": self.guidance.prompt,
            },
        }


class IntentRecognitionPipeline:
    def __init__(
        self,
        *,
        rewrite_service: RewriteServiceProtocol,
        intent_resolver: IntentResolverProtocol,
        guidance_service: GuidanceServiceProtocol,
    ) -> None:
        self._rewrite_service = rewrite_service
        self._intent_resolver = intent_resolver
        self._guidance_service = guidance_service

    async def recognize(
        self,
        question: str,
        history: list[MemoryMessage] | None = None,
    ) -> IntentRecognitionResult:
        started_at = time.perf_counter()
        rewrite_started_at = time.perf_counter()
        rewrite_result = await self._rewrite_service.rewrite(question, history)
        record_stage(
            "intent_rewrite",
            elapsed_ms=_elapsed_ms(rewrite_started_at),
            subQuestionCount=len(rewrite_result.sub_questions),
        )
        resolve_started_at = time.perf_counter()
        sub_intents = await self._intent_resolver.resolve(rewrite_result)
        record_stage(
            "intent_resolution",
            elapsed_ms=_elapsed_ms(resolve_started_at),
            subIntentCount=sum(len(item.node_scores) for item in sub_intents),
            subQuestionCount=len(sub_intents),
        )
        guidance_started_at = time.perf_counter()
        guidance = await self._guidance_service.detect_ambiguity(
            rewrite_result.rewritten_question,
            sub_intents,
        )
        record_stage(
            "intent_guidance",
            elapsed_ms=_elapsed_ms(guidance_started_at),
            action=guidance.action.name,
        )
        group = self._intent_resolver.merge_intent_group(sub_intents)
        result = IntentRecognitionResult(
            rewrite_result=rewrite_result,
            sub_intents=sub_intents,
            guidance=guidance,
            kb_intents=group.kb_intents,
            mcp_intents=group.mcp_intents,
            is_system_only=self._intent_resolver.is_system_only(sub_intents),
        )
        record_stage(
            "intent_pipeline",
            elapsed_ms=_elapsed_ms(started_at),
            kbIntentCount=len(result.kb_intents),
            mcpIntentCount=len(result.mcp_intents),
            systemOnly=result.is_system_only,
        )
        return result


class RepositoryIntentTreeSource:
    def __init__(self, repository: IntentNodeRepository | None = None) -> None:
        self._repository = repository or IntentNodeRepository()
        self._data: IntentTreeData | None = None

    def load(self) -> IntentTreeData:
        if self._data is None:
            records: list[IntentNodeRecord] = self._repository.list_enabled_nodes()
            self._data = build_intent_tree(records)
        return self._data


def get_intent_recognition_pipeline() -> IntentRecognitionPipeline:
    llm_service = get_llm_service()
    tree_source = RepositoryIntentTreeSource()
    classifier = IntentClassifier(tree_source, llm_service=llm_service)
    resolver = IntentResolver(classifier)
    guidance = IntentGuidanceService(
        checker=AmbiguityLLMChecker(llm_service=llm_service),
    )
    return IntentRecognitionPipeline(
        rewrite_service=QueryRewriteService(llm_service=llm_service),
        intent_resolver=resolver,
        guidance_service=guidance,
    )


def _sub_intent_to_dict(sub_intent: SubQuestionIntent) -> dict[str, Any]:
    return {
        "subQuestion": sub_intent.sub_question,
        "intents": [_score_to_dict(score) for score in sub_intent.node_scores],
    }


def _score_to_dict(score: NodeScore) -> dict[str, Any]:
    node = score.node
    return {
        "id": node.id,
        "name": node.name,
        "kind": node.kind.name,
        "level": node.level.name,
        "score": score.score,
        "reason": score.reason,
        "fullPath": node.full_path,
        "collectionName": node.collection_name,
        "mcpToolId": node.mcp_tool_id,
        "topK": node.top_k,
    }


def _elapsed_ms(started_at: float) -> int:
    return int((time.perf_counter() - started_at) * 1000)
