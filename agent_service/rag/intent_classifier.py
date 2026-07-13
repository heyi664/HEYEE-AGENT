from __future__ import annotations

import time
from typing import Protocol

from agent_service.core.observability import record_stage
from agent_service.rag.intent_models import IntentKind, IntentNode, IntentTreeData, NodeScore
from agent_service.rag.llm_response_cleaner import parse_json_array
from agent_service.rag.prompt_template_loader import PromptTemplateLoader
from agent_service.services.llm_service import LLMService

INTENT_CLASSIFIER_PROMPT = "intent-classifier.st"


class IntentTreeSource(Protocol):
    def load(self) -> IntentTreeData: ...


class IntentClassifier:
    def __init__(
        self,
        tree_source: IntentTreeSource,
        *,
        llm_service: LLMService,
        prompt_loader: PromptTemplateLoader | None = None,
    ) -> None:
        self._tree_source = tree_source
        self._llm_service = llm_service
        self._prompt_loader = prompt_loader or PromptTemplateLoader()

    async def classify_targets(self, question: str) -> list[NodeScore]:
        tree = self._tree_source.load()
        prompt = self._prompt_loader.render(
            INTENT_CLASSIFIER_PROMPT,
            {"intent_list": _serialize_leaf_nodes(tree.leaf_nodes)},
        )
        started_at = time.perf_counter()
        try:
            result = await self._llm_service.complete(
                [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": question},
                ],
                use_tools=False,
            )
        except Exception:
            record_stage(
                "intent_classification_llm",
                elapsed_ms=_elapsed_ms(started_at),
                candidateNodeCount=len(tree.leaf_nodes),
                intentCount=0,
                status="failed",
            )
            raise
        scores = _parse_scores(result.reply, tree)
        scores.sort(key=lambda item: item.score, reverse=True)
        record_stage(
            "intent_classification_llm",
            elapsed_ms=_elapsed_ms(started_at),
            candidateNodeCount=len(tree.leaf_nodes),
            intentCount=len(scores),
            status="success",
        )
        return scores


def _serialize_leaf_nodes(nodes: list[IntentNode]) -> str:
    parts: list[str] = []
    for node in nodes:
        lines = [
            f"- id={node.id}",
            f"  path={node.full_path or node.name}",
            f"  description={node.description or ''}",
            f"  type={_kind_label(node.kind)}",
        ]
        if node.mcp_tool_id:
            lines.append(f"  toolId={node.mcp_tool_id}")
        if node.examples:
            lines.append("  examples=" + " / ".join(node.examples))
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def _kind_label(kind: IntentKind) -> str:
    return kind.name


def _parse_scores(reply: str, tree: IntentTreeData) -> list[NodeScore]:
    items = parse_json_array(reply)
    scores: list[NodeScore] = []
    for item in items:
        node_id = item.get("id")
        score = item.get("score")
        if node_id is None or score is None:
            continue
        node = tree.id_to_node.get(str(node_id))
        if node is None:
            continue
        try:
            value = float(score)
        except (TypeError, ValueError):
            continue
        reason = item.get("reason")
        scores.append(NodeScore(node=node, score=value, reason=str(reason) if reason else None))
    return scores


def _elapsed_ms(started_at: float) -> int:
    return int((time.perf_counter() - started_at) * 1000)
