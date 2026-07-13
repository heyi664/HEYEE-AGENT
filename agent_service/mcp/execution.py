from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

from agent_service.core.observability import record_stage
from agent_service.mcp.parameter_extractor import McpParameterExtractor
from agent_service.rag.intent_models import IntentKind, SubQuestionIntent
from agent_service.tools.registry import ToolRegistry, tool_registry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class McpToolExecution:
    tool_id: str
    sub_question: str
    status: str
    content: str = ""
    missing_required: tuple[str, ...] = ()


@dataclass(frozen=True)
class McpExecutionResult:
    executions: list[McpToolExecution]

    @classmethod
    def empty(cls) -> McpExecutionResult:
        return cls([])

    @property
    def tool_calls(self) -> list[str]:
        return [item.tool_id for item in self.executions if item.status == "success"]

    @property
    def context(self) -> str:
        fragments = [_format_execution(item) for item in self.executions]
        return "\n\n".join(fragment for fragment in fragments if fragment).strip()


class McpExecutionService:
    """Run MCP intents through discovered tools with Schema-bound arguments."""

    def __init__(
        self,
        *,
        registry: ToolRegistry | None = None,
        parameter_extractor: McpParameterExtractor,
        max_context_chars: int = 6000,
    ) -> None:
        self._registry = registry or tool_registry
        self._parameter_extractor = parameter_extractor
        self._max_context_chars = max_context_chars

    async def execute(self, sub_intents: list[SubQuestionIntent]) -> McpExecutionResult:
        requests = [
            (sub_intent.sub_question, score.node)
            for sub_intent in sub_intents
            for score in sub_intent.node_scores
            if score.node.kind is IntentKind.MCP and score.node.mcp_tool_id
        ]
        if not requests:
            return McpExecutionResult.empty()

        raw_results = await asyncio.gather(
            *(self._execute_one(question, node) for question, node in requests),
            return_exceptions=True,
        )
        executions: list[McpToolExecution] = []
        for (question, node), result in zip(requests, raw_results, strict=True):
            if isinstance(result, Exception):
                logger.error(
                    "unexpected MCP execution failure tool=%s error=%s",
                    node.mcp_tool_id,
                    result,
                )
                executions.append(
                    McpToolExecution(
                        tool_id=str(node.mcp_tool_id),
                        sub_question=question,
                        status="failed",
                    )
                )
            else:
                executions.append(result)
        return McpExecutionResult(executions)

    async def _execute_one(self, question: str, node: Any) -> McpToolExecution:
        tool_id = str(node.mcp_tool_id)
        tool = self._registry.get(tool_id)
        if tool is None:
            return McpToolExecution(
                tool_id=tool_id,
                sub_question=question,
                status="unavailable",
            )

        started_at = time.perf_counter()
        extracted = await self._parameter_extractor.extract(
            question,
            tool,
            getattr(node, "param_prompt_template", None),
        )
        if extracted.invalid_fields:
            record_stage(
                "mcp_parameter_extraction",
                elapsed_ms=_elapsed_ms(started_at),
                toolId=tool_id,
                status="invalid",
            )
            return McpToolExecution(tool_id=tool_id, sub_question=question, status="invalid")
        if extracted.missing_required:
            record_stage(
                "mcp_parameter_extraction",
                elapsed_ms=_elapsed_ms(started_at),
                toolId=tool_id,
                status="needs_clarification",
            )
            return McpToolExecution(
                tool_id=tool_id,
                sub_question=question,
                status="needs_clarification",
                missing_required=tuple(extracted.missing_required),
            )

        record_stage(
            "mcp_parameter_extraction",
            elapsed_ms=_elapsed_ms(started_at),
            toolId=tool_id,
            status="fallback" if extracted.fallback_used else "success",
        )
        call_started_at = time.perf_counter()
        try:
            content = await tool.handler(extracted.arguments)
        except Exception:
            logger.exception("MCP tool invocation failed tool=%s", tool_id)
            record_stage(
                "mcp_tool_call",
                elapsed_ms=_elapsed_ms(call_started_at),
                toolId=tool_id,
                status="failed",
            )
            return McpToolExecution(tool_id=tool_id, sub_question=question, status="failed")

        record_stage(
            "mcp_tool_call",
            elapsed_ms=_elapsed_ms(call_started_at),
            toolId=tool_id,
            status="success",
        )
        return McpToolExecution(
            tool_id=tool_id,
            sub_question=question,
            status="success",
            content=content[: self._max_context_chars],
        )


def _format_execution(item: McpToolExecution) -> str:
    if item.status == "success":
        return (
            f"[MCP tool: {item.tool_id}]\n"
            f"Question: {item.sub_question}\n"
            f"Real-time result:\n{item.content}"
        )
    if item.status == "needs_clarification":
        missing = ", ".join(item.missing_required)
        return (
            f"[MCP tool: {item.tool_id}]\n"
            f"The real-time query cannot run because required parameters are missing: {missing}. "
            "Ask the user only for the missing information; do not guess or perform a broad query."
        )
    if item.status == "unavailable":
        return (
            f"[MCP tool: {item.tool_id}]\n"
            "The real-time data service is not configured or this tool is unavailable. "
            "Tell the user that real-time data cannot currently be obtained; do not fabricate it."
        )
    if item.status == "invalid":
        return (
            f"[MCP tool: {item.tool_id}]\n"
            "The request parameters could not be validated. Ask the user to clarify the query; "
            "do not call the tool with guessed values."
        )
    return (
        f"[MCP tool: {item.tool_id}]\n"
        "The real-time data service did not return a usable result. "
        "State that the real-time query is temporarily unavailable; do not fabricate data."
    )


def _elapsed_ms(started_at: float) -> int:
    return int((time.perf_counter() - started_at) * 1000)
