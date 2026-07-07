from __future__ import annotations

import logging
from typing import Any, cast

from agent_service.core.config import get_settings
from agent_service.memory.models import MemoryMessage
from agent_service.rag.llm_response_cleaner import parse_json_object
from agent_service.rag.query_splitter import split_query
from agent_service.rag.query_term_mapping import QueryTermMappingService
from agent_service.rag.schemas import RewriteResult
from agent_service.services.llm_service import LLMService, get_llm_service

logger = logging.getLogger(__name__)


class QueryRewriteService:
    def __init__(
        self,
        llm_service: LLMService | object | None = None,
        term_mapping_service: QueryTermMappingService | None = None,
        term_mappings: dict[str, str] | None = None,
        enabled: bool | None = None,
        history_turns: int | None = None,
        max_sub_questions: int | None = None,
        history_max_chars: int | None = None,
        history_message_max_chars: int | None = None,
    ) -> None:
        settings = get_settings()
        self._llm_service: Any = llm_service
        self._term_mapping_service = term_mapping_service or QueryTermMappingService(
            mappings=term_mappings
        )
        self._enabled = settings.rag_query_rewrite_enabled if enabled is None else enabled
        self._history_turns = history_turns or settings.rag_query_rewrite_history_turns
        self._max_sub_questions = max_sub_questions or settings.rag_query_rewrite_max_sub_questions
        self._history_max_chars = history_max_chars or settings.rag_query_rewrite_history_max_chars
        self._history_message_max_chars = (
            history_message_max_chars or settings.rag_query_rewrite_history_message_max_chars
        )

    async def rewrite(
        self,
        question: str,
        history: list[MemoryMessage] | None = None,
    ) -> RewriteResult:
        fallback = self._fallback_result(question)
        if not self._enabled or self._llm_service is None:
            return fallback

        try:
            result = await self._llm_service.complete(self._build_messages(question, history or []))
            parsed = parse_json_object(result.reply)
            rewritten_question = str(
                parsed.get("rewritten_question")
                or parsed.get("rewrite")
                or fallback.rewritten_question
            ).strip()
            sub_questions = self._sub_questions_from_response(parsed, rewritten_question)
            return RewriteResult(
                original_question=question,
                rewritten_question=rewritten_question,
                sub_questions=sub_questions[: self._max_sub_questions],
            )
        except Exception:
            logger.exception("rag query rewrite failed; fallback to rule split")
            return fallback

    def _fallback_result(self, question: str) -> RewriteResult:
        normalized = self._term_mapping_service.normalize(question)
        return RewriteResult(
            original_question=question,
            rewritten_question=normalized,
            sub_questions=split_query(normalized)[: self._max_sub_questions],
        )

    def _build_messages(
        self,
        question: str,
        history: list[MemoryMessage],
    ) -> list[dict[str, str]]:
        messages = [
            {
                "role": "system",
                "content": self._system_prompt(),
            }
        ]
        recent_history = self._latest_history_messages(history)
        if recent_history:
            history_text = "\n".join(
                f"{message.role}: {message.content}" for message in recent_history
            )
            messages.append({"role": "system", "content": f"最近对话：\n{history_text}"})
        messages.append({"role": "user", "content": question})
        return messages

    def _latest_history_messages(self, history: list[MemoryMessage]) -> list[MemoryMessage]:
        if self._history_turns <= 0 or self._history_max_chars <= 0:
            return []

        user_seen = 0
        selected: list[MemoryMessage] = []
        for message in reversed(history):
            selected.append(message)
            if message.role == "user":
                user_seen += 1
                if user_seen >= self._history_turns:
                    break

        budgeted: list[MemoryMessage] = []
        total_chars = 0
        for message in selected:
            content = self._trim_message_content(message.content)
            if not content:
                continue
            projected = total_chars + len(message.role) + len(content)
            if budgeted and projected > self._history_max_chars:
                continue
            if projected > self._history_max_chars:
                content = content[: max(0, self._history_max_chars - len(message.role))]
                if not content:
                    continue
            budgeted.append(MemoryMessage(id=message.id, role=message.role, content=content))
            total_chars += len(message.role) + len(content)
        return list(reversed(budgeted))

    def _trim_message_content(self, content: str) -> str:
        stripped = content.strip()
        if len(stripped) <= self._history_message_max_chars:
            return stripped
        return stripped[: self._history_message_max_chars]

    def _sub_questions_from_response(
        self,
        parsed: dict[str, Any],
        rewritten_question: str,
    ) -> list[str]:
        should_split = parsed.get("should_split")
        if should_split is False:
            return [rewritten_question]
        raw_sub_questions = parsed.get("sub_questions")
        sub_questions = self._coerce_sub_questions(raw_sub_questions)
        if not sub_questions:
            sub_questions = split_query(rewritten_question)
        return sub_questions

    def _coerce_sub_questions(self, value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in cast(list[object], value) if str(item).strip()]

    def _system_prompt(self) -> str:
        return """
# 角色
你是查询改写助手，用于 RAG 检索阶段。

# 任务
1. 将用户问题改写成适合检索的自然语言查询
2. 判断是否需要拆分成多个子问题

# 输出格式
严格返回 JSON，不要额外文字。兼容字段如下：
{
  "rewritten_question": "改写后的查询",
  "should_split": true,
  "sub_questions": ["子问题1", "子问题2"]
}

# 改写规则
- 保留专有名词、系统名、产品名、模块名，不得修改写法
- 保留时间范围、环境、终端类型、角色身份等限制条件
- 删除礼貌用语，如“请帮我”“麻烦”“谢谢”
- 删除回答指令，如“详细说明”“分点回答”“一步步分析”
- 不得添加原文没有的条件、维度、假设
- 指代词如“它”“这个”“刚才说的”，结合最近历史消息还原具体实体

# 拆分规则
只在以下情况拆分：多个问号、显式列举、分号或换行分隔、明确要求分别回答。
以下情况不拆分：抽象对比、笼统询问、不确定是否需要拆分。

# 一致性约束
- 如果 should_split=false：sub_questions 必须只有一条，且等于 rewritten_question
- 如果 should_split=true：每个子问题必须是完整、可独立检索的问题
""".strip()


def get_query_rewrite_service() -> QueryRewriteService:
    return QueryRewriteService(llm_service=get_llm_service())
