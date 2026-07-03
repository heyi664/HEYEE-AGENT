from __future__ import annotations

from agent_service.memory.models import MemoryMessage
from agent_service.services.llm_service import LLMService, get_llm_service


class ConversationSummaryService:
    def __init__(
        self,
        llm_service: LLMService | object | None = None,
        max_chars: int = 300,
    ) -> None:
        self._llm_service = llm_service or get_llm_service()
        self._max_chars = max_chars

    async def summarize(
        self,
        *,
        existing_summary: str | None,
        pending_messages: list[MemoryMessage],
    ) -> str:
        transcript = "\n".join(
            f"{message.role.upper()}: {message.content}" for message in pending_messages
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "你是会话记忆摘要器。只记录用户讨论过的话题、状态、偏好、限制和关键约束。"
                    "不要记录具体答案，不要把历史摘要当作事实新增来源。"
                    f"输出不超过 {self._max_chars} 个中文字符。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"已有历史摘要：\n{existing_summary or '无'}\n\n"
                    f"本次待合并对话：\n{transcript}\n\n"
                    "请合并去重，输出新的会话摘要。"
                ),
            },
        ]
        result = await self._llm_service.complete(messages)
        return result.reply.strip()

