from __future__ import annotations

from agent_service.memory.models import MemoryContext

SYSTEM_PROMPT = (
    "你是 HYEEE AI，一个面向本地生活点评、店铺推荐和用户问答场景的助手。"
    "回答要简洁、自然、可靠；如果信息不足，先说明需要哪些补充信息。"
)


def build_messages(memory_context: MemoryContext, message: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(memory_context.to_prompt_messages())
    messages.append({"role": "user", "content": message})
    return messages
