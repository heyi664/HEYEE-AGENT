from __future__ import annotations

from agent_service.schemas.chat import ChatHistoryItem

SYSTEM_PROMPT = (
    "你是 HYEEE AI，一个面向本地生活点评、店铺推荐和用户问答场景的助手。"
    "回答要简洁、自然、可靠；如果信息不足，先说明需要哪些补充信息。"
)


def build_messages(history: list[ChatHistoryItem], message: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for item in history[-10:]:
        messages.append({"role": item.role, "content": item.content})
    messages.append({"role": "user", "content": message})
    return messages

