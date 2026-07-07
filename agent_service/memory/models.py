from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

MemoryRole = Literal["user", "assistant"]


@dataclass(frozen=True)
class MemoryMessage:
    id: str
    role: MemoryRole
    content: str
    create_time: datetime | None = None


@dataclass(frozen=True)
class MemorySummary:
    id: str
    conversation_id: str
    user_id: str
    last_message_id: str
    content: str


@dataclass(frozen=True)
class ConversationRecord:
    conversation_id: str
    title: str | None = None
    last_time: datetime | None = None
    update_time: datetime | None = None


@dataclass(frozen=True)
class MemoryContext:
    summary: MemorySummary | None = None
    messages: list[MemoryMessage] = field(default_factory=list)

    def to_prompt_messages(self) -> list[dict[str, str]]:
        prompt_messages: list[dict[str, str]] = []
        if self.summary and self.summary.content.strip():
            prompt_messages.append(
                {"role": "system", "content": f"对话摘要：{self.summary.content.strip()}"}
            )
        for message in self.messages:
            prompt_messages.append({"role": message.role, "content": message.content})
        return prompt_messages
