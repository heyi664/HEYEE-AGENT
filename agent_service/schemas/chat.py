from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ChatHistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    userId: int | None = None
    conversationId: str | None = None
    message: str
    history: list[ChatHistoryItem] = Field(default_factory=list)

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("message must not be blank")
        return value.strip()


class ChatResponse(BaseModel):
    conversationId: str
    reply: str = Field(min_length=1)
    createdAt: str = Field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    sources: list[str] = Field(default_factory=list)
    toolCalls: list[str] = Field(default_factory=list)

