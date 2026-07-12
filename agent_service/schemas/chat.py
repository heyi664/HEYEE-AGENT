from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ChatHistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    userId: int | None = None
    conversationId: str | None = Field(default=None, max_length=20)
    message: str
    history: list[ChatHistoryItem] = Field(default_factory=list)

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("message must not be blank")
        return value.strip()


class RagIntentScore(BaseModel):
    id: str
    name: str
    kind: str
    level: str
    score: float
    reason: str | None = None
    fullPath: str = ""
    collectionName: str | None = None
    mcpToolId: str | None = None
    topK: int | None = None


class RagSubIntent(BaseModel):
    subQuestion: str
    intents: list[RagIntentScore] = Field(default_factory=list)


class RagGuidanceResult(BaseModel):
    action: str
    prompt: str | None = None


class RagIntentRecognitionResult(BaseModel):
    originalQuestion: str
    rewrittenQuestion: str
    subQuestions: list[str] = Field(default_factory=list)
    subIntents: list[RagSubIntent] = Field(default_factory=list)
    kbIntents: list[RagIntentScore] = Field(default_factory=list)
    mcpIntents: list[RagIntentScore] = Field(default_factory=list)
    isSystemOnly: bool = False
    guidance: RagGuidanceResult | None = None


class ChatSource(BaseModel):
    id: str | None = None
    title: str
    content: str
    score: float | None = None
    sourceType: str | None = None
    url: str | None = None
    collectionName: str | None = None
    channel: str | None = None


class ChatResponse(BaseModel):
    conversationId: str
    reply: str = Field(min_length=1)
    createdAt: str = Field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    sources: list[ChatSource] = Field(default_factory=list)
    toolCalls: list[str] = Field(default_factory=list)
    ragIntent: RagIntentRecognitionResult | None = None
