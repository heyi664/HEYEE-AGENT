from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ConversationListItem(BaseModel):
    conversationId: str
    userId: str
    title: str
    lastTime: datetime | None = None
    createTime: datetime | None = None
    updateTime: datetime | None = None


class ConversationMessageItem(BaseModel):
    id: str
    role: Literal["user", "assistant"]
    content: str
    createTime: datetime | None = None


class ConversationSummaryItem(BaseModel):
    id: str
    lastMessageId: str
    content: str
    updateTime: datetime | None = None


class ConversationMessagesResponse(BaseModel):
    conversationId: str
    userId: str
    summary: ConversationSummaryItem | None = None
    messages: list[ConversationMessageItem] = Field(default_factory=list)


class ConversationDeleteResponse(BaseModel):
    conversationId: str
    userId: str
    deleted: bool = True