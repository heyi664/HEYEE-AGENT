from __future__ import annotations

from datetime import datetime
from typing import Literal

from fastapi import APIRouter, Depends, Header, HTTPException, Path, Query, Response, status
from pydantic import BaseModel

from agent_service.memory.conversation_memory_service import ConversationMemoryService

router = APIRouter(tags=["conversation-memory"])


class ConversationSummaryResponse(BaseModel):
    conversationId: str
    title: str | None = None
    lastTime: datetime | None = None
    updateTime: datetime | None = None


class ConversationMessageResponse(BaseModel):
    id: str
    role: Literal["user", "assistant"]
    content: str
    createTime: datetime | None = None


def get_conversation_memory_service() -> ConversationMemoryService:
    return ConversationMemoryService()


def resolve_request_user_id(
    userId: str = Query(..., min_length=1),
    authenticated_user_id: str | None = Header(default=None, alias="X-User-Id"),
) -> str:
    if authenticated_user_id is not None and authenticated_user_id != userId:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")
    return userId


@router.get("/conversations", response_model=list[ConversationSummaryResponse])
def list_conversations(
    user_id: str = Depends(resolve_request_user_id),
    limit: int = Query(default=50, ge=1, le=200),
    service: ConversationMemoryService = Depends(get_conversation_memory_service),
) -> list[ConversationSummaryResponse]:
    return [
        ConversationSummaryResponse(
            conversationId=item.conversation_id,
            title=item.title,
            lastTime=item.last_time,
            updateTime=item.update_time,
        )
        for item in service.list_conversations(user_id, limit)
    ]


@router.get(
    "/conversations/{conversation_id}/messages",
    response_model=list[ConversationMessageResponse],
)
def list_conversation_messages(
    conversation_id: str = Path(..., min_length=1),
    user_id: str = Depends(resolve_request_user_id),
    service: ConversationMemoryService = Depends(get_conversation_memory_service),
) -> list[ConversationMessageResponse]:
    return [
        ConversationMessageResponse(
            id=item.id,
            role=item.role,
            content=item.content,
            createTime=item.create_time,
        )
        for item in service.list_history(conversation_id, user_id)
    ]


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(
    conversation_id: str,
    user_id: str = Depends(resolve_request_user_id),
    service: ConversationMemoryService = Depends(get_conversation_memory_service),
) -> Response:
    service.delete_conversation(conversation_id, user_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete("/conversations", status_code=status.HTTP_204_NO_CONTENT)
def clear_conversations(
    user_id: str = Depends(resolve_request_user_id),
    service: ConversationMemoryService = Depends(get_conversation_memory_service),
) -> Response:
    service.clear_conversations(user_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
