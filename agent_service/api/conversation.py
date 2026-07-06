from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from agent_service.repositories.conversation_memory_repository import ConversationMemoryRepository
from agent_service.schemas.conversation import (
    ConversationDeleteResponse,
    ConversationListItem,
    ConversationMessageItem,
    ConversationMessagesResponse,
    ConversationSummaryItem,
)

router = APIRouter(tags=["conversations"])


def get_memory_repository() -> ConversationMemoryRepository:
    return ConversationMemoryRepository()


@router.get("/conversations", response_model=list[ConversationListItem])
def list_conversations(
    userId: str = Query(..., min_length=1),
    limit: int = Query(50, ge=1, le=200),
    repository: ConversationMemoryRepository = Depends(get_memory_repository),
) -> list[ConversationListItem]:
    rows = repository.list_conversations(userId, limit=limit)
    return [
        ConversationListItem(
            conversationId=str(row["conversation_id"]),
            userId=str(row["user_id"]),
            title=str(row["title"]),
            lastTime=row.get("last_time"),
            createTime=row.get("create_time"),
            updateTime=row.get("update_time"),
        )
        for row in rows
    ]


@router.get(
    "/conversations/{conversation_id}/messages",
    response_model=ConversationMessagesResponse,
)
def get_conversation_messages(
    conversation_id: str,
    userId: str = Query(..., min_length=1),
    limit: int = Query(100, ge=1, le=500),
    repository: ConversationMemoryRepository = Depends(get_memory_repository),
) -> ConversationMessagesResponse:
    summary = repository.get_latest_summary(conversation_id, userId)
    messages = repository.list_messages(conversation_id, userId, limit=limit)
    return ConversationMessagesResponse(
        conversationId=conversation_id,
        userId=userId,
        summary=(
            ConversationSummaryItem(
                id=summary.id,
                lastMessageId=summary.last_message_id,
                content=summary.content,
            )
            if summary
            else None
        ),
        messages=[
            ConversationMessageItem(
                id=message.id,
                role=message.role,
                content=message.content,
                createTime=message.create_time,
            )
            for message in messages
        ],
    )


@router.delete("/conversations/{conversation_id}", response_model=ConversationDeleteResponse)
def delete_conversation(
    conversation_id: str,
    userId: str = Query(..., min_length=1),
    repository: ConversationMemoryRepository = Depends(get_memory_repository),
) -> ConversationDeleteResponse:
    deleted = repository.soft_delete_conversation(conversation_id, userId)
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return ConversationDeleteResponse(conversationId=conversation_id, userId=userId)