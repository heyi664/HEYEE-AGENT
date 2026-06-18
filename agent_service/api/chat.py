from __future__ import annotations

from fastapi import APIRouter, Depends

from agent_service.schemas.chat import ChatRequest, ChatResponse
from agent_service.services.chat_service import ChatService, get_chat_service

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    return await chat_service.chat(request)

