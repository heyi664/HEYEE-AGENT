from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse

from agent_service.schemas.chat import ChatRequest, ChatResponse
from agent_service.services.chat_service import ChatService, get_chat_service
from agent_service.services.stream_chat_service import StreamChatService, get_stream_chat_service

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    return await chat_service.chat(request)


@router.post("/chat/stream")
async def stream_chat(
    chat_request: ChatRequest,
    request: Request,
    stream_service: StreamChatService = Depends(get_stream_chat_service),
) -> StreamingResponse:
    return StreamingResponse(
        stream_service.stream(chat_request, is_disconnected=request.is_disconnected),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/chat/stream/{task_id}/cancel")
async def cancel_stream_chat(
    task_id: str,
    stream_service: StreamChatService = Depends(get_stream_chat_service),
) -> JSONResponse:
    return JSONResponse({"taskId": task_id, "cancelled": stream_service.cancel(task_id)})

