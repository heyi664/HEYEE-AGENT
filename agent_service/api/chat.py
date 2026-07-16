from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse

from agent_service.core.config import get_settings
from agent_service.schemas.chat import ChatRequest, ChatResponse
from agent_service.services.chat_service import ChatService, get_chat_service
from agent_service.services.stream_chat_service import StreamChatService, get_stream_chat_service

router = APIRouter(tags=["chat"])


def _with_trusted_identity(
    chat_request: ChatRequest, authenticated_user_id: str | None
) -> ChatRequest:
    """Prefer the gateway identity over body data, while retaining local-demo compatibility."""

    requested_user_id = str(chat_request.userId) if chat_request.userId is not None else None
    if authenticated_user_id is not None:
        authenticated_user_id = authenticated_user_id.strip()
        if not authenticated_user_id:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
        if requested_user_id is not None and requested_user_id != authenticated_user_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")
        return chat_request.model_copy(update={"userId": authenticated_user_id})
    if get_settings().agent_require_authenticated_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    return chat_request.model_copy(update={"userId": requested_user_id or "0"})


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    authenticated_user_id: str | None = Header(default=None, alias="X-User-Id"),
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    return await chat_service.chat(_with_trusted_identity(request, authenticated_user_id))


@router.post("/chat/stream")
async def stream_chat(
    chat_request: ChatRequest,
    request: Request,
    authenticated_user_id: str | None = Header(default=None, alias="X-User-Id"),
    stream_service: StreamChatService = Depends(get_stream_chat_service),
) -> StreamingResponse:
    return StreamingResponse(
        stream_service.stream(
            _with_trusted_identity(chat_request, authenticated_user_id),
            is_disconnected=request.is_disconnected,
        ),
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
    authenticated_user_id: str | None = Header(default=None, alias="X-User-Id"),
    stream_service: StreamChatService = Depends(get_stream_chat_service),
) -> JSONResponse:
    if get_settings().agent_require_authenticated_user and not authenticated_user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    owner_id = authenticated_user_id.strip() if authenticated_user_id else "0"
    return JSONResponse(
        {"taskId": task_id, "cancelled": await stream_service.cancel(task_id, owner_id=owner_id)}
    )

