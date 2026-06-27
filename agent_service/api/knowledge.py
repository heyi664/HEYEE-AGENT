from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from starlette.datastructures import UploadFile

from agent_service.schemas.knowledge import (
    KnowledgeBaseCreateRequest,
    KnowledgeBaseCreateResponse,
    KnowledgeBaseSummary,
    KnowledgeDocumentUploadResult,
    KnowledgeDocumentUrlUploadRequest,
)
from agent_service.services.knowledge_document_service import (
    KnowledgeDocumentService,
    get_knowledge_document_service,
)

router = APIRouter(tags=["knowledge"])


@router.get("/knowledge-bases", response_model=list[KnowledgeBaseSummary])
def list_knowledge_bases(
    service: KnowledgeDocumentService = Depends(get_knowledge_document_service),
) -> list[KnowledgeBaseSummary]:
    return service.list_knowledge_bases()



@router.post("/knowledge-bases", response_model=KnowledgeBaseCreateResponse)
def create_knowledge_base(
    request: KnowledgeBaseCreateRequest,
    service: KnowledgeDocumentService = Depends(get_knowledge_document_service),
) -> KnowledgeBaseCreateResponse:
    return service.create_knowledge_base(
        name=request.name,
        embedding_model=request.embeddingModel,
        collection_name=request.collectionName,
    )

@router.post("/knowledge-documents/upload", response_model=KnowledgeDocumentUploadResult)
async def upload_knowledge_document(
    request: Request,
    service: KnowledgeDocumentService = Depends(get_knowledge_document_service),
) -> KnowledgeDocumentUploadResult:
    content_type = request.headers.get("content-type", "").lower()
    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        return await _upload_file_from_form(form, service)
    if content_type.startswith("application/json"):
        payload = await request.json()
        return await _upload_url_from_json(payload, service)
    raise HTTPException(
        status_code=415,
        detail="Content-Type must be multipart/form-data or application/json",
    )


async def _upload_file_from_form(
    form: Any,
    service: KnowledgeDocumentService,
) -> KnowledgeDocumentUploadResult:
    knowledge_base_name = _required_form_text(form, "knowledgeBaseName")
    source_type = _required_form_text(form, "sourceType").upper()
    chunk_strategy = _required_form_text(form, "chunkStrategy")
    chunk_config = _required_form_text(form, "chunkConfig")
    if source_type != "FILE":
        raise HTTPException(
            status_code=400,
            detail="multipart upload only supports sourceType FILE",
        )
    if _optional_form_text(form, "scheduleEnabled") in {"true", "1"} or _optional_form_text(
        form, "scheduleCron"
    ):
        # TODO: URL schedule sync is intentionally left for a later iteration.
        raise HTTPException(status_code=400, detail="URL schedule sync is not supported yet")
    file = form.get("file")
    if not isinstance(file, UploadFile):
        raise HTTPException(status_code=400, detail="file is required")
    return await service.upload_file(
        knowledge_base_name=knowledge_base_name,
        source_type=source_type,
        upload_file=file,
        chunk_strategy=chunk_strategy,
        chunk_config=chunk_config,
    )


async def _upload_url_from_json(
    payload: Any,
    service: KnowledgeDocumentService,
) -> KnowledgeDocumentUploadResult:
    try:
        request = KnowledgeDocumentUrlUploadRequest.model_validate(payload)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid knowledge document upload request",
        ) from exc
    return await service.upload_url(
        knowledge_base_name=request.knowledgeBaseName,
        source_type=request.sourceType,
        url=request.url,
        chunk_strategy=request.chunkStrategy,
        chunk_config=request.chunkConfig,
    )


def _required_form_text(form: Any, key: str) -> str:
    value = form.get(key)
    if value is None or isinstance(value, UploadFile):
        raise HTTPException(status_code=400, detail=f"{key} is required")
    text = str(value).strip()
    if not text:
        raise HTTPException(status_code=400, detail=f"{key} is required")
    return text


def _optional_form_text(form: Any, key: str) -> str | None:
    value = form.get(key)
    if value is None or isinstance(value, UploadFile):
        return None
    text = str(value).strip()
    return text or None
