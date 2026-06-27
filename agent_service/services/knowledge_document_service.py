from __future__ import annotations

import json
import mimetypes
import socket
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from ipaddress import ip_address
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import unquote, urlparse

import httpx
from fastapi import HTTPException
from starlette.datastructures import UploadFile

from agent_service.core.config import get_settings
from agent_service.repositories.knowledge_repository import (
    KnowledgeBaseRecord,
    KnowledgeDocumentRecord,
    KnowledgeRepository,
)
from agent_service.schemas.knowledge import (
    ALLOWED_CHUNK_STRATEGIES,
    KnowledgeBaseCreateResponse,
    KnowledgeBaseSummary,
    KnowledgeDocumentUploadResult,
)
from agent_service.services.object_storage_service import ObjectStorageService, StoredObject


@dataclass(frozen=True)
class ChunkOptions:
    strategy: str
    config: dict[str, object]


class KnowledgeDocumentService:
    def __init__(
        self,
        repository: KnowledgeRepository | None = None,
        storage: ObjectStorageService | None = None,
    ) -> None:
        self.repository = repository or KnowledgeRepository()
        self.storage = storage or ObjectStorageService()

    def list_knowledge_bases(self) -> list[KnowledgeBaseSummary]:
        return self.repository.list_knowledge_bases()


    def create_knowledge_base(
        self,
        *,
        name: str,
        embedding_model: str,
        collection_name: str,
    ) -> KnowledgeBaseCreateResponse:
        name = name.strip()
        embedding_model = embedding_model.strip()
        collection_name = collection_name.strip()
        if self.repository.find_knowledge_base_by_name(name) is not None:
            raise HTTPException(status_code=409, detail="Knowledge base name already exists")
        if self.repository.find_knowledge_base_by_collection_name(collection_name) is not None:
            raise HTTPException(status_code=409, detail="Collection name already exists")

        settings = get_settings()
        kb_id = _new_id()
        record = KnowledgeBaseRecord(
            id=kb_id,
            name=name,
            embedding_model=embedding_model,
            collection_name=collection_name,
            created_by=settings.upload_created_by,
        )
        self.repository.insert_knowledge_base(record)
        return KnowledgeBaseCreateResponse(
            id=kb_id,
            name=name,
            embeddingModel=embedding_model,
            collectionName=collection_name,
            createdBy=settings.upload_created_by,
        )
    async def upload_file(
        self,
        *,
        knowledge_base_name: str,
        source_type: str,
        upload_file: UploadFile,
        chunk_strategy: str,
        chunk_config: str,
    ) -> KnowledgeDocumentUploadResult:
        if source_type != "FILE":
            raise HTTPException(status_code=400, detail="sourceType must be FILE")
        kb = self._get_knowledge_base_or_404(knowledge_base_name)
        chunk_options = self._resolve_chunk_options(chunk_strategy, chunk_config)
        original_name = _safe_file_name(upload_file.filename or "upload.bin")
        file_type = _file_type_from_name(original_name)
        object_key = _build_object_key(kb.id, original_name)
        content_type = upload_file.content_type or mimetypes.guess_type(original_name)[0]
        stored = await self.storage.upload_async_stream(
            object_key=object_key,
            file_name=original_name,
            content_type=content_type,
            content=_upload_file_chunks(upload_file),
        )
        return self._insert_document(
            kb=kb,
            stored=stored,
            source_type="FILE",
            source_location=None,
            file_type=file_type,
            chunk_options=chunk_options,
        )

    async def upload_url(
        self,
        *,
        knowledge_base_name: str,
        source_type: str,
        url: str,
        chunk_strategy: str,
        chunk_config: str,
    ) -> KnowledgeDocumentUploadResult:
        if source_type != "URL":
            raise HTTPException(status_code=400, detail="sourceType must be URL")
        kb = self._get_knowledge_base_or_404(knowledge_base_name)
        chunk_options = self._resolve_chunk_options(chunk_strategy, chunk_config)
        self._validate_remote_url(url)

        temp_path: Path | None = None
        try:
            downloaded = await self._download_remote_file(url)
            temp_path = downloaded.path
            file_type = _file_type_from_name(downloaded.file_name)
            object_key = _build_object_key(kb.id, downloaded.file_name)
            stored = await self.storage.upload_local_file(
                path=temp_path,
                object_key=object_key,
                file_name=downloaded.file_name,
                content_type=downloaded.content_type,
            )
            return self._insert_document(
                kb=kb,
                stored=stored,
                source_type="URL",
                source_location=url,
                file_type=file_type,
                chunk_options=chunk_options,
            )
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    def _get_knowledge_base_or_404(self, name: str) -> KnowledgeBaseSummary:
        kb = self.repository.find_knowledge_base_by_name(name.strip())
        if kb is None:
            raise HTTPException(status_code=404, detail="知识库不存在")
        return kb

    def _resolve_chunk_options(self, strategy: str, config_text: str) -> ChunkOptions:
        strategy = strategy.strip()
        if strategy not in ALLOWED_CHUNK_STRATEGIES:
            raise HTTPException(status_code=400, detail=f"非法分块策略: {strategy}")
        try:
            config = json.loads(config_text)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=400,
                detail="chunkConfig 必须是合法 JSON 字符串",
            ) from exc
        if not isinstance(config, dict):
            raise HTTPException(status_code=400, detail="chunkConfig 必须是 JSON 对象")
        self._validate_chunk_config(config)
        return ChunkOptions(strategy=strategy, config=config)

    def _validate_chunk_config(self, config: dict[str, object]) -> None:
        numeric_fields = ["targetChars", "maxChars", "minChars", "overlapChars"]
        for field in numeric_fields:
            if field in config and not isinstance(config[field], int):
                raise HTTPException(status_code=400, detail=f"chunkConfig.{field} 必须是整数")

        target = config.get("targetChars")
        max_chars = config.get("maxChars")
        min_chars = config.get("minChars")
        overlap = config.get("overlapChars")
        if isinstance(target, int) and target <= 0:
            raise HTTPException(status_code=400, detail="chunkConfig.targetChars 必须大于 0")
        if isinstance(max_chars, int) and isinstance(target, int) and max_chars < target:
            raise HTTPException(
                status_code=400,
                detail="chunkConfig.maxChars 必须大于等于 targetChars",
            )
        if isinstance(min_chars, int) and isinstance(target, int) and min_chars > target:
            raise HTTPException(
                status_code=400,
                detail="chunkConfig.minChars 必须小于等于 targetChars",
            )
        if isinstance(overlap, int) and overlap < 0:
            raise HTTPException(status_code=400, detail="chunkConfig.overlapChars 不能为负数")

    def _validate_remote_url(self, url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise HTTPException(status_code=400, detail="URL 只支持 http/https")
        try:
            for info in socket.getaddrinfo(parsed.hostname, parsed.port or 443):
                address = ip_address(info[4][0])
                if address.is_private or address.is_loopback or address.is_link_local:
                    raise HTTPException(status_code=400, detail="URL 不允许指向内网地址")
        except HTTPException:
            raise
        except OSError as exc:
            raise HTTPException(status_code=400, detail="URL 域名无法解析") from exc

    async def _download_remote_file(self, url: str) -> DownloadedFile:
        settings = get_settings()
        max_bytes = settings.upload_max_size_mb * 1024 * 1024
        temp_dir = Path(settings.upload_temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        file_name = _safe_file_name(_file_name_from_url(url))
        content_type: str | None = None
        bytes_written = 0
        with NamedTemporaryFile(
            delete=False,
            dir=temp_dir,
            prefix="remote-",
            suffix=".tmp",
        ) as temp:
            temp_path = Path(temp.name)
            try:
                async with httpx.AsyncClient(
                    timeout=settings.remote_download_timeout_seconds,
                    follow_redirects=True,
                    max_redirects=3,
                ) as client:
                    async with client.stream("GET", url) as response:
                        response.raise_for_status()
                        content_type = response.headers.get("content-type")
                        header_name = _file_name_from_content_disposition(
                            response.headers.get("content-disposition")
                        )
                        if header_name:
                            file_name = _safe_file_name(header_name)
                        async for chunk in response.aiter_bytes(1024 * 1024):
                            bytes_written += len(chunk)
                            if bytes_written > max_bytes:
                                raise HTTPException(status_code=400, detail="远程文件超过最大限制")
                            temp.write(chunk)
            except Exception:
                temp_path.unlink(missing_ok=True)
                raise
        return DownloadedFile(path=temp_path, file_name=file_name, content_type=content_type)

    def _insert_document(
        self,
        *,
        kb: KnowledgeBaseSummary,
        stored: StoredObject,
        source_type: str,
        source_location: str | None,
        file_type: str,
        chunk_options: ChunkOptions,
    ) -> KnowledgeDocumentUploadResult:
        settings = get_settings()
        document_id = _new_id()
        record = KnowledgeDocumentRecord(
            id=document_id,
            kb_id=kb.id,
            doc_name=stored.original_file_name,
            file_url=stored.file_url,
            file_type=file_type,
            file_size=stored.file_size,
            source_type=source_type,
            source_location=source_location,
            chunk_strategy=chunk_options.strategy,
            chunk_config=chunk_options.config,
            created_by=settings.upload_created_by,
        )
        self.repository.insert_document(record)
        return KnowledgeDocumentUploadResult(
            id=document_id,
            knowledgeBaseId=kb.id,
            knowledgeBaseName=kb.name,
            docName=stored.original_file_name,
            fileUrl=stored.file_url,
            fileType=file_type,
            fileSize=stored.file_size,
            sourceType=source_type,  # type: ignore[arg-type]
            sourceLocation=source_location,
            chunkStrategy=chunk_options.strategy,
            chunkConfig=chunk_options.config,
        )


@dataclass(frozen=True)
class DownloadedFile:
    path: Path
    file_name: str
    content_type: str | None


async def _upload_file_chunks(upload_file: UploadFile) -> AsyncIterator[bytes]:
    while True:
        chunk = await upload_file.read(1024 * 1024)
        if not chunk:
            break
        yield chunk


def _new_id() -> str:
    return uuid.uuid4().hex[:20]


def _safe_file_name(name: str) -> str:
    name = unquote(name).replace("\\", "/").split("/")[-1].strip()
    cleaned = "".join(ch if ch.isalnum() or ch in {".", "-", "_"} else "_" for ch in name)
    return cleaned or "upload.bin"


def _file_name_from_url(url: str) -> str:
    path = urlparse(url).path
    name = unquote(path.rsplit("/", 1)[-1]) if path else ""
    return name or "remote-file.bin"


def _file_name_from_content_disposition(value: str | None) -> str | None:
    if not value:
        return None
    for part in value.split(";"):
        part = part.strip()
        if part.lower().startswith("filename="):
            return part.split("=", 1)[1].strip().strip('"')
    return None


def _file_type_from_name(name: str) -> str:
    suffix = Path(name).suffix.lower().lstrip(".")
    if suffix == "md":
        return "markdown"
    return suffix or "bin"


def _build_object_key(kb_id: str, file_name: str) -> str:
    from datetime import datetime

    today = datetime.now().strftime("%Y%m%d")
    return f"knowledge-base/{kb_id}/{today}/{uuid.uuid4().hex}-{file_name}"


_service: KnowledgeDocumentService | None = None


def get_knowledge_document_service() -> KnowledgeDocumentService:
    global _service
    if _service is None:
        _service = KnowledgeDocumentService()
    return _service
