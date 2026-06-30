from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO
from urllib.parse import quote, urlparse

import httpx

from agent_service.core.config import get_settings


@dataclass(frozen=True)
class StoredObject:
    original_file_name: str
    file_url: str
    object_key: str
    file_size: int


class ObjectStorageService:
    async def upload_async_stream(
        self,
        *,
        bucket_name: str,
        object_key: str,
        file_name: str,
        content_type: str | None,
        content: AsyncIterator[bytes],
        content_length: int | None = None,
    ) -> StoredObject:
        settings = get_settings()
        put_url = self._create_presigned_put_url(bucket_name, object_key, content_type)
        counted = _CountingAsyncIterator(content, settings.upload_max_size_mb * 1024 * 1024)
        headers = {"Content-Type": content_type or "application/octet-stream"}
        if content_length is not None:
            headers["Content-Length"] = str(content_length)
        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.put(put_url, content=counted, headers=headers)
            if response.is_error:
                raise RuntimeError(
                    "RustFS upload failed "
                    f"status={response.status_code} body={response.text[:500]}"
                )
        return StoredObject(
            original_file_name=file_name,
            file_url=self._build_file_url(bucket_name, object_key),
            object_key=object_key,
            file_size=counted.bytes_read,
        )

    async def upload_local_file(
        self,
        *,
        path: Path,
        bucket_name: str,
        object_key: str,
        file_name: str,
        content_type: str | None,
    ) -> StoredObject:
        settings = get_settings()
        file_size = path.stat().st_size
        max_size = settings.upload_max_size_mb * 1024 * 1024
        if file_size > max_size:
            raise ValueError(f"文件超过最大限制 {settings.upload_max_size_mb}MB")

        put_url = self._create_presigned_put_url(bucket_name, object_key, content_type)
        headers = {
            "Content-Type": content_type or "application/octet-stream",
            "Content-Length": str(file_size),
        }
        async with httpx.AsyncClient(timeout=None) as client:
            with path.open("rb") as file_obj:
                response = await client.put(
                    put_url,
                    content=_read_file_chunks(file_obj),
                    headers=headers,
                )
            if response.is_error:
                raise RuntimeError(
                    "RustFS upload failed "
                    f"status={response.status_code} body={response.text[:500]}"
                )
        return StoredObject(
            original_file_name=file_name,
            file_url=self._build_file_url(bucket_name, object_key),
            object_key=object_key,
            file_size=file_size,
        )


    def download_file_url(self, file_url: str) -> bytes:
        bucket_name, object_key = _parse_s3_file_url(file_url)
        client = self._create_s3_client()
        response = client.get_object(Bucket=bucket_name, Key=object_key)
        body = response["Body"]
        try:
            return body.read()
        finally:
            body.close()
    def ensure_bucket(self, bucket_name: str) -> None:
        client = self._create_s3_client()
        try:
            client.head_bucket(Bucket=bucket_name)
            return
        except Exception:
            client.create_bucket(Bucket=bucket_name)

    def _create_presigned_put_url(
        self,
        bucket_name: str,
        object_key: str,
        content_type: str | None,
    ) -> str:
        client = self._create_s3_client()
        params = {"Bucket": bucket_name, "Key": object_key}
        if content_type:
            params["ContentType"] = content_type
        return str(client.generate_presigned_url("put_object", Params=params, ExpiresIn=900))

    def _create_s3_client(self):
        settings = get_settings()
        if not settings.rustfs_access_key or not settings.rustfs_secret_key:
            raise RuntimeError("RustFS credentials are not configured")

        try:
            import boto3
            from botocore.client import Config
        except ImportError as exc:  # pragma: no cover - depends on runtime environment
            raise RuntimeError("boto3 is required for RustFS presigned uploads") from exc

        return boto3.client(
            "s3",
            endpoint_url=settings.rustfs_endpoint,
            aws_access_key_id=settings.rustfs_access_key,
            aws_secret_access_key=settings.rustfs_secret_key,
            region_name=settings.rustfs_region,
            config=Config(signature_version="s3v4"),
        )

    def _build_file_url(self, bucket_name: str, object_key: str) -> str:
        settings = get_settings()
        if settings.rustfs_public_base_url:
            base = settings.rustfs_public_base_url.rstrip("/")
            return f"{base}/{quote(object_key)}"
        return f"s3://{bucket_name}/{object_key}"


class _CountingAsyncIterator:
    def __init__(self, source: AsyncIterator[bytes], max_bytes: int) -> None:
        self.source = source
        self.max_bytes = max_bytes
        self.bytes_read = 0

    def __aiter__(self) -> _CountingAsyncIterator:
        return self

    async def __anext__(self) -> bytes:
        chunk = await self.source.__anext__()
        self.bytes_read += len(chunk)
        if self.bytes_read > self.max_bytes:
            raise ValueError("文件超过最大限制")
        return chunk


def _read_file_chunks(file_obj: BinaryIO, chunk_size: int = 1024 * 1024) -> Iterator[bytes]:
    while True:
        chunk = file_obj.read(chunk_size)
        if not chunk:
            break
        yield chunk

def _parse_s3_file_url(file_url: str) -> tuple[str, str]:
    parsed = urlparse(file_url)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
        raise ValueError("fileUrl must be an s3://bucket/object-key URL")
    return parsed.netloc, parsed.path.lstrip("/")