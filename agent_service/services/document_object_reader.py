from __future__ import annotations

from urllib.parse import urlparse

from agent_service.services.object_storage_service import ObjectStorageService


class DocumentObjectReader:
    def __init__(self, storage: ObjectStorageService | None = None) -> None:
        self.storage = storage or ObjectStorageService()

    def read(self, file_url: str) -> bytes:
        return self.storage.download_file_url(file_url)


def parse_s3_file_url(file_url: str) -> tuple[str, str]:
    parsed = urlparse(file_url)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
        raise ValueError("fileUrl must be an s3://bucket/object-key URL")
    return parsed.netloc, parsed.path.lstrip("/")
