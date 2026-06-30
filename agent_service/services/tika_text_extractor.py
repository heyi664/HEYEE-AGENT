from __future__ import annotations

from urllib.parse import quote

import httpx

from agent_service.core.config import get_settings


class TikaTextExtractor:
    async def extract_text(self, content: bytes, file_name: str | None = None) -> str:
        settings = get_settings()
        url = settings.tika_server_url.rstrip("/") + "/tika"
        headers = {"Accept": "text/plain"}
        if file_name:
            safe_name = quote(file_name, safe="")
            headers["Content-Disposition"] = f"attachment; filename*=UTF-8''{safe_name}"
        async with httpx.AsyncClient(timeout=settings.tika_timeout_seconds) as client:
            response = await client.put(url, content=content, headers=headers)
        response.raise_for_status()
        text = response.text.strip()
        if not text:
            raise ValueError("Tika extracted empty text")
        return text
