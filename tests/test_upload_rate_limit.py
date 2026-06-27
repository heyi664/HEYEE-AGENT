from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_service.middleware.upload_rate_limit import UploadRateLimitMiddleware


class FakeSettings:
    upload_rate_limit_enabled: bool = True
    upload_rate_limit_redis_url: str = "redis://localhost:6379/0"
    upload_rate_limit_key: str = "test:upload:semaphore"
    upload_rate_limit_permits: int = 1
    upload_rate_limit_lease_seconds: int = 10
    upload_rate_limit_acquire_timeout_ms: int = 0


class FakeMiddleware(UploadRateLimitMiddleware):
    def __init__(self, app: Any, permit: str | None = "permit") -> None:
        super().__init__(app, settings=FakeSettings())
        self.permit = permit
        self.released: list[str] = []

    async def _acquire_permit(self) -> str | None:
        return self.permit

    async def _release_permit(self, token: str) -> None:
        self.released.append(token)


class DisabledSettings(FakeSettings):
    upload_rate_limit_enabled: bool = False


def make_app(middleware: type[UploadRateLimitMiddleware], *args: Any) -> FastAPI:
    app = FastAPI()
    app.add_middleware(middleware, *args)

    @app.post("/v1/knowledge-documents/upload")
    async def upload() -> dict[str, bool]:
        return {"ok": True}

    @app.post("/v1/agent/chat")
    async def chat() -> dict[str, bool]:
        return {"ok": True}

    return app


def test_upload_rate_limit_returns_429_when_no_permit() -> None:
    class NoPermitMiddleware(FakeMiddleware):
        def __init__(self, app: Any) -> None:
            super().__init__(app, permit=None)

    client = TestClient(make_app(NoPermitMiddleware))

    response = client.post("/v1/knowledge-documents/upload")

    assert response.status_code == 429
    assert response.json()["detail"] == "Too many concurrent upload requests"


def test_upload_rate_limit_does_not_limit_other_paths() -> None:
    class NoPermitMiddleware(FakeMiddleware):
        def __init__(self, app: Any) -> None:
            super().__init__(app, permit=None)

    client = TestClient(make_app(NoPermitMiddleware))

    response = client.post("/v1/agent/chat")

    assert response.status_code == 200


def test_upload_rate_limit_can_be_disabled() -> None:
    app = FastAPI()
    app.add_middleware(UploadRateLimitMiddleware, settings=DisabledSettings())

    @app.post("/v1/knowledge-documents/upload")
    async def upload() -> dict[str, bool]:
        return {"ok": True}

    client = TestClient(app)

    response = client.post("/v1/knowledge-documents/upload")

    assert response.status_code == 200

def test_upload_rate_limit_returns_503_when_redis_fails() -> None:
    class BrokenMiddleware(FakeMiddleware):
        def __init__(self, app: Any) -> None:
            super().__init__(app)

        async def _acquire_permit(self) -> str | None:
            raise RuntimeError("redis failed")

    client = TestClient(make_app(BrokenMiddleware))

    response = client.post("/v1/knowledge-documents/upload")

    assert response.status_code == 503
    assert response.json()["detail"] == "Upload rate limiter is unavailable"
