from __future__ import annotations

from fastapi.testclient import TestClient

from agent_service.main import create_app


def test_health() -> None:
    client = TestClient(create_app())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "heyee-agent",
        "version": "0.1.0",
    }

