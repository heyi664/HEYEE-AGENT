from __future__ import annotations

from fastapi.testclient import TestClient

from agent_service.main import create_app


def test_frontend_chat_page_is_served() -> None:
    client = TestClient(create_app())

    response = client.get("/ui/chat.html")

    assert response.status_code == 200
    assert "HYEEE AI" in response.text
    assert "/v1/agent/chat" in response.text


def test_root_redirects_to_chat_page() -> None:
    client = TestClient(create_app(), follow_redirects=False)

    response = client.get("/")

    assert response.status_code == 307
    assert response.headers["location"] == "/ui/chat.html"
