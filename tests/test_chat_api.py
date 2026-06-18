from __future__ import annotations

from fastapi.testclient import TestClient

from agent_service.main import create_app


def test_chat_generates_conversation_id_and_reply() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/v1/agent/chat",
        json={
            "userId": 1,
            "conversationId": None,
            "message": "你好",
            "history": [],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["conversationId"].startswith("conv_")
    assert body["reply"]
    assert body["sources"] == []
    assert body["toolCalls"] == []


def test_chat_rejects_blank_message() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/v1/agent/chat",
        json={
            "userId": 1,
            "conversationId": "conv_001",
            "message": "   ",
            "history": [],
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid chat request"

