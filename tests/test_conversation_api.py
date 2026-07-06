from __future__ import annotations

from datetime import datetime

from fastapi.testclient import TestClient

from agent_service.api.conversation import get_memory_repository
from agent_service.main import create_app
from agent_service.memory.models import MemoryMessage, MemorySummary


class FakeConversationRepository:
    def __init__(self) -> None:
        self.deleted: list[tuple[str, str]] = []

    def list_conversations(self, user_id: str, limit: int = 50) -> list[dict[str, object]]:
        assert user_id == "7"
        assert limit == 10
        return [
            {
                "conversation_id": "conv_1",
                "user_id": "7",
                "title": "hello",
                "last_time": datetime(2026, 7, 6, 12, 0, 0),
                "create_time": datetime(2026, 7, 6, 11, 0, 0),
                "update_time": datetime(2026, 7, 6, 12, 0, 0),
            }
        ]

    def get_latest_summary(self, conversation_id: str, user_id: str) -> MemorySummary | None:
        return MemorySummary(
            id="sum_1",
            conversation_id=conversation_id,
            user_id=user_id,
            last_message_id="msg_1",
            content="early summary",
        )

    def list_messages(
        self,
        conversation_id: str,
        user_id: str,
        limit: int = 100,
    ) -> list[MemoryMessage]:
        assert conversation_id == "conv_1"
        assert user_id == "7"
        assert limit == 20
        return [MemoryMessage(id="msg_2", role="user", content="question")]

    def soft_delete_conversation(self, conversation_id: str, user_id: str) -> bool:
        self.deleted.append((conversation_id, user_id))
        return conversation_id == "conv_1"


def test_conversation_management_routes() -> None:
    app = create_app()
    repository = FakeConversationRepository()
    app.dependency_overrides[get_memory_repository] = lambda: repository
    client = TestClient(app)

    conversations = client.get("/v1/agent/conversations", params={"userId": "7", "limit": 10})
    assert conversations.status_code == 200
    assert conversations.json()[0]["conversationId"] == "conv_1"

    messages = client.get(
        "/v1/agent/conversations/conv_1/messages",
        params={"userId": "7", "limit": 20},
    )
    assert messages.status_code == 200
    assert messages.json()["summary"]["lastMessageId"] == "msg_1"
    assert messages.json()["messages"][0]["content"] == "question"

    deleted = client.delete("/v1/agent/conversations/conv_1", params={"userId": "7"})
    assert deleted.status_code == 200
    assert deleted.json()["deleted"] is True
    assert repository.deleted == [("conv_1", "7")]


def test_delete_conversation_returns_404_when_missing() -> None:
    app = create_app()
    repository = FakeConversationRepository()
    app.dependency_overrides[get_memory_repository] = lambda: repository
    client = TestClient(app)

    response = client.delete("/v1/agent/conversations/missing", params={"userId": "7"})

    assert response.status_code == 404