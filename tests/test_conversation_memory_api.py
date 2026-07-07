from __future__ import annotations

from fastapi.testclient import TestClient

from agent_service.api.conversation_memory import get_conversation_memory_service
from agent_service.main import create_app
from agent_service.memory.models import ConversationRecord, MemoryMessage


class FakeConversationMemoryService:
    def __init__(self) -> None:
        self.deleted: list[tuple[str, str]] = []
        self.cleared: list[str] = []

    def list_conversations(self, user_id: str, limit: int = 50) -> list[ConversationRecord]:
        return [
            ConversationRecord(
                conversation_id="conv_1",
                title="hello",
                last_time=None,
                update_time=None,
            )
        ]

    def list_history(self, conversation_id: str, user_id: str) -> list[MemoryMessage]:
        return [MemoryMessage(id="msg_1", role="user", content="hello")]

    def delete_conversation(self, conversation_id: str, user_id: str) -> None:
        self.deleted.append((conversation_id, user_id))

    def clear_conversations(self, user_id: str) -> None:
        self.cleared.append(user_id)


def test_conversation_memory_api_lists_history_and_deletes() -> None:
    app = create_app()
    service = FakeConversationMemoryService()
    app.dependency_overrides[get_conversation_memory_service] = lambda: service
    client = TestClient(app)

    conversations = client.get("/v1/agent/conversations", params={"userId": "7"})
    history = client.get("/v1/agent/conversations/conv_1/messages", params={"userId": "7"})
    deleted = client.delete("/v1/agent/conversations/conv_1", params={"userId": "7"})
    cleared = client.delete("/v1/agent/conversations", params={"userId": "7"})

    assert conversations.status_code == 200
    assert conversations.json()[0]["conversationId"] == "conv_1"
    assert history.status_code == 200
    assert history.json()[0]["content"] == "hello"
    assert deleted.status_code == 204
    assert cleared.status_code == 204
    assert service.deleted == [("conv_1", "7")]
    assert service.cleared == ["7"]
