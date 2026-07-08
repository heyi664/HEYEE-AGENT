from __future__ import annotations

from agent_service.memory.summary_compression_lock import RedisSummaryCompressionLock


class FakeRedisClient:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.deleted_keys: list[str] = []

    def set(self, key: str, value: str, nx: bool = False, ex: int | None = None) -> bool:
        if nx and key in self.values:
            return False
        self.values[key] = value
        return True

    def delete(self, key: str) -> None:
        self.deleted_keys.append(key)
        self.values.pop(key, None)

    def eval(self, script: str, numkeys: int, key: str, expected_value: str) -> int:
        if self.values.get(key) != expected_value:
            return 0
        self.values.pop(key, None)
        return 1


def test_redis_summary_lock_release_does_not_delete_lock_owned_by_another_worker() -> None:
    client = FakeRedisClient()
    lock = RedisSummaryCompressionLock("redis://localhost:6379/0", ttl_seconds=120)
    lock._client = client

    assert lock.acquire("conv_1", "user_1") is True
    key = "heyee:conversation-summary-lock:user_1:conv_1"
    first_owner_value = client.values[key]
    client.values[key] = "another-worker-token"

    lock.release("conv_1", "user_1")

    assert client.values[key] == "another-worker-token"
    assert client.deleted_keys == []
    assert first_owner_value != "another-worker-token"
