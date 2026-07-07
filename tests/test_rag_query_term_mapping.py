from __future__ import annotations

import json

from agent_service.rag.query_term_mapping import QueryTermMappingService


class FakeTermRepository:
    def __init__(self) -> None:
        self.calls = 0

    def list_mappings(self) -> dict[str, str]:
        self.calls += 1
        return {"售后": "售后服务"}


class FakeRedis:
    def __init__(self, cached: dict[str, str] | None = None) -> None:
        self.cached = cached
        self.writes: list[tuple[str, int, str]] = []

    def get(self, key: str) -> str | None:
        if self.cached is None:
            return None
        return json.dumps(self.cached, ensure_ascii=False)

    def setex(self, key: str, ttl: int, value: str) -> None:
        self.writes.append((key, ttl, value))


def test_term_mapping_replaces_longest_terms_first() -> None:
    service = QueryTermMappingService(
        mappings={
            "苹果": "Apple",
            "苹果手机": "iPhone",
        }
    )

    assert service.normalize("苹果手机保修多久？") == "iPhone保修多久？"


def test_term_mapping_prefers_redis_cache_over_repository() -> None:
    repository = FakeTermRepository()
    service = QueryTermMappingService(
        repository=repository,
        redis_client=FakeRedis({"苹果手机": "iPhone"}),
    )

    assert service.normalize("苹果手机保修多久？") == "iPhone保修多久？"
    assert repository.calls == 0


def test_term_mapping_writes_repository_result_to_cache_on_miss() -> None:
    redis_client = FakeRedis()
    repository = FakeTermRepository()
    service = QueryTermMappingService(repository=repository, redis_client=redis_client)

    assert service.normalize("售后电话是多少？") == "售后服务电话是多少？"
    assert repository.calls == 1
    assert redis_client.writes

def test_term_mapping_does_not_replace_inside_existing_target_term() -> None:
    service = QueryTermMappingService(mappings={"平安": "平安保司"})

    assert service.normalize("平安保司的售后规则") == "平安保司的售后规则"
