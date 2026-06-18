from __future__ import annotations

import pytest

from agent_service.core.config import get_settings


@pytest.fixture(autouse=True)
def use_mock_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_MOCK_MODE", "true")
    monkeypatch.setenv("AI_PROVIDER", "openai")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
