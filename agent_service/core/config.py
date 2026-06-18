from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    service_name: str = "heyee-agent"
    service_version: str = "0.1.0"
    log_level: str = "INFO"

    agent_host: str = "127.0.0.1"
    agent_port: int = 8000
    agent_reload: bool = True
    agent_mock_mode: bool = True

    ai_provider: str = "openai"
    ai_api_key: str | None = None
    ai_base_url: str = "https://api.openai.com/v1"
    ai_model: str = "gpt-4o-mini"
    ai_timeout_seconds: float = Field(default=25.0, gt=0)

    java_service_url: str = "http://127.0.0.1:8081"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
