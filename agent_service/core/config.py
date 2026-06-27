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
    agent_max_steps: int = Field(default=5, ge=1, le=10)

    java_service_url: str = "http://127.0.0.1:8081"

    mcp_enabled: bool = False
    mcp_server_url: str = "http://127.0.0.1:8081/mcp"
    mcp_server_token: str | None = None
    mcp_timeout_seconds: float = Field(default=10.0, gt=0)
    mcp_tool_prefix: str = ""
    mcp_fail_fast: bool = False

    database_url: str | None = None

    rustfs_endpoint: str = "http://127.0.0.1:9000"
    rustfs_access_key: str | None = None
    rustfs_secret_key: str | None = None
    rustfs_bucket: str = "knowledge-base"
    rustfs_region: str = "us-east-1"
    rustfs_public_base_url: str | None = None

    upload_temp_dir: str = "./tmp/uploads"
    upload_max_size_mb: int = Field(default=100, gt=0)
    remote_download_timeout_seconds: float = Field(default=60.0, gt=0)
    upload_created_by: str = "agent"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
