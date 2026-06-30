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

    embedding_provider: str = "siliconflow"
    embedding_api_key: str | None = None
    embedding_base_url: str = "https://api.siliconflow.cn/v1"
    embedding_model: str = "BAAI/bge-m3"
    embedding_dimension: int = Field(default=1024, gt=0)
    embedding_batch_size: int = Field(default=32, ge=1)
    embedding_timeout_seconds: float = Field(default=60.0, gt=0)

    tika_server_url: str = "http://127.0.0.1:9998"
    tika_timeout_seconds: float = Field(default=60.0, gt=0)
    chunk_pipeline_max_retries: int = Field(default=2, ge=0)
    chunk_pipeline_retry_backoff_seconds: float = Field(default=0.5, ge=0)
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
    upload_rate_limit_enabled: bool = True
    upload_rate_limit_redis_url: str = "redis://:123456@192.168.23.129:6379/0"
    upload_rate_limit_key: str = "heyee:knowledge-upload:semaphore"
    upload_rate_limit_permits: int = Field(default=3, ge=1)
    upload_rate_limit_lease_seconds: int = Field(default=900, ge=1)
    upload_rate_limit_acquire_timeout_ms: int = Field(default=0, ge=0)

    rocketmq_mock_mode: bool = True
    rocketmq_name_server: str = "127.0.0.1:9876"
    rocketmq_producer_group: str = "heyee-agent-chunk-producer"
    rocketmq_consumer_group: str = "heyee-agent-chunk-consumer"
    rocketmq_chunk_topic: str = "heyee-knowledge-document-chunk"
    rocketmq_chunk_tag: str = "START_CHUNK"
    rocketmq_access_key: str | None = None
    rocketmq_secret_key: str | None = None
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()

