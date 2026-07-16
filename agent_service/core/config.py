from __future__ import annotations

from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    service_name: str = "heyee-agent"
    service_version: str = "0.1.0"
    log_level: str = "INFO"
    log_dir: str = "./logs"
    log_max_bytes: int = Field(default=20 * 1024 * 1024, ge=1024)
    log_backup_count: int = Field(default=7, ge=1, le=100)

    agent_host: str = "127.0.0.1"
    agent_port: int = 8000
    agent_reload: bool = True
    agent_mock_mode: bool = True

    ai_provider: str = "openai"
    ai_api_key: str | None = None
    ai_base_url: str = "https://api.openai.com/v1"
    ai_model: str = "gpt-4o-mini"
    ai_timeout_seconds: float = Field(default=25.0, gt=0)
    ai_models_json: str | None = None
    ai_circuit_failure_threshold: int = Field(default=3, ge=1)
    ai_circuit_open_seconds: float = Field(default=60.0, gt=0)
    ai_circuit_half_open_max_in_flight: int = Field(default=1, ge=1)
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
    # The business-project MCP Server is intentionally not configured yet.
    # Set MCP_ENABLED=true and MCP_SERVER_URL only after the Server is available.
    mcp_server_url: str | None = None
    mcp_server_token: str | None = None
    mcp_timeout_seconds: float = Field(default=10.0, gt=0)
    mcp_tool_prefix: str = ""
    mcp_fail_fast: bool = False
    mcp_context_max_chars: int = Field(default=6000, ge=500, le=50000)

    database_url: str | None = None
    memory_enabled: bool = True
    memory_history_keep_turns: int = Field(default=8, ge=1, le=100)
    memory_summary_enabled: bool = False
    memory_summary_batch_size: int = Field(default=3, ge=1, le=20)
    memory_summary_max_chars: int = Field(default=300, ge=100, le=2000)
    memory_async_compress: bool = True
    memory_context_max_chars: int = Field(default=6000, ge=500, le=50000)
    memory_redis_url: str | None = None
    memory_lock_ttl_seconds: int = Field(default=120, ge=1, le=3600)

    # Optional Redis coordination for cross-instance stream cancellation. Keep empty for
    # local/single-instance development; cancellation still works through local memory.
    stream_cancel_redis_url: str | None = None
    stream_cancel_key_prefix: str = "heyee:stream:cancel"
    stream_cancel_channel: str = "heyee:stream:cancel"
    stream_cancel_ttl_seconds: int = Field(default=1800, ge=30, le=86400)
    stream_cancel_max_tasks: int = Field(default=10000, ge=100, le=100000)

    # Stream admission control. Without a Redis URL this remains a FIFO limiter for one
    # process; set the URL on every instance to enforce one cluster-wide limit.
    stream_queue_enabled: bool = True
    stream_queue_redis_url: str | None = None
    stream_queue_key_prefix: str = "heyee:stream:queue"
    stream_queue_max_concurrent: int = Field(default=3, ge=1, le=1000)
    stream_queue_max_wait_seconds: float = Field(default=20.0, gt=0, le=3600)
    stream_queue_lease_seconds: float = Field(default=600.0, gt=0, le=86400)
    stream_queue_poll_interval_ms: int = Field(default=200, ge=20, le=5000)

    rag_query_rewrite_enabled: bool = False
    rag_query_rewrite_history_turns: int = Field(default=2, ge=0, le=5)
    rag_query_rewrite_max_sub_questions: int = Field(default=5, ge=1, le=10)
    rag_query_rewrite_history_max_chars: int = Field(default=1500, ge=100, le=10000)
    rag_query_rewrite_history_message_max_chars: int = Field(default=500, ge=50, le=5000)
    rag_term_mapping_cache_ttl_seconds: int = Field(default=300, ge=1)
    rag_intent_enabled: bool = False
    rag_intent_min_score: float = Field(default=0.35, ge=0, le=1)
    rag_max_intent_count: int = Field(default=3, ge=1, le=10)
    rag_intent_cache_enabled: bool = True
    rag_intent_cache_ttl_seconds: int = Field(default=604800, ge=60)
    rag_intent_llm_temperature: float = Field(default=0.1, ge=0, le=2)
    rag_intent_llm_top_p: float = Field(default=0.3, ge=0, le=1)
    rag_guidance_enabled: bool = True
    rag_guidance_score_ratio: float = Field(default=0.8, ge=0, le=1)
    rag_guidance_margin: float = Field(default=0.15, ge=0, le=1)
    rag_guidance_max_options: int = Field(default=6, ge=2, le=20)
    rag_retrieval_candidate_top_k: int = Field(default=10, ge=1, le=100)
    rag_retrieval_final_top_k: int = Field(default=5, ge=1, le=20)
    rag_retrieval_keyword_enabled: bool = False
    rag_retrieval_global_vector_enabled: bool = False
    rag_retrieval_rerank_enabled: bool = True

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

    @model_validator(mode="after")
    def validate_stream_queue_lease(self) -> Settings:
        """Keep the crash-recovery lease comfortably beyond the visible queue wait.

        A worker can be delayed while claiming a permit, beginning preparation, or shutting
        down.  The lease is a last-resort recovery mechanism, not the normal completion timer;
        requiring at least three wait windows prevents an active stream from being admitted
        twice merely because it exceeded the user-facing queue timeout.
        """

        if (
            self.stream_queue_enabled
            and self.stream_queue_lease_seconds <= self.stream_queue_max_wait_seconds * 3
        ):
            raise ValueError(
                "stream_queue_lease_seconds must be greater than "
                "stream_queue_max_wait_seconds * 3"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
