from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from agent_service.core.config import get_settings
from agent_service.core.errors import MessageQueueUnavailableError
from agent_service.schemas.chunking import KnowledgeChunkMessage
from agent_service.services.knowledge_chunk_pipeline import KnowledgeChunkPipeline

logger = logging.getLogger(__name__)


class RocketMqKnowledgeChunkConsumer:
    def __init__(self, pipeline: KnowledgeChunkPipeline | None = None) -> None:
        self.settings = get_settings()
        self.pipeline = pipeline or KnowledgeChunkPipeline()

    def start_forever(self) -> None:
        try:
            from rocketmq.client import (  # type: ignore[import-not-found]
                ConsumeStatus,
                PushConsumer,
            )
        except Exception as exc:  # pragma: no cover - depends on deployment package/platform
            raise MessageQueueUnavailableError(
                "RocketMQ 客户端不可用：请确认已安装 rocketmq-client-python，"
                "并在 Linux/WSL/Docker 等受支持环境运行"
            ) from exc

        def callback(message: Any) -> int:
            try:
                payload = json.loads(message.body.decode("utf-8"))
                payload.setdefault("messageId", getattr(message, "id", None))
                chunk_message = KnowledgeChunkMessage.from_dict(payload)
                asyncio.run(self.pipeline.process_message(chunk_message))
                return ConsumeStatus.CONSUME_SUCCESS
            except Exception:
                logger.exception("failed to consume knowledge chunk message")
                return ConsumeStatus.RECONSUME_LATER

        consumer = PushConsumer(self.settings.rocketmq_consumer_group)
        consumer.set_name_server_address(self.settings.rocketmq_name_server)
        consumer.subscribe(self.settings.rocketmq_chunk_topic, callback)
        consumer.start()
        logger.info(
            "RocketMQ knowledge chunk consumer started topic=%s group=%s",
            self.settings.rocketmq_chunk_topic,
            self.settings.rocketmq_consumer_group,
        )
        try:
            while True:
                time.sleep(3600)
        finally:
            consumer.shutdown()
