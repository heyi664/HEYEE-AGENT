from __future__ import annotations

from agent_service.consumers.rocketmq_knowledge_chunk_consumer import RocketMqKnowledgeChunkConsumer
from agent_service.core.config import get_settings
from agent_service.core.logging import configure_logging


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    RocketMqKnowledgeChunkConsumer().start_forever()


if __name__ == "__main__":
    main()
