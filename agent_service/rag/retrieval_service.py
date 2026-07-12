from __future__ import annotations

from agent_service.core.config import get_settings
from agent_service.rag.retrieval_pipeline import (
    DeduplicationPostProcessor,
    MultiChannelRetriever,
    RerankPostProcessor,
)
from agent_service.rag.search_channels import (
    GlobalVectorSearchChannel,
    IntentDirectedSearchChannel,
    KeywordSearchChannel,
)


def get_multi_channel_retriever() -> MultiChannelRetriever:
    settings = get_settings()
    processors = [DeduplicationPostProcessor()]
    if settings.rag_retrieval_rerank_enabled:
        processors.append(RerankPostProcessor())
    return MultiChannelRetriever(
        channels=[
            IntentDirectedSearchChannel(),
            KeywordSearchChannel(),
            GlobalVectorSearchChannel(),
        ],
        post_processors=processors,
    )
