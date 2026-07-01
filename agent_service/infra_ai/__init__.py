from __future__ import annotations

from agent_service.infra_ai.config import get_ai_model_config
from agent_service.infra_ai.health_store import ModelHealthStore, get_model_health_store
from agent_service.infra_ai.routing_executor import ModelRoutingExecutor
from agent_service.infra_ai.selector import ModelSelector


def get_model_selector() -> ModelSelector:
    return ModelSelector(get_ai_model_config(), get_model_health_store())


def get_model_routing_executor() -> ModelRoutingExecutor:
    return ModelRoutingExecutor(get_model_health_store())


__all__ = [
    "ModelHealthStore",
    "ModelRoutingExecutor",
    "ModelSelector",
    "get_ai_model_config",
    "get_model_health_store",
    "get_model_routing_executor",
    "get_model_selector",
]
