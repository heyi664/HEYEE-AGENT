from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s - %(message)s"
APP_HANDLER_NAME = "heyee-agent-file"
METRICS_HANDLER_NAME = "heyee-rag-metrics-file"


def configure_logging(
    level: str,
    *,
    log_dir: str | Path = "./logs",
    max_bytes: int = 20 * 1024 * 1024,
    backup_count: int = 7,
) -> None:
    resolved_level = getattr(logging, level.upper(), logging.INFO)
    formatter = logging.Formatter(LOG_FORMAT)
    directory = Path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(resolved_level)
    if not root_logger.handlers:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)
    _replace_named_handler(
        root_logger,
        _rotating_handler(directory / "heyee-agent.log", formatter, max_bytes, backup_count),
        APP_HANDLER_NAME,
    )

    metrics_logger = logging.getLogger("agent_service.metrics")
    metrics_logger.setLevel(logging.INFO)
    metrics_logger.propagate = True
    _replace_named_handler(
        metrics_logger,
        _rotating_handler(directory / "rag-metrics.log", formatter, max_bytes, backup_count),
        METRICS_HANDLER_NAME,
    )


def _rotating_handler(
    path: Path,
    formatter: logging.Formatter,
    max_bytes: int,
    backup_count: int,
) -> RotatingFileHandler:
    handler = RotatingFileHandler(
        path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    handler.name = METRICS_HANDLER_NAME if path.name == "rag-metrics.log" else APP_HANDLER_NAME
    handler.setFormatter(formatter)
    return handler


def _replace_named_handler(
    logger: logging.Logger,
    handler: logging.Handler,
    handler_name: str,
) -> None:
    for existing in list(logger.handlers):
        if existing.name != handler_name:
            continue
        logger.removeHandler(existing)
        existing.close()
    logger.addHandler(handler)

