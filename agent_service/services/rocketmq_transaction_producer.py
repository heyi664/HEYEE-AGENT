from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from agent_service.core.config import get_settings
from agent_service.core.errors import MessageQueueUnavailableError

T = TypeVar("T")


@dataclass(frozen=True)
class TransactionSendResult:
    message_id: str | None = None
    transaction_id: str | None = None


class TransactionMessageProducer(Protocol):
    def send_in_transaction(
        self,
        *,
        topic: str,
        tag: str,
        key: str,
        local_transaction: Callable[[], T],
        message_builder: Callable[[T], dict[str, Any]],
    ) -> TransactionSendResult:
        ...


class RocketMqTransactionProducer:
    def __init__(self) -> None:
        self.settings = get_settings()

    def send_in_transaction(
        self,
        *,
        topic: str,
        tag: str,
        key: str,
        local_transaction: Callable[[], T],
        message_builder: Callable[[T], dict[str, Any]],
    ) -> TransactionSendResult:
        if self.settings.rocketmq_mock_mode:
            transaction_result = local_transaction()
            json.dumps(message_builder(transaction_result), ensure_ascii=False)
            return TransactionSendResult(message_id=f"mock-{key}", transaction_id=None)

        return self._send_with_rocketmq(
            topic=topic,
            tag=tag,
            key=key,
            local_transaction=local_transaction,
            message_builder=message_builder,
        )

    def _send_with_rocketmq(
        self,
        *,
        topic: str,
        tag: str,
        key: str,
        local_transaction: Callable[[], T],
        message_builder: Callable[[T], dict[str, Any]],
    ) -> TransactionSendResult:
        try:
            from rocketmq.client import (  # type: ignore[import-not-found]
                Message,
                TransactionMQProducer,
                TransactionStatus,
            )
        except Exception as exc:  # pragma: no cover - depends on deployment package/platform
            raise MessageQueueUnavailableError(
                "RocketMQ 客户端不可用：请确认已安装 rocketmq-client-python，"
                "并在 Linux/WSL/Docker 等受支持环境运行，或开启 ROCKETMQ_MOCK_MODE"
            ) from exc

        transaction_holder: dict[str, T] = {}

        def check_callback(_message: object) -> int:
            if "result" in transaction_holder:
                return TransactionStatus.COMMIT
            return TransactionStatus.ROLLBACK

        def local_execute(_message: object, _user_args: object) -> int:
            try:
                transaction_result = local_transaction()
                transaction_holder["result"] = transaction_result
                # Validate the final consumer payload shape. The consumer can still recover all
                # fields from docId by querying the database if a broker/client cannot mutate body.
                json.dumps(message_builder(transaction_result), ensure_ascii=False)
                return TransactionStatus.COMMIT
            except Exception:
                return TransactionStatus.ROLLBACK

        producer = TransactionMQProducer(self.settings.rocketmq_producer_group, check_callback)
        producer.set_name_server_address(self.settings.rocketmq_name_server)
        producer.start()
        try:
            message = Message(topic)
            message.set_tags(tag)
            message.set_keys(key)
            message.set_body(json.dumps({"docId": key}, ensure_ascii=False).encode("utf-8"))
            send_result = producer.send_message_in_transaction(message, local_execute, None)
            return TransactionSendResult(
                message_id=getattr(send_result, "msg_id", None),
                transaction_id=getattr(send_result, "transaction_id", None),
            )
        finally:
            producer.shutdown()


_producer: RocketMqTransactionProducer | None = None


def get_rocketmq_transaction_producer() -> RocketMqTransactionProducer:
    global _producer
    if _producer is None:
        _producer = RocketMqTransactionProducer()
    return _producer
