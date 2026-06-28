from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from agent_service.core.config import get_settings

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
            # Keep the same payload validation path in local/mock mode.
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
            )
        except Exception as exc:  # pragma: no cover - depends on deployment package
            raise RuntimeError(
                "rocketmq-client-python is required when ROCKETMQ_MOCK_MODE=false"
            ) from exc

        transaction_holder: dict[str, T] = {}

        def execute_local_transaction(_message: object, _user_args: object) -> int:
            try:
                transaction_holder["result"] = local_transaction()
                return 1
            except Exception:
                return 2

        producer = TransactionMQProducer(self.settings.rocketmq_producer_group)
        producer.set_name_server_address(self.settings.rocketmq_name_server)
        producer.set_session_credentials(
            self.settings.rocketmq_access_key or "",
            self.settings.rocketmq_secret_key or "",
            "",
        )
        producer.set_transaction_listener(execute_local_transaction, None)
        producer.start()
        try:
            # The transaction message API requires a half-message before the local transaction.
            # The complete chunk payload is still derived from the CAS result in the callback path.
            initial_body = json.dumps({"docId": key}, ensure_ascii=False).encode("utf-8")
            message = Message(topic)
            message.set_tags(tag)
            message.set_keys(key)
            message.set_body(initial_body)
            send_result = producer.send_message_in_transaction(message, None)
            tx_result = transaction_holder.get("result")
            if tx_result is not None:
                json.dumps(message_builder(tx_result), ensure_ascii=False)
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
