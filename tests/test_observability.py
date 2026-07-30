from __future__ import annotations

from agent_service.core.logging import configure_logging
from agent_service.core.observability import bind_conversation, pop_trace, push_trace, record_stage


def test_stage_metrics_are_written_to_the_dedicated_log_file(tmp_path) -> None:
    configure_logging("INFO", log_dir=tmp_path)

    trace = push_trace("trace_metrics", task_id="task_metrics")
    try:
        with bind_conversation("conv_metrics"):
            record_stage(
                "intent_classification_llm",
                elapsed_ms=123,
                subQuestionCount=2,
                intentCount=1,
            )
    finally:
        pop_trace(trace)

    content = (tmp_path / "rag-metrics.log").read_text(encoding="utf-8")
    assert "stage=intent_classification_llm" in content
    assert "conversationId=conv_metrics" in content
    assert "traceId=trace_metrics" in content
    assert "taskId=task_metrics" in content
    assert "elapsedMs=123" in content
    assert "intentCount=1" in content
