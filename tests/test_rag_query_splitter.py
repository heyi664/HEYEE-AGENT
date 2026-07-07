from __future__ import annotations

from agent_service.rag.query_splitter import split_query


def test_split_query_splits_multiple_questions() -> None:
    assert split_query("苹果手机怎么保修？换电池多少钱？") == [
        "苹果手机怎么保修？",
        "换电池多少钱？",
    ]


def test_split_query_splits_semicolon_and_newline() -> None:
    assert split_query("查询订单状态；然后告诉我退款规则\n还要发票规则") == [
        "查询订单状态",
        "然后告诉我退款规则",
        "还要发票规则",
    ]


def test_split_query_keeps_comparison_question_together() -> None:
    assert split_query("A 和 B 有什么区别？") == ["A 和 B 有什么区别？"]


def test_split_query_returns_original_when_blank_separators_do_not_create_parts() -> None:
    assert split_query("苹果手机保修多久？") == ["苹果手机保修多久？"]
