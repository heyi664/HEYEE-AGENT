from __future__ import annotations

from agent_service.services.chunking_service import ChunkingService


def test_fixed_size_chunker_aligns_to_sentence_boundary() -> None:
    chunks = ChunkingService().split(
        "第一段内容。第二段内容。第三段内容。",
        "fixed_size",
        {"targetChars": 9, "overlapChars": 4},
    )

    assert len(chunks) >= 2
    assert chunks[0].content.endswith("。")
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1


def test_fixed_size_chunker_repairs_soft_cjk_line_break() -> None:
    chunks = ChunkingService().split(
        "商\n保通是一款系统。",
        "fixed_size",
        {"targetChars": 100, "overlapChars": 0},
    )

    assert chunks[0].content == "商保通是一款系统。"


def test_structure_aware_chunker_keeps_markdown_code_fence_together() -> None:
    text = "# 标题\n\n介绍段落。\n\n```python\nprint('hello')\n```\n\n## 下一节\n正文内容。"

    chunks = ChunkingService().split(
        text,
        "structure_aware",
        {"targetChars": 18, "maxChars": 32, "minChars": 8, "overlapChars": 0},
    )

    assert len(chunks) >= 2
    assert any("```python\nprint('hello')\n```" in chunk.content for chunk in chunks)
    assert all(chunk.chunk_index == index for index, chunk in enumerate(chunks))


def test_structure_aware_chunker_only_splits_on_block_boundaries() -> None:
    text = "# A\n\nparagraph one.\n\nparagraph two.\n\n# B\n\nparagraph three."

    chunks = ChunkingService().split(
        text,
        "structure_aware",
        {"targetChars": 20, "maxChars": 28, "minChars": 5, "overlapChars": 0},
    )

    assert "paragraph" not in chunks[0].content or chunks[0].content.endswith("\n\n")
    assert "# B" in "".join(chunk.content for chunk in chunks)
