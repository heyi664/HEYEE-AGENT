from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass
from typing import Any

from agent_service.schemas.chunking import TextChunk

_HEADING_RE = re.compile(r"^#{1,6}\s+.*$")
_CODE_FENCE_RE = re.compile(r"^```.*$")
_ATOMIC_IMAGE_RE = re.compile(r'^!\[[^\]]*]\([^)]+\)(?:\s*"[^"]*")?\s*$')
_ATOMIC_LINK_RE = re.compile(r"^\[[^\]]+]\([^)]+\)\s*$")


@dataclass(frozen=True)
class _Block:
    kind: str
    start: int
    end: int


class ChunkingService:
    def split(self, text: str, strategy: str, config: dict[str, Any]) -> list[TextChunk]:
        if not text or not text.strip():
            raise ValueError("document text is empty")
        if strategy == "fixed_size":
            return self._split_fixed_size(text, config)
        if strategy == "structure_aware":
            return self._split_structure_aware(text, config)
        raise ValueError(f"unsupported chunk strategy: {strategy}")

    def _split_fixed_size(self, text: str, config: dict[str, Any]) -> list[TextChunk]:
        normalized = _normalize_for_fixed_size(text)
        chunk_size = _positive_int(config.get("chunkSize"), 0) or _positive_int(
            config.get("targetChars"), 1400
        )
        if chunk_size == -1:
            return [_make_chunk(0, normalized, {"startChar": 0, "endChar": len(normalized)})]
        overlap = _non_negative_int(config.get("overlapSize"), -1)
        if overlap < 0:
            overlap = _non_negative_int(config.get("overlapChars"), 0)
        if chunk_size > 1:
            overlap = min(overlap, chunk_size - 1)
        else:
            overlap = 0

        chunks: list[TextChunk] = []
        start = 0
        last_end = -1
        while start < len(normalized):
            target_end = min(start + chunk_size, len(normalized))
            end = _adjust_to_boundary(normalized, start, target_end, overlap)
            if end <= start or end <= last_end:
                end = target_end
            content = normalized[start:end]
            if content.strip():
                chunks.append(
                    _make_chunk(len(chunks), content, {"startChar": start, "endChar": end})
                )
            last_end = end
            if end >= len(normalized):
                break
            next_start = max(0, end - overlap)
            if next_start <= start:
                next_start = end
            start = next_start
        return chunks

    def _split_structure_aware(self, text: str, config: dict[str, Any]) -> list[TextChunk]:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        target = _positive_int(config.get("targetChars"), 1400)
        max_chars = _positive_int(config.get("maxChars"), max(target, 1800))
        min_chars = _non_negative_int(config.get("minChars"), 0)
        overlap = _non_negative_int(config.get("overlapChars"), 0)

        blocks = _segment_to_blocks(normalized)
        if not blocks:
            return [_make_chunk(0, normalized, {"startChar": 0, "endChar": len(normalized)})]
        ranges = _pack_blocks_to_chunks(blocks, min_chars, target, max_chars)
        return _materialize_structure_chunks(normalized, ranges, overlap)


def _make_chunk(index: int, content: str, metadata: dict[str, Any]) -> TextChunk:
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return TextChunk(
        chunk_id=uuid.uuid4().hex[:20],
        chunk_index=index,
        content=content,
        content_hash=content_hash,
        char_count=len(content),
        token_count=_estimate_token_count(content),
        metadata=metadata,
    )


def _adjust_to_boundary(text: str, start: int, target_end: int, overlap: int) -> int:
    if target_end <= start:
        return target_end
    max_lookback = min(overlap, target_end - start)
    if max_lookback <= 0:
        return target_end

    for offset in range(max_lookback + 1):
        pos = target_end - offset - 1
        if pos <= start:
            break
        if text[pos] == "\n":
            return pos + 1

    for offset in range(max_lookback + 1):
        pos = target_end - offset - 1
        if pos <= start:
            break
        if text[pos] in {"。", "！", "？"}:
            return pos + 1

    for offset in range(max_lookback + 1):
        pos = target_end - offset - 1
        if pos <= start:
            break
        if text[pos] in {".", "!", "?"}:
            next_pos = pos + 1
            if next_pos >= len(text) or text[next_pos].isspace():
                return next_pos
    return target_end


def _normalize_for_fixed_size(text: str) -> str:
    source = text.replace("\r", "")
    output: list[str] = []
    in_url = False
    index = 0
    while index < len(source):
        if not in_url and _looks_like_url_start(source, index):
            in_url = True
        char = source[index]
        if in_url:
            if char.isspace():
                next_index = index
                saw_newline = False
                while next_index < len(source) and source[next_index].isspace():
                    saw_newline = saw_newline or source[next_index] == "\n"
                    next_index += 1
                prev = source[index - 1] if index > 0 else ""
                next_char = source[next_index] if next_index < len(source) else ""
                if saw_newline and next_char and _should_join_broken_url(
                    prev, next_char, source, next_index
                ):
                    index = next_index
                    continue
                output.append(source[index:next_index])
                in_url = False
                index = next_index
                continue
            output.append(char)
            if not _is_url_char(char):
                in_url = False
            index += 1
            continue
        if char == "\n":
            prev = source[index - 1] if index > 0 else ""
            next_char = source[index + 1] if index + 1 < len(source) else ""
            if _is_cjk_word_char(prev) and _is_cjk_word_char(next_char):
                index += 1
                continue
        output.append(char)
        index += 1
    return "".join(output)


def _segment_to_blocks(text: str) -> list[_Block]:
    blocks: list[_Block] = []
    pos = 0
    in_fence = False
    fence_start = -1
    in_para = False
    para_start = -1
    while pos < len(text):
        line_end = text.find("\n", pos)
        if line_end < 0:
            line_end = len(text)
        line_end_nl = line_end + 1 if line_end < len(text) and text[line_end] == "\n" else line_end
        line = text[pos:line_end]
        trimmed = line.rstrip(" \t")

        if not in_fence and _CODE_FENCE_RE.match(trimmed):
            if in_para:
                blocks.append(_Block("PARA", para_start, pos))
                in_para = False
            in_fence = True
            fence_start = pos
            pos = line_end_nl
            continue
        if in_fence:
            if _CODE_FENCE_RE.match(trimmed):
                blocks.append(_Block("CODE", fence_start, line_end_nl))
                in_fence = False
            pos = line_end_nl
            continue
        if not trimmed:
            if in_para:
                blocks.append(_Block("PARA", para_start, pos))
                in_para = False
            pos = line_end_nl
            continue
        if _HEADING_RE.match(trimmed):
            if in_para:
                blocks.append(_Block("PARA", para_start, pos))
                in_para = False
            blocks.append(_Block("HEADING", pos, line_end_nl))
            pos = line_end_nl
            continue
        if _ATOMIC_IMAGE_RE.match(trimmed) or _ATOMIC_LINK_RE.match(trimmed):
            if in_para:
                blocks.append(_Block("PARA", para_start, pos))
                in_para = False
            blocks.append(_Block("ATOMIC", pos, line_end_nl))
            pos = line_end_nl
            continue
        if not in_para:
            in_para = True
            para_start = pos
        pos = line_end_nl
    if in_fence:
        blocks.append(_Block("CODE", fence_start, len(text)))
    elif in_para:
        blocks.append(_Block("PARA", para_start, len(text)))
    return _coalesce_trailing_blanks(blocks, text)


def _coalesce_trailing_blanks(blocks: list[_Block], text: str) -> list[_Block]:
    if not blocks:
        return blocks
    output: list[_Block] = []
    previous = blocks[0]
    for current in blocks[1:]:
        if _is_all_blank(text, previous.end, current.start):
            previous = _Block(previous.kind, previous.start, current.start)
        output.append(previous)
        previous = current
    output.append(previous)
    return output


def _pack_blocks_to_chunks(
    blocks: list[_Block], min_chars: int, target_chars: int, max_chars: int
) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    index = 0
    while index < len(blocks):
        chunk_start = blocks[index].start
        chunk_end = blocks[index].end
        size = chunk_end - chunk_start
        next_index = index + 1
        while next_index < len(blocks):
            block = blocks[next_index]
            after_add = block.end - chunk_start
            if after_add <= max_chars:
                chunk_end = block.end
                size = after_add
                next_index += 1
            else:
                if size < min_chars:
                    chunk_end = block.end
                    next_index += 1
                break
        ranges.append((chunk_start, chunk_end))
        index = next_index

    if len(ranges) >= 2:
        last_start, last_end = ranges[-1]
        if last_end - last_start < min(min_chars, target_chars // 2):
            prev_start, _prev_end = ranges[-2]
            if last_end - prev_start <= max_chars * 2:
                ranges[-2] = (prev_start, last_end)
                ranges.pop()
    return ranges


def _materialize_structure_chunks(
    text: str, ranges: list[tuple[int, int]], overlap: int
) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    prev_tail = ""
    for start, end in ranges:
        body = text[start:end]
        if overlap > 0 and prev_tail:
            body = prev_tail + body
        chunks.append(_make_chunk(len(chunks), body, {"startChar": start, "endChar": end}))
        prev_tail = body[-overlap:] if overlap > 0 else ""
    return chunks


def _positive_int(value: Any, default: int) -> int:
    return value if isinstance(value, int) and value > 0 else default


def _non_negative_int(value: Any, default: int) -> int:
    return value if isinstance(value, int) and value >= 0 else default


def _estimate_token_count(text: str) -> int:
    return max(1, len(text) // 2)


def _looks_like_url_start(text: str, index: int) -> bool:
    return text.startswith("http://", index) or text.startswith("https://", index)


def _should_join_broken_url(prev: str, next_char: str, text: str, next_index: int) -> bool:
    if _is_list_item_start(text, next_index):
        return False
    if prev == "." and next_char.isalpha():
        return True
    if prev in {"/", "?", "&", "=", "#", "%", "-", "_", ":"}:
        return True
    return next_char in {"/", "?", "&", "=", "#"}


def _is_list_item_start(text: str, index: int) -> bool:
    pos = index
    while pos < len(text) and text[pos] in {" ", "\t"}:
        pos += 1
    start = pos
    while pos < len(text) and text[pos].isdigit():
        pos += 1
    return pos > start and pos < len(text) and text[pos] in {".", "）", ")"}


def _is_url_char(char: str) -> bool:
    return char.isalnum() or char in "-._~:/?#[]@!$&'()*+,;=%"


def _is_cjk_word_char(char: str) -> bool:
    if not char or char.isspace() or _is_cjk_punctuation(char):
        return False
    code = ord(char)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0x20000 <= code <= 0x2A6DF
        or 0xF900 <= code <= 0xFAFF
        or 0xFF00 <= code <= 0xFFEF
    )


def _is_cjk_punctuation(char: str) -> bool:
    return char in "。，、；：！？（）【】《》“”‘’"


def _is_all_blank(text: str, start: int, end: int) -> bool:
    return all(char in {" ", "\t", "\r", "\n"} for char in text[start:end])
