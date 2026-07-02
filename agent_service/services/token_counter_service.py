from __future__ import annotations

from typing import Protocol


class TokenCounterService(Protocol):
    def count_tokens(self, text: str | None) -> int: ...


class HeuristicTokenCounterService:
    def count_tokens(self, text: str | None) -> int:
        if not text or not text.strip():
            return 0

        ascii_count = 0
        cjk_count = 0
        other_count = 0
        for char in text:
            if char.isspace():
                continue
            if ord(char) <= 0x7F:
                ascii_count += 1
            elif _is_cjk(char):
                cjk_count += 1
            else:
                other_count += 1

        total = (ascii_count + 3) // 4
        total += cjk_count
        total += (other_count + 1) // 2
        return max(total, 1)


def _is_cjk(char: str) -> bool:
    code = ord(char)
    ranges = (
        (0x3400, 0x4DBF),
        (0x4E00, 0x9FFF),
        (0xF900, 0xFAFF),
        (0x20000, 0x2A6DF),
        (0x2A700, 0x2B73F),
        (0x2B740, 0x2B81F),
        (0x2B820, 0x2CEAF),
        (0x2CEB0, 0x2EBEF),
        (0x30000, 0x3134F),
        (0x31350, 0x323AF),
        (0x3040, 0x309F),
        (0x30A0, 0x30FF),
        (0x31F0, 0x31FF),
        (0xAC00, 0xD7AF),
        (0x1100, 0x11FF),
        (0x3130, 0x318F),
        (0x3000, 0x303F),
        (0xFF00, 0xFFEF),
    )
    return any(start <= code <= end for start, end in ranges)
