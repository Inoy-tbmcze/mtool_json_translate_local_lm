"""Shared utility functions and regex definitions for Japanese text processing."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Set
from json_repair import repair_json

DEFAULT_MIN_JAPANESE_RATIO = 0.8

DEFAULT_JP_SYMBOLS = [
    "）", "」", "…", "（", "「", "『", "』", "【", "】",
    "・", "！", "？", "〜", "ー", "、", "。"
]

FILE_EXTENSIONS = (
    ".png", ".jpg", ".jpeg", ".bmp", ".tga", ".webp",
    ".wav", ".mp3", ".ogg", ".flac", ".webm", ".mp4", ".ogv", ".avi", ".bik", ".bk2",
    ".cpp", ".h", ".cs", ".py", ".json", ".xml", ".asset", ".mat", ".prefab", ".txt"
)

PURE_ASCII_IDENTIFIER_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\.\(\)\s]+$")

ENGINE_KEY_RE = re.compile(
    r"(?:.*フレーム\s*\d+$|"
    r"^event\d+【\d+】$|"
    r"^[a-zA-Z0-9_]{4,}_[\u3040-\u30ff\u4e00-\u9faf]|"
    r"^(?:event|pers|scene|cutscene)\d*_[0-9a-zA-Z_]+$|"
    r"^\d+_\d+_[\u3040-\u30ff\u4e00-\u9faf]|"
    r"^(?:sound|voice|snd|bgm|se)[\\/])",
    re.IGNORECASE
)

FANTASY_ITEM_PATTERN = re.compile(
    r".*?(?:の花|の草|の薬|の種|の根|の芽|の果実|の石|の剣|の盾|の鎧|の指輪|の巻物|の鍵|の壺|の瓶|の尾|の角|の羽|の皮|の骨)$"
)

KATAKANA_WORD_PATTERN = re.compile(r"^[\u30A0-\u30FF\u30FC\u30FB\s]{2,}$")

DEV_COMMENT_RE = re.compile(
    r"^\s*(?://|/\*|#|<!--|【(?:開発|仕様|デバッグ|テスト|メモ|TODO|FIXME|仮|作業用|消去予定|実装予定|処理|補足)】|"
    r"(?:TODO|FIXME|DEBUG|HACK|BUG|NOTE|メモ|仮置き|未実装|要修正|後で修正|仕様|開発メモ)\s*[:：])",
    re.IGNORECASE
)

JAPANESE_SENTENCE_PUNCTUATION = ("。", "！", "？", "…", "...", "」", "♪", "〜")

JP_CHAR_PATTERN = re.compile(r"[\u3040-\u30ff\u4e00-\u9faf]")


def is_protected_sentence(text: str) -> bool:
    """Returns True if text contains full Japanese sentence or dialogue punctuation."""
    s = text.strip() if text else ""
    return any(symbol in s for symbol in JAPANESE_SENTENCE_PUNCTUATION)


def is_protected_short_ui_label(text: str) -> bool:
    """Returns True for short Japanese UI labels, skill names, and menu items."""
    s = text.strip() if text else ""
    if 0 < len(s) <= 10 and JP_CHAR_PATTERN.search(s):
        code_symbols = ("/", "\\", "{", "}", "=", "<", ">", "//", "/*")
        if not any(sym in s for sym in code_symbols):
            return True
    return False


def is_protected_game_item(text: str) -> bool:
    """Returns True if string matches standard fantasy item naming patterns."""
    s = text.strip() if text else ""
    if len(s) <= 20 and FANTASY_ITEM_PATTERN.match(s):
        return True
    return False


def is_protected_katakana_word(text: str) -> bool:
    """Returns True if string is pure Katakana game vocabulary."""
    s = text.strip() if text else ""
    return bool(KATAKANA_WORD_PATTERN.match(s))


def load_japanese_symbols(symbols_path: Path) -> Set[str]:
    """Loads Japanese symbols from a JSON file."""
    if symbols_path.exists():
        try:
            with open(symbols_path, "r", encoding="utf-8") as f:
                symbols_list = json.load(f)
                if isinstance(symbols_list, list):
                    return set(symbols_list)
        except (json.JSONDecodeError, OSError):
            pass
    return set(DEFAULT_JP_SYMBOLS)


def build_japanese_regex(symbols: Set[str]) -> re.Pattern:
    """Builds regex matching Japanese characters and custom symbols."""
    escaped_symbols = "".join(re.escape(s) for s in symbols)
    return re.compile(rf"[\u3040-\u30ff\u4e00-\u9faf{escaped_symbols}]")


def calculate_japanese_ratio(text: str, jp_regex: re.Pattern) -> float:
    """Calculates the proportion of Japanese characters in a string."""
    if not text:
        return 0.0
    jp_char_count = len(jp_regex.findall(text))
    return jp_char_count / len(text)


def has_japanese_characters(text: str, jp_regex: re.Pattern) -> bool:
    """Checks if text contains Japanese characters."""
    return bool(jp_regex.search(text))


def is_ascii_art_or_symbol_heavy(text: str, jp_regex: re.Pattern) -> bool:
    """Detects ASCII art or symbol-heavy lines."""
    if not text:
        return False
    symbol_chars = set(r"=-_*+#/\|~<>[]{}()!@$%^&:`';")
    symbol_count = sum(1 for c in text if c in symbol_chars)
    ratio = symbol_count / len(text)
    if len(text) > 5 and ratio > 0.5:
        return True
    if re.search(r"(.)\1{4,}", text) and not has_japanese_characters(text, jp_regex):
        return True
    return False


def clean_japanese_text(text: str) -> str:
    """Replaces full-width symbols with standard equivalents for translation readability."""
    return text.replace("…", "...").replace("！", "!").replace("”", '"').replace("“", '"')


def parse_llm_json_response(response_text: str) -> dict:
    """Parses raw text from LLM into a dictionary using json_repair."""
    if not response_text or not response_text.strip():
        raise ValueError("Response text is empty or whitespace.")

    repaired_json_str = repair_json(response_text)
    parsed = json.loads(repaired_json_str)

    if isinstance(parsed, dict):
        return parsed

    raise ValueError("Parsed output is not a dictionary.")


def parse_json_array_safely(content: str) -> list:
    """Parses JSON array and repairs truncated strings."""
    content = content.strip()
    if "```" in content:
        content = re.sub(r"```(?:json)?|```", "", content).strip()

    json_match = re.search(r"\[.*\]", content, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except (json.JSONDecodeError, ValueError):
            pass

    start_idx = content.find("[")
    if start_idx != -1:
        truncated = content[start_idx:]
        last_comma = truncated.rfind(",")
        if last_comma != -1:
            repaired = truncated[:last_comma] + "]"
            try:
                return json.loads(repaired)
            except (json.JSONDecodeError, ValueError):
                pass

    return []
