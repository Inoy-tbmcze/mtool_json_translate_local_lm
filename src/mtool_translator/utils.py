"""Shared utility functions and regex definitions for Japanese text processing."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional, Set, Union

from json_repair import repair_json

try:
    import orjson

    def fast_json_loads(data: Union[str, bytes]) -> Any:
        return orjson.loads(data)

    def fast_json_dumps(obj: Any, indent: bool = False) -> str:
        opt = orjson.OPT_INDENT_2 if indent else 0
        return orjson.dumps(obj, option=opt).decode("utf-8")

    def fast_json_dumps_bytes(obj: Any, indent: bool = False) -> bytes:
        opt = orjson.OPT_INDENT_2 if indent else 0
        return orjson.dumps(obj, option=opt)

except ImportError:
    import json

    def fast_json_loads(data: Union[str, bytes]) -> Any:
        return json.loads(data)

    def fast_json_dumps(obj: Any, indent: bool = False) -> str:
        return json.dumps(obj, ensure_ascii=False, indent=2 if indent else None)

    def fast_json_dumps_bytes(obj: Any, indent: bool = False) -> bytes:
        return json.dumps(obj, ensure_ascii=False, indent=2 if indent else None).encode("utf-8")


def load_json_file(file_path: Union[str, Path]) -> Any:
    """Fast binary JSON file loader."""
    with open(file_path, "rb") as f:
        return fast_json_loads(f.read())


def dump_json_file(file_path: Union[str, Path], data: Any, indent: bool = True) -> None:
    """Fast binary JSON file dumper."""
    target = Path(file_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = fast_json_dumps_bytes(data, indent=indent)
    with open(target, "wb") as f:
        f.write(payload)


DEFAULT_MIN_JAPANESE_RATIO = 0.8

DEFAULT_JP_SYMBOLS = [
    "）",
    "」",
    "…",
    "（",
    "「",
    "『",
    "』",
    "【",
    "】",
    "・",
    "！",
    "？",
    "〜",
    "ー",
    "、",
    ")",
    "_",
    "(",
    "♡",
    "。",
]

FILE_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tga",
    ".webp",
    ".wav",
    ".mp3",
    ".ogg",
    ".flac",
    ".webm",
    ".mp4",
    ".ogv",
    ".avi",
    ".bik",
    ".bk2",
    ".cpp",
    ".h",
    ".cs",
    ".py",
    ".json",
    ".xml",
    ".asset",
    ".mat",
    ".prefab",
    ".txt",
)

PURE_ASCII_IDENTIFIER_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\.\(\)\s]+$")

ENGINE_KEY_RE = re.compile(
    r"(?:フレーム\s*\d+$|"
    r"^event\d+【\d+】$|"
    r"^[a-zA-Z0-9_]{4,}_[\u3040-\u30ff\u4e00-\u9faf]|"
    r"^(?:event|pers|scene|cutscene)\d*_[0-9a-zA-Z_]+$|"
    r"^\d+_\d+_[\u3040-\u30ff\u4e00-\u9faf]|"
    r"^(?:sound|voice|snd|bgm|se)[\\/])",
    re.IGNORECASE,
)

FANTASY_ITEM_PATTERN = re.compile(
    r".*?(?:の花|の草|の薬|の種|の根|の芽|の果実|の石|の剣|の盾|の鎧|の指輪|の巻物|の鍵|の壺|の瓶|の尾|の角|の羽|の皮|の骨)$"
)

FANTASY_SUFFIXES = (
    "の花",
    "の草",
    "の薬",
    "の種",
    "の根",
    "の芽",
    "の果実",
    "の石",
    "の剣",
    "の盾",
    "の鎧",
    "の指輪",
    "の巻物",
    "の鍵",
    "の壺",
    "の瓶",
    "の尾",
    "の角",
    "の羽",
    "の皮",
    "の骨",
)

KATAKANA_WORD_PATTERN = re.compile(r"^[\u30A0-\u30FF\u30FC\u30FB\s]{2,}$")

DEV_COMMENT_RE = re.compile(
    r"^\s*(?://|/\*|#|<!--|【(?:開発|仕様|デバッグ|テスト|メモ|TODO|FIXME|仮|作業用|消去予定|実装予定|処理|補足)】|"
    r"(?:TODO|FIXME|DEBUG|HACK|BUG|NOTE|メモ|仮置き|未実装|要修正|後で修正|仕様|開発メモ)\s*[:：])",
    re.IGNORECASE,
)

DEV_COMMENT_STARTERS = frozenset("/#<【tfdhbnメ仮未要後仕開TFDHBN")

JAPANESE_SENTENCE_PUNCTUATION = ("。", "！", "？", "…", "...", "」", "♪", "〜")

_SENTENCE_PUNCT_RE = re.compile(r"[。！？…」♪〜]|\.\.\.")
_CODE_CHARS = frozenset(r"/\}{}=<>")
_SYMBOL_CHARS = frozenset(r"=-_*+#/\|~<>[]{}()!@$%^&:`';")
_REPEATED_CHAR_RE = re.compile(r"(.)\1{4,}")

JP_CHAR_PATTERN = re.compile(r"[\u3040-\u30ff\u4e00-\u9faf]")


def is_protected_sentence(text: str) -> bool:
    """Returns True if text contains full Japanese sentence or dialogue punctuation."""
    return bool(text) and bool(_SENTENCE_PUNCT_RE.search(text))


def is_protected_short_ui_label(text: str) -> bool:
    """Returns True for short Japanese UI labels, skill names, and menu items."""
    if not text:
        return False
    s = text.strip()
    if 0 < len(s) <= 10 and JP_CHAR_PATTERN.search(s):
        return not any(c in _CODE_CHARS for c in s)
    return False


def is_protected_game_item(text: str) -> bool:
    """Returns True if string matches standard fantasy item naming patterns."""
    if not text:
        return False
    s = text.strip()
    return len(s) <= 20 and s.endswith(FANTASY_SUFFIXES)


def is_protected_katakana_word(text: str) -> bool:
    """Returns True if string is pure Katakana game vocabulary."""
    if not text:
        return False
    s = text.strip()
    return bool(KATAKANA_WORD_PATTERN.match(s))


def load_japanese_symbols(symbols_path: Optional[Union[Path, str]] = None) -> Set[str]:
    """Loads Japanese symbols from a JSON file, falling back to default symbols."""
    if symbols_path is not None:
        target_path = Path(symbols_path)
        if target_path.exists():
            try:
                symbols_list = load_json_file(target_path)
                if isinstance(symbols_list, list):
                    return set(symbols_list)
            except Exception:
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
    jp_char_count = jp_regex.subn("", text)[1]
    return jp_char_count / len(text)


def has_japanese_characters(text: str, jp_regex: re.Pattern) -> bool:
    """Checks if text contains Japanese characters."""
    return bool(text) and bool(jp_regex.search(text))


def is_ascii_art_or_symbol_heavy(text: str, jp_regex: re.Pattern) -> bool:
    """Detects ASCII art or symbol-heavy lines."""
    if not text:
        return False
    n = len(text)
    if n > 5:
        symbol_count = sum(1 for c in text if c in _SYMBOL_CHARS)
        if (symbol_count / n) > 0.5:
            return True
    if _REPEATED_CHAR_RE.search(text) and not jp_regex.search(text):
        return True
    return False


def clean_japanese_text(text: str) -> str:
    """Replaces full-width symbols with standard equivalents for translation readability."""
    if not text:
        return ""
    if "…" in text:
        text = text.replace("…", "...")
    if "！" in text:
        text = text.replace("！", "!")
    if "”" in text:
        text = text.replace("”", '"')
    if "“" in text:
        text = text.replace("“", '"')
    return text


def parse_llm_json_response(response_text: str) -> dict:
    """Parses raw text from LLM into a dictionary using fast path with json_repair fallback."""
    if not response_text or not response_text.strip():
        raise ValueError("Response text is empty or whitespace.")

    raw = response_text.strip()
    if raw.startswith("```"):
        first_nl = raw.find("\n")
        last_fence = raw.rfind("```")
        if first_nl != -1 and last_fence > first_nl:
            raw = raw[first_nl + 1 : last_fence].strip()

    try:
        parsed = fast_json_loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    repaired_json_str = repair_json(raw)
    parsed = fast_json_loads(repaired_json_str)
    if isinstance(parsed, dict):
        return parsed

    raise ValueError("Parsed output is not a dictionary.")


def parse_json_array_safely(content: str) -> list:
    """Parses JSON array and repairs truncated strings."""
    if not content:
        return []

    s = content.strip()
    start_idx = s.find("[")
    if start_idx != -1:
        end_idx = s.rfind("]")
        if end_idx > start_idx:
            try:
                res = fast_json_loads(s[start_idx : end_idx + 1])
                if isinstance(res, list):
                    return res
            except Exception:
                pass

        last_comma = s.rfind(",")
        if last_comma > start_idx:
            repaired = s[start_idx:last_comma] + "]"
            try:
                res = fast_json_loads(repaired)
                if isinstance(res, list):
                    return res
            except Exception:
                pass

    try:
        repaired = repair_json(s)
        res = fast_json_loads(repaired)
        if isinstance(res, list):
            return res
    except Exception:
        pass

    return []
