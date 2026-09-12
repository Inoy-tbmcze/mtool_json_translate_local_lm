"""Shared utility functions and regex definitions for Japanese text processing."""

from __future__ import annotations

import contextlib
import json
import os
import re
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .native_core import fast_count_symbols, fast_has_repeated_chars

try:
    import orjson

    def fast_json_loads(data: str | bytes) -> Any:
        """Fast JSON deserializer using high-performance orjson."""
        return orjson.loads(data)

    def fast_json_dumps(obj: Any, indent: bool = False) -> str:
        """Fast JSON serializer to string using high-performance orjson."""
        opt = orjson.OPT_INDENT_2 if indent else 0
        return orjson.dumps(obj, option=opt).decode("utf-8")

    def fast_json_dumps_bytes(obj: Any, indent: bool = False) -> bytes:
        """Fast JSON serializer to bytes using high-performance orjson."""
        opt = orjson.OPT_INDENT_2 if indent else 0
        return orjson.dumps(obj, option=opt)

except ImportError:

    def fast_json_loads(data: str | bytes) -> Any:
        """Fallback JSON deserializer using standard library json."""
        return json.loads(data)

    def fast_json_dumps(obj: Any, indent: bool = False) -> str:
        """Fallback JSON serializer to string using standard library json."""
        return json.dumps(obj, ensure_ascii=False, indent=2 if indent else None)

    def fast_json_dumps_bytes(obj: Any, indent: bool = False) -> bytes:
        """Fallback JSON serializer to bytes using standard library json."""
        return json.dumps(obj, ensure_ascii=False, indent=2 if indent else None).encode("utf-8")


def load_json_file(file_path: str | Path) -> Any:
    """Fast binary JSON file loader."""
    with open(file_path, "rb") as file_handle:
        return fast_json_loads(file_handle.read())


def dump_json_file(file_path: str | Path, data: Any, indent: bool = True) -> None:
    """Fast binary JSON file dumper with crash-resilient atomic replacement."""
    target = Path(file_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = fast_json_dumps_bytes(data, indent=indent)
    tmp_target = target.with_name(f".{target.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with open(tmp_target, "wb") as file_handle:
            file_handle.write(payload)
        os.replace(tmp_target, target)
    except BaseException:
        if tmp_target.exists():
            with contextlib.suppress(OSError):
                tmp_target.unlink()
        raise


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

FILE_EXTENSIONS_SET = frozenset(FILE_EXTENSIONS)

ENGINE_KEY_RE = re.compile(
    r"(?:フレーム\s*\d+$|"
    r"^event\d+【\d+】$|"
    r"^[a-zA-Z0-9_]{4,}_[\u3040-\u30ff\u4e00-\u9faf]|"
    r"^(?:event|pers|scene|cutscene)\d*_[0-9a-zA-Z_]+$|"
    r"^\d+_\d+_[\u3040-\u30ff\u4e00-\u9faf]|"
    r"^(?:sound|voice|snd|bgm|se)[\\/])",
    re.IGNORECASE,
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

RPG_ESCAPE_CODE_RE = re.compile(
    r"\\[A-Za-z]+\[[^\]\r\n]+\]|\\[!^.><{}_|\$\\]|\\[Gg](?![A-Za-z0-9_])"
)

_SENTENCE_PUNCT_RE = re.compile(r"[。！？…」♪〜]|\.\.\.")
_CODE_CHARS = frozenset(r"/\\}{}=<>")

JP_CHAR_PATTERN = re.compile(r"[\u3040-\u30ff\u4e00-\u9faf]")
_RE_TRAILING_COMMA = re.compile(r",\s*([}\]])")


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


def strip_engine_escape_codes(text: str) -> str:
    """Strips RPG Maker and MTool engine control codes from text.

    Uses fast substring check to bypass regex engine on strings without escape prefixes.
    """
    if not text or "\\" not in text:
        return text
    return RPG_ESCAPE_CODE_RE.sub("", text)


def load_japanese_symbols(symbols_path: Path | str | None = None) -> set[str]:
    """Loads Japanese symbols from a JSON file, falling back to default symbols."""
    if symbols_path is not None:
        target_path = Path(symbols_path)
        if target_path.exists():
            try:
                symbols_list = load_json_file(target_path)
                if isinstance(symbols_list, list):
                    return set(symbols_list)
            except (OSError, ValueError, TypeError):
                pass
    return set(DEFAULT_JP_SYMBOLS)


def build_japanese_regex(symbols: set[str]) -> re.Pattern:
    """Builds regex matching Japanese characters and custom symbols."""
    escaped_symbols = "".join(re.escape(s) for s in symbols)
    return re.compile(rf"[\u3040-\u30ff\u4e00-\u9faf{escaped_symbols}]")


_FILTER_NOISE_RE = re.compile(r"[\s\d\W_]")


def calculate_japanese_ratio(
    text: str, jp_regex: re.Pattern, exclude_noise: bool = True
) -> float:
    """Calculates the proportion of Japanese characters in a string.

    When exclude_noise is True (default), calculates ratio against meaningful
    linguistic characters (excluding whitespace, digits, and non-word symbols)
    to prevent valid Japanese text containing numbers or ASCII identifiers from
    being falsely quarantined.
    """
    if not text:
        return 0.0

    if not exclude_noise:
        jp_char_count = jp_regex.subn("", text)[1]
        return jp_char_count / len(text)

    filtered = _FILTER_NOISE_RE.sub("", text)
    target = filtered if filtered else text
    den = len(target)
    if den == 0:
        return 0.0
    jp_char_count = jp_regex.subn("", target)[1]
    return jp_char_count / den


def has_japanese_characters(text: str, jp_regex: re.Pattern) -> bool:
    """Checks if text contains Japanese characters."""
    return bool(text) and bool(jp_regex.search(text))


def is_ascii_art_or_symbol_heavy(text: str, jp_regex: re.Pattern) -> bool:
    """Detects ASCII art or symbol-heavy lines using low-level symbol counting."""
    if not text:
        return False
    n = len(text)
    if n > 5:
        symbol_count = fast_count_symbols(text)
        if (symbol_count / n) > 0.5:
            return True
    return bool(fast_has_repeated_chars(text, min_repeat=5) and not jp_regex.search(text))


def clean_japanese_text(text: str) -> str:
    """Replaces full-width symbols with standard equivalents with fast rejection check."""
    if not text:
        return ""
    if "…" not in text and "！" not in text and "”" not in text and "“" not in text:
        return text
    if "…" in text:
        text = text.replace("…", "...")
    if "！" in text:
        text = text.replace("！", "!")
    if "”" in text:
        text = text.replace("”", '"')
    if "“" in text:
        text = text.replace("“", '"')
    return text


def _strip_markdown_fences(text: str) -> str:
    """Strips outer markdown code block fences (e.g. ```json ... ```)."""
    if "```" not in text:
        return text

    f1 = text.find("```")
    nl = text.find("\n", f1)
    if nl != -1:
        f2 = text.rfind("```")
        return text[nl + 1 : f2].strip() if f2 > nl else text[nl + 1 :].strip()

    f2 = text.rfind("```")
    return text[f1 + 3 : f2].strip() if f2 > f1 + 3 else text[f1 + 3 :].strip()


def _extract_container_slice(text: str) -> tuple[int, int]:
    """Finds starting and ending indices of outermost JSON container."""
    if not text:
        return -1, -1

    n = len(text)
    first_char = text[0]
    last_char = text[-1]

    # Fast O(1) path for standard, well-formed JSON strings
    if (first_char == "{" or first_char == "[") and (last_char == "}" or last_char == "]"):
        return 0, n - 1

    if first_char == "{" or first_char == "[":
        start = 0
    else:
        idx_brace = text.find("{")
        idx_bracket = text.find("[")
        if idx_brace == -1:
            start = idx_bracket
        elif idx_bracket == -1:
            start = idx_brace
        else:
            start = min(idx_brace, idx_bracket)

    if start == -1:
        return -1, -1

    if last_char == "}" or last_char == "]":
        end = n - 1
    else:
        last_brace = text.rfind("}")
        last_bracket = text.rfind("]")
        end = max(last_brace, last_bracket)

    return start, end


# pylint: disable=too-many-instance-attributes,too-few-public-methods
class _JsonRepairMachine:
    """High-speed state-machine token reconstructor for malformed LLM JSON."""

    __slots__ = (
        "escaped",
        "i",
        "in_string",
        "is_key",
        "last_comma_idx",
        "last_non_ws",
        "n",
        "out",
        "quote_char",
        "s",
        "stack",
    )

    def __init__(self, s: str) -> None:
        self.s = s
        self.n = len(s)
        self.i = 0
        self.out: list[str] = []
        self.stack: list[str] = []
        self.in_string = False
        self.quote_char = '"'
        self.escaped = False
        self.last_non_ws = ""
        self.last_comma_idx = -1
        self.is_key = False

    def repair(self) -> str:
        """Executes single-pass state-machine repair sweep."""
        while self.i < self.n:
            ch = self.s[self.i]
            if self.in_string:
                self._handle_string_char(ch)
            else:
                self._handle_structural_char(ch)
        return self._finalize()

    def _handle_string_char(self, ch: str) -> None:
        if self.escaped:
            self.out.append(ch)
            self.escaped = False
            self.i += 1
            return

        if ch == "\\":
            self.out.append(ch)
            self.escaped = True
            self.i += 1
            return

        if ch == self.quote_char:
            j = self.i + 1
            while j < self.n and self.s[j] in (" ", "\t", "\r", "\n"):
                j += 1
            if j == self.n or self.s[j] in (",", "}", "]", ":"):
                self.out.append('"')
                self.in_string = False
                self.last_non_ws = '"'
            else:
                self.out.append('\\"')
            self.i += 1
            return

        if ch == "\n":
            self.out.append("\\n")
        elif ch == "\r":
            if self.i + 1 >= self.n or self.s[self.i + 1] != "\n":
                self.out.append("\\r")
        elif ch == "\t":
            self.out.append("\\t")
        elif ord(ch) < 0x20:
            self.out.append(f"\\u{ord(ch):04x}")
        else:
            self.out.append(ch)
        self.i += 1

    def _handle_structural_char(self, ch: str) -> None:
        if ch in (" ", "\t", "\r", "\n"):
            self.out.append(ch)
            self.i += 1
        elif ch == "/" and self.i + 1 < self.n:
            if self.s[self.i + 1] == "/":
                nl = self.s.find("\n", self.i + 2)
                self.i = self.n if nl == -1 else nl
            elif self.s[self.i + 1] == "*":
                end_c = self.s.find("*/", self.i + 2)
                self.i = self.n if end_c == -1 else end_c + 2
        elif ch in ('"', "'"):
            self.in_string = True
            self.quote_char = ch
            self.out.append('"')
            self.is_key = bool(
                self.stack and self.stack[-1] == "}" and self.last_non_ws in ("{", ",")
            )
            self.last_non_ws = '"'
            self.last_comma_idx = -1
            self.i += 1
        elif ch == "{":
            self.stack.append("}")
            self.out.append("{")
            self.last_non_ws = "{"
            self.last_comma_idx = -1
            self.i += 1
        elif ch == "[":
            self.stack.append("]")
            self.out.append("[")
            self.last_non_ws = "["
            self.last_comma_idx = -1
            self.i += 1
        elif ch in ("}", "]"):
            self._close_delimiter(ch)
            self.i += 1
        elif ch == ",":
            if self.last_non_ws not in ("{", "[", ",", ":"):
                self.last_comma_idx = len(self.out)
                self.out.append(",")
                self.last_non_ws = ","
            self.i += 1
        elif ch == ":":
            self.out.append(":")
            self.last_non_ws = ":"
            self.last_comma_idx = -1
            self.i += 1
        else:
            self._handle_literals_and_chars(ch)

    def _close_delimiter(self, closing: str) -> None:
        if self.last_comma_idx != -1:
            self.out[self.last_comma_idx] = ""
            self.last_comma_idx = -1
        if self.last_non_ws == ":":
            self.out.append('""')

        if self.stack:
            if self.stack[-1] == closing:
                self.stack.pop()
                self.out.append(closing)
            elif closing in self.stack:
                while self.stack and self.stack[-1] != closing:
                    self.out.append(self.stack.pop())
                if self.stack and self.stack[-1] == closing:
                    self.stack.pop()
                    self.out.append(closing)
            else:
                other = "]" if closing == "}" else "}"
                self.out.append(other)
                self.stack.pop()

        self.last_non_ws = closing

    def _handle_literals_and_chars(self, ch: str) -> None:
        if self.s.startswith("True", self.i):
            self.out.append("true")
            self.i += 4
            self.last_non_ws = "e"
        elif self.s.startswith("False", self.i):
            self.out.append("false")
            self.i += 5
            self.last_non_ws = "e"
        elif self.s.startswith("None", self.i):
            self.out.append("null")
            self.i += 4
            self.last_non_ws = "l"
        else:
            self.out.append(ch)
            self.last_non_ws = ch
            self.i += 1
        self.last_comma_idx = -1

    def _finalize(self) -> str:
        if self.in_string:
            if self.escaped:
                self.out.append("\\")
            self.out.append('"')
            self.last_non_ws = '"'
            if self.is_key:
                self.out.append(': ""')

        if self.last_comma_idx != -1:
            self.out[self.last_comma_idx] = ""

        if self.last_non_ws == ":":
            self.out.append('""')

        while self.stack:
            self.out.append(self.stack.pop())

        return "".join(self.out)


def repair_json_string(raw: str) -> str:
    """Repairs and extracts valid JSON from malformed or conversational LLM output."""
    if not raw or not raw.strip():
        return ""

    s = _strip_markdown_fences(raw.strip())
    start_idx, end_idx = _extract_container_slice(s)
    if start_idx == -1:
        return ""

    if end_idx > start_idx:
        candidate = s[start_idx : end_idx + 1]
        try:
            fast_json_loads(candidate)
            return candidate
        except (ValueError, TypeError, json.JSONDecodeError):
            cand_no_comma = _RE_TRAILING_COMMA.sub(r"\1", candidate)
            try:
                fast_json_loads(cand_no_comma)
                return cand_no_comma
            except (ValueError, TypeError, json.JSONDecodeError):
                pass

    return _JsonRepairMachine(s[start_idx:]).repair()


def parse_llm_json_response(response_text: str) -> dict:
    """Parses raw text from LLM into a dictionary using fast path with in-house repair."""
    if not response_text or not response_text.strip():
        raise ValueError("Response text is empty or whitespace.")

    raw = _strip_markdown_fences(response_text.strip())

    try:
        parsed = fast_json_loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except (ValueError, TypeError, json.JSONDecodeError):
        pass

    try:
        repaired_json_str = repair_json_string(raw)
        parsed = fast_json_loads(repaired_json_str)
        if isinstance(parsed, dict):
            return parsed
    except (ValueError, TypeError, json.JSONDecodeError) as err:
        raise ValueError("Failed to repair JSON output.") from err

    raise ValueError("Parsed output is not a dictionary.")


def parse_json_array_safely(content: str) -> list:
    """Parses JSON array and repairs truncated strings using in-house high-speed repair."""
    if not content:
        return []

    s = _strip_markdown_fences(content.strip())
    start_idx = s.find("[")
    if start_idx != -1:
        end_idx = s.rfind("]")
        if end_idx > start_idx:
            try:
                res = fast_json_loads(s[start_idx : end_idx + 1])
                if isinstance(res, list):
                    return res
            except (ValueError, TypeError, json.JSONDecodeError):
                pass

        last_comma = s.rfind(",")
        if last_comma > start_idx:
            repaired = s[start_idx:last_comma] + "]"
            try:
                res = fast_json_loads(repaired)
                if isinstance(res, list):
                    return res
            except (ValueError, TypeError, json.JSONDecodeError):
                pass

    try:
        repaired = repair_json_string(s)
        res = fast_json_loads(repaired)
        if isinstance(res, list):
            return res
    except (ValueError, TypeError, json.JSONDecodeError):
        pass

    return []


def run_batch_wave(
    wave: list[list[Any]],
    executor: ThreadPoolExecutor,
    worker_fn: Callable[..., Any],
    config: dict[str, Any],
    apply_result_fn: Callable[[list[Any], Any], None],
    session: Any = None,
    error_tag: str = "Worker thread",
) -> None:
    """Executes a parallel wave of batches across threads with resilient retry fallback."""
    future_to_batch = {
        executor.submit(worker_fn, batch, config, session): batch
        for batch in wave
    }
    for future in as_completed(future_to_batch):
        batch = future_to_batch[future]
        try:
            results = future.result()
        except Exception as err:
            print(f"\nWarning: {error_tag} failed ({err}). Retrying batch...")
            results = worker_fn(batch, config, session)

        apply_result_fn(batch, results)

