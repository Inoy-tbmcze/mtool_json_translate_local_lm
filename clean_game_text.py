"""
Stage 1 Cleaning Entrypoint (Backward Compatible Wrapper).
Delegates to mtool_translator.cleaner.
"""

import sys
from pathlib import Path

# Ensure src/ is on sys.path for direct script execution
_SRC_DIR = Path(__file__).resolve().parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# pylint: disable=wrong-import-position
from mtool_translator.cleaner import (
    process_json_file,
    is_stage1_junk,
    call_batch_classification,
    save_progress,
)
from mtool_translator.utils import (
    DEFAULT_MIN_JAPANESE_RATIO,
    DEFAULT_JP_SYMBOLS,
    FILE_EXTENSIONS,
    PURE_ASCII_IDENTIFIER_PATTERN,
    ENGINE_KEY_RE,
    FANTASY_ITEM_PATTERN,
    KATAKANA_WORD_PATTERN,
    DEV_COMMENT_RE,
    JAPANESE_SENTENCE_PUNCTUATION,
    JP_CHAR_PATTERN,
    is_protected_sentence,
    is_protected_short_ui_label,
    is_protected_game_item,
    is_protected_katakana_word,
    load_japanese_symbols,
    build_japanese_regex,
    calculate_japanese_ratio,
    has_japanese_characters,
    is_ascii_art_or_symbol_heavy,
    parse_json_array_safely,
)

__all__ = [
    "process_json_file",
    "is_stage1_junk",
    "call_batch_classification",
    "save_progress",
    "DEFAULT_MIN_JAPANESE_RATIO",
    "DEFAULT_JP_SYMBOLS",
    "FILE_EXTENSIONS",
    "PURE_ASCII_IDENTIFIER_PATTERN",
    "ENGINE_KEY_RE",
    "FANTASY_ITEM_PATTERN",
    "KATAKANA_WORD_PATTERN",
    "DEV_COMMENT_RE",
    "JAPANESE_SENTENCE_PUNCTUATION",
    "JP_CHAR_PATTERN",
    "is_protected_sentence",
    "is_protected_short_ui_label",
    "is_protected_game_item",
    "is_protected_katakana_word",
    "load_japanese_symbols",
    "build_japanese_regex",
    "calculate_japanese_ratio",
    "has_japanese_characters",
    "is_ascii_art_or_symbol_heavy",
    "parse_json_array_safely",
]

if __name__ == "__main__":
    process_json_file()
