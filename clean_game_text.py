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
    call_batch_classification,
    is_stage1_junk,
    process_json_file,
    save_progress,
)
from mtool_translator.utils import (
    DEFAULT_JP_SYMBOLS,
    DEFAULT_MIN_JAPANESE_RATIO,
    DEV_COMMENT_RE,
    ENGINE_KEY_RE,
    FILE_EXTENSIONS,
    JP_CHAR_PATTERN,
    KATAKANA_WORD_PATTERN,
    RPG_ESCAPE_CODE_RE,
    build_japanese_regex,
    calculate_japanese_ratio,
    has_japanese_characters,
    is_ascii_art_or_symbol_heavy,
    is_protected_game_item,
    is_protected_katakana_word,
    is_protected_sentence,
    is_protected_short_ui_label,
    load_japanese_symbols,
    parse_json_array_safely,
    strip_engine_escape_codes,
)

__all__ = [
    "DEFAULT_JP_SYMBOLS",
    "DEFAULT_MIN_JAPANESE_RATIO",
    "DEV_COMMENT_RE",
    "ENGINE_KEY_RE",
    "FILE_EXTENSIONS",
    "JP_CHAR_PATTERN",
    "KATAKANA_WORD_PATTERN",
    "RPG_ESCAPE_CODE_RE",
    "build_japanese_regex",
    "calculate_japanese_ratio",
    "call_batch_classification",
    "has_japanese_characters",
    "is_ascii_art_or_symbol_heavy",
    "is_protected_game_item",
    "is_protected_katakana_word",
    "is_protected_sentence",
    "is_protected_short_ui_label",
    "is_stage1_junk",
    "load_japanese_symbols",
    "parse_json_array_safely",
    "process_json_file",
    "save_progress",
    "strip_engine_escape_codes",
]

if __name__ == "__main__":
    process_json_file()
