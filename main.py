"""
MTool JSON Translator - Unified CLI & Translation Entry Point.
"""

import sys
from pathlib import Path

from mtool_translator.cli import main
from mtool_translator.translator import (
    JSONTranslator,
    TokenAwareChunker,
    clean_japanese_text,
    parse_llm_json_response,
    process_translation,
)

# Ensure src/ is on sys.path for direct script execution
_SRC_DIR = Path(__file__).resolve().parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

__all__ = [
    "JSONTranslator",
    "TokenAwareChunker",
    "clean_japanese_text",
    "main",
    "parse_llm_json_response",
    "process_translation",
]

if __name__ == "__main__":
    main()
