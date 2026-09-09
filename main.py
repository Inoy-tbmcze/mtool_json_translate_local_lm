"""
MTool JSON Translator - Unified CLI & Translation Entry Point.
"""

import sys
from pathlib import Path

# Ensure src/ is on sys.path for direct script execution
_SRC_DIR = Path(__file__).resolve().parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from mtool_translator.cli import main
from mtool_translator.translator import (
    JSONTranslator,
    TokenAwareChunker,
    process_translation,
    clean_japanese_text,
    parse_llm_json_response
)

if __name__ == "__main__":
    main()
