"""
Stage 3 Validation Entrypoint (Backward Compatible Wrapper).
Delegates to mtool_translator.validator.
"""

import sys
from pathlib import Path

from mtool_translator.validator import (
    call_batch_validation,
    process_validation,
    save_progress,
)

# Ensure src/ is on sys.path for direct script execution
_SRC_DIR = Path(__file__).resolve().parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

__all__ = [
    "process_validation",
    "call_batch_validation",
    "save_progress",
]

if __name__ == "__main__":
    process_validation()
