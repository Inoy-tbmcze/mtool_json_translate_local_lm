# MTool JSON Translator

Translates Japanese game localization JSON files (MTool, RPG Maker, Unity text dumps) into English using a local LLM server (LM Studio or any OpenAI-compatible API).

The system preserves JSON dictionary structures, game context, character voices, and UI markers across three modular stages: **Cleanup**, **Translation**, and **Validation**.

## High-Performance Architecture

The core pipeline is tuned for maximum throughput and minimal memory overhead:
- **Native x86-64 Machine Code Engine**: JIT-allocated machine code routines via Win32 `VirtualAlloc` for zero-overhead ASCII identifier scanning, repeated byte detection, and 256-LUT symbol counting, with automatic fallback for non-x86 platforms.
- **L1-Cache Unicode BMP Bitmask**: 8,192-byte direct bitmask for $O(1)$ classification across all 65,536 Basic Multilingual Plane characters.
- **Fast-Rejection Text Processing**: Zero-allocation substring short-circuit checks skipping string mutation for >95% of standard game lines.
- **SIMD JSON Serialization**: Native C/Rust accelerated JSON serialization and deserialization via `orjson`.
- **Low-Latency Network Socket Tuning**: Persistent HTTP connection pools with `TCP_NODELAY` (Nagle's algorithm disabled) to eliminate packet buffering latency on loopback (`127.0.0.1`) local LLM calls.

## Requirements

- Python 3.10 or newer (Windows AMD64 recommended for native machine code kernels)
- Local LLM server running at `http://127.0.0.1:1234` (LM Studio or OpenAI-compatible)
- Suggested models:
  - `gemma-4-e4b` for text cleaning and translation audit
  - `google/gemma-4-12b-qat` for translation

## Setup

Install runtime dependencies:

```powershell
pip install -r requirements.txt
```

Optional editable install for the `mtool-translate` CLI and development tools (SCA/linters):

```powershell
pip install -e ".[dev]"
```

## Quick Start

1. Place your raw source JSON file into `data/raw/` (e.g., `data/raw/ManualTransFile.json`).
2. Run the complete pipeline:

```powershell
python main.py pipeline -i data/raw/ManualTransFile.json -y
```

All processed output files are saved to `data/processed/`.

## Pipeline Stages

### 1. Clean
Strips engine identifiers, developer comments, file paths, and non-Japanese lines while protecting dialogue, items, and UI labels:

```powershell
python main.py clean -i data/raw/ManualTransFile.json
```

Outputs:
- `data/processed/<name>_cleaned.json`
- `data/processed/<name>_quarantine.json`

### 2. Translate
Generates a character and tone Translation Blueprint, applies common dictionary matches, and translates batches with token-aware chunking:

```powershell
python main.py translate -i data/processed/ManualTransFile_cleaned.json
```

Outputs:
- `data/processed/<name>_translated.json`
- `data/processed/summary.txt`

### 3. Validate
Audits translation quality against Japanese source text and separates passed lines from lines needing retranslation:

```powershell
python main.py validate -i data/processed/<name>_translated.json
```

Outputs:
- `data/processed/<name>_validated.json` (passed translations)
- `data/processed/<name>_retranslate.json` (failed lines reset to Japanese for second pass)

## Development & Static Code Analysis

The codebase enforces strict static analysis standards:

```powershell
# Run full static analysis suite (Ruff, Mypy, Pylint 10.00/10, Isort)
.\Make.ps1 sca

# Format code with Ruff and Isort
.\Make.ps1 format

# Automatically apply safe fixes
.\Make.ps1 fix
```

## Configuration

Settings for API endpoints, models, token limits, batch sizes, and worker thread counts live in `config.json`.
