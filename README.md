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
- **Automated Model Lifecycle Management**: Native LM Studio REST manager detects server availability, dynamically unloads inactive models, and loads required models with stage-specific hardware parameters (`context_length`, `eval_batch_size`, `parallel`, `flash_attention`, `offload_kv_cache_to_gpu`). Gracefully falls back for mock or third-party OpenAI-compatible endpoints.

## Requirements

- Python 3.10 or newer (Windows AMD64 recommended for native machine code kernels)
- Local LLM server running at `http://127.0.0.1:1234` (LM Studio or OpenAI-compatible)
- Default models:
  - `gemma-4-e4b-uncensored-hauhaucs-aggressive` for text cleaning (`cleanup`) and translation audit (`validation`)
  - `gemma4-12b-qat-uncensored-hauhaucs-balanced` for translation blueprint generation (`summary`) and batch translation (`translation`)

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
python main.py translate -i data/processed/ManualTransFile_cleaned.json -y
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

## Testing & Verification

Deterministic offline test harness and unit test suites:

```powershell
# Run deterministic pipeline performance and accuracy harness
python tests/test_pipeline_harness.py --json

# Run unit test suite
python -m unittest discover tests
```

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

Pipeline settings are configured modularly in `config.json` with separate blocks for `cleanup`, `summary`, `translation`, and `validation`.

### Modular Sections
- **`cleanup`**: Stage 1 & 2 pre-processing, heuristic Japanese character ratios, regex filters, and LLM dev-junk classification. Uses `gemma-4-e4b-uncensored-hauhaucs-aggressive`.
- **`summary`**: Extraction of character personas, tone, and setting blueprint from raw dialogue. Uses `gemma4-12b-qat-uncensored-hauhaucs-balanced`.
- **`translation`**: Main batch translation engine with token-aware chunking and dictionary pre-translation. Uses `gemma4-12b-qat-uncensored-hauhaucs-balanced`.
- **`validation`**: Stage 3 semantic verification auditing target English against Japanese source text. Uses `gemma-4-e4b-uncensored-hauhaucs-aggressive`.

### Configurable Parameters
Each stage section supports independent configuration for:
- **Sampling Parameters**: `temperature`, `max_tokens`, `top_p`, `top_k`, `min_p`, `repeat_penalty`, `presence_penalty`, `frequency_penalty`.
- **Hardware & Model Load Parameters**: `context_length`, `eval_batch_size`, `physical_batch_size`, `parallel`, `flash_attention`, `offload_kv_cache_to_gpu`, `context_checkpoints`, `ttl`.
- **Operational Parameters**: `api_endpoint`, `api_key`, `batch_size`, `max_workers`, `save_interval`, `request_timeout`.
