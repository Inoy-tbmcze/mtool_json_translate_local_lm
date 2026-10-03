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


```powershell
python main.py pipeline -i data/raw/ManualTransFile.json -t data/processed/ManualTransFile_translated.json -y
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

## Incremental Updates (Diff & Merge)

When a game is updated with new content, translate only the newly introduced strings instead of re-processing the entire script:

### Automated End-to-End Pipeline with `-t`
Pass `-t / --translated` to the `pipeline` command to automate the complete update lifecycle (Diff -> Clean -> Translate -> Validate -> Merge) in one step:

```powershell
python main.py pipeline -i data/raw/ManualTransFile.json -t data/processed/ManualTransFile_translated.json -y
```

> **Note on Retranslation Recovery:**
> Any lines flagged as needing retranslation during Stage 3 validation are automatically routed back to Stage 2 for a second translation pass using the cached `summary.txt` (Translation Blueprint). To maximize playable in-game English coverage, all successfully retranslated lines are merged directly into the final translation files (`ManualTransFile_translated.json` and master reference). Truly untranslatable lines (where output matches source) remain in `_retranslate.json` for manual review.

### Manual Step-by-Step Workflow

#### 1. Diff (Filter Out Already Translated Keys)
Compares updated `ManualTransFile.json` against an existing `ManualTransFile_translated.json` and removes already translated keys in-place (or writes to a separate file via `-o`):

```powershell
# Modifies input file in-place (keeps only untranslated keys)
python main.py diff -i data/raw/ManualTransFile.json -t data/processed/ManualTransFile_translated.json

# Or using the standalone script wrapper:
python diff_untranslated.py -i data/raw/ManualTransFile.json -t data/processed/ManualTransFile_translated.json
```

Options:
- `-o / --output`: Write untranslated keys to a new file instead of modifying input in-place.
- `--key-presence`: Filter keys solely by existence in the translated file rather than verifying a non-empty translation differing from the source key (safe mode).

#### 2. Translate New Lines
Translate the isolated new strings:

```powershell
python main.py translate -i data/raw/ManualTransFile.json -o data/processed/ManualTransFile_new_translated.json -y
```

#### 3. Merge Back into Master Translation
Merges newly translated strings back into the master `ManualTransFile_translated.json`:

```powershell
python main.py merge -b data/processed/ManualTransFile_translated.json -n data/processed/ManualTransFile_new_translated.json

# Or using the standalone script wrapper:
python diff_untranslated.py merge -b data/processed/ManualTransFile_translated.json -n data/processed/ManualTransFile_new_translated.json
```

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
