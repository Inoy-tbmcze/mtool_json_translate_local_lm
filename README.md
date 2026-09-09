# MTool JSON Translator

Translates Japanese game localization JSON files into English using a local LLM server (LM Studio or any OpenAI-compatible API).

## Requirements

- Python 3.10 or newer
- Local LLM server running at `http://127.0.0.1:1234`
- Suggested models:
  - `gemma-4-e4b` for text cleaning and translation audit
  - `google/gemma-4-12b-qat` for translation

## Setup

Install required dependencies:

```powershell
pip install -r requirements.txt
```

Optional editable install for the `mtool-translate` command:

```powershell
pip install -e .
```

## Quick start

1. Put your source JSON file into `data/raw/` (for example, `data/raw/ManualTransFile.json`).
2. Run the full pipeline:

```powershell
python main.py pipeline -i data/raw/ManualTransFile.json -y
```

All output files are saved to `data/processed/`.

## Pipeline stages

### 1. Clean

Strips engine identifiers, developer comments, file paths, and non-Japanese lines while preserving game terms and dialogue:

```powershell
python main.py clean -i data/raw/ManualTransFile.json
```

Outputs:
- `data/processed/<name>_cleaned.json`
- `data/processed/<name>_quarantine.json`

### 2. Translate

Generates a translation blueprint and translates text batches into English:

```powershell
python main.py translate -i data/processed/ManualTransFile_cleaned.json
```

Outputs:
- `data/processed/translated_<timestamp>.json`
- `data/processed/summary.txt`

### 3. Validate

Audits translation quality and separates passing lines from lines needing retranslation:

```powershell
python main.py validate -i data/processed/translated_<timestamp>.json
```

Outputs:
- `data/processed/<name>_validated.json` (passed)
- `data/processed/<name>_retranslate.json` (failed lines reset to Japanese for another pass)

## Configuration

Settings for endpoints, models, batch sizes, and worker counts live in `config.json`.
