# Project Context: Game Localization JSON Translator

## Purpose
Automated game localization (JP -> EN) using local LLMs, focusing on high-fidelity translation of JSON-based game text while preserving structure and context.

## Key Components
- `main.py`: Orchestrates the end-to-end workflow.
- `translate_config.json`: Global configuration (models, paths, parameters).
- `clean_game_text.py`: Preprocessing to filter junk/non-translatable text and protect UI elements.
- `validate_translation.py`: LLM-based automated quality audit.
- `TokenAwareChunker`: Splits text into chunks fitting local LLM context windows.
- `Translation Blueprint`: Provides world-building, tone, and character context for consistency.

## Workflow
Input -> Preprocessing (Cleaning + Blueprint) -> Chunking -> Translation (Retries/Fallbacks) -> Persistence -> Output.

## Technical Details
- **Error Handling**: Manages 429 (Rate Limit) errors.
- **Persistence**: Saves progress to `.progress.json` for resumability.
- **Integrity**: Preserves original JSON keys and structure.