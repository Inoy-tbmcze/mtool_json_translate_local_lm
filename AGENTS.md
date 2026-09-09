# Project: Game Localization JSON Translator (JP -> EN)

## Purpose
Translate Japanese game text into English using local LLMs. Preserve JSON structures, context, and UI elements.

## Environment
- **Operating System:** Windows
- **Command Shell:** PowerShell / CMD

## Execution Commands
- **Run Workflow:** `python src/main.py`
- **Preprocess Text:** `python src/clean_game_text.py --input <path>`
- **Validate Output:** `python src/validate_translation.py --file <path>`

## Agent Rules & Constraints
1. **Windows Commands Only:** Use Windows PowerShell or CMD commands. Do not use Linux or Bash commands (for example: `ls`, `rm`, `cat`, `grep`).
2. **Keep Files Small:** Keep individual source code files under 200 lines of code. Split large classes or functions into separate files in `src/`.

## Recommended File Structure
game_translator/
├── config/
├── src/
├── data/
└── AGENTS.md

## Key Components
- `src/main.py`: Entry point. Manages the end-to-end translation pipeline.
- `src/clean_game_text.py`: Preprocesses text and masks UI tags.
- `src/validate_translation.py`: Checks translation quality.