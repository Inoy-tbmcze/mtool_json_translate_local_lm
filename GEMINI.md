# Project: Game Localization JSON Translator (JP -> EN)

## Purpose & Architecture
Translate Japanese game localization text (MTool, RPG Maker, Unity JSON dumps) into English using local LLMs (LM Studio / OpenAI-compatible API). The system preserves JSON dictionary structures, game context, character voices, and UI markers across three modular stages: **Cleanup**, **Translation**, and **Validation**.

Performance-critical paths incorporate low-level acceleration:
- JIT-allocated x86-64 machine code kernels via Win32 `VirtualAlloc` (`is_ascii_ident`, `has_repeated_bytes`, `count_symbols`, `find_json_bounds`).
- Unicode BMP 8,192-byte bitmask lookup table for character set checks.
- Fast-rejection zero-copy string algorithms for text cleanup.
- SIMD JSON serialization via `orjson`.
- Resilient in-house JSON repair engine (`repair_json_string`) replacing third-party `json_repair`.
- In-house persistent HTTP/1.1 client (`FastLocalHttpClient`) utilizing standard library `http.client`, Windows `SIO_LOOPBACK_FAST_PATH`, `TCP_NODELAY`, and LIFO connection pooling, completely eliminating `requests` and its transitive dependencies.

---

## Project Structure

The project follows a standard `src/` layout with separate input/output data staging and backward-compatible root CLI wrappers:

```
mtool_json_translate_local_lm/
├── .agents/skills/               # Agent skills
│   └── unslop/
│       └── SKILL.md              # Writing & anti-slop rules
├── .gemini/
│   ├── settings.json             # Environment config
│   └── skills/                   # Workspace mirror of .agents/skills/
├── data/
│   ├── raw/                      # Original game text dumps (e.g. ManualTransFile.json)
│   ├── processed/                # Intermediate, cleaned, translated, and validated outputs
│   └── reference/                # Reference dictionaries and symbol definitions (common_translations.json, jp_symbols.json)
├── src/
│   └── mtool_translator/        # Core package
│       ├── __init__.py
│       ├── cli.py                # Unified CLI entrypoint
│       ├── config.py             # Configuration & path resolution
│       ├── cleaner.py            # Stage 1: Heuristic & LLM text cleaner
│       ├── http_client.py        # Fast in-house persistent HTTP/1.1 client
│       ├── native_core.py        # JIT x86-64 machine code & bitmask acceleration
│       ├── translator.py         # Stage 2: Token-aware chunker & translation engine
│       ├── validator.py          # Stage 3: Translation validation auditor
│       └── utils.py              # Shared Japanese regex, in-house JSON repair & parsing helpers
├── clean_game_text.py            # Backward-compatible Stage 1 script wrapper
├── main.py                       # Unified CLI and Stage 2 script wrapper
├── validate_translation.py       # Backward-compatible Stage 3 script wrapper
├── Make.ps1                      # Development task runner (sca, format, fix)
├── config.json                   # Central configuration
├── requirements.txt              # Project dependencies
├── pyproject.toml                # Package metadata, tool configurations & entry points
├── .gitignore                    # Git exclusions
├── AGENTS.md                     # Agent operational guidelines
└── GEMINI.md                     # Gemini CLI instructions mirror (this file)
```

---

## Registered Skills Index

| Skill Name | Scope | Description | Trigger |
| :--- | :--- | :--- | :--- |
| **`unslop`** | **Always Active** | Strips AI clichés, sycophancy, chatbot filler, superficial -ing clauses, and mannered prose. | In effect for every response, edit, explanation, and translation. |
| **`check-llm-service`** | On Demand | Verifies LM Studio endpoint connectivity, loaded models, and chat completion responsiveness. | Before launching tasks, or when debugging connection / model errors. |
| **`clean-game-text`** | On Demand | Sanitizes raw localization JSON dumps via rule-based regex and lightweight LLM classification. | When preparing raw game JSON text for translation. |
| **`translate-game-json`** | On Demand | Generates Translation Blueprint (`summary.txt`) and translates batches with token-aware chunking. | When translating cleaned files or retranslating failed subsets. |
| **`validate-translation`** | On Demand | Audits Japanese-to-English translation quality and isolates lines needing retranslation. | When validating completed translations. |

### Progressive Disclosure & Execution Model
1. **Always-On Skills:** Skills flagged with `alwaysApply` (such as `unslop`) govern all generated text, code, and communication unconditionally.
2. **Discovery:** Agents read the `name` and `description` from the YAML frontmatter in `.agents/skills/<skill>/SKILL.md`.
3. **Activation:** When a task matches the skill's trigger, the agent reads the full `SKILL.md` body for step-by-step procedures.
4. **Execution:** The agent invokes the prescribed script or command using the project virtualenv.

---

## Environment & Python Setup
- **Operating System:** Windows
- **Command Shell:** PowerShell (`pwsh`)
- **Configured Python Virtualenv:**
  ```powershell
  # Full path to project Python interpreter:
  C:\Users\inoy\PycharmProjects\mtool_translate\.venv\Scripts\python.exe

  # Or activate venv in PowerShell:
  C:\Users\inoy\PycharmProjects\mtool_translate\.venv\Scripts\Activate.ps1
  ```
  *Important:* Always invoke Python using the virtual environment interpreter above. System Python lacks required dependencies (`orjson`).

- **Local LLM Server (LM Studio):**
  - Default endpoint: `http://127.0.0.1:1234/v1/chat/completions`
  - Ensure LM Studio has the required model loaded (`gemma-4-e4b` for cleanup/validation, `google/gemma-4-12b-qat` for translation).

---

## Execution Pipeline

### Unified CLI
Run any stage or the full pipeline via `main.py`:
```powershell
# Stage 1: Clean
& "C:\Users\inoy\PycharmProjects\mtool_translate\.venv\Scripts\python.exe" main.py clean -i data/raw/ManualTransFile.json

# Stage 2: Translate
& "C:\Users\inoy\PycharmProjects\mtool_translate\.venv\Scripts\python.exe" main.py translate -i data/processed/ManualTransFile_cleaned.json

# Stage 3: Validate
& "C:\Users\inoy\PycharmProjects\mtool_translate\.venv\Scripts\python.exe" main.py validate -i data/processed/ManualTransFile_translated.json

# Full Pipeline: Clean -> Translate -> Validate
& "C:\Users\inoy\PycharmProjects\mtool_translate\.venv\Scripts\python.exe" main.py pipeline -i data/raw/ManualTransFile.json
```

### Backward-Compatible Individual Commands

#### Stage 1: Preprocess & Clean Text
- **Skill:** `clean-game-text` ([SKILL.md](file:///E:/ai/projects/mtool_json_translate_local_lm/.agents/skills/clean-game-text/SKILL.md))
- **Script:** [clean_game_text.py](file:///E:/ai/projects/mtool_json_translate_local_lm/clean_game_text.py)
- **Config Section:** `"cleanup"` in [config.json](file:///E:/ai/projects/mtool_json_translate_local_lm/config.json)
- **Command:**
  ```powershell
  & "C:\Users\inoy\PycharmProjects\mtool_translate\.venv\Scripts\python.exe" clean_game_text.py
  ```
- **Outputs:**
  - `data/processed/<stem>_cleaned.json`: Filtered dialogue, item, UI, and story lines ready for translation.
  - `data/processed/<stem>_quarantine.json`: Quarantined engine variables, asset paths, and developer comments.

#### Stage 2: Translation Engine
- **Skill:** `translate-game-json` ([SKILL.md](file:///E:/ai/projects/mtool_json_translate_local_lm/.agents/skills/translate-game-json/SKILL.md))
- **Script:** [main.py](file:///E:/ai/projects/mtool_json_translate_local_lm/main.py)
- **Config Section:** `"translation"` in [config.json](file:///E:/ai/projects/mtool_json_translate_local_lm/config.json)
- **Command:**
  ```powershell
  & "C:\Users\inoy\PycharmProjects\mtool_translate\.venv\Scripts\python.exe" main.py
  ```
- **Outputs:**
  - `data/processed/ManualTransFile_translated.json`: Final translated JSON file.
  - `data/processed/summary.txt`: Hierarchical Translation Blueprint.

#### Stage 3: Translation Validation & Retranslate Loop
- **Skill:** `validate-translation` ([SKILL.md](file:///E:/ai/projects/mtool_json_translate_local_lm/.agents/skills/validate-translation/SKILL.md))
- **Script:** [validate_translation.py](file:///E:/ai/projects/mtool_json_translate_local_lm/validate_translation.py)
- **Config Section:** `"validation"` in [config.json](file:///E:/ai/projects/mtool_json_translate_local_lm/config.json)
- **Command:**
  ```powershell
  & "C:\Users\inoy\PycharmProjects\mtool_translate\.venv\Scripts\python.exe" validate_translation.py
  ```
- **Outputs:**
  - `data/processed/<stem>_validated.json`: High-confidence translations passing audit (`JP -> EN`).
  - `data/processed/<stem>_retranslate.json`: Failed translations reset to original Japanese (`JP -> JP`) for second-pass translation via `main.py`.
  - `data/processed/<stem>_checkpoint.json`: Resumption index.

---

## Static Code Analysis & Quality Standards

Every modification must pass all static analysis checks cleanly:
```powershell
pwsh -ExecutionPolicy Bypass -File .\Make.ps1 sca
```
Tool requirements:
- **Ruff:** Clean pass (zero errors or warnings).
- **Mypy:** Clean pass across all source files with strict type hints.
- **Pylint:** Clean **10.00/10** rating across all modules.
- **Isort:** Clean import order matching Black profile.

---

## Configuration Reference (`config.json`)

```json
{
  "api_endpoint": "http://127.0.0.1:1234/v1/chat/completions",
  "api_key": "lm-studio",
  "cleanup": {
    "api_endpoint": "http://127.0.0.1:1234/v1/chat/completions",
    "api_key": "lm-studio",
    "model": "gemma-4-e4b",
    "input_filename": "ManualTransFile.json",
    "symbols_filename": "jp_symbols.json",
    "min_japanese_ratio": 0.8,
    "batch_size": 30,
    "max_workers": 4,
    "save_interval": 30,
    "request_timeout": 60
  },
  "translation": {
    "api_endpoint": "http://127.0.0.1:1234/v1/chat/completions",
    "api_key": "",
    "model": "google/gemma-4-12b-qat",
    "source_language": "Japanese",
    "target_language": "English",
    "input_filename": "ManualTransFile_cleaned.json",
    "batch_size": 40,
    "max_retries": 10,
    "retry_delay": 0.1,
    "request_timeout": 1200,
    "save_interval": 1,
    "max_workers": 4,
    "api_type": "openai"
  },
  "validation": {
    "api_endpoint": "http://127.0.0.1:1234/v1/chat/completions",
    "api_key": "lm-studio",
    "model": "gemma-4-e4b",
    "input_filename": "ManualTransFile_translated.json",
    "batch_size": 20,
    "max_workers": 4,
    "save_interval": 10,
    "request_timeout": 60
  }
}
```

---

## Agent Operational Rules & Guidelines
1. **Always-On Unslop Skill:** Follow [.agents/skills/unslop/SKILL.md](file:///E:/ai/projects/mtool_json_translate_local_lm/.agents/skills/unslop/SKILL.md) unconditionally: eliminate filler, sycophancy, AI buzzwords (delve, tapestry, pivotal, crucial, enhance), superficial -ing clauses, em dashes, and decorative emojis. Speak and write plainly and directly.
2. **Windows PowerShell Commands Only:** Always use Windows PowerShell commands. Never use Linux or Bash commands (`ls`, `rm`, `cat`, `grep`).
3. **Virtualenv Interpreter:** Always run Python scripts using `C:\Users\inoy\PycharmProjects\mtool_translate\.venv\Scripts\python.exe`.
4. **Encoding & Serialization:** All JSON inputs/outputs must use UTF-8 encoding via `fast_json_dumps_bytes` or `fast_json_loads` (`orjson` accelerated).
5. **JSON Output Resilience:** Always utilize in-house `repair_json_string` to safely deserialize model responses that contain markdown fences, commentary, truncated tokens, or unescaped quotes.
6. **Autosave & Checkpoints:** Preserve checkpoint logic (`checkpoint.json`, `translation_progress.json`, autosaves) to guarantee that interruptions can resume without data loss.
7. **Skills Adherence:** Refer to the corresponding `.agents/skills/<skill>/SKILL.md` before executing or modifying pipeline steps.
8. **Static Code Analysis:** Always verify changes with `.\Make.ps1 sca` and ensure Pylint stays at 10.00/10.
