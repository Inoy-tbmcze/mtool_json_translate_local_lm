# Project: Game Localization JSON Translator (JP -> EN)

## Purpose & Architecture
Translate Japanese game localization text (MTool, RPG Maker, Unity JSON dumps) into English using local LLMs (LM Studio / OpenAI-compatible API). The system preserves JSON dictionary structures, game context, character voices, and UI markers across three modular stages: **Cleanup**, **Translation**, and **Validation**.

---

## Recommended Skills Architecture

This repository adopts the open **Agent Skills specification** for procedural execution, progressive disclosure, and lightweight agent context.

### Skills Directory Structure
Agent skills reside under `.agents/skills/` (mirrored in `.gemini/skills/`):

```
mtool_json_translate_local_lm/
├── .agents/skills/
│   ├── unslop/
│   │   └── SKILL.md              # Always-applied writing & anti-slop rules
├── .gemini/
│   ├── settings.json             # Environment config (fileName, alwaysApply skills, systemInstructions)
│   └── skills/                   # Workspace mirror of .agents/skills/
├── config.json                   # Central configuration
├── jp_symbols.json               # Regex symbol matching definitions
├── clean_game_text.py            # Cleanup engine
├── main.py                       # Translation engine
├── validate_translation.py       # Validation engine
├── ManualTransFile.json          # Working / sample input file
└── AGENTS.md                     # Agent operational guidelines (this file)
```

### Registered Skills Index

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

### Creating New Skills
When adding new skills to `.agents/skills/<skill-name>/`:
1. Use lowercase alphanumeric characters and hyphens only (`[a-z0-9-]+`) for the skill folder and `name`.
2. Include a required `SKILL.md` with YAML frontmatter delimited by `---`:
   ```markdown
   ---
   name: your-skill-name
   description: Specific explanation of what the skill does. Use when [trigger condition].
   compatibility: Windows PowerShell, Python >= 3.10
   metadata:
     author: inoy
     version: "1.0.0"
   ---

   # Your Skill Title
   ## Workflow
   ...
   ```
3. Store optional helper code in `scripts/`, supplementary guidelines in `references/`, and templates in `assets/`.
4. Mirror changes to `.gemini/skills/` to preserve workspace interoperability.

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
  *Important:* Always invoke Python using the virtual environment interpreter above. System Python lacks required dependencies (`requests`, `json_repair`).

- **Local LLM Server (LM Studio):**
  - Default endpoint: `http://127.0.0.1:1234/v1/chat/completions`
  - Ensure LM Studio has the required model loaded (`gemma-4-e4b` for cleanup/validation, `google/gemma-4-12b-qat` for translation).

---

## Execution Pipeline

### Stage 1: Preprocess & Clean Text
- **Skill:** `clean-game-text` ([SKILL.md](file:///E:/ai/projects/mtool_json_translate_local_lm/.agents/skills/clean-game-text/SKILL.md))
- **Script:** [clean_game_text.py](file:///E:/ai/projects/mtool_json_translate_local_lm/clean_game_text.py)
- **Config Section:** `\"cleanup\"` in [config.json](file:///E:/ai/projects/mtool_json_translate_local_lm/config.json)
- **Command:**
  ```powershell
  & "C:\Users\inoy\PycharmProjects\mtool_translate\.venv\Scripts\python.exe" clean_game_text.py
  ```
- **Outputs:**
  - `<stem>_cleaned.json`: Filtered dialogue, item, UI, and story lines ready for translation.
  - `<stem>_quarantine.json`: Quarantined engine variables, asset paths, and developer comments.

### Stage 2: Translation Engine
- **Skill:** `translate-game-json` ([SKILL.md](file:///E:/ai/projects/mtool_json_translate_local_lm/.agents/skills/translate-game-json/SKILL.md))
- **Script:** [main.py](file:///E:/ai/projects/mtool_json_translate_local_lm/main.py)
- **Config Section:** `\"translation\"` in [config.json](file:///E:/ai/projects/mtool_json_translate_local_lm/config.json)
- **Command:**
  ```powershell
  & "C:\Users\inoy\PycharmProjects\mtool_translate\.venv\Scripts\python.exe" main.py
  ```
- **Interactive Prompts Notice:**
  - When `summary.txt` does not exist, `main.py` generates the Translation Blueprint and pauses with `input()`. Review `summary.txt` and press Enter to proceed.
  - When `translation_progress.json` exists, answer `y` to resume or `n` to start over.
- **Outputs:**
  - `translated_<YYYYMMDD_HHMMSS>.json`: Final translated JSON file.
  - `summary.txt`: Hierarchical Translation Blueprint.
  - `translation.log`: Runtime execution log.

### Stage 3: Translation Validation & Retranslate Loop
- **Skill:** `validate-translation` ([SKILL.md](file:///E:/ai/projects/mtool_json_translate_local_lm/.agents/skills/validate-translation/SKILL.md))
- **Script:** [validate_translation.py](file:///E:/ai/projects/mtool_json_translate_local_lm/validate_translation.py)
- **Config Section:** `\"validation\"` in [config.json](file:///E:/ai/projects/mtool_json_translate_local_lm/config.json)
- **Command:**
  ```powershell
  & "C:\Users\inoy\PycharmProjects\mtool_translate\.venv\Scripts\python.exe" validate_translation.py
  ```
- **Outputs:**
  - `<stem>_validated.json`: High-confidence translations passing audit (`JP -> EN`).
  - `<stem>_retranslate.json`: Failed translations reset to original Japanese (`JP -> JP`) for second-pass translation via `main.py`.
  - `<stem>_checkpoint.json`: Resumption index.

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
    "api_type": "lmstudio"
  },
  "validation": {
    "api_endpoint": "http://127.0.0.1:1234/v1/chat/completions",
    "api_key": "lm-studio",
    "model": "gemma-4-e4b",
    "input_filename": "translated_output.json",
    "batch_size": 1,
    "max_workers": 20,
    "save_interval": 120,
    "request_timeout": 60
  }
}
```

---

## Agent Operational Rules & Guidelines
1. **Always-On Unslop Skill:** Follow [.agents/skills/unslop/SKILL.md](file:///E:/ai/projects/mtool_json_translate_local_lm/.agents/skills/unslop/SKILL.md) unconditionally: eliminate filler, sycophancy, AI buzzwords (delve, tapestry, pivotal, crucial, enhance), superficial -ing clauses, em dashes, and decorative emojis. Speak and write plainly and directly.
2. **Windows PowerShell Commands Only:** Always use Windows PowerShell commands. Never use Linux or Bash commands (`ls`, `rm`, `cat`, `grep`).
3. **Virtualenv Interpreter:** Always run Python scripts using `C:\\Users\\inoy\\PycharmProjects\\mtool_translate\\.venv\\Scripts\\python.exe`.
4. **Encoding & Formatting:** All JSON inputs/outputs must use UTF-8 encoding and `ensure_ascii=False, indent=2`.
5. **JSON Output Resilience:** Always utilize `json_repair.repair_json` to safely deserialize model responses that contain markdown fences, commentary, or unescaped quotes.
6. **Autosave & Checkpoints:** Preserve checkpoint logic (`checkpoint.json`, `translation_progress.json`, autosaves) to guarantee that interruptions can resume without data loss.
7. **Skills Adherence:** Refer to the corresponding `.agents/skills/<skill>/SKILL.md` before executing or modifying pipeline steps.
