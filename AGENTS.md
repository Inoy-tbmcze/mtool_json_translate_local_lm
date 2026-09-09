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
│   │   └── SKILL.md
├── .gemini/skills/               # Workspace mirror of .agents/skills/
├── config.json                   # Central configuration
├── jp_symbols.json               # Regex symbol matching definitions
├── clean_game_text.py            # Cleanup engine
├── main.py                       # Translation engine
├── validate_translation.py       # Validation engine
├── ManualTransFile.json          # Working / sample input file
└── AGENTS.md                     # Agent operational guidelines (this file)
```

### Registered Skills Index

| Skill Name | Description | Trigger |
| :--- | :--- | :--- |
| **`check-llm-service`** | Verifies LM Studio endpoint connectivity, loaded models, and chat completion responsiveness. | Before launching tasks, or when debugging connection / model errors. |
| **`clean-game-text`** | Sanitizes raw localization JSON dumps via rule-based regex and lightweight LLM classification. | When preparing raw game JSON text for translation. |
| **`translate-game-json`** | Generates Translation Blueprint (`summary.txt`) and translates batches with token-aware chunking. | When translating cleaned files or retranslating failed subsets. |
| **`validate-translation`** | Audits Japanese-to-English translation quality and isolates lines needing retranslation. | When validating completed translations. |

### Progressive Disclosure & Execution Model
1. **Discovery:** Agents read the `name` and `description` from the YAML frontmatter in `.agents/skills/<skill>/SKILL.md`.
2. **Activation:** When a task matches the skill's trigger, the agent reads the full `SKILL.md` body for step-by-step procedures.
3. **Execution:** The agent invokes the prescribed script or command using the project virtualenv.

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
- **Config Section:** `"cleanup"` in [config.json](file:///E:/ai/projects/mtool_json_translate_local_lm/config.json)
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
- **Config Section:** `"translation"` in [config.json](file:///E:/ai/projects/mtool_json_translate_local_lm/config.json)
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
- **Config Section:** `"validation"` in [config.json](file:///E:/ai/projects/mtool_json_translate_local_lm/config.json)
- **Command:**
  ```powershell
  & "C:\Users\inoy\PycharmProjects\mtool_translate\.venv\Scripts\python.exe" validate_translation.py
  ```
- **Outputs:**
  - `<stem>_validated.json`: High-confidence translations passing audit (`JP -> EN`).
  - `<stem>_retranslate.json`: Failed translations reset to original Japanese (`JP -> JP`) for second-pass translation via `main.py`.
  - `<stem>_checkpoint.json`: Resumption index.

---

## Agent Operational Rules & Guidelines
1. **Windows PowerShell Commands Only:** Always use Windows PowerShell commands. Never use Linux or Bash commands (`ls`, `rm`, `cat`, `grep`).
2. **Virtualenv Interpreter:** Always run Python scripts using `C:\Users\inoy\PycharmProjects\mtool_translate\.venv\Scripts\python.exe`.
3. **Encoding & Formatting:** All JSON inputs/outputs must use UTF-8 encoding and `ensure_ascii=False, indent=2`.
4. **JSON Output Resilience:** Always utilize `json_repair.repair_json` to safely deserialize model responses that contain markdown fences, commentary, or unescaped quotes.
5. **Autosave & Checkpoints:** Preserve checkpoint logic (`checkpoint.json`, `translation_progress.json`, autosaves) to guarantee that interruptions can resume without data loss.
6. **Skills Adherence:** Refer to the corresponding `.agents/skills/<skill>/SKILL.md` before executing or modifying pipeline steps.
