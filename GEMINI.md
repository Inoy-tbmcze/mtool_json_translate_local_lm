# AGENT DIRECTIVES: mtool_json_translate_local_lm

## Runtime & Environment
- **Interpreter**: `C:\Users\inoy\PycharmProjects\mtool_translate\.venv\Scripts\python.exe` (shorthand `$PY`)
- **Shell**: Windows PowerShell (`pwsh`)
- **LLM Endpoint**: `http://127.0.0.1:1234/v1/chat/completions` (LM Studio)
- **Config**: [config.json](file:///E:/ai/projects/mtool_json_translate_local_lm/config.json)

## Execution Matrix
| Goal | Command | Primary Output / Verification |
| :--- | :--- | :--- |
| **Verify / Benchmark** | `& $PY tests/test_pipeline_harness.py --json` | stdout JSON metrics |
| **Stage 1 (Clean)** | `& $PY main.py clean -i <raw.json>` | `data/processed/<stem>_cleaned.json` |
| **Stage 2 (Translate)** | `& $PY main.py translate -i <cleaned.json>` | `data/processed/<stem>_translated.json` |
| **Stage 3 (Validate)** | `& $PY main.py validate -i <translated.json>` | `data/processed/<stem>_validated.json` |
| **Full Pipeline** | `& $PY main.py pipeline -i <raw.json>` | `data/processed/<stem>_validated.json` |
| **Live LM Studio Test** | `& $PY tests/test_pipeline_harness.py --live` | Benchmarks live model endpoints |
| **Quality Gate (SCA)** | `pwsh -ExecutionPolicy Bypass -File .\Make.ps1 sca` | Pylint 10/10, Mypy, Ruff, Isort |

## Hard Invariants
1. **Execution Priority**: On any pipeline execution, test, or verification request, prioritize running `tests/test_pipeline_harness.py --json` to assert key conservation, latency, throughput, and error absence.
2. **Quality Standards**: All code modifications must pass `pwsh -ExecutionPolicy Bypass -File .\Make.ps1 sca` with zero warnings and Pylint 10.00/10.
3. **Serialization & Resilience**:
   - Use `fast_json_dumps_bytes` / `fast_json_loads` (`orjson` accelerated) for UTF-8 I/O.
   - Use in-house `repair_json_string` (in [src/mtool_translator/utils.py](file:///E:/ai/projects/mtool_json_translate_local_lm/src/mtool_translator/utils.py)) to parse LLM outputs.
   - Retain checkpointing logic (`checkpoint.json`, `translation_progress.json`).
4. **Shell Constraints**: Windows PowerShell only (no bash `ls`, `rm`, `cat`, `grep`). Pass multiline code or complex strings via stdin here-strings (`@' ... '@`) to avoid quoting issues.
5. **Anti-Slop / Tone**: Adhere strictly to [.agents/skills/unslop/SKILL.md](file:///E:/ai/projects/mtool_json_translate_local_lm/.agents/skills/unslop/SKILL.md).

## Editing Guidelines for Agents
When modifying this file:
- **High Signal-to-Noise**: Keep file under 100 lines. Do not add narrative explanations, design rationale, or decorative text.
- **No Embedded Copies**: Never inline file contents (e.g. `config.json`) or file trees. Reference paths directly.
- **Append via Matrix**: Add new CLI tasks or scripts directly as rows in the Execution Matrix rather than new descriptive sections.
- **Preserve Invariants**: Do not alter or remove SCA requirements, interpreter paths, or test harness rules without explicit user request.
- **Keep Mirror in Sync**: Mirror any updates to [AGENTS.md](file:///E:/ai/projects/mtool_json_translate_local_lm/AGENTS.md).
- **Block-Patching Guidelines for Agents**: When editing blocks in existing files, NEVER rewrite the full file. Use `block-patcher` ([.agents/skills/block-patcher/scripts/patch_block.exe](file:///E:/ai/projects/mtool_json_translate_local_lm/.agents/skills/block-patcher/scripts/patch_block.exe)) via one of two zero-escaping methods:
  1. *Dual-File Method (Recommended)*: Write search block to `.search.tmp` and replacement to `.replace.tmp` via `client_create_file`, then run:
     `& ".agents/skills/block-patcher/scripts/patch_block.exe" --file "path/to/target" --search-file ".search.tmp" --replace-file ".replace.tmp" --clean-tmp`
  2. *Stdin Verbatim Here-String*:
     `@'\n{"target_file": "path", "search_block": "...", "replace_block": "..."}\n'@ | & ".agents/skills/block-patcher/scripts/patch_block.exe" --stdin`
  *Avoid*: Never pass multiline Python code via `python -c "..."` on PowerShell. Prefer IDE tools (`pycharm`/`apply_patch`) or `block-patcher`.
