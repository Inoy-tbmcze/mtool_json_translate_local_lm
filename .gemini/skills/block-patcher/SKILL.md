---
name: block-patcher
description: Atomically replaces an exact contiguous block of code or text in a file with new content. Use this instead of rewriting entire files or running complex shell sed/awk commands.
---

# Block Patcher

Ultra-high-performance atomic block-patching tool for agent environments. Features dual-engine execution (AVX2-vectorized native C-microkernel + zero-copy memoryview Python fallback) with byte-level fidelity, automatic CRLF/LF normalization, and Win32 atomic swap guarantees.

## When to Use

- Modifying 5–50 lines in a medium or large file (100–10,000+ lines).
- Eliminating context-window waste and latency from full-file rewrites (`client_edit_file`).
- Completely immune to shell-escaping, PowerShell `$variable` expansions, and quote stripping.

---

## AI Agent Quick Start (Windows PowerShell)

AI agents frequently encounter quoting, backslash, and parenthesis issues when running shell commands on Windows. Use **Method 1** or **Method 2** to guarantee 100% reliable execution.

### Method 1: Dual-File Pattern (Recommended for Agents - Zero Shell Escaping)

This is the most reliable method for AI agents. Write the search and replace blocks to temporary files using `client_create_file` (which bypasses the shell entirely), then let `patch_block.exe` replace the block and automatically delete the temp files.

1. Use `client_create_file` to write the exact search block to `.patch_search.tmp`.
2. Use `client_create_file` to write the replacement block to `.patch_replace.tmp`.
3. Run the atomic patch with `--clean-tmp`:

```powershell
& "<path_to_skill>\scripts\patch_block.exe" --file "path/to/target.py" --search-file ".patch_search.tmp" --replace-file ".patch_replace.tmp" --clean-tmp
```

*(Note: `--clean-tmp` automatically deletes `.patch_search.tmp` and `.patch_replace.tmp` on success).*

---

### Method 2: Verbatim Here-String via stdin (Single Shell Command)

PowerShell verbatim here-strings (`@' ... '@`) completely disable variable interpolation (`$`), quote stripping, parentheses evaluation, and escape sequences.

```powershell
@'
{
  "target_file": "path/to/target.py",
  "search_block": "def old_func():\n    return False\n",
  "replace_block": "def old_func():\n    return True\n",
  "allow_multiple": false
}
'@ | & "<path_to_skill>\scripts\patch_block.exe" --stdin
```

---

### Method 3: PowerShell Helper Script (`patch-block.ps1`)

If you prefer a direct PowerShell command:

```powershell
& "<path_to_skill>\scripts\patch-block.ps1" -File "path/to/target.py" -SearchFile ".patch_search.tmp" -ReplaceFile ".patch_replace.tmp" -CleanTmp
```

---

### Method 4: Payload File (`--payload-file`)

Write the full JSON configuration to a temporary `.patch.json` file:

```powershell
& "<path_to_skill>\scripts\patch_block.exe" --payload-file ".patch.json" --clean-tmp
```

---

### Method 5: In-Process Python API (Direct AVX2 C-Kernel via `ctypes`)

Inside a Python script or module:

```python
import sys
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, r"<path_to_skill>\scripts")
from patch_block import apply_block_patch, apply_block_patch_files

# Direct string blocks
res = apply_block_patch(
    target_file=r"path/to/target.py",
    search_block="""def old_func():\n    return False\n""",
    replace_block="""def old_func():\n    return True\n""",
    allow_multiple=False,
)
print(res.to_dict())

# Or via files:
res = apply_block_patch_files(
    target_file=r"path/to/target.py",
    search_file=r".patch_search.tmp",
    replace_file=r".patch_replace.tmp",
    clean_tmp=True,
)
```

---

## Anti-Patterns to Avoid

- **DO NOT USE `python -c "..."` with embedded Python code in PowerShell**:
  PowerShell will interpret parentheses (e.g. `(default)` in docstrings) as cmdlet invocations and strip quotes, causing fatal syntax errors.
- **DO NOT USE Bash heredocs (`<< 'EOF'`) on Windows**:
  PowerShell does not support `<<`. Always pipe from `@' ... '@` instead.

---

## Return Payload

### On success (exit code 0):
```json
{
  "success": true,
  "target_file": "E:/path/to/file.py",
  "start_line": 42,
  "end_line": 50,
  "lines_delta": 3,
  "occurrences": 1
}
```

### On ambiguity / not found (exit code 2):
```json
{
  "success": false,
  "target_file": "E:/path/to/file.py",
  "error": "Ambiguity error: found 2 occurrences of search block. Provide more surrounding context or set allow_multiple=true."
}
```
