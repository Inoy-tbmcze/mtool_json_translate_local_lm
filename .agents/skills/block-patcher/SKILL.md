---
name: block-patcher
description: Atomically replaces an exact contiguous block of code or text in a file with new content. Use this instead of rewriting entire files or running complex shell sed/awk commands.
---

# Block Patcher

High-performance atomic block-patching tool for agent environments. Performs exact contiguous string/code block replacement with byte-level fidelity, automatic CRLF/LF normalization, and atomic swap guarantees.

## When to Use

- Use when modifying 5–50 lines in a medium or large file (100–10,000+ lines).
- Eliminates context-window waste and latency from full-file rewrites (`client_edit_file`).
- Completely immune to shell-escaping, PowerShell `$variable` expansions, and quote stripping.

## Usage

### Method 1: Via Python `stdin` with JSON (Recommended for Agents)

Passing JSON via stdin avoids PowerShell quoting and escaping bugs with complex code blocks:

```powershell
python -c "
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path('.agents/skills/block-patcher/scripts').resolve()))
from patch_block import apply_block_patch

res = apply_block_patch(
    target_file=r'path/to/target.py',
    search_block='''def old_function():
    return False
''',
    replace_block='''def old_function():
    return True
''',
    allow_multiple=False
)
print(json.dumps(res.to_dict(), indent=2))
"
```

Or piping JSON directly to the CLI:
```bash
python .agents/skills/block-patcher/scripts/patch_block.py --stdin << 'EOF'
{
  "target_file": "path/to/file.py",
  "search_block": "def old():\n    pass\n",
  "replace_block": "def new():\n    return 42\n",
  "allow_multiple": false
}
EOF
```

### Method 2: CLI Arguments (Quick Single-Line Edits)

```powershell
python .agents/skills/block-patcher/scripts/patch_block.py --file "path/to/file.py" --search "DEBUG = False" --replace "DEBUG = True"
```

## Return Payload

On success (exit code 0):
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

On ambiguity / error (exit code 2):
```json
{
  "success": false,
  "target_file": "E:/path/to/file.py",
  "error": "Ambiguity error: found 2 occurrences of search block. Provide more surrounding context or set allow_multiple=true."
}
```
