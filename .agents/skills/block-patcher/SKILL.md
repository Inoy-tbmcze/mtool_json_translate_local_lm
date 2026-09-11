---
name: block-patcher
description: Atomically replaces an exact contiguous block of code or text in a file with new content. Use this instead of rewriting entire files or running complex shell sed/awk commands.
---

# Block Patcher

Ultra-high-performance atomic block-patching tool for agent environments. Features dual-engine execution (AVX2-vectorized native C-microkernel + zero-copy memoryview Python fallback) with byte-level fidelity, automatic CRLF/LF normalization, and Win32 atomic swap guarantees.

## When to Use

- Modifying 5–50 lines in a medium or large file (100–10,000+ lines).
- Eliminates context-window waste and latency from full-file rewrites (`client_edit_file`).
- Completely immune to shell-escaping, PowerShell `$variable` expansions, and quote stripping.

---

## Invocation Methods

### Method 1: Sub-Millisecond Native CLI with `stdin` JSON (Recommended)

Pipe JSON directly to the standalone `patch_block.exe` (or `patch_block.py`):

```bash
<path_to_skill>/scripts/patch_block.exe --stdin << 'EOF'
{
  "target_file": "path/to/target.py",
  "search_block": "def old_func():\n    return False\n",
  "replace_block": "def old_func():\n    return True\n",
  "allow_multiple": false
}
EOF
```

### Method 2: In-Process Python API (Direct Native C-Kernel via `ctypes`)

```powershell
python -c "
import sys, json
from pathlib import Path
skill_dir = Path('<path_to_skill>/scripts').resolve()
sys.path.insert(0, str(skill_dir))
from patch_block import apply_block_patch

res = apply_block_patch(
    target_file=r'path/to/target.py',
    search_block='''def old_func():
    return False
''',
    replace_block='''def old_func():
    return True
''',
    allow_multiple=False
)
print(json.dumps(res.to_dict(), indent=2))
"
```

### Method 3: CLI Arguments (Quick Edits)

```powershell
<path_to_skill>/scripts/patch_block.exe --file "path/to/file.py" --search "DEBUG = False" --replace "DEBUG = True"
```

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
