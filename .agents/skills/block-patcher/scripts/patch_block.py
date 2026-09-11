#!/usr/bin/env python3
"""
Ultra-High-Performance Targeted Block-Patching Kernel & CLI for Agent Environments.
Features dual-engine execution:
  1. Direct hardware-accelerated AVX2 C-kernel (patch_block.dll / patch_block.exe)
  2. Streamlined zero-copy memoryview fallback engine
"""

from __future__ import annotations

import ctypes
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

# File size threshold to switch from in-memory byte buffer to memory-mapped I/O in Python fallback
MMAP_THRESHOLD_BYTES = 16 * 1024 * 1024  # 16 MiB


@dataclass(slots=True, frozen=True)
class PatchResult:
    success: bool
    target_file: str
    start_line: int = 0
    end_line: int = 0
    lines_delta: int = 0
    occurrences: int = 0
    error: str = ""

    def to_dict(self) -> dict[str, object]:
        if not self.success:
            return {
                "success": False,
                "target_file": self.target_file,
                "error": self.error,
            }
        return {
            "success": True,
            "target_file": self.target_file,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "lines_delta": self.lines_delta,
            "occurrences": self.occurrences,
        }


# --- Native C Kernel Integration via ctypes ---

class _CPatchResult(ctypes.Structure):
    _fields_ = [
        ("success", ctypes.c_int32),
        ("target_file", ctypes.c_char_p),
        ("start_line", ctypes.c_int64),
        ("end_line", ctypes.c_int64),
        ("lines_delta", ctypes.c_int64),
        ("occurrences", ctypes.c_int64),
        ("error", ctypes.c_char_p),
    ]


_NATIVE_LIB = None
_NATIVE_LOADED = False


def _get_native_lib():
    global _NATIVE_LIB, _NATIVE_LOADED
    if _NATIVE_LOADED:
        return _NATIVE_LIB
    _NATIVE_LOADED = True

    script_dir = Path(__file__).resolve().parent
    dll_candidates = [
        script_dir / "patch_block.dll",
        script_dir / "libpatch_block.so",
        script_dir / "libpatch_block.dylib",
    ]

    for candidate in dll_candidates:
        if candidate.is_file():
            try:
                lib = ctypes.CDLL(str(candidate))
                lib.apply_block_patch_c.argtypes = [
                    ctypes.c_char_p,
                    ctypes.c_char_p,
                    ctypes.c_char_p,
                    ctypes.c_bool,
                ]
                lib.apply_block_patch_c.restype = _CPatchResult
                lib.free_patch_result_c.argtypes = [ctypes.POINTER(_CPatchResult)]
                lib.free_patch_result_c.restype = None
                _NATIVE_LIB = lib
                return _NATIVE_LIB
            except Exception:
                continue
    return None


def _apply_block_patch_native(
    target_file: str,
    search_block: str,
    replace_block: str,
    allow_multiple: bool,
) -> PatchResult | None:
    lib = _get_native_lib()
    if lib is None:
        return None

    try:
        c_res = lib.apply_block_patch_c(
            target_file.encode("utf-8"),
            search_block.encode("utf-8"),
            replace_block.encode("utf-8"),
            allow_multiple,
        )

        res = PatchResult(
            success=bool(c_res.success),
            target_file=c_res.target_file.decode("utf-8") if c_res.target_file else target_file,
            start_line=int(c_res.start_line),
            end_line=int(c_res.end_line),
            lines_delta=int(c_res.lines_delta),
            occurrences=int(c_res.occurrences),
            error=c_res.error.decode("utf-8") if c_res.error else "",
        )
        lib.free_patch_result_c(ctypes.byref(c_res))
        return res
    except Exception:
        return None


# --- High-Performance Python Fallback Engine ---

def _detect_dominant_newline(content: bytes | memoryview) -> bytes:
    probe_slice = content[:65536] if len(content) > 65536 else content
    crlf_count = probe_slice.count(b"\r\n")
    lf_count = probe_slice.count(b"\n") - crlf_count
    return b"\r\n" if crlf_count > lf_count else b"\n"


def _normalize_newlines(text: str, target_newline: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", target_newline)


def _find_matches_early_exit(
    haystack: bytes | memoryview,
    needle: bytes,
    allow_multiple: bool,
) -> tuple[list[int], str | None]:
    if not needle:
        return [], "Search block cannot be empty."

    first_offset = haystack.find(needle)
    if first_offset == -1:
        return [], "Search block not found in target file."

    if not allow_multiple:
        second_offset = haystack.find(needle, first_offset + 1)
        if second_offset != -1:
            total = 2
            curr = second_offset
            while True:
                curr = haystack.find(needle, curr + 1)
                if curr == -1:
                    break
                total += 1
            return (
                [],
                f"Ambiguity error: found {total} occurrences of search block. "
                "Provide more surrounding context or set allow_multiple=true.",
            )
        return [first_offset], None

    offsets = [first_offset]
    curr = first_offset
    needle_len = max(1, len(needle))
    while True:
        curr = haystack.find(needle, curr + needle_len)
        if curr == -1:
            break
        offsets.append(curr)

    return offsets, None


def _atomic_replace_win32(temp_path: Path, target_path: Path, max_retries: int = 5) -> None:
    delay = 0.005
    for attempt in range(max_retries):
        try:
            os.replace(temp_path, target_path)
            return
        except PermissionError:
            if attempt == max_retries - 1:
                raise
            time.sleep(delay)
            delay *= 2


def _apply_block_patch_python(
    target_file: str | Path,
    search_block: str,
    replace_block: str,
    allow_multiple: bool = False,
) -> PatchResult:
    path = Path(target_file).resolve()
    norm_path_str = str(path).replace("\\", "/")

    if not path.is_file():
        return PatchResult(
            success=False,
            target_file=norm_path_str,
            error=f"File not found: {path}",
        )

    file_size = path.stat().st_size
    if file_size == 0:
        return PatchResult(
            success=False,
            target_file=norm_path_str,
            error="Target file is empty.",
        )

    import mmap
    file_obj: BinaryIO | None = None
    mm: mmap.mmap | None = None
    raw_content: bytes | mmap.mmap

    try:
        if file_size >= MMAP_THRESHOLD_BYTES:
            file_obj = open(path, "rb")
            mm = mmap.mmap(file_obj.fileno(), 0, access=mmap.ACCESS_READ)
            raw_content = mm
        else:
            with open(path, "rb") as f:
                raw_content = f.read()

        dominant_nl_bytes = _detect_dominant_newline(raw_content)
        dominant_nl_str = "\r\n" if dominant_nl_bytes == b"\r\n" else "\n"

        search_normalized = _normalize_newlines(search_block, dominant_nl_str)
        needle_bytes = search_normalized.encode("utf-8")

        offsets, err = _find_matches_early_exit(raw_content, needle_bytes, allow_multiple)

        if not offsets and err and "not found" in err:
            alt_nl_str = "\n" if dominant_nl_str == "\r\n" else "\r\n"
            search_alt = _normalize_newlines(search_block, alt_nl_str)
            needle_alt_bytes = search_alt.encode("utf-8")
            offsets_alt, err_alt = _find_matches_early_exit(
                raw_content, needle_alt_bytes, allow_multiple
            )
            if offsets_alt:
                offsets = offsets_alt
                err = None
                needle_bytes = needle_alt_bytes
                dominant_nl_str = alt_nl_str

        if err or not offsets:
            return PatchResult(
                success=False,
                target_file=norm_path_str,
                error=err or "Search block not found.",
            )

        first_offset = offsets[0]
        needle_len = len(needle_bytes)

        start_line = raw_content.count(b"\n", 0, first_offset) + 1
        lines_in_search = needle_bytes.count(b"\n")
        end_line = start_line + lines_in_search

        replace_normalized = _normalize_newlines(replace_block, dominant_nl_str)
        replacement_bytes = replace_normalized.encode("utf-8")
        lines_in_replace = replacement_bytes.count(b"\n")
        lines_delta = (lines_in_replace - lines_in_search) * len(offsets)

        mv = memoryview(raw_content)
        fast_id = f"{os.getpid()}_{time.time_ns():x}"
        temp_file_path = path.with_name(f"{path.name}.tmp.{fast_id}")

        try:
            with open(temp_file_path, "wb") as out_f:
                last_idx = 0
                for offset in offsets:
                    out_f.write(mv[last_idx:offset])
                    if replacement_bytes:
                        out_f.write(replacement_bytes)
                    last_idx = offset + needle_len

                out_f.write(mv[last_idx:])
        except Exception:
            if temp_file_path.exists():
                temp_file_path.unlink(missing_ok=True)
            raise

    finally:
        if mm is not None:
            mm.close()
        if file_obj is not None:
            file_obj.close()

    try:
        _atomic_replace_win32(temp_file_path, path)
    except Exception as ex:
        if temp_file_path.exists():
            temp_file_path.unlink(missing_ok=True)
        return PatchResult(
            success=False,
            target_file=norm_path_str,
            error=f"Atomic rename failed: {ex}",
        )

    return PatchResult(
        success=True,
        target_file=norm_path_str,
        start_line=start_line,
        end_line=end_line,
        lines_delta=lines_delta,
        occurrences=len(offsets),
    )


# --- Public API ---

def apply_block_patch(
    target_file: str | Path,
    search_block: str,
    replace_block: str,
    allow_multiple: bool = False,
) -> PatchResult:
    """
    Atomically replaces an exact contiguous block of text in target_file with new content.
    Automatically leverages AVX2 C-kernel when available, falling back seamlessly to Python.
    """
    target_str = str(Path(target_file).resolve())

    # Try native C-kernel first (sub-millisecond in-process execution)
    res = _apply_block_patch_native(target_str, search_block, replace_block, allow_multiple)
    if res is not None:
        return res

    # Fallback to zero-copy memoryview Python engine
    return _apply_block_patch_python(target_str, search_block, replace_block, allow_multiple)


# --- CLI Interface ---

def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="High-performance atomic block-patching tool."
    )
    parser.add_argument("--file", "-f", help="Path to target file.")
    parser.add_argument("--search", "-s", help="Contiguous text block to find.")
    parser.add_argument("--replace", "-r", help="New replacement text block.")
    parser.add_argument(
        "--allow-multiple",
        action="store_true",
        help="Allow replacing all occurrences (default: false).",
    )
    parser.add_argument(
        "--json",
        "-j",
        help="JSON payload containing target_file, search_block, replace_block, [allow_multiple].",
    )
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read JSON payload from standard input (recommended for multiline content).",
    )
    parser.add_argument(
        "--payload-file",
        help="Path to a JSON file containing the parameters.",
    )

    args = parser.parse_args()

    payload: dict[str, object] = {}

    if args.stdin:
        try:
            stdin_content = sys.stdin.read()
            if not stdin_content.strip():
                print(
                    json.dumps({"success": False, "error": "Empty stdin payload."}),
                    file=sys.stderr,
                )
                return 1
            payload = json.loads(stdin_content)
        except Exception as e:
            print(
                json.dumps({"success": False, "error": f"Invalid JSON on stdin: {e}"}),
                file=sys.stderr,
            )
            return 1
    elif args.payload_file:
        try:
            payload = json.loads(Path(args.payload_file).read_text(encoding="utf-8"))
        except Exception as e:
            print(
                json.dumps(
                    {"success": False, "error": f"Failed reading payload file: {e}"}
                ),
                file=sys.stderr,
            )
            return 1
    elif args.json:
        try:
            payload = json.loads(args.json)
        except Exception as e:
            print(
                json.dumps({"success": False, "error": f"Invalid JSON argument: {e}"}),
                file=sys.stderr,
            )
            return 1
    elif args.file:
        payload = {
            "target_file": args.file,
            "search_block": args.search if args.search is not None else "",
            "replace_block": args.replace if args.replace is not None else "",
            "allow_multiple": args.allow_multiple,
        }
    else:
        parser.print_help(file=sys.stderr)
        return 1

    target_file = str(payload.get("target_file", ""))
    search_block = str(payload.get("search_block", ""))
    replace_block = str(payload.get("replace_block", ""))
    allow_multiple = bool(payload.get("allow_multiple", False))

    if not target_file:
        print(
            json.dumps({"success": False, "error": "Missing target_file parameter."}),
            file=sys.stderr,
        )
        return 1

    res = apply_block_patch(
        target_file=target_file,
        search_block=search_block,
        replace_block=replace_block,
        allow_multiple=allow_multiple,
    )

    out_json = json.dumps(res.to_dict(), indent=2, ensure_ascii=False)
    if res.success:
        print(out_json)
        return 0
    else:
        print(out_json, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
