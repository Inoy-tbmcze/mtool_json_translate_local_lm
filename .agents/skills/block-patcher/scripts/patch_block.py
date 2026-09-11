#!/usr/bin/env python3
"""
High-Performance Targeted Block-Patching Kernel & CLI for Agent Environments.
Designed for maximum execution speed, minimal memory overhead, byte-fidelity,
and Win32 atomic file operations.
"""

from __future__ import annotations

import argparse
import json
import mmap
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

# File size threshold to switch from in-memory byte buffer to memory-mapped I/O
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


def _detect_encoding_and_bom(raw_prefix: bytes) -> tuple[str, bytes]:
    """Detects UTF-8 BOM or defaults to strict UTF-8 with zero decoding overhead."""
    if raw_prefix.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig", b"\xef\xbb\xbf"
    return "utf-8", b""


def _detect_dominant_newline(content: bytes | mmap.mmap) -> bytes:
    """
    Scans the buffer to determine dominant newline convention.
    Uses SIMD-accelerated count operations.
    """
    probe_slice = content[:65536] if len(content) > 65536 else content
    crlf_count = probe_slice.count(b"\r\n")
    lf_count = probe_slice.count(b"\n") - crlf_count
    return b"\r\n" if crlf_count > lf_count else b"\n"


def _normalize_newlines(text: str, target_newline: str) -> str:
    """Normalizes all newline variations in a string to the target newline."""
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", target_newline)


def _find_matches_early_exit(
    haystack: bytes | mmap.mmap,
    needle: bytes,
    allow_multiple: bool,
) -> tuple[list[int], str | None]:
    """
    Performs fast search with early exit on ambiguity.
    Returns (match_offsets, error_message).
    """
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
    """
    Atomically replaces target_path with temp_path, with exponential backoff
    to handle transient Windows file locks (antivirus, search indexing).
    """
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


def apply_block_patch(
    target_file: str | Path,
    search_block: str,
    replace_block: str,
    allow_multiple: bool = False,
) -> PatchResult:
    """
    Atomically replaces an exact contiguous block of text in target_file with new content.
    Guarantees zero mutation to untouched bytes, preservation of native newlines,
    and single-pass sub-millisecond execution for typical files.
    """
    path = Path(target_file).resolve()
    if not path.is_file():
        return PatchResult(
            success=False,
            target_file=str(path),
            error=f"File not found: {path}",
        )

    file_size = path.stat().st_size
    if file_size == 0:
        return PatchResult(
            success=False,
            target_file=str(path),
            error="Target file is empty.",
        )

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

        # Step 1: Detect Encoding & Dominant Line Ending
        _encoding, _bom = _detect_encoding_and_bom(raw_content[:4])
        dominant_nl_bytes = _detect_dominant_newline(raw_content)
        dominant_nl_str = "\r\n" if dominant_nl_bytes == b"\r\n" else "\n"

        # Step 2: Prepare Candidate Needles
        search_normalized = _normalize_newlines(search_block, dominant_nl_str)
        needle_bytes = search_normalized.encode("utf-8")

        offsets, err = _find_matches_early_exit(raw_content, needle_bytes, allow_multiple)

        # Fallback: Alternate line endings if not found
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
                target_file=str(path),
                error=err or "Search block not found.",
            )

        # Step 3: Compute Line Metrics using SIMD memchr (buffer.count(b'\n'))
        first_offset = offsets[0]
        needle_len = len(needle_bytes)

        start_line = raw_content.count(b"\n", 0, first_offset) + 1
        lines_in_search = needle_bytes.count(b"\n")
        end_line = start_line + lines_in_search

        # Step 4: Prepare Replacement Block
        replace_normalized = _normalize_newlines(replace_block, dominant_nl_str)
        replacement_bytes = replace_normalized.encode("utf-8")
        lines_in_replace = replacement_bytes.count(b"\n")
        lines_delta = (lines_in_replace - lines_in_search) * len(offsets)

        # Step 5: Construct Patched File with Zero Untouched Byte Mutation
        temp_file_path = path.with_name(f"{path.name}.tmp.{os.urandom(8).hex()}")
        try:
            with open(temp_file_path, "wb") as out_f:
                last_idx = 0
                for offset in offsets:
                    out_f.write(raw_content[last_idx:offset])
                    out_f.write(replacement_bytes)
                    last_idx = offset + needle_len

                out_f.write(raw_content[last_idx:])
                out_f.flush()
                os.fsync(out_f.fileno())

        except Exception:
            if temp_file_path.exists():
                temp_file_path.unlink(missing_ok=True)
            raise

    finally:
        if mm is not None:
            mm.close()
        if file_obj is not None:
            file_obj.close()

    # Step 6: Atomic Swap
    try:
        _atomic_replace_win32(temp_file_path, path)
    except Exception as ex:
        if temp_file_path.exists():
            temp_file_path.unlink(missing_ok=True)
        return PatchResult(
            success=False,
            target_file=str(path),
            error=f"Atomic rename failed: {ex}",
        )

    return PatchResult(
        success=True,
        target_file=str(path),
        start_line=start_line,
        end_line=end_line,
        lines_delta=lines_delta,
        occurrences=len(offsets),
    )


def main() -> int:
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
