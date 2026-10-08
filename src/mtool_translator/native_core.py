"""Native assembly and low-level CPU acceleration kernel for text processing.

Provides JIT-allocated x86-64 machine code execution via ctypes and
L1-cache-optimized bitmask lookups for high-throughput string inspection.
"""

from __future__ import annotations

import atexit
import ctypes
import platform
import re
import sys
from collections.abc import Callable

# Character set definitions for symbol checking
SYMBOL_CHARS_BYTES = b"=-_*+#/\\|~<>[]{}()!@$%^&:`';"
SYMBOL_CHARS_SET = frozenset("=-_*+#/\\|~<>[]{}()!@$%^&:`';")
_SYMBOL_DELETE_TRANS = str.maketrans("", "", SYMBOL_CHARS_BYTES.decode("ascii"))
_ASCII_IDENT_RE = re.compile(r"^[a-zA-Z0-9_\-.\(\) \t\r\n]+$")

# Precomputed 256-byte LUT for instant symbol identification
SYMBOL_LUT = bytearray(256)
for _b in SYMBOL_CHARS_BYTES:
    SYMBOL_LUT[_b] = 1
SYMBOL_LUT_BYTES = bytes(SYMBOL_LUT)


def _assemble_with_labels(instructions: list[object]) -> bytes:
    """Two-pass assembler resolving relative 8-bit and 32-bit jump offsets."""
    pos = 0
    labels: dict[str, int] = {}
    for item in instructions:
        if isinstance(item, str):
            labels[item] = pos
        elif isinstance(item, tuple):
            tag, _ = item
            pos += 4 if tag == "rel32" else 1
        elif isinstance(item, (bytes, bytearray)):
            pos += len(item)

    out = bytearray()
    for item in instructions:
        if isinstance(item, str):
            continue
        if isinstance(item, tuple):
            tag, target = item
            if tag == "rel32":
                disp = labels[str(target)] - (len(out) + 4)
                out.extend(disp.to_bytes(4, "little", signed=True))
            else:
                disp = labels[str(target)] - (len(out) + 1)
                out.append(disp & 0xFF)
        elif isinstance(item, (bytes, bytearray)):
            out.extend(item)
    return bytes(out)


def _build_token_count_machine_code() -> bytes:
    """Builds x86-64 machine code to count Japanese (Kana/Kanji) and ASCII/Latin chars in UTF-8."""
    instrs: list[object] = [
        # test rdx, rdx (if len == 0, return 0)
        b"\x48\x85\xd2",
        b"\x0f\x84",
        ("rel32", "done_zero"),
        # add rdx, rcx (rdx = end ptr = rcx + len)
        b"\x48\x01\xca",
        # xor r8d, r8d (jp_count = 0)
        b"\x45\x31\xc0",
        # xor r9d, r9d (ascii_count = 0)
        b"\x45\x31\xc9",
        "loop",
        # cmp rcx, rdx
        b"\x48\x39\xd1",
        # jae finish
        b"\x0f\x83",
        ("rel32", "finish"),
        # movzx eax, byte ptr [rcx]
        b"\x0f\xb6\x01",
        # test al, 0x80 (is ASCII?)
        b"\xa8\x80",
        # jnz multibyte
        b"\x75",
        ("rel8", "multibyte"),
        # inc r9d (ascii_count++)
        b"\x41\xff\xc1",
        # inc rcx
        b"\x48\xff\xc1",
        # jmp loop
        b"\xeb",
        ("rel8", "loop"),
        "multibyte",
        # cmp al, 0xE0
        b"\x3c\xe0",
        # jae check_3byte
        b"\x73",
        ("rel8", "check_3byte"),
        # 2-byte UTF-8 sequence: lea r10, [rcx + 2]
        b"\x4c\x8d\x51\x02",
        # cmp r10, rdx
        b"\x49\x39\xd2",
        # ja consume_remaining
        b"\x0f\x87",
        ("rel32", "consume_remaining"),
        # and eax, 0x1F
        b"\x83\xe0\x1f",
        # shl eax, 6
        b"\xc1\xe0\x06",
        # movzx r11d, byte ptr [rcx + 1]
        b"\x44\x0f\xb6\x59\x01",
        # and r11d, 0x3F
        b"\x41\x83\xe3\x3f",
        # or eax, r11d
        b"\x44\x09\xd8",
        # cmp eax, 0x024F (Latin script upper bound)
        b"\x3d\x4f\x02\x00\x00",
        # ja skip_2byte
        b"\x77",
        ("rel8", "skip_2byte"),
        # inc r9d (ascii_count++)
        b"\x41\xff\xc1",
        "skip_2byte",
        # add rcx, 2
        b"\x48\x83\xc1\x02",
        # jmp loop
        b"\xe9",
        ("rel32", "loop"),
        "check_3byte",
        # cmp al, 0xF0
        b"\x3c\xf0",
        # jae check_4byte
        b"\x73",
        ("rel8", "check_4byte"),
        # 3-byte UTF-8 sequence: lea r10, [rcx + 3]
        b"\x4c\x8d\x51\x03",
        # cmp r10, rdx
        b"\x49\x39\xd2",
        # ja consume_remaining
        b"\x0f\x87",
        ("rel32", "consume_remaining"),
        # Fast filter on lead byte: Kana/Kanji are strictly in 0xE3..0xE9
        # cmp al, 0xE3
        b"\x3c\xe3",
        # jb skip_3byte
        b"\x72",
        ("rel8", "skip_3byte"),
        # cmp al, 0xE9
        b"\x3c\xe9",
        # ja skip_3byte
        b"\x77",
        ("rel8", "skip_3byte"),
        # Decode 3-byte UTF-8 to 16-bit code point
        # and eax, 0x0F
        b"\x83\xe0\x0f",
        # shl eax, 12
        b"\xc1\xe0\x0c",
        # movzx r11d, byte ptr [rcx + 1]
        b"\x44\x0f\xb6\x59\x01",
        # and r11d, 0x3F
        b"\x41\x83\xe3\x3f",
        # shl r11d, 6
        b"\x41\xc1\xe3\x06",
        # or eax, r11d
        b"\x44\x09\xd8",
        # movzx r11d, byte ptr [rcx + 2]
        b"\x44\x0f\xb6\x59\x02",
        # and r11d, 0x3F
        b"\x41\x83\xe3\x3f",
        # or eax, r11d
        b"\x44\x09\xd8",
        # Kana check: 0x3040..0x30FF
        # cmp eax, 0x3040
        b"\x3d\x40\x30\x00\x00",
        # jb check_kanji
        b"\x72",
        ("rel8", "check_kanji"),
        # cmp eax, 0x30FF
        b"\x3d\xff\x30\x00\x00",
        # jbe is_jp
        b"\x76",
        ("rel8", "is_jp"),
        "check_kanji",
        # Kanji check: 0x4E00..0x9FAF
        # cmp eax, 0x4E00
        b"\x3d\x00\x4e\x00\x00",
        # jb skip_3byte
        b"\x72",
        ("rel8", "skip_3byte"),
        # cmp eax, 0x9FAF
        b"\x3d\xaf\x9f\x00\x00",
        # ja skip_3byte
        b"\x77",
        ("rel8", "skip_3byte"),
        "is_jp",
        # inc r8d (jp_count++)
        b"\x41\xff\xc0",
        "skip_3byte",
        # add rcx, 3
        b"\x48\x83\xc1\x03",
        # jmp loop
        b"\xe9",
        ("rel32", "loop"),
        "check_4byte",
        # cmp al, 0xF8
        b"\x3c\xf8",
        # jae consume_one
        b"\x73",
        ("rel8", "consume_one"),
        # lea r10, [rcx + 4]
        b"\x4c\x8d\x51\x04",
        # cmp r10, rdx
        b"\x49\x39\xd2",
        # ja consume_remaining
        b"\x0f\x87",
        ("rel32", "consume_remaining"),
        # add rcx, 4
        b"\x48\x83\xc1\x04",
        # jmp loop
        b"\xe9",
        ("rel32", "loop"),
        "consume_remaining",
        # mov rcx, rdx
        b"\x48\x89\xd1",
        # jmp finish
        b"\xeb",
        ("rel8", "finish"),
        "consume_one",
        # inc rcx
        b"\x48\xff\xc1",
        # jmp loop
        b"\xe9",
        ("rel32", "loop"),
        "finish",
        # mov rax, r8 (jp_count)
        b"\x4c\x89\xc0",
        # shl rax, 32
        b"\x48\xc1\xe0\x20",
        # or rax, r9 (ascii_count)
        b"\x4c\x09\xc8",
        # ret
        b"\xc3",
        "done_zero",
        # xor eax, eax; ret
        b"\x31\xc0\xc3",
    ]
    return _assemble_with_labels(instrs)


class NativeKernelManager:
    """Manages JIT allocation and lifetime of native x86-64 execution buffers."""

    def __init__(self) -> None:
        self.is_available = False
        self._allocated_pages: list[int] = []
        self._fn_count_tokens: Callable[[bytes, int], int] | None = None

        if sys.platform == "win32" and platform.machine().lower() in (
            "amd64",
            "x86_64",
        ):
            self._init_windows_x64()

        if self.is_available:
            atexit.register(self.cleanup)

    def _init_windows_x64(self) -> None:
        """Initializes native machine code execution on Windows AMD64 with W^X protection."""
        try:
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            kernel32.VirtualAlloc.restype = ctypes.c_void_p
            kernel32.VirtualAlloc.argtypes = [
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_uint32,
                ctypes.c_uint32,
            ]
            kernel32.VirtualProtect.argtypes = [
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_uint32),
            ]
            kernel32.VirtualFree.argtypes = [
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_uint32,
            ]

            mem_commit = 0x1000 | 0x2000
            page_rw = 0x04  # PAGE_READWRITE (enforce W^X)
            page_rx = 0x20  # PAGE_EXECUTE_READ

            def alloc_native_func(payload: bytes, func_type: type) -> Callable:
                addr = kernel32.VirtualAlloc(None, len(payload), mem_commit, page_rw)
                if not addr:
                    raise OSError("VirtualAlloc failed")
                self._allocated_pages.append(addr)
                ctypes.memmove(addr, payload, len(payload))
                old_protect = ctypes.c_uint32(0)
                if not kernel32.VirtualProtect(
                    addr, len(payload), page_rx, ctypes.byref(old_protect)
                ):
                    raise OSError("VirtualProtect failed")
                return func_type(addr)

            fn_tok_t = ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.c_char_p, ctypes.c_size_t)
            self._fn_count_tokens = alloc_native_func(_build_token_count_machine_code(), fn_tok_t)
            self.is_available = True
        except (OSError, AttributeError, RuntimeError):
            self.is_available = False

    def count_jp_and_ascii(self, raw_bytes: bytes) -> tuple[int, int]:
        """Counts Japanese (Kana/Kanji) and ASCII/Latin chars via native machine code."""
        if not raw_bytes:
            return 0, 0
        if self._fn_count_tokens is not None:
            packed = int(self._fn_count_tokens(raw_bytes, len(raw_bytes)))
            return packed >> 32, packed & 0xFFFFFFFF
        return 0, 0

    def cleanup(self) -> None:
        """Frees all JIT-allocated native executable pages."""
        if sys.platform == "win32" and self._allocated_pages:
            try:
                kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
                for page in self._allocated_pages:
                    kernel32.VirtualFree(page, 0, 0x8000)
                self._allocated_pages.clear()
            except (OSError, AttributeError):
                pass


# Global singleton kernel manager
NATIVE_MANAGER = NativeKernelManager()


def fast_is_ascii_identifier(text: str) -> bool:
    """Checks pure ASCII identifier characters using compiled C regex."""
    return bool(text and text.isascii() and _ASCII_IDENT_RE.fullmatch(text))


def fast_has_repeated_chars(text: str, min_repeat: int = 5) -> bool:
    """Checks for consecutive identical characters using zero-allocation linear sweep."""
    if not text or len(text) < min_repeat:
        return False
    count = 1
    prev = text[0]
    for char in text[1:]:
        if char == prev:
            count += 1
            if count >= min_repeat:
                return True
        else:
            prev = char
            count = 1
    return False


def fast_count_symbols(text: str) -> int:
    """Counts symbol characters using C-accelerated str.translate deletion."""
    if not text:
        return 0
    return len(text) - len(text.translate(_SYMBOL_DELETE_TRANS))


def fast_count_jp_and_ascii(text: str) -> tuple[int, int]:
    """Counts Japanese Kana/Kanji and ASCII/Latin characters with native machine code acceleration.

    Returns:
        Tuple of (jp_count, ascii_count).
    """
    if not text:
        return 0, 0
    if text.isascii():
        return 0, len(text)

    # Fast inline bypass for short strings to avoid FFI marshalling
    if len(text) <= 3:
        jp_count = 0
        ascii_count = 0
        for char in text:
            code_point = ord(char)
            if (0x3040 <= code_point <= 0x30FF) or (0x4E00 <= code_point <= 0x9FAF):
                jp_count += 1
            elif code_point <= 0x024F:
                ascii_count += 1
        return jp_count, ascii_count

    if NATIVE_MANAGER.is_available:
        return NATIVE_MANAGER.count_jp_and_ascii(text.encode("utf-8"))

    # Pure Python fallback loop
    jp_count = 0
    ascii_count = 0
    for char in text:
        code_point = ord(char)
        if (0x3040 <= code_point <= 0x30FF) or (0x4E00 <= code_point <= 0x9FAF):
            jp_count += 1
        elif code_point <= 0x024F:
            ascii_count += 1
    return jp_count, ascii_count
