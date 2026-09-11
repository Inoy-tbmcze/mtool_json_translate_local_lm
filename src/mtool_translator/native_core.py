"""Native assembly and low-level CPU acceleration kernel for text processing.

Provides JIT-allocated x86-64 machine code execution via ctypes and
L1-cache-optimized bitmask lookups for high-throughput string inspection.
"""

from __future__ import annotations

import atexit
import ctypes
import platform
import sys
from typing import Callable, Optional, Set

# Character set definitions for symbol checking
SYMBOL_CHARS_BYTES = b"=-_*+#/\\|~<>[]{}()!@$%^&:`';"
SYMBOL_CHARS_SET = frozenset("=-_*+#/\\|~<>[]{}()!@$%^&:`';")

# Precomputed 256-byte LUT for instant symbol identification
SYMBOL_LUT = bytearray(256)
for _b in SYMBOL_CHARS_BYTES:
    SYMBOL_LUT[_b] = 1
SYMBOL_LUT_BYTES = bytes(SYMBOL_LUT)


def _assemble_with_labels(instructions: list[object]) -> bytes:
    """Two-pass assembler resolving relative 8-bit jump offsets."""
    pos = 0
    labels: dict[str, int] = {}
    for item in instructions:
        if isinstance(item, str):
            labels[item] = pos
        elif isinstance(item, tuple):
            pos += 1
        elif isinstance(item, (bytes, bytearray)):
            pos += len(item)

    out = bytearray()
    for item in instructions:
        if isinstance(item, str):
            continue
        if isinstance(item, tuple):
            _tag, target = item
            disp = labels[str(target)] - (len(out) + 1)
            out.append(disp & 0xFF)
        elif isinstance(item, (bytes, bytearray)):
            out.extend(item)
    return bytes(out)


def _build_ascii_ident_machine_code() -> bytes:
    """Builds self-contained x86-64 machine code for pure ASCII identifier matching."""
    allowed = set(b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.() \t\r\n")
    table = bytearray(256)
    for char_byte in allowed:
        table[char_byte] = 1

    code = bytearray(
        [
            0x48,
            0x85,
            0xD2,  # test rdx, rdx (len == 0?)
            0x74,
            0x24,  # jz .fail (+36 -> 0x29)
            0x4C,
            0x8D,
            0x05,
            0x20,
            0x00,
            0x00,
            0x00,  # lea r8, [rip + 0x20] (table at 0x2C)
            0x4D,
            0x31,
            0xC9,  # xor r9, r9 (i = 0)
            # .loop (0x0F):
            0x42,
            0x0F,
            0xB6,
            0x04,
            0x09,  # movzx eax, byte ptr [rcx + r9]
            0x41,
            0x80,
            0x3C,
            0x00,
            0x00,  # cmp byte ptr [r8 + rax], 0
            0x74,
            0x0E,  # je .fail (+14 -> 0x29)
            0x49,
            0xFF,
            0xC1,  # inc r9
            0x49,
            0x39,
            0xD1,  # cmp r9, rdx
            0x72,
            0xEC,  # jb .loop (-20 -> 0x0F)
            0xB8,
            0x01,
            0x00,
            0x00,
            0x00,  # mov eax, 1
            0xC3,  # ret
            # .fail (0x29):
            0x31,
            0xC0,  # xor eax, eax
            0xC3,  # ret
        ]
    )
    return bytes(code + table)


def _build_repeated_bytes_machine_code() -> bytes:
    """Builds x86-64 machine code for finding consecutive identical byte runs."""
    instrs: list[object] = [
        b"\x48\x85\xD2",  # test rdx, rdx
        b"\x74",
        ("rel8", "fail"),  # jz fail
        b"\x49\x63\xC0",  # movsxd rax, r8d
        b"\x48\x39\xC2",  # cmp rdx, rax
        b"\x72",
        ("rel8", "fail"),  # jb fail
        b"\x41\x83\xF8\x01",  # cmp r8d, 1
        b"\x7E",
        ("rel8", "check_one"),  # jle check_one
        b"\x8A\x01",  # mov al, byte ptr [rcx]
        b"\x41\xB9\x01\x00\x00\x00",  # mov r9d, 1
        b"\x49\xC7\xC2\x01\x00\x00\x00",  # mov r10, 1
        "loop",
        b"\x42\x38\x04\x11",  # cmp byte ptr [rcx + r10], al
        b"\x75",
        ("rel8", "diff"),  # jne diff
        b"\x41\xFF\xC1",  # inc r9d
        b"\x45\x39\xC1",  # cmp r9d, r8d (REX.R=1, REX.B=1)
        b"\x7D",
        ("rel8", "success"),  # jge success
        b"\xEB",
        ("rel8", "next"),  # jmp next
        "diff",
        b"\x42\x8A\x04\x11",  # mov al, byte ptr [rcx + r10]
        b"\x41\xB9\x01\x00\x00\x00",  # mov r9d, 1
        "next",
        b"\x49\xFF\xC2",  # inc r10
        b"\x49\x39\xD2",  # cmp r10, rdx
        b"\x72",
        ("rel8", "loop"),  # jb loop
        "fail",
        b"\x31\xC0\xC3",  # xor eax, eax; ret
        "success",
        b"\xB8\x01\x00\x00\x00\xC3",  # mov eax, 1; ret
        "check_one",
        b"\x45\x85\xC0",  # test r8d, r8d
        b"\x7E",
        ("rel8", "fail"),  # jle fail
        b"\xEB",
        ("rel8", "success"),  # jmp success
    ]
    return _assemble_with_labels(instrs)


def _build_symbols_machine_code() -> bytes:
    """Builds x86-64 machine code for symbol counting via 256-byte LUT."""
    instrs: list[object] = [
        b"\x48\x31\xC0",  # xor rax, rax (count = 0)
        b"\x4D\x31\xC9",  # xor r9, r9   (i = 0)
        "loop",
        b"\x49\x39\xD1",  # cmp r9, rdx
        b"\x73",
        ("rel8", "done"),  # jae done
        b"\x46\x0F\xB6\x14\x09",  # movzx r10d, byte ptr [rcx + r9]
        b"\x47\x0F\xB6\x1C\x10",  # movzx r11d, byte ptr [r8 + r10]
        b"\x4C\x01\xD8",  # add rax, r11
        b"\x41\xFF\xC1",  # inc r9
        b"\xEB",
        ("rel8", "loop"),  # jmp loop
        "done",
        b"\xC3",  # ret
    ]
    return _assemble_with_labels(instrs)


class NativeKernelManager:
    """Manages JIT allocation and lifetime of native x86-64 execution buffers."""

    def __init__(self) -> None:
        self.is_available = False
        self._allocated_pages: list[int] = []
        self._fn_ascii_ident: Optional[Callable[[bytes, int], int]] = None
        self._fn_repeated_bytes: Optional[Callable[[bytes, int, int], int]] = None
        self._fn_count_symbols: Optional[Callable[[bytes, int, bytes], int]] = None

        if sys.platform == "win32" and platform.machine().lower() in ("amd64", "x86_64"):
            self._init_windows_x64()

        if self.is_available:
            atexit.register(self.cleanup)

    def _init_windows_x64(self) -> None:
        """Initializes native machine code execution on Windows AMD64."""
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
            kernel32.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32]

            mem_commit = 0x1000 | 0x2000
            page_rwx = 0x40

            def alloc_native_func(payload: bytes, func_type: type) -> Callable:
                addr = kernel32.VirtualAlloc(None, len(payload), mem_commit, page_rwx)
                if not addr:
                    raise OSError("VirtualAlloc failed")
                self._allocated_pages.append(addr)
                ctypes.memmove(addr, payload, len(payload))
                return func_type(addr)

            fn_ascii_t = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_char_p, ctypes.c_size_t)
            fn_rep_t = ctypes.CFUNCTYPE(
                ctypes.c_int, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_int
            )
            fn_sym_t = ctypes.CFUNCTYPE(
                ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p
            )

            self._fn_ascii_ident = alloc_native_func(_build_ascii_ident_machine_code(), fn_ascii_t)
            self._fn_repeated_bytes = alloc_native_func(
                _build_repeated_bytes_machine_code(), fn_rep_t
            )
            self._fn_count_symbols = alloc_native_func(_build_symbols_machine_code(), fn_sym_t)

            self.is_available = True
        except (OSError, AttributeError, RuntimeError):
            self.is_available = False

    def is_ascii_ident(self, raw_bytes: bytes) -> bool:
        """Invokes native machine code to check pure ASCII identifier status."""
        if not raw_bytes:
            return False
        if self._fn_ascii_ident is not None:
            return bool(self._fn_ascii_ident(raw_bytes, len(raw_bytes)) == 1)
        return False

    def has_repeated_bytes(self, raw_bytes: bytes, min_repeat: int = 5) -> bool:
        """Invokes native machine code to check for consecutive repeated bytes."""
        if not raw_bytes:
            return False
        if self._fn_repeated_bytes is not None:
            return bool(self._fn_repeated_bytes(raw_bytes, len(raw_bytes), min_repeat) == 1)
        return False

    def count_symbols(self, raw_bytes: bytes) -> int:
        """Invokes native machine code to count symbols via 256-byte LUT."""
        if not raw_bytes:
            return 0
        if self._fn_count_symbols is not None:
            return int(self._fn_count_symbols(raw_bytes, len(raw_bytes), SYMBOL_LUT_BYTES))
        return 0

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


def build_japanese_bmp_bitmask(symbols: Set[str]) -> bytes:
    """Builds an 8,192-byte bitmask covering all 65,536 code points of the Unicode BMP."""
    bitmask = bytearray(8192)

    def _set_bit(cp: int) -> None:
        if 0 <= cp < 65536:
            bitmask[cp >> 3] |= 1 << (cp & 7)

    # Hiragana (0x3040..0x309F)
    for code_point in range(0x3040, 0x30A0):
        _set_bit(code_point)

    # Katakana (0x30A0..0x30FF)
    for code_point in range(0x30A0, 0x3100):
        _set_bit(code_point)

    # CJK Unified Ideographs (0x4E00..0x9FAF)
    for code_point in range(0x4E00, 0x9FB0):
        _set_bit(code_point)

    # Custom Japanese symbols and punctuation
    for sym in symbols:
        for char in sym:
            _set_bit(ord(char))

    return bytes(bitmask)


def fast_is_ascii_identifier(text: str) -> bool:
    """Checks pure ASCII identifier characters using native assembly or bitmask."""
    if not text:
        return False
    if not text.isascii():
        return False

    raw = text.encode("ascii", "replace")
    if NATIVE_MANAGER.is_available:
        return NATIVE_MANAGER.is_ascii_ident(raw)

    allowed = set(b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.() \t\r\n")
    return all(b in allowed for b in raw)


def fast_has_repeated_chars(text: str, min_repeat: int = 5) -> bool:
    """Checks for consecutive identical characters using native assembly with linear sweep."""
    if not text or len(text) < min_repeat:
        return False

    if text.isascii():
        raw = text.encode("ascii", "replace")
        if NATIVE_MANAGER.is_available:
            return NATIVE_MANAGER.has_repeated_bytes(raw, min_repeat)

    # Fast linear sweep fallback
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
    """Counts symbol characters using native assembly 256-byte LUT or set containment."""
    if not text:
        return 0

    if text.isascii():
        raw = text.encode("ascii", "replace")
        if NATIVE_MANAGER.is_available:
            return NATIVE_MANAGER.count_symbols(raw)

    return sum(1 for char in text if char in SYMBOL_CHARS_SET)
