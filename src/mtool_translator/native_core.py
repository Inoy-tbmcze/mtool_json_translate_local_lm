"""Native assembly and low-level CPU acceleration kernel for text processing.

Provides JIT-allocated x86-64 machine code execution via ctypes and
L1-cache-optimized bitmask lookups for high-throughput string inspection.
"""

from __future__ import annotations

import atexit
import ctypes
import platform
import sys
from collections.abc import Callable

# Character set definitions for symbol checking
SYMBOL_CHARS_BYTES = b"=-_*+#/\\|~<>[]{}()!@$%^&:`';"
SYMBOL_CHARS_SET = frozenset("=-_*+#/\\|~<>[]{}()!@$%^&:`';")

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
        b"\x48\x85\xd2",  # test rdx, rdx
        b"\x74",
        ("rel8", "fail"),  # jz fail
        b"\x49\x63\xc0",  # movsxd rax, r8d
        b"\x48\x39\xc2",  # cmp rdx, rax
        b"\x72",
        ("rel8", "fail"),  # jb fail
        b"\x41\x83\xf8\x01",  # cmp r8d, 1
        b"\x7e",
        ("rel8", "check_one"),  # jle check_one
        b"\x8a\x01",  # mov al, byte ptr [rcx]
        b"\x41\xb9\x01\x00\x00\x00",  # mov r9d, 1
        b"\x49\xc7\xc2\x01\x00\x00\x00",  # mov r10, 1
        "loop",
        b"\x42\x38\x04\x11",  # cmp byte ptr [rcx + r10], al
        b"\x75",
        ("rel8", "diff"),  # jne diff
        b"\x41\xff\xc1",  # inc r9d
        b"\x45\x39\xc1",  # cmp r9d, r8d (REX.R=1, REX.B=1)
        b"\x7d",
        ("rel8", "success"),  # jge success
        b"\xeb",
        ("rel8", "next"),  # jmp next
        "diff",
        b"\x42\x8a\x04\x11",  # mov al, byte ptr [rcx + r10]
        b"\x41\xb9\x01\x00\x00\x00",  # mov r9d, 1
        "next",
        b"\x49\xff\xc2",  # inc r10
        b"\x49\x39\xd2",  # cmp r10, rdx
        b"\x72",
        ("rel8", "loop"),  # jb loop
        "fail",
        b"\x31\xc0\xc3",  # xor eax, eax; ret
        "success",
        b"\xb8\x01\x00\x00\x00\xc3",  # mov eax, 1; ret
        "check_one",
        b"\x45\x85\xc0",  # test r8d, r8d
        b"\x7e",
        ("rel8", "fail"),  # jle fail
        b"\xeb",
        ("rel8", "success"),  # jmp success
    ]
    return _assemble_with_labels(instrs)


def _build_symbols_machine_code() -> bytes:
    """Builds x86-64 machine code for symbol counting via 256-byte LUT."""
    instrs: list[object] = [
        b"\x48\x31\xc0",  # xor rax, rax (count = 0)
        b"\x4d\x31\xc9",  # xor r9, r9   (i = 0)
        "loop",
        b"\x49\x39\xd1",  # cmp r9, rdx
        b"\x73",
        ("rel8", "done"),  # jae done
        b"\x46\x0f\xb6\x14\x09",  # movzx r10d, byte ptr [rcx + r9]
        b"\x47\x0f\xb6\x1c\x10",  # movzx r11d, byte ptr [r8 + r10]
        b"\x4c\x01\xd8",  # add rax, r11
        b"\x41\xff\xc1",  # inc r9
        b"\xeb",
        ("rel8", "loop"),  # jmp loop
        "done",
        b"\xc3",  # ret
    ]
    return _assemble_with_labels(instrs)


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


def _build_json_bounds_machine_code() -> bytes:
    """Builds x86-64 machine code to scan buffer for first and last JSON container delimiters."""
    instrs: list[object] = [
        # test rdx, rdx (len == 0?)
        b"\x48\x85\xd2",
        b"\x0f\x84",
        ("rel32", "done_none"),
        # r8d = -1 (first = -1)
        b"\x41\xb8\xff\xff\xff\xff",
        # r9d = -1 (last = -1)
        b"\x41\xb9\xff\xff\xff\xff",
        # r10 = 0 (offset i = 0)
        b"\x4d\x31\xd2",
        "loop",
        # cmp r10, rdx
        b"\x49\x39\xd2",
        b"\x73",
        ("rel8", "done"),
        # movzx eax, byte ptr [rcx + r10]
        b"\x42\x0f\xb6\x04\x11",
        # cmp al, 0x7B ('{')
        b"\x3c\x7b",
        b"\x74",
        ("rel8", "check_open"),
        # cmp al, 0x5B ('[')
        b"\x3c\x5b",
        b"\x74",
        ("rel8", "check_open"),
        # cmp al, 0x7D ('}')
        b"\x3c\x7d",
        b"\x74",
        ("rel8", "check_close"),
        # cmp al, 0x5D (']')
        b"\x3c\x5d",
        b"\x74",
        ("rel8", "check_close"),
        "next",
        # inc r10
        b"\x49\xff\xc2",
        b"\xeb",
        ("rel8", "loop"),
        "check_open",
        # cmp r8d, -1
        b"\x41\x83\xf8\xff",
        b"\x75",
        ("rel8", "next"),
        # mov r8d, r10d
        b"\x45\x89\xd0",
        b"\xeb",
        ("rel8", "next"),
        "check_close",
        # mov r9d, r10d
        b"\x45\x89\xd1",
        b"\xeb",
        ("rel8", "next"),
        "done",
        # movsxd rax, r8d
        b"\x49\x63\xc0",
        # shl rax, 32
        b"\x48\xc1\xe0\x20",
        # mov r11d, r9d
        b"\x45\x8b\xd9",
        # or rax, r11
        b"\x4c\x09\xd8",
        b"\xc3",
        "done_none",
        # rax = -1 (0xFFFFFFFFFFFFFFFF)
        b"\x48\xc7\xc0\xff\xff\xff\xff",
        b"\xc3",
    ]
    return _assemble_with_labels(instrs)


class NativeKernelManager:
    """Manages JIT allocation and lifetime of native x86-64 execution buffers."""

    def __init__(self) -> None:
        self.is_available = False
        self._allocated_pages: list[int] = []
        self._fn_ascii_ident: Callable[[bytes, int], int] | None = None
        self._fn_repeated_bytes: Callable[[bytes, int, int], int] | None = None
        self._fn_count_symbols: Callable[[bytes, int, bytes], int] | None = None
        self._fn_count_tokens: Callable[[bytes, int], int] | None = None
        self._fn_json_bounds: Callable[[bytes, int], int] | None = None

        if sys.platform == "win32" and platform.machine().lower() in (
            "amd64",
            "x86_64",
        ):
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
            kernel32.VirtualFree.argtypes = [
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_uint32,
            ]

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
            fn_tok_t = ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.c_char_p, ctypes.c_size_t)
            fn_bnd_t = ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.c_char_p, ctypes.c_size_t)

            self._fn_ascii_ident = alloc_native_func(_build_ascii_ident_machine_code(), fn_ascii_t)
            self._fn_repeated_bytes = alloc_native_func(
                _build_repeated_bytes_machine_code(), fn_rep_t
            )
            self._fn_count_symbols = alloc_native_func(_build_symbols_machine_code(), fn_sym_t)
            self._fn_count_tokens = alloc_native_func(_build_token_count_machine_code(), fn_tok_t)
            self._fn_json_bounds = alloc_native_func(_build_json_bounds_machine_code(), fn_bnd_t)

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

    def count_jp_and_ascii(self, raw_bytes: bytes) -> tuple[int, int]:
        """Counts Japanese (Kana/Kanji) and ASCII/Latin chars via native machine code."""
        if not raw_bytes:
            return 0, 0
        if self._fn_count_tokens is not None:
            packed = int(self._fn_count_tokens(raw_bytes, len(raw_bytes)))
            return packed >> 32, packed & 0xFFFFFFFF
        return 0, 0

    def find_json_bounds(self, raw_bytes: bytes) -> tuple[int, int]:
        """Scans buffer via native machine code to locate first and last JSON delimiters."""
        if not raw_bytes:
            return -1, -1
        if self._fn_json_bounds is not None:
            packed = int(self._fn_json_bounds(raw_bytes, len(raw_bytes)))
            first = packed >> 32
            last = packed & 0xFFFFFFFF
            first_ret = -1 if first == 0xFFFFFFFF else first
            last_ret = -1 if last == 0xFFFFFFFF else last
            return first_ret, last_ret
        return -1, -1

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


def build_japanese_bmp_bitmask(symbols: set[str]) -> bytes:
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


def fast_count_jp_and_ascii(text: str) -> tuple[int, int]:
    """Counts Japanese Kana/Kanji and ASCII/Latin characters with native machine code acceleration.

    Returns:
        Tuple of (jp_count, ascii_count).
    """
    if not text:
        return 0, 0
    if text.isascii():
        return 0, len(text)

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


def fast_find_json_bounds(raw_bytes: bytes) -> tuple[int, int]:
    """Finds first and last JSON container delimiter indices using native assembly or fallback."""
    if not raw_bytes:
        return -1, -1
    if NATIVE_MANAGER.is_available:
        first, last = NATIVE_MANAGER.find_json_bounds(raw_bytes)
        if first != -1 or last != -1:
            return first, last

    first_brace = raw_bytes.find(b"{")
    first_bracket = raw_bytes.find(b"[")
    if first_brace == -1:
        first = first_bracket
    elif first_bracket == -1:
        first = first_brace
    else:
        first = min(first_brace, first_bracket)

    last_brace = raw_bytes.rfind(b"}")
    last_bracket = raw_bytes.rfind(b"]")
    last = max(last_brace, last_bracket)
    return first, last
