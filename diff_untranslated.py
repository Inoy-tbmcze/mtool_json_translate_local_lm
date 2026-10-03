"""Translation Diff & Merge Entrypoint (Standalone Script & Library Wrapper).
Delegates to mtool_translator.diff.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src/ is on sys.path for direct script execution
_SRC_DIR = Path(__file__).resolve().parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# pylint: disable=wrong-import-position
from mtool_translator.diff import (
    filter_untranslated_keys,
    is_entry_translated,
    merge_translations,
    process_diff,
    process_merge,
)

__all__ = [
    "filter_untranslated_keys",
    "is_entry_translated",
    "main",
    "merge_translations",
    "process_diff",
    "process_merge",
]


def build_parser() -> argparse.ArgumentParser:
    """Constructs the argument parser for diff and merge operations."""
    parser = argparse.ArgumentParser(
        description="MTool Translation Diff & Merge Tool (Game Update Support)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="subcommand", help="Subcommand to execute")

    # diff subcommand
    diff_p = subparsers.add_parser(
        "diff",
        help="Filter out already translated keys from current ManualTransFile.json",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    diff_p.add_argument(
        "-i",
        "--input",
        default="ManualTransFile.json",
        help="Path to current/updated game JSON file",
    )
    diff_p.add_argument(
        "-t",
        "--translated",
        default="ManualTransFile_translated.json",
        help="Path to existing translated JSON reference file",
    )
    diff_p.add_argument(
        "-o",
        "--output",
        default=None,
        help="Path to save remaining untranslated keys (default: modifies --input in-place)",
    )
    diff_p.add_argument(
        "--key-presence",
        action="store_true",
        help="Treat any key present in translated file as translated (bypasses safe check)",
    )

    # merge subcommand
    merge_p = subparsers.add_parser(
        "merge",
        help="Merge newly translated JSON lines back into master translated file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    merge_p.add_argument(
        "-b",
        "--base",
        default="ManualTransFile_translated.json",
        help="Path to master translated JSON file to update",
    )
    merge_p.add_argument(
        "-n",
        "--new",
        required=True,
        help="Path to newly translated JSON file",
    )
    merge_p.add_argument(
        "-o",
        "--output",
        default=None,
        help="Path to save merged JSON file (default: modifies --base in-place)",
    )
    merge_p.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Do not overwrite existing keys in base file",
    )

    # Top-level fallback flags when run without subcommands (defaults to diff)
    parser.add_argument(
        "-i",
        "--input",
        default="ManualTransFile.json",
        help="[Diff mode] Path to current/updated game JSON file",
    )
    parser.add_argument(
        "-t",
        "--translated",
        default="ManualTransFile_translated.json",
        help="[Diff mode] Path to existing translated JSON reference file",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Path to output JSON file (default: modifies input/base in-place)",
    )
    parser.add_argument(
        "--key-presence",
        action="store_true",
        help="[Diff mode] Treat any key in translated file as translated",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Switch top-level mode to merge",
    )
    parser.add_argument(
        "-b",
        "--base",
        default="ManualTransFile_translated.json",
        help="[Merge mode] Path to master translated JSON file",
    )
    parser.add_argument(
        "-n",
        "--new",
        default=None,
        help="[Merge mode] Path to newly translated JSON file",
    )

    return parser


def main() -> None:
    """CLI execution entrypoint."""
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.subcommand == "merge" or args.merge:
            if not args.new:
                parser.error("Merge mode requires -n / --new argument.")
            process_merge(
                base_file=args.base,
                new_file=args.new,
                output_file=args.output,
                overwrite_existing=not getattr(args, "no_overwrite", False),
            )
        else:
            process_diff(
                current_file=args.input,
                translated_file=args.translated,
                output_file=args.output,
                key_presence_only=args.key_presence,
            )
    except (FileNotFoundError, TypeError, ValueError) as err:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
