"""Unified Command Line Interface for MTool JSON Translator."""

import argparse
import contextlib
import sys
from pathlib import Path
from typing import Any

from .cleaner import process_json_file
from .config import resolve_output_path
from .diff import process_diff, process_merge
from .translator import process_translation
from .utils import dump_json_file, load_json_file
from .validator import process_validation


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-branches
def _run_retranslation_recovery(
    config_file: str,
    stem: str,
    output_dir: str | None,
    retranslate_path: Path,
    summary_file: Path,
    validated_path: Path,
    translated_path: Path,
    auto_confirm: bool,
) -> None:
    """Retranslates lines that failed initial validation and merges them into final output files."""
    if not retranslate_path.exists():
        return

    failed_items: dict[str, Any] = {}
    try:
        loaded_failed = load_json_file(retranslate_path)
        if isinstance(loaded_failed, dict):
            failed_items = loaded_failed
    except (OSError, ValueError, TypeError):
        return

    if not failed_items:
        return

    print(
        f"\n[Recovery Pass] Retranslating {len(failed_items)} line(s) "
        "using existing Translation Blueprint..."
    )
    if output_dir:
        retrans_out = Path(output_dir) / f"{stem}_retranslated_pass2.json"
        retrans_prog = Path(output_dir) / f"{stem}_retranslate_progress.json"
    else:
        retrans_out = resolve_output_path(
            f"{stem}_retranslated_pass2.json", default_subfolder="processed"
        )
        retrans_prog = resolve_output_path(
            f"{stem}_retranslate_progress.json", default_subfolder="processed"
        )

    if retrans_prog.exists():
        with contextlib.suppress(OSError):
            retrans_prog.unlink()

    process_translation(
        config_file=config_file,
        input_file=str(retranslate_path),
        output_file=str(retrans_out),
        auto_confirm=auto_confirm,
        progress_file=retrans_prog,
        summary_file=summary_file,
    )

    if not retrans_out.exists():
        return

    print("\nMerging retranslated lines into final translation...")
    process_merge(base_file=str(validated_path), new_file=str(retrans_out))
    process_merge(base_file=str(translated_path), new_file=str(retrans_out))

    # Update retranslate file: retain only entries that failed completely (v == k or empty)
    still_untranslated: dict[str, Any] = {}
    try:
        retrans_data = load_json_file(retrans_out)
        if isinstance(retrans_data, dict):
            for k in failed_items:
                v = retrans_data.get(k)
                if v is None or not isinstance(v, str) or not v.strip() or v == k:
                    still_untranslated[k] = k
    except (OSError, ValueError, TypeError):
        still_untranslated = failed_items

    dump_json_file(retranslate_path, still_untranslated, indent=True)
    merged_count = len(failed_items) - len(still_untranslated)
    print(
        f"Retranslation complete: {merged_count}/{len(failed_items)} line(s) "
        "merged into final translation."
    )
    if still_untranslated:
        print(
            f"Preserved {len(still_untranslated)} untranslated line(s) "
            f"in {retranslate_path}."
        )


def run_pipeline(
    config_file: str = "config.json",
    input_file: str | None = None,
    output_dir: str | None = None,
    auto_confirm: bool = False,
    translated_file: str | None = None,
) -> tuple[Path, Path]:
    """Executes the localization pipeline (Clean -> Translate -> Validate).

    If translated_file is provided, pre-filters input_file using diff against
    existing translations and automatically merges passed translations back into
    translated_file upon completion.
    """
    pipeline_input: str | None
    if translated_file:
        print("=" * 60)
        print("Starting Incremental Pipeline: Diff -> Clean -> Translate -> Validate -> Merge")
        print("=" * 60)
        print("\n[Step 0/4] Filtering untranslated keys (Diff)...")
        target_diff_input = input_file or "ManualTransFile.json"
        process_diff(current_file=target_diff_input, translated_file=translated_file)
        pipeline_input = target_diff_input
    else:
        print("=" * 60)
        print("Starting Localization Pipeline: Clean -> Translate -> Validate")
        print("=" * 60)
        pipeline_input = input_file

    # 1. Clean
    print("\n[Step 1/3] Preprocessing and Cleaning...")
    cleaned_path, _ = process_json_file(
        config_file=config_file, input_file=pipeline_input, output_dir=output_dir
    )

    # 2. Translate
    print("\n[Step 2/3] Translating cleaned text...")
    stem = cleaned_path.stem.replace("_cleaned", "")
    trans_stem = f"{stem}_incremental" if translated_file else stem
    if output_dir:
        out_trans = Path(output_dir) / f"{trans_stem}_translated.json"
        progress_file = Path(output_dir) / f"{trans_stem}_progress.json"
        summary_file = Path(output_dir) / f"{stem}_summary.txt"
    else:
        out_trans = resolve_output_path(
            f"{trans_stem}_translated.json", default_subfolder="processed"
        )
        progress_file = resolve_output_path(
            f"{trans_stem}_progress.json", default_subfolder="processed"
        )
        summary_file = resolve_output_path(
            f"{stem}_summary.txt", default_subfolder="processed"
        )

    # Collision guard: Ensure out_trans never collides with or overwrites translated_file
    if translated_file and out_trans.resolve() == Path(translated_file).resolve():
        out_trans = out_trans.with_name(f"{stem}_incremental_translated.json")
        progress_file = progress_file.with_name(f"{stem}_incremental_progress.json")

    translated_path = process_translation(
        config_file=config_file,
        input_file=str(cleaned_path),
        output_file=str(out_trans),
        auto_confirm=auto_confirm,
        progress_file=progress_file,
        summary_file=summary_file,
    )

    # 3. Validate
    print("\n[Step 3/3] Auditing translation quality...")
    validated_path, retranslate_path = process_validation(
        config_file=config_file, input_file=str(translated_path), output_dir=output_dir
    )

    # 4. Optional Automatic Recovery Pass: Retranslate failed lines using existing Blueprint
    _run_retranslation_recovery(
        config_file=config_file,
        stem=trans_stem,
        output_dir=output_dir,
        retranslate_path=retranslate_path,
        summary_file=summary_file,
        validated_path=validated_path,
        translated_path=translated_path,
        auto_confirm=auto_confirm,
    )

    # 5. Merge if translated_file was provided
    if translated_file:
        print("\n[Final Step] Merging validated translations back into master file...")
        process_merge(base_file=translated_file, new_file=str(validated_path))

    print("\n" + "=" * 60)
    print("Pipeline Execution Complete!")
    print(f"Passed Translations:  {validated_path}")
    print(f"Retranslate File:     {retranslate_path}")
    print("=" * 60)

    return validated_path, retranslate_path


def build_parser() -> argparse.ArgumentParser:
    """Builds and configures the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="MTool Game Localization JSON Translator (Local LLM)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Clean sub-command
    clean_p = subparsers.add_parser(
        "clean", help="Stage 1: Preprocess raw game JSON and quarantine junk"
    )
    clean_p.add_argument("-i", "--input", help="Path to input JSON file", default=None)
    clean_p.add_argument("-c", "--config", help="Path to config.json", default="config.json")
    clean_p.add_argument("-o", "--output-dir", help="Target output directory", default=None)

    # Translate sub-command
    trans_p = subparsers.add_parser(
        "translate", help="Stage 2: Translate Japanese JSON text to English"
    )
    trans_p.add_argument("-i", "--input", help="Path to input JSON file", default=None)
    trans_p.add_argument("-o", "--output", help="Path to output JSON file", default=None)
    trans_p.add_argument("-c", "--config", help="Path to config.json", default="config.json")
    trans_p.add_argument(
        "-y", "--yes", action="store_true", help="Auto-confirm prompts without pausing"
    )

    # Validate sub-command
    val_p = subparsers.add_parser(
        "validate",
        help="Stage 3: Validate translations and isolate lines for retranslation",
    )
    val_p.add_argument("-i", "--input", help="Path to input JSON file", default=None)
    val_p.add_argument("-c", "--config", help="Path to config.json", default="config.json")
    val_p.add_argument("-o", "--output-dir", help="Target output directory", default=None)

    # Full pipeline sub-command
    pipe_p = subparsers.add_parser(
        "pipeline", help="Run full pipeline: Clean -> Translate -> Validate"
    )
    pipe_p.add_argument("-i", "--input", help="Path to initial raw input JSON file", default=None)
    pipe_p.add_argument(
        "-t",
        "--translated",
        help="Path to existing master translated JSON (enables incremental diff and merge)",
        default=None,
    )
    pipe_p.add_argument("-o", "--output-dir", help="Target output directory", default=None)
    pipe_p.add_argument("-c", "--config", help="Path to config.json", default="config.json")
    pipe_p.add_argument("-y", "--yes", action="store_true", help="Auto-confirm prompts")

    # Diff sub-command
    diff_p = subparsers.add_parser(
        "diff",
        help="Filter out already translated keys from current ManualTransFile.json",
    )
    diff_p.add_argument(
        "-i",
        "--input",
        help="Path to current/updated game JSON file",
        default="ManualTransFile.json",
    )
    diff_p.add_argument(
        "-t",
        "--translated",
        help="Path to existing translated JSON reference file",
        default="ManualTransFile_translated.json",
    )
    diff_p.add_argument(
        "-o",
        "--output",
        help="Path to save remaining untranslated keys (default: modifies --input in-place)",
        default=None,
    )
    diff_p.add_argument(
        "--key-presence",
        action="store_true",
        help="Treat any key present in translated file as translated",
    )

    # Merge sub-command
    merge_p = subparsers.add_parser(
        "merge",
        help="Merge newly translated lines back into master translated file",
    )
    merge_p.add_argument(
        "-b",
        "--base",
        help="Path to master translated JSON file to update",
        default="ManualTransFile_translated.json",
    )
    merge_p.add_argument(
        "-n",
        "--new",
        help="Path to newly translated JSON file to merge in",
        required=True,
    )
    merge_p.add_argument(
        "-o",
        "--output",
        help="Path to save merged JSON file (default: modifies --base in-place)",
        default=None,
    )
    merge_p.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Do not overwrite existing keys in base file",
    )

    return parser


def main():
    """Main CLI dispatch entrypoint."""
    parser = build_parser()

    # If no arguments provided, default to translation stage for backward compatibility
    if len(sys.argv) == 1:
        print("MTool JSON Translation Engine (Default Mode: Translate)")
        print(
            "Use --help to view available commands: "
            "clean, translate, validate, pipeline, diff, merge.\n"
        )
        process_translation(config_file="config.json", auto_confirm=False)
        return

    args = parser.parse_args()

    if args.command == "clean":
        process_json_file(
            config_file=args.config, input_file=args.input, output_dir=args.output_dir
        )
    elif args.command == "translate":
        process_translation(
            config_file=args.config,
            input_file=args.input,
            output_file=args.output,
            auto_confirm=args.yes,
        )
    elif args.command == "validate":
        process_validation(
            config_file=args.config, input_file=args.input, output_dir=args.output_dir
        )
    elif args.command == "pipeline":
        run_pipeline(
            config_file=args.config,
            input_file=args.input,
            output_dir=args.output_dir,
            auto_confirm=args.yes,
            translated_file=args.translated,
        )
    elif args.command == "diff":
        process_diff(
            current_file=args.input,
            translated_file=args.translated,
            output_file=args.output,
            key_presence_only=args.key_presence,
        )
    elif args.command == "merge":
        process_merge(
            base_file=args.base,
            new_file=args.new,
            output_file=args.output,
            overwrite_existing=not args.no_overwrite,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
