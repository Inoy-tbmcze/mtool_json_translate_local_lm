"""Unified Command Line Interface for MTool JSON Translator."""

import argparse
import sys
from pathlib import Path
from typing import Optional

from .cleaner import process_json_file
from .translator import process_translation
from .validator import process_validation


def run_pipeline(config_file: str = "config.json", input_file: Optional[str] = None, auto_confirm: bool = False):
    """Executes the full Stage 1 -> Stage 2 -> Stage 3 pipeline."""
    print("=" * 60)
    print("Starting Localization Pipeline: Clean -> Translate -> Validate")
    print("=" * 60)

    # 1. Clean
    print("\n[Step 1/3] Preprocessing and Cleaning...")
    cleaned_path, _ = process_json_file(config_file=config_file, input_file=input_file)

    # 2. Translate
    print("\n[Step 2/3] Translating cleaned text...")
    translated_path = process_translation(
        config_file=config_file,
        input_file=str(cleaned_path),
        auto_confirm=auto_confirm
    )

    # 3. Validate
    print("\n[Step 3/3] Auditing translation quality...")
    validated_path, retranslate_path = process_validation(
        config_file=config_file,
        input_file=str(translated_path)
    )

    print("\n" + "=" * 60)
    print("Pipeline Execution Complete!")
    print(f"Passed Translations:  {validated_path}")
    print(f"Retranslate File:     {retranslate_path}")
    print("=" * 60)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MTool Game Localization JSON Translator (Local LLM)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Clean sub-command
    clean_p = subparsers.add_parser("clean", help="Stage 1: Preprocess raw game JSON and quarantine junk")
    clean_p.add_argument("-i", "--input", help="Path to input JSON file", default=None)
    clean_p.add_argument("-c", "--config", help="Path to config.json", default="config.json")
    clean_p.add_argument("-o", "--output-dir", help="Target output directory", default=None)

    # Translate sub-command
    trans_p = subparsers.add_parser("translate", help="Stage 2: Translate Japanese JSON text to English")
    trans_p.add_argument("-i", "--input", help="Path to input JSON file", default=None)
    trans_p.add_argument("-o", "--output", help="Path to output JSON file", default=None)
    trans_p.add_argument("-c", "--config", help="Path to config.json", default="config.json")
    trans_p.add_argument("-y", "--yes", action="store_true", help="Auto-confirm prompts without pausing")

    # Validate sub-command
    val_p = subparsers.add_parser("validate", help="Stage 3: Validate translations and isolate lines for retranslation")
    val_p.add_argument("-i", "--input", help="Path to input JSON file", default=None)
    val_p.add_argument("-c", "--config", help="Path to config.json", default="config.json")
    val_p.add_argument("-o", "--output-dir", help="Target output directory", default=None)

    # Full pipeline sub-command
    pipe_p = subparsers.add_parser("pipeline", help="Run full pipeline: Clean -> Translate -> Validate")
    pipe_p.add_argument("-i", "--input", help="Path to initial raw input JSON file", default=None)
    pipe_p.add_argument("-c", "--config", help="Path to config.json", default="config.json")
    pipe_p.add_argument("-y", "--yes", action="store_true", help="Auto-confirm prompts")

    return parser


def main():
    parser = build_parser()

    # If no arguments provided, default to translation stage for backward compatibility
    if len(sys.argv) == 1:
        print("MTool JSON Translation Engine (Default Mode: Translate)")
        print("Use --help to view available commands: clean, translate, validate, pipeline.\n")
        process_translation(config_file="config.json", auto_confirm=False)
        return

    args = parser.parse_args()

    if args.command == "clean":
        process_json_file(
            config_file=args.config,
            input_file=args.input,
            output_dir=args.output_dir
        )
    elif args.command == "translate":
        process_translation(
            config_file=args.config,
            input_file=args.input,
            output_file=args.output,
            auto_confirm=args.yes
        )
    elif args.command == "validate":
        process_validation(
            config_file=args.config,
            input_file=args.input,
            output_dir=args.output_dir
        )
    elif args.command == "pipeline":
        run_pipeline(
            config_file=args.config,
            input_file=args.input,
            auto_confirm=args.yes
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
