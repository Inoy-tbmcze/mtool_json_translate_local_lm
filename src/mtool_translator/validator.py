"""
Stage 3 Translation Validation Module for Game Localization JSON.
------------------------------------------------------------------
Validates Japanese -> English translations using local LLM auditor.
Separates passed translations from failed lines needing retranslation.
"""

from __future__ import annotations

import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import requests

from .config import load_config, resolve_input_path, resolve_output_path


@dataclass
class ValidationPaths:
    """Encapsulates input, output, and checkpoint paths for validation."""

    input_file: Path
    valid: Path
    retranslate: Path
    checkpoint: Path


@dataclass
class ValidationState:
    """Encapsulates validation dictionaries and tracked keys."""

    validated_data: Dict[str, Any]
    retranslate_data: Dict[str, Any]
    processed_keys: Set[str]


def _parse_validation_json(content: str) -> Dict[str, Any]:
    """Extracts and parses JSON object from LLM response content."""
    text = re.sub(r"```(?:json)?|```", "", content).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    parsed: Dict[str, Any] = json.loads(match.group(0) if match else text)
    return parsed


def call_batch_validation(
    batch: List[Tuple[int, str, str]], config: Dict[str, Any]
) -> Dict[str, bool]:
    """Sends batch of (ID, JP_source, EN_target) to LLM for translation validation."""
    items_str = "\n".join(f"{idx} | JP: {jp} | EN: {en}" for idx, jp, en in batch)
    prompt = (
        "You are a translation quality auditor. Check if the English(second value) "
        "text is a valid, plausible translation or equivalent representation "
        "of the Japanese(first value) source text.\n\n"
        "Rules:\n"
        "- Return 1 only if the English text accurately matches or contextually "
        "represents translation of Japanese source.\n"
        "- Return 0 if anything else.\n\n"
        "Output ONLY a raw JSON object mapping ID to 1 or 0.\n"
        'Example format: {"0":1, "1":0, "2":1}\n\n'
        f"Pairs to validate:\n{items_str}"
    )

    data = {
        "model": config["model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 1024,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.get('api_key', 'lm-studio')}",
    }

    results = {jp: True for _idx, jp, _en in batch}
    try:
        resp = requests.post(
            config["api_endpoint"], headers=headers, json=data, timeout=config["request_timeout"]
        )
        resp.raise_for_status()
        raw_content = resp.json()["choices"][0]["message"]["content"].strip()
        parsed = _parse_validation_json(raw_content)
        for idx, jp, _en in batch:
            val = parsed.get(str(idx), 1)
            results[jp] = bool(val == 1 or val is True or str(val).lower() == "true")
    except (requests.RequestException, json.JSONDecodeError, KeyError, ValueError) as err:
        print(f"\nWarning: Batch validation request failed ({err}). Defaulting items to VALID (1).")

    return results


def save_progress(
    paths: ValidationPaths, state: ValidationState, lock: Optional[threading.RLock] = None
) -> None:
    """Safely writes validation results (*_validated.json, *_retranslate.json) and checkpoints."""

    def _write_files() -> None:
        paths.valid.parent.mkdir(parents=True, exist_ok=True)
        paths.retranslate.parent.mkdir(parents=True, exist_ok=True)
        paths.checkpoint.parent.mkdir(parents=True, exist_ok=True)

        with open(paths.valid, "w", encoding="utf-8") as file_v:
            json.dump(state.validated_data, file_v, ensure_ascii=False, indent=2)

        with open(paths.retranslate, "w", encoding="utf-8") as file_r:
            json.dump(state.retranslate_data, file_r, ensure_ascii=False, indent=2)

        checkpoint_keys = list(state.validated_data.keys()) + list(state.retranslate_data.keys())
        with open(paths.checkpoint, "w", encoding="utf-8") as file_c:
            json.dump({"processed_keys": checkpoint_keys}, file_c, ensure_ascii=False, indent=2)
        print(" -> Autosave successful.")

    try:
        if lock is not None:
            with lock:
                _write_files()
        else:
            _write_files()
    except OSError as err:
        print(f" -> Error during autosave: {err}")


def _init_validation_paths(
    config: Dict[str, Any], input_file: Optional[str], output_dir: Optional[str]
) -> ValidationPaths:
    """Resolves input, output, and checkpoint paths for translation validation."""
    input_filename = input_file or config.get("input_filename", "translated_game_text.json")
    input_path = resolve_input_path(input_filename, default_subfolder="processed")
    if not input_path.exists():
        input_path = resolve_input_path(input_filename, default_subfolder="raw")

    if not input_path.exists():
        print(f"Error: Input file '{input_filename}' not found at {input_path}.")
        sys.exit(1)

    stem, ext = input_path.stem, input_path.suffix

    if output_dir:
        out_dir = Path(output_dir)
        valid_path = out_dir / f"{stem}_validated{ext}"
        retranslate_path = out_dir / f"{stem}_retranslate{ext}"
        checkpoint_path = out_dir / f"{stem}_checkpoint.json"
    else:
        valid_path = resolve_output_path(f"{stem}_validated{ext}", default_subfolder="processed")
        retranslate_path = resolve_output_path(
            f"{stem}_retranslate{ext}", default_subfolder="processed"
        )
        checkpoint_path = resolve_output_path(
            f"{stem}_checkpoint.json", default_subfolder="processed"
        )

    return ValidationPaths(input_path, valid_path, retranslate_path, checkpoint_path)


def _load_validation_state(paths: ValidationPaths) -> ValidationState:
    """Loads existing validated data, retranslation data, and checkpoint keys."""
    validated_data: Dict[str, Any] = {}
    retranslate_data: Dict[str, Any] = {}
    processed_keys: Set[str] = set()

    if paths.checkpoint.exists():
        try:
            with open(paths.checkpoint, "r", encoding="utf-8") as f:
                chk = json.load(f)
                processed_keys = set(chk.get("processed_keys", []))
            print(f"--> Found existing checkpoint: {len(processed_keys)} items already processed.")
        except (json.JSONDecodeError, OSError) as err:
            print(f"Warning: Could not read checkpoint file ({err}). Starting fresh.")

    for check_p, data_dict in [
        (paths.valid, validated_data),
        (paths.retranslate, retranslate_data),
    ]:
        if check_p.exists():
            try:
                with open(check_p, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    data_dict.update(loaded)
                    processed_keys.update(loaded.keys())
            except (json.JSONDecodeError, OSError):
                pass

    return ValidationState(validated_data, retranslate_data, processed_keys)


def _process_validation_wave(
    current_wave: List[List[Tuple[int, str, str]]],
    config: Dict[str, Any],
    lock: threading.RLock,
    state: ValidationState,
) -> None:
    """Processes a single parallel wave of validation batches."""
    with ThreadPoolExecutor(max_workers=len(current_wave)) as executor:
        future_to_batch = {
            executor.submit(call_batch_validation, batch, config): batch for batch in current_wave
        }
        for future in as_completed(future_to_batch):
            batch = future_to_batch[future]
            results = future.result()
            with lock:
                for _idx, jp, en in batch:
                    if results.get(jp, True):
                        state.validated_data[jp] = en
                    else:
                        state.retranslate_data[jp] = jp
                    state.processed_keys.add(jp)


def _run_validation_waves(
    batches: List[List[Tuple[int, str, str]]],
    config: Dict[str, Any],
    paths: ValidationPaths,
    state: ValidationState,
    lock: threading.RLock,
) -> None:
    """Runs batch waves and saves checkpoints periodically."""
    max_workers = config.get("max_workers", 4)
    save_interval = config.get("save_interval", 10)
    waves = [batches[i : i + max_workers] for i in range(0, len(batches), max_workers)]

    for wave_idx, current_wave in enumerate(waves, 1):
        _process_validation_wave(current_wave, config, lock, state)
        if wave_idx % save_interval == 0 or wave_idx == len(waves):
            print(f"Wave {wave_idx}/{len(waves)} complete. Autosaving progress...")
            save_progress(paths, state, lock)


def process_validation(
    config_file: str = "config.json",
    input_file: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Tuple[Path, Path]:
    """Audits translated JSON file, splitting passed items and items needing retranslation."""
    config = load_config(config_file, section="validation")
    paths = _init_validation_paths(config, input_file, output_dir)
    state = _load_validation_state(paths)

    with open(paths.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    unprocessed_items = [(jp, en) for jp, en in data.items() if jp not in state.processed_keys]

    if not unprocessed_items:
        print("All items have already been validated.")
        return paths.valid, paths.retranslate

    batch_size = config.get("batch_size", 20)
    batches = [
        [(idx, jp, en) for idx, (jp, en) in enumerate(unprocessed_items[i : i + batch_size])]
        for i in range(0, len(unprocessed_items), batch_size)
    ]

    print(
        f"--- Validating {len(unprocessed_items)} remaining lines across {len(batches)} batches ---"
    )

    lock = threading.RLock()
    _run_validation_waves(batches, config, paths, state, lock)
    save_progress(paths, state, lock)

    print("\n=== Validation Complete ===")
    print(f"Total Lines Processed: {len(data)}")
    print(f"Validated File (1s):   {len(state.validated_data)} lines -> {paths.valid}")
    print(f"Retranslate File (0s): {len(state.retranslate_data)} lines -> {paths.retranslate}")

    return paths.valid, paths.retranslate
