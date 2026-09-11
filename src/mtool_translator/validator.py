"""Stage 3 Translation Validation Module for Game Localization JSON.

Validates Japanese -> English translations using local LLM auditor.
Separates passed translations from failed lines needing retranslation.
"""

from __future__ import annotations

import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import load_config, resolve_input_path, resolve_output_path
from .http_client import FastLocalHttpClient, HttpRequestError, default_client
from .utils import (
    dump_json_file,
    fast_json_dumps_bytes,
    fast_json_loads,
    load_json_file,
    repair_json_string,
)


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

    validated_data: dict[str, Any]
    retranslate_data: dict[str, Any]
    processed_keys: set[str]


@dataclass
class ValidationWaveContext:
    """Bundles shared context parameters for validation wave execution."""

    config: dict[str, Any]
    session: FastLocalHttpClient
    lock: threading.RLock
    state: ValidationState


def _parse_validation_json(content: str) -> dict[str, Any]:
    """Extracts and parses JSON object from LLM response content."""
    if not content:
        return {}

    s = content.strip()
    start_idx = s.find("{")
    if start_idx != -1:
        end_idx = s.rfind("}")
        if end_idx > start_idx:
            try:
                res = fast_json_loads(s[start_idx : end_idx + 1])
                if isinstance(res, dict):
                    return res
            except (ValueError, TypeError, json.JSONDecodeError):
                pass

    try:
        repaired = repair_json_string(s)
        if repaired:
            res = fast_json_loads(repaired)
            if isinstance(res, dict):
                return res
    except (ValueError, TypeError, json.JSONDecodeError):
        pass

    return {}


def _send_validation_request(
    prompt: str, config: dict[str, Any], session: FastLocalHttpClient | None = None
) -> dict[str, Any]:
    """Sends validation payload to LLM and returns parsed mapping."""
    req_data = {
        "model": config["model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 1024,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.get('api_key', 'lm-studio')}",
    }
    requester = session or default_client
    resp = requester.post(
        config["api_endpoint"],
        headers=headers,
        data=fast_json_dumps_bytes(req_data),
        timeout=config["request_timeout"],
    )
    resp.raise_for_status()
    raw_content = resp.json()["choices"][0]["message"]["content"].strip()
    return _parse_validation_json(raw_content)


def call_batch_validation(
    batch: list[tuple[int, str, str]],
    config: dict[str, Any],
    session: FastLocalHttpClient | None = None,
) -> dict[str, bool]:
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

    results = {jp: True for _idx, jp, _en in batch}
    try:
        parsed = _send_validation_request(prompt, config, session)
        for idx, jp, _en in batch:
            val = parsed.get(str(idx), 1)
            results[jp] = bool(val == 1 or val is True or str(val).lower() == "true")
    except (HttpRequestError, ValueError, KeyError) as err:
        print(f"\nWarning: Batch validation request failed ({err}). Defaulting items to VALID (1).")

    return results


def save_progress(
    paths: ValidationPaths, state: ValidationState, lock: threading.RLock | None = None
) -> None:
    """Safely writes validation results (*_validated.json, *_retranslate.json) and checkpoints."""

    def _write_files() -> None:
        dump_json_file(paths.valid, state.validated_data, indent=True)
        dump_json_file(paths.retranslate, state.retranslate_data, indent=True)
        dump_json_file(
            paths.checkpoint,
            {"processed_keys": list(state.processed_keys)},
            indent=True,
        )
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
    config: dict[str, Any], input_file: str | None, output_dir: str | None
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
    validated_data: dict[str, Any] = {}
    retranslate_data: dict[str, Any] = {}
    processed_keys: set[str] = set()

    if paths.checkpoint.exists():
        try:
            chk = load_json_file(paths.checkpoint)
            if isinstance(chk, dict):
                processed_keys = set(chk.get("processed_keys", []))
            print(f"--> Found existing checkpoint: {len(processed_keys)} items already processed.")
        except (OSError, ValueError, TypeError, KeyError) as err:
            print(f"Warning: Could not read checkpoint file ({err}). Starting fresh.")

    for check_p, data_dict in (
        (paths.valid, validated_data),
        (paths.retranslate, retranslate_data),
    ):
        if check_p.exists():
            try:
                loaded = load_json_file(check_p)
                if isinstance(loaded, dict):
                    data_dict.update(loaded)
                    processed_keys.update(loaded.keys())
            except (OSError, ValueError, TypeError):
                pass

    return ValidationState(validated_data, retranslate_data, processed_keys)


def _process_validation_wave_with_executor(
    executor: ThreadPoolExecutor,
    current_wave: list[list[tuple[int, str, str]]],
    ctx: ValidationWaveContext,
) -> None:
    """Processes a single parallel wave of validation batches using persistent thread pool."""
    future_to_batch = {
        executor.submit(call_batch_validation, batch, ctx.config, ctx.session): batch
        for batch in current_wave
    }
    for future in as_completed(future_to_batch):
        batch = future_to_batch[future]
        results = future.result()
        with ctx.lock:
            for _idx, jp, en in batch:
                if results.get(jp, True):
                    ctx.state.validated_data[jp] = en
                else:
                    ctx.state.retranslate_data[jp] = jp
                ctx.state.processed_keys.add(jp)


def _run_validation_waves(
    batches: list[list[tuple[int, str, str]]],
    config: dict[str, Any],
    paths: ValidationPaths,
    state: ValidationState,
    lock: threading.RLock,
) -> None:
    """Runs batch waves and saves checkpoints periodically."""
    max_workers = config.get("max_workers", 4)
    save_interval = config.get("save_interval", 10)
    waves = [batches[i : i + max_workers] for i in range(0, len(batches), max_workers)]

    session = FastLocalHttpClient(max_connections=max_workers)
    ctx = ValidationWaveContext(config=config, session=session, lock=lock, state=state)
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for wave_idx, current_wave in enumerate(waves, 1):
                _process_validation_wave_with_executor(executor, current_wave, ctx)
                if wave_idx % save_interval == 0 or wave_idx == len(waves):
                    print(f"Wave {wave_idx}/{len(waves)} complete. Autosaving progress...")
                    save_progress(paths, state, lock)
    finally:
        session.close()


def process_validation(
    config_file: str = "config.json",
    input_file: str | None = None,
    output_dir: str | None = None,
) -> tuple[Path, Path]:
    """Audits translated JSON file, splitting passed items and items needing retranslation."""
    config = load_config(config_file, section="validation")
    paths = _init_validation_paths(config, input_file, output_dir)
    state = _load_validation_state(paths)

    data = load_json_file(paths.input_file)

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
