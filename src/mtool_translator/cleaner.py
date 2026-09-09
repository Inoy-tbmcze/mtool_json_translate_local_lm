"""
Stage 1 & Stage 2 Text Cleaning Module for Game Localization JSON.
-------------------------------------------------------------------
Preprocesses raw game text dumps, preserves game elements and valid dialogue,
and quarantines developer junk and engine markers.
"""

from __future__ import annotations

import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Set
import requests

from .config import load_config, resolve_input_path, resolve_output_path
from .utils import (
    DEFAULT_MIN_JAPANESE_RATIO,
    FILE_EXTENSIONS,
    PURE_ASCII_IDENTIFIER_PATTERN,
    ENGINE_KEY_RE,
    DEV_COMMENT_RE,
    load_japanese_symbols,
    build_japanese_regex,
    calculate_japanese_ratio,
    has_japanese_characters,
    is_ascii_art_or_symbol_heavy,
    is_protected_sentence,
    is_protected_short_ui_label,
    is_protected_game_item,
    is_protected_katakana_word,
    parse_json_array_safely,
)


@dataclass
class CleanerPaths:
    """Encapsulates input and output file paths for text cleaning."""
    input_file: Path
    cleaned: Path
    quarantine: Path


@dataclass
class CleanerState:
    """Encapsulates processing state dictionaries and tracked keys."""
    cleaned_data: Dict[str, Any]
    quarantine_data: Dict[str, Any]
    processed_keys: Set[str]


def _is_filepath_or_engine_junk(key: str, text: str) -> Optional[str]:
    """Checks fast filepath, asset, identifier, or engine marker junk."""
    if not text:
        return "empty_string"
    if "/" in key or "\\" in key or "/" in text or "\\" in text:
        return "filepath_or_asset"
    if key.lower().endswith(FILE_EXTENSIONS) or text.lower().endswith(FILE_EXTENSIONS):
        return "filepath_or_asset"
    if PURE_ASCII_IDENTIFIER_PATTERN.match(text):
        return "pure_ascii_identifier"
    if ENGINE_KEY_RE.search(key) or ENGINE_KEY_RE.search(text):
        return "game_engine_key_or_marker"
    return None


def _is_content_junk(text: str, jp_regex, min_ratio: float) -> Optional[str]:
    """Checks character ratio, comment, or symbol junk."""
    if not has_japanese_characters(text, jp_regex):
        return "non_japanese_text"
    if len(text) > 20:
        jp_ratio = calculate_japanese_ratio(text, jp_regex)
        if jp_ratio < min_ratio:
            return f"low_japanese_ratio ({jp_ratio:.1%} < {min_ratio:.1%})"
    if DEV_COMMENT_RE.search(text):
        return "developer_comment"
    if is_ascii_art_or_symbol_heavy(text, jp_regex):
        return "ascii_art_or_symbol_heavy"
    return None


def is_stage1_junk(
    key: str,
    text: str,
    jp_regex,
    min_ratio: float = DEFAULT_MIN_JAPANESE_RATIO
) -> Tuple[bool, str]:
    """Returns (is_junk, reason) based on fast regex rules."""
    s = text.strip() if text else ""
    k = key.strip() if key else ""
    reason = _is_filepath_or_engine_junk(k, s) or _is_content_junk(s, jp_regex, min_ratio)
    if reason:
        return True, reason
    return False, ""


def call_batch_classification(
    batch: List[Tuple[int, str, str]],
    config: Dict[str, Any],
    session: Optional[requests.Session] = None
) -> Dict[str, bool]:
    """Sends a batch to the LLM for classification."""
    items_str = "\n".join(f"{idx}:{text}" for idx, _k, text in batch)
    combined_prompt = (
        "Identify developer junk in game text.\n"
        "Return a JSON array of integer IDs to DISCARD (e.g., [0, 3]).\n"
        "Discard ONLY if 100% sure it is dev junk "
        "(TODOs, specs, debug logs, internal engine triggers).\n"
        "Keep ALL story text, dialogue, tutorial text, menu text, "
        "skill names, and user instructions.\n"
        "Return [] if no items are junk.\n\n"
        f"Items:\n{items_str}"
    )

    data = {
        "model": config["model"],
        "messages": [{"role": "user", "content": combined_prompt}],
        "temperature": 0.0,
        "max_tokens": 512
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.get('api_key', 'lm-studio')}"
    }

    results = {key: True for _idx, key, _text in batch}

    try:
        resp = (session or requests).post(
            config["api_endpoint"],
            headers=headers,
            json=data,
            timeout=config["request_timeout"]
        )
        resp.raise_for_status()
        raw_msg = resp.json()["choices"][0]["message"]["content"].strip()
        discard_ids = set(parse_json_array_safely(raw_msg))
        for idx, key, _text in batch:
            if idx in discard_ids or str(idx) in discard_ids:
                results[key] = False
    except (requests.RequestException, json.JSONDecodeError, KeyError, ValueError) as err:
        print(f"\nWarning: Batch classification request failed ({err}). Defaulting items to KEEP.")

    return results


def save_progress(
    cleaned_path: Path,
    quarantine_path: Path,
    cleaned_data: dict,
    quarantine_data: dict,
    lock: Optional[threading.RLock] = None
) -> None:
    """Saves progress to disk."""
    if lock is not None:
        with lock:
            cleaned_snap = cleaned_data.copy()
            quarantine_snap = quarantine_data.copy()
    else:
        cleaned_snap = cleaned_data.copy()
        quarantine_snap = quarantine_data.copy()

    try:
        cleaned_path.parent.mkdir(parents=True, exist_ok=True)
        quarantine_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cleaned_path, "w", encoding="utf-8") as f:
            json.dump(cleaned_snap, f, ensure_ascii=False, indent=2)
        with open(quarantine_path, "w", encoding="utf-8") as f:
            json.dump(quarantine_snap, f, ensure_ascii=False, indent=2)
        print(" -> Autosave successful.")
    except OSError as err:
        print(f" -> Error during autosave: {err}")


def _init_cleaner_environment(
    config: Dict[str, Any],
    input_file: Optional[str],
    output_dir: Optional[str]
) -> Tuple[CleanerPaths, Any, float]:
    """Initializes paths, regexes, and ratio thresholds."""
    input_filename = input_file or config.get("input_filename", "ManualTransFile.json")
    min_ratio = config.get("min_japanese_ratio", DEFAULT_MIN_JAPANESE_RATIO)
    symbols_filename = config.get("symbols_filename", "jp_symbols.json")

    symbols_path = resolve_input_path(symbols_filename, default_subfolder="")
    jp_symbols = load_japanese_symbols(symbols_path)
    jp_regex = build_japanese_regex(jp_symbols)

    input_path = resolve_input_path(input_filename, default_subfolder="raw")
    if not input_path.exists():
        print(f"Error: Input file '{input_filename}' not found at {input_path}.")
        sys.exit(1)

    stem, ext = input_path.stem, input_path.suffix
    if output_dir:
        out_dir = Path(output_dir)
        cleaned_path = out_dir / f"{stem}_cleaned{ext}"
        quarantine_path = out_dir / f"{stem}_quarantine{ext}"
    else:
        cleaned_path = resolve_output_path(
            f"{stem}_cleaned{ext}", default_subfolder="processed"
        )
        quarantine_path = resolve_output_path(
            f"{stem}_quarantine{ext}", default_subfolder="processed"
        )

    return CleanerPaths(input_path, cleaned_path, quarantine_path), jp_regex, min_ratio


def _load_existing_progress(paths: CleanerPaths) -> CleanerState:
    """Loads existing progress from cleaned and quarantined files if present."""
    cleaned_data: Dict[str, Any] = {}
    quarantine_data: Dict[str, Any] = {}
    processed_keys: Set[str] = set()

    for check_p, data_dict in [
        (paths.cleaned, cleaned_data),
        (paths.quarantine, quarantine_data)
    ]:
        if check_p.exists():
            try:
                with open(check_p, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    data_dict.update(loaded)
                    processed_keys.update(loaded.keys())
            except (json.JSONDecodeError, OSError):
                pass

    if processed_keys:
        print(f"--> Found existing progress: {len(processed_keys)} items already processed.")

    return CleanerState(cleaned_data, quarantine_data, processed_keys)


def _run_stage1_filter(
    data: Dict[str, Any],
    state: CleanerState,
    jp_regex,
    min_ratio: float
) -> List[Tuple[str, Any]]:
    """Filters lines using Stage 1 heuristic rules and protected patterns."""
    stage2_candidates = []
    print(f"\n--- Stage 1: Rule-Based Filtering ({len(data)} total lines) ---")
    stage1_junk_count = 0
    protected_count = 0

    for key, text in data.items():
        if key in state.processed_keys:
            continue

        check_target = text if text else key
        text_str = str(check_target)
        key_str = str(key)

        is_junk, reason = is_stage1_junk(key_str, text_str, jp_regex, min_ratio=min_ratio)
        if is_junk:
            state.quarantine_data[key] = {
                "val": text, "stage": "Stage 1 (Rule)", "reason": reason
            }
            state.processed_keys.add(key)
            stage1_junk_count += 1
            continue

        if (
            is_protected_sentence(text_str)
            or is_protected_short_ui_label(text_str)
            or is_protected_game_item(text_str)
            or is_protected_katakana_word(text_str)
        ):
            state.cleaned_data[key] = text
            state.processed_keys.add(key)
            protected_count += 1
            continue

        stage2_candidates.append((key, text))

    print("Stage 1 Complete:")
    print(
        f" - {protected_count} sentences, skill names, items, "
        "and UI labels protected automatically."
    )
    print(f" - {stage1_junk_count} junk lines quarantined.")
    print(f" - {len(stage2_candidates)} ambiguous strings sent to Stage 2 LLM.")
    return stage2_candidates


def _process_wave(
    current_wave: List[List[Tuple[int, str, str]]],
    config: Dict[str, Any],
    session: requests.Session,
    lock: threading.RLock,
    state: CleanerState
) -> None:
    """Processes a single wave of classification batches in parallel."""
    with ThreadPoolExecutor(max_workers=len(current_wave)) as executor:
        future_to_batch = {
            executor.submit(call_batch_classification, batch, config, session): batch
            for batch in current_wave
        }
        for future in as_completed(future_to_batch):
            batch = future_to_batch[future]
            results = future.result()
            with lock:
                for _idx, key, text in batch:
                    if results.get(key, True):
                        state.cleaned_data[key] = text
                    else:
                        state.quarantine_data[key] = {
                            "val": text,
                            "stage": "Stage 2 (LLM)",
                            "reason": f"Flagged as internal dev junk by {config['model']}"
                        }
                    state.processed_keys.add(key)


def _execute_stage2_waves(
    batches: List[List[Tuple[int, str, str]]],
    config: Dict[str, Any],
    paths: CleanerPaths,
    state: CleanerState
) -> None:
    """Orchestrates Stage 2 synchronized batch waves and autosaving."""
    max_workers = config.get("max_workers", 4)
    save_interval = config.get("save_interval", 10)
    waves = [batches[i: i + max_workers] for i in range(0, len(batches), max_workers)]
    print(
        f"Processing {len(batches)} batches across {len(waves)} "
        f"synchronized waves (Wave size: {max_workers})...\n"
    )

    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=max_workers, pool_maxsize=max_workers
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    lock = threading.RLock()
    try:
        for wave_idx, current_wave in enumerate(waves, 1):
            _process_wave(current_wave, config, session, lock, state)
            if wave_idx % save_interval == 0 or wave_idx == len(waves):
                print(f"Wave {wave_idx}/{len(waves)} complete. Autosaving progress...")
                save_progress(
                    paths.cleaned, paths.quarantine,
                    state.cleaned_data, state.quarantine_data, lock
                )
    finally:
        session.close()


def process_json_file(
    config_file: str = "config.json",
    input_file: Optional[str] = None,
    output_dir: Optional[str] = None
) -> Tuple[Path, Path]:
    """Cleans a raw game localization JSON file based on config and heuristics."""
    config = load_config(config_file, section="cleanup")
    paths, jp_regex, min_japanese_ratio = _init_cleaner_environment(
        config, input_file, output_dir
    )
    state = _load_existing_progress(paths)

    with open(paths.input_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    stage2_candidates = _run_stage1_filter(
        raw_data, state, jp_regex, min_japanese_ratio
    )

    if stage2_candidates:
        print(f"\n--- Stage 2: Multithreaded LLM Classification ({config['model']}) ---")
        batch_size = config.get("batch_size", 30)
        batches = [
            [(idx, k, v) for idx, (k, v) in enumerate(stage2_candidates[i:i + batch_size])]
            for i in range(0, len(stage2_candidates), batch_size)
        ]
        _execute_stage2_waves(batches, config, paths, state)

    save_progress(
        paths.cleaned, paths.quarantine, state.cleaned_data, state.quarantine_data
    )

    print("\n=== Processing Complete ===")
    print(f"Total Original Lines: {len(raw_data)}")
    print(f"Cleaned Lines Saved:  {len(state.cleaned_data)} -> {paths.cleaned}")
    print(f"Quarantined Lines:    {len(state.quarantine_data)} -> {paths.quarantine}")

    return paths.cleaned, paths.quarantine
