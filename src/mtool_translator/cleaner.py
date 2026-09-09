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


def is_stage1_junk(
    key: str,
    text: str,
    jp_regex,
    min_ratio: float = DEFAULT_MIN_JAPANESE_RATIO
) -> Tuple[bool, str]:
    """Returns (is_junk, reason) based on fast regex rules."""
    s = text.strip() if text else ""
    k = key.strip() if key else ""

    if not s:
        return True, "empty_string"

    # Check key and text for paths or file extensions
    if "/" in k or "\\" in k or "/" in s or "\\" in s:
        return True, "filepath_or_asset"

    if k.lower().endswith(FILE_EXTENSIONS) or s.lower().endswith(FILE_EXTENSIONS):
        return True, "filepath_or_asset"

    if PURE_ASCII_IDENTIFIER_PATTERN.match(s):
        return True, "pure_ascii_identifier"

    if ENGINE_KEY_RE.search(k) or ENGINE_KEY_RE.search(s):
        return True, "game_engine_key_or_marker"

    if not has_japanese_characters(s, jp_regex):
        return True, "non_japanese_text"

    if len(s) > 20:
        jp_ratio = calculate_japanese_ratio(s, jp_regex)
        if jp_ratio < min_ratio:
            return True, f"low_japanese_ratio ({jp_ratio:.1%} < {min_ratio:.1%})"

    if DEV_COMMENT_RE.search(s):
        return True, "developer_comment"

    if is_ascii_art_or_symbol_heavy(s, jp_regex):
        return True, "ascii_art_or_symbol_heavy"

    return False, ""


def call_batch_classification(
    batch: List[Tuple[int, str, str]],
    config: Dict[str, Any],
    session: Optional[requests.Session] = None
) -> Dict[str, bool]:
    """Sends a batch to the LLM for classification."""
    prompt_items = [f"{idx}:{text}" for idx, key, text in batch]
    items_str = "\n".join(prompt_items)

    combined_prompt = (
        "Identify developer junk in game text.\n"
        "Return a JSON array of integer IDs to DISCARD (e.g., [0, 3]).\n"
        "Discard ONLY if 100% sure it is dev junk (TODOs, specs, debug logs, internal engine triggers).\n"
        "Keep ALL story text, dialogue, tutorial text, menu text, skill names, and user instructions.\n"
        "Return [] if no items are junk.\n\n"
        f"Items:\n{items_str}"
    )

    data = {
        "model": config["model"],
        "messages": [
            {"role": "user", "content": combined_prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 512
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.get('api_key', 'lm-studio')}"
    }

    results = {key: True for idx, key, text in batch}
    http_client = session if session is not None else requests

    try:
        resp = http_client.post(
            config["api_endpoint"],
            headers=headers,
            json=data,
            timeout=config["request_timeout"]
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()

        discard_ids = set(parse_json_array_safely(content))
        for idx, key, text in batch:
            if idx in discard_ids or str(idx) in discard_ids:
                results[key] = False
    except Exception as e:
        print(f"\nWarning: Batch classification request failed ({e}). Defaulting items to KEEP.")

    return results


def save_progress(
    cleaned_path: Path,
    quarantine_path: Path,
    cleaned_data: dict,
    quarantine_data: dict,
    lock: Optional[threading.RLock] = None
):
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
    except Exception as e:
        print(f" -> Error during autosave: {e}")


def process_json_file(
    config_file: str = "config.json",
    input_file: Optional[str] = None,
    output_dir: Optional[str] = None
) -> Tuple[Path, Path]:
    """Cleans a raw game localization JSON file based on config and heuristics."""
    config = load_config(config_file, section="cleanup")

    input_filename = input_file or config.get("input_filename", "ManualTransFile.json")
    min_japanese_ratio = config.get("min_japanese_ratio", DEFAULT_MIN_JAPANESE_RATIO)
    symbols_filename = config.get("symbols_filename", "jp_symbols.json")

    symbols_path = resolve_input_path(symbols_filename, default_subfolder="")
    jp_symbols = load_japanese_symbols(symbols_path)
    jp_regex = build_japanese_regex(jp_symbols)

    input_path = resolve_input_path(input_filename, default_subfolder="raw")

    if not input_path.exists():
        print(f"Error: Input file '{input_filename}' not found at {input_path}.")
        sys.exit(1)

    stem = input_path.stem
    ext = input_path.suffix

    if output_dir:
        out_dir = Path(output_dir)
        out_cleaned_path = out_dir / f"{stem}_cleaned{ext}"
        out_quarantine_path = out_dir / f"{stem}_quarantine{ext}"
    else:
        out_cleaned_path = resolve_output_path(f"{stem}_cleaned{ext}", default_subfolder="processed")
        out_quarantine_path = resolve_output_path(f"{stem}_quarantine{ext}", default_subfolder="processed")

    cleaned_data: Dict[str, Any] = {}
    quarantine_data: Dict[str, Any] = {}
    processed_keys: Set[str] = set()

    for check_p, data_dict in [(out_cleaned_path, cleaned_data), (out_quarantine_path, quarantine_data)]:
        if check_p.exists():
            try:
                with open(check_p, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    data_dict.update(loaded)
                    processed_keys.update(loaded.keys())
            except Exception:
                pass

    if processed_keys:
        print(f"--> Found existing progress: {len(processed_keys)} items already processed.")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    stage2_candidates = []

    print(f"\n--- Stage 1: Rule-Based Filtering ({len(data)} total lines) ---")
    stage1_junk_count = 0
    protected_count = 0

    for key, text in data.items():
        if key in processed_keys:
            continue

        check_target = text if text else key
        text_str = str(check_target)
        key_str = str(key)

        # 1. Filter engine keys, file paths, and developer junk first
        is_junk, reason = is_stage1_junk(key_str, text_str, jp_regex, min_ratio=min_japanese_ratio)

        if is_junk:
            quarantine_data[key] = {"val": text, "stage": "Stage 1 (Rule)", "reason": reason}
            processed_keys.add(key)
            stage1_junk_count += 1
            continue

        # 2. Protect Japanese sentences, dialogue, short UI labels / skill names, fantasy items, and Katakana vocabulary
        if (
            is_protected_sentence(text_str)
            or is_protected_short_ui_label(text_str)
            or is_protected_game_item(text_str)
            or is_protected_katakana_word(text_str)
        ):
            cleaned_data[key] = text
            processed_keys.add(key)
            protected_count += 1
            continue

        stage2_candidates.append((key, text))

    print("Stage 1 Complete:")
    print(f" - {protected_count} sentences, skill names, items, and UI labels protected automatically.")
    print(f" - {stage1_junk_count} junk lines quarantined.")
    print(f" - {len(stage2_candidates)} ambiguous strings sent to Stage 2 LLM.")

    batch_size = config.get("batch_size", 30)
    max_workers = config.get("max_workers", 4)
    save_interval = config.get("save_interval", 10)

    if stage2_candidates:
        print(f"\n--- Stage 2: Multithreaded LLM Classification ({config['model']}) ---")

        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=max_workers, pool_maxsize=max_workers)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        batches = []
        for i in range(0, len(stage2_candidates), batch_size):
            chunk = stage2_candidates[i:i + batch_size]
            batch_data = [(idx, k, v) for idx, (k, v) in enumerate(chunk)]
            batches.append(batch_data)

        wave_size = max_workers
        waves = [batches[i: i + wave_size] for i in range(0, len(batches), wave_size)]

        print(
            f"Processing {len(batches)} batches across {len(waves)} synchronized waves (Wave size: {wave_size})...\n"
        )

        lock = threading.RLock()
        completed_batches = 0

        try:
            for wave_idx, current_wave in enumerate(waves, 1):
                with ThreadPoolExecutor(max_workers=len(current_wave)) as executor:
                    future_to_batch = {
                        executor.submit(call_batch_classification, batch, config, session): batch
                        for batch in current_wave
                    }

                    for future in as_completed(future_to_batch):
                        batch = future_to_batch[future]
                        results = future.result()

                        with lock:
                            for idx, key, text in batch:
                                is_user_facing = results.get(key, True)

                                if is_user_facing:
                                    cleaned_data[key] = text
                                else:
                                    quarantine_data[key] = {
                                        "val": text,
                                        "stage": "Stage 2 (LLM)",
                                        "reason": f"Flagged as internal dev junk by {config['model']}"
                                    }
                                processed_keys.add(key)

                            completed_batches += 1

                if completed_batches % save_interval == 0 or completed_batches == len(batches):
                    print(f"Wave {wave_idx}/{len(waves)} complete. Autosaving progress...")
                    save_progress(
                        out_cleaned_path, out_quarantine_path, cleaned_data, quarantine_data, lock
                    )
        finally:
            session.close()

    save_progress(
        out_cleaned_path, out_quarantine_path, cleaned_data, quarantine_data
    )

    print("\n=== Processing Complete ===")
    print(f"Total Original Lines: {len(data)}")
    print(f"Cleaned Lines Saved:  {len(cleaned_data)} -> {out_cleaned_path}")
    print(f"Quarantined Lines:    {len(quarantine_data)} -> {out_quarantine_path}")

    return out_cleaned_path, out_quarantine_path
