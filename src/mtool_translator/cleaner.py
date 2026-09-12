"""Stage 1 & Stage 2 Text Cleaning Module for Game Localization JSON.

Preprocesses raw game text dumps, preserves game elements and valid dialogue,
and quarantines developer junk and engine markers.
"""

from __future__ import annotations

import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import load_config, resolve_input_path, resolve_output_path
from .http_client import FastLocalHttpClient, HttpRequestError, default_client
from .native_core import fast_is_ascii_identifier
from .utils import (
    _SENTENCE_PUNCT_RE,
    DEFAULT_MIN_JAPANESE_RATIO,
    DEV_COMMENT_RE,
    DEV_COMMENT_STARTERS,
    ENGINE_KEY_RE,
    FILE_EXTENSIONS_SET,
    JP_CHAR_PATTERN,
    build_japanese_regex,
    calculate_japanese_ratio,
    dump_json_file,
    fast_json_dumps_bytes,
    has_japanese_characters,
    is_ascii_art_or_symbol_heavy,
    is_protected_game_item,
    is_protected_katakana_word,
    is_protected_sentence,
    is_protected_short_ui_label,
    load_japanese_symbols,
    load_json_file,
    parse_json_array_safely,
    strip_engine_escape_codes,
)

COMMENT_PREFIXES: tuple[str, ...] = ("//", "/*", "#", "<!--")


@dataclass
class CleanerPaths:
    """Encapsulates input and output file paths for text cleaning."""

    input_file: Path
    cleaned: Path
    quarantine: Path


@dataclass
class CleanerState:
    """Encapsulates processing state dictionaries and tracked keys."""

    cleaned_data: dict[str, Any]
    quarantine_data: dict[str, Any]
    processed_keys: set[str]
    dirty: bool = False


@dataclass
class CleanerWaveContext:
    """Bundles shared context parameters for wave execution."""

    config: dict[str, Any]
    session: FastLocalHttpClient
    lock: threading.RLock
    state: CleanerState


def _has_engine_markers(key: str, text: str) -> bool:
    """Checks key and text for engine trigger markers."""
    if ("_" in key or "フレーム" in key or (key and key[0] in "eE")) and ENGINE_KEY_RE.search(key):
        return True
    return bool(
        ("_" in text or "フレーム" in text or (text and text[0] in "eE"))
        and ENGINE_KEY_RE.search(text)
    )


def _is_dev_comment(text: str) -> bool:
    """Evaluates whether text represents a developer comment with O(1) starter pruning."""
    return bool(text and text[0] in DEV_COMMENT_STARTERS and DEV_COMMENT_RE.match(text))


def _is_filepath_candidate(key: str, text: str) -> bool:
    """Evaluates whether key or text represents a file or asset path."""
    key_dot = key.rfind(".")
    if key_dot != -1 and key[key_dot:].lower() in FILE_EXTENSIONS_SET:
        return True
    text_dot = text.rfind(".")
    if text_dot != -1 and text[text_dot:].lower() in FILE_EXTENSIONS_SET:
        return True

    if "/" in key or "\\" in key or "/" in text or "\\" in text:
        clean_text = strip_engine_escape_codes(text) if "\\" in text else text
        clean_key = strip_engine_escape_codes(key) if "\\" in key else key

        # Exclude developer comment starters from being treated as path separators
        text_has_path = (
            "/" in clean_text or "\\" in clean_text
        ) and not clean_text.startswith(COMMENT_PREFIXES)
        key_has_path = (
            "/" in clean_key or "\\" in clean_key
        ) and not clean_key.startswith(COMMENT_PREFIXES)

        if text_has_path or key_has_path:
            is_dialogue = bool(JP_CHAR_PATTERN.search(clean_text)) or bool(
                _SENTENCE_PUNCT_RE.search(clean_text)
            )
            key_is_dialogue = bool(JP_CHAR_PATTERN.search(clean_key)) or bool(
                _SENTENCE_PUNCT_RE.search(clean_key)
            )
            if text_has_path and not is_dialogue:
                return True
            if key_has_path and not key_is_dialogue:
                return True

    return False


def _is_filepath_or_engine_junk(key: str, text: str) -> str | None:
    """Checks fast filepath, asset, identifier, or engine marker junk."""
    if not text:
        return "empty_string"

    if _is_filepath_candidate(key, text):
        return "filepath_or_asset"

    # Fast ASCII check via native machine code kernel
    if text.isascii() and fast_is_ascii_identifier(text):
        return "pure_ascii_identifier"

    if _has_engine_markers(key, text):
        return "game_engine_key_or_marker"

    return None


def _is_content_junk(text: str, jp_regex: Any, min_ratio: float) -> str | None:
    """Checks character ratio or symbol junk."""
    clean_text = strip_engine_escape_codes(text) if "\\" in text else text
    if not has_japanese_characters(clean_text, jp_regex):
        return "non_japanese_text"

    if is_ascii_art_or_symbol_heavy(clean_text, jp_regex):
        return "ascii_art_or_symbol_heavy"

    # Semantic Protection: dialogue and valid game strings must not be
    # quarantined by alphanumeric/formatting ratio cutoffs.
    if is_protected_sentence(clean_text) or is_protected_game_item(clean_text):
        return None

    n = len(clean_text)
    if n > 20:
        jp_ratio = calculate_japanese_ratio(clean_text, jp_regex)
        if jp_ratio < min_ratio:
            return f"low_japanese_ratio ({jp_ratio:.1%} < {min_ratio:.1%})"

    return None


def is_stage1_junk(
    key: str, text: str, jp_regex: Any, min_ratio: float = DEFAULT_MIN_JAPANESE_RATIO
) -> tuple[bool, str]:
    """Returns (is_junk, reason) based on fast heuristic rules."""
    s = text.strip() if text else ""
    k = key.strip() if key else ""

    if not s:
        return True, "empty_string"

    # Fast developer comment detection before generic slash or path evaluation
    if _is_dev_comment(s) or _is_dev_comment(k):
        return True, "developer_comment"

    reason = _is_filepath_or_engine_junk(k, s) or _is_content_junk(s, jp_regex, min_ratio)
    if reason:
        return True, reason
    return False, ""


def _send_classification_request(
    prompt: str, config: dict[str, Any], session: FastLocalHttpClient | None = None
) -> set[Any]:
    """Sends the classification request to the LLM and extracts discarded IDs."""
    req_body = {
        "model": config["model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 512,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.get('api_key', 'lm-studio')}",
    }
    requester = session or default_client
    resp = requester.post(
        config["api_endpoint"],
        headers=headers,
        data=fast_json_dumps_bytes(req_body),
        timeout=config["request_timeout"],
    )
    resp.raise_for_status()
    raw_msg = resp.json()["choices"][0]["message"]["content"].strip()
    return set(parse_json_array_safely(raw_msg))


def call_batch_classification(
    batch: list[tuple[int, str, str]],
    config: dict[str, Any],
    session: FastLocalHttpClient | None = None,
) -> dict[str, bool]:
    """Sends a batch to the LLM for classification."""
    items_str = "\n".join(f"{idx}:{text}" for idx, _k, text in batch)
    prompt = (
        "Identify developer junk in game text.\n"
        "Return a JSON array of integer IDs to DISCARD (e.g., [0, 3]).\n"
        "Discard ONLY if 100% sure it is dev junk "
        "(TODOs, specs, debug logs, internal engine triggers).\n"
        "Keep ALL story text, dialogue, tutorial text, menu text, "
        "skill names, and user instructions.\n"
        "Return [] if no items are junk.\n\n"
        f"Items:\n{items_str}"
    )

    results = {key: True for _idx, key, _text in batch}
    try:
        discard_ids = _send_classification_request(prompt, config, session)
        for idx, key, _text in batch:
            if idx in discard_ids or str(idx) in discard_ids:
                results[key] = False
    except (HttpRequestError, ValueError, KeyError) as err:
        print(f"\nWarning: Batch classification request failed ({err}). Defaulting items to KEEP.")

    return results


def save_progress(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    cleaned_path: Path,
    quarantine_path: Path,
    cleaned_data: dict,
    quarantine_data: dict,
    lock: threading.RLock | None = None,
    state: CleanerState | None = None,
) -> None:
    """Saves progress to disk using high-speed binary serialization."""
    if state is not None and not state.dirty:
        return

    def _write_files() -> None:
        dump_json_file(cleaned_path, cleaned_data, indent=True)
        dump_json_file(quarantine_path, quarantine_data, indent=True)
        if state is not None:
            state.dirty = False
        print(" -> Autosave successful.")

    try:
        if lock is not None:
            with lock:
                _write_files()
        else:
            _write_files()
    except OSError as err:
        print(f" -> Error during autosave: {err}")


def _init_cleaner_environment(
    config: dict[str, Any], input_file: str | None, output_dir: str | None
) -> tuple[CleanerPaths, Any, float]:
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
        cleaned_path = resolve_output_path(f"{stem}_cleaned{ext}", default_subfolder="processed")
        quarantine_path = resolve_output_path(
            f"{stem}_quarantine{ext}", default_subfolder="processed"
        )

    return CleanerPaths(input_path, cleaned_path, quarantine_path), jp_regex, min_ratio


def _load_existing_progress(paths: CleanerPaths) -> CleanerState:
    """Loads existing progress from cleaned and quarantined files if present."""
    cleaned_data: dict[str, Any] = {}
    quarantine_data: dict[str, Any] = {}
    processed_keys: set[str] = set()

    for check_p, data_dict in (
        (paths.cleaned, cleaned_data),
        (paths.quarantine, quarantine_data),
    ):
        if check_p.exists():
            try:
                loaded = load_json_file(check_p)
                if isinstance(loaded, dict):
                    data_dict.update(loaded)
                    processed_keys.update(loaded.keys())
            except (OSError, ValueError, TypeError):
                pass

    if processed_keys:
        print(f"--> Found existing progress: {len(processed_keys)} items already processed.")

    return CleanerState(cleaned_data, quarantine_data, processed_keys, dirty=False)


def _is_protected_entry(text: str, stripped: str) -> bool:
    """Checks whether original or stripped text qualifies as protected game text."""
    if is_protected_sentence(text):
        return True
    if stripped is not text and is_protected_sentence(stripped):
        return True
    return (
        is_protected_short_ui_label(stripped)
        or is_protected_game_item(stripped)
        or is_protected_katakana_word(stripped)
    )


def _run_stage1_filter(
    data: dict[str, Any], state: CleanerState, jp_regex: Any, min_ratio: float
) -> list[tuple[str, Any]]:
    """Filters lines using Stage 1 heuristic rules and protected patterns."""
    stage2_candidates = []
    print(f"\n--- Stage 1: Rule-Based Filtering ({len(data)} total lines) ---")
    stage1_junk_count = 0
    protected_count = 0

    for key, text in data.items():
        if key in state.processed_keys:
            continue

        check_target = text if text else key
        text_str = text if isinstance(check_target, str) else str(check_target)
        key_str = key if isinstance(key, str) else str(key)

        is_junk, reason = is_stage1_junk(key_str, text_str, jp_regex, min_ratio=min_ratio)
        if is_junk:
            state.quarantine_data[key] = {
                "val": text,
                "stage": "Stage 1 (Rule)",
                "reason": reason,
            }
            state.processed_keys.add(key)
            stage1_junk_count += 1
            continue

        eval_text = strip_engine_escape_codes(text_str) if "\\" in text_str else text_str
        if _is_protected_entry(text_str, eval_text):
            state.cleaned_data[key] = text
            state.processed_keys.add(key)
            protected_count += 1
            continue

        stage2_candidates.append((key, text))

    print("Stage 1 Complete:")
    print(
        f" - {protected_count} sentences, skill names, "
        "items, and UI labels protected automatically."
    )
    print(f" - {stage1_junk_count} junk lines quarantined.")
    print(f" - {len(stage2_candidates)} ambiguous strings sent to Stage 2 LLM.")
    if stage1_junk_count > 0 or protected_count > 0:
        state.dirty = True
    return stage2_candidates


def _process_wave_with_executor(
    executor: ThreadPoolExecutor,
    current_wave: list[list[tuple[int, str, str]]],
    ctx: CleanerWaveContext,
) -> None:
    """Processes a single wave of classification batches using the persistent thread pool."""
    future_to_batch = {
        executor.submit(call_batch_classification, batch, ctx.config, ctx.session): batch
        for batch in current_wave
    }
    for future in as_completed(future_to_batch):
        batch = future_to_batch[future]
        results = future.result()
        with ctx.lock:
            for _idx, key, text in batch:
                if results.get(key, True):
                    ctx.state.cleaned_data[key] = text
                else:
                    ctx.state.quarantine_data[key] = {
                        "val": text,
                        "stage": "Stage 2 (LLM)",
                        "reason": f"Flagged as internal dev junk by {ctx.config['model']}",
                    }
                ctx.state.processed_keys.add(key)
            if batch:
                ctx.state.dirty = True


def _execute_stage2_waves(
    batches: list[list[tuple[int, str, str]]],
    config: dict[str, Any],
    paths: CleanerPaths,
    state: CleanerState,
) -> None:
    """Orchestrates Stage 2 synchronized batch waves and autosaves."""
    max_workers = config.get("max_workers", 4)
    save_interval = config.get("save_interval", 10)
    waves = [batches[i : i + max_workers] for i in range(0, len(batches), max_workers)]
    print(
        f"Processing {len(batches)} batches across {len(waves)} "
        f"synchronized waves (Wave size: {max_workers})...\n"
    )

    session = FastLocalHttpClient(max_connections=max_workers)
    lock = threading.RLock()
    ctx = CleanerWaveContext(config=config, session=session, lock=lock, state=state)
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for wave_idx, current_wave in enumerate(waves, 1):
                _process_wave_with_executor(executor, current_wave, ctx)
                if wave_idx % save_interval == 0 and wave_idx < len(waves):
                    print(f"Wave {wave_idx}/{len(waves)} complete. Autosaving progress...")
                    save_progress(
                        paths.cleaned,
                        paths.quarantine,
                        state.cleaned_data,
                        state.quarantine_data,
                        lock,
                        state=state,
                    )
    finally:
        session.close()


def process_json_file(
    config_file: str = "config.json",
    input_file: str | None = None,
    output_dir: str | None = None,
) -> tuple[Path, Path]:
    """Cleans a raw game localization JSON file based on config and heuristics."""
    config = load_config(config_file, section="cleanup")
    paths, jp_regex, min_japanese_ratio = _init_cleaner_environment(config, input_file, output_dir)
    state = _load_existing_progress(paths)

    raw_data = load_json_file(paths.input_file)

    stage2_candidates = _run_stage1_filter(raw_data, state, jp_regex, min_japanese_ratio)

    if stage2_candidates:
        print(f"\n--- Stage 2: Multithreaded LLM Classification ({config['model']}) ---")
        batch_size = config.get("batch_size", 30)
        batches = [
            [(idx, k, v) for idx, (k, v) in enumerate(stage2_candidates[i : i + batch_size])]
            for i in range(0, len(stage2_candidates), batch_size)
        ]
        _execute_stage2_waves(batches, config, paths, state)

    save_progress(
        paths.cleaned,
        paths.quarantine,
        state.cleaned_data,
        state.quarantine_data,
        state=state,
    )

    print("\n=== Processing Complete ===")
    print(f"Total Original Lines: {len(raw_data)}")
    print(f"Cleaned Lines Saved:  {len(state.cleaned_data)} -> {paths.cleaned}")
    print(f"Quarantined Lines:    {len(state.quarantine_data)} -> {paths.quarantine}")

    return paths.cleaned, paths.quarantine
