"""
Two-Stage Hybrid Cleaning Script for Game Localization JSON (V15 - Fixed Path and Config Logic)
-----------------------------------------------------------------------------------------------
Stage 1:
  - Fast rule-based junk filtering.
  - File Path and Voice Key Filtering: Quarantines audio files and paths before text checks.
  - Sentence Whitelist: Protects full sentences automatically.
  - Short UI Whitelist: Protects short Japanese UI labels and skill names automatically.
Stage 2:
  - Fast multithreaded classification for ambiguous strings.
"""

import json
import re
import sys
import threading
import requests
from pathlib import Path
from typing import Dict, Any, List, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

DEFAULT_MIN_JAPANESE_RATIO = 0.8

DEFAULT_JP_SYMBOLS = [
    "）", "」", "…", "（", "「", "『", "』", "【", "】",
    "・", "！", "？", "〜", "ー", "、", "。"
]

FILE_EXTENSIONS = (
    ".png", ".jpg", ".jpeg", ".bmp", ".tga", ".webp",
    ".wav", ".mp3", ".ogg", ".flac", ".webm", ".mp4", ".ogv", ".avi", ".bik", ".bk2",
    ".cpp", ".h", ".cs", ".py", ".json", ".xml", ".asset", ".mat", ".prefab", ".txt"
)

PURE_ASCII_IDENTIFIER_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\.\(\)\s]+$")

ENGINE_KEY_PATTERNS = [
    re.compile(r".*フレーム\s*\d+$", re.IGNORECASE),
    re.compile(r"^event\d+【\d+】$", re.IGNORECASE),
    re.compile(r"^[a-zA-Z0-9_]{4,}_[\u3040-\u30ff\u4e00-\u9faf]"),
    re.compile(r"^(?:event|pers|scene|cutscene)\d*_[0-9a-zA-Z_]+$", re.IGNORECASE),
    re.compile(r"^\d+_\d+_[\u3040-\u30ff\u4e00-\u9faf]", re.IGNORECASE),
    re.compile(r"^(?:sound|voice|snd|bgm|se)[\\/]", re.IGNORECASE),
]

FANTASY_ITEM_PATTERN = re.compile(
    r".*?(?:の花|の草|の薬|の種|の根|の芽|の果実|の石|の剣|の盾|の鎧|の指輪|の巻物|の鍵|の壺|の瓶|の尾|の角|の羽|の皮|の骨)$"
)

DEV_COMMENT_PATTERNS = [
    r"^\s*//", r"^\s*/\*", r"^\s*#", r"^\s*<!--",
    r"^\s*【(?:開発|仕様|デバッグ|テスト|メモ|TODO|FIXME|仮|作業用|消去予定|実装予定|処理|補足)】",
    r"^\s*(?:TODO|FIXME|DEBUG|HACK|BUG|NOTE|メモ|仮置き|未実装|要修正|後で修正|仕様|開発メモ)\s*[:：]",
]

JAPANESE_SENTENCE_PUNCTUATION = ("。", "！", "？", "…", "...", "」", "♪", "〜")

JP_CHAR_PATTERN = re.compile(r"[\u3040-\u30ff\u4e00-\u9faf]")


def is_protected_sentence(text: str) -> bool:
    """Returns True if text contains full Japanese sentence or dialogue punctuation."""
    s = text.strip()
    return any(symbol in s for symbol in JAPANESE_SENTENCE_PUNCTUATION)


def is_protected_short_ui_label(text: str) -> bool:
    """Returns True for short Japanese UI labels, skill names, and menu items."""
    s = text.strip()
    if 0 < len(s) <= 10 and JP_CHAR_PATTERN.search(s):
        code_symbols = ("/", "\\", "{", "}", "=", "<", ">", "//", "/*")
        if not any(sym in s for sym in code_symbols):
            return True
    return False


def is_protected_game_item(text: str) -> bool:
    """Returns True if string matches standard fantasy item naming patterns."""
    s = text.strip()
    if len(s) <= 20 and FANTASY_ITEM_PATTERN.match(s):
        return True
    return False


def load_japanese_symbols(symbols_filename: str = "jp_symbols.json") -> Set[str]:
    """Loads Japanese symbols from a JSON file."""
    script_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()
    symbols_path = Path(symbols_filename)
    if not symbols_path.is_absolute():
        symbols_path = script_dir / symbols_path

    if symbols_path.exists():
        try:
            with open(symbols_path, "r", encoding="utf-8") as f:
                symbols_list = json.load(f)
                if isinstance(symbols_list, list):
                    return set(symbols_list)
        except Exception:
            pass

    return set(DEFAULT_JP_SYMBOLS)


def build_japanese_regex(symbols: Set[str]) -> re.Pattern:
    """Builds regex matching Japanese characters and custom symbols."""
    escaped_symbols = "".join(re.escape(s) for s in symbols)
    return re.compile(rf"[\u3040-\u30ff\u4e00-\u9faf{escaped_symbols}]")


def load_config(config_file: str = "cleanup_config.json") -> Dict[str, Any]:
    """Loads settings from configuration file or uses defaults."""
    script_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()
    config_path = Path(config_file)
    if not config_path.is_absolute():
        config_path = script_dir / config_path

    config = {}
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to read '{config_file}' ({e}). Using default settings.")
    else:
        print(f"Notice: Config file '{config_file}' not found. Using default settings.")

    config.setdefault("request_timeout", 60)
    config.setdefault("batch_size", 30)
    config.setdefault("max_workers", 4)
    config.setdefault("save_interval", 10)
    config.setdefault("input_filename", "game_text.json")
    config.setdefault("api_endpoint", "http://127.0.0.1:1234/v1/chat/completions")
    config.setdefault("model", "gemma-4-e4b")
    config.setdefault("min_japanese_ratio", DEFAULT_MIN_JAPANESE_RATIO)
    config.setdefault("symbols_filename", "jp_symbols.json")

    return config


def calculate_japanese_ratio(text: str, jp_regex: re.Pattern) -> float:
    """Calculates the proportion of Japanese characters in a string."""
    if not text:
        return 0.0
    jp_char_count = len(jp_regex.findall(text))
    return jp_char_count / len(text)


def has_japanese_characters(text: str, jp_regex: re.Pattern) -> bool:
    """Checks if text contains Japanese characters."""
    return bool(jp_regex.search(text))


def is_ascii_art_or_symbol_heavy(text: str, jp_regex: re.Pattern) -> bool:
    """Detects ASCII art or symbol-heavy lines."""
    if not text:
        return False
    symbol_chars = set(r"=-_*+#/\|~<>[]{}()!@$%^&:`';")
    symbol_count = sum(1 for c in text if c in symbol_chars)
    ratio = symbol_count / len(text)
    if len(text) > 5 and ratio > 0.5:
        return True
    if re.search(r"(.)\1{4,}", text) and not has_japanese_characters(text, jp_regex):
        return True
    return False


def is_stage1_junk(
    key: str,
    text: str,
    jp_regex: re.Pattern,
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

    if any(k.lower().endswith(ext) for ext in FILE_EXTENSIONS) or any(s.lower().endswith(ext) for ext in FILE_EXTENSIONS):
        return True, "filepath_or_asset"

    if PURE_ASCII_IDENTIFIER_PATTERN.match(s):
        return True, "pure_ascii_identifier"

    for pattern in ENGINE_KEY_PATTERNS:
        if pattern.search(k) or pattern.search(s):
            return True, "game_engine_key_or_marker"

    if not has_japanese_characters(s, jp_regex):
        return True, "non_japanese_text"

    jp_ratio = calculate_japanese_ratio(s, jp_regex)
    if len(s) > 20 and jp_ratio < min_ratio:
        return True, f"low_japanese_ratio ({jp_ratio:.1%} < {min_ratio:.1%})"

    for pattern in DEV_COMMENT_PATTERNS:
        if re.search(pattern, s, re.IGNORECASE):
            return True, "developer_comment"

    if is_ascii_art_or_symbol_heavy(s, jp_regex):
        return True, "ascii_art_or_symbol_heavy"

    return False, ""


def parse_json_array_safely(content: str) -> list:
    """Parses JSON array and repairs truncated strings."""
    content = content.strip()
    if "```" in content:
        content = re.sub(r"```(?:json)?|```", "", content).strip()

    json_match = re.search(r"\[.*\]", content, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except Exception:
            pass

    start_idx = content.find("[")
    if start_idx != -1:
        truncated = content[start_idx:]
        last_comma = truncated.rfind(",")
        if last_comma != -1:
            repaired = truncated[:last_comma] + "]"
            try:
                return json.loads(repaired)
            except Exception:
                pass

    return []


def call_batch_classification(batch: List[Tuple[int, str, str]], config: Dict[str, Any]) -> Dict[str, bool]:
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

    try:
        resp = requests.post(
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
    checkpoint_path: Path,
    cleaned_data: dict,
    quarantine_data: dict,
    lock: threading.RLock
):
    """Saves progress to disk."""
    with lock:
        try:
            with open(cleaned_path, "w", encoding="utf-8") as f:
                json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
            with open(quarantine_path, "w", encoding="utf-8") as f:
                json.dump(quarantine_data, f, ensure_ascii=False, indent=2)

            checkpoint_keys = list(cleaned_data.keys()) + list(quarantine_data.keys())
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump({"processed_keys": checkpoint_keys}, f, ensure_ascii=False, indent=2)
            print(" -> Autosave successful.")
        except Exception as e:
            print(f" -> Error during autosave: {e}")


def process_json_file(config_file: str = "cleanup_config.json"):
    config = load_config(config_file)
    input_filename = config.get("input_filename", "game_text.json")
    min_japanese_ratio = config.get("min_japanese_ratio", DEFAULT_MIN_JAPANESE_RATIO)
    symbols_filename = config.get("symbols_filename", "jp_symbols.json")

    jp_symbols = load_japanese_symbols(symbols_filename)
    jp_regex = build_japanese_regex(jp_symbols)

    script_dir = Path(__file__).parent if "__file__" in globals() else Path.cwd()
    input_path = Path(input_filename)
    if not input_path.is_absolute():
        input_path = script_dir / input_path

    if not input_path.exists():
        print(f"Error: Input file '{input_filename}' not found at {input_path}.")
        sys.exit(1)

    stem = input_path.stem
    ext = input_path.suffix
    out_cleaned_path = input_path.parent / f"{stem}_cleaned{ext}"
    out_quarantine_path = input_path.parent / f"{stem}_quarantine{ext}"
    checkpoint_path = input_path.parent / f"{stem}_checkpoint.json"

    cleaned_data = {}
    quarantine_data = {}
    processed_keys = set()

    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                chk = json.load(f)
                processed_keys = set(chk.get("processed_keys", []))
            print(f"--> Found existing checkpoint: {len(processed_keys)} items already processed.")
        except Exception as e:
            print(f"Warning: Could not read checkpoint file ({e}). Starting fresh.")

    if out_cleaned_path.exists():
        try:
            with open(out_cleaned_path, "r", encoding="utf-8") as f:
                cleaned_data = json.load(f)
                processed_keys.update(cleaned_data.keys())
        except Exception:
            pass

    if out_quarantine_path.exists():
        try:
            with open(out_quarantine_path, "r", encoding="utf-8") as f:
                quarantine_data = json.load(f)
                processed_keys.update(quarantine_data.keys())
        except Exception:
            pass

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

        # 2. Protect Japanese sentences, dialogue, and short UI labels / skill names
        if is_protected_sentence(text_str) or is_protected_short_ui_label(text_str):
            cleaned_data[key] = text
            processed_keys.add(key)
            protected_count += 1
            continue

        stage2_candidates.append((key, text))

    print(f"Stage 1 Complete:")
    print(f" - {protected_count} sentences, skill names, and UI labels protected automatically.")
    print(f" - {stage1_junk_count} junk lines quarantined.")
    print(f" - {len(stage2_candidates)} ambiguous strings sent to Stage 2 LLM.")

    batch_size = config.get("batch_size", 30)
    max_workers = config.get("max_workers", 4)
    save_interval = config.get("save_interval", 10)

    if stage2_candidates:
        print(f"\n--- Stage 2: Multithreaded LLM Classification ({config['model']}) ---")

        batches = []
        for i in range(0, len(stage2_candidates), batch_size):
            chunk = stage2_candidates[i:i + batch_size]
            batch_data = [(idx, k, v) for idx, (k, v) in enumerate(chunk)]
            batches.append(batch_data)

        wave_size = max_workers
        waves = [batches[i : i + wave_size] for i in range(0, len(batches), wave_size)]

        print(
            f"Processing {len(batches)} batches across {len(waves)} synchronized waves (Wave size: {wave_size})...\n"
        )

        lock = threading.RLock()
        completed_batches = 0

        for wave_idx, current_wave in enumerate(waves, 1):
            with ThreadPoolExecutor(max_workers=len(current_wave)) as executor:
                future_to_batch = {
                    executor.submit(call_batch_classification, batch, config): batch
                    for batch in current_wave
                }

                for future in as_completed(future_to_batch):
                    batch = future_to_batch[future]
                    results = future.result()

                    with lock:
                        for idx, key, text in batch:
                            if is_protected_game_item(text):
                                is_user_facing = True
                            else:
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
                    out_cleaned_path, out_quarantine_path, checkpoint_path, cleaned_data, quarantine_data, lock
                )

    save_progress(
        out_cleaned_path, out_quarantine_path, checkpoint_path, cleaned_data, quarantine_data, threading.RLock()
    )

    print("\n=== Processing Complete ===")
    print(f"Total Original Lines: {len(data)}")
    print(f"Cleaned Lines Saved:  {len(cleaned_data)} -> {out_cleaned_path.name}")
    print(f"Quarantined Lines:    {len(quarantine_data)} -> {out_quarantine_path.name}")


if __name__ == "__main__":
    process_json_file()