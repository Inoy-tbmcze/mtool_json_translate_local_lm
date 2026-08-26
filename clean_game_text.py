"""
Two-Stage Hybrid Cleaning Script for Game Localization JSON (V6 - Gemma 4 Optimized)
------------------------------------------------------------------------------------------
Stage 1:
  - Fast rule-based junk filtering (code, non-Japanese lines, file paths, comment syntax,
    low Japanese character ratio including custom Japanese punctuation/symbols).
Stage 2:
  - Multithreaded classification optimized for Gemma-4-e4b (single-turn user prompt layout).
"""

import json
import re
import sys
import threading
import requests
from pathlib import Path
from typing import Dict, Any, List, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

# Global default threshold for Japanese character ratio (80%)
DEFAULT_MIN_JAPANESE_RATIO = 0.8

# Hardcoded fallback symbols counted as Japanese if jp_symbols.json is not found
DEFAULT_JP_SYMBOLS = [
    "）", "」", "…", "（", "「", "『", "』", "【", "】",
    "・", "！", "？", "〜", "ー", "、", "。"
]

# Fantasy item & plant suffix pattern
FANTASY_ITEM_PATTERN = re.compile(
    r".*?(?:の花|の草|の薬|の種|の根|の芽|の果実|の石|の剣|の盾|の鎧|の指輪|の巻物|の鍵|の壺|の瓶|の尾|の角|の羽|の皮|の骨)$"
)


def is_protected_game_item(text: str) -> bool:
    """Returns True if string matches standard fantasy item/material naming patterns."""
    s = text.strip()
    if len(s) <= 20 and FANTASY_ITEM_PATTERN.match(s):
        return True
    return False


def load_japanese_symbols(symbols_filename: str = "jp_symbols.json") -> Set[str]:
    """Loads Japanese symbols from JSON file in the script's directory, falling back to defaults."""
    script_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()
    symbols_path = script_dir / symbols_filename

    if symbols_path.exists():
        try:
            with open(symbols_path, "r", encoding="utf-8") as f:
                symbols_list = json.load(f)
                if isinstance(symbols_list, list):
                    print(f"--> Loaded {len(symbols_list)} Japanese symbols from '{symbols_filename}'.")
                    return set(symbols_list)
        except Exception as e:
            print(f"Warning: Failed to parse '{symbols_filename}' ({e}). Using default symbols.")

    print("--> Using default hardcoded Japanese symbol set.")
    return set(DEFAULT_JP_SYMBOLS)


def build_japanese_regex(symbols: Set[str]) -> re.Pattern:
    """Builds regex matching Hiragana, Katakana, Kanji, and approved Japanese symbols."""
    escaped_symbols = "".join(re.escape(s) for s in symbols)
    return re.compile(rf"[\u3040-\u30ff\u4e00-\u9faf{escaped_symbols}]")


def load_config(config_file: str = "cleanup_config.json") -> Dict[str, Any]:
    """Loads configuration file and applies defaults matching your translation pipeline."""
    config_path = Path(__file__).parent / config_file if '__file__' in globals() else Path(config_file)
    if not config_path.exists():
        print(f"Error: Configuration file '{config_file}' not found.")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

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
    """Calculates the proportion of Japanese characters and custom symbols in a string."""
    if not text:
        return 0.0
    jp_char_count = len(jp_regex.findall(text))
    return jp_char_count / len(text)


def has_japanese_characters(text: str, jp_regex: re.Pattern) -> bool:
    """Checks if text contains Hiragana, Katakana, Kanji, or approved Japanese symbols."""
    return bool(jp_regex.search(text))


def is_ascii_art_or_symbol_heavy(text: str, jp_regex: re.Pattern) -> bool:
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


DEV_COMMENT_PATTERNS = [
    r"^\s*//", r"^\s*/\*", r"^\s*#", r"^\s*<!--",
    r"^\s*※",
    r"^\s*【(?:開発|仕様|デバッグ|テスト|メモ|TODO|FIXME|仮|作業用|消去予定|実装予定|処理|補足)】",
    r"^\s*(?:TODO|FIXME|DEBUG|HACK|BUG|NOTE|メモ|仮置き|未実装|要修正|後で修正|仕様|開発メモ)\s*[:：]",
]


def is_stage1_junk(
    text: str,
    jp_regex: re.Pattern,
    min_ratio: float = DEFAULT_MIN_JAPANESE_RATIO
) -> Tuple[bool, str]:
    """Returns (is_junk, reason) based on fast regex and Japanese language presence/ratio."""
    if not text or not text.strip():
        return True, "empty_string"

    s = text.strip()
    if not has_japanese_characters(s, jp_regex):
        return True, "non_japanese_text"

    # Enforce minimum Japanese character ratio threshold
    jp_ratio = calculate_japanese_ratio(s, jp_regex)
    if jp_ratio < min_ratio:
        return True, f"low_japanese_ratio ({jp_ratio:.1%} < {min_ratio:.1%})"

    # Check for obvious code comments
    for pattern in DEV_COMMENT_PATTERNS:
        if re.search(pattern, s, re.IGNORECASE):
            return True, "developer_comment"

    # File paths or code assets
    file_exts = (
        ".png", ".jpg", ".jpeg", ".wav", ".mp3", ".ogg", ".cpp", ".h", ".cs",
        ".json", ".xml", ".asset", ".mat", ".prefab"
    )
    if any(s.lower().endswith(ext) for ext in file_exts) or "/" in s or "\\" in s:
        return True, "filepath_or_asset"

    if is_ascii_art_or_symbol_heavy(s, jp_regex):
        return True, "ascii_art_or_symbol_heavy"

    return False, ""


def call_batch_classification(batch: List[Tuple[int, str, str]], config: Dict[str, Any]) -> Dict[str, bool]:
    """Sends batch to Gemma 4 via single user turn to eliminate LM Studio template format warnings."""
    prompt_items = [f"{idx}:{text}" for idx, key, text in batch]
    items_str = "\n".join(prompt_items)

    combined_prompt = (
        "Classify Japanese video game text strings into JSON mapping ID to 1 or 0.\n\n"
        "1 (KEEP / PLAYER-FACING):\n"
        "- Dialogue, quest text, UI labels, skills, items, plants, weapons (e.g., '毒消しの花', '薬草').\n\n"
        "0 (DISCARD / DEV JUNK):\n"
        "- Dev notes, specs, TODOs, comments, debug logs (e.g., '※仕様', '処理用', '実装予定').\n\n"
        "Output ONLY a raw JSON object with keys as string IDs.\n"
        'Example format: {"0":1, "1":0, "2":1}\n\n'
        f"Strings to classify:\n{items_str}"
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

    results = {}
    try:
        resp = requests.post(
            config["api_endpoint"],
            headers=headers,
            json=data,
            timeout=config["request_timeout"]
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()

        if "```" in content:
            content = re.sub(r"```(?:json)?|```", "", content).strip()

        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            content = json_match.group(0)

        parsed = json.loads(content)
        for idx, key, text in batch:
            val = parsed.get(str(idx), parsed.get(idx, 0))
            results[key] = bool(val == 1 or val is True or str(val).lower() == "true")
    except Exception as e:
        print(f"\nWarning: Batch classification request failed ({e}). Defaulting items to KEEP.")
        for idx, key, text in batch:
            results[key] = True

    return results


def save_progress(
    cleaned_path: Path,
    quarantine_path: Path,
    checkpoint_path: Path,
    cleaned_data: dict,
    quarantine_data: dict,
    lock: threading.RLock
):
    """Safely writes current results and progress checkpoints to disk using RLock."""
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

    # Load Japanese symbols and construct regex
    jp_symbols = load_japanese_symbols(symbols_filename)
    jp_regex = build_japanese_regex(jp_symbols)

    script_dir = Path(__file__).parent if "__file__" in globals() else Path.cwd()
    input_path = script_dir / input_filename

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

    print(f"\n--- Stage 1: Rule-Based Junk Filtering ({len(data)} total lines) ---")
    print(f"Minimum Japanese Character Ratio Threshold: {min_japanese_ratio:.1%}")
    stage1_junk_count = 0

    for key, text in data.items():
        if key in processed_keys:
            continue

        check_target = text if text else key
        is_junk, reason = is_stage1_junk(str(check_target), jp_regex, min_ratio=min_japanese_ratio)

        if is_junk:
            quarantine_data[key] = {"val": text, "stage": "Stage 1 (Rule)", "reason": reason}
            processed_keys.add(key)
            stage1_junk_count += 1
        else:
            stage2_candidates.append((key, text))

    print(f"Stage 1 Complete:")
    print(f" - {stage1_junk_count} code/non-Japanese/low-ratio/dev junk lines quarantined.")
    print(f" - {len(stage2_candidates)} Japanese strings sent to Stage 2 LLM for accuracy auditing.")

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