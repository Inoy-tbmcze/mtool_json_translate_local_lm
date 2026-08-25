"""
Two-Stage Hybrid Cleaning Script for Game Localization JSON (V5 - High Accuracy)
------------------------------------------------------------------------------------------
Stage 1:
  - Strict Rule-Based Junk Filtering (quarantines code, non-Japanese lines, file paths, comment syntax).
  - NO blind auto-keeping: All Japanese lines are sent to Stage 2 LLM to distinguish 
    player-facing text from developer specs/notes/comments.
Stage 2:
  - Multithreaded LLM classification focused on distinguishing true game content vs. internal dev text.
"""

import json
import re
import sys
import threading
import requests
from pathlib import Path
from typing import Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def load_config(config_file: str = "config.json") -> Dict[str, Any]:
    """Loads configuration file and applies defaults matching your translation pipeline."""
    config_path = Path(__file__).parent / config_file if '__file__' in globals() else Path(config_file)
    if not config_path.exists():
        print(f"Error: Configuration file '{config_file}' not found.")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    config.setdefault("request_timeout", 60)
    config.setdefault("batch_size", 30)
    config.setdefault("max_workers", 4)  # Lowered to 4 to prevent LM Studio LRU slot thrashing
    config.setdefault("save_interval", 10)
    config.setdefault("input_filename", "game_text.json")
    config.setdefault("api_endpoint", "http://127.0.0.1:1234/v1/chat/completions")
    config.setdefault("model", "qwen3-4b-instruct")

    return config


def has_japanese_characters(text: str) -> bool:
    """Checks if text contains Hiragana, Katakana, or Kanji characters."""
    return bool(re.search(r"[\u3040-\u30ff\u4e00-\u9faf]", text))


def is_ascii_art_or_symbol_heavy(text: str) -> bool:
    if not text:
        return False
    symbol_chars = set(r"=-_*+#/\|~<>[]{}()!@$%^&:`';")
    symbol_count = sum(1 for c in text if c in symbol_chars)
    ratio = symbol_count / len(text)
    if len(text) > 5 and ratio > 0.5:
        return True
    if re.search(r"(.)\1{4,}", text) and not has_japanese_characters(text):
        return True
    return False


DEV_COMMENT_PATTERNS = [
    r"^\s*//", r"^\s*/\*", r"^\s*#", r"^\s*<!--",
    r"^\s*※",
    r"^\s*【(?:開発|仕様|デバッグ|テスト|メモ|TODO|FIXME|仮|作業用|消去予定|実装予定|処理|補足)】",
    r"^\s*(?:TODO|FIXME|DEBUG|HACK|BUG|NOTE|メモ|仮置き|未実装|要修正|後で修正|仕様|開発メモ)\s*[:：]",
]


def is_stage1_junk(text: str) -> Tuple[bool, str]:
    """Returns (is_junk, reason) based on fast regex and Japanese language presence."""
    if not text or not text.strip():
        return True, "empty_string"

    s = text.strip()
    if not has_japanese_characters(s):
        return True, "non_japanese_text"

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

    if is_ascii_art_or_symbol_heavy(s):
        return True, "ascii_art_or_symbol_heavy"

    return False, ""


def call_qwen_batch_classification(batch: List[Tuple[int, str, str]], config: Dict[str, Any]) -> Dict[str, bool]:
    """Sends batch to LLM to accurately separate player-facing game text from dev notes."""
    prompt_items = [f"{idx}:{text}" for idx, key, text in batch]
    items_str = "\n".join(prompt_items)

    system_prompt = (
        "You are a video game localization auditor classifying Japanese text strings.\n\n"
        "Classify as 1 (KEEP / PLAYER-FACING):\n"
        "- Items, consumables, crafting materials, plants, herbs (e.g., '毒消しの花', '薬草', '鉄の剣').\n"
        "- In-game dialogue, story lines, NPC conversations.\n"
        "- Quest text, item names, item descriptions, equipment stats.\n"
        "- Skill names, spell descriptions, status effects.\n"
        "- UI labels (e.g., '男', '女', '装備', '属性', '決定', 'キャンセル').\n"
        "- Location, map, dungeon, or level names.\n\n"
        "Classify as 0 (DISCARD / INTERNAL DEV JUNK):\n"
        "- Developer notes, specifications, design comments (e.g., '※〜の仕様', '処理用', '実装予定').\n"
        "- Internal tool instructions, bug reports, design memos, variable explanations.\n"
        "- Technical system debug logs not intended for players.\n\n"
        "IMPORTANT: If a string is a short fantasy item name, plant, or object name, ALWAYS output 1.\n\n"
        "Output strictly a JSON object mapping each numeric ID to 1 or 0.\n"
        'Example: {"0":1, "1":0, "2":1}'
    )

    data = {
        "model": config["model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Classify these strings:\n{items_str}"}
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


def process_json_file(config_file: str = "config.json"):
    config = load_config(config_file)
    input_filename = config.get("input_filename", "game_text.json")

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
    stage1_junk_count = 0

    for key, text in data.items():
        if key in processed_keys:
            continue

        check_target = text if text else key
        is_junk, reason = is_stage1_junk(str(check_target))

        if is_junk:
            quarantine_data[key] = {"val": text, "stage": "Stage 1 (Rule)", "reason": reason}
            processed_keys.add(key)
            stage1_junk_count += 1
        else:
            stage2_candidates.append((key, text))

    print(f"Stage 1 Complete:")
    print(f" - {stage1_junk_count} code/non-Japanese/dev junk lines quarantined.")
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

        print(f"Total candidates: {len(stage2_candidates)} across {len(batches)} batches.")
        print(
            f"Running with max_workers={max_workers}, batch_size={batch_size}, autosave every {save_interval} batches.\n")

        lock = threading.RLock()
        completed_batches = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Group batches into synchronized waves matching max_workers
            wave_size = config.get("max_workers", 4)
            waves = [batches[i: i + wave_size] for i in range(0, len(batches), wave_size)]

            print(
                f"Processing {len(batches)} batches across {len(waves)} synchronized waves (Wave size: {wave_size})...\n")

            lock = threading.RLock()
            completed_batches = 0

            for wave_idx, current_wave in enumerate(waves, 1):
                # Process only the current wave in parallel
                with ThreadPoolExecutor(max_workers=len(current_wave)) as executor:
                    future_to_batch = {
                        executor.submit(call_qwen_batch_classification, batch, config): batch
                        for batch in current_wave
                    }

                    # Wait for EVERY worker in this wave to complete before starting the next wave
                    for future in as_completed(future_to_batch):
                        batch = future_to_batch[future]
                        results = future.result()

                        with lock:
                            for idx, key, text in batch:
                                # If string matches standard item/plant patterns, force keep = True
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

                # Autosave after wave completes
                if completed_batches % save_interval == 0 or completed_batches == len(batches):
                    print(f"Wave {wave_idx}/{len(waves)} complete. Autosaving progress...")
                    save_progress(out_cleaned_path, out_quarantine_path, checkpoint_path, cleaned_data, quarantine_data,
                                  lock)

    # Final save & sync
    save_progress(out_cleaned_path, out_quarantine_path, checkpoint_path, cleaned_data, quarantine_data,
                  threading.RLock())

    print("\n=== Processing Complete ===")
    print(f"Total Original Lines: {len(data)}")
    print(f"Cleaned Lines Saved:  {len(cleaned_data)} -> {out_cleaned_path.name}")
    print(f"Quarantined Lines:    {len(quarantine_data)} -> {out_quarantine_path.name}")


if __name__ == "__main__":
    process_json_file()
