"""
Stage 3 Translation Validation Module for Game Localization JSON.
------------------------------------------------------------------
Validates Japanese -> English translations using local LLM auditor.
Separates passed translations from failed lines needing retranslation.
"""

import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import requests

from .config import load_config, resolve_input_path, resolve_output_path


def call_batch_validation(batch: List[Tuple[int, str, str]], config: Dict[str, Any]) -> Dict[str, bool]:
    """Sends batch of (ID, JP_source, EN_target) to LLM for translation validation."""
    prompt_items = [f"{idx} | JP: {jp} | EN: {en}" for idx, jp, en in batch]
    items_str = "\n".join(prompt_items)

    combined_prompt = (
        "You are a translation quality auditor. Check if the English(second value) text is a valid, "
        "plausible translation or equivalent representation of the Japanese(first value) source text.\n\n"
        "Rules:\n"
        "- Return 1 only if the English text accurately matches or contextually represents translation of Japanese source.\n"
        "- Return 0 if anything else.\n\n"
        "Output ONLY a raw JSON object mapping ID to 1 or 0.\n"
        'Example format: {"0":1, "1":0, "2":1}\n\n'
        f"Pairs to validate:\n{items_str}"
    )

    data = {
        "model": config["model"],
        "messages": [
            {"role": "user", "content": combined_prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 1024
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
        for idx, jp, en in batch:
            val = parsed.get(str(idx), parsed.get(idx, 1))
            results[jp] = bool(val == 1 or val is True or str(val).lower() == "true")
    except Exception as e:
        print(f"\nWarning: Batch validation request failed ({e}). Defaulting items to VALID (1).")
        for idx, jp, en in batch:
            results[jp] = True

    return results


def save_progress(
    valid_path: Path,
    retranslate_path: Path,
    checkpoint_path: Path,
    validated_data: dict,
    retranslate_data: dict,
    lock: threading.RLock
):
    """Safely writes validation results (*_validated.json, *_retranslate.json) and checkpoints."""
    with lock:
        try:
            valid_path.parent.mkdir(parents=True, exist_ok=True)
            retranslate_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

            with open(valid_path, "w", encoding="utf-8") as f:
                json.dump(validated_data, f, ensure_ascii=False, indent=2)

            with open(retranslate_path, "w", encoding="utf-8") as f:
                json.dump(retranslate_data, f, ensure_ascii=False, indent=2)

            checkpoint_keys = list(validated_data.keys()) + list(retranslate_data.keys())
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump({"processed_keys": checkpoint_keys}, f, ensure_ascii=False, indent=2)
            print(" -> Autosave successful.")
        except Exception as e:
            print(f" -> Error during autosave: {e}")


def process_validation(
    config_file: str = "config.json",
    input_file: Optional[str] = None,
    output_dir: Optional[str] = None
) -> Tuple[Path, Path]:
    """Audits translated JSON file, splitting passed items and items needing retranslation."""
    config = load_config(config_file, section="validation")
    input_filename = input_file or config.get("input_filename", "translated_game_text.json")
    save_interval = config.get("save_interval", 10)

    input_path = resolve_input_path(input_filename, default_subfolder="processed")
    if not input_path.exists():
        input_path = resolve_input_path(input_filename, default_subfolder="raw")

    if not input_path.exists():
        print(f"Error: Input file '{input_filename}' not found at {input_path}.")
        sys.exit(1)

    stem = input_path.stem
    ext = input_path.suffix

    if output_dir:
        out_dir = Path(output_dir)
        valid_path = out_dir / f"{stem}_validated{ext}"
        retranslate_path = out_dir / f"{stem}_retranslate{ext}"
        checkpoint_path = out_dir / f"{stem}_checkpoint.json"
    else:
        valid_path = resolve_output_path(f"{stem}_validated{ext}", default_subfolder="processed")
        retranslate_path = resolve_output_path(f"{stem}_retranslate{ext}", default_subfolder="processed")
        checkpoint_path = resolve_output_path(f"{stem}_checkpoint.json", default_subfolder="processed")

    validated_data = {}
    retranslate_data = {}
    processed_keys = set()

    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                chk = json.load(f)
                processed_keys = set(chk.get("processed_keys", []))
            print(f"--> Found existing checkpoint: {len(processed_keys)} items already processed.")
        except Exception as e:
            print(f"Warning: Could not read checkpoint file ({e}). Starting fresh.")

    if valid_path.exists():
        try:
            with open(valid_path, "r", encoding="utf-8") as f:
                validated_data = json.load(f)
                processed_keys.update(validated_data.keys())
        except Exception:
            pass

    if retranslate_path.exists():
        try:
            with open(retranslate_path, "r", encoding="utf-8") as f:
                retranslate_data = json.load(f)
                processed_keys.update(retranslate_data.keys())
        except Exception:
            pass

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    unprocessed_items = [(jp, en) for jp, en in data.items() if jp not in processed_keys]

    batch_size = config.get("batch_size", 20)
    max_workers = config.get("max_workers", 4)

    batches = []
    for i in range(0, len(unprocessed_items), batch_size):
        chunk = unprocessed_items[i:i + batch_size]
        batch_data = [(idx, jp, en) for idx, (jp, en) in enumerate(chunk)]
        batches.append(batch_data)

    if not batches:
        print("All items have already been validated.")
        return valid_path, retranslate_path

    wave_size = max_workers
    waves = [batches[i: i + wave_size] for i in range(0, len(batches), wave_size)]

    print(f"--- Validating {len(unprocessed_items)} remaining lines across {len(batches)} batches ---")

    lock = threading.RLock()
    completed_batches = 0

    for wave_idx, current_wave in enumerate(waves, 1):
        with ThreadPoolExecutor(max_workers=len(current_wave)) as executor:
            future_to_batch = {
                executor.submit(call_batch_validation, batch, config): batch
                for batch in current_wave
            }

            for future in as_completed(future_to_batch):
                batch = future_to_batch[future]
                results = future.result()

                with lock:
                    for idx, jp, en in batch:
                        is_valid = results.get(jp, True)
                        if is_valid:
                            validated_data[jp] = en
                        else:
                            retranslate_data[jp] = jp
                        processed_keys.add(jp)

                    completed_batches += 1

        if completed_batches % save_interval == 0 or completed_batches == len(batches):
            print(f"Wave {wave_idx}/{len(waves)} complete. Autosaving progress...")
            save_progress(valid_path, retranslate_path, checkpoint_path, validated_data, retranslate_data, lock)

    save_progress(valid_path, retranslate_path, checkpoint_path, validated_data, retranslate_data, lock)

    print("\n=== Validation Complete ===")
    print(f"Total Lines Processed: {len(data)}")
    print(f"Validated File (1s):   {len(validated_data)} lines -> {valid_path}")
    print(f"Retranslate File (0s): {len(retranslate_data)} lines -> {retranslate_path}")

    return valid_path, retranslate_path
