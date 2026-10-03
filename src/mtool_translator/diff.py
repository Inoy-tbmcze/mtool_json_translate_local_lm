"""Translation Diff & Merge Module for Game Localization JSON.

Enables incremental translation workflows when games are updated:
1. Diff: Compares updated ManualTransFile.json with existing ManualTransFile_translated.json
   and isolates only new/untranslated keys for translation.
2. Merge: Combines newly translated keys back into the master ManualTransFile_translated.json.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import resolve_input_path, resolve_output_path
from .utils import dump_json_file, load_json_file


def is_entry_translated(
    key: str,
    translated_data: dict[str, Any],
    key_presence_only: bool = False,
) -> bool:
    """Checks whether a key has a valid translation in translated_data.

    Args:
        key: The original Japanese source key.
        translated_data: Dictionary of existing translations.
        key_presence_only: If True, key existence is sufficient. If False (safe mode),
            requires a non-empty string value differing from the original key.

    Returns:
        True if the key is considered translated, False otherwise.
    """
    if key not in translated_data:
        return False
    if key_presence_only:
        return True

    val = translated_data[key]
    if val is None or not isinstance(val, str):
        return False

    val_stripped = val.strip()
    if not val_stripped:
        return False

    return val != key


def filter_untranslated_keys(
    current_data: dict[str, Any],
    translated_data: dict[str, Any],
    key_presence_only: bool = False,
) -> dict[str, Any]:
    """Filters out keys from current_data that are already translated in translated_data.

    Args:
        current_data: Raw or current game JSON mapping (source strings).
        translated_data: Existing translated game JSON mapping.
        key_presence_only: If True, treat any existing key as translated.
            If False (default, safe mode), a key is considered translated only
            if it exists in translated_data, has a non-empty string value,
            and its value is different from the original key.

    Returns:
        A new dictionary containing only untranslated keys from current_data,
        preserving original insertion order.
    """
    untranslated: dict[str, Any] = {}
    for key, val in current_data.items():
        if not is_entry_translated(key, translated_data, key_presence_only=key_presence_only):
            untranslated[key] = val
    return untranslated


def merge_translations(
    base_data: dict[str, Any],
    new_data: dict[str, Any],
    overwrite_existing: bool = True,
    safe_check: bool = True,
) -> tuple[dict[str, Any], int]:
    """Merges newly translated entries into base translation dictionary.

    Args:
        base_data: Master translated dictionary to update.
        new_data: Newly translated dictionary.
        overwrite_existing: Whether to overwrite existing translations in base_data.
        safe_check: If True, only copy non-empty translations that differ from the key.

    Returns:
        Tuple of (merged_dictionary, updated_keys_count).
    """
    merged = dict(base_data)
    updated_count = 0

    for k, v in new_data.items():
        if safe_check and (v is None or not isinstance(v, str) or not v.strip() or v == k):
            continue

        if not overwrite_existing and k in merged:
            continue

        if k not in merged or merged[k] != v:
            merged[k] = v
            updated_count += 1

    return merged, updated_count


def process_diff(
    current_file: str | Path = "ManualTransFile.json",
    translated_file: str | Path = "ManualTransFile_translated.json",
    output_file: str | Path | None = None,
    key_presence_only: bool = False,
) -> tuple[Path, int, int]:
    """Isolates untranslated keys from updated game JSON against existing translations.

    Args:
        current_file: Path or filename of the updated raw game JSON.
        translated_file: Path or filename of existing translated JSON.
        output_file: Target output path. If None, modifies current_file in-place.
        key_presence_only: Whether to treat any key present in translated_file as translated.

    Returns:
        Tuple of (output_path, removed_count, remaining_count).
    """
    current_path = resolve_input_path(current_file, default_subfolder="raw")
    if not current_path.exists():
        raise FileNotFoundError(f"Current game JSON not found: {current_path}")

    trans_path = resolve_input_path(translated_file, default_subfolder="processed")
    if not trans_path.exists():
        trans_path = resolve_input_path(translated_file, default_subfolder="raw")
    if not trans_path.exists():
        raise FileNotFoundError(f"Translated reference JSON not found: {trans_path}")

    current_data = load_json_file(current_path)
    if not isinstance(current_data, dict):
        raise TypeError(
            f"Expected JSON object in {current_path}, got {type(current_data).__name__}"
        )

    translated_data = load_json_file(trans_path)
    if not isinstance(translated_data, dict):
        raise TypeError(
            f"Expected JSON object in {trans_path}, got {type(translated_data).__name__}"
        )

    untranslated_data = filter_untranslated_keys(
        current_data, translated_data, key_presence_only=key_presence_only
    )

    total_keys = len(current_data)
    remaining_keys = len(untranslated_data)
    removed_keys = total_keys - remaining_keys

    if output_file is not None:
        target_path = Path(output_file)
        if not target_path.is_absolute():
            target_path = resolve_output_path(target_path, default_subfolder="processed")
    else:
        target_path = current_path

    dump_json_file(target_path, untranslated_data, indent=True)

    print("=" * 60)
    print("Translation Diff Summary")
    print("=" * 60)
    print(f"Current game file:     {current_path}")
    print(f"Translated reference:  {trans_path}")
    print(f"Total keys checked:    {total_keys}")
    print(f"Already translated:    {removed_keys} (removed)")
    print(f"Untranslated keys:     {remaining_keys} (retained)")
    mode_str = (
        "Key presence only"
        if key_presence_only
        else "Safe (non-empty & different from key)"
    )
    print(f"Filter mode:           {mode_str}")
    inplace_str = " (in-place)" if target_path == current_path else ""
    print(f"Output saved to:       {target_path}{inplace_str}")
    print("=" * 60)

    return target_path, removed_keys, remaining_keys


def process_merge(
    base_file: str | Path = "ManualTransFile_translated.json",
    new_file: str | Path = "ManualTransFile_translated.json",
    output_file: str | Path | None = None,
    overwrite_existing: bool = True,
    safe_check: bool = True,
) -> tuple[Path, int]:
    """Merges newly translated JSON lines back into the master translated file.

    Args:
        base_file: Path or filename of the master translated JSON file.
        new_file: Path or filename of newly translated JSON file.
        output_file: Target output path. If None, modifies base_file in-place.
        overwrite_existing: Whether to overwrite existing entries in base_file.
        safe_check: Whether to ensure merged values are valid non-empty translations.

    Returns:
        Tuple of (output_path, updated_keys_count).
    """
    base_path = resolve_input_path(base_file, default_subfolder="processed")
    if not base_path.exists():
        base_path = resolve_input_path(base_file, default_subfolder="raw")
    if not base_path.exists():
        raise FileNotFoundError(f"Master translated file not found: {base_path}")

    new_path = resolve_input_path(new_file, default_subfolder="processed")
    if not new_path.exists():
        new_path = resolve_input_path(new_file, default_subfolder="raw")
    if not new_path.exists():
        raise FileNotFoundError(f"New translated file not found: {new_path}")

    base_data = load_json_file(base_path)
    if not isinstance(base_data, dict):
        raise TypeError(f"Expected JSON object in {base_path}, got {type(base_data).__name__}")

    new_data = load_json_file(new_path)
    if not isinstance(new_data, dict):
        raise TypeError(f"Expected JSON object in {new_path}, got {type(new_data).__name__}")

    merged_data, updated_count = merge_translations(
        base_data,
        new_data,
        overwrite_existing=overwrite_existing,
        safe_check=safe_check,
    )

    if output_file is not None:
        target_path = Path(output_file)
        if not target_path.is_absolute():
            target_path = resolve_output_path(target_path, default_subfolder="processed")
    else:
        target_path = base_path

    dump_json_file(target_path, merged_data, indent=True)

    print("=" * 60)
    print("Translation Merge Summary")
    print("=" * 60)
    print(f"Master translated file: {base_path}")
    print(f"New translations file:  {new_path}")
    print(f"Original master keys:   {len(base_data)}")
    print(f"New keys provided:      {len(new_data)}")
    print(f"Keys updated / added:   {updated_count}")
    print(f"Total merged keys:      {len(merged_data)}")
    inplace_str = " (in-place)" if target_path == base_path else ""
    print(f"Output saved to:        {target_path}{inplace_str}")
    print("=" * 60)

    return target_path, updated_count
