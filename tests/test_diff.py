"""Unit tests for Translation Diff & Merge functionality."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mtool_translator.diff import (
    filter_untranslated_keys,
    is_entry_translated,
    merge_translations,
    process_diff,
    process_merge,
)
from mtool_translator.utils import dump_json_file, load_json_file


class TestDiffLogic(unittest.TestCase):
    """Test suite for core diff and merge helper logic."""

    def test_is_entry_translated_safe_mode(self) -> None:
        """Verifies safe check requires non-empty string differing from key."""
        trans = {
            "剣": "Sword",
            "盾": "盾",
            "薬": "",
            "石": "   ",
            "杖": None,
        }
        self.assertTrue(is_entry_translated("剣", trans, key_presence_only=False))
        self.assertFalse(is_entry_translated("盾", trans, key_presence_only=False))
        self.assertFalse(is_entry_translated("薬", trans, key_presence_only=False))
        self.assertFalse(is_entry_translated("石", trans, key_presence_only=False))
        self.assertFalse(is_entry_translated("杖", trans, key_presence_only=False))
        self.assertFalse(is_entry_translated("弓", trans, key_presence_only=False))

    def test_is_entry_translated_key_presence_only(self) -> None:
        """Verifies key presence mode treats any existing key as translated."""
        trans = {
            "剣": "Sword",
            "盾": "盾",
            "薬": "",
        }
        self.assertTrue(is_entry_translated("剣", trans, key_presence_only=True))
        self.assertTrue(is_entry_translated("盾", trans, key_presence_only=True))
        self.assertTrue(is_entry_translated("薬", trans, key_presence_only=True))
        self.assertFalse(is_entry_translated("弓", trans, key_presence_only=True))

    def test_filter_untranslated_keys(self) -> None:
        """Asserts already translated keys are removed while untranslated are kept."""
        current = {
            "剣": "剣",
            "盾": "盾",
            "鎧": "鎧",
            "新ダンジョン": "新ダンジョン",
        }
        translated = {
            "剣": "Sword",
            "盾": "盾",  # Untranslated in previous translation
        }
        # Safe mode: "剣" is removed; "盾" remains because value == key; "鎧" and "新ダンジョン" remain
        res = filter_untranslated_keys(current, translated, key_presence_only=False)
        self.assertEqual(
            res,
            {
                "盾": "盾",
                "鎧": "鎧",
                "新ダンジョン": "新ダンジョン",
            },
        )

        # Key presence only: both "剣" and "盾" are removed
        res_kp = filter_untranslated_keys(current, translated, key_presence_only=True)
        self.assertEqual(
            res_kp,
            {
                "鎧": "鎧",
                "新ダンジョン": "新ダンジョン",
            },
        )

    def test_merge_translations(self) -> None:
        """Asserts newly translated lines merge into master translation correctly."""
        base = {
            "剣": "Sword",
            "盾": "Old Shield",
        }
        new = {
            "盾": "Steel Shield",
            "鎧": "Armor",
            "未翻訳": "未翻訳",  # Should be skipped under safe_check=True
            "空": "",  # Should be skipped under safe_check=True
        }
        merged, count = merge_translations(
            base, new, overwrite_existing=True, safe_check=True
        )
        self.assertEqual(count, 2)  # "盾" updated, "鎧" added
        self.assertEqual(
            merged,
            {
                "剣": "Sword",
                "盾": "Steel Shield",
                "鎧": "Armor",
            },
        )

    def test_merge_translations_no_overwrite(self) -> None:
        """Asserts overwrite_existing=False preserves existing base translations."""
        base = {
            "剣": "Sword",
            "盾": "Old Shield",
        }
        new = {
            "盾": "Steel Shield",
            "鎧": "Armor",
        }
        merged, count = merge_translations(
            base, new, overwrite_existing=False, safe_check=True
        )
        self.assertEqual(count, 1)  # Only "鎧" added
        self.assertEqual(merged["盾"], "Old Shield")
        self.assertEqual(merged["鎧"], "Armor")


class TestDiffProcess(unittest.TestCase):
    """Test suite for process_diff and process_merge file operations."""

    def test_process_diff_in_place(self) -> None:
        """Verifies process_diff modifies current game JSON in-place when output_file is None."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cur_file = Path(tmp_dir) / "ManualTransFile.json"
            trans_file = Path(tmp_dir) / "ManualTransFile_translated.json"

            dump_json_file(
                cur_file,
                {
                    "ロード": "ロード",
                    "セーブ": "セーブ",
                    "新しいボス": "新しいボス",
                },
            )
            dump_json_file(
                trans_file,
                {
                    "ロード": "Load",
                    "セーブ": "Save",
                },
            )

            out_path, removed, remaining = process_diff(
                current_file=cur_file,
                translated_file=trans_file,
                output_file=None,
            )

            self.assertEqual(out_path, cur_file)
            self.assertEqual(removed, 2)
            self.assertEqual(remaining, 1)

            updated_data = load_json_file(cur_file)
            self.assertEqual(updated_data, {"新しいボス": "新しいボス"})

    def test_process_diff_custom_output(self) -> None:
        """Verifies process_diff writes to specified output without modifying input."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cur_file = Path(tmp_dir) / "ManualTransFile.json"
            trans_file = Path(tmp_dir) / "ManualTransFile_translated.json"
            diff_file = Path(tmp_dir) / "ManualTransFile_diff.json"

            initial_data = {
                "ロード": "ロード",
                "新しいボス": "新しいボス",
            }
            dump_json_file(cur_file, initial_data)
            dump_json_file(trans_file, {"ロード": "Load"})

            out_path, removed, remaining = process_diff(
                current_file=cur_file,
                translated_file=trans_file,
                output_file=diff_file,
            )

            self.assertEqual(out_path, diff_file)
            self.assertEqual(removed, 1)
            self.assertEqual(remaining, 1)

            # Original current file should remain untouched
            self.assertEqual(load_json_file(cur_file), initial_data)
            # Output file contains only remaining untranslated lines
            self.assertEqual(load_json_file(diff_file), {"新しいボス": "新しいボス"})

    def test_process_merge_in_place(self) -> None:
        """Verifies process_merge updates master translated file in-place."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_file = Path(tmp_dir) / "ManualTransFile_translated.json"
            new_file = Path(tmp_dir) / "ManualTransFile_diff_translated.json"

            dump_json_file(base_file, {"ロード": "Load"})
            dump_json_file(new_file, {"新しいボス": "New Boss"})

            out_path, count = process_merge(
                base_file=base_file,
                new_file=new_file,
                output_file=None,
            )

            self.assertEqual(out_path, base_file)
            self.assertEqual(count, 1)
            self.assertEqual(
                load_json_file(base_file),
                {
                    "ロード": "Load",
                    "新しいボス": "New Boss",
                },
            )

    def test_process_diff_missing_file_raises(self) -> None:
        """Asserts FileNotFoundError is raised when an input file does not exist."""
        with self.assertRaises(FileNotFoundError):
            process_diff(
                current_file="nonexistent_current_file_xyz.json",
                translated_file="nonexistent_trans_file_xyz.json",
            )

    def test_merge_guarantees_no_duplicate_keys(self) -> None:
        """Asserts that merged JSON contains strictly unique keys without duplicates."""
        import json

        def strict_pairs_hook(pairs: list[tuple[str, str]]) -> dict[str, str]:
            seen: dict[str, str] = {}
            for k, v in pairs:
                if k in seen:
                    raise ValueError(f"Duplicate key detected in JSON: {k}")
                seen[k] = v
            return seen

        with tempfile.TemporaryDirectory() as tmp_dir:
            base_file = Path(tmp_dir) / "ManualTransFile_translated.json"
            retrans_pass2 = Path(tmp_dir) / "ManualTransFile_retranslated_pass2.json"

            # Stage 2 initial translations
            dump_json_file(
                base_file,
                {
                    "攻撃": "Attack",
                    "防御": "Defense",
                    "逃走": "Escape_Failed_Raw",
                },
            )

            # Stage 3 retranslation recovery pass
            dump_json_file(
                retrans_pass2,
                {
                    "逃走": "Flee",
                    "魔法": "Magic",
                },
            )

            process_merge(base_file=base_file, new_file=retrans_pass2)

            # Read raw bytes and verify strict uniqueness at JSON parser level
            raw_content = base_file.read_text(encoding="utf-8")
            parsed_strict = json.loads(raw_content, object_pairs_hook=strict_pairs_hook)

            self.assertEqual(len(parsed_strict), 4)
            self.assertEqual(parsed_strict["攻撃"], "Attack")
            self.assertEqual(parsed_strict["防御"], "Defense")
            self.assertEqual(parsed_strict["逃走"], "Flee")  # Successfully recovered
            self.assertEqual(parsed_strict["魔法"], "Magic")

    def test_retranslation_recovery_merged_to_final_translation(self) -> None:
        """Verifies that lines retranslated on pass 2 are merged into final output files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            trans_file = Path(tmp_dir) / "ManualTransFile_translated.json"
            valid_file = Path(tmp_dir) / "ManualTransFile_translated_validated.json"
            retrans_pass2 = Path(tmp_dir) / "pass2_retranslated.json"

            # Pass 1 initial state: "剣" succeeded, "謎の呪文" failed validation
            dump_json_file(trans_file, {"剣": "Sword", "謎の呪文": "BadHallucination"})
            dump_json_file(valid_file, {"剣": "Sword"})

            # Pass 2 retranslation result: "謎の呪文" successfully gets translated to English
            dump_json_file(retrans_pass2, {"謎の呪文": "Mysterious Spell"})

            # Recovery merges retranslated lines directly into final files
            process_merge(base_file=valid_file, new_file=retrans_pass2)
            process_merge(base_file=trans_file, new_file=retrans_pass2)

            # Both files must now contain the retranslated line
            valid_data = load_json_file(valid_file)
            self.assertEqual(valid_data["謎の呪文"], "Mysterious Spell")
            self.assertEqual(valid_data["剣"], "Sword")

            trans_data = load_json_file(trans_file)
            self.assertEqual(trans_data["謎の呪文"], "Mysterious Spell")
            self.assertEqual(trans_data["剣"], "Sword")


if __name__ == "__main__":
    unittest.main()
