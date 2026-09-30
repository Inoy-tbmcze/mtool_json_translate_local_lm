"""Unit tests for Stage 1 cleaner heuristics and box-drawing/grid-art detection."""

from __future__ import annotations

import unittest
from pathlib import Path

from mtool_translator.cleaner import is_stage1_junk
from mtool_translator.config import resolve_input_path
from mtool_translator.utils import (
    build_japanese_regex,
    is_box_drawing_or_grid_art,
    is_protected_sentence,
    load_japanese_symbols,
)


class TestCleanerHeuristics(unittest.TestCase):
    """Test suite asserting Stage 1 junk classification accuracy and edge cases."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initializes Japanese symbol regex from project configuration."""
        symbols_path = resolve_input_path("jp_symbols.json", default_subfolder="")
        if not Path(symbols_path).exists():
            symbols_path = resolve_input_path("jp_symbols.json", default_subfolder="reference")
        cls.jp_symbols = load_japanese_symbols(symbols_path)
        cls.jp_regex = build_japanese_regex(cls.jp_symbols)

    def test_user_samples_detected_as_grid_art(self) -> None:
        """Verifies that the user's three multiline ASCII/Unicode battle maps are quarantined."""
        s1 = (
            "┏━━━┳━━━┳…………︻…………┳━━━┳━━━┓\r\n"
            "10┃～～～￤　　　　　　　　①　　　　☖　　　┃㊦　　┃\r\n"
            "  ┃～～～￤　　　　　　　　　　　　　┃　　　┃　　　┃\r\n"
            " 9┃～～～￤　　　　　　　　　　　　　┃　　　┃　　　┃\r\n"
            "  ┃　　　￤　　　┃～～～～～～～～～┃　　　┃　　　┃\r\n"
            " 8┃　　　￤㊦　　┃～～～～～～～～～┃　　Ⓑ┃　　　┃\r\n"
            "  ┣☖━━┻━━━┛～～～～～～～～～┗━━━┻━━☖┫\r\n"
            " 7┃　　　～～～～～～～～～⃞～～～～～～～～～　　　┃"
        )
        s2 = (
            "┏━━━┳━━━━━┳━━━━━┳━━━━━┳━━━━━━━━━┓\r\n"
            "10┃∨～∨┃　　　　　☖　　　　　☖　　　　　￤㊤　　　　　　　　┃\r\n"
            "  ┃～～～┃　～～～　┃～～　～～┃　　　　　￤　　　～～～～～　┃\r\n"
            " 9┃∨～∨┃　～～～　┃～～　～～┃　　　　　￤　　　～～～～～　┃\r\n"
            "  ┃～～～┃　～～～　┃～～　～～┣━━☖━━╋━━━━━┓～～　┃\r\n"
            " 8┃∨～∨┃㊤～～～　┃～～　～～┃　　敵　　┃～～　　　┃～～　┃\r\n"
            "  ┃～～～┣……………┫～～　～～┃　　　　　┃～～　　　┃～～　┃\r\n"
            "  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛"
        )
        s3 = (
            "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━┓\r\n"
            "11┃～～～～～～～～～～～～～～～～～～～～～～～～　　　☖　　　　　┃\r\n"
            "  ┃～～～～～～～～～～～～～～～～～～～～～～～～　　　┃　　　　　┃\r\n"
            "10┃～～　　　　　　　　　　　　　　　　　　　　　　　　　┃　　　　❶┃\r\n"
            "  ┃～～　┏━━━━━━━┳………………………┓　　　┏━┻━━━━━┛\r\n"
            " 9┃～～　┃\t\t\t\t\t\t\t┃　　　　　～～～～￤　　　┃\r\n"
            "  ┗━━━┻━━━━━┛\t┗━━━━━━━━━━━━━┛\t┗━━━━━┛"
        )
        for s in (s1, s2, s3):
            self.assertTrue(is_box_drawing_or_grid_art(s))
            self.assertFalse(is_protected_sentence(s))
            is_junk, reason = is_stage1_junk(s, s, self.jp_regex)
            self.assertTrue(is_junk)
            self.assertEqual(reason, "box_drawing_or_grid_art")

    def test_single_grid_rows_quarantined(self) -> None:
        """Verifies individual tactical grid rows with row numbers and walls are quarantined."""
        row1 = "18\t\t\t\t┃■■宝　　　宝　宝┃"
        row2 = "6┃　　敵　　　　　　　　　　　　　　　　敵　　┃"
        row3 = "1┃　　≫　　／敵　　／敵　　／敵　　／敵　⑤"
        for row in (row1, row2, row3):
            self.assertTrue(is_box_drawing_or_grid_art(row))
            self.assertFalse(is_protected_sentence(row))
            is_junk, reason = is_stage1_junk(row, row, self.jp_regex)
            self.assertTrue(is_junk)
            self.assertEqual(reason, "box_drawing_or_grid_art")

    def test_valid_dialogue_with_box_dashes_preserved(self) -> None:
        """Verifies dialogue containing box drawing dashes (e.g. em-dashes) is preserved."""
        d1 = "「蕗──これ蕗なのか？　じゃあそっちの鋭い花は？」"
        d2 = "「ははは、由来を聞けば納得できるやもしれませんぞ」"
        d3 = "ふと、ベルタの畦に咲いたものに留まった。"
        for d in (d1, d2, d3):
            self.assertFalse(is_box_drawing_or_grid_art(d))
            self.assertTrue(is_protected_sentence(d))
            is_junk, reason = is_stage1_junk(d, d, self.jp_regex)
            self.assertFalse(is_junk, f"Expected {d!r} not to be junk, got reason: {reason}")

    def test_world_map_and_flowchart_preserved(self) -> None:
        """Verifies fast-travel location trees and quest flowcharts are preserved."""
        tree = (
            "―――――――――モイーズ湿地帯・西――┐\r\n"
            "														｜																・水の宮殿			｜\r\n"
            "	大峡谷―――――――コリンズの森																		モイーズ湿地帯・東\r\n"
            "・ラドミラの塔				・森の城塞																				・沼地の廃墟"
        )
        flowchart = (
            "＜ベトリヌスの酒場・クエストの流れ＞\r\n\r\n"
            "墓参り\r\n"
            "ポラリス神殿を守ってほしい―┬→コボルド退治―┬→ ワイルド・ブル"
        )
        for text in (tree, flowchart):
            self.assertFalse(is_box_drawing_or_grid_art(text))
            is_junk, reason = is_stage1_junk(text, text, self.jp_regex)
            self.assertFalse(is_junk, f"Expected tree/chart not to be junk, got: {reason}")

    def test_protected_sentence_requires_japanese_chars(self) -> None:
        """Verifies is_protected_sentence rejects strings lacking Japanese characters."""
        self.assertFalse(is_protected_sentence("... [12:00:00] INFO: Server initialized!"))
        self.assertFalse(is_protected_sentence("... --- ..."))
        self.assertFalse(is_protected_sentence("12345!?"))
        self.assertTrue(is_protected_sentence("待って……！"))
        self.assertTrue(is_protected_sentence("何…？"))


if __name__ == "__main__":
    unittest.main()
