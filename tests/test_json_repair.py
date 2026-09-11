"""Unit and integration test suite for in-house JSON repair and resilient LLM parsing."""

from __future__ import annotations

import unittest

from mtool_translator.native_core import NATIVE_MANAGER, fast_find_json_bounds
from mtool_translator.utils import (
    fast_json_loads,
    parse_json_array_safely,
    parse_llm_json_response,
    repair_json_string,
)
from mtool_translator.validator import _parse_validation_json


class TestJsonRepair(unittest.TestCase):
    """Verifies in-house JSON repair across typical and adversarial LLM outputs."""

    def test_valid_json_untouched(self) -> None:
        """Valid JSON objects and arrays parse without modification."""
        payload_obj = '{"key": "value", "num": 42, "flag": true}'
        payload_arr = '[1, 2, "three", null]'
        self.assertEqual(fast_json_loads(repair_json_string(payload_obj)), {"key": "value", "num": 42, "flag": True})
        self.assertEqual(fast_json_loads(repair_json_string(payload_arr)), [1, 2, "three", None])

    def test_markdown_code_fences_and_preamble(self) -> None:
        """Markdown code blocks with commentary preambles and postambles extract cleanly."""
        raw = (
            "Here is the requested JSON output:\n"
            "```json\n"
            '{\n  "hero": "勇者",\n  "level": 99\n}\n'
            "```\n"
            "Hope this helps with your translation!"
        )
        repaired = repair_json_string(raw)
        data = fast_json_loads(repaired)
        self.assertEqual(data, {"hero": "勇者", "level": 99})

    def test_trailing_commas_in_objects_and_arrays(self) -> None:
        """Trailing commas inside objects and lists are sanitized."""
        raw = '{"items": [1, 2, 3, ], "config": {"debug": false, }, }'
        repaired = repair_json_string(raw)
        data = fast_json_loads(repaired)
        self.assertEqual(data, {"items": [1, 2, 3], "config": {"debug": False}})

    def test_truncated_string(self) -> None:
        """Strings cut off at token limits are closed safely."""
        raw = '{"status": "in_progress", "dialogue": "To be continued in the next epis'
        repaired = repair_json_string(raw)
        data = fast_json_loads(repaired)
        self.assertEqual(data["status"], "in_progress")
        self.assertTrue(data["dialogue"].startswith("To be continued"))

    def test_truncated_key_at_eof(self) -> None:
        """Keys cut off before colon or value are completed."""
        raw = '{"id": 101, "pending_key'
        repaired = repair_json_string(raw)
        data = fast_json_loads(repaired)
        self.assertEqual(data.get("id"), 101)
        self.assertIn("pending_key", data)

    def test_missing_value_after_colon(self) -> None:
        """Colons missing values at delimiter or EOF receive empty string fallback."""
        raw1 = '{"first": 1, "second": }'
        raw2 = '{"first": 1, "second": '
        data1 = fast_json_loads(repair_json_string(raw1))
        data2 = fast_json_loads(repair_json_string(raw2))
        self.assertEqual(data1["first"], 1)
        self.assertEqual(data1["second"], "")
        self.assertEqual(data2["first"], 1)
        self.assertEqual(data2["second"], "")

    def test_unclosed_nested_structures(self) -> None:
        """Deeply nested structures cut off mid-stream close in reverse delimiter order."""
        raw = '{"game": {"scenes": [{"id": 1, "actors": ["Hero", "Mage"'
        repaired = repair_json_string(raw)
        data = fast_json_loads(repaired)
        self.assertEqual(data["game"]["scenes"][0]["actors"], ["Hero", "Mage"])

    def test_unescaped_internal_and_trailing_quotes(self) -> None:
        """Internal dialogue quotes within string values are escaped properly."""
        raw1 = '{"line": "She shouted "Wait for me!" before running."}'
        raw2 = '{"line": "He whispered "Goodbye!""}'
        data1 = fast_json_loads(repair_json_string(raw1))
        data2 = fast_json_loads(repair_json_string(raw2))
        self.assertIn("Wait for me!", data1["line"])
        self.assertIn("Goodbye!", data2["line"])

    def test_raw_control_characters_in_strings(self) -> None:
        """Raw unescaped newlines and tabs inside strings are converted to valid JSON escapes."""
        raw = "{\n  \"line\": \"First line\nSecond line\tIndented\"\n}"
        repaired = repair_json_string(raw)
        data = fast_json_loads(repaired)
        self.assertEqual(data["line"], "First line\nSecond line\tIndented")

    def test_single_quoted_json(self) -> None:
        """Python-style single-quoted JSON dicts and arrays are repaired to double quotes."""
        raw = "{'code': 200, 'message': 'Success', 'tags': ['rpg', 'vn']}"
        repaired = repair_json_string(raw)
        data = fast_json_loads(repaired)
        self.assertEqual(data, {"code": 200, "message": "Success", "tags": ["rpg", "vn"]})

    def test_python_literals(self) -> None:
        """Python literals (True, False, None) normalize to JSON equivalents."""
        raw = '{"a": True, "b": False, "c": None}'
        repaired = repair_json_string(raw)
        data = fast_json_loads(repaired)
        self.assertEqual(data, {"a": True, "b": False, "c": None})

    def test_javascript_comments(self) -> None:
        """Line and block JavaScript-style comments are discarded."""
        raw = """
        {
          // Starting configuration
          "title": "Fantasy Game", /* Inline comment */
          "version": 1.2
        }
        """
        repaired = repair_json_string(raw)
        data = fast_json_loads(repaired)
        self.assertEqual(data["title"], "Fantasy Game")
        self.assertEqual(data["version"], 1.2)

    def test_empty_and_non_json_inputs(self) -> None:
        """Empty, whitespace, or delimiter-free strings return empty string."""
        self.assertEqual(repair_json_string(""), "")
        self.assertEqual(repair_json_string("   \n\t  "), "")
        self.assertEqual(repair_json_string("Just plain text with no braces"), "")

    def test_japanese_game_text_fidelity(self) -> None:
        """Full-width Japanese text, kanji, kana, and punctuation remain bit-exact."""
        raw = '{"101": "【勇者の剣】を手に入れた！", "102": "魔王：「覚悟しろ……！」"}'
        repaired = repair_json_string(raw)
        data = fast_json_loads(repaired)
        self.assertEqual(data["101"], "【勇者の剣】を手に入れた！")
        self.assertEqual(data["102"], "魔王：「覚悟しろ……！」")


class TestParseLlmJsonResponse(unittest.TestCase):
    """Verifies parse_llm_json_response behavior and error cases."""

    def test_parse_valid_and_fenced_dicts(self) -> None:
        """Parses direct and markdown-fenced dictionary responses."""
        resp1 = '{"0": "Hello", "1": "World"}'
        resp2 = '```json\n{"0": "Hello", "1": "World"}\n```'
        self.assertEqual(parse_llm_json_response(resp1), {"0": "Hello", "1": "World"})
        self.assertEqual(parse_llm_json_response(resp2), {"0": "Hello", "1": "World"})

    def test_parse_malformed_and_truncated(self) -> None:
        """Parses malformed responses with trailing commas and unclosed brackets."""
        resp = '{\n  "0": "Hello",\n  "1": "World",\n'
        self.assertEqual(parse_llm_json_response(resp), {"0": "Hello", "1": "World"})

    def test_raises_on_invalid_output(self) -> None:
        """Raises ValueError when input is empty or resolves to a non-dict."""
        with self.assertRaises(ValueError):
            parse_llm_json_response("")
        with self.assertRaises(ValueError):
            parse_llm_json_response("[1, 2, 3]")
        with self.assertRaises(ValueError):
            parse_llm_json_response("No JSON here")


class TestParseJsonArraySafely(unittest.TestCase):
    """Verifies parse_json_array_safely array extraction and repair."""

    def test_valid_and_fenced_arrays(self) -> None:
        """Parses valid array payloads."""
        self.assertEqual(parse_json_array_safely('["apple", "banana"]'), ["apple", "banana"])
        self.assertEqual(parse_json_array_safely('```json\n[1, 2, 3]\n```'), [1, 2, 3])

    def test_truncated_array(self) -> None:
        """Repairs truncated array at trailing comma or EOF."""
        self.assertEqual(parse_json_array_safely('["item1", "item2", '), ["item1", "item2"])

    def test_empty_or_non_array(self) -> None:
        """Returns empty list on invalid, empty, or dict responses."""
        self.assertEqual(parse_json_array_safely(""), [])
        self.assertEqual(parse_json_array_safely('{"a": 1}'), [])


class TestValidatorParsing(unittest.TestCase):
    """Verifies validator._parse_validation_json."""

    def test_validator_mapping(self) -> None:
        """Parses validation mapping output."""
        raw = '{"0": 1, "1": 0, "2": 1, }'
        self.assertEqual(_parse_validation_json(raw), {"0": 1, "1": 0, "2": 1})

    def test_validator_empty(self) -> None:
        """Returns empty dict on invalid input."""
        self.assertEqual(_parse_validation_json(""), {})


class TestNativeJsonBounds(unittest.TestCase):
    """Verifies native assembly and fallback JSON boundary scanners."""

    def test_json_bounds_detection(self) -> None:
        """Scans buffer and returns correct first and last delimiter indices."""
        text = "Leading text {\"key\": [1, 2, 3]} trailing note"
        raw_b = text.encode("utf-8")
        first, last = fast_find_json_bounds(raw_b)
        self.assertEqual(first, text.find("{"))
        self.assertEqual(last, text.rfind("}"))

    def test_native_manager_availability(self) -> None:
        """Asserts native manager is active on Windows x64."""
        self.assertTrue(NATIVE_MANAGER.is_available)


if __name__ == "__main__":
    unittest.main()
