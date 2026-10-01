"""Unit and integration test suite for in-house JSON repair and resilient LLM parsing."""

from __future__ import annotations

import unittest
from typing import Any

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

    def test_repair_trailing_backslash_before_newline(self) -> None:
        """Trailing backslashes before newlines are normalized into valid JSON escapes."""
        raw = '{"8": "Yes, repair is complete!\\\nMy costs are fair~~"}'
        repaired = repair_json_string(raw)
        data = fast_json_loads(repaired)
        self.assertEqual(data["8"], "Yes, repair is complete!\nMy costs are fair~~")

    def test_repair_double_quoted_keys(self) -> None:
        """Accidental double quotes on numeric keys are normalized."""
        raw = '{"1": "Well - Underwater", ""2": "Well - Above Water", ""3": "Dead Wasteland"}'
        repaired = repair_json_string(raw)
        data = fast_json_loads(repaired)
        self.assertEqual(data["2"], "Well - Above Water")
        self.assertEqual(data["3"], "Dead Wasteland")

    def test_repair_unquoted_opening_key(self) -> None:
        """Numeric keys missing an opening quote are restored."""
        raw = '{"2": "Pink Bikini",3": "Bottom"}'
        repaired = repair_json_string(raw)
        data = fast_json_loads(repaired)
        self.assertEqual(data["3"], "Bottom")

    def test_repair_colon_digit_suffix(self) -> None:
        """RPG maker colon-digit suffixes closed prematurely are repaired cleanly."""
        raw = '{"14": "No more!":69, "15": "I don\'t know!"}'
        repaired = repair_json_string(raw)
        data = fast_json_loads(repaired)
        self.assertEqual(data["14"], "No more!")
        self.assertEqual(data["15"], "I don't know!")


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

    def test_missing_colon_digit_key_repair(self) -> None:
        """Repairs keys where opening quote and colon are missing between digits and value."""
        raw = '{"13": "Mining", "14Atl. Gratitude", "15": "Done"}'
        res = parse_llm_json_response(raw)
        self.assertEqual(res.get("14"), "Atl. Gratitude")

    def test_parse_llm_json_response_recovers_wave47_malformed_batch(self) -> None:
        """Recovers 25/25 keys from Wave 47 Batch 2 comma-split value."""
        raw = (
            '{"1":"A","2":"B","3":"C","4":"D","5":"E","6":"F","7":"G","8":"H",'
            '"9":"I","10":"J","11":"K","12":"L","13":"M","14":"N","15":"O","16":"P",'
            '"17":"Q","18":"R","19":"S","20":"T","21":"U","22":"V","23":"W","24":"X",'
            '"25":"Deleting \\"○○○.\\",\\"sd\\" is sufficient."}'
        )
        res = parse_llm_json_response(raw)
        self.assertEqual(len(res), 25)
        self.assertIn("Deleting", res["25"])

    def test_parse_llm_json_response_recovers_wave49_malformed_batch(self) -> None:
        """Recovers 25/25 keys from Wave 49 Batch 1 missing colon on line 14."""
        raw = (
            '{"1":"A","2":"B","3":"C","4":"D","5":"E","6":"F","7":"G","8":"H",'
            '"9":"I","10":"J","11":"K","12":"L","13":"M",'
            '"14Atl. Since we owe you a debt of gratitude,",'
            '"15":"O","16":"P","17":"Q","18":"R","19":"S","20":"T","21":"U","22":"V",'
            '"23":"W","24":"X","25":"Y"}'
        )
        res = parse_llm_json_response(raw)
        self.assertEqual(len(res), 25)
        self.assertEqual(res["14"], "Atl. Since we owe you a debt of gratitude,")


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


class TestTranslatorTemperatureJitter(unittest.TestCase):
    """Verifies temperature jitter escalation on retry attempts in JSONTranslator."""

    def test_temperature_escalation_on_retries(self) -> None:
        """Asserts attempt 0 uses base temp (0.05), attempt 1 uses 0.1, attempt 2 uses 0.2."""
        from unittest.mock import MagicMock
        from mtool_translator.translator import JSONTranslator

        translator = JSONTranslator.__new__(JSONTranslator)
        translator.config = {
            "max_retries": 3,
            "retry_delay": 0.001,
            "request_timeout": 1,
            "temperature": 0.05,
        }
        translator.logger = MagicMock()
        translator.session = MagicMock()

        captured_temps: list[float] = []

        def fake_post(*_args: Any, **kwargs: Any) -> MagicMock:
            json_payload = kwargs.get("json", {})
            captured_temps.append(json_payload.get("temperature"))
            resp = MagicMock()
            if len(captured_temps) < 3:
                resp.status_code = 500
                resp.text = "Internal Server Error"
            else:
                resp.status_code = 200
                resp.json.return_value = {
                    "choices": [{"message": {"content": '{"1": "translated"}'}}]
                }
            return resp

        translator.session.post.side_effect = fake_post
        translator.is_valid_translation = MagicMock(return_value=True)

        res = translator._send_translation_request(
            api_url="http://test/v1/chat/completions",
            headers={},
            data={"model": "test-model"},
            texts=[("orig_1", "orig_1")],
            fallback_results={"orig_1": "orig_1"},
        )

        self.assertEqual(captured_temps, [0.05, 0.1, 0.2])
        self.assertEqual(res, {"orig_1": "translated"})


class TestKeySchemaAndBoundaryRecovery(unittest.TestCase):
    """Tests non-sequential key mapping, translatable text filtering, and boundary recovery."""

    def test_is_translatable_text(self) -> None:
        from mtool_translator.utils import is_translatable_text

        self.assertFalse(is_translatable_text("・"))
        self.assertFalse(is_translatable_text("・Fixed error"))
        self.assertFalse(is_translatable_text("------"))
        self.assertFalse(is_translatable_text("   "))
        self.assertTrue(is_translatable_text("「さて、"))
        self.assertTrue(is_translatable_text("。早速なんだけど……」"))
        self.assertTrue(is_translatable_text("こんにちは"))

    def test_extract_translated_item_variants(self) -> None:
        from mtool_translator.translator import JSONTranslator

        self.assertEqual(JSONTranslator._extract_translated_item(0, {"k_01": "Hello"}), "Hello")
        self.assertEqual(JSONTranslator._extract_translated_item(0, {"1": "Hello"}), "Hello")
        self.assertEqual(JSONTranslator._extract_translated_item(0, {"01": "Hello"}), "Hello")
        self.assertEqual(JSONTranslator._extract_translated_item(0, {"k_1": "Hello"}), "Hello")
        self.assertIsNone(JSONTranslator._extract_translated_item(0, {"k_02": "Hello"}))

    def test_missing_colon_key_repair(self) -> None:
        from mtool_translator.utils import repair_json_string

        malformed = '{"k_01Hello", "k_02": "World"}'
        repaired = repair_json_string(malformed)
        self.assertIn('"k_01": "Hello"', repaired)

    def test_boundary_recovery_retranslates_predecessor_and_missing(self) -> None:
        from mtool_translator.translator import JSONTranslator

        translator = JSONTranslator("config.json")
        retranslated: list[str] = []

        def mock_translate_single(text_tuple: tuple[str, str]) -> str:
            retranslated.append(text_tuple[0])
            return f"Recovered {text_tuple[0]}"

        translator._translate_single_item = mock_translate_single  # type: ignore[method-assign]

        texts = [
            ("k5", "「…………（どきどき）」"),
            ("k6", "「さて、"),
            ("k7", "。早速なんだけど……」"),
            ("k8", "「な、なんですか？」"),
        ]
        # Simulate LLM merged k6 and k7 into k_02, omitting k_03
        llm_response = {
            "k_01": "...Heart pounding...",
            "k_02": "Well, without further ado...",
            "k_04": "Wh-what is it?",
        }

        res = translator._map_translation_response(texts, llm_response)
        # Both predecessor (k6 at index 1) and missing line (k7 at index 2) must be retranslated
        self.assertIn("k6", retranslated)
        self.assertIn("k7", retranslated)
        self.assertEqual(res["k6"], "Recovered k6")
        self.assertEqual(res["k7"], "Recovered k7")
        self.assertEqual(res["k5"], "...Heart pounding...")
        self.assertEqual(res["k8"], "Wh-what is it?")


if __name__ == "__main__":
    unittest.main()
