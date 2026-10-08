"""Tests for JSONTranslator logic, parameter jitter, key schema, and boundary recovery."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

# Ensure src/ is on sys.path
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mtool_translator.translator import JSONTranslator
from mtool_translator.utils import is_translatable_text, repair_json_string


class TestTranslatorTemperatureJitter(unittest.TestCase):
    """Verifies temperature jitter escalation on retry attempts in JSONTranslator."""

    def test_temperature_escalation_on_retries(self) -> None:
        """Asserts attempt 0 uses base temp (0.05), attempt 1 uses 0.1, attempt 2 uses 0.2."""
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
        """Verifies filtering of translatable vs non-translatable text patterns."""
        self.assertFalse(is_translatable_text("・"))
        self.assertFalse(is_translatable_text("・Fixed error"))
        self.assertFalse(is_translatable_text("------"))
        self.assertFalse(is_translatable_text("   "))
        self.assertTrue(is_translatable_text("「さて、"))
        self.assertTrue(is_translatable_text("。早々なんだけど……」"))
        self.assertTrue(is_translatable_text("こんにちは"))

    def test_extract_translated_item_variants(self) -> None:
        """Verifies key extraction under varying prefix formats."""
        self.assertEqual(JSONTranslator._extract_translated_item(0, {"k_01": "Hello"}), "Hello")
        self.assertEqual(JSONTranslator._extract_translated_item(0, {"1": "Hello"}), "Hello")
        self.assertEqual(JSONTranslator._extract_translated_item(0, {"01": "Hello"}), "Hello")
        self.assertEqual(JSONTranslator._extract_translated_item(0, {"k_1": "Hello"}), "Hello")
        self.assertIsNone(JSONTranslator._extract_translated_item(0, {"k_02": "Hello"}))

    def test_missing_colon_key_repair(self) -> None:
        """Verifies recovery of keys with missing colons."""
        malformed = '{"k_01Hello", "k_02": "World"}'
        repaired = repair_json_string(malformed)
        self.assertIn('"k_01": "Hello"', repaired)

    def test_boundary_recovery_retranslates_predecessor_and_missing(self) -> None:
        """Verifies missing keys and predecessor retranslation on merged outputs."""
        translator = JSONTranslator("config.json")
        retranslated: list[str] = []

        def mock_translate_single(text_tuple: tuple[str, str]) -> str:
            retranslated.append(text_tuple[0])
            return f"Recovered {text_tuple[0]}"

        translator._translate_single_item = mock_translate_single  # type: ignore[method-assign]

        texts = [
            ("k5", "「…………（どきどき）」"),
            ("k6", "「さて、"),
            ("k7", "。早々なんだけど……」"),
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
