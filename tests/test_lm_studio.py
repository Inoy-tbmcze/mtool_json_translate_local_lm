"""Unit tests for LM Studio model lifecycle management."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from mtool_translator.http_client import HttpResponse, HttpStatusError
from mtool_translator.lm_studio import (
    ensure_model_loaded,
    get_lm_studio_base_url,
    get_loaded_instances,
    is_lm_studio_available,
    load_model,
    unload_all_models,
)


class TestLMStudio(unittest.TestCase):
    """Test suite for LM Studio probing, unloader, and dynamic loader."""

    def test_get_lm_studio_base_url(self) -> None:
        """Verifies root base URL extraction from various API endpoints."""
        self.assertEqual(
            get_lm_studio_base_url("http://127.0.0.1:1234/v1/chat/completions"),
            "http://127.0.0.1:1234",
        )
        self.assertEqual(
            get_lm_studio_base_url("http://localhost:5000/v1/models"),
            "http://localhost:5000",
        )
        self.assertEqual(
            get_lm_studio_base_url("http://192.168.1.100:8080/v1"),
            "http://192.168.1.100:8080",
        )
        self.assertEqual(
            get_lm_studio_base_url("http://127.0.0.1:1234"),
            "http://127.0.0.1:1234",
        )

    @patch("mtool_translator.lm_studio.default_client.get")
    def test_is_lm_studio_available_success(self, mock_get: MagicMock) -> None:
        """Verifies detection when LM Studio /api/v1/models responds 200."""
        mock_get.return_value = HttpResponse(
            status_code=200,
            content=b'{"models": [{"key": "test-model"}]}',
        )
        self.assertTrue(is_lm_studio_available("http://127.0.0.1:1234"))

    @patch("mtool_translator.lm_studio.default_client.get")
    def test_is_lm_studio_available_offline(self, mock_get: MagicMock) -> None:
        """Verifies graceful False when LM Studio is unreachable."""
        mock_get.side_effect = HttpStatusError("Connection refused")
        self.assertFalse(is_lm_studio_available("http://127.0.0.1:1234"))

    @patch("mtool_translator.lm_studio.default_client.get")
    def test_get_loaded_instances(self, mock_get: MagicMock) -> None:
        """Verifies extraction of all loaded instance IDs."""
        mock_get.return_value = HttpResponse(
            status_code=200,
            content=(
                b'{"models": ['
                b'  {"key": "model-1", "loaded_instances": [{"id": "inst-1"}, {"id": "inst-2"}]},'
                b'  {"key": "model-2", "loaded_instances": []}'
                b']}'
            ),
        )
        instances = get_loaded_instances("http://127.0.0.1:1234")
        self.assertEqual(instances, ["inst-1", "inst-2"])

    @patch("mtool_translator.lm_studio.default_client.post")
    @patch("mtool_translator.lm_studio.get_loaded_instances")
    def test_unload_all_models(
        self, mock_get_instances: MagicMock, mock_post: MagicMock
    ) -> None:
        """Verifies unloading all active model instances."""
        mock_get_instances.return_value = ["inst-1", "inst-2"]
        mock_post.return_value = HttpResponse(status_code=200, content=b'{"instance_id": "inst-1"}')

        count = unload_all_models("http://127.0.0.1:1234")
        self.assertEqual(count, 2)
        self.assertEqual(mock_post.call_count, 2)

    @patch("mtool_translator.lm_studio.default_client.post")
    def test_load_model(self, mock_post: MagicMock) -> None:
        """Verifies model loading payload filters load keys."""
        mock_post.return_value = HttpResponse(
            status_code=200,
            content=b'{"instance_id": "inst-loaded", "load_time_seconds": 1.25}',
        )
        config = {
            "model": "my-target-model",
            "context_length": 8192,
            "eval_batch_size": 512,
            "temperature": 0.2,  # Not in LOAD_CONFIG_KEYS
        }
        success = load_model("http://127.0.0.1:1234", "my-target-model", config)
        self.assertTrue(success)

        called_url, kwargs = mock_post.call_args
        self.assertEqual(called_url[0], "http://127.0.0.1:1234/api/v1/models/load")
        sent_json = kwargs["json"]
        self.assertEqual(sent_json["model"], "my-target-model")
        self.assertEqual(sent_json["context_length"], 8192)
        self.assertNotIn("temperature", sent_json)

    @patch("mtool_translator.lm_studio.load_model")
    @patch("mtool_translator.lm_studio.unload_all_models")
    @patch("mtool_translator.lm_studio.is_lm_studio_available")
    def test_ensure_model_loaded_when_available(
        self,
        mock_avail: MagicMock,
        mock_unload: MagicMock,
        mock_load: MagicMock,
    ) -> None:
        """Verifies unloads existing and loads target model when available."""
        mock_avail.return_value = True
        mock_unload.return_value = 1
        mock_load.return_value = True

        config = {
            "api_endpoint": "http://127.0.0.1:1234/v1/chat/completions",
            "model": "gemma-4-e4b-uncensored-hauhaucs-aggressive",
        }
        result = ensure_model_loaded(config, stage_name="cleaning")
        self.assertTrue(result)
        mock_unload.assert_called_once_with("http://127.0.0.1:1234")
        mock_load.assert_called_once_with(
            "http://127.0.0.1:1234",
            "gemma-4-e4b-uncensored-hauhaucs-aggressive",
            config,
        )

    @patch("mtool_translator.lm_studio.is_lm_studio_available")
    def test_ensure_model_loaded_when_offline(self, mock_avail: MagicMock) -> None:
        """Verifies graceful skip when server is offline."""
        mock_avail.return_value = False
        config = {
            "api_endpoint": "http://127.0.0.1:1234/v1/chat/completions",
            "model": "gemma-4-e4b-uncensored-hauhaucs-aggressive",
        }
        result = ensure_model_loaded(config, stage_name="cleaning")
        self.assertFalse(result)

    @patch("mtool_translator.lm_studio.load_model")
    @patch("mtool_translator.lm_studio.unload_all_models")
    @patch("mtool_translator.lm_studio.is_lm_studio_available")
    def test_ensure_model_loaded_failure_raises(
        self,
        mock_avail: MagicMock,
        mock_unload: MagicMock,
        mock_load: MagicMock,
    ) -> None:
        """Verifies RuntimeError is raised if load_model fails when server is available."""
        mock_avail.return_value = True
        mock_unload.return_value = 1
        mock_load.return_value = False

        config = {
            "api_endpoint": "http://127.0.0.1:1234/v1/chat/completions",
            "model": "gemma-4-e4b-uncensored-hauhaucs-aggressive",
        }
        with self.assertRaises(RuntimeError) as ctx:
            ensure_model_loaded(config, stage_name="cleaning")
        self.assertIn("Failed to load required model", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
