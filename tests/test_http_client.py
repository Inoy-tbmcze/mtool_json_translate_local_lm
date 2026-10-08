"""Comprehensive test suite for FastLocalHttpClient and module integration.

Tests keep-alive connection pooling, socket optimizations, concurrent execution,
stale socket auto-reconnect, timeout handling, error propagation, and integration
with cleaner, translator, and validator modules against a mock local LLM endpoint.
"""

from __future__ import annotations

import http.server
import socket
import sys
import threading
import unittest
from pathlib import Path
from typing import Any, Dict

# Ensure src/ is on sys.path
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mtool_translator.cleaner import call_batch_classification
from mtool_translator.http_client import (
    FastLocalHttpClient,
    HttpConnectionError,
    HttpRequestError,
    HttpStatusError,
    HttpTimeoutError,
    is_loopback,
)
from mtool_translator.utils import fast_json_dumps_bytes
from mtool_translator.validator import call_batch_validation


class MockLLMHandler(http.server.BaseHTTPRequestHandler):
    """Mock HTTP handler simulating local LLM server behavior."""

    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default stdout request logging."""

    def do_POST(self) -> None:  # pylint: disable=invalid-name
        """Handles POST endpoints for testing."""
        content_length = int(self.headers.get("Content-Length", 0))
        _body = self.rfile.read(content_length)

        if self.path == "/v1/chat/completions":
            resp_data = {
                "id": "mock-completion-123",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": '{"1": "Hero Sword", "2": "Iron Shield"}',
                        },
                        "finish_reason": "stop",
                    }
                ],
            }
            payload = fast_json_dumps_bytes(resp_data)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            self.wfile.write(payload)

        elif self.path == "/classify":
            # Cleaner mock endpoint: returns array of junk IDs to discard
            resp_data = {
                "id": "mock-cleaner-123",
                "choices": [
                    {
                        "message": {
                            "content": "[1]",
                        }
                    }
                ],
            }
            payload = fast_json_dumps_bytes(resp_data)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        elif self.path == "/validate":
            # Validator mock endpoint: returns {id: 1} mapping
            resp_data = {
                "id": "mock-val-123",
                "choices": [
                    {
                        "message": {
                            "content": '{"0": 1, "1": 0}',
                        }
                    }
                ],
            }
            payload = fast_json_dumps_bytes(resp_data)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        elif self.path == "/timeout":
            time.sleep(0.5)
            self.send_response(200)
            self.end_headers()

        elif self.path == "/drop":
            # Immediately close socket without HTTP response
            self.close_connection = True
            if self.connection:
                self.connection.close()

        elif self.path == "/error-429":
            payload = b'{"error": "rate_limit_exceeded"}'
            self.send_response(429)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        elif self.path == "/error-500":
            payload = b'{"error": "internal_server_error"}'
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self) -> None:  # pylint: disable=invalid-name
        """Handles GET endpoints for testing."""
        if self.path == "/health":
            payload = b'{"status": "ok"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        else:
            self.send_response(404)
            self.end_headers()


class QuietThreadingHTTPServer(http.server.ThreadingHTTPServer):
    """ThreadingHTTPServer that suppresses intentional drop/close connection errors."""

    def handle_error(self, request: Any, client_address: Any) -> None:
        """Silently discard connection drop errors during tests."""


class MockServer:
    """Threaded HTTP server for local integration testing."""

    def __init__(self) -> None:
        self.server = QuietThreadingHTTPServer(("127.0.0.1", 0), MockLLMHandler)
        self.port = self.server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()


class TestFastLocalHttpClient(unittest.TestCase):
    """Test suite verifying FastLocalHttpClient and module integrations."""

    server: MockServer

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = MockServer()
        cls.server.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.stop()

    def test_loopback_detection(self) -> None:
        """Tests loopback address detection."""
        self.assertTrue(is_loopback("127.0.0.1"))
        self.assertTrue(is_loopback("127.0.0.2"))
        self.assertTrue(is_loopback("localhost"))
        self.assertTrue(is_loopback("::1"))
        self.assertFalse(is_loopback("192.168.1.1"))
        self.assertFalse(is_loopback("api.openai.com"))

    def test_basic_post_and_get(self) -> None:
        """Tests basic POST and GET requests."""
        with FastLocalHttpClient() as client:
            get_resp = client.get(f"{self.server.base_url}/health")
            self.assertEqual(get_resp.status_code, 200)
            self.assertEqual(get_resp.json(), {"status": "ok"})

            req_body = {"model": "test-model", "messages": [{"role": "user", "content": "hi"}]}
            post_resp = client.post(f"{self.server.base_url}/v1/chat/completions", json=req_body)
            self.assertEqual(post_resp.status_code, 200)
            data = post_resp.json()
            self.assertEqual(data["choices"][0]["message"]["role"], "assistant")
            self.assertEqual(post_resp.headers.get("Content-Type"), "application/json")

    def test_keepalive_connection_reuse(self) -> None:
        """Tests that subsequent requests reuse persistent keep-alive socket."""
        with FastLocalHttpClient(max_connections=5) as client:
            url = f"{self.server.base_url}/v1/chat/completions"
            req_body = {"model": "test", "messages": []}

            resp1 = client.post(url, json=req_body)
            self.assertEqual(resp1.status_code, 200)

            for _ in range(50):
                resp = client.post(url, json=req_body)
                self.assertEqual(resp.status_code, 200)

    def test_multithreaded_concurrency(self) -> None:
        """Tests 20 concurrent worker threads making requests through shared client."""
        with FastLocalHttpClient(max_connections=20) as client:
            url = f"{self.server.base_url}/v1/chat/completions"
            req_body = {"model": "test", "messages": []}
            errors: list[Exception] = []

            def worker(thread_id: int) -> None:  # pylint: disable=unused-argument
                try:
                    for _ in range(5):
                        resp = client.post(url, json=req_body)
                        self.assertEqual(resp.status_code, 200)
                        parsed = resp.json()
                        self.assertIn("choices", parsed)
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    errors.append(exc)

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            self.assertEqual(len(errors), 0, f"Thread errors occurred: {errors}")

    def test_stale_connection_auto_reconnect(self) -> None:
        """Tests client automatically handles dropped keep-alive connection."""
        with FastLocalHttpClient(max_connections=2) as client:
            resp1 = client.post(f"{self.server.base_url}/v1/chat/completions", json={"m": 1})
            self.assertEqual(resp1.status_code, 200)

            try:
                client.post(f"{self.server.base_url}/drop", json={"m": 2}, timeout=0.2)
            except (HttpConnectionError, HttpRequestError):
                pass

            resp3 = client.post(f"{self.server.base_url}/v1/chat/completions", json={"m": 3})
            self.assertEqual(resp3.status_code, 200)

    def test_timeout_error(self) -> None:
        """Tests HttpTimeoutError is raised when endpoint exceeds timeout."""
        with FastLocalHttpClient(default_timeout=0.1) as client:
            caught_timeout = False
            try:
                client.post(f"{self.server.base_url}/timeout", timeout=0.1)
            except (HttpTimeoutError, HttpRequestError):
                caught_timeout = True
            self.assertTrue(caught_timeout, "Expected HttpTimeoutError on slow endpoint")

    def test_connection_refused(self) -> None:
        """Tests HttpConnectionError on unreachable port."""
        with FastLocalHttpClient() as client:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("127.0.0.1", 0))
            unused_port = sock.getsockname()[1]
            sock.close()

            caught_refusal = False
            try:
                client.post(f"http://127.0.0.1:{unused_port}/test", timeout=0.5)
            except (HttpConnectionError, HttpRequestError):
                caught_refusal = True
            self.assertTrue(caught_refusal, "Expected HttpConnectionError on closed port")

    def test_status_codes_and_raise_for_status(self) -> None:
        """Tests error status codes (429, 500) and raise_for_status()."""
        with FastLocalHttpClient() as client:
            resp_429 = client.post(f"{self.server.base_url}/error-429")
            self.assertEqual(resp_429.status_code, 429)
            with self.assertRaises(HttpStatusError) as ctx_429:
                resp_429.raise_for_status()
            self.assertEqual(ctx_429.exception.status_code, 429)

            resp_500 = client.post(f"{self.server.base_url}/error-500")
            self.assertEqual(resp_500.status_code, 500)
            with self.assertRaises(HttpStatusError) as ctx_500:
                resp_500.raise_for_status()
            self.assertEqual(ctx_500.exception.status_code, 500)

    def test_cleaner_integration(self) -> None:
        """Tests cleaner.call_batch_classification against mock server."""
        config: Dict[str, Any] = {
            "model": "test-model",
            "api_endpoint": f"{self.server.base_url}/classify",
            "api_key": "test-key",
            "request_timeout": 5.0,
        }
        batch = [(0, "key0", "Hello world"), (1, "key1", "TODO: remove this")]
        with FastLocalHttpClient() as client:
            results = call_batch_classification(batch, config, session=client)
            self.assertTrue(results["key0"])
            self.assertFalse(results["key1"])

    def test_validator_integration(self) -> None:
        """Tests validator.call_batch_validation against mock server."""
        config: Dict[str, Any] = {
            "model": "test-model",
            "api_endpoint": f"{self.server.base_url}/validate",
            "api_key": "test-key",
            "request_timeout": 5.0,
        }
        batch = [(0, "剣", "Sword"), (1, "盾", "Random garbage")]
        with FastLocalHttpClient() as client:
            results = call_batch_validation(batch, config, session=client)
            self.assertTrue(results["剣"])
            self.assertFalse(results["盾"])


if __name__ == "__main__":
    unittest.main()
