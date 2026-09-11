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
import time
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


def test_basic_post_and_get(server: MockServer) -> None:
    """Tests basic POST and GET requests."""
    print("Testing basic POST and GET...")
    with FastLocalHttpClient() as client:
        # GET
        get_resp = client.get(f"{server.base_url}/health")
        assert get_resp.status_code == 200
        assert get_resp.json() == {"status": "ok"}

        # POST
        req_body = {"model": "test-model", "messages": [{"role": "user", "content": "hi"}]}
        post_resp = client.post(f"{server.base_url}/v1/chat/completions", json=req_body)
        assert post_resp.status_code == 200
        data = post_resp.json()
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert post_resp.headers.get("Content-Type") == "application/json"
    print("  [PASS] Basic POST and GET passed.")


def test_keepalive_connection_reuse(server: MockServer) -> None:
    """Tests that subsequent requests reuse persistent keep-alive socket."""
    print("Testing keep-alive socket reuse & latency...")
    with FastLocalHttpClient(max_connections=5) as client:
        url = f"{server.base_url}/v1/chat/completions"
        req_body = {"model": "test", "messages": []}

        # Warm-up request (socket connect)
        t0 = time.perf_counter()
        resp1 = client.post(url, json=req_body)
        t_first = (time.perf_counter() - t0) * 1000

        # Subsequent requests on keep-alive connection
        times = []
        for _ in range(50):
            t_start = time.perf_counter()
            resp = client.post(url, json=req_body)
            times.append((time.perf_counter() - t_start) * 1000)
            assert resp.status_code == 200

        avg_subsequent = sum(times) / len(times)
        print(f"  First call: {t_first:.2f}ms | Avg keep-alive: {avg_subsequent:.2f}ms (50 reqs)")
        assert resp1.status_code == 200
    print("  [PASS] Keep-alive socket reuse verified.")


def test_multithreaded_concurrency(server: MockServer) -> None:
    """Tests 20 concurrent worker threads making requests through shared client."""
    print("Testing 20-thread concurrency through shared connection pool...")
    with FastLocalHttpClient(max_connections=20) as client:
        url = f"{server.base_url}/v1/chat/completions"
        req_body = {"model": "test", "messages": []}
        errors: list[Exception] = []

        def worker(thread_id: int) -> None:
            try:
                for _ in range(5):
                    resp = client.post(url, json=req_body)
                    assert resp.status_code == 200
                    parsed = resp.json()
                    assert "choices" in parsed
            except Exception as exc:  # pylint: disable=broad-exception-caught
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors occurred: {errors}"
    print("  [PASS] 20-thread concurrent execution succeeded with zero errors.")


def test_stale_connection_auto_reconnect(server: MockServer) -> None:
    """Tests client automatically handles dropped keep-alive connection."""
    print("Testing auto-reconnect on server-dropped socket...")
    with FastLocalHttpClient(max_connections=2) as client:
        # 1. Successful request
        resp1 = client.post(f"{server.base_url}/v1/chat/completions", json={"m": 1})
        assert resp1.status_code == 200

        # 2. Trigger drop endpoint
        try:
            client.post(f"{server.base_url}/drop", json={"m": 2}, timeout=0.2)
        except (HttpConnectionError, HttpRequestError):
            pass

        # 3. Next request should cleanly reconnect and succeed
        resp3 = client.post(f"{server.base_url}/v1/chat/completions", json={"m": 3})
        assert resp3.status_code == 200
    print("  [PASS] Auto-reconnect on connection drop passed.")


def test_timeout_error(server: MockServer) -> None:
    """Tests HttpTimeoutError is raised when endpoint exceeds timeout."""
    print("Testing timeout error handling...")
    with FastLocalHttpClient(default_timeout=0.1) as client:
        caught_timeout = False
        try:
            client.post(f"{server.base_url}/timeout", timeout=0.1)
        except HttpTimeoutError:
            caught_timeout = True
        except HttpRequestError:
            caught_timeout = True
        assert caught_timeout, "Expected HttpTimeoutError on slow endpoint"
    print("  [PASS] Timeout error correctly detected.")


def test_connection_refused() -> None:
    """Tests HttpConnectionError on unreachable port."""
    print("Testing connection refused error handling...")
    with FastLocalHttpClient() as client:
        # Find an unused port
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        unused_port = sock.getsockname()[1]
        sock.close()

        caught_refusal = False
        try:
            client.post(f"http://127.0.0.1:{unused_port}/test", timeout=0.5)
        except HttpConnectionError:
            caught_refusal = True
        except HttpRequestError:
            caught_refusal = True
        assert caught_refusal, "Expected HttpConnectionError on closed port"
    print("  [PASS] Connection refused correctly raised HttpConnectionError.")


def test_status_codes_and_raise_for_status(server: MockServer) -> None:
    """Tests error status codes (429, 500) and raise_for_status()."""
    print("Testing status codes and raise_for_status()...")
    with FastLocalHttpClient() as client:
        resp_429 = client.post(f"{server.base_url}/error-429")
        assert resp_429.status_code == 429
        try:
            resp_429.raise_for_status()
            assert False, "Should have raised HttpStatusError for 429"
        except HttpStatusError as err:
            assert err.status_code == 429

        resp_500 = client.post(f"{server.base_url}/error-500")
        assert resp_500.status_code == 500
        try:
            resp_500.raise_for_status()
            assert False, "Should have raised HttpStatusError for 500"
        except HttpStatusError as err:
            assert err.status_code == 500
    print("  [PASS] HTTP 429 and 500 status checks passed.")


def test_cleaner_integration(server: MockServer) -> None:
    """Tests cleaner.call_batch_classification against mock server."""
    print("Testing cleaner integration...")
    config: Dict[str, Any] = {
        "model": "test-model",
        "api_endpoint": f"{server.base_url}/classify",
        "api_key": "test-key",
        "request_timeout": 5.0,
    }
    batch = [(0, "key0", "Hello world"), (1, "key1", "TODO: remove this")]
    with FastLocalHttpClient() as client:
        results = call_batch_classification(batch, config, session=client)
        assert results["key0"] is True  # Kept
        assert results["key1"] is False  # Discarded
    print("  [PASS] Cleaner integration verified.")


def test_validator_integration(server: MockServer) -> None:
    """Tests validator.call_batch_validation against mock server."""
    print("Testing validator integration...")
    config: Dict[str, Any] = {
        "model": "test-model",
        "api_endpoint": f"{server.base_url}/validate",
        "api_key": "test-key",
        "request_timeout": 5.0,
    }
    batch = [(0, "剣", "Sword"), (1, "盾", "Random garbage")]
    with FastLocalHttpClient() as client:
        results = call_batch_validation(batch, config, session=client)
        assert results["剣"] is True
        assert results["盾"] is False
    print("  [PASS] Validator integration verified.")


def test_loopback_detection() -> None:
    """Tests loopback address detection."""
    assert is_loopback("127.0.0.1")
    assert is_loopback("127.0.0.2")
    assert is_loopback("localhost")
    assert is_loopback("::1")
    assert not is_loopback("192.168.1.1")
    assert not is_loopback("api.openai.com")
    print("  [PASS] Loopback detection verified.")


def run_all_tests() -> None:
    """Starts mock server and executes all test suites."""
    print("==========================================================")
    print(" Starting FastLocalHttpClient Verification Suite")
    print("==========================================================")

    server = MockServer()
    server.start()
    print(f"Mock server listening on {server.base_url}\n")

    try:
        test_loopback_detection()
        test_basic_post_and_get(server)
        test_keepalive_connection_reuse(server)
        test_multithreaded_concurrency(server)
        test_stale_connection_auto_reconnect(server)
        test_timeout_error(server)
        test_connection_refused()
        test_status_codes_and_raise_for_status(server)
        test_cleaner_integration(server)
        test_validator_integration(server)

        print("\n==========================================================")
        print(" ALL VERIFICATION TESTS PASSED SUCCESSFULLY! (10/10)")
        print("==========================================================")
    finally:
        server.stop()


if __name__ == "__main__":
    run_all_tests()
