"""High-performance, low-latency persistent HTTP client for local LLM inference.

Replaces external 'requests' dependency with a zero-overhead, connection-pooled
HTTP/1.1 client built on standard library http.client and native socket optimizations.
Applies TCP_NODELAY, SO_KEEPALIVE, and Windows SIO_LOOPBACK_FAST_PATH for ultra-low
latency IPC communication with local language model servers.
"""

from __future__ import annotations

import atexit
import functools
import http.client
import socket
import ssl
import sys
import threading
import time
import urllib.parse
from types import TracebackType
from typing import Any, Self

from .utils import fast_json_dumps_bytes, fast_json_loads

# Windows Winsock SIO_LOOPBACK_FAST_PATH control code (0x98000010)
SIO_LOOPBACK_FAST_PATH = 0x98000010

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "0.0.0.0"})


def is_loopback(host: str) -> bool:
    """Returns True if the host is a local loopback interface."""
    return host in _LOOPBACK_HOSTS or host.startswith("127.")


class HttpRequestError(Exception):
    """Base exception for all HTTP client errors."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response: HttpResponse | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response = response


class HttpTimeoutError(HttpRequestError):
    """Raised when an HTTP request or socket operation times out."""


class HttpConnectionError(HttpRequestError):
    """Raised when network connection fails, is refused, or drops unexpectedly."""


class HttpStatusError(HttpRequestError):
    """Raised by raise_for_status() when the response status code is 4xx or 5xx."""


# Backward compatibility alias for requests.RequestException
RequestException = HttpRequestError


class HttpResponse:
    """Lightweight, slot-optimized HTTP response wrapper."""

    __slots__ = ("_text", "content", "headers", "status_code")

    def __init__(
        self,
        status_code: int,
        content: bytes,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self.content = content
        self._text: str | None = None
        self.headers: dict[str, str] = headers or {}

    @property
    def text(self) -> str:
        """Decodes content to UTF-8 text with lazy caching."""
        if self._text is None:
            self._text = self.content.decode("utf-8", errors="replace")
        return self._text

    def json(self) -> Any:
        """Parses content directly from raw UTF-8 bytes using fast SIMD orjson."""
        return fast_json_loads(self.content)

    def raise_for_status(self) -> None:
        """Raises HttpStatusError if the response status code indicates an error."""
        if self.status_code >= 400:
            preview = self.text[:200]
            raise HttpStatusError(
                f"HTTP {self.status_code} Error: {preview}",
                status_code=self.status_code,
                response=self,
            )

    def __repr__(self) -> str:
        return f"<HttpResponse [{self.status_code}] len={len(self.content)}>"


def _create_optimized_socket(
    host: str,
    port: int,
    timeout: float | None = None,
) -> socket.socket:
    """Creates a raw TCP socket with TCP_NODELAY, SO_KEEPALIVE, and SIO_LOOPBACK_FAST_PATH."""
    addrinfo = socket.getaddrinfo(host, port, socket.AF_UNSPEC, socket.SOCK_STREAM)
    last_err: Exception | None = None

    for af, socktype, proto, _canonname, sa in addrinfo:
        sock = socket.socket(af, socktype, proto)
        try:
            if sys.platform == "win32" and is_loopback(host):
                try:
                    sock.ioctl(SIO_LOOPBACK_FAST_PATH, 1)
                except OSError:
                    pass

            if timeout is not None:
                sock.settimeout(timeout)

            sock.connect(sa)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            return sock
        except OSError as err:
            last_err = err
            sock.close()
            continue

    if last_err:
        raise last_err
    raise OSError(f"Could not resolve or connect to {host}:{port}")


# pylint: disable=too-few-public-methods
class _FastSendOutputMixin:
    """Zero-overhead mixin providing atomic header-body socket buffering."""

    _buffer: list[bytes]
    send: Any

    def _send_output(self, message_body: Any = None, encode_chunked: bool = False) -> None:
        """Sends headers and payload in a single contiguous buffer to eliminate fragmentation."""
        self._buffer.extend((b"", b""))
        header_bytes = b"\r\n".join(self._buffer)
        del self._buffer[:]
        if (
            message_body is not None
            and isinstance(message_body, (bytes, bytearray, memoryview))
            and not encode_chunked
        ):
            self.send(header_bytes + message_body)
        else:
            self.send(header_bytes)
            if message_body is not None:
                self.send(message_body)


class _FastHTTPConnection(_FastSendOutputMixin, http.client.HTTPConnection):
    """Custom HTTPConnection applying native socket optimizations and atomic header-body sends."""

    _buffer: list[bytes]

    def __init__(self, host: str, port: int, timeout: float = 60.0) -> None:
        super().__init__(host, port, timeout=timeout)
        self.is_closed = False

    def connect(self) -> None:
        self.sock = _create_optimized_socket(self.host, self.port, self.timeout)


class _FastHTTPSConnection(_FastSendOutputMixin, http.client.HTTPSConnection):
    """Custom HTTPSConnection applying TLS wrapping and atomic header-body sends."""

    _buffer: list[bytes]

    def __init__(self, host: str, port: int, timeout: float = 60.0) -> None:
        super().__init__(host, port, timeout=timeout)
        self.is_closed = False

    def connect(self) -> None:
        raw_sock = _create_optimized_socket(self.host, self.port, self.timeout)
        context = ssl.create_default_context()
        self.sock = context.wrap_socket(raw_sock, server_hostname=self.host)


class _HostConnectionPool:
    """LIFO connection pool for a specific (scheme, host, port) endpoint."""

    def __init__(
        self,
        scheme: str,
        host: str,
        port: int,
        max_size: int = 10,
        keepalive_timeout: float = 30.0,
    ) -> None:
        self.scheme = scheme
        self.host = host
        self.port = port
        self.max_size = max_size
        self.keepalive_timeout = keepalive_timeout
        self._pool: list[tuple[_FastHTTPConnection | _FastHTTPSConnection, float]] = []
        self._lock = threading.Lock()

    def _create_conn(self, timeout: float) -> _FastHTTPConnection | _FastHTTPSConnection:
        if self.scheme == "https":
            return _FastHTTPSConnection(self.host, self.port, timeout=timeout)
        return _FastHTTPConnection(self.host, self.port, timeout=timeout)

    def acquire(self, timeout: float) -> tuple[_FastHTTPConnection | _FastHTTPSConnection, bool]:
        """Acquires a connection from pool or creates a new one. Returns (conn, is_reused)."""
        now = time.monotonic()
        with self._lock:
            while self._pool:
                conn, last_used = self._pool.pop()
                if now - last_used > self.keepalive_timeout or conn.is_closed:
                    try:
                        conn.close()
                    except (OSError, http.client.HTTPException):
                        pass
                    continue

                if conn.sock is not None:
                    try:
                        conn.sock.settimeout(timeout)
                    except OSError:
                        pass
                return conn, True

        return self._create_conn(timeout), False

    def release(self, conn: _FastHTTPConnection | _FastHTTPSConnection) -> None:
        """Releases an open connection back to the LIFO pool."""
        if getattr(conn, "is_closed", False) or conn.sock is None:
            return
        with self._lock:
            if len(self._pool) < self.max_size:
                self._pool.append((conn, time.monotonic()))
            else:
                try:
                    conn.close()
                except (OSError, http.client.HTTPException):
                    pass

    def close(self) -> None:
        """Closes all active pooled connections."""
        with self._lock:
            for conn, _ in self._pool:
                try:
                    conn.close()
                except (OSError, http.client.HTTPException):
                    pass
            self._pool.clear()


@functools.lru_cache(maxsize=128)
def _parse_url(url: str) -> tuple[str, str, int, str]:
    """Parses URL into scheme, host, port, path_and_query with LRU cache."""
    parsed = urllib.parse.urlsplit(url)
    scheme = parsed.scheme.lower() or "http"
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if scheme == "https" else 80)
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    return scheme, host, port, path


class FastLocalHttpClient:
    """Thread-safe persistent HTTP client with connection pooling and TCP_NODELAY."""

    def __init__(
        self,
        max_connections: int = 10,
        default_timeout: float = 60.0,
        keepalive_timeout: float = 30.0,
    ) -> None:
        self.max_connections = max_connections
        self.default_timeout = default_timeout
        self.keepalive_timeout = keepalive_timeout
        self._pools: dict[tuple[str, str, int], _HostConnectionPool] = {}
        self._pools_lock = threading.Lock()
        self._is_closed = False

    def _get_pool(self, scheme: str, host: str, port: int) -> _HostConnectionPool:
        key = (scheme, host, port)
        with self._pools_lock:
            if key not in self._pools:
                self._pools[key] = _HostConnectionPool(
                    scheme,
                    host,
                    port,
                    max_size=self.max_connections,
                    keepalive_timeout=self.keepalive_timeout,
                )
            return self._pools[key]

    @staticmethod
    def _execute_on_conn(
        conn: _FastHTTPConnection | _FastHTTPSConnection,
        method: str,
        path: str,
        body: bytes | None,
        req_headers: dict[str, str],
    ) -> HttpResponse:
        if conn.sock is None:
            conn.connect()

        conn.putrequest(method, path, skip_host=False, skip_accept_encoding=True)
        for header_name, header_val in req_headers.items():
            conn.putheader(header_name, header_val)
        conn.endheaders(message_body=body)

        http_resp = conn.getresponse()
        resp_bytes = http_resp.read()
        resp_headers = dict(http_resp.getheaders())

        if http_resp.will_close:
            conn.is_closed = True
            conn.close()

        return HttpResponse(
            status_code=http_resp.status,
            content=resp_bytes,
            headers=resp_headers,
        )

    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    def request(
        self,
        method: str,
        url: str,
        data: bytes | str | None = None,
        json: Any | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> HttpResponse:
        """Sends an HTTP request with automatic keep-alive retry on stale connection."""
        if self._is_closed:
            raise HttpRequestError("FastLocalHttpClient is closed")

        req_timeout = timeout if timeout is not None else self.default_timeout
        scheme, host, port, path = _parse_url(url)
        pool = self._get_pool(scheme, host, port)

        payload: bytes | None = None
        req_headers: dict[str, str] = {k: str(v) for k, v in headers.items()} if headers else {}
        if json is not None:
            payload = fast_json_dumps_bytes(json)
            req_headers.setdefault("Content-Type", "application/json")
        elif isinstance(data, str):
            payload = data.encode("utf-8")
        elif isinstance(data, (bytes, bytearray)):
            payload = bytes(data)

        if payload is not None:
            req_headers["Content-Length"] = str(len(payload))
        req_headers.setdefault("Connection", "keep-alive")
        req_headers.setdefault("Accept", "*/*")

        conn, is_reused = pool.acquire(req_timeout)
        try:
            resp = self._execute_on_conn(conn, method, path, payload, req_headers)
            pool.release(conn)
            return resp
        except (
            http.client.RemoteDisconnected,
            ConnectionResetError,
            BrokenPipeError,
            http.client.CannotSendRequest,
            http.client.ResponseNotReady,
        ) as conn_err:
            try:
                conn.close()
            except (OSError, http.client.HTTPException):
                pass
            if is_reused:
                fresh_conn, _ = pool.acquire(req_timeout)
                try:
                    resp = self._execute_on_conn(fresh_conn, method, path, payload, req_headers)
                    pool.release(fresh_conn)
                    return resp
                except TimeoutError as to_err:
                    fresh_conn.close()
                    raise HttpTimeoutError(
                        f"Request timed out after {req_timeout}s: {url}"
                    ) from to_err
                except (OSError, http.client.HTTPException) as fresh_err:
                    fresh_conn.close()
                    raise HttpConnectionError(
                        f"Failed to connect to {url}: {fresh_err}"
                    ) from fresh_err
            raise HttpConnectionError(f"Connection error to {url}: {conn_err}") from conn_err
        except TimeoutError as to_err:
            conn.close()
            raise HttpTimeoutError(f"Request timed out after {req_timeout}s: {url}") from to_err
        except (OSError, http.client.HTTPException) as err:
            conn.close()
            raise HttpConnectionError(f"Network error communicating with {url}: {err}") from err

    def post(
        self,
        url: str,
        data: bytes | str | None = None,
        json: Any | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> HttpResponse:
        """Sends a POST request."""
        return self.request("POST", url, data=data, json=json, headers=headers, timeout=timeout)

    def get(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> HttpResponse:
        """Sends a GET request."""
        return self.request("GET", url, headers=headers, timeout=timeout)

    def close(self) -> None:
        """Closes all pooled connections across all host targets."""
        self._is_closed = True
        with self._pools_lock:
            for pool in self._pools.values():
                pool.close()
            self._pools.clear()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()


default_client = FastLocalHttpClient()
atexit.register(default_client.close)


def post(
    url: str,
    data: bytes | str | None = None,
    json: Any | None = None,
    headers: dict[str, str] | None = None,
    timeout: float | None = None,
) -> HttpResponse:
    """Convenience module-level POST using default persistent client."""
    return default_client.post(url, data=data, json=json, headers=headers, timeout=timeout)


def get(
    url: str,
    headers: dict[str, str] | None = None,
    timeout: float | None = None,
) -> HttpResponse:
    """Convenience module-level GET using default persistent client."""
    return default_client.get(url, headers=headers, timeout=timeout)
