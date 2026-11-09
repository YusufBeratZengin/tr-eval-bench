"""Shared test fixtures: a fake OpenAI-compatible embedding server and helpers."""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np
import pytest


def _vector_for(text: str, dim: int = 8) -> list[float]:
    """Deterministic pseudo-embedding derived from SHA-256 (no network)."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [digest[i] / 255.0 + 0.01 for i in range(dim)]


def default_responder(payload: dict[str, Any], path: str) -> tuple[int, Any]:
    inputs = payload.get("input", [])
    if isinstance(inputs, str):
        inputs = [inputs]
    data = [
        {"object": "embedding", "index": index, "embedding": _vector_for(text)}
        for index, text in enumerate(inputs)
    ]
    return 200, {"object": "list", "data": data, "model": payload.get("model", "")}


class _EmbeddingHandler(BaseHTTPRequestHandler):
    server_version = "FakeEmbeddingServer/1.0"

    def log_message(self, format: str, *args: object) -> None:
        pass

    def do_POST(self) -> None:
        state: dict[str, Any] = self.server.state  # type: ignore[attr-defined]
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {}
        state["requests"].append(
            {"path": self.path, "payload": payload, "headers": dict(self.headers)}
        )
        responder: Callable[[dict[str, Any], str], tuple[int, Any]] = state["responder"]
        status, body = responder(payload, self.path)
        encoded = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


class FakeApiServer:
    """Threaded local HTTP server used to exercise the API runner offline."""

    def __init__(self) -> None:
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _EmbeddingHandler)
        self._server.state = {"responder": default_responder, "requests": []}  # type: ignore[attr-defined]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/v1"

    @property
    def requests(self) -> list[dict[str, Any]]:
        return self._server.state["requests"]  # type: ignore[attr-defined,no-any-return]

    def respond_with(self, responder: Callable[[dict[str, Any], str], tuple[int, Any]]) -> None:
        self._server.state["responder"] = responder  # type: ignore[attr-defined]

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


@pytest.fixture
def api_server() -> Iterator[FakeApiServer]:
    server = FakeApiServer()
    try:
        yield server
    finally:
        server.close()


class KeywordRunner:
    """Runner that maps known texts to fixed vectors (optionally recording calls)."""

    def __init__(
        self,
        mapping: dict[str, Sequence[float]],
        *,
        dim: int = 3,
        fail_on: str | None = None,
        name: str = "fake:keyword",
    ) -> None:
        self.mapping = {text: list(vector) for text, vector in mapping.items()}
        self._dim = dim
        self.fail_on = fail_on
        self._name = name
        self.calls: list[list[str]] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str], *, batch_size: int = 32) -> np.ndarray:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.calls.append(list(texts))
        if self.fail_on is not None and self.fail_on in texts:
            raise RuntimeError("boom")
        rows = [self.mapping.get(text, [0.0] * self._dim) for text in texts]
        matrix = np.asarray(rows, dtype=np.float64)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return matrix / norms


@contextmanager
def fake_sentence_transformers_module(encode_impl: Callable[..., Any]) -> Iterator[None]:
    """Install a stub ``sentence_transformers`` module for the duration of a test."""
    import sys
    import types

    module = types.ModuleType("sentence_transformers")

    class _FakeModel:
        def __init__(self, model_name: str, **kwargs: object) -> None:
            self.model_name = model_name
            self.kwargs = kwargs

        def encode(self, texts: Sequence[str], **kwargs: object) -> Any:
            return encode_impl(texts, **kwargs)

        def get_sentence_embedding_dimension(self) -> int:
            return 3

    module.SentenceTransformer = _FakeModel  # type: ignore[attr-defined]
    previous = sys.modules.get("sentence_transformers")
    sys.modules["sentence_transformers"] = module
    try:
        yield
    finally:
        if previous is None:
            sys.modules.pop("sentence_transformers", None)
        else:
            sys.modules["sentence_transformers"] = previous
