"""Embedding runners: Hugging Face sentence-transformers, OpenAI-compatible APIs
(Ollama, vLLM, TGI, llama.cpp server) and a dependency-free hashing baseline.

Every runner implements :class:`EmbeddingRunner` and returns a ``(n_texts, dim)``
float64 matrix with L2-normalized rows.  Runners are constructed eagerly so that
configuration problems surface before any dataset is scored.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

FloatMatrix = npt.NDArray[np.float64]

DEFAULT_API_URL = "http://localhost:11434/v1"
DEFAULT_TIMEOUT = 60.0
MAX_API_BATCH = 64


class RunnerError(RuntimeError):
    """Raised when an embedding backend cannot produce vectors."""


class RunnerUnavailable(RunnerError):
    """Raised when an optional dependency for a backend is not installed."""


@runtime_checkable
class EmbeddingRunner(Protocol):
    """Minimal interface every backend must satisfy."""

    @property
    def name(self) -> str:
        """Short human-readable identifier, e.g. ``hf:BAAI/bge-m3``."""

    @property
    def dimension(self) -> int | None:
        """Embedding size, or ``None`` when it is only known after the first call."""

    def embed(self, texts: Sequence[str], *, batch_size: int = 32) -> FloatMatrix:
        """Embed ``texts`` into an ``(n, dim)`` matrix of unit-length rows."""


def normalize_rows(matrix: npt.NDArray[np.floating]) -> FloatMatrix:
    """L2-normalize each row of a 2-D matrix; zero rows stay zero."""
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got shape {array.shape}")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return array / norms


def turkish_lower(text: str) -> str:
    """Lowercase Turkish text with the dotted/dotless i rules applied first."""
    return text.replace("İ", "i").replace("I", "ı").lower()


_WORD_RE = re.compile(r"\w+", re.UNICODE)


class HashingRunner:
    """Deterministic feature-hashing embedder over word and character n-grams.

    This is a real (if weak) lexical embedder, not a placeholder: it gives a
    reproducible offline baseline, powers the test suite, and lets users smoke
    test the CLI without downloading a model.  It is *not* a semantic model, so
    its STS scores are near zero by design.
    """

    def __init__(self, dimension: int = 256, ngram_range: tuple[int, int] = (3, 4)) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        low, high = ngram_range
        if not 1 <= low <= high:
            raise ValueError("ngram_range must satisfy 1 <= low <= high")
        self._dimension = int(dimension)
        self._ngram_range = (int(low), int(high))
        self._seed = b"tr-eval-bench-v1"

    @property
    def name(self) -> str:
        return f"hashing:{self._dimension}"

    @property
    def dimension(self) -> int:
        return self._dimension

    def _bucket(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(
            self._seed + b"|" + feature.encode("utf-8"), digest_size=9
        ).digest()
        index = int.from_bytes(digest[:8], "big") % self._dimension
        sign = 1.0 if digest[8] & 1 else -1.0
        return index, sign

    def _features(self, text: str) -> list[str]:
        lowered = turkish_lower(text)
        words = _WORD_RE.findall(lowered)
        features = [f"w:{word}" for word in words]
        low, high = self._ngram_range
        for word in words:
            padded = f"<{word}>"
            for size in range(low, high + 1):
                for start in range(0, max(len(padded) - size + 1, 0)):
                    features.append(f"c:{padded[start : start + size]}")
        return features

    def embed(self, texts: Sequence[str], *, batch_size: int = 32) -> FloatMatrix:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        matrix = np.zeros((len(texts), self._dimension), dtype=np.float64)
        for row, text in enumerate(texts):
            if not isinstance(text, str):
                raise RunnerError(f"text at index {row} is not a string")
            counts: dict[int, float] = {}
            for feature in self._features(text):
                index, sign = self._bucket(feature)
                counts[index] = counts.get(index, 0.0) + sign
            for index, value in counts.items():
                matrix[row, index] = np.sign(value) * (1.0 + np.log1p(abs(value)))
        return normalize_rows(matrix)


class SentenceTransformerRunner:
    """Hugging Face ``SentenceTransformer`` models, executed locally."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cpu",
        revision: str | None = None,
        trust_remote_code: bool = False,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name must not be empty")
        self._model_name = model_name.strip()
        self._device = device
        self._revision = revision
        self._trust_remote_code = trust_remote_code
        self._model = self._load()

    def _load(self) -> object:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RunnerUnavailable(
                "sentence-transformers is not installed. Install the optional extra with "
                "`pip install 'tr-eval-bench[hf]'`, or benchmark a local API endpoint with "
                "`--backend api --api-url http://localhost:11434/v1`."
            ) from exc
        try:
            return SentenceTransformer(
                self._model_name,
                device=self._device,
                revision=self._revision,
                trust_remote_code=self._trust_remote_code,
            )
        except Exception as exc:  # pragma: no cover - network/model specific
            raise RunnerError(
                f"could not load sentence-transformers model {self._model_name!r}: {exc}"
            ) from exc

    @property
    def name(self) -> str:
        return f"hf:{self._model_name}"

    @property
    def dimension(self) -> int | None:
        dim = getattr(self._model, "get_sentence_embedding_dimension", lambda: None)()
        return int(dim) if dim else None

    def embed(self, texts: Sequence[str], *, batch_size: int = 32) -> FloatMatrix:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if len(texts) == 0:
            return np.zeros((0, self.dimension or 0), dtype=np.float64)
        try:
            vectors = self._model.encode(  # type: ignore[attr-defined]
                list(texts),
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception as exc:  # pragma: no cover - backend specific
            raise RunnerError(f"embedding failed for {self.name}: {exc}") from exc
        matrix = np.asarray(vectors, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.shape[0] != len(texts):
            raise RunnerError(
                f"{self.name} returned {matrix.shape[0]} vectors for {len(texts)} texts"
            )
        if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
            raise RunnerError(f"{self.name} returned non-finite or malformed embeddings")
        return normalize_rows(matrix)


class OpenAICompatRunner:
    """Any OpenAI-compatible ``/v1/embeddings`` endpoint (Ollama, vLLM, TGI...).

    No third-party HTTP client is required: the standard library is used so the
    core install stays at two dependencies.
    """

    def __init__(
        self,
        model_name: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_batch: int = MAX_API_BATCH,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name must not be empty")
        resolved_url = (base_url or os.environ.get("TR_EVAL_API_URL") or DEFAULT_API_URL).strip()
        if not resolved_url.startswith(("http://", "https://")):
            raise ValueError(f"base_url must start with http:// or https://, got {resolved_url!r}")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if max_batch <= 0:
            raise ValueError("max_batch must be positive")
        self._model_name = model_name.strip()
        self._base_url = resolved_url.rstrip("/")
        self._api_key = api_key or os.environ.get("TR_EVAL_API_KEY")
        self._timeout = float(timeout)
        self._max_batch = int(max_batch)
        self._dimension: int | None = None

    @property
    def name(self) -> str:
        return f"api:{self._model_name}@{self._base_url}"

    @property
    def dimension(self) -> int | None:
        return self._dimension

    @property
    def endpoint(self) -> str:
        """Full embeddings URL, tolerating base URLs that already include ``/v1``."""
        if self._base_url.endswith("/embeddings"):
            return self._base_url
        return f"{self._base_url}/embeddings"

    def _request(self, payload: dict[str, object]) -> dict[str, object]:
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        request = urllib.request.Request(self.endpoint, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read()[:500].decode("utf-8", "replace") if exc.fp else ""
            raise RunnerError(
                f"{self.endpoint} returned HTTP {exc.code}: {detail or exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RunnerError(
                f"could not reach {self.endpoint}: {exc.reason}. Start the server "
                "(e.g. `ollama serve`) or pass --api-url."
            ) from exc
        except TimeoutError as exc:
            raise RunnerError(
                f"request to {self.endpoint} timed out after {self._timeout}s"
            ) from exc
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RunnerError(f"{self.endpoint} returned invalid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise RunnerError(
                f"{self.endpoint} returned a JSON {type(parsed).__name__}, not an object"
            )
        return parsed

    def _embed_batch(self, texts: Sequence[str]) -> FloatMatrix:
        payload: dict[str, object] = {"model": self._model_name, "input": list(texts)}
        response = self._request(payload)
        data = response.get("data")
        if not isinstance(data, list):
            raise RunnerError(f"{self.endpoint} response has no 'data' list")
        if len(data) != len(texts):
            raise RunnerError(
                f"{self.endpoint} returned {len(data)} embeddings for {len(texts)} inputs"
            )
        if not all(isinstance(item, dict) for item in data):
            raise RunnerError(f"{self.endpoint} response 'data' contains non-object entries")
        items: list[dict[str, object]] = list(data)
        expected = list(range(len(texts)))
        slots: list[dict[str, object] | None] = [None] * len(texts)
        for item in items:
            index = item.get("index")
            if not isinstance(index, int) or isinstance(index, bool):
                raise RunnerError(f"{self.endpoint} response items need an integer 'index'")
            if not 0 <= index < len(texts) or slots[index] is not None:
                raise RunnerError(
                    f"{self.endpoint} response indexes must be unique and cover 0..{len(texts) - 1}"
                )
            slots[index] = item
        if any(item is None for item in slots):
            raise RunnerError(
                f"{self.endpoint} response indexes must be unique and cover 0..{len(texts) - 1}"
            )
        ordered = [item for item in slots if item is not None]
        assert len(ordered) == len(expected)
        vectors: list[npt.NDArray[np.float64]] = []
        for item in ordered:
            embedding = item.get("embedding")
            if not isinstance(embedding, list) or not embedding:
                raise RunnerError(f"{self.endpoint} returned an empty embedding")
            if not all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in embedding
            ):
                raise RunnerError(f"{self.endpoint} returned a non-numeric embedding value")
            vector = np.asarray(embedding, dtype=np.float64)
            if vector.ndim != 1 or vector.size == 0:
                raise RunnerError(f"{self.endpoint} returned a malformed embedding")
            if not np.all(np.isfinite(vector)):
                raise RunnerError(f"{self.endpoint} returned an embedding with NaN or infinity")
            vectors.append(vector)
        dimensions = {vector.size for vector in vectors}
        if len(dimensions) != 1:
            raise RunnerError(
                f"{self.endpoint} returned embeddings with inconsistent dimensions "
                f"{sorted(dimensions)}"
            )
        matrix = np.vstack(vectors)
        if matrix.shape[0] != len(texts):
            raise RunnerError(
                f"{self.endpoint} returned {matrix.shape[0]} embeddings for {len(texts)} inputs"
            )
        if self._dimension is None:
            self._dimension = int(matrix.shape[1])
        elif matrix.shape[1] != self._dimension:
            raise RunnerError(
                f"{self.endpoint} changed embedding size mid-run "
                f"({self._dimension} -> {matrix.shape[1]})"
            )
        return normalize_rows(matrix)

    def embed(self, texts: Sequence[str], *, batch_size: int = 32) -> FloatMatrix:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if len(texts) == 0:
            return np.zeros((0, self._dimension or 0), dtype=np.float64)
        chunk = min(int(batch_size), self._max_batch)
        batches = [list(texts[start : start + chunk]) for start in range(0, len(texts), chunk)]
        matrices = [self._embed_batch(batch) for batch in batches]
        return np.vstack(matrices)


def load_runner(
    spec: str,
    *,
    backend: str = "auto",
    device: str = "cpu",
    api_url: str | None = None,
    api_key: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    trust_remote_code: bool = False,
) -> EmbeddingRunner:
    """Build a runner from a CLI/library model spec.

    Supported specs::

        hashing            -> HashingRunner(dimension=256)
        hashing:512        -> HashingRunner(dimension=512)
        hf:BAAI/bge-m3     -> SentenceTransformerRunner
        BAAI/bge-m3        -> SentenceTransformerRunner (auto)
        bge-m3             -> OpenAI-compatible endpoint (auto, only with --backend api)

    The endpoint for the ``api`` backend comes from ``api_url`` or the
    ``TR_EVAL_API_URL`` environment variable (default ``http://localhost:11434/v1``),
    never from the model spec, so a model name is never mistaken for a URL.
    """
    if not spec or not spec.strip():
        raise ValueError("model spec must not be empty")
    spec = spec.strip()
    resolved = backend.lower()
    if resolved not in {"auto", "hf", "api", "hashing"}:
        raise ValueError(f"unknown backend {backend!r}; expected auto, hf, api or hashing")
    if spec.startswith(("http://", "https://", "openai:")):
        raise ValueError(
            "pass the model name as the model spec and the endpoint with --api-url, "
            "e.g. `tr-eval-bench nomic-embed-text --backend api "
            "--api-url http://localhost:11434/v1`"
        )

    if resolved == "auto":
        if spec.startswith("hashing"):
            resolved = "hashing"
        elif spec.startswith("hf:"):
            resolved = "hf"
        else:
            resolved = "hf"

    if resolved == "hashing":
        _, _, dimension = spec.partition(":")
        if not dimension:
            return HashingRunner()
        try:
            return HashingRunner(dimension=int(dimension))
        except ValueError as exc:
            raise ValueError(f"invalid hashing dimension in {spec!r}: {exc}") from exc

    if resolved == "hf":
        return SentenceTransformerRunner(
            spec.removeprefix("hf:"), device=device, trust_remote_code=trust_remote_code
        )

    base_url = api_url
    return OpenAICompatRunner(spec, base_url=base_url, api_key=api_key, timeout=timeout)
