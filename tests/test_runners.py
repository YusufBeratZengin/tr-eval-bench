"""Runner tests: hashing baseline, Hugging Face adapter (stubbed) and HTTP API."""

from __future__ import annotations

import json

import numpy as np
import pytest
from tr_eval_bench.runners import (
    HashingRunner,
    OpenAICompatRunner,
    RunnerError,
    RunnerUnavailable,
    SentenceTransformerRunner,
    load_runner,
    normalize_rows,
    turkish_lower,
)

from tests.conftest import KeywordRunner, fake_sentence_transformers_module

# Obviously fake credential used only to assert header forwarding; never a real key.
FAKE_BEARER = "example-fake-key-000000"


def test_hashing_runner_is_deterministic_and_normalized() -> None:
    first = HashingRunner(128)
    second = HashingRunner(128)
    texts = ["Türkçe metin gömme", "İkinci cümle"]
    matrix = first.embed(texts, batch_size=1)
    assert matrix.shape == (2, 128)
    np.testing.assert_allclose(matrix, second.embed(texts))
    np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), np.ones(2), atol=1e-9)


def test_hashing_runner_separates_unrelated_texts() -> None:
    runner = HashingRunner(256)
    vectors = runner.embed(["kedi bahçede uyuyor", "kuantum bilgisayarları"])
    similarity = float(vectors[0] @ vectors[1])
    assert similarity < 0.5


def test_hashing_runner_handles_empty_input() -> None:
    matrix = HashingRunner(16).embed([])
    assert matrix.shape == (0, 16)


def test_hashing_runner_validates_arguments() -> None:
    with pytest.raises(ValueError, match="dimension"):
        HashingRunner(0)
    with pytest.raises(ValueError, match="ngram_range"):
        HashingRunner(16, ngram_range=(4, 2))
    with pytest.raises(ValueError, match="batch_size"):
        HashingRunner(16).embed(["x"], batch_size=0)
    with pytest.raises(RunnerError, match="not a string"):
        HashingRunner(16).embed([None])  # type: ignore[list-item]


def test_turkish_lower_handles_dotted_and_dotless_i() -> None:
    assert turkish_lower("İSTANBUL") == "istanbul"
    assert turkish_lower("IŞIK") == "ışık"


def test_normalize_rows_keeps_zero_rows() -> None:
    matrix = normalize_rows(np.array([[3.0, 4.0], [0.0, 0.0]]))
    np.testing.assert_allclose(matrix[0], [0.6, 0.8])
    np.testing.assert_allclose(matrix[1], [0.0, 0.0], atol=1e-12)


def test_normalize_rows_rejects_non_matrix() -> None:
    with pytest.raises(ValueError, match="2-D"):
        normalize_rows(np.array([1.0, 2.0]))


def test_sentence_transformer_runner_embeds_with_stub() -> None:
    def encode(texts: list[str], **kwargs: object) -> np.ndarray:
        assert kwargs["normalize_embeddings"] is True
        return np.array([[3.0, 0.0, 4.0] for _ in texts])

    with fake_sentence_transformers_module(encode):
        runner = SentenceTransformerRunner("fake/model", device="cpu")
        matrix = runner.embed(["bir", "iki"], batch_size=8)
    assert matrix.shape == (2, 3)
    np.testing.assert_allclose(matrix[0], [0.6, 0.0, 0.8])
    assert runner.name == "hf:fake/model"
    assert runner.dimension == 3


def test_sentence_transformer_runner_reshapes_single_vector() -> None:
    with fake_sentence_transformers_module(lambda texts, **kwargs: np.array([1.0, 0.0, 0.0])):
        runner = SentenceTransformerRunner("fake/model")
        matrix = runner.embed(["tek"])
    assert matrix.shape == (1, 3)


def test_sentence_transformer_runner_detects_count_mismatch() -> None:
    with fake_sentence_transformers_module(lambda texts, **kwargs: np.zeros((1, 3))):
        runner = SentenceTransformerRunner("fake/model")
        with pytest.raises(RunnerError, match="returned 1 vectors"):
            runner.embed(["a", "b"])


def test_sentence_transformer_runner_rejects_non_finite_vectors() -> None:
    with fake_sentence_transformers_module(
        lambda texts, **kwargs: np.array([[float("nan"), 1.0, 0.0] for _ in texts])
    ):
        runner = SentenceTransformerRunner("fake/model")
        with pytest.raises(RunnerError, match="non-finite"):
            runner.embed(["a"])


def test_sentence_transformer_runner_wraps_backend_errors() -> None:
    def failing(texts: list[str], **kwargs: object) -> np.ndarray:
        raise RuntimeError("CUDA out of memory")

    with fake_sentence_transformers_module(failing):
        runner = SentenceTransformerRunner("fake/model")
        with pytest.raises(RunnerError, match="CUDA out of memory"):
            runner.embed(["a"])


def test_sentence_transformer_runner_requires_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(__import__("sys").modules, "sentence_transformers", None)
    with pytest.raises(RunnerUnavailable, match="pip install"):
        SentenceTransformerRunner("fake/model")


def test_sentence_transformer_runner_rejects_empty_model_name() -> None:
    with pytest.raises(ValueError, match="model_name"):
        SentenceTransformerRunner("   ")


def test_api_runner_success_and_payload(api_server) -> None:  # type: ignore[no-untyped-def]
    runner = OpenAICompatRunner("nomic-embed-text", base_url=api_server.url)
    matrix = runner.embed(["birinci metin", "ikinci metin"])
    assert matrix.shape == (2, 8)
    assert runner.dimension == 8
    np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), np.ones(2), atol=1e-9)
    request = api_server.requests[0]
    assert request["path"] == "/v1/embeddings"
    assert request["payload"]["model"] == "nomic-embed-text"
    assert request["payload"]["input"] == ["birinci metin", "ikinci metin"]
    assert "Authorization" not in request["headers"]


def test_api_runner_sends_bearer_token(api_server) -> None:  # type: ignore[no-untyped-def]
    runner = OpenAICompatRunner("m", base_url=api_server.url, api_key=FAKE_BEARER)
    runner.embed(["x"])
    assert api_server.requests[0]["headers"]["Authorization"] == f"Bearer {FAKE_BEARER}"


def test_api_runner_batches_requests(api_server) -> None:  # type: ignore[no-untyped-def]
    runner = OpenAICompatRunner("m", base_url=api_server.url, max_batch=2)
    matrix = runner.embed(["a", "b", "c", "d", "e"], batch_size=10)
    assert matrix.shape == (5, 8)
    assert len(api_server.requests) == 3
    assert [len(r["payload"]["input"]) for r in api_server.requests] == [2, 2, 1]


def test_api_runner_reorders_by_index(api_server) -> None:  # type: ignore[no-untyped-def]
    def reversed_response(payload: dict, path: str) -> tuple[int, dict]:
        inputs = payload["input"]
        data = [
            {"index": index, "embedding": [float(index + 1), 1.0, 0.0, 0.0]}
            for index in range(len(inputs))
        ]
        return 200, {"data": list(reversed(data))}

    api_server.respond_with(reversed_response)
    runner = OpenAICompatRunner("m", base_url=api_server.url)
    matrix = runner.embed(["a", "b", "c"])
    np.testing.assert_allclose(
        matrix[:, 1],
        [1 / np.sqrt(2), 1 / np.sqrt(5), 1 / np.sqrt(10)],
        atol=1e-9,
    )


def test_api_runner_empty_input_makes_no_request(api_server) -> None:  # type: ignore[no-untyped-def]
    runner = OpenAICompatRunner("m", base_url=api_server.url)
    matrix = runner.embed([])
    assert matrix.shape == (0, 0)
    assert api_server.requests == []


def test_api_runner_reports_http_errors(api_server) -> None:  # type: ignore[no-untyped-def]
    api_server.respond_with(lambda payload, path: (500, {"error": "model not found"}))
    runner = OpenAICompatRunner("m", base_url=api_server.url)
    with pytest.raises(RunnerError, match="HTTP 500"):
        runner.embed(["x"])


def test_api_runner_reports_invalid_json(api_server) -> None:  # type: ignore[no-untyped-def]
    api_server.respond_with(lambda payload, path: (200, b"<html>not json</html>"))
    runner = OpenAICompatRunner("m", base_url=api_server.url)
    with pytest.raises(RunnerError, match="invalid JSON"):
        runner.embed(["x"])


def test_api_runner_reports_missing_data(api_server) -> None:  # type: ignore[no-untyped-def]
    api_server.respond_with(lambda payload, path: (200, {"object": "list"}))
    runner = OpenAICompatRunner("m", base_url=api_server.url)
    with pytest.raises(RunnerError, match="no 'data'"):
        runner.embed(["x"])


def test_api_runner_reports_count_mismatch(api_server) -> None:  # type: ignore[no-untyped-def]
    api_server.respond_with(
        lambda payload, path: (200, {"data": [{"index": 0, "embedding": [1.0]}]})
    )
    runner = OpenAICompatRunner("m", base_url=api_server.url)
    with pytest.raises(RunnerError, match="1 embeddings for 2 inputs"):
        runner.embed(["a", "b"])


def test_api_runner_reports_empty_embedding(api_server) -> None:  # type: ignore[no-untyped-def]
    api_server.respond_with(lambda payload, path: (200, {"data": [{"index": 0, "embedding": []}]}))
    runner = OpenAICompatRunner("m", base_url=api_server.url)
    with pytest.raises(RunnerError, match="empty embedding"):
        runner.embed(["a"])


def test_api_runner_rejects_non_object_items(api_server) -> None:  # type: ignore[no-untyped-def]
    api_server.respond_with(
        lambda payload, path: (200, {"data": [{"index": 0, "embedding": [1.0]}, "nope"]})
    )
    runner = OpenAICompatRunner("m", base_url=api_server.url)
    with pytest.raises(RunnerError, match="non-object entries"):
        runner.embed(["a", "b"])


def test_api_runner_rejects_duplicate_indexes(api_server) -> None:  # type: ignore[no-untyped-def]
    api_server.respond_with(
        lambda payload, path: (
            200,
            {
                "data": [
                    {"index": 0, "embedding": [1.0]},
                    {"index": 0, "embedding": [2.0]},
                ]
            },
        )
    )
    runner = OpenAICompatRunner("m", base_url=api_server.url)
    with pytest.raises(RunnerError, match="unique and cover"):
        runner.embed(["a", "b"])


def test_api_runner_rejects_missing_indexes(api_server) -> None:  # type: ignore[no-untyped-def]
    api_server.respond_with(
        lambda payload, path: (
            200,
            {
                "data": [
                    {"index": 0, "embedding": [1.0]},
                    {"index": 2, "embedding": [2.0]},
                ]
            },
        )
    )
    runner = OpenAICompatRunner("m", base_url=api_server.url)
    with pytest.raises(RunnerError, match="unique and cover"):
        runner.embed(["a", "b"])


def test_api_runner_rejects_missing_index_field(api_server) -> None:  # type: ignore[no-untyped-def]
    api_server.respond_with(
        lambda payload, path: (
            200,
            {
                "data": [
                    {"embedding": [1.0]},
                    {"index": 1, "embedding": [2.0]},
                ]
            },
        )
    )
    runner = OpenAICompatRunner("m", base_url=api_server.url)
    with pytest.raises(RunnerError, match="integer 'index'"):
        runner.embed(["a", "b"])


def test_api_runner_rejects_non_numeric_embedding(api_server) -> None:  # type: ignore[no-untyped-def]
    api_server.respond_with(
        lambda payload, path: (200, {"data": [{"index": 0, "embedding": ["1.0", 2.0]}]})
    )
    runner = OpenAICompatRunner("m", base_url=api_server.url)
    with pytest.raises(RunnerError, match="non-numeric"):
        runner.embed(["a"])


def test_api_runner_rejects_nan_embedding(api_server) -> None:  # type: ignore[no-untyped-def]
    api_server.respond_with(
        lambda payload, path: (
            200,
            {"data": [{"index": 0, "embedding": [float("nan"), 1.0]}]},
        )
    )
    runner = OpenAICompatRunner("m", base_url=api_server.url)
    with pytest.raises(RunnerError, match="NaN or infinity"):
        runner.embed(["a"])


def test_api_runner_rejects_ragged_embeddings(api_server) -> None:  # type: ignore[no-untyped-def]
    api_server.respond_with(
        lambda payload, path: (
            200,
            {
                "data": [
                    {"index": 0, "embedding": [1.0, 2.0]},
                    {"index": 1, "embedding": [1.0, 2.0, 3.0]},
                ]
            },
        )
    )
    runner = OpenAICompatRunner("m", base_url=api_server.url)
    with pytest.raises(RunnerError, match="inconsistent dimensions"):
        runner.embed(["a", "b"])


def test_api_runner_detects_dimension_change(api_server) -> None:  # type: ignore[no-untyped-def]
    sizes = iter([4, 6])

    def variable_dimension(payload: dict, path: str) -> tuple[int, dict]:
        size = next(sizes)
        data = [
            {"index": index, "embedding": [1.0] * size} for index, _ in enumerate(payload["input"])
        ]
        return 200, {"data": data}

    api_server.respond_with(variable_dimension)
    runner = OpenAICompatRunner("m", base_url=api_server.url, max_batch=1)
    with pytest.raises(RunnerError, match="changed embedding size"):
        runner.embed(["a", "b"])


def test_api_runner_unreachable_endpoint() -> None:
    runner = OpenAICompatRunner("m", base_url="http://127.0.0.1:1/v1", timeout=1.0)
    with pytest.raises(RunnerError, match="could not reach"):
        runner.embed(["x"])


def test_api_runner_validates_configuration(api_server) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match="base_url"):
        OpenAICompatRunner("m", base_url="localhost:11434")
    with pytest.raises(ValueError, match="model_name"):
        OpenAICompatRunner("  ", base_url=api_server.url)
    with pytest.raises(ValueError, match="timeout"):
        OpenAICompatRunner("m", base_url=api_server.url, timeout=0)
    with pytest.raises(ValueError, match="max_batch"):
        OpenAICompatRunner("m", base_url=api_server.url, max_batch=0)
    with pytest.raises(ValueError, match="batch_size"):
        OpenAICompatRunner("m", base_url=api_server.url).embed(["a"], batch_size=0)


def test_api_runner_endpoint_accepts_full_url() -> None:
    runner = OpenAICompatRunner("m", base_url="http://localhost:11434/v1/embeddings")
    assert runner.endpoint == "http://localhost:11434/v1/embeddings"
    assert runner.dimension is None


def test_api_runner_reads_env_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TR_EVAL_API_URL", "http://127.0.0.1:9999/v1")
    monkeypatch.setenv("TR_EVAL_API_KEY", "env-key-not-real")
    runner = OpenAICompatRunner("m")
    assert runner.endpoint == "http://127.0.0.1:9999/v1/embeddings"
    assert runner._api_key == "env-key-not-real"


def test_load_runner_factory_paths() -> None:
    assert isinstance(load_runner("hashing"), HashingRunner)
    assert load_runner("hashing:64").dimension == 64
    with pytest.raises(ValueError, match="dimension"):
        load_runner("hashing:abc")
    with pytest.raises(ValueError, match="must not be empty"):
        load_runner("  ")
    with pytest.raises(ValueError, match="unknown backend"):
        load_runner("hashing", backend="grpc")
    with pytest.raises(ValueError, match="api-url"):
        load_runner("http://localhost:11434/v1", backend="api")
    api = load_runner("nomic", backend="api", api_url="http://127.0.0.1:11434/v1")
    assert isinstance(api, OpenAICompatRunner)
    assert api.name.startswith("api:nomic@")


def test_load_runner_auto_prefers_hf_for_plain_names() -> None:
    with fake_sentence_transformers_module(lambda texts, **kwargs: np.ones((len(texts), 2))):
        runner = load_runner("some/model")
    assert isinstance(runner, SentenceTransformerRunner)
    assert runner.name == "hf:some/model"


def test_keyword_runner_helper_normalizes() -> None:
    runner = KeywordRunner({"a": [3.0, 4.0, 0.0]})
    matrix = runner.embed(["a"])
    np.testing.assert_allclose(matrix[0], [0.6, 0.8, 0.0])
    assert json.dumps(runner.name) == '"fake:keyword"'
