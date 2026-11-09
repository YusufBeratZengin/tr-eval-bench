"""Task-level tests: STS scoring, retrieval ranking and hybrid RRF/BM25."""

from __future__ import annotations

import math

import numpy as np
import pytest
from tr_eval_bench.datasets import Passage, Query, RagDataset, StsPair
from tr_eval_bench.tasks.hybrid import (
    BM25Index,
    fuse_score_matrix,
    rankings_from_scores,
    rrf_scores,
    tokenize,
)
from tr_eval_bench.tasks.retrieval import evaluate_retrieval
from tr_eval_bench.tasks.sts import evaluate_sts

from tests.conftest import KeywordRunner


def _sts_pairs() -> list[StsPair]:
    return [
        StsPair("a", "yakın", "benzer", 5.0, "test"),
        StsPair("b", "orta", "ilgili", 3.0, "test"),
        StsPair("c", "uzak", "alakasız", 0.0, "test"),
    ]


def test_sts_perfect_ordering_gives_spearman_one() -> None:
    runner = KeywordRunner(
        {
            "yakın": [1.0, 0.0, 0.0],
            "benzer": [1.0, 0.0, 0.0],
            "orta": [0.8, 0.6, 0.0],
            "ilgili": [0.6, 0.8, 0.0],  # cosine 0.96 with "orta", not a tie
            "uzak": [0.0, 0.0, 1.0],
            "alakasız": [0.0, 1.0, 0.0],
        }
    )
    result = evaluate_sts(runner, _sts_pairs())
    assert result.spearman == pytest.approx(1.0)
    assert result.n_pairs == 3
    assert result.elapsed_seconds >= 0.0


def test_sts_on_bundled_dataset_with_hashing_runner() -> None:
    from tr_eval_bench.runners import HashingRunner

    result = evaluate_sts(HashingRunner(128))
    assert result.n_pairs == 100
    assert -1.0 <= result.spearman <= 1.0


def test_sts_limit_and_validation() -> None:
    from tr_eval_bench.runners import HashingRunner

    limited = evaluate_sts(HashingRunner(64), limit=10)
    assert limited.n_pairs == 10
    with pytest.raises(ValueError, match="limit"):
        evaluate_sts(HashingRunner(64), limit=0)
    with pytest.raises(ValueError, match="batch_size"):
        evaluate_sts(HashingRunner(64), batch_size=0)
    with pytest.raises(ValueError, match="at least two pairs"):
        evaluate_sts(HashingRunner(64), _sts_pairs()[:1])
    with pytest.raises(ValueError, match="at least two pairs"):
        evaluate_sts(HashingRunner(64), _sts_pairs(), limit=1)


def test_sts_detects_shape_mismatch() -> None:
    class OddRunner:
        name = "odd"

        @property
        def dimension(self) -> None:
            return None

        def embed(self, texts, *, batch_size: int = 32):  # type: ignore[no-untyped-def]
            size = 3 if texts[0] in {"yakın", "orta", "uzak"} else 2
            return np.ones((len(texts), size))

    with pytest.raises(ValueError, match="shapes differ"):
        evaluate_sts(OddRunner(), _sts_pairs())  # type: ignore[arg-type]


def _tiny_dataset() -> RagDataset:
    corpus = (
        Passage("p1", "Birinci", "elma armut", "test"),
        Passage("p2", "İkinci", "araba motor", "test"),
        Passage("p3", "Üçüncü", "deniz kum", "test"),
    )
    queries = (
        Query("q1", "meyve?", (("p1", 1.0),), "test"),
        Query("q2", "taşıt?", (("p2", 1.0),), "test"),
        Query("q3", "kumsal?", (("p3", 2.0),), "test"),
    )
    return RagDataset(corpus=corpus, queries=queries)


def _tiny_runner() -> KeywordRunner:
    return KeywordRunner(
        {
            "elma armut": [1.0, 0.0, 0.0],
            "araba motor": [0.0, 1.0, 0.0],
            "deniz kum": [0.0, 0.0, 1.0],
            "meyve?": [1.0, 0.0, 0.0],
            "taşıt?": [0.0, 1.0, 0.0],
            "kumsal?": [0.0, 0.0, 1.0],
        }
    )


def test_retrieval_perfect_ranking() -> None:
    result = evaluate_retrieval(_tiny_runner(), _tiny_dataset())
    assert result.ndcg10 == pytest.approx(1.0)
    assert result.mrr10 == pytest.approx(1.0)
    assert result.recall10 == pytest.approx(1.0)
    assert result.n_queries == 3
    assert result.corpus_size == 3


def test_retrieval_second_rank_halves_reciprocal_rank() -> None:
    runner = KeywordRunner(
        {
            "elma armut": [0.0, 1.0, 0.0],
            "araba motor": [1.0, 0.0, 0.0],
            "deniz kum": [0.0, 0.0, 1.0],
            "meyve?": [1.0, 0.0, 0.0],  # p2 first, p1 second -> RR 0.5
            "taşıt?": [1.0, 0.0, 0.0],  # p2 first -> RR 1.0
            "kumsal?": [0.0, 0.0, 1.0],  # p3 first -> RR 1.0
        }
    )
    result = evaluate_retrieval(runner, _tiny_dataset())
    assert result.mrr10 == pytest.approx((0.5 + 1.0 + 1.0) / 3)
    # A 3-passage corpus always fits in the top 10, so recall stays perfect.
    assert result.recall10 == pytest.approx(1.0)
    assert result.ndcg10 < 1.0


def test_retrieval_with_hybrid_fusion() -> None:
    result = evaluate_retrieval(_tiny_runner(), _tiny_dataset(), hybrid=True)
    assert result.hybrid_ndcg10 is not None
    assert result.hybrid_mrr10 is not None
    assert result.hybrid_recall10 is not None
    assert 0.0 <= result.hybrid_mrr10 <= 1.0
    # Fusion can only reorder the dense result, never invent a relevant document.
    assert result.hybrid_recall10 <= 1.0


def test_retrieval_limit_keeps_first_queries() -> None:
    result = evaluate_retrieval(_tiny_runner(), _tiny_dataset(), limit=2)
    assert result.n_queries == 2


def test_retrieval_validation_errors() -> None:
    dataset = _tiny_dataset()
    with pytest.raises(ValueError, match="k must be positive"):
        evaluate_retrieval(_tiny_runner(), dataset, k=0)
    with pytest.raises(ValueError, match="batch_size"):
        evaluate_retrieval(_tiny_runner(), dataset, batch_size=0)
    with pytest.raises(ValueError, match="limit"):
        evaluate_retrieval(_tiny_runner(), dataset, limit=0)
    empty_corpus = RagDataset(corpus=(), queries=dataset.queries)
    with pytest.raises(ValueError, match="corpus is empty"):
        evaluate_retrieval(_tiny_runner(), empty_corpus)
    no_queries = RagDataset(corpus=dataset.corpus, queries=())
    with pytest.raises(ValueError, match="at least one query"):
        evaluate_retrieval(_tiny_runner(), no_queries)


def test_retrieval_detects_wrong_embedding_count() -> None:
    class ShortRunner:
        name = "short"

        @property
        def dimension(self) -> int:
            return 3

        def embed(self, texts, *, batch_size: int = 32):  # type: ignore[no-untyped-def]
            return np.ones((len(texts) - 1, 3))

    with pytest.raises(ValueError, match="unexpected number"):
        evaluate_retrieval(ShortRunner(), _tiny_dataset())  # type: ignore[arg-type]


def test_retrieval_detects_dimension_mismatch() -> None:
    class MismatchRunner:
        name = "mismatch"
        calls = 0

        @property
        def dimension(self) -> None:
            return None

        def embed(self, texts, *, batch_size: int = 32):  # type: ignore[no-untyped-def]
            MismatchRunner.calls += 1
            size = 3 if MismatchRunner.calls == 1 else 4
            return np.ones((len(texts), size))

    with pytest.raises(ValueError, match="different dimensions"):
        evaluate_retrieval(MismatchRunner(), _tiny_dataset())  # type: ignore[arg-type]


def test_tokenize_is_turkish_aware() -> None:
    assert tokenize("İSTANBUL'da güzel bir gün!") == ["istanbul", "da", "güzel", "bir", "gün"]
    assert tokenize("a b c") == []


def test_bm25_scores_follow_the_formula() -> None:
    index = BM25Index(["elma armut", "araba"])
    scores = index.scores("elma")
    idf = math.log(1 + (2 - 1 + 0.5) / (1 + 0.5))
    expected = idf * 2.5 / (1 + 1.5 * (1 - 0.75 + 0.75 * 2 / 1.5))
    assert scores[0] == pytest.approx(expected)
    assert scores[1] == 0.0


def test_bm25_batch_and_empty_query() -> None:
    index = BM25Index(["elma armut", "araba motor", "deniz kum"])
    matrix = index.score_batch(["elma", "araba", "zzz"])
    assert matrix.shape == (3, 3)
    assert matrix[0, 0] > 0
    assert matrix[1, 1] > 0
    assert np.all(matrix[2] == 0.0)
    assert index.vocabulary_size == 6
    assert index.score_batch([]).shape == (0, 3)


def test_bm25_validation() -> None:
    with pytest.raises(ValueError, match="at least one document"):
        BM25Index([])
    with pytest.raises(ValueError, match="k1"):
        BM25Index(["a b"], k1=0)
    with pytest.raises(ValueError, match="b must be"):
        BM25Index(["a b"], b=1.5)


def test_rankings_from_scores_is_stable() -> None:
    rankings = rankings_from_scores(np.array([[0.1, 0.9, 0.5], [1.0, 0.0, 0.0]]))
    assert rankings == [[1, 2, 0], [0, 1, 2]]
    with pytest.raises(ValueError, match="2-D"):
        rankings_from_scores(np.array([1.0, 2.0]))


def test_rrf_scores_hand_computed() -> None:
    fused = rrf_scores([[0, 1], [1, 0]], n_docs=2, k=60)
    assert fused[0] == pytest.approx(1 / 61 + 1 / 62)
    assert fused[1] == pytest.approx(1 / 62 + 1 / 61)
    top = rrf_scores([[0, 1]], n_docs=2, k=60)
    assert top[0] == pytest.approx(1 / 61)
    assert top[1] == pytest.approx(1 / 62)
    assert top[0] > top[1]


def test_rrf_validation() -> None:
    with pytest.raises(ValueError, match="n_docs"):
        rrf_scores([[0]], n_docs=0)
    with pytest.raises(ValueError, match="k must be positive"):
        rrf_scores([[0]], n_docs=1, k=0)
    with pytest.raises(ValueError, match="at least one ranking"):
        rrf_scores([], n_docs=1)
    with pytest.raises(ValueError, match="weights must match"):
        rrf_scores([[0]], n_docs=1, weights=[1.0, 2.0])
    with pytest.raises(ValueError, match="outside"):
        rrf_scores([[5]], n_docs=1)
    with pytest.raises(ValueError, match="non-negative"):
        rrf_scores([[0]], n_docs=1, weights=[-1.0])


def test_fuse_score_matrix_prefers_agreement() -> None:
    scores = np.array([[0.9, 0.1, 0.0], [0.8, 0.2, 0.0]])
    fused = fuse_score_matrix(scores)
    assert int(np.argmax(fused)) == 0
    with pytest.raises(ValueError, match="shape"):
        fuse_score_matrix(np.array([1.0, 2.0]))


def test_fuse_score_matrix_skips_silent_systems() -> None:
    """A zero BM25 row must not outvote a confident dense ranking."""
    dense_only = np.array([[0.0, 0.9, 0.1]])
    with_silent = np.vstack([dense_only, np.zeros(3)])
    np.testing.assert_allclose(fuse_score_matrix(with_silent), fuse_score_matrix(dense_only))
    np.testing.assert_allclose(fuse_score_matrix(np.zeros((2, 3))), np.zeros(3))
    np.testing.assert_allclose(fuse_score_matrix(np.zeros((0, 3))), np.zeros(3))
    with pytest.raises(ValueError, match="weights must match"):
        fuse_score_matrix(np.zeros((2, 3)), weights=[1.0])
