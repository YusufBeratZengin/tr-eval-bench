"""Metric correctness tests with hand-computed expectations."""

from __future__ import annotations

import math

import numpy as np
import pytest
from tr_eval_bench.metrics import (
    cosine_similarity,
    dcg_at_k,
    mean,
    mrr_at_k,
    ndcg_at_k,
    pearson,
    rankdata_average,
    recall_at_k,
    spearman,
)


def test_rankdata_average_handles_ties() -> None:
    ranks = rankdata_average([10.0, 20.0, 20.0, 30.0])
    assert ranks.tolist() == [1.0, 2.5, 2.5, 4.0]


def test_rankdata_rejects_nan() -> None:
    with pytest.raises(ValueError, match="NaN"):
        rankdata_average([1.0, float("nan")])


def test_spearman_perfect_and_reversed() -> None:
    a = [1, 2, 3, 4, 5]
    assert spearman(a, [10, 20, 30, 40, 50]) == pytest.approx(1.0)
    assert spearman(a, [5, 4, 3, 2, 1]) == pytest.approx(-1.0)


def test_spearman_matches_manual_rank_correlation() -> None:
    # Ranks: x -> 1,2,3,4 ; y -> 1,3,4,2  => Pearson of ranks = 0.8
    assert spearman([1, 2, 3, 4], [1, 2, 4, 3]) == pytest.approx(0.8)


def test_spearman_with_ties_is_between_bounds() -> None:
    value = spearman([1, 1, 2, 3], [1, 2, 2, 4])
    assert -1.0 <= value <= 1.0
    assert value == pytest.approx(
        pearson(rankdata_average([1, 1, 2, 3]), rankdata_average([1, 2, 2, 4]))
    )


def test_spearman_constant_series_is_nan() -> None:
    assert math.isnan(spearman([1, 1, 1], [1, 2, 3]))


def test_pearson_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        pearson([1, 2, 3], [1, 2])


def test_pearson_single_observation_raises() -> None:
    with pytest.raises(ValueError, match="two observations"):
        pearson([1.0], [2.0])


def test_dcg_at_k_matches_definition() -> None:
    # DCG = 3/log2(2) + 2/log2(3) + 0/log2(4) + 1/log2(5)
    expected = 3 / 1 + 2 / math.log2(3) + 0 + 1 / math.log2(5)
    assert dcg_at_k([3, 2, 0, 1], k=4) == pytest.approx(expected)


def test_ndcg_perfect_ranking_is_one() -> None:
    assert ndcg_at_k([3, 2, 1, 0], [3, 2, 1, 0], k=10) == pytest.approx(1.0)


def test_ndcg_hand_computed() -> None:
    ranked = [0, 3, 0, 1]  # gains at positions 2 and 4
    ideal = [3, 1]
    dcg = 3 / math.log2(3) + 1 / math.log2(5)
    idcg = 3 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(ranked, ideal, k=10) == pytest.approx(dcg / idcg)


def test_ndcg_without_relevant_documents_is_zero() -> None:
    assert ndcg_at_k([0, 0, 0], [0, 0], k=3) == 0.0


def test_ndcg_respects_cutoff() -> None:
    # The only relevant document sits at rank 2, outside k=1.
    assert ndcg_at_k([0, 5], [5], k=1) == 0.0


def test_ndcg_rejects_negative_ideal() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        ndcg_at_k([1, 0], [1, -1], k=5)


def test_mrr_first_relevant_position() -> None:
    assert mrr_at_k([0, 0, 1, 0], k=10) == pytest.approx(1 / 3)
    assert mrr_at_k([1, 0], k=10) == pytest.approx(1.0)
    assert mrr_at_k([0, 0, 0], k=10) == 0.0


def test_mrr_cutoff_excludes_late_hit() -> None:
    assert mrr_at_k([0, 0, 1], k=2) == 0.0


def test_recall_counts_relevant_hits() -> None:
    assert recall_at_k([1, 0, 1, 0], total_relevant=4, k=10) == pytest.approx(0.5)
    assert recall_at_k([0, 0], total_relevant=2, k=10) == 0.0
    assert recall_at_k([1], total_relevant=0, k=10) == 0.0


def test_rejects_invalid_k() -> None:
    for func in (
        lambda: dcg_at_k([1], k=0),
        lambda: ndcg_at_k([1], [1], k=0),
        lambda: mrr_at_k([1], k=-1),
        lambda: recall_at_k([1], 1, k=0),
    ):
        with pytest.raises(ValueError, match="k must be positive"):
            func()


def test_cosine_similarity_normalizes_rows() -> None:
    left = np.array([[2.0, 0.0], [0.0, 5.0]])
    right = np.array([[1.0, 1.0]])
    matrix = cosine_similarity(left, right)
    assert matrix.shape == (2, 1)
    assert matrix[0, 0] == pytest.approx(1 / math.sqrt(2))
    assert matrix[1, 0] == pytest.approx(1 / math.sqrt(2))


def test_cosine_similarity_zero_row_is_zero() -> None:
    matrix = cosine_similarity(np.zeros((1, 3)), np.ones((1, 3)))
    assert matrix[0, 0] == 0.0


def test_cosine_similarity_dimension_mismatch() -> None:
    with pytest.raises(ValueError, match="dimension mismatch"):
        cosine_similarity(np.ones((1, 2)), np.ones((1, 3)))


def test_mean_rejects_empty_sequence() -> None:
    with pytest.raises(ValueError, match="empty"):
        mean([])
