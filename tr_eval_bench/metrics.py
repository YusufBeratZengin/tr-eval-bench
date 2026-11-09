"""Metric implementations used by every task.

The module is deliberately dependency-light: only NumPy is used, and every
formula follows the standard definitions (Spearman rank correlation, DCG/NDCG
with linear gains, reciprocal rank).  All functions are pure and covered by
``tests/test_metrics.py`` with hand-computed expectations.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def _as_vector(values: Sequence[float] | FloatArray, name: str) -> FloatArray:
    """Convert ``values`` to a finite 1-D float64 array or raise ``ValueError``."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


def rankdata_average(values: Sequence[float] | FloatArray) -> FloatArray:
    """Ranks of ``values`` (1-based) with tied values sharing their average rank."""
    array = _as_vector(values, "values")
    order = np.argsort(array, kind="mergesort")
    ordered = array[order]
    ranks = np.empty(array.size, dtype=np.float64)
    ranks[order] = np.arange(1, array.size + 1, dtype=np.float64)
    start = 0
    while start < array.size:
        stop = start
        while stop + 1 < array.size and ordered[stop + 1] == ordered[start]:
            stop += 1
        if stop > start:
            ranks[order[start : stop + 1]] = (start + stop + 2) / 2.0
        start = stop + 1
    return ranks


def pearson(x: Sequence[float] | FloatArray, y: Sequence[float] | FloatArray) -> float:
    """Pearson product-moment correlation; ``nan`` when either input is constant."""
    a = _as_vector(x, "x")
    b = _as_vector(y, "y")
    if a.size != b.size:
        raise ValueError(f"length mismatch: {a.size} != {b.size}")
    if a.size < 2:
        raise ValueError("pearson correlation needs at least two observations")
    a_centered = a - a.mean()
    b_centered = b - b.mean()
    denominator = float(np.sqrt(np.dot(a_centered, a_centered) * np.dot(b_centered, b_centered)))
    if denominator == 0.0:
        return float("nan")
    return float(np.dot(a_centered, b_centered) / denominator)


def spearman(x: Sequence[float] | FloatArray, y: Sequence[float] | FloatArray) -> float:
    """Spearman rank correlation (Pearson correlation of average ranks)."""
    return pearson(rankdata_average(x), rankdata_average(y))


def dcg_at_k(relevances: Sequence[float] | FloatArray, k: int = 10) -> float:
    """Discounted cumulative gain with linear gains, truncated to ``k`` items."""
    if k <= 0:
        raise ValueError("k must be positive")
    array = np.asarray(relevances, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("relevances must be one-dimensional")
    if array.size == 0:
        return 0.0
    if not np.all(np.isfinite(array)):
        raise ValueError("relevances contain NaN or infinite values")
    gains = array[:k]
    discounts = np.log2(np.arange(2, gains.size + 2, dtype=np.float64))
    return float(np.sum(gains / discounts))


def ndcg_at_k(
    ranked_relevances: Sequence[float] | FloatArray,
    ideal_relevances: Sequence[float] | FloatArray,
    k: int = 10,
) -> float:
    """Normalized DCG@k.

    ``ranked_relevances`` are the graded relevance labels in the order produced
    by the retriever, ``ideal_relevances`` are all labels for the query (any
    order; they are sorted descending to build the ideal ranking).  Returns
    ``0.0`` when the query has no positive relevance, which matches the
    convention used by MTEB/scikit-learn for empty ideal DCG.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    ideal = _as_vector(ideal_relevances, "ideal_relevances")
    if np.any(ideal < 0):
        raise ValueError("ideal_relevances must be non-negative")
    ideal_dcg = dcg_at_k(np.sort(ideal)[::-1], k=k)
    if ideal_dcg == 0.0:
        return 0.0
    score = dcg_at_k(ranked_relevances, k=k) / ideal_dcg
    return float(min(max(score, 0.0), 1.0))


def mrr_at_k(ranked_relevances: Sequence[float] | FloatArray, k: int = 10) -> float:
    """Reciprocal rank of the first relevant (label > 0) item within the top ``k``."""
    if k <= 0:
        raise ValueError("k must be positive")
    array = np.asarray(ranked_relevances, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("ranked_relevances must be one-dimensional")
    if array.size == 0:
        return 0.0
    if not np.all(np.isfinite(array)):
        raise ValueError("ranked_relevances contain NaN or infinite values")
    hits = np.flatnonzero(array[:k] > 0)
    if hits.size == 0:
        return 0.0
    return float(1.0 / (int(hits[0]) + 1))


def recall_at_k(
    ranked_relevances: Sequence[float] | FloatArray,
    total_relevant: int,
    k: int = 10,
) -> float:
    """Share of the relevant documents retrieved within the top ``k``."""
    if k <= 0:
        raise ValueError("k must be positive")
    if total_relevant < 0:
        raise ValueError("total_relevant must be non-negative")
    if total_relevant == 0:
        return 0.0
    array = np.asarray(ranked_relevances, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("ranked_relevances must be one-dimensional")
    found = int(np.count_nonzero(array[:k] > 0))
    return float(min(found / total_relevant, 1.0))


def cosine_similarity(
    left: npt.NDArray[np.floating],
    right: npt.NDArray[np.floating],
) -> npt.NDArray[np.float64]:
    """Row-wise cosine similarity matrix for two 2-D arrays.

    Rows are L2-normalized here, so callers may pass raw (unnormalized)
    embeddings.  Zero rows yield zero similarity instead of ``NaN``.
    """
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("cosine_similarity expects two 2-D arrays")
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"embedding dimension mismatch: {a.shape[1]} != {b.shape[1]}")
    a_norms = np.linalg.norm(a, axis=1, keepdims=True)
    b_norms = np.linalg.norm(b, axis=1, keepdims=True)
    a_norms[a_norms == 0.0] = 1.0
    b_norms[b_norms == 0.0] = 1.0
    return (a / a_norms) @ (b / b_norms).T


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean, raising ``ValueError`` for an empty sequence."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("mean of empty sequence")
    return float(array.mean())
