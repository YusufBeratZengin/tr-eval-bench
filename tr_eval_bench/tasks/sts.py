"""Semantic textual similarity task (Spearman rank correlation)."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from tr_eval_bench.datasets import StsPair, load_sts_mini
from tr_eval_bench.metrics import pearson, spearman
from tr_eval_bench.runners import EmbeddingRunner


@dataclass(frozen=True)
class StsResult:
    """Outcome of the STS task."""

    spearman: float
    pearson: float
    n_pairs: int
    elapsed_seconds: float


def evaluate_sts(
    runner: EmbeddingRunner,
    pairs: Sequence[StsPair] | None = None,
    *,
    batch_size: int = 32,
    limit: int | None = None,
) -> StsResult:
    """Score graded sentence pairs with the cosine of their embeddings.

    ``limit`` keeps the first N pairs (dataset order) for quick smoke runs.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    dataset = tuple(pairs) if pairs is not None else load_sts_mini()
    if len(dataset) < 2:
        raise ValueError("STS evaluation needs at least two pairs")
    if limit is not None:
        dataset = dataset[:limit]
        if len(dataset) < 2:
            raise ValueError("limit must leave at least two pairs")

    started = time.perf_counter()
    left = runner.embed([pair.sentence1 for pair in dataset], batch_size=batch_size)
    right = runner.embed([pair.sentence2 for pair in dataset], batch_size=batch_size)
    if left.shape != right.shape:
        raise ValueError(
            f"embedding shapes differ between the two sides: {left.shape} vs {right.shape}"
        )
    norms = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    norms[norms == 0.0] = 1.0
    predicted = [float(value) for value in np.sum(left * right, axis=1) / norms]
    gold = [pair.score for pair in dataset]
    elapsed = time.perf_counter() - started
    return StsResult(
        spearman=spearman(gold, predicted),
        pearson=pearson(gold, predicted),
        n_pairs=len(dataset),
        elapsed_seconds=elapsed,
    )
