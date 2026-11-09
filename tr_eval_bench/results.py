"""Result objects shared by the library API, the CLI and the reporters."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

SCHEMA_VERSION = "1.1"

_NON_FINITE_REASON = "not defined for these inputs (constant labels or too few samples)"

METRIC_LABELS: dict[str, str] = {
    "sts_spearman": "Spearman",
    "sts_pearson": "Pearson",
    "retrieval_ndcg10": "NDCG@10",
    "retrieval_mrr10": "MRR@10",
    "retrieval_recall10": "Recall@10",
    "hybrid_ndcg10": "Hybrid NDCG@10",
    "hybrid_mrr10": "Hybrid MRR@10",
    "hybrid_recall10": "Hybrid Recall@10",
}


@dataclass(frozen=True)
class EvalResults:
    """Scores for one model on the bundled mini benchmark.

    Attribute names follow the public API documented in the README, e.g.
    ``results.sts_spearman`` and ``results.retrieval_mrr10``.  A metric is
    ``None`` when its task was not run.
    """

    model: str
    tasks: tuple[str, ...] = ()
    sts_spearman: float | None = None
    sts_pearson: float | None = None
    retrieval_ndcg10: float | None = None
    retrieval_mrr10: float | None = None
    retrieval_recall10: float | None = None
    hybrid_ndcg10: float | None = None
    hybrid_mrr10: float | None = None
    hybrid_recall10: float | None = None
    n_sts_pairs: int = 0
    n_queries: int = 0
    corpus_size: int = 0
    elapsed_seconds: float = 0.0
    dataset_version: str = ""
    errors: tuple[str, ...] = field(default_factory=tuple)

    def metric(self, name: str) -> float | None:
        """Return a metric by its snake_case key (``KeyError`` for unknown keys)."""
        if name not in METRIC_LABELS:
            raise KeyError(f"unknown metric {name!r}; known metrics: {sorted(METRIC_LABELS)}")
        value = getattr(self, name)
        return None if value is None else float(value)

    def metrics(self) -> dict[str, float]:
        """All metrics with a finite value, keyed by snake_case name.

        ``NaN`` (an undefined correlation, e.g. when every gold label is
        identical) is deliberately excluded so the payload is strict JSON.
        """
        return {
            name: float(value)
            for name in METRIC_LABELS
            if (value := getattr(self, name)) is not None and math.isfinite(value)
        }

    def undefined_metrics(self) -> dict[str, str]:
        """Metrics that were computed but are non-finite, with the reason."""
        return {
            name: _NON_FINITE_REASON
            for name in METRIC_LABELS
            if (value := getattr(self, name)) is not None and not math.isfinite(value)
        }

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable payload (stable schema, see ``SCHEMA_VERSION``)."""
        return {
            "schema_version": SCHEMA_VERSION,
            "model": self.model,
            "dataset_version": self.dataset_version,
            "tasks": list(self.tasks),
            "metrics": self.metrics(),
            "undefined_metrics": self.undefined_metrics(),
            "counts": {
                "sts_pairs": self.n_sts_pairs,
                "retrieval_queries": self.n_queries,
                "corpus_passages": self.corpus_size,
            },
            "elapsed_seconds": round(float(self.elapsed_seconds), 3),
            "errors": list(self.errors),
        }


def _finite_or_none(value: float | None) -> float | None:
    """Return ``value`` when finite, else ``None`` (strict-JSON safe)."""
    if value is None or not math.isfinite(value):
        return None
    return float(value)


@dataclass(frozen=True)
class MetricDelta:
    """Difference between two models on a single metric."""

    metric: str
    baseline: float | None
    candidate: float | None

    @property
    def delta(self) -> float | None:
        if self.baseline is None or self.candidate is None:
            return None
        return self.candidate - self.baseline

    @property
    def status(self) -> str:
        """``"ok"``, ``"missing"`` (one side absent) or ``"undefined"`` (NaN)."""
        if self.baseline is None and self.candidate is None:
            return "missing"
        if self.baseline is None or self.candidate is None:
            return "missing"
        if not math.isfinite(self.baseline) or not math.isfinite(self.candidate):
            return "undefined"
        return "ok"

    @property
    def winner(self) -> str:
        """``"candidate"``, ``"baseline"`` or ``"tie"`` (``"n/a"`` when missing/NaN)."""
        delta = self.delta
        if delta is None or not math.isfinite(delta):
            return "n/a"
        if abs(delta) < 1e-12:
            return "tie"
        return "candidate" if delta > 0 else "baseline"


@dataclass(frozen=True)
class DiffReport:
    """Side-by-side comparison of a baseline and a candidate model."""

    baseline: EvalResults
    candidate: EvalResults
    rows: tuple[MetricDelta, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "baseline": self.baseline.to_dict(),
            "candidate": self.candidate.to_dict(),
            "deltas": [
                {
                    "metric": row.metric,
                    "baseline": _finite_or_none(row.baseline),
                    "candidate": _finite_or_none(row.candidate),
                    "delta": _finite_or_none(row.delta),
                    "winner": row.winner,
                    "status": row.status,
                }
                for row in self.rows
            ],
        }
