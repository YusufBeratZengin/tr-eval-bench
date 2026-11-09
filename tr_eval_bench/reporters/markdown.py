"""Markdown rendering: single-model report and leaderboard table."""

from __future__ import annotations

import math
from collections.abc import Sequence

from tr_eval_bench.results import EvalResults

_LEADERBOARD_METRICS = (
    ("sts_spearman", "STS Spearman"),
    ("retrieval_ndcg10", "NDCG@10"),
    ("retrieval_mrr10", "MRR@10"),
)


def _fmt(value: float | None) -> str:
    if value is None:
        return "-"
    if math.isnan(value):
        return "nan"
    return f"{value:.4f}"


def render_markdown(result: EvalResults) -> str:
    """Single-model report with metadata and a metric table."""
    lines = [
        f"# tr-eval-bench · `{result.model}`",
        "",
        f"Dataset: `{result.dataset_version}`  ",
        f"Elapsed: {result.elapsed_seconds:.2f}s  ",
        f"Counts: {result.n_sts_pairs} STS pairs, {result.n_queries} retrieval queries, "
        f"{result.corpus_size} passages",
        "",
        "| Task | Metric | Score |",
        "| --- | --- | ---: |",
    ]
    if result.sts_spearman is not None:
        lines.append(f"| STS-TR | Spearman | {_fmt(result.sts_spearman)} |")
        lines.append(f"| STS-TR | Pearson | {_fmt(result.sts_pearson)} |")
    if result.retrieval_ndcg10 is not None:
        lines.append(f"| Retrieval-TR | NDCG@10 | {_fmt(result.retrieval_ndcg10)} |")
        lines.append(f"| Retrieval-TR | MRR@10 | {_fmt(result.retrieval_mrr10)} |")
        lines.append(f"| Retrieval-TR | Recall@10 | {_fmt(result.retrieval_recall10)} |")
    if result.hybrid_ndcg10 is not None:
        lines.append(f"| Hybrid (dense+BM25, RRF) | NDCG@10 | {_fmt(result.hybrid_ndcg10)} |")
        lines.append(f"| Hybrid (dense+BM25, RRF) | MRR@10 | {_fmt(result.hybrid_mrr10)} |")
        lines.append(f"| Hybrid (dense+BM25, RRF) | Recall@10 | {_fmt(result.hybrid_recall10)} |")
    lines.append("")
    if result.errors:
        lines.append("> Warnings: " + "; ".join(result.errors))
        lines.append("")
    lines.append(
        "_Scores come from the bundled mini sets (100 STS pairs, 100 queries). "
        "They are a fast model-selection signal, not an official MTEB ranking._"
    )
    return "\n".join(lines)


def render_leaderboard(results: Sequence[EvalResults]) -> str:
    """Comparative Markdown table suitable for a README or HF Space card."""
    if not results:
        raise ValueError("render_leaderboard needs at least one result")
    header = "| Model | " + " | ".join(label for _, label in _LEADERBOARD_METRICS) + " |"
    separator = "| --- | " + " | ".join("---:" for _ in _LEADERBOARD_METRICS) + " |"
    lines = [header, separator]
    for result in results:
        cells = [_fmt(result.metric(name)) for name, _ in _LEADERBOARD_METRICS]
        lines.append(f"| `{result.model}` | " + " | ".join(cells) + " |")
    return "\n".join(lines)
