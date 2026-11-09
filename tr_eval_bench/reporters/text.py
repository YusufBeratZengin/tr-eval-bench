"""Plain-text terminal rendering for evaluation results."""

from __future__ import annotations

import math

from tr_eval_bench.results import METRIC_LABELS, EvalResults


def _fmt(value: float | None) -> str:
    if value is None:
        return "-"
    if math.isnan(value):
        return "nan"
    return f"{value:.4f}"


def metric_rows(result: EvalResults) -> list[tuple[str, str, str]]:
    """Rows of ``(task, metric label, formatted score)`` in stable order."""
    rows: list[tuple[str, str, str]] = []
    if result.sts_spearman is not None:
        rows.append(("STS-TR", METRIC_LABELS["sts_spearman"], _fmt(result.sts_spearman)))
        rows.append(("STS-TR", METRIC_LABELS["sts_pearson"], _fmt(result.sts_pearson)))
    if result.retrieval_ndcg10 is not None:
        rows.append(
            ("Retrieval-TR", METRIC_LABELS["retrieval_ndcg10"], _fmt(result.retrieval_ndcg10))
        )
        rows.append(
            ("Retrieval-TR", METRIC_LABELS["retrieval_mrr10"], _fmt(result.retrieval_mrr10))
        )
        rows.append(
            ("Retrieval-TR", METRIC_LABELS["retrieval_recall10"], _fmt(result.retrieval_recall10))
        )
    if result.hybrid_ndcg10 is not None:
        rows.append(("Hybrid", METRIC_LABELS["hybrid_ndcg10"], _fmt(result.hybrid_ndcg10)))
        rows.append(("Hybrid", METRIC_LABELS["hybrid_mrr10"], _fmt(result.hybrid_mrr10)))
        rows.append(("Hybrid", METRIC_LABELS["hybrid_recall10"], _fmt(result.hybrid_recall10)))
    return rows


def render_table(result: EvalResults) -> str:
    """Render a single model's scores as a fixed-width table."""
    rows = metric_rows(result)
    header = ("TASK", "METRIC", "SCORE")
    widths = [len(header[i]) for i in range(3)]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    lines = [
        f"model    {result.model}",
        f"dataset  {result.dataset_version}",
    ]
    counts = []
    if result.n_sts_pairs:
        counts.append(f"STS {result.n_sts_pairs} pairs")
    if result.n_queries:
        counts.append(f"retrieval {result.n_queries} queries / {result.corpus_size} passages")
    if counts:
        lines.append(f"data     {' | '.join(counts)}")
    lines.append("")
    lines.append("  ".join(header[i].ljust(widths[i]) for i in range(3)).rstrip())
    lines.append("  ".join("-" * widths[i] for i in range(3)))
    for row in rows:
        lines.append("  ".join(row[i].ljust(widths[i]) for i in range(3)).rstrip())
    lines.append("")
    lines.append(f"elapsed  {result.elapsed_seconds:.2f}s")
    for error in result.errors:
        lines.append(f"warning  {error}")
    return "\n".join(lines)
