"""Two-model comparison ("diff") reports for the CLI and for CI comments."""

from __future__ import annotations

import math
from collections.abc import Iterable

from tr_eval_bench.results import METRIC_LABELS, DiffReport, EvalResults, MetricDelta


def build_diff(baseline: EvalResults, candidate: EvalResults) -> DiffReport:
    """Compare two results metric by metric (only metrics computed for at least one)."""
    rows: list[MetricDelta] = []
    for name, _label in METRIC_LABELS.items():
        left = baseline.metric(name)
        right = candidate.metric(name)
        if left is None and right is None:
            continue
        rows.append(MetricDelta(metric=name, baseline=left, candidate=right))
    if not rows:
        raise ValueError("no comparable metrics: run the same tasks for both models")
    return DiffReport(baseline=baseline, candidate=candidate, rows=tuple(rows))


def _fmt(value: float | None) -> str:
    if value is None:
        return "-"
    if math.isnan(value):
        return "nan"
    return f"{value:.4f}"


def _fmt_delta(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "nan" if value is not None else "-"
    return f"{value:+.4f}"


def _label(name: str) -> str:
    return METRIC_LABELS.get(name, name)


def render_diff_text(report: DiffReport) -> str:
    """Aligned terminal table with an explicit winner column."""
    header = ("METRIC", "BASELINE", "CANDIDATE", "DELTA", "WINNER")
    rows = [
        (
            _label(row.metric),
            _fmt(row.baseline),
            _fmt(row.candidate),
            _fmt_delta(row.delta),
            row.winner,
        )
        for row in report.rows
    ]
    widths = [len(header[i]) for i in range(len(header))]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    lines = [
        f"baseline   {report.baseline.model}",
        f"candidate  {report.candidate.model}",
        "",
        "  ".join(header[i].ljust(widths[i]) for i in range(len(header))).rstrip(),
        "  ".join("-" * widths[i] for i in range(len(header))),
    ]
    for row in rows:
        lines.append("  ".join(row[i].ljust(widths[i]) for i in range(len(header))).rstrip())
    improved = sum(1 for row in report.rows if row.winner == "candidate")
    regressed = sum(1 for row in report.rows if row.winner == "baseline")
    lines.append("")
    lines.append(f"candidate better on {improved}/{len(report.rows)} metrics, worse on {regressed}")
    return "\n".join(lines)


def render_diff_markdown(report: DiffReport) -> str:
    """Markdown table suitable for a pull-request comment."""
    lines = [
        "### tr-eval-bench diff",
        "",
        f"Baseline: `{report.baseline.model}`  ",
        f"Candidate: `{report.candidate.model}`",
        "",
        "| Metric | Baseline | Candidate | Δ | Winner |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in report.rows:
        lines.append(
            f"| {_label(row.metric)} | {_fmt(row.baseline)} | {_fmt(row.candidate)} | "
            f"{_fmt_delta(row.delta)} | {row.winner} |"
        )
    improved = sum(1 for row in report.rows if row.winner == "candidate")
    lines.append("")
    lines.append(f"Candidate is better on **{improved}/{len(report.rows)}** metrics.")
    return "\n".join(lines)


def regression_failures(
    report: DiffReport,
    thresholds: Iterable[tuple[str, float]],
) -> list[str]:
    """Return human-readable failures for metrics that dropped more than allowed.

    Thresholds are walked independently of the report rows so that a requested
    metric which neither model produced fails loudly instead of being skipped.
    Non-finite (``NaN``) scores also fail: a regression gate must never pass on
    a value that cannot be compared.
    """
    messages: list[str] = []
    rows = {row.metric: row for row in report.rows}
    for metric, limit in dict(thresholds).items():
        label = _label(metric)
        row = rows.get(metric)
        if row is None:
            messages.append(f"{label}: not computed for either model, cannot compare")
            continue
        if row.baseline is None:
            messages.append(f"{label}: missing on the baseline side, cannot compare")
            continue
        if row.candidate is None:
            messages.append(f"{label}: missing on the candidate side, cannot compare")
            continue
        if not math.isfinite(row.baseline):
            messages.append(f"{label}: baseline score is not finite (nan), cannot compare")
            continue
        if not math.isfinite(row.candidate):
            messages.append(f"{label}: candidate score is not finite (nan), cannot compare")
            continue
        delta = row.candidate - row.baseline
        if delta < -abs(limit):
            messages.append(f"{label} dropped by {abs(delta):.4f} (allowed drop {abs(limit):.4f})")
    return messages
