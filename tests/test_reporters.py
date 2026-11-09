"""Reporter tests: terminal table, Markdown, leaderboard and diff gates."""

from __future__ import annotations

import math
from typing import Any, cast

import pytest
from tr_eval_bench.reporters import (
    build_diff,
    regression_failures,
    render_diff_markdown,
    render_diff_text,
    render_leaderboard,
    render_markdown,
    render_table,
)
from tr_eval_bench.results import EvalResults


def _result(model: str = "hashing:256", **overrides: object) -> EvalResults:
    base: dict[str, object] = {
        "model": model,
        "tasks": ("sts", "retrieval"),
        "sts_spearman": 0.5123,
        "sts_pearson": 0.5001,
        "retrieval_ndcg10": 0.6345,
        "retrieval_mrr10": 0.6012,
        "retrieval_recall10": 0.9,
        "n_sts_pairs": 100,
        "n_queries": 100,
        "corpus_size": 100,
        "elapsed_seconds": 1.234,
        "dataset_version": "tr-eval-mini-0.1.0",
    }
    base.update(overrides)
    return EvalResults(**base)  # type: ignore[arg-type]


def test_render_table_contains_scores_and_metadata() -> None:
    text = render_table(_result())
    assert "hashing:256" in text
    assert "tr-eval-mini-0.1.0" in text
    assert "Spearman" in text
    assert "0.5123" in text
    assert "MRR@10" in text
    assert "0.6012" in text
    assert "elapsed  1.23s" in text


def test_render_table_shows_hybrid_and_warnings() -> None:
    result = _result(
        hybrid_ndcg10=0.61,
        hybrid_mrr10=0.58,
        hybrid_recall10=0.9,
        errors=("model card missing",),
    )
    text = render_table(result)
    assert "Hybrid NDCG@10" in text
    assert "warning  model card missing" in text


def test_render_table_handles_partial_and_nan_results() -> None:
    result = EvalResults(
        model="only-sts",
        tasks=("sts",),
        sts_spearman=float("nan"),
        sts_pearson=float("nan"),
        n_sts_pairs=100,
        dataset_version="tr-eval-mini-0.1.0",
    )
    text = render_table(result)
    assert "nan" in text
    assert "Retrieval" not in text


def test_render_markdown_single_result() -> None:
    text = render_markdown(_result())
    assert text.startswith("# tr-eval-bench")
    assert "| Task | Metric | Score |" in text
    assert "| STS-TR | Spearman | 0.5123 |" in text
    assert "not an official MTEB ranking" in text


def test_render_leaderboard_orders_metrics() -> None:
    text = render_leaderboard([_result("a"), _result("b")])
    assert "| Model | STS Spearman | NDCG@10 | MRR@10 |" in text
    assert "| `a` | 0.5123 | 0.6345 | 0.6012 |" in text
    with pytest.raises(ValueError, match="at least one result"):
        render_leaderboard([])


def test_build_diff_only_includes_computed_metrics() -> None:
    baseline = EvalResults(model="a", sts_spearman=0.5, n_sts_pairs=100)
    candidate = EvalResults(model="b", sts_spearman=0.6, retrieval_mrr10=0.4)
    report = build_diff(baseline, candidate)
    metrics = [row.metric for row in report.rows]
    assert metrics == ["sts_spearman", "retrieval_mrr10"]
    assert report.rows[0].winner == "candidate"
    assert report.rows[1].winner == "n/a"
    payload = report.to_dict()
    baseline_payload = cast(dict[str, Any], payload["baseline"])
    assert baseline_payload["model"] == "a"


def test_build_diff_requires_overlap() -> None:
    with pytest.raises(ValueError, match="no comparable metrics"):
        build_diff(EvalResults(model="a"), EvalResults(model="b"))


def test_diff_winner_and_tie() -> None:
    baseline = _result("base")
    candidate = _result("cand", sts_spearman=0.5123)
    report = build_diff(baseline, candidate)
    assert report.rows[0].winner == "tie"
    assert report.rows[0].delta == pytest.approx(0.0)


def test_render_diff_text_and_markdown() -> None:
    baseline = _result("base", retrieval_mrr10=0.70)
    candidate = _result("cand", retrieval_mrr10=0.65)
    report = build_diff(baseline, candidate)
    text = render_diff_text(report)
    assert "baseline   base" in text
    assert "candidate  cand" in text
    assert "-0.0500" in text
    assert "candidate better on" in text
    markdown = render_diff_markdown(report)
    assert "### tr-eval-bench diff" in markdown
    assert "| Metric | Baseline | Candidate | Δ | Winner |" in markdown
    assert "`base`" in markdown


def test_regression_failures_detects_drops() -> None:
    baseline = _result("base", retrieval_mrr10=0.70)
    candidate = _result("cand", retrieval_mrr10=0.60)
    report = build_diff(baseline, candidate)
    failures = regression_failures(report, [("retrieval_mrr10", 0.05)])
    assert len(failures) == 1
    assert "dropped by 0.1000" in failures[0]
    assert regression_failures(report, [("retrieval_mrr10", 0.2)]) == []


def test_regression_failures_reports_missing_metric() -> None:
    baseline = EvalResults(model="a", sts_spearman=0.5)
    candidate = EvalResults(model="b", sts_spearman=0.5, retrieval_mrr10=0.4)
    report = build_diff(baseline, candidate)
    failures = regression_failures(report, [("retrieval_mrr10", 0.1)])
    assert failures == ["MRR@10: missing on the baseline side, cannot compare"]


def test_regression_failures_fails_when_metric_missing_on_both_sides() -> None:
    baseline = EvalResults(model="a", sts_spearman=0.5)
    candidate = EvalResults(model="b", sts_spearman=0.6)
    report = build_diff(baseline, candidate)
    failures = regression_failures(report, [("retrieval_mrr10", 0.1)])
    assert failures == ["MRR@10: not computed for either model, cannot compare"]


@pytest.mark.parametrize("side", ["baseline", "candidate"])
def test_regression_failures_rejects_nan_scores(side: str) -> None:
    baseline = _result("base")
    candidate = _result("cand")
    if side == "baseline":
        baseline = _result("base", retrieval_mrr10=math.nan)
    else:
        candidate = _result("cand", retrieval_mrr10=math.nan)
    failures = regression_failures(build_diff(baseline, candidate), [("retrieval_mrr10", 0.05)])
    assert len(failures) == 1
    assert f"{side} score is not finite (nan)" in failures[0]


def test_diff_dict_is_strict_json_with_undefined_metrics() -> None:
    import json

    baseline = _result("base", sts_spearman=math.nan)
    candidate = _result("cand", sts_spearman=0.6)
    payload = build_diff(baseline, candidate).to_dict()
    encoded = json.dumps(payload, allow_nan=False)
    assert "NaN" not in encoded
    deltas = payload["deltas"]
    assert isinstance(deltas, list)
    spearman = next(row for row in deltas if row["metric"] == "sts_spearman")
    assert spearman["baseline"] is None
    assert spearman["delta"] is None
    assert spearman["status"] == "undefined"
    assert spearman["winner"] == "n/a"


def test_nan_metric_rendering_is_safe() -> None:
    result = _result(sts_spearman=math.nan, sts_pearson=math.nan)
    assert "nan" in render_markdown(result)
    baseline = _result()
    diff = render_diff_text(build_diff(result, baseline))
    assert "nan" in diff  # NaN is printed, not crashed on
    assert "n/a" in diff  # and NaN deltas have no winner
