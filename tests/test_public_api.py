"""Public API tests, including the <=60 s CPU budget acceptance criterion."""

from __future__ import annotations

import json
import math
import time
from typing import Any, cast

import pytest
import tr_eval_bench
from tr_eval_bench import compare_models, evaluate_model
from tr_eval_bench.results import EvalResults


def test_evaluate_model_full_run_is_under_60_seconds() -> None:
    started = time.perf_counter()
    results = evaluate_model("hashing:256")
    elapsed = time.perf_counter() - started
    assert elapsed < 60.0, f"mini benchmark took {elapsed:.1f}s on CPU"
    assert results.elapsed_seconds < 60.0
    assert results.n_sts_pairs == 100
    assert results.n_queries == 100
    assert results.corpus_size == 100
    assert results.sts_spearman is not None
    assert results.retrieval_mrr10 is not None
    assert results.tasks == ("sts", "retrieval")


def test_evaluate_model_retrieval_only() -> None:
    results = evaluate_model("hashing:64", tasks="retrieval", limit=5)
    assert results.sts_spearman is None
    assert results.sts_pearson is None
    assert results.n_sts_pairs == 0
    assert results.n_queries == 5


def test_evaluate_model_sts_only() -> None:
    results = evaluate_model("hashing:64", tasks=["sts"], limit=5)
    assert results.retrieval_mrr10 is None
    assert results.n_sts_pairs == 5


def test_evaluate_model_hybrid_requires_retrieval() -> None:
    with pytest.raises(ValueError, match="requires the retrieval task"):
        evaluate_model("hashing:64", tasks="sts", hybrid=True)


def test_evaluate_model_rejects_unknown_task() -> None:
    with pytest.raises(ValueError, match="unknown task"):
        evaluate_model("hashing:64", tasks="sts,clustering")


def test_evaluate_model_rejects_empty_task_list() -> None:
    with pytest.raises(ValueError, match="at least one task"):
        evaluate_model("hashing:64", tasks="")


def test_evaluate_model_validates_limit_and_batch() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        evaluate_model("hashing:64", batch_size=0)
    with pytest.raises(ValueError, match="limit"):
        evaluate_model("hashing:64", limit=0)


def test_evaluate_model_accepts_injected_runner() -> None:
    from tr_eval_bench.runners import HashingRunner

    results = evaluate_model("ignored", tasks="sts", runner=HashingRunner(64))
    assert results.model == "hashing:64"
    assert results.sts_spearman is not None
    assert results.n_sts_pairs == 100


def test_compare_models_with_strings() -> None:
    report = compare_models("hashing:64", "hashing:128", tasks="retrieval", limit=5)
    assert report.baseline.model == "hashing:64"
    assert report.candidate.model == "hashing:128"
    assert len(report.rows) == 3


def test_compare_models_with_results_objects() -> None:
    first = EvalResults(model="a", sts_spearman=0.5)
    second = EvalResults(model="b", sts_spearman=0.6)
    report = compare_models(first, second)
    assert report.rows[0].winner == "candidate"


def test_results_dict_is_json_friendly() -> None:
    results = evaluate_model("hashing:64", tasks="retrieval", limit=3)
    payload = results.to_dict()
    assert payload["schema_version"] == "1.1"
    metrics = cast(dict[str, float], payload["metrics"])
    counts = cast(dict[str, int], payload["counts"])
    assert set(metrics) == {"retrieval_ndcg10", "retrieval_mrr10", "retrieval_recall10"}
    assert counts["retrieval_queries"] == 3
    assert isinstance(payload["elapsed_seconds"], float)


def test_results_metric_helper() -> None:
    results = EvalResults(model="m", sts_spearman=0.25)
    assert results.metric("sts_spearman") == pytest.approx(0.25)
    assert results.metric("sts_pearson") is None
    assert results.metrics() == {"sts_spearman": 0.25}
    with pytest.raises(KeyError, match="unknown metric"):
        results.metric("does_not_exist")


def test_spearman_nan_is_serialized_as_null_with_status() -> None:
    results = EvalResults(model="m", sts_spearman=math.nan)
    payload = results.to_dict()
    metrics: Any = payload["metrics"]
    undefined: Any = payload["undefined_metrics"]
    assert "sts_spearman" not in metrics
    assert "sts_spearman" in undefined
    assert "constant labels" in undefined["sts_spearman"]
    json.dumps(payload, allow_nan=False)


def test_package_exports_expected_symbols() -> None:
    for name in ("evaluate_model", "compare_models", "EvalResults", "__version__"):
        assert hasattr(tr_eval_bench, name)
    assert tr_eval_bench.__version__ == "0.1.0"
