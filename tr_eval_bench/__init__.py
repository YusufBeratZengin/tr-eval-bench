"""tr-eval-bench: a fast, offline-first benchmark for Turkish embedding models.

Public API::

    from tr_eval_bench import evaluate_model, compare_models

    results = evaluate_model("hashing:256")
    print(results.sts_spearman, results.retrieval_mrr10)
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version

from tr_eval_bench.datasets import (
    DATASET_VERSION,
    DatasetError,
    dataset_info,
    load_rag_mini,
    load_sts_mini,
    verify_datasets,
)
from tr_eval_bench.reporters.diff_report import build_diff
from tr_eval_bench.results import SCHEMA_VERSION, DiffReport, EvalResults
from tr_eval_bench.runners import (
    EmbeddingRunner,
    HashingRunner,
    OpenAICompatRunner,
    RunnerError,
    RunnerUnavailable,
    SentenceTransformerRunner,
    load_runner,
    normalize_rows,
)
from tr_eval_bench.tasks import (
    BM25Index,
    RetrievalResult,
    StsResult,
    evaluate_retrieval,
    evaluate_sts,
    rrf_scores,
)

try:  # pragma: no cover - depends on installation mode
    __version__ = version("tr-eval-bench")
except PackageNotFoundError:  # pragma: no cover - source checkout without install
    __version__ = "0.1.0"

SUPPORTED_TASKS: tuple[str, ...] = ("sts", "retrieval")


def _normalize_tasks(tasks: Sequence[str] | str) -> tuple[str, ...]:
    if isinstance(tasks, str):
        requested = [chunk.strip() for chunk in tasks.split(",") if chunk.strip()]
    else:
        requested = [str(task).strip() for task in tasks if str(task).strip()]
    if not requested:
        raise ValueError("at least one task is required (sts, retrieval)")
    unknown = [task for task in requested if task not in SUPPORTED_TASKS]
    if unknown:
        raise ValueError(
            f"unknown task(s): {', '.join(unknown)}; supported: {', '.join(SUPPORTED_TASKS)}"
        )
    ordered = tuple(task for task in SUPPORTED_TASKS if task in requested)
    return ordered


def evaluate_model(
    model: str,
    *,
    tasks: Sequence[str] | str = SUPPORTED_TASKS,
    backend: str = "auto",
    device: str = "cpu",
    batch_size: int = 32,
    limit: int | None = None,
    hybrid: bool = False,
    rrf_k: int = 60,
    api_url: str | None = None,
    runner: EmbeddingRunner | None = None,
) -> EvalResults:
    """Evaluate one embedding model on the bundled mini benchmark.

    Parameters
    ----------
    model:
        Hugging Face model id (``sentence-transformers/...``), a local API model
        name (with ``backend="api"``), or ``hashing[:dim]`` for the offline baseline.
    tasks:
        Iterable or comma-separated string drawn from ``("sts", "retrieval")``.
    runner:
        Pre-built runner (mainly for tests and custom integrations); when given,
        ``model`` is only used as the display name.
    """
    selected = _normalize_tasks(tasks)
    if hybrid and "retrieval" not in selected:
        raise ValueError("hybrid ranking requires the retrieval task")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")

    active_runner = runner or load_runner(model, backend=backend, device=device, api_url=api_url)
    display_name = model if runner is None else active_runner.name
    started = time.perf_counter()

    sts_result: StsResult | None = None
    retrieval_result: RetrievalResult | None = None
    if "sts" in selected:
        sts_result = evaluate_sts(active_runner, batch_size=batch_size, limit=limit)
    if "retrieval" in selected:
        retrieval_result = evaluate_retrieval(
            active_runner, batch_size=batch_size, limit=limit, hybrid=hybrid, rrf_k=rrf_k
        )

    elapsed = time.perf_counter() - started
    return EvalResults(
        model=display_name,
        tasks=selected,
        sts_spearman=sts_result.spearman if sts_result else None,
        sts_pearson=sts_result.pearson if sts_result else None,
        retrieval_ndcg10=retrieval_result.ndcg10 if retrieval_result else None,
        retrieval_mrr10=retrieval_result.mrr10 if retrieval_result else None,
        retrieval_recall10=retrieval_result.recall10 if retrieval_result else None,
        hybrid_ndcg10=retrieval_result.hybrid_ndcg10 if retrieval_result else None,
        hybrid_mrr10=retrieval_result.hybrid_mrr10 if retrieval_result else None,
        hybrid_recall10=retrieval_result.hybrid_recall10 if retrieval_result else None,
        n_sts_pairs=sts_result.n_pairs if sts_result else 0,
        n_queries=retrieval_result.n_queries if retrieval_result else 0,
        corpus_size=retrieval_result.corpus_size if retrieval_result else 0,
        elapsed_seconds=elapsed,
        dataset_version=DATASET_VERSION,
    )


def compare_models(
    baseline: str | EvalResults,
    candidate: str | EvalResults,
    **kwargs: object,
) -> DiffReport:
    """Evaluate two models (or diff two existing results) metric by metric."""
    baseline_result = (
        baseline if isinstance(baseline, EvalResults) else evaluate_model(baseline, **kwargs)  # type: ignore[arg-type]
    )
    candidate_result = (
        candidate if isinstance(candidate, EvalResults) else evaluate_model(candidate, **kwargs)  # type: ignore[arg-type]
    )
    return build_diff(baseline_result, candidate_result)


__all__ = [
    "DATASET_VERSION",
    "SCHEMA_VERSION",
    "SUPPORTED_TASKS",
    "BM25Index",
    "DatasetError",
    "DiffReport",
    "EmbeddingRunner",
    "EvalResults",
    "HashingRunner",
    "OpenAICompatRunner",
    "RetrievalResult",
    "RunnerError",
    "RunnerUnavailable",
    "SentenceTransformerRunner",
    "StsResult",
    "__version__",
    "compare_models",
    "dataset_info",
    "evaluate_model",
    "evaluate_retrieval",
    "evaluate_sts",
    "load_rag_mini",
    "load_runner",
    "load_sts_mini",
    "normalize_rows",
    "rrf_scores",
    "verify_datasets",
]
