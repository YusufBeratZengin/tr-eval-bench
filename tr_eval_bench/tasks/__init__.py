"""Evaluation tasks bundled with tr-eval-bench."""

from tr_eval_bench.tasks.hybrid import BM25Index, rrf_scores
from tr_eval_bench.tasks.retrieval import RetrievalResult, evaluate_retrieval
from tr_eval_bench.tasks.sts import StsResult, evaluate_sts

__all__ = [
    "BM25Index",
    "RetrievalResult",
    "StsResult",
    "evaluate_retrieval",
    "evaluate_sts",
    "rrf_scores",
]
