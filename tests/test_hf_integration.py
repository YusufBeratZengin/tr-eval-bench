"""Optional integration test that downloads a real model.

Skipped by default so the suite stays offline and fast. Enable with::

    TR_EVAL_HF_TEST=1 pytest tests/test_hf_integration.py

The test scores the *full* bundled STS set: the first twenty pairs all carry
the same gold label, so a ``limit=20`` smoke run produces an undefined
(Spearman = ``NaN``) correlation and would not prove anything.
"""

from __future__ import annotations

import math
import os
import time

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("TR_EVAL_HF_TEST") != "1",
        reason="set TR_EVAL_HF_TEST=1 to download and evaluate a real model",
    ),
]

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# Generous ceiling for a cached model on a shared CPU container; the point is to
# catch accidental downloads or pathological slowdowns, not to micro-benchmark.
MAX_TOTAL_SECONDS = 300.0


def test_real_huggingface_model_scores_within_budget() -> None:
    from tr_eval_bench import evaluate_model

    # Wall time is measured outside evaluate_model so model loading is included;
    # ``results.elapsed_seconds`` starts only after the runner is constructed.
    started = time.perf_counter()
    results = evaluate_model(MODEL)
    total_seconds = time.perf_counter() - started

    assert results.model == MODEL
    assert results.n_sts_pairs == 100
    assert results.n_queries == 100
    assert results.corpus_size == 100
    assert results.sts_spearman is not None
    assert results.sts_pearson is not None
    assert results.retrieval_mrr10 is not None
    assert math.isfinite(results.sts_spearman)
    assert math.isfinite(results.sts_pearson)
    assert math.isfinite(results.retrieval_mrr10)
    # A real semantic model must clear the random-correlation band.
    assert results.sts_spearman > 0.5
    assert total_seconds < MAX_TOTAL_SECONDS, (
        f"full evaluation incl. model load took {total_seconds:.1f}s"
    )
