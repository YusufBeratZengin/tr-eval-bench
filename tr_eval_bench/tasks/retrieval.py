"""RAG retrieval task: NDCG@10, MRR@10 and Recall@10 over the bundled corpus."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from tr_eval_bench.datasets import Query, RagDataset, load_rag_mini
from tr_eval_bench.metrics import mrr_at_k, ndcg_at_k, recall_at_k
from tr_eval_bench.runners import EmbeddingRunner
from tr_eval_bench.tasks.hybrid import BM25Index, fuse_score_matrix


@dataclass(frozen=True)
class RetrievalResult:
    """Outcome of the retrieval task (dense, plus hybrid when enabled)."""

    ndcg10: float
    mrr10: float
    recall10: float
    n_queries: int
    corpus_size: int
    elapsed_seconds: float
    k: int = 10
    hybrid_ndcg10: float | None = None
    hybrid_mrr10: float | None = None
    hybrid_recall10: float | None = None


@dataclass(frozen=True)
class _QueryScores:
    ndcg: float
    mrr: float
    recall: float


def _score_ranking(
    ranking: Sequence[int],
    passage_ids: Sequence[str],
    qrels: dict[str, float],
    *,
    k: int,
    total_relevant: int,
) -> _QueryScores:
    labels = [qrels.get(passage_ids[index], 0.0) for index in ranking[:k]]
    ideal = [value for value in qrels.values() if value > 0]
    return _QueryScores(
        ndcg=ndcg_at_k(labels, ideal, k=k),
        mrr=mrr_at_k(labels, k=k),
        recall=recall_at_k(labels, total_relevant, k=k),
    )


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else 0.0


def evaluate_retrieval(
    runner: EmbeddingRunner,
    dataset: RagDataset | None = None,
    *,
    k: int = 10,
    batch_size: int = 32,
    limit: int | None = None,
    hybrid: bool = False,
    rrf_k: int = 60,
) -> RetrievalResult:
    """Embed the corpus and queries, rank by cosine similarity and score.

    With ``hybrid=True`` a BM25 run over the same corpus is fused with the dense
    ranking using Reciprocal Rank Fusion (``rrf_k``), and the fused scores are
    reported alongside the dense ones.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")

    data = dataset if dataset is not None else load_rag_mini()
    if data.corpus_size == 0:
        raise ValueError("retrieval corpus is empty")
    queries: tuple[Query, ...] = data.queries[:limit] if limit is not None else data.queries
    if not queries:
        raise ValueError("retrieval evaluation needs at least one query")

    started = time.perf_counter()
    passage_ids = [passage.passage_id for passage in data.corpus]
    corpus_vectors = runner.embed([passage.text for passage in data.corpus], batch_size=batch_size)
    query_vectors = runner.embed([query.text for query in queries], batch_size=batch_size)
    if corpus_vectors.shape[0] != data.corpus_size or query_vectors.shape[0] != len(queries):
        raise ValueError("runner returned an unexpected number of embeddings")
    if corpus_vectors.shape[1] != query_vectors.shape[1]:
        raise ValueError("query and corpus embeddings have different dimensions")

    corpus_norms = np.linalg.norm(corpus_vectors, axis=1, keepdims=True)
    corpus_norms[corpus_norms == 0.0] = 1.0
    query_norms = np.linalg.norm(query_vectors, axis=1, keepdims=True)
    query_norms[query_norms == 0.0] = 1.0
    dense_scores = (query_vectors / query_norms) @ (corpus_vectors / corpus_norms).T

    dense_rankings = [np.argsort(-row, kind="stable").tolist() for row in dense_scores]

    fused_rankings: list[list[int]] = []
    if hybrid:
        bm25_scores = BM25Index([passage.text for passage in data.corpus]).score_batch(
            [query.text for query in queries]
        )
        for row in range(len(queries)):
            fused = fuse_score_matrix(np.vstack([dense_scores[row], bm25_scores[row]]), k=rrf_k)
            fused_rankings.append(np.argsort(-fused, kind="stable").tolist())

    dense_scores_per_query: list[_QueryScores] = []
    hybrid_scores_per_query: list[_QueryScores] = []
    for index, query in enumerate(queries):
        qrels = data.relevance_labels(query)
        total_relevant = sum(1 for value in qrels.values() if value > 0)
        dense_scores_per_query.append(
            _score_ranking(
                dense_rankings[index], passage_ids, qrels, k=k, total_relevant=total_relevant
            )
        )
        if hybrid:
            hybrid_scores_per_query.append(
                _score_ranking(
                    fused_rankings[index], passage_ids, qrels, k=k, total_relevant=total_relevant
                )
            )

    elapsed = time.perf_counter() - started
    result = RetrievalResult(
        ndcg10=_mean([row.ndcg for row in dense_scores_per_query]),
        mrr10=_mean([row.mrr for row in dense_scores_per_query]),
        recall10=_mean([row.recall for row in dense_scores_per_query]),
        n_queries=len(queries),
        corpus_size=data.corpus_size,
        elapsed_seconds=elapsed,
        k=k,
    )
    if not hybrid:
        return result
    return RetrievalResult(
        ndcg10=result.ndcg10,
        mrr10=result.mrr10,
        recall10=result.recall10,
        n_queries=result.n_queries,
        corpus_size=result.corpus_size,
        elapsed_seconds=result.elapsed_seconds,
        k=k,
        hybrid_ndcg10=_mean([row.ndcg for row in hybrid_scores_per_query]),
        hybrid_mrr10=_mean([row.mrr for row in hybrid_scores_per_query]),
        hybrid_recall10=_mean([row.recall for row in hybrid_scores_per_query]),
    )
