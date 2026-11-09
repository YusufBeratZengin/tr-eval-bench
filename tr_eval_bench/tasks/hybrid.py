"""Lexical BM25 retrieval and Reciprocal Rank Fusion (RRF).

The dense retriever alone misses exact-term matches (product codes, proper
nouns, rare morphology); hybrid search fuses both rankings with RRF, the
strategy used by most production RAG stacks.  Implementations follow the
standard formulas:

* BM25 (Robertson & Zaragoza, 2009) with ``k1=1.5``, ``b=0.75`` and the
  ``log(1 + (N - df + 0.5) / (df + 0.5))`` IDF variant.
* RRF (Cormack et al., 2009): ``score(d) = sum_i w_i / (k + rank_i(d))`` with
  ``k=60`` by default.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence

import numpy as np
import numpy.typing as npt

from tr_eval_bench.runners import turkish_lower

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lowercase (Turkish-aware) word tokenizer, dropping 1-character tokens."""
    return [token for token in _WORD_RE.findall(turkish_lower(text)) if len(token) > 1]


class BM25Index:
    """In-memory BM25 index over a small document collection."""

    def __init__(
        self,
        documents: Sequence[str],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not documents:
            raise ValueError("BM25Index needs at least one document")
        if k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0.0 <= b <= 1.0:
            raise ValueError("b must be within [0, 1]")
        self.k1 = float(k1)
        self.b = float(b)
        self._doc_tokens = [tokenize(document) for document in documents]
        self._doc_lengths = np.array([len(tokens) for tokens in self._doc_tokens], dtype=np.float64)
        total = float(self._doc_lengths.sum())
        self._avg_length = total / len(self._doc_tokens) if total > 0 else 0.0
        self._postings: dict[str, list[tuple[int, int]]] = {}
        for doc_index, tokens in enumerate(self._doc_tokens):
            for token, tf in Counter(tokens).items():
                self._postings.setdefault(token, []).append((doc_index, tf))
        n_docs = len(self._doc_tokens)
        self._idf = {
            token: math.log(1.0 + (n_docs - len(postings) + 0.5) / (len(postings) + 0.5))
            for token, postings in self._postings.items()
        }
        self.n_docs = n_docs

    @property
    def vocabulary_size(self) -> int:
        return len(self._postings)

    def scores(self, query: str) -> npt.NDArray[np.float64]:
        """BM25 score for every document in corpus order."""
        scores = np.zeros(self.n_docs, dtype=np.float64)
        if self._avg_length == 0.0:
            return scores
        for token, query_tf in Counter(tokenize(query)).items():
            postings = self._postings.get(token)
            if not postings:
                continue
            idf = self._idf[token]
            for doc_index, term_freq in postings:
                length_norm = self.k1 * (
                    1.0 - self.b + self.b * self._doc_lengths[doc_index] / self._avg_length
                )
                scores[doc_index] += (
                    query_tf * idf * term_freq * (self.k1 + 1.0) / (term_freq + length_norm)
                )
        return scores

    def score_batch(self, queries: Iterable[str]) -> npt.NDArray[np.float64]:
        """Score matrix of shape ``(n_queries, n_docs)``."""
        rows = [self.scores(query) for query in queries]
        if not rows:
            return np.zeros((0, self.n_docs), dtype=np.float64)
        return np.vstack(rows)


def rankings_from_scores(scores: npt.NDArray[np.floating]) -> list[list[int]]:
    """Convert a ``(n_queries, n_docs)`` score matrix to best-first doc rankings."""
    matrix = np.asarray(scores, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("scores must be a 2-D array")
    return [np.argsort(-row, kind="stable").tolist() for row in matrix]


def rrf_scores(
    rankings: Sequence[Sequence[int]],
    n_docs: int,
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> npt.NDArray[np.float64]:
    """Reciprocal Rank Fusion of one or more best-first document rankings."""
    if n_docs <= 0:
        raise ValueError("n_docs must be positive")
    if k <= 0:
        raise ValueError("k must be positive")
    if not rankings:
        raise ValueError("rrf_scores needs at least one ranking")
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights must match the number of rankings")
    fused = np.zeros(n_docs, dtype=np.float64)
    for ranking, weight in zip(rankings, weights, strict=True):
        if weight < 0:
            raise ValueError("weights must be non-negative")
        for rank, doc_index in enumerate(ranking, start=1):
            if doc_index < 0 or doc_index >= n_docs:
                raise ValueError(f"document index {doc_index} outside [0, {n_docs})")
            fused[doc_index] += weight / (k + rank)
    return fused


def fuse_score_matrix(
    score_matrix: npt.NDArray[np.floating],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> npt.NDArray[np.float64]:
    """RRF-fuse several systems stacked along axis 0 of ``(n_systems, n_docs)``.

    A system that produced no signal at all for this query (every score is
    zero, e.g. BM25 when no query term occurs in the corpus) is skipped instead
    of contributing an arbitrary corpus-order ranking: ranking noise would
    otherwise hand rank 1 to document 0 and change the fused order for reasons
    unrelated to relevance.
    """
    matrix = np.asarray(score_matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("score_matrix must have shape (n_systems, n_docs)")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        return np.zeros(matrix.shape[1], dtype=np.float64)
    row_weights = [1.0] * matrix.shape[0] if weights is None else list(weights)
    if len(row_weights) != matrix.shape[0]:
        raise ValueError("weights must match the number of systems")
    active = [index for index in range(matrix.shape[0]) if np.any(matrix[index] != 0.0)]
    if not active:
        return np.zeros(matrix.shape[1], dtype=np.float64)
    rankings = [rankings_from_scores(matrix[index : index + 1])[0] for index in active]
    return rrf_scores(rankings, matrix.shape[1], k=k, weights=[row_weights[i] for i in active])
