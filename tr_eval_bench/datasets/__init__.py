"""Bundled mini gold datasets for Turkish embedding evaluation.

The datasets ship inside the wheel as Parquet files, so evaluation works with
no network access.  Provenance and licensing are documented in
``tr_eval_bench/datasets/PROVENANCE.md``; the files are rebuilt by
``scripts/build_datasets.py``.
"""

from __future__ import annotations

import functools
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

DATA_DIR = Path(__file__).resolve().parent
STS_PATH = DATA_DIR / "sts_mini.parquet"
RAG_QUERIES_PATH = DATA_DIR / "rag_mini.parquet"
RAG_CORPUS_PATH = DATA_DIR / "rag_corpus.parquet"
MANIFEST_PATH = DATA_DIR / "MANIFEST.json"

DATASET_VERSION = "tr-eval-mini-0.1.0"


class DatasetError(RuntimeError):
    """Raised when bundled data is missing, corrupt or internally inconsistent."""


@dataclass(frozen=True)
class StsPair:
    """One graded semantic-similarity pair (score in 0..5)."""

    pair_id: str
    sentence1: str
    sentence2: str
    score: float
    domain: str


@dataclass(frozen=True)
class Passage:
    """A retrieval corpus document."""

    passage_id: str
    title: str
    text: str
    domain: str


@dataclass(frozen=True)
class Query:
    """A retrieval query with graded qrels ``(passage_id, relevance)``."""

    query_id: str
    text: str
    qrels: tuple[tuple[str, float], ...]
    domain: str


@dataclass(frozen=True)
class RagDataset:
    """Corpus + queries for the retrieval task."""

    corpus: tuple[Passage, ...]
    queries: tuple[Query, ...]

    @property
    def corpus_size(self) -> int:
        return len(self.corpus)

    def relevance_labels(self, query: Query) -> dict[str, float]:
        """Map every corpus passage id to its relevance label (0.0 when absent)."""
        return dict(query.qrels)


def _read_parquet(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise DatasetError(
            f"bundled dataset not found: {path.name}. Reinstall the package or run "
            "`python scripts/build_datasets.py` from a source checkout."
        )
    try:
        table = pq.read_table(path)
    except Exception as exc:  # pragma: no cover - depends on pyarrow internals
        raise DatasetError(f"could not read {path.name}: {exc}") from exc
    return [dict(row) for row in table.to_pylist()]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DatasetError(message)


@functools.lru_cache(maxsize=1)
def load_sts_mini() -> tuple[StsPair, ...]:
    """Load the 100-pair Turkish STS mini set."""
    rows = _read_parquet(STS_PATH)
    _require(len(rows) > 0, "sts_mini.parquet is empty")
    pairs: list[StsPair] = []
    for row in rows:
        sentence1 = str(row["sentence1"]).strip()
        sentence2 = str(row["sentence2"]).strip()
        _require(bool(sentence1) and bool(sentence2), "sts pair with empty sentence")
        score = float(str(row["score"]))
        _require(0.0 <= score <= 5.0, f"sts score out of range: {score}")
        pairs.append(
            StsPair(
                pair_id=str(row["pair_id"]),
                sentence1=sentence1,
                sentence2=sentence2,
                score=score,
                domain=str(row["domain"]),
            )
        )
    return tuple(pairs)


@functools.lru_cache(maxsize=1)
def load_rag_mini() -> RagDataset:
    """Load the 100-query retrieval set and its 100-passage corpus."""
    corpus_rows = _read_parquet(RAG_CORPUS_PATH)
    query_rows = _read_parquet(RAG_QUERIES_PATH)
    _require(len(corpus_rows) > 0, "rag_corpus.parquet is empty")
    _require(len(query_rows) > 0, "rag_mini.parquet is empty")

    corpus: list[Passage] = []
    seen_passages: set[str] = set()
    for row in corpus_rows:
        passage_id = str(row["passage_id"])
        _require(passage_id not in seen_passages, f"duplicate passage id: {passage_id}")
        seen_passages.add(passage_id)
        text = str(row["text"]).strip()
        _require(bool(text), f"empty passage text: {passage_id}")
        corpus.append(
            Passage(
                passage_id=passage_id,
                title=str(row["title"]),
                text=text,
                domain=str(row["domain"]),
            )
        )

    qrels_by_query: dict[str, list[tuple[str, float]]] = {}
    texts: dict[str, str] = {}
    domains: dict[str, str] = {}
    order: list[str] = []
    for row in query_rows:
        query_id = str(row["query_id"])
        passage_id = str(row["passage_id"])
        _require(
            passage_id in seen_passages,
            f"query {query_id} references unknown passage {passage_id}",
        )
        relevance = float(str(row["relevance"]))
        _require(relevance >= 0.0, f"negative relevance in query {query_id}")
        if query_id not in qrels_by_query:
            qrels_by_query[query_id] = []
            texts[query_id] = str(row["query"]).strip()
            domains[query_id] = str(row["domain"])
            order.append(query_id)
        qrels_by_query[query_id].append((passage_id, relevance))

    queries: list[Query] = []
    for query_id in order:
        qrels = tuple(qrels_by_query[query_id])
        _require(any(rel > 0 for _, rel in qrels), f"query {query_id} has no relevant passage")
        _require(bool(texts[query_id]), f"query {query_id} has empty text")
        queries.append(
            Query(
                query_id=query_id,
                text=texts[query_id],
                qrels=qrels,
                domain=domains[query_id],
            )
        )

    query_ids = {query.query_id for query in queries}
    _require(len(query_ids) == len(queries), "duplicate query ids in rag_mini.parquet")
    return RagDataset(corpus=tuple(corpus), queries=tuple(queries))


def dataset_info() -> dict[str, object]:
    """Summary of the bundled datasets (used by the CLI and the manifest check)."""
    sts = load_sts_mini()
    rag = load_rag_mini()
    return {
        "version": DATASET_VERSION,
        "sts_pairs": len(sts),
        "retrieval_queries": len(rag.queries),
        "corpus_passages": rag.corpus_size,
        "sts_domains": sorted({pair.domain for pair in sts}),
        "retrieval_domains": sorted({query.domain for query in rag.queries}),
    }


def file_digests() -> dict[str, str]:
    """SHA-256 digests of the packaged Parquet files."""
    digests: dict[str, str] = {}
    for path in (STS_PATH, RAG_QUERIES_PATH, RAG_CORPUS_PATH):
        digests[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digests


def load_manifest() -> dict[str, object]:
    """Read ``MANIFEST.json`` (dataset version, sizes, digests)."""
    if not MANIFEST_PATH.is_file():
        raise DatasetError("MANIFEST.json is missing from the packaged datasets")
    try:
        data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetError(f"MANIFEST.json is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise DatasetError("MANIFEST.json must contain an object")
    return data


def verify_datasets() -> None:
    """Raise ``DatasetError`` when the packaged files do not match the manifest."""
    manifest = load_manifest()
    expected = manifest.get("files")
    if not isinstance(expected, dict):
        raise DatasetError("MANIFEST.json has no 'files' object")
    actual = file_digests()
    for name, digest in actual.items():
        if expected.get(name) != digest:
            raise DatasetError(f"checksum mismatch for {name}: packaged file was modified")
    for name in expected:
        if name not in actual:
            raise DatasetError(f"MANIFEST.json lists an unknown dataset file: {name}")
