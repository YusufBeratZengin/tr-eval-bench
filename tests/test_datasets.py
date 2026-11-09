"""Dataset loader tests: schema, integrity and failure modes."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import tr_eval_bench.datasets as datasets
from tr_eval_bench.datasets import (
    DATASET_VERSION,
    DatasetError,
    dataset_info,
    file_digests,
    load_manifest,
    load_rag_mini,
    load_sts_mini,
    verify_datasets,
)


@pytest.fixture(autouse=True)
def _clear_dataset_caches() -> Iterator[None]:
    load_sts_mini.cache_clear()
    load_rag_mini.cache_clear()
    yield
    load_sts_mini.cache_clear()
    load_rag_mini.cache_clear()


def test_sts_mini_has_100_graded_pairs() -> None:
    pairs = load_sts_mini()
    assert len(pairs) == 100
    assert len({pair.pair_id for pair in pairs}) == 100
    assert {pair.score for pair in pairs} == {0.0, 1.0, 2.0, 3.0, 4.0, 5.0}
    assert all(pair.sentence1 and pair.sentence2 for pair in pairs)


def test_rag_mini_has_100_queries_and_passages() -> None:
    data = load_rag_mini()
    assert data.corpus_size == 100
    assert len(data.queries) == 100
    passage_ids = {passage.passage_id for passage in data.corpus}
    for query in data.queries:
        labels = data.relevance_labels(query)
        assert set(labels).issubset(passage_ids)
        assert any(value > 0 for value in labels.values())
        assert query.text.endswith("?")


def test_dataset_info_reports_counts() -> None:
    info = cast(dict[str, Any], dataset_info())
    assert info["version"] == DATASET_VERSION
    assert info["sts_pairs"] == 100
    assert info["retrieval_queries"] == 100
    assert info["corpus_passages"] == 100
    assert len(info["retrieval_domains"]) == 10


def test_manifest_and_checksums_are_consistent() -> None:
    verify_datasets()
    manifest = load_manifest()
    files = cast(dict[str, str], manifest["files"])
    counts = cast(dict[str, int], manifest["counts"])
    assert manifest["license"] == "MIT"
    assert counts["sts_pairs"] == 100
    assert set(files) == set(file_digests())


def test_missing_dataset_file_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(datasets, "STS_PATH", tmp_path / "missing.parquet")
    with pytest.raises(DatasetError, match="bundled dataset not found"):
        load_sts_mini()


def test_corrupt_parquet_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    broken = tmp_path / "broken.parquet"
    broken.write_text("this is not parquet", encoding="utf-8")
    monkeypatch.setattr(datasets, "STS_PATH", broken)
    with pytest.raises(DatasetError, match="could not read"):
        load_sts_mini()


def test_sts_score_out_of_range_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "sts.parquet"
    pq.write_table(
        pa.table(
            {
                "pair_id": ["a"],
                "sentence1": ["bir"],
                "sentence2": ["iki"],
                "score": pa.array([9.0], type=pa.float32()),
                "domain": ["test"],
            }
        ),
        path,
    )
    monkeypatch.setattr(datasets, "STS_PATH", path)
    with pytest.raises(DatasetError, match="out of range"):
        load_sts_mini()


def test_unknown_passage_in_qrels_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "queries.parquet"
    pq.write_table(
        pa.table(
            {
                "query_id": ["q1"],
                "query": ["soru?"],
                "passage_id": ["does-not-exist"],
                "relevance": pa.array([1.0], type=pa.float32()),
                "domain": ["test"],
            }
        ),
        path,
    )
    monkeypatch.setattr(datasets, "RAG_QUERIES_PATH", path)
    with pytest.raises(DatasetError, match="unknown passage"):
        load_rag_mini()


def test_query_without_relevant_passage_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    corpus = tmp_path / "corpus.parquet"
    pq.write_table(
        pa.table(
            {
                "passage_id": ["p1"],
                "title": ["t"],
                "text": ["metin"],
                "domain": ["test"],
            }
        ),
        corpus,
    )
    queries = tmp_path / "queries.parquet"
    pq.write_table(
        pa.table(
            {
                "query_id": ["q1"],
                "query": ["soru?"],
                "passage_id": ["p1"],
                "relevance": pa.array([0.0], type=pa.float32()),
                "domain": ["test"],
            }
        ),
        queries,
    )
    monkeypatch.setattr(datasets, "RAG_CORPUS_PATH", corpus)
    monkeypatch.setattr(datasets, "RAG_QUERIES_PATH", queries)
    with pytest.raises(DatasetError, match="no relevant passage"):
        load_rag_mini()


def test_duplicate_passage_ids_raise(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.parquet"
    pq.write_table(
        pa.table(
            {
                "passage_id": ["p1", "p1"],
                "title": ["a", "b"],
                "text": ["bir", "iki"],
                "domain": ["test", "test"],
            }
        ),
        corpus,
    )
    monkeypatch.setattr(datasets, "RAG_CORPUS_PATH", corpus)
    with pytest.raises(DatasetError, match="duplicate passage id"):
        load_rag_mini()


def test_manifest_checksum_mismatch_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    manifest = tmp_path / "MANIFEST.json"
    payload = {
        "files": {
            "sts_mini.parquet": "0" * 64,
            "rag_mini.parquet": "0" * 64,
            "rag_corpus.parquet": "0" * 64,
        }
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(datasets, "MANIFEST_PATH", manifest)
    with pytest.raises(DatasetError, match="checksum mismatch"):
        verify_datasets()


def test_manifest_missing_file_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(datasets, "MANIFEST_PATH", tmp_path / "nope.json")
    with pytest.raises(DatasetError, match=r"MANIFEST\.json is missing"):
        load_manifest()


def test_manifest_invalid_json_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    manifest = tmp_path / "MANIFEST.json"
    manifest.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(datasets, "MANIFEST_PATH", manifest)
    with pytest.raises(DatasetError, match="not valid JSON"):
        load_manifest()


def test_manifest_without_files_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    manifest = tmp_path / "MANIFEST.json"
    manifest.write_text('{"name": "x"}', encoding="utf-8")
    monkeypatch.setattr(datasets, "MANIFEST_PATH", manifest)
    with pytest.raises(DatasetError, match="no 'files'"):
        verify_datasets()
