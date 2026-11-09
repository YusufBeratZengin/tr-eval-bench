#!/usr/bin/env python3
"""Build the bundled Parquet datasets from the curated Turkish source files.

Run from the repository root::

    python scripts/build_datasets.py

The script validates the source data (unique ids, score ranges, one relevant
passage per query, balanced score distribution), writes three Parquet files and
refreshes ``MANIFEST.json`` with sizes and SHA-256 digests.  It is a development
tool: the generated Parquet files are committed and shipped in the wheel, so
users never need PyArrow-writing code or network access.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_passages import PASSAGES
from data_sts import PAIRS

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "tr_eval_bench" / "datasets"
DATASET_VERSION = "tr-eval-mini-0.1.0"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate() -> None:
    if len(PASSAGES) != 100:
        raise SystemExit(f"expected 100 passages, found {len(PASSAGES)}")
    if len(PAIRS) != 100:
        raise SystemExit(f"expected 100 STS pairs, found {len(PAIRS)}")

    passage_ids = [row[0] for row in PASSAGES]
    if len(set(passage_ids)) != len(passage_ids):
        raise SystemExit("duplicate passage ids in data_passages.py")
    questions = [row[4] for row in PASSAGES]
    if len(set(questions)) != len(questions):
        raise SystemExit("duplicate questions in data_passages.py")
    for passage_id, domain, title, text, question in PASSAGES:
        if not all([passage_id, domain, title, text.strip(), question.strip()]):
            raise SystemExit(f"incomplete passage record: {passage_id}")
        if not question.strip().endswith("?"):
            raise SystemExit(f"question does not end with '?': {question}")

    pair_ids = [row[0] for row in PAIRS]
    if len(set(pair_ids)) != len(pair_ids):
        raise SystemExit("duplicate STS pair ids in data_sts.py")
    scores = Counter()
    for pair_id, domain, first, second, score in PAIRS:
        if score not in {0.0, 1.0, 2.0, 3.0, 4.0, 5.0}:
            raise SystemExit(f"invalid STS score for {pair_id}: {score}")
        if not first.strip() or not second.strip():
            raise SystemExit(f"empty sentence in STS pair {pair_id}")
        if not domain:
            raise SystemExit(f"missing domain in STS pair {pair_id}")
        scores[score] += 1
    if min(scores.values()) < 5:
        raise SystemExit(f"STS score distribution too skewed: {dict(scores)}")

    domains = Counter(row[1] for row in PASSAGES)
    if max(domains.values()) - min(domains.values()) > 1:
        raise SystemExit(f"passage domains are unbalanced: {dict(domains)}")
    print("validation OK")
    print(f"  STS score distribution: {dict(sorted(scores.items()))}")
    print(f"  passage domains:        {dict(domains)}")


def write_datasets() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    sts_table = pa.table(
        {
            "pair_id": [row[0] for row in PAIRS],
            "sentence1": [row[2] for row in PAIRS],
            "sentence2": [row[3] for row in PAIRS],
            "score": pa.array([row[4] for row in PAIRS], type=pa.float32()),
            "domain": [row[1] for row in PAIRS],
        }
    )
    pq.write_table(sts_table, DATA_DIR / "sts_mini.parquet", compression="zstd")

    corpus_table = pa.table(
        {
            "passage_id": [row[0] for row in PASSAGES],
            "title": [row[2] for row in PASSAGES],
            "text": [row[3] for row in PASSAGES],
            "domain": [row[1] for row in PASSAGES],
        }
    )
    pq.write_table(corpus_table, DATA_DIR / "rag_corpus.parquet", compression="zstd")

    query_table = pa.table(
        {
            "query_id": [f"q{index:03d}" for index in range(1, len(PASSAGES) + 1)],
            "query": [row[4] for row in PASSAGES],
            "passage_id": [row[0] for row in PASSAGES],
            "relevance": pa.array([1.0] * len(PASSAGES), type=pa.float32()),
            "domain": [row[1] for row in PASSAGES],
        }
    )
    pq.write_table(query_table, DATA_DIR / "rag_mini.parquet", compression="zstd")

    files = {
        name: _sha256(DATA_DIR / name)
        for name in ("sts_mini.parquet", "rag_corpus.parquet", "rag_mini.parquet")
    }
    manifest = {
        "name": DATASET_VERSION,
        "license": "MIT",
        "language": "tr",
        "created": "2026-11-03",
        "source": "Original text authored for tr-eval-bench from widely known general knowledge. "
        "No external corpus was copied; see PROVENANCE.md.",
        "counts": {
            "sts_pairs": len(PAIRS),
            "retrieval_queries": len(PASSAGES),
            "corpus_passages": len(PASSAGES),
        },
        "files": files,
    }
    (DATA_DIR / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for name, digest in files.items():
        size_kb = (DATA_DIR / name).stat().st_size / 1024
        print(f"  {name:<22} {size_kb:6.1f} KiB  sha256:{digest[:16]}…")


def main() -> int:
    validate()
    write_datasets()
    total = sum(path.stat().st_size for path in DATA_DIR.glob("*.parquet"))
    print(f"datasets written to {DATA_DIR} (total {total / 1024:.1f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
