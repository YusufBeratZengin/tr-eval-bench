# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-11-03

### Added

- `evaluate_model()` and `compare_models()` library API returning a frozen `EvalResults`
  (JSON schema version 1.1).
- STS-TR task: 100 graded Turkish sentence pairs, Spearman and Pearson correlation.
- Retrieval-TR task: 100 queries over a 100-passage corpus, NDCG@10, MRR@10 and Recall@10.
- Hybrid retrieval (`--hybrid`): BM25 + dense Reciprocal Rank Fusion scoring.
- Embedding backends: Hugging Face `SentenceTransformer` (`hf`), any OpenAI-compatible
  `/v1/embeddings` endpoint such as Ollama/vLLM/TGI (`api`), and a deterministic offline
  `hashing[:dim]` lexical baseline.
- CLI `tr-eval-bench` with `run`, `diff` and `datasets` commands; `--format table|json|markdown`,
  `--output`, `--export-markdown`, `--limit`, `--tasks`, `--batch-size`, `--device`, `--api-url`.
- CI regression gates: `--fail-under METRIC=VALUE` and `diff --max-drop METRIC=VALUE`
  (exit code 1 on violation).
- Bundled Parquet datasets with `MANIFEST.json` SHA-256 checksums, `datasets --verify` and a
  reproducible builder (`scripts/build_datasets.py`).
- Test suite: 150 tests, 95% statement/branch coverage, offline fake embedding server for API
  runner tests, hand-computed metric expectations.
- `scripts/check.sh` quality gate (ruff lint + format, mypy, pytest with an 85% coverage floor).
- GitHub Actions CI on Linux and Windows with Python 3.10 and 3.13, Dependabot, issue and pull
  request templates.
- Documentation: `README.md`, `README.tr.md`, dataset provenance note, contributing, security and
  code of conduct guides.

### Changed

- JSON schema 1.1: metrics with no defined value (constant gold labels) are omitted from `metrics`
  and listed with a reason under `undefined_metrics`; diff rows use `null` plus a `status` field
  instead of serialising `NaN`.
- Dataset text was rewritten to remove every real person name; people are described by role.
  Parquet files and `MANIFEST.json` were regenerated, and the measured reference scores updated.

### Fixed

- `--fail-under` and `diff --max-drop` now fail on non-finite (`NaN`) scores instead of passing.
- `diff --max-drop` now evaluates the requested metrics even when both models omitted them, and
  fails with an explicit message.
- The OpenAI-compatible runner rejects non-object response items, missing/duplicate/out-of-range
  indexes, non-numeric or non-finite values, and inconsistent embedding dimensions.
- The optional Hugging Face integration test scores all 100 STS pairs (a 20-pair slice had a
  constant label and produced `NaN`) and measures total wall time including model loading.
- `LICENSE` and `pyproject.toml` now spell the copyright holder as **Yusuf Berat Zengin**.
- Distribution archives were rebuilt so the sdist matches the sources and ships `CONTRIBUTING.md`,
  `SECURITY.md` and `CODE_OF_CONDUCT.md`.

### Known limitations

- The mini sets are a fast smoke test, not an official MTEB/TR-MTEB ranking; retrieval scores
  saturate for competent models (documented in the README).
- Turkish only; single-turn factoid retrieval; no reranker, classification or clustering tasks yet.
- Only the OpenAI-compatible embeddings protocol is supported (not Ollama's native endpoint).

[0.1.0]: https://github.com/YusufBeratZengin/tr-eval-bench/releases/tag/v0.1.0
