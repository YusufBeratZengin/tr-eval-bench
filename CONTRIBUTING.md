# Contributing to tr-eval-bench

Thanks for helping. This project is small on purpose: a short, correct, offline benchmark kit.
Small, well-tested changes are more likely to be merged than large rewrites.

## Development setup

```bash
git clone https://github.com/YusufBeratZengin/tr-eval-bench
cd tr-eval-bench
python3 -m venv .venv
. .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
bash scripts/check.sh
```

`scripts/check.sh` is the single gate: dataset checksum verification, `ruff` lint and formatting,
`mypy`, then `pytest` with an 85% coverage floor. A pull request is ready when it exits 0.

## What we accept

- Bug fixes with a regression test that fails before the fix.
- New metrics or tasks that follow the published definition and come with hand-computed tests.
- Dataset improvements (harder negatives, new domains) that keep the licence clean and the builder
  validation green.
- Documentation fixes, especially in `README.tr.md` (natural Turkish, not machine translation).

Out of scope for 0.1.x: training code, proprietary corpora, telemetry, network calls during tests.

## Adding or editing data

1. Edit `scripts/data_passages.py` or `scripts/data_sts.py` (human-readable sources).
2. Run `python scripts/build_datasets.py` — it validates unique ids, question formatting, STS score
   range/distribution, balanced domains, then rewrites the Parquet files and `MANIFEST.json`.
3. Commit both the sources and the regenerated Parquet + manifest in the same commit.
4. Additions must be original text or clearly licensed public-domain/CC-BY material **with
   attribution**, must contain no personal data, and must not include official rates, taxes, legal
   values or prices. If a fact needs a source you cannot provide, leave it out.

## Tests

- Never hit the network in tests. Use the `api_server` fixture in `tests/conftest.py` for HTTP
  behaviour and `fake_sentence_transformers_module` for the Hugging Face adapter.
- Tests that genuinely need a downloaded model must be skipped unless `TR_EVAL_HF_TEST=1`.
- Metric tests should assert hand-computed numbers, not just "between 0 and 1".

## Style

- Python 3.10+, type hints on every public function, `from __future__ import annotations`.
- `ruff` (line length 100) and `mypy` must pass; the config lives in `pyproject.toml`.
- Keep runtime dependencies at two (`numpy`, `pyarrow`). Anything else must be an optional extra
  with a written justification.
- CLI output stays stable and greppable: plain tables, JSON with `schema_version`, Markdown tables.

## Commits and pull requests

- Short imperative commit subjects ("Add NDCG cutoff test"), one logical change per commit.
- PR description: what changed, why, how it was verified (`bash scripts/check.sh` output), and any
  behaviour change to the JSON schema or CLI.
- By contributing you agree that your work is licensed under the project's MIT licence.
- Be decent in reviews and discussions; see [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## Good first issues

The issue tracker labels beginner-friendly work as `good first issue`. If you want to claim one,
comment on it so nobody duplicates the effort.
