# Pull request

## What changed

<!-- One or two sentences. Link the issue if there is one: "Closes #12". -->

## Why

<!-- The user-visible problem this solves. -->

## How it was verified

<!-- Paste the last lines of `bash scripts/check.sh`, or describe the extra manual check. -->

```
```

## Checklist

- [ ] `bash scripts/check.sh` exits 0 (lint, format, mypy, tests, ≥85% coverage)
- [ ] Tests added or updated for the change, including error paths
- [ ] No network access in tests (HTTP tests use the local fixture)
- [ ] No secrets, tokens or personal data added anywhere
- [ ] Docs updated when behaviour changed (`README.md`, `README.tr.md`, `CHANGELOG.md`)
- [ ] If datasets changed: `python scripts/build_datasets.py` was re-run and the Parquet files plus
      `MANIFEST.json` are committed together
- [ ] Runtime dependencies stay at `numpy` + `pyarrow` (or the new dependency is justified in the README)
- [ ] Commits are small and use imperative subjects
