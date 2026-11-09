# Bundled dataset provenance

## What ships

| File | Rows | Columns |
| --- | ---: | --- |
| `sts_mini.parquet` | 100 | `pair_id`, `sentence1`, `sentence2`, `score` (0–5), `domain` |
| `rag_mini.parquet` | 100 | `query_id`, `query`, `passage_id`, `relevance`, `domain` |
| `rag_corpus.parquet` | 100 | `passage_id`, `title`, `text`, `domain` |
| `MANIFEST.json` | — | dataset name, license, counts, SHA-256 of each Parquet file |

Every query has exactly one relevant passage (`relevance = 1.0`). Domains: geography,
history, science, space, technology, nature, culture, literature, sport, language
(10 passages each, 30/19/16/12/7/16 STS pairs per score bucket from 5 down to 0).

## How the text was produced

- The sentences and passages were **written for this project** from widely known general
  knowledge. No corpus, website, book or dataset was copied, scraped or paraphrased
  mechanically, so there is no upstream licence to carry.
- Facts were kept to stable, non-numeric general knowledge (e.g. "water boils at 100 °C at sea
  level", "Jüpiter is the largest planet", "cumhuriyet was proclaimed in 1923"). The dataset
  deliberately contains **no official statistic, tax rate, legal value, price or forecast**; if such
  a value were ever needed it would be marked `NEEDS_SOURCE` and excluded instead of guessed.
- **No personal data and no real person names**: no names of individuals (public or private), no
  addresses, no user content, no credentials. People are described by role ("Osmanlı ordusu",
  "halk şairleri") or not mentioned; places, works and institutions keep their usual names.

## Licence

The datasets are released under the MIT licence, the same as the code (see `LICENSE` at the
repository root). You may redistribute and modify them with attribution.

## Rebuilding and verification

```bash
python scripts/build_datasets.py     # validates sources, rewrites Parquet + MANIFEST.json
tr-eval-bench datasets --verify      # checks SHA-256 against the manifest
```

`scripts/data_passages.py` and `scripts/data_sts.py` are the human-editable sources; the builder
fails on duplicate ids, missing questions, out-of-range scores, unbalanced domains or a skewed STS
distribution.

## Known limitations

- 100 items per task is a smoke-test scale. Retrieval scores saturate for competent models and the
  lexical `hashing` baseline is competitive; the set is meant to catch broken pipelines and
  regressions, not to rank the top of a leaderboard.
- One gold passage per query means Recall@10 cannot exceed 1.0 by construction.
- Turkish only, single-turn factoid questions; no multi-hop or long-document cases yet.
