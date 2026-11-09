# tr-eval-bench

**Score Turkish text-embedding and RAG-retrieval models on CPU in under a minute with cached models — offline, from one command.**

`tr-eval-bench` is a small, dependency-light benchmark kit for the question every Turkish RAG
project starts with: *which embedding model should I use?* It runs a curated 100-pair semantic
similarity set and a 100-query retrieval set (100-passage corpus) that ship inside the package, so
you get Spearman, NDCG@10, MRR@10 and Recall@10 without downloading gigabytes of data or renting a
GPU.

```
$ tr-eval-bench "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

model    sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
dataset  tr-eval-mini-0.1.0
data     STS 100 pairs | retrieval 100 queries / 100 passages

TASK          METRIC     SCORE
------------  ---------  ------
STS-TR        Spearman   0.8538
STS-TR        Pearson    0.8865
Retrieval-TR  NDCG@10    0.9457
Retrieval-TR  MRR@10     0.9348
Retrieval-TR  Recall@10  0.9800

elapsed  11.68s
```

That is a real run on an AMD Ryzen 9 5950X container (5 vCPU allocated, no GPU), Python 3.14,
model already in the Hugging Face cache. A first run that downloads the model takes longer; the
benchmark itself is the ~2 s that remain after loading.

## Install

```bash
pip install tr-eval-bench              # core: NumPy + PyArrow only
pip install "tr-eval-bench[hf]"        # + sentence-transformers for Hugging Face models
```

Python 3.10+ on Linux, macOS or Windows. No API keys are required for local models.

## Why this exists

- The academic Turkish benchmark ([`selmanbaysan/mteb_tr`](https://github.com/selmanbaysan/mteb_tr),
  MTEB-TR) is thorough but downloads gigabytes of data, needs a GPU and takes hours — too heavy for
  a quick model choice or a CI gate.
- The public [MTEB Turkish leaderboard](https://huggingface.co/spaces/alibayram/mteb-turkish)
  tracks large, well-known models; teams running their own small fine-tunes have no fast local way
  to see whether a new checkpoint actually improved.
- `tr-eval-bench` trades coverage for speed: 200 labelled items, bundled, CPU-only, offline, one
  command, machine-readable output.

It complements MTEB-TR; it does not replace it. See [Limitations](#limitations-and-honesty).

## What it measures

| Task | Items | Metric | What it tells you |
| --- | ---: | --- | --- |
| `sts` — STS-TR | 100 graded pairs (0–5) | Spearman, Pearson | Does the model order sentence similarity the way humans do? |
| `retrieval` — Retrieval-TR | 100 queries / 100 passages | NDCG@10, MRR@10, Recall@10 | Can the model find the one passage that answers a Turkish question? |
| `--hybrid` (optional) | same retrieval set | NDCG@10, MRR@10, Recall@10 | Does BM25 + dense fusion (RRF) beat dense retrieval alone? |

Metric definitions are the standard ones and are unit-tested against hand-computed values:

- **Spearman** — Pearson correlation of average ranks (ties averaged).
- **NDCG@10** — `DCG = Σ rel_i / log2(i+1)`, linear gains, ideal ranking from the query's labels,
  `NDCG = DCG / IDCG` (0 when a query has no relevant document).
- **MRR@10** — reciprocal rank of the first relevant result within the top 10, averaged over queries.
- **Recall@10** — share of relevant passages found in the top 10.
- **Hybrid** — BM25 (`k1=1.5`, `b=0.75`) fused with the dense ranking by Reciprocal Rank Fusion
  (`score = Σ 1/(60 + rank)`), then scored with the same metrics.

Scores are computed on L2-normalized embeddings with cosine similarity.

## CLI

```bash
tr-eval-bench MODEL [options]                 # evaluate one model
tr-eval-bench diff BASELINE CANDIDATE [opts]  # compare two models metric by metric
tr-eval-bench datasets [--verify]             # dataset counts + checksums
```

`MODEL` is a Hugging Face model id, an API model name (with `--backend api`), or `hashing[:dim]`
for the built-in lexical baseline.

| Option | Default | Meaning |
| --- | --- | --- |
| `--backend {auto,hf,api,hashing}` | `auto` | `auto` uses `hashing` for `hashing*` specs, Hugging Face otherwise. |
| `--device` | `cpu` | Torch device for Hugging Face models (`cpu`, `cuda`, `mps`). |
| `--api-url` | `$TR_EVAL_API_URL` or `http://localhost:11434/v1` | OpenAI-compatible endpoint. |
| `--batch-size` | `32` | Embedding batch size; lower it if a large model runs out of RAM. |
| `--tasks` | `sts,retrieval` | Run a subset of tasks. |
| `--limit` | off | Score only the first N items (quick smoke runs). |
| `--hybrid` | off | Also report BM25+dense RRF scores. |
| `--rrf-k` | `60` | RRF constant. |
| `--format {table,json,markdown}` | `table` | Report format. |
| `--output PATH` | off | Write the rendered report to a file. |
| `--export-markdown PATH` | off | Also write a Markdown report. |
| `--fail-under METRIC=VALUE` | off | Exit 1 when a metric is below the threshold (repeatable). |
| `--max-drop METRIC=VALUE` | off | `diff` only: exit 1 when the candidate drops more than VALUE. |
| `-q`, `--quiet` | off | Do not print the report (useful with `--output`). |

Metric aliases accepted by `--fail-under` / `--max-drop`: `spearman`, `pearson`, `ndcg`/`ndcg10`,
`mrr`/`mrr10`, `recall`/`recall10`, `hybrid_ndcg`, `hybrid_mrr`.

Exit codes: `0` success, `1` a gate failed, `2` configuration/runtime error.

### Local API servers (Ollama, vLLM, TGI, llama.cpp)

The API runner speaks the OpenAI-compatible `POST /v1/embeddings` protocol. Ollama exposes it on
`/v1`; vLLM and TGI serve it too.

```bash
ollama pull nomic-embed-text
tr-eval-bench nomic-embed-text --backend api --api-url http://localhost:11434/v1
```

Credentials, when a server needs them, come from `TR_EVAL_API_KEY` — never from the command line, so
keys do not end up in shell history or CI logs.

### Regression gate in CI

```bash
# Fail if the candidate model is worse than the current one on retrieval
tr-eval-bench diff old-model new-model --max-drop mrr=0.02 --max-drop ndcg=0.02 --quiet

# Fail if a fine-tune dropped below an absolute floor
tr-eval-bench ./checkpoints/my-model --fail-under spearman=0.80 --format json --output scores.json
```

### Diff output

```
$ tr-eval-bench diff hashing:256 hashing:1024 --format markdown

### tr-eval-bench diff

Baseline: `hashing:256`  
Candidate: `hashing:1024`

| Metric | Baseline | Candidate | Δ | Winner |
| --- | ---: | ---: | ---: | --- |
| Spearman | 0.7753 | 0.7727 | -0.0026 | baseline |
| Pearson | 0.7769 | 0.7887 | +0.0118 | candidate |
| NDCG@10 | 0.9748 | 0.9873 | +0.0125 | candidate |
| MRR@10 | 0.9674 | 0.9836 | +0.0162 | candidate |
| Recall@10 | 1.0000 | 1.0000 | +0.0000 | tie |

Candidate is better on **3/5** metrics.
```

## Library API

```python
from tr_eval_bench import evaluate_model, compare_models

results = evaluate_model("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
print(results.sts_spearman)      # 0.8538...
print(results.retrieval_mrr10)   # 0.9348...
print(results.to_dict())         # JSON-ready payload (schema_version 1.1)

diff = compare_models("hashing:256", "hashing:1024", tasks="retrieval")
print(diff.rows[3].metric, diff.rows[3].delta)
```

`evaluate_model` accepts `tasks`, `backend`, `device`, `batch_size`, `limit`, `hybrid`, `rrf_k`,
`api_url`, and `runner` (inject any object with `name`, `dimension` and
`embed(texts, batch_size=...)`).

## Bundled datasets

| File | Rows | Content |
| --- | ---: | --- |
| `sts_mini.parquet` | 100 | Graded sentence pairs (0–5) across 10 domains |
| `rag_mini.parquet` | 100 | Question + graded qrels (`passage_id`, `relevance`) |
| `rag_corpus.parquet` | 100 | Passage id, title, text, domain |
| `MANIFEST.json` | — | Version, counts, SHA-256 checksums |

- **License:** MIT, like the code. The text is original, written for this project from widely known
  general knowledge; no external corpus was copied, there is no personal data and no real person
  names.
- **Offline:** the Parquet files are inside the wheel. `tr-eval-bench datasets --verify` checks
  their checksums; nothing is downloaded at evaluation time.
- **Rebuild from source:** `python scripts/build_datasets.py` (validates ids, score range,
  distribution and rewrites the manifest).
- Full provenance and the licensing rationale: [`tr_eval_bench/datasets/PROVENANCE.md`](tr_eval_bench/datasets/PROVENANCE.md).

## Measured results (reference machine)

AMD Ryzen 9 5950X, 5 vCPU, no GPU, Python 3.14, `tr-eval-bench 0.1.0`:

| Model | STS Spearman | NDCG@10 | MRR@10 | Wall time |
| --- | ---: | ---: | ---: | ---: |
| `hashing:256` (lexical baseline) | 0.7753 | 0.9748 | 0.9674 | 0.2 s |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 0.8538 | 0.9457 | 0.9348 | 18.9 s |
| `emrecan/bert-base-turkish-cased-mean-nli-stsb-tr` | 0.8920 | 0.9795 | 0.9725 | 29.2 s |

Wall time is measured end-to-end (process start of the evaluation call, including model loading)
with both models already in the Hugging Face cache on the reference machine. Warm-cache runs of
the compact model measured 11.7 s inside `elapsed_seconds` (after loading); the 18.9 s column adds
the load. A cold download or a busy 5-vCPU container can take considerably longer — the earlier
"under a minute" target describes cached-model runs, not first downloads.

> Note the lexical baseline: the retrieval questions are factoid and share vocabulary with their
> gold passage, so `hashing` is competitive there while losing clearly on STS. Use `sts` to separate
> good models and `retrieval` to catch broken pipelines — see [Limitations](#limitations-and-honesty).

## Dependencies, and why

| Package | Role | Why not stdlib |
| --- | --- | --- |
| `numpy` | similarity matrices, rank statistics | Pure-Python dot products for 100×768 matrices are 20–50× slower and simpler to get wrong. |
| `pyarrow` | read the bundled Parquet files | Parquet is compact, typed and checksummed; a JSON mirror would duplicate the data. |
| `sentence-transformers` (extra) | local Hugging Face models | Only needed for the `hf` backend; the API and `hashing` backends work without it. |

Everything else — CLI parsing, HTTP, JSON — is standard library.

## Limitations and honesty

- **Mini, not official.** 100 STS pairs and 100 retrieval queries are a fast signal. They do not
  reproduce MTEB or TR-MTEB rankings and must not be cited as such.
- **Retrieval saturates.** The corpus is small and the questions are factoid, so every working model
  scores high (MRR@10 ≈ 0.93–0.98) and the lexical baseline is competitive. Use retrieval scores to
  catch broken pipelines and regressions, and `sts`/`--hybrid` plus `diff` to separate good models.
  A hard-negative corpus is the top item on the roadmap.
- **Turkish only.** The bundled data is Turkish; the code itself is language-agnostic.
- **Single-turn QA.** No multi-hop, long-document or cross-lingual retrieval yet.
- **API protocol.** Only OpenAI-compatible `/v1/embeddings` is supported, not Ollama's native
  `/api/embeddings`.
- **Timings** depend on CPU, model size and whether the model is already cached.

## Development

```bash
git clone https://github.com/YusufBeratZengin/tr-eval-bench
cd tr-eval-bench
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
bash scripts/check.sh        # ruff lint + format, mypy, pytest with 85% coverage gate
```

Tests never touch the network: the API runner is exercised against a local threaded HTTP server, and
Hugging Face models are stubbed. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Roadmap

- 0.2 — hard-negative retrieval corpus, classification and clustering mini tasks
- 0.3 — reranker (cross-encoder) evaluation, cross-lingual TR→EN retrieval
- Later — distractor-resistance set, embedding quantization/latency profiler, HF Space JSON export

## License

MIT © 2026 Yusuf Berat Zengin. See [LICENSE](LICENSE).
