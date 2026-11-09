# Security policy

## Supported versions

| Version | Supported |
| --- | --- |
| 0.1.x | yes |

## Reporting a vulnerability

Please do **not** open a public issue for security problems. Use GitHub's private
[Security Advisories](https://github.com/YusufBeratZengin/tr-eval-bench/security/advisories/new)
form. Include the version, a minimal reproduction and the impact you expect. You should get an
answer within seven days; please allow time for a fix and a release before public disclosure.

## Scope and threat model

`tr-eval-bench` is a local evaluation tool. It reads bundled Parquet files and calls embedding
backends you configure. Relevant risks and how they are handled:

- **Credentials.** API tokens are read only from the `TR_EVAL_API_KEY` environment variable. They
  are never accepted as CLI arguments (so they do not land in shell history or CI logs), never
  written to reports and never logged. Reports contain model names, scores and timings only.
- **Network exposure.** The only outbound requests are to the embedding endpoint you pass with
  `--api-url` (or the default `http://localhost:11434/v1`). Model downloads are performed by
  `sentence-transformers`/`huggingface_hub` under their own policies; pin a `revision` in your own
  code if you need reproducibility. There is no telemetry.
- **Untrusted input.** Datasets inside the wheel are checksummed (`tr-eval-bench datasets --verify`).
  The API runner validates response shape, ordering and embedding dimension, and fails closed with a
  clear error instead of scoring malformed vectors.
- **Remote code.** `trust_remote_code` is off by default; enabling it executes code from the model
  repository, so only do that with models you trust.
- **Denial of service.** Large models can exhaust RAM. Use `--batch-size 16` (or lower) and `--limit`
  for a quick check.

## Not in scope

- Vulnerabilities in third-party models, model hubs or inference servers (report those upstream).
- Accuracy or quality of benchmark scores — those are documented limitations, not security issues.
- Anything that requires running the tool on data you do not trust and then executing that data;
  the packaged data is text only and is never executed.
