"""Command line interface for tr-eval-bench.

Examples::

    tr-eval-bench "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    tr-eval-bench hashing:512 --tasks retrieval --format markdown
    tr-eval-bench diff model-a model-b --format markdown
    tr-eval-bench ollama/nomic-embed-text --backend api --api-url http://localhost:11434/v1
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path

from tr_eval_bench import __version__, evaluate_model
from tr_eval_bench.datasets import (
    DATASET_VERSION,
    DatasetError,
    dataset_info,
    file_digests,
    verify_datasets,
)
from tr_eval_bench.reporters import (
    build_diff,
    regression_failures,
    render_diff_markdown,
    render_diff_text,
    render_markdown,
    render_table,
)
from tr_eval_bench.results import METRIC_LABELS, DiffReport, EvalResults
from tr_eval_bench.runners import RunnerError

FORMATS = ("table", "json", "markdown")
COMMANDS = ("run", "diff", "datasets")

METRIC_ALIASES: dict[str, str] = {
    "spearman": "sts_spearman",
    "sts": "sts_spearman",
    "sts_spearman": "sts_spearman",
    "pearson": "sts_pearson",
    "sts_pearson": "sts_pearson",
    "ndcg": "retrieval_ndcg10",
    "ndcg10": "retrieval_ndcg10",
    "retrieval_ndcg10": "retrieval_ndcg10",
    "mrr": "retrieval_mrr10",
    "mrr10": "retrieval_mrr10",
    "retrieval_mrr10": "retrieval_mrr10",
    "recall": "retrieval_recall10",
    "recall10": "retrieval_recall10",
    "retrieval_recall10": "retrieval_recall10",
    "hybrid_ndcg": "hybrid_ndcg10",
    "hybrid_ndcg10": "hybrid_ndcg10",
    "hybrid_mrr": "hybrid_mrr10",
    "hybrid_mrr10": "hybrid_mrr10",
}


def _threshold(raw: str) -> tuple[str, float]:
    """Parse ``metric=value`` CLI thresholds into ``(canonical_metric, value)``."""
    if "=" not in raw:
        raise argparse.ArgumentTypeError(f"expected METRIC=VALUE (e.g. mrr=0.35), got {raw!r}")
    name, _, value = raw.partition("=")
    key = METRIC_ALIASES.get(name.strip().lower())
    if key is None:
        raise argparse.ArgumentTypeError(
            f"unknown metric {name.strip()!r}; use one of: {', '.join(sorted(METRIC_ALIASES))}"
        )
    try:
        parsed = float(value.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid threshold value in {raw!r}") from exc
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("thresholds must be within [0, 1]")
    return key, parsed


def _positive_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {raw!r}") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _tasks(raw: str) -> str:
    allowed = {"sts", "retrieval"}
    requested = {chunk.strip() for chunk in raw.split(",") if chunk.strip()}
    unknown = requested - allowed
    if unknown or not requested:
        raise argparse.ArgumentTypeError(
            f"tasks must be a comma-separated subset of sts,retrieval; got {raw!r}"
        )
    return ",".join(sorted(requested))


def _add_eval_options(parser: argparse.ArgumentParser, *, hybrid_default: bool = False) -> None:
    parser.add_argument(
        "--backend",
        choices=("auto", "hf", "api", "hashing"),
        default="auto",
        help="embedding backend (auto picks hf unless the spec is hashing[:dim])",
    )
    parser.add_argument("--device", default="cpu", help="torch device for Hugging Face models")
    parser.add_argument(
        "--api-url",
        default=None,
        help="OpenAI-compatible endpoint base URL (default: $TR_EVAL_API_URL or "
        "http://localhost:11434/v1)",
    )
    parser.add_argument(
        "--batch-size", type=_positive_int, default=32, help="embedding batch size (default: 32)"
    )
    parser.add_argument(
        "--tasks", type=_tasks, default="sts,retrieval", help="comma-separated tasks to run"
    )
    parser.add_argument(
        "--limit", type=_positive_int, default=None, help="score only the first N items"
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        default=hybrid_default,
        help="also report BM25+dense RRF fusion scores (retrieval task)",
    )
    parser.add_argument(
        "--rrf-k", type=_positive_int, default=60, help="RRF constant (default: 60)"
    )
    parser.add_argument("--format", choices=FORMATS, default="table", help="output format")
    parser.add_argument(
        "--output", type=Path, default=None, help="write the rendered report to a file"
    )
    parser.add_argument(
        "--export-markdown",
        type=Path,
        default=None,
        help="also write a Markdown report (leaderboard-ready)",
    )
    parser.add_argument(
        "--fail-under",
        type=_threshold,
        action="append",
        default=[],
        metavar="METRIC=VALUE",
        help="exit 1 when a metric is below VALUE (repeatable, e.g. --fail-under mrr=0.4)",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="do not print the report")
    parser.add_argument("--version", action="version", version=f"tr-eval-bench {__version__}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tr-eval-bench",
        description="Benchmark Turkish text embeddings on STS and RAG retrieval, offline, on CPU.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="evaluate one model")
    run.add_argument("model", help="Hugging Face model id, API model name, or hashing[:dim]")
    _add_eval_options(run)

    diff = subparsers.add_parser("diff", help="compare a baseline and a candidate model")
    diff.add_argument("baseline", help="baseline model spec")
    diff.add_argument("candidate", help="candidate model spec")
    _add_eval_options(diff)
    diff.add_argument(
        "--max-drop",
        type=_threshold,
        action="append",
        default=[],
        metavar="METRIC=VALUE",
        help="exit 1 when the candidate drops more than VALUE on a metric (repeatable)",
    )

    datasets = subparsers.add_parser("datasets", help="show bundled dataset information")
    datasets.add_argument("--verify", action="store_true", help="verify Parquet checksums")
    datasets.add_argument("--format", choices=FORMATS, default="table", help="output format")
    return parser


def _configure_output_streams() -> None:
    """Force UTF-8 on redirected stdout/stderr so non-ASCII reports never crash.

    On Windows, redirected output keeps the locale encoding (often cp1252),
    which cannot represent characters such as the ``Δ`` in the markdown diff
    header. Real consoles use a wide-character API, so only text streams whose
    encoding is not already Unicode get reconfigured; files and pipes then
    receive valid UTF-8.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        raw_encoding = getattr(stream, "encoding", None) or ""
        encoding = raw_encoding.lower().replace("-", "").replace("_", "")
        if reconfigure is None or encoding.startswith(("utf8", "utf16", "utf32")):
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):  # pragma: no cover - exotic stream wrappers
            continue


def _render(result: EvalResults, output_format: str) -> str:
    if output_format == "json":
        return json.dumps(result.to_dict(), ensure_ascii=False, indent=2, allow_nan=False)
    if output_format == "markdown":
        return render_markdown(result)
    return render_table(result)


def _render_diff(report: DiffReport, output_format: str) -> str:
    if output_format == "json":
        return json.dumps(report.to_dict(), ensure_ascii=False, indent=2, allow_nan=False)
    if output_format == "markdown":
        return render_diff_markdown(report)
    return render_diff_text(report)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _emit(text: str, *, quiet: bool) -> None:
    if not quiet:
        print(text)


def _check_fail_under(result: EvalResults, thresholds: Sequence[tuple[str, float]]) -> list[str]:
    failures: list[str] = []
    for metric, minimum in thresholds:
        value = result.metric(metric)
        label = METRIC_LABELS[metric]
        if value is None:
            failures.append(f"{label} was not computed (is its task enabled?)")
        elif not math.isfinite(value):
            failures.append(
                f"{label} is not a finite score (nan), so the {minimum:.4f} threshold cannot be met"
            )
        elif value < minimum:
            failures.append(f"{label} {value:.4f} is below the required {minimum:.4f}")
    return failures


def _cmd_run(args: argparse.Namespace) -> int:
    result = evaluate_model(
        args.model,
        tasks=args.tasks,
        backend=args.backend,
        device=args.device,
        batch_size=args.batch_size,
        limit=args.limit,
        hybrid=args.hybrid,
        rrf_k=args.rrf_k,
        api_url=args.api_url,
    )
    text = _render(result, args.format)
    _emit(text, quiet=args.quiet)
    if args.output:
        _write(args.output, text)
    if args.export_markdown:
        _write(args.export_markdown, render_markdown(result))
    failures = _check_fail_under(result, args.fail_under)
    for message in failures:
        print(f"fail-under: {message}", file=sys.stderr)
    return 1 if failures else 0


def _cmd_diff(args: argparse.Namespace) -> int:
    shared: dict[str, object] = {
        "tasks": args.tasks,
        "backend": args.backend,
        "device": args.device,
        "batch_size": args.batch_size,
        "limit": args.limit,
        "hybrid": args.hybrid,
        "rrf_k": args.rrf_k,
        "api_url": args.api_url,
    }
    baseline = evaluate_model(args.baseline, **shared)  # type: ignore[arg-type]
    candidate = evaluate_model(args.candidate, **shared)  # type: ignore[arg-type]
    report = build_diff(baseline, candidate)
    text = _render_diff(report, args.format)
    _emit(text, quiet=args.quiet)
    if args.output:
        _write(args.output, text)
    if args.export_markdown:
        _write(args.export_markdown, render_diff_markdown(report))
    failures = _check_fail_under(candidate, args.fail_under)
    failures.extend(regression_failures(report, args.max_drop))
    for message in failures:
        print(f"regression: {message}", file=sys.stderr)
    return 1 if failures else 0


def _cmd_datasets(args: argparse.Namespace) -> int:
    info = dataset_info()
    digests = file_digests()
    if args.verify:
        verify_datasets()
    payload = {**info, "files": digests, "verified": bool(args.verify)}
    if args.format == "json":
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    if args.format == "markdown":
        print(f"# Bundled datasets `{DATASET_VERSION}`\n")
        print("| Item | Count |")
        print("| --- | ---: |")
        print(f"| STS pairs | {info['sts_pairs']} |")
        print(f"| Retrieval queries | {info['retrieval_queries']} |")
        print(f"| Corpus passages | {info['corpus_passages']} |")
        return 0
    print(f"dataset      {DATASET_VERSION}")
    print(f"sts pairs    {info['sts_pairs']}")
    print(f"queries      {info['retrieval_queries']}")
    print(f"passages     {info['corpus_passages']}")
    sts_domains = ", ".join(info["sts_domains"])  # type: ignore[arg-type]
    rag_domains = ", ".join(info["retrieval_domains"])  # type: ignore[arg-type]
    print(f"sts domains  {sts_domains}")
    print(f"rag domains  {rag_domains}")
    for name, digest in digests.items():
        print(f"{name:<22} sha256:{digest[:16]}…")
    if args.verify:
        print("checksums    OK")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns a process exit code (0 success, 1 gate failure, 2 error)."""
    _configure_output_streams()
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if not raw_args:
        build_parser().print_help()
        return 2
    if raw_args[0] not in (*COMMANDS, "-h", "--help") and not raw_args[0].startswith("-"):
        raw_args = ["run", *raw_args]
    args = build_parser().parse_args(raw_args)
    try:
        if args.command == "run":
            return _cmd_run(args)
        if args.command == "diff":
            return _cmd_diff(args)
        return _cmd_datasets(args)
    except (RunnerError, DatasetError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
