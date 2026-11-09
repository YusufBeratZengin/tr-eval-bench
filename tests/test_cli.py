"""CLI tests: exit codes, formats, file output and regression gates."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from tr_eval_bench import cli
from tr_eval_bench.results import EvalResults


def _result(model: str, mrr: float = 0.5) -> EvalResults:
    return EvalResults(
        model=model,
        tasks=("sts", "retrieval"),
        sts_spearman=0.4,
        sts_pearson=0.4,
        retrieval_ndcg10=mrr + 0.05,
        retrieval_mrr10=mrr,
        retrieval_recall10=0.8,
        n_sts_pairs=100,
        n_queries=100,
        corpus_size=100,
        elapsed_seconds=0.5,
        dataset_version="tr-eval-mini-0.1.0",
    )


def test_run_prints_table(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["hashing:128", "--tasks", "retrieval", "--limit", "10"]) == 0
    output = capsys.readouterr().out
    assert "hashing:128" in output
    assert "MRR@10" in output


def test_run_json_format(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["hashing:64", "--tasks", "sts", "--limit", "10", "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["model"] == "hashing:64"
    assert payload["counts"]["sts_pairs"] == 10
    # The first ten pairs all share score 5.0, so Spearman is undefined and is
    # reported under undefined_metrics instead of leaking NaN into JSON.
    assert "sts_spearman" in {**payload["metrics"], **payload["undefined_metrics"]}
    assert "sts_spearman" in payload["undefined_metrics"]


def test_run_writes_output_and_markdown(tmp_path: Path) -> None:
    json_path = tmp_path / "reports" / "scores.json"
    md_path = tmp_path / "reports" / "scores.md"
    code = cli.main(
        [
            "hashing:64",
            "--tasks",
            "retrieval",
            "--limit",
            "5",
            "--format",
            "json",
            "--output",
            str(json_path),
            "--export-markdown",
            str(md_path),
            "--quiet",
        ]
    )
    assert code == 0
    assert json.loads(json_path.read_text(encoding="utf-8"))["model"] == "hashing:64"
    assert json_path.read_text(encoding="utf-8").endswith("\n")
    markdown = md_path.read_text(encoding="utf-8")
    assert markdown.startswith("# tr-eval-bench")
    assert "Retrieval-TR" in markdown


def test_run_markdown_format_to_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["hashing:64", "--tasks", "sts", "--limit", "5", "--format", "markdown"]) == 0
    assert "| Task | Metric | Score |" in capsys.readouterr().out


def test_fail_under_triggers_exit_one(capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(
        [
            "hashing:64",
            "--tasks",
            "retrieval",
            "--limit",
            "10",
            "--fail-under",
            "mrr=1.0",
            "--quiet",
        ]
    )
    assert code == 1
    assert "fail-under" in capsys.readouterr().err


def test_fail_under_missing_metric(capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(
        ["hashing:64", "--tasks", "sts", "--limit", "5", "--fail-under", "mrr=0.1", "--quiet"]
    )
    assert code == 1
    assert "was not computed" in capsys.readouterr().err


def test_fail_under_nan_metric_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    nan_result = replace(_result("m"), sts_spearman=math.nan, sts_pearson=math.nan)
    monkeypatch.setattr(cli, "evaluate_model", lambda model, **kwargs: nan_result)
    assert cli.main(["m", "--fail-under", "spearman=0.5", "--quiet"]) == 1
    error = capsys.readouterr().err
    assert "not a finite score (nan)" in error


def test_fail_under_nan_metric_passes_without_threshold(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    nan_result = replace(_result("m"), sts_spearman=math.nan, sts_pearson=math.nan)
    monkeypatch.setattr(cli, "evaluate_model", lambda model, **kwargs: nan_result)
    assert cli.main(["m", "--quiet"]) == 0
    assert capsys.readouterr().err == ""


def test_invalid_threshold_is_usage_error() -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["hashing:64", "--fail-under", "nope"])
    assert excinfo.value.code == 2
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["hashing:64", "--fail-under", "mrr=abc"])
    assert excinfo.value.code == 2
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["hashing:64", "--fail-under", "mrr=2"])
    assert excinfo.value.code == 2


def test_invalid_task_and_limit_are_usage_errors() -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["hashing:64", "--tasks", "sts,nope"])
    assert excinfo.value.code == 2
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["hashing:64", "--limit", "0"])
    assert excinfo.value.code == 2


def test_no_arguments_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main([]) == 2
    assert "usage: tr-eval-bench" in capsys.readouterr().out


def test_help_and_version_exit_zero() -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--help"])
    assert excinfo.value.code == 0
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["hashing:64", "--version"])
    assert excinfo.value.code == 0


def test_diff_runs_both_models(capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(["diff", "hashing:64", "hashing:128", "--limit", "10"])
    assert code == 0
    output = capsys.readouterr().out
    assert "baseline   hashing:64" in output
    assert "candidate  hashing:128" in output
    assert "WINNER" in output


def test_diff_markdown_survives_cp1252_redirected_stdout() -> None:
    """A redirected Windows-style cp1252 stdout must still produce valid markdown."""
    env = {**os.environ, "PYTHONIOENCODING": "cp1252"}
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from tr_eval_bench.cli import main; raise SystemExit(main())",
            "diff",
            "hashing:64",
            "hashing:128",
            "--tasks",
            "retrieval",
            "--limit",
            "5",
            "--format",
            "markdown",
        ],
        capture_output=True,
        env=env,
        timeout=300,
        check=False,
    )
    stderr = completed.stderr.decode("utf-8", "replace")
    assert completed.returncode == 0, stderr
    assert "Traceback" not in stderr
    assert "UnicodeEncodeError" not in stderr
    output = completed.stdout.decode("utf-8")
    assert "| Metric | Baseline | Candidate | Δ | Winner |" in output


def test_diff_markdown_export(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    md_path = tmp_path / "diff.md"
    code = cli.main(
        [
            "diff",
            "hashing:64",
            "hashing:128",
            "--tasks",
            "retrieval",
            "--limit",
            "10",
            "--format",
            "json",
            "--export-markdown",
            str(md_path),
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["candidate"]["model"] == "hashing:128"
    assert "diff" in md_path.read_text(encoding="utf-8")


def test_diff_gate_uses_injected_results(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fake_evaluate(model: str, **kwargs: object) -> EvalResults:
        return _result(model, mrr=0.7) if model == "base" else _result(model, mrr=0.4)

    monkeypatch.setattr(cli, "evaluate_model", fake_evaluate)
    code = cli.main(["diff", "base", "cand", "--max-drop", "mrr=0.1", "--quiet"])
    assert code == 1
    assert "regression: MRR@10 dropped by 0.3000" in capsys.readouterr().err


def test_diff_gate_passes_within_tolerance(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "evaluate_model", lambda model, **kwargs: _result(model, mrr=0.6))
    assert cli.main(["diff", "base", "cand", "--max-drop", "mrr=0.1", "--quiet"]) == 0
    assert capsys.readouterr().err == ""


def test_diff_fail_under_applies_to_candidate(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "evaluate_model", lambda model, **kwargs: _result(model, mrr=0.2))
    assert cli.main(["diff", "base", "cand", "--fail-under", "mrr=0.5", "--quiet"]) == 1
    assert "below the required" in capsys.readouterr().err


def test_diff_nan_score_fails_fail_under(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    nan_result = replace(_result("cand"), retrieval_mrr10=math.nan)
    monkeypatch.setattr(
        cli,
        "evaluate_model",
        lambda model, **kwargs: _result(model) if model == "base" else nan_result,
    )
    assert cli.main(["diff", "base", "cand", "--fail-under", "mrr=0.1", "--quiet"]) == 1
    assert "not a finite score (nan)" in capsys.readouterr().err


def test_diff_nan_score_fails_max_drop(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    nan_result = replace(_result("cand"), retrieval_mrr10=math.nan)
    monkeypatch.setattr(
        cli,
        "evaluate_model",
        lambda model, **kwargs: _result(model) if model == "base" else nan_result,
    )
    assert cli.main(["diff", "base", "cand", "--max-drop", "mrr=0.5", "--quiet"]) == 1
    assert "candidate score is not finite (nan)" in capsys.readouterr().err


def test_diff_max_drop_fails_when_metric_missing_on_both_sides(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    sts_only = EvalResults(model="m", tasks=("sts",), sts_spearman=0.5, sts_pearson=0.5)
    monkeypatch.setattr(cli, "evaluate_model", lambda model, **kwargs: sts_only)
    assert cli.main(["diff", "base", "cand", "--max-drop", "mrr=0.1", "--quiet"]) == 1
    assert "not computed for either model" in capsys.readouterr().err


def test_datasets_command(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["datasets"]) == 0
    output = capsys.readouterr().out
    assert "tr-eval-mini-0.1.0" in output
    assert "sha256:" in output


def test_datasets_verify_and_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["datasets", "--verify", "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["verified"] is True
    assert payload["sts_pairs"] == 100
    assert payload["corpus_passages"] == 100
    assert "sts_mini.parquet" in payload["files"]


def test_datasets_markdown(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["datasets", "--format", "markdown"]) == 0
    assert "| STS pairs | 100 |" in capsys.readouterr().out


def test_runtime_error_maps_to_exit_two(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["http://localhost:11434/v1", "--backend", "api"]) == 2
    assert "error:" in capsys.readouterr().err


def test_hybrid_flag_adds_scores(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["hashing:64", "--tasks", "retrieval", "--limit", "10", "--hybrid"]) == 0
    assert "Hybrid MRR@10" in capsys.readouterr().out
