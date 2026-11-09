#!/usr/bin/env bash
# Single quality gate for tr-eval-bench: lint, formatting, types, tests, coverage.
# Usage (from anywhere): bash scripts/check.sh
# Environment: PYTHON=/path/to/python to force an interpreter.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

CYAN=""; RESET=""
if [ -t 1 ]; then CYAN="$(printf '\033[36m')"; RESET="$(printf '\033[0m')"; fi

step() { printf '%s==> %s%s\n' "$CYAN" "$1" "$RESET"; }

PYTHON="${PYTHON:-python3}"
if [ -x ".venv/bin/python" ]; then
  PYTHON=".venv/bin/python"
elif [ -x ".venv/Scripts/python.exe" ]; then
  PYTHON=".venv/Scripts/python.exe"
fi

if ! command -v "$PYTHON" >/dev/null 2>&1 && [ ! -x "$PYTHON" ]; then
  echo "error: no Python interpreter found (set PYTHON=/path/to/python)" >&2
  exit 1
fi

if ! "$PYTHON" -c "import pytest, coverage, ruff, mypy" >/dev/null 2>&1; then
  step "Installing development dependencies into .venv"
  if [ ! -x ".venv/bin/python" ] && [ ! -x ".venv/Scripts/python.exe" ]; then
    "$PYTHON" -m venv .venv
    if [ -x ".venv/bin/python" ]; then
      PYTHON=".venv/bin/python"
    else
      PYTHON=".venv/Scripts/python.exe"
    fi
  fi
  "$PYTHON" -m pip install --quiet --upgrade pip
  "$PYTHON" -m pip install --quiet -e ".[dev]"
fi

step "Python: $("$PYTHON" -c 'import sys; print(sys.version.split()[0], sys.executable)')"

step "Dataset checksums"
"$PYTHON" -c "from tr_eval_bench.datasets import verify_datasets, dataset_info; verify_datasets(); info = dataset_info(); print('datasets OK:', info['version'], info['sts_pairs'], 'STS pairs,', info['retrieval_queries'], 'queries,', info['corpus_passages'], 'passages')"

step "ruff lint"
"$PYTHON" -m ruff check tr_eval_bench tests scripts

step "ruff format"
"$PYTHON" -m ruff format --check tr_eval_bench tests scripts

step "mypy"
"$PYTHON" -m mypy

step "pytest with coverage (gate: 85%)"
"$PYTHON" -m pytest -q --cov=tr_eval_bench --cov-report=term-missing --cov-fail-under=85

step "All checks passed"
