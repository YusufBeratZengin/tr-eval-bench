"""Report renderers (terminal table, Markdown, diff)."""

from tr_eval_bench.reporters.diff_report import (
    build_diff,
    regression_failures,
    render_diff_markdown,
    render_diff_text,
)
from tr_eval_bench.reporters.markdown import render_leaderboard, render_markdown
from tr_eval_bench.reporters.text import render_table

__all__ = [
    "build_diff",
    "regression_failures",
    "render_diff_markdown",
    "render_diff_text",
    "render_leaderboard",
    "render_markdown",
    "render_table",
]
