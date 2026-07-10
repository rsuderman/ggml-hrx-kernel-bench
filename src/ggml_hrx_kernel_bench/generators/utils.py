"""Shared helpers for the asset code generators (copy, unary, ...)."""
from __future__ import annotations

from pathlib import Path
from string import Template


def repo_root() -> Path:
    # This module lives at src/ggml_hrx_kernel_bench/generators/utils.py.
    return Path(__file__).resolve().parents[3]


def load_template(*parts: str) -> Template:
    """Read a template file (path parts relative to the repo root) as a ``string.Template``."""
    return Template(repo_root().joinpath(*parts).read_text(encoding="utf-8"))
