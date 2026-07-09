"""Code generator for the pointwise-unary v2 kernels + routes.

Like ``generators/copy.py``, this is part of the build pipeline: ``materialized_assets.py`` calls
``render_kernel_artifacts()`` / ``render_catalog_artifacts()`` during ``materialize_asset_root`` and
injects ``router_route_lists()`` into the materialized ``router.json``. The generated
``kernels/v2/<op>/*.loom`` and ``catalog/v2/<op>/*.json`` live only under ``build/generated/assets``;
nothing here is checked into git except this module and the templates.

The base files are the ``.tmpl`` templates (readable as ordinary kernels/routes with ``$`` holes):
  - ``kernels/v2/pointwise/{contiguous_4d,non_contiguous_4d}.loom.tmpl``  (holes: ``$op``, ``$preamble``, ``$compute_block``)
  - ``catalog/v2/pointwise/{contiguous_4d,non_contiguous_4d}.json.tmpl``  (holes: ``$route_id``, ``$family``, ``$source_id``, ``$kernel_path``, ``$root_symbol``, ``$export_name``, ``$dtype``)
The whole family differs only by op name and a single compute snippet, so each op is one ``UNARY_OPS``
row substituted into those templates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .utils import load_template, repo_root

CONTIGUOUS = "contiguous_4d"
NON_CONTIGUOUS = "non_contiguous_4d"

DTYPES = (("f16", "F16"), ("f32", "F32"))


@dataclass(frozen=True)
class UnaryOp:
    # f32 snippet producing %result from %value inside scf.if (4-space indented; may span lines). The
    # f16 path up-converts to f32, runs this, then truncates back to f16 (loom scalar math is f32).
    compute_block: str
    variants: tuple[str, ...]
    # Optional launch-body lines emitted before scf.if (e.g. relu's shared %zero constant). Each line
    # must include its own trailing newline; "" emits nothing.
    preamble: str = ""


UNARY_OPS: dict[str, UnaryOp] = {
    "abs": UnaryOp(
        "    %result = scalar.absf %value : f32", (CONTIGUOUS, NON_CONTIGUOUS)
    ),
    "exp": UnaryOp(
        "    %result = scalar.expf<afn> %value : f32", (CONTIGUOUS, NON_CONTIGUOUS)
    ),
    "neg": UnaryOp(
        "    %result = scalar.negf<nnan|ninf|nsz> %value : f32",
        (CONTIGUOUS, NON_CONTIGUOUS),
    ),
    "relu": UnaryOp(
        "    %result = scalar.maxnumf %value, %zero : f32",
        (CONTIGUOUS, NON_CONTIGUOUS),
        preamble="  %zero = scalar.constant 0.0 : f32\n",
    ),
    "sqr": UnaryOp(
        "    %result = scalar.mulf<nnan|ninf|nsz> %value, %value : f32", (CONTIGUOUS,)
    ),
    "sqrt": UnaryOp("    %result = scalar.sqrtf %value : f32", (CONTIGUOUS,)),
}


def _route_symbol(op: str, dt: str, variant: str) -> str:
    return f"{op}_{dt}" if variant == CONTIGUOUS else f"{op}_{dt}_non_contiguous_4d"


def _route_relpath(op: str, dt: str, variant: str) -> str:
    name = f"{op}_{dt}" if variant == CONTIGUOUS else f"{op}_{dt}_non_contiguous_4d"
    return f"{op}/{name}.json"


def render_kernel_artifacts() -> dict[Path, str]:
    """Kernel .loom contents keyed by path relative to ``kernels/v2``."""
    out: dict[Path, str] = {}
    for op, spec in UNARY_OPS.items():
        for variant in spec.variants:
            out[Path(op) / f"{variant}.loom"] = load_template(
                "kernels", "v2", "pointwise", f"{variant}.loom.tmpl"
            ).substitute(op=op, preamble=spec.preamble, compute_block=spec.compute_block)
    return out


def render_catalog_artifacts() -> dict[Path, str]:
    """Route .json contents keyed by path relative to ``catalog/v2``."""
    out: dict[Path, str] = {}
    for op, spec in UNARY_OPS.items():
        for variant in spec.variants:
            template = load_template(
                "catalog", "v2", "pointwise", f"{variant}.json.tmpl"
            )
            for dt, DT in DTYPES:
                sym = _route_symbol(op, dt, variant)
                out[Path(_route_relpath(op, dt, variant))] = template.substitute(
                    route_id=(
                        f"{op}_{dt}_contiguous_4d" if variant == CONTIGUOUS else sym
                    ),
                    family=f"{op}_{dt}",
                    source_id=f"pointwise_{dt}",
                    kernel_path=f"{op}/{variant}.loom",
                    root_symbol=f"@{sym}",
                    export_name=sym,
                    dtype=DT,
                )
    return out


def router_route_lists() -> dict[str, list[str]]:
    """Router ``routes`` entries (op key -> ordered catalog-relative route paths) to inject."""
    lists: dict[str, list[str]] = {}
    for op, spec in UNARY_OPS.items():
        paths: list[str] = []
        for variant in spec.variants:
            for dt, _DT in DTYPES:
                paths.append(_route_relpath(op, dt, variant))
        lists[op.upper()] = paths
    return lists


def generator_input_paths() -> tuple[Path, ...]:
    """Files whose changes should retrigger materialization (for mtime tracking)."""
    return (
        Path(__file__).resolve(),
        *sorted((repo_root() / "kernels" / "v2" / "pointwise").glob("*.tmpl")),
        *sorted((repo_root() / "catalog" / "v2" / "pointwise").glob("*.tmpl")),
    )
