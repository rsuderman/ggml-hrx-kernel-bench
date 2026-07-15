"""Code generator for pointwise-binary v2 kernels and routes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .utils import load_template, repo_root

DTYPES = (("f16", "F16"), ("f32", "F32"))
OPS_BY_KEY = (
    ("ADD", "add"),
    ("DIV", "div"),
    ("MUL", "mul"),
    ("SUB", "sub"),
)
PLAIN_VARIANTS = (
    "generic_2d",
    "contiguous_2d",
    "rhs_row_broadcast_2d",
    "contiguous_1d",
    "generic_4d",
)
ROUTER_VARIANTS_BY_DTYPE = (
    ("generic_2d", "f16"),
    ("generic_2d", "f32"),
    ("contiguous_2d", "f16"),
    ("rhs_row_broadcast_2d", "f16"),
    ("contiguous_1d", "f16"),
    ("generic_4d", "f16"),
    ("contiguous_2d", "f32"),
    ("rhs_row_broadcast_2d", "f32"),
    ("contiguous_1d", "f32"),
    ("generic_4d", "f32"),
)

SPECIAL_STATIC_ROUTES_BY_KEY = {
    "MUL": ("mul/rms_norm_mul_f32_n16_r60_vector_tail.json",),
}


@dataclass(frozen=True)
class BinaryOp:
    scalar_op: str

    @property
    def compute_expr(self) -> str:
        return f"scalar.{self.scalar_op}<nnan|ninf|nsz> %a, %b : f32"


BINARY_OPS: dict[str, BinaryOp] = {
    "add": BinaryOp("addf"),
    "div": BinaryOp("divf"),
    "mul": BinaryOp("mulf"),
    "sub": BinaryOp("subf"),
}


def _route_id(op: str, dt: str, variant: str) -> str:
    return f"{op}_{dt}_{variant}"


def _route_relpath(op: str, dt: str, variant: str) -> str:
    return f"{op}/{_route_id(op, dt, variant)}.json"


def _kernel_path(op: str, variant: str) -> str:
    return f"{op}/{variant}.loom"


def _symbol(op: str, dt: str, variant: str) -> str:
    return f"{op}_{dt}" if variant == "generic_2d" else _route_id(op, dt, variant)


def _source_id(op: str, dt: str, variant: str) -> str:
    if variant in {"generic_2d", "contiguous_2d", "rhs_row_broadcast_2d"}:
        return f"{op}_pointwise_{dt}"
    return f"{op}_{dt}"


def render_kernel_artifacts() -> dict[Path, str]:
    out: dict[Path, str] = {}
    for op in BINARY_OPS:
        spec = BINARY_OPS[op]
        for variant in PLAIN_VARIANTS:
            rendered = load_template(
                "kernels", "v2", "binary", f"{variant}.loom.tmpl"
            ).substitute(
                op=op,
                f32_compute_expr=spec.compute_expr,
                f16_compute_expr=spec.compute_expr,
            )
            out[Path(_kernel_path(op, variant))] = rendered
    return out


def render_catalog_artifacts() -> dict[Path, str]:
    out: dict[Path, str] = {}
    for op in BINARY_OPS:
        for variant in PLAIN_VARIANTS:
            template = load_template("catalog", "v2", "binary", f"{variant}.json.tmpl")
            for dt, DT in DTYPES:
                symbol = _symbol(op, dt, variant)
                rendered = template.substitute(
                    op=op,
                    route_id=_route_id(op, dt, variant),
                    family=f"{op}_{dt}",
                    source_id=_source_id(op, dt, variant),
                    kernel_path=_kernel_path(op, variant),
                    root_symbol=f"@{symbol}",
                    export_name=symbol,
                    dtype=DT,
                )
                out[Path(_route_relpath(op, dt, variant))] = rendered
    return out


def router_route_lists() -> dict[str, list[str]]:
    lists: dict[str, list[str]] = {}
    for op_key, op in OPS_BY_KEY:
        paths = [
            _route_relpath(op, dt, variant)
            for variant, dt in ROUTER_VARIANTS_BY_DTYPE
        ]
        paths.extend(SPECIAL_STATIC_ROUTES_BY_KEY.get(op_key, ()))
        lists[op_key] = paths
    return lists


def generator_input_paths() -> tuple[Path, ...]:
    return (
        Path(__file__).resolve(),
        *sorted((repo_root() / "kernels" / "v2" / "binary").glob("*.tmpl")),
        *sorted((repo_root() / "catalog" / "v2" / "binary").glob("*.tmpl")),
    )
