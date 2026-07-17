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

import json
from dataclasses import dataclass
from pathlib import Path

from .utils import load_template, repo_root

CONTIGUOUS = "contiguous_4d"
NON_CONTIGUOUS = "non_contiguous_4d"

DTYPES = (("f16", "F16"), ("f32", "F32"))
F32_DTYPES = (("f32", "F32"),)


@dataclass(frozen=True)
class UnaryOp:
    # f32 snippet producing %result from %value inside scf.if (4-space indented; may span lines). The
    # f16 path up-converts to f32, runs this, then truncates back to f16 (loom scalar math is f32).
    compute_block: str
    variants: tuple[str, ...]
    # Optional launch-body lines emitted before scf.if (e.g. relu's shared %zero constant). Each line
    # must include its own trailing newline; "" emits nothing.
    preamble: str = ""
    dtypes: tuple[tuple[str, str], ...] = DTYPES
    # Optional exact YAML attributes required for this route family.
    attributes: dict[str, object] | None = None


UNARY_OPS: dict[str, UnaryOp] = {
    "abs": UnaryOp(
        "    %result = scalar.absf %value : f32", (CONTIGUOUS, NON_CONTIGUOUS)
    ),
    "ceil": UnaryOp("    %result = scalar.ceilf %value : f32", (CONTIGUOUS, NON_CONTIGUOUS)),
    "cos": UnaryOp("    %result = scalar.cosf<afn> %value : f32", (CONTIGUOUS,)),
    "elu": UnaryOp(
        "\n".join(
            (
                "    %positive = scalar.cmpf ogt, %value, %zero : f32",
                "    %exp = scalar.expf<afn> %value : f32",
                "    %negative_result = scalar.subf<nnan|ninf|nsz> %exp, %one : f32",
                "    %result = scf.select %positive, %value, %negative_result : f32",
            )
        ),
        (CONTIGUOUS, NON_CONTIGUOUS),
        preamble="  %zero = scalar.constant 0.0 : f32\n  %one = scalar.constant 1.0 : f32\n",
    ),
    "exp": UnaryOp(
        "    %result = scalar.expf<afn> %value : f32", (CONTIGUOUS, NON_CONTIGUOUS)
    ),
    "expm1": UnaryOp(
        "\n".join(
            (
                "    %exp = scalar.expf<afn> %value : f32",
                "    %result = scalar.subf<nnan|ninf|nsz> %exp, %one : f32",
            )
        ),
        (CONTIGUOUS, NON_CONTIGUOUS),
        preamble="  %one = scalar.constant 1.0 : f32\n",
    ),
    "floor": UnaryOp("    %result = scalar.floorf %value : f32", (CONTIGUOUS, NON_CONTIGUOUS)),
    "gelu": UnaryOp(
        "\n".join(
            (
                "    %square = scalar.mulf<nnan|ninf|nsz> %value, %value : f32",
                "    %coef_term = scalar.mulf<nnan|ninf|nsz> %coef, %square : f32",
                "    %inner_scale = scalar.addf<nnan|ninf|nsz> %one, %coef_term : f32",
                "    %inner = scalar.mulf<nnan|ninf|nsz> %value, %inner_scale : f32",
                "    %scaled = scalar.mulf<nnan|ninf|nsz> %sqrt_two_over_pi, %inner : f32",
                "    %tanh = scalar.tanhf<afn> %scaled : f32",
                "    %one_plus = scalar.addf<nnan|ninf|nsz> %one, %tanh : f32",
                "    %half_value = scalar.mulf<nnan|ninf|nsz> %half, %value : f32",
                "    %result = scalar.mulf<nnan|ninf|nsz> %half_value, %one_plus : f32",
            )
        ),
        (CONTIGUOUS, NON_CONTIGUOUS),
        preamble=(
            "  %half = scalar.constant 0.5 : f32\n"
            "  %one = scalar.constant 1.0 : f32\n"
            "  %coef = scalar.constant 0.044715 : f32\n"
            "  %sqrt_two_over_pi = scalar.constant 0.7978845608028654 : f32\n"
        ),
    ),
    "gelu_erf": UnaryOp(
        "\n".join(
            (
                "    %scaled = scalar.mulf<nnan|ninf|nsz> %value, %sqrt_two_inv : f32",
                "    %erf = scalar.erff<afn> %scaled : f32",
                "    %one_plus = scalar.addf<nnan|ninf|nsz> %one, %erf : f32",
                "    %half_value = scalar.mulf<nnan|ninf|nsz> %half, %value : f32",
                "    %result = scalar.mulf<nnan|ninf|nsz> %half_value, %one_plus : f32",
            )
        ),
        (CONTIGUOUS, NON_CONTIGUOUS),
        preamble=(
            "  %half = scalar.constant 0.5 : f32\n"
            "  %one = scalar.constant 1.0 : f32\n"
            "  %sqrt_two_inv = scalar.constant 0.7071067811865476 : f32\n"
        ),
    ),
    "gelu_quick": UnaryOp(
        "\n".join(
            (
                "    %scaled = scalar.mulf<nnan|ninf|nsz> %value, %coef : f32",
                "    %exp = scalar.expf<afn> %scaled : f32",
                "    %denom = scalar.addf<nnan|ninf|nsz> %one, %exp : f32",
                "    %result = scalar.divf<nnan|ninf|nsz|arcp> %value, %denom : f32",
            )
        ),
        (CONTIGUOUS, NON_CONTIGUOUS),
        preamble="  %one = scalar.constant 1.0 : f32\n  %coef = scalar.constant -1.702 : f32\n",
    ),
    "hardsigmoid": UnaryOp(
        "\n".join(
            (
                "    %biased = scalar.addf<nnan|ninf|nsz> %value, %three : f32",
                "    %scaled = scalar.divf<nnan|ninf|nsz|arcp> %biased, %six : f32",
                "    %lower = scalar.maxnumf %scaled, %zero : f32",
                "    %result = scalar.minnumf %lower, %one : f32",
            )
        ),
        (CONTIGUOUS, NON_CONTIGUOUS),
        preamble=(
            "  %zero = scalar.constant 0.0 : f32\n"
            "  %one = scalar.constant 1.0 : f32\n"
            "  %three = scalar.constant 3.0 : f32\n"
            "  %six = scalar.constant 6.0 : f32\n"
        ),
    ),
    "hardswish": UnaryOp(
        "\n".join(
            (
                "    %biased = scalar.addf<nnan|ninf|nsz> %value, %three : f32",
                "    %scaled = scalar.divf<nnan|ninf|nsz|arcp> %biased, %six : f32",
                "    %lower = scalar.maxnumf %scaled, %zero : f32",
                "    %gate = scalar.minnumf %lower, %one : f32",
                "    %result = scalar.mulf<nnan|ninf|nsz> %value, %gate : f32",
            )
        ),
        (CONTIGUOUS, NON_CONTIGUOUS),
        preamble=(
            "  %zero = scalar.constant 0.0 : f32\n"
            "  %one = scalar.constant 1.0 : f32\n"
            "  %three = scalar.constant 3.0 : f32\n"
            "  %six = scalar.constant 6.0 : f32\n"
        ),
    ),
    "leaky_relu": UnaryOp(
        "\n".join(
            (
                "    %nonnegative = scalar.cmpf oge, %value, %zero : f32",
                "    %negative_result = scalar.mulf<nnan|ninf|nsz> %value, %negative_slope : f32",
                "    %result = scf.select %nonnegative, %value, %negative_result : f32",
            )
        ),
        (CONTIGUOUS,),
        preamble=(
            "  %zero = scalar.constant 0.0 : f32\n"
            "  %negative_slope = scalar.constant 0.1 : f32\n"
        ),
        attributes={"negative_slope": 0.1},
    ),
    "log": UnaryOp(
        "\n".join(
            (
                "    %log2 = scalar.log2f<afn> %value : f32",
                "    %result = scalar.mulf<nnan|ninf|nsz> %log2, %ln_two : f32",
            )
        ),
        (CONTIGUOUS,),
        preamble="  %ln_two = scalar.constant 0.6931471805599453 : f32\n",
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
    "round": UnaryOp("    %result = scalar.roundf %value : f32", (CONTIGUOUS, NON_CONTIGUOUS)),
    "sgn": UnaryOp(
        "\n".join(
            (
                "    %greater = scalar.cmpf ogt, %value, %zero : f32",
                "    %less = scalar.cmpf olt, %value, %zero : f32",
                "    %positive = scf.select %greater, %one, %zero : f32",
                "    %result = scf.select %less, %minus_one, %positive : f32",
            )
        ),
        (CONTIGUOUS, NON_CONTIGUOUS),
        preamble=(
            "  %minus_one = scalar.constant -1.0 : f32\n"
            "  %zero = scalar.constant 0.0 : f32\n"
            "  %one = scalar.constant 1.0 : f32\n"
        ),
    ),
    "sigmoid": UnaryOp(
        "\n".join(
            (
                "    %neg = scalar.negf<nnan|ninf|nsz> %value : f32",
                "    %exp = scalar.expf<afn> %neg : f32",
                "    %denom = scalar.addf<nnan|ninf|nsz> %one, %exp : f32",
                "    %result = scalar.divf<nnan|ninf|nsz|arcp> %one, %denom : f32",
            )
        ),
        (CONTIGUOUS, NON_CONTIGUOUS),
        preamble="  %one = scalar.constant 1.0 : f32\n",
    ),
    "silu": UnaryOp("    %result = scalar.siluf %value : f32", (CONTIGUOUS, NON_CONTIGUOUS)),
    "sin": UnaryOp("    %result = scalar.sinf<afn> %value : f32", (CONTIGUOUS,)),
    "softcap": UnaryOp(
        "\n".join(
            (
                "    %scaled = scalar.divf<nnan|ninf|nsz|arcp> %value, %softcap : f32",
                "    %tanh = scalar.tanhf<afn> %scaled : f32",
                "    %result = scalar.mulf<nnan|ninf|nsz> %tanh, %softcap : f32",
            )
        ),
        (CONTIGUOUS,),
        preamble="  %softcap = scalar.constant 50.0 : f32\n",
        dtypes=(("f32", "F32"),),
        attributes={"softcap": 50.0},
    ),
    "softplus": UnaryOp("    %result = scalar.softplusf<afn> %value : f32", (CONTIGUOUS, NON_CONTIGUOUS)),
    "sqr": UnaryOp(
        "    %result = scalar.mulf<nnan|ninf|nsz> %value, %value : f32", (CONTIGUOUS,)
    ),
    "sqrt": UnaryOp("    %result = scalar.sqrtf %value : f32", (CONTIGUOUS,)),
    "step": UnaryOp(
        "\n".join(
            (
                "    %positive = scalar.cmpf ogt, %value, %zero : f32",
                "    %result = scf.select %positive, %one, %zero : f32",
            )
        ),
        (CONTIGUOUS, NON_CONTIGUOUS),
        preamble="  %zero = scalar.constant 0.0 : f32\n  %one = scalar.constant 1.0 : f32\n",
    ),
    "tanh": UnaryOp(
        "    %result = scalar.tanhf<afn> %value : f32",
        (CONTIGUOUS, NON_CONTIGUOUS),
    ),
    "trunc": UnaryOp("    %result = scalar.truncf %value : f32", (CONTIGUOUS, NON_CONTIGUOUS)),
    "xielu": UnaryOp(
        "\n".join(
            (
                "    %positive = scalar.cmpf ogt, %value, %zero : f32",
                "    %square = scalar.mulf<nnan|ninf|nsz> %value, %value : f32",
                "    %positive_scaled = scalar.mulf<nnan|ninf|nsz> %alpha_p, %square : f32",
                "    %beta_value = scalar.mulf<nnan|ninf|nsz> %beta, %value : f32",
                "    %positive_result = scalar.addf<nnan|ninf|nsz> %positive_scaled, %beta_value : f32",
                "    %min_x_eps = scalar.minnumf %value, %eps : f32",
                "    %exp = scalar.expf<afn> %min_x_eps : f32",
                "    %expm1 = scalar.subf<nnan|ninf|nsz> %exp, %one : f32",
                "    %delta = scalar.subf<nnan|ninf|nsz> %expm1, %value : f32",
                "    %negative_scaled = scalar.mulf<nnan|ninf|nsz> %delta, %alpha_n : f32",
                "    %negative_result = scalar.addf<nnan|ninf|nsz> %negative_scaled, %beta_value : f32",
                "    %result = scf.select %positive, %positive_result, %negative_result : f32",
            )
        ),
        (CONTIGUOUS,),
        preamble=(
            "  %zero = scalar.constant 0.0 : f32\n"
            "  %one = scalar.constant 1.0 : f32\n"
            "  %alpha_n = scalar.constant 4.0 : f32\n"
            "  %alpha_p = scalar.constant 20.0 : f32\n"
            "  %beta = scalar.constant 0.5 : f32\n"
            "  %eps = scalar.constant 0.0000001 : f32\n"
        ),
        dtypes=(("f32", "F32"),),
    ),
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
            for dt, DT in spec.dtypes:
                sym = _route_symbol(op, dt, variant)
                rendered = json.loads(
                    template.substitute(
                        route_id=(
                            f"{op}_{dt}_contiguous_4d" if variant == CONTIGUOUS else sym
                        ),
                        family=f"{op}_{dt}",
                        op=op.upper(),
                        source_id=f"pointwise_{dt}",
                        kernel_path=f"{op}/{variant}.loom",
                        root_symbol=f"@{sym}",
                        export_name=sym,
                        dtype=DT,
                    )
                )
                if spec.attributes is not None:
                    rendered["attributes"] = spec.attributes
                out[Path(_route_relpath(op, dt, variant))] = json.dumps(
                    rendered,
                    indent=2,
                )
    return out


def router_route_lists() -> dict[str, list[str]]:
    """Router ``routes`` entries (op key -> ordered catalog-relative route paths) to inject."""
    lists: dict[str, list[str]] = {}
    for op, spec in UNARY_OPS.items():
        paths: list[str] = []
        for variant in spec.variants:
            for dt, _DT in spec.dtypes:
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
