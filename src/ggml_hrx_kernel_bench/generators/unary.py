"""Code generator for the pointwise-unary v2 kernels + routes.

The whole pointwise-unary family differs only by op name and a single f32 compute snippet, so it is
generated from two kernel templates + two route templates + the ``UNARY_OPS`` table below. To add an op:
add a ``UNARY_OPS`` row (a f32 ``%result``-from-``%value`` snippet + which variants).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from string import Template

CONTIGUOUS = "contiguous_4d"
NON_CONTIGUOUS = "non_contiguous_4d"

DTYPES = (("f16", "F16"), ("f32", "F32"))


@dataclass(frozen=True)
class UnaryOp:
    # f32 snippet producing %result from %value inside scf.if (4-space indented; may span lines). The
    # f16 path up-converts to f32, runs this, then truncates back to f16 (loom scalar math is f32).
    compute: str
    variants: tuple[str, ...]
    # Optional launch-body lines emitted before scf.if (e.g. relu's shared %zero constant). Each line
    # must include its own trailing newline; "" emits nothing.
    preamble: str = ""


UNARY_OPS: dict[str, UnaryOp] = {
    "abs": UnaryOp("    %result = scalar.absf %value : f32", (CONTIGUOUS, NON_CONTIGUOUS)),
    "exp": UnaryOp("    %result = scalar.expf<afn> %value : f32", (CONTIGUOUS, NON_CONTIGUOUS)),
    "neg": UnaryOp("    %result = scalar.negf<nnan|ninf|nsz> %value : f32", (CONTIGUOUS, NON_CONTIGUOUS)),
    "relu": UnaryOp(
        "    %result = scalar.maxnumf %value, %zero : f32",
        (CONTIGUOUS, NON_CONTIGUOUS),
        preamble="  %zero = scalar.constant 0.0 : f32\n",
    ),
    "sqr": UnaryOp("    %result = scalar.mulf<nnan|ninf|nsz> %value, %value : f32", (CONTIGUOUS,)),
    "sqrt": UnaryOp("    %result = scalar.sqrtf %value : f32", (CONTIGUOUS,)),
}


# --- kernel templates -------------------------------------------------------------------------------
# One export block per (op, dtype); a file is CONFIG + f32 export + f16 export. The scf.if body loads
# src0 at $src_index (contiguous: %linear, non-contiguous: %src0_idx) and stores dst at %linear.

def _f32_body(compute: str, src_index: str) -> str:
    return "\n".join(
        [
            f"    %value = view.load %src0_view[{src_index}] : view<1073741824xf32, #dense> -> f32",
            compute,
            "    view.store %result, %dst_view[%linear] : f32, view<1073741824xf32, #dense>",
        ]
    )


def _f16_body(compute: str, src_index: str) -> str:
    return "\n".join(
        [
            f"    %half_in = view.load %src0_view[{src_index}] : view<1073741824xf16, #dense> -> f16",
            "    %value = scalar.extf %half_in : f16 to f32",
            compute,
            "    %half_out = scalar.fptrunc %result : f32 to f16",
            "    %half_bits = scalar.bitcast %half_out : f16 to i16",
            "    view.store %half_bits, %dst_view[%linear] : i16, view<1073741824xi16, #dense>",
        ]
    )


_CONTIGUOUS_CONFIG = """config.decl @hrx2.shape.pointwise.d0 : %value: index where [range(%value, 1, 65536)]
config.decl @hrx2.shape.pointwise.d1 : %value: index where [range(%value, 1, 1048576)]
config.decl @hrx2.tuning.pointwise.workgroup_size : %value: index where [range(%value, 32, 1024), mul(%value, 32)]
"""

_CONTIGUOUS_EXPORT = Template(
    """kernel.def export("$sym") @$sym() {
  %unit = index.constant 1 : index
  %minus_one = index.constant -1 : index
  %d0 = config.get @hrx2.shape.pointwise.d0 : index
  %d1 = config.get @hrx2.shape.pointwise.d1 : index
  %workgroup_size = config.get @hrx2.tuning.pointwise.workgroup_size : index
  %total = index.mul %d0, %d1 : index
  %rounding = index.add %workgroup_size, %minus_one : index
  %rounded = index.add %total, %rounding : index
  %workgroups = index.div %rounded, %workgroup_size : index
  kernel.launch.config workgroups(%workgroups, %unit, %unit) workgroup_size(%workgroup_size, %unit, %unit) : index
} launch(%src0: buffer, %dst: buffer) {
  %base = index.constant 0 : offset
$preamble  %d0 = config.get @hrx2.shape.pointwise.d0 : index
  %d1 = config.get @hrx2.shape.pointwise.d1 : index
  %workgroup_size = config.get @hrx2.tuning.pointwise.workgroup_size : index
  %d0_bounded = index.assume %d0 [range(%d0, 1, 65536)] : index
  %d1_bounded = index.assume %d1 [range(%d1, 1, 1048576)] : index
  %workgroup0 = kernel.workgroup.id<x> : index
  %lane0 = kernel.workitem.id<x> : index
  %workgroup = index.assume %workgroup0 [range(%workgroup0, 0, 4194303)] : index
  %lane = index.assume %lane0 [range(%lane0, 0, 1023)] : index
  %linear_mul0 = index.mul %workgroup, %workgroup_size : index
  %linear_mul = index.assume %linear_mul0 [range(%linear_mul0, 0, 1073741823)] : index
  %linear_add0 = index.add %linear_mul, %lane : index
  %linear = index.assume %linear_add0 [range(%linear_add0, 0, 1073741823)] : index
  %total0 = index.mul %d0_bounded, %d1_bounded : index
  %total = index.assume %total0 [range(%total0, 1, 1073741824)] : index
  %in_bounds = index.cmp ult, %linear, %total : index

  %src0_global = buffer.assume.memory_space<global> %src0 : buffer
  %dst_global = buffer.assume.memory_space<global> %dst : buffer
  %src0_view = buffer.view %src0_global[%base] : buffer -> view<1073741824x${src_type}, #dense>
  %dst_view = buffer.view %dst_global[%base] : buffer -> view<1073741824x${dst_type}, #dense>

  scf.if %in_bounds {
$body
  }
  kernel.return
}
"""
)

_NON_CONTIGUOUS_CONFIG = """config.decl @hrx2.shape.pointwise.ne0 : %value: index where [range(%value, 1, 1073741824)]
config.decl @hrx2.shape.pointwise.ne1 : %value: index where [range(%value, 1, 1073741824)]
config.decl @hrx2.shape.pointwise.ne2 : %value: index where [range(%value, 1, 1073741824)]
config.decl @hrx2.shape.pointwise.ne3 : %value: index where [range(%value, 1, 1073741824)]
config.decl @hrx2.stride.pointwise.src0_nb0 : %value: index where [range(%value, 0, 1073741824)]
config.decl @hrx2.stride.pointwise.src0_nb1 : %value: index where [range(%value, 0, 1073741824)]
config.decl @hrx2.stride.pointwise.src0_nb2 : %value: index where [range(%value, 0, 1073741824)]
config.decl @hrx2.stride.pointwise.src0_nb3 : %value: index where [range(%value, 0, 1073741824)]
config.decl @hrx2.tuning.pointwise.workgroup_size : %value: index where [range(%value, 32, 1024), mul(%value, 32)]
"""

_NON_CONTIGUOUS_EXPORT = Template(
    """kernel.def export("$sym") @$sym() {
  %unit = index.constant 1 : index
  %minus_one = index.constant -1 : index
  %ne0 = config.get @hrx2.shape.pointwise.ne0 : index
  %ne1 = config.get @hrx2.shape.pointwise.ne1 : index
  %ne2 = config.get @hrx2.shape.pointwise.ne2 : index
  %ne3 = config.get @hrx2.shape.pointwise.ne3 : index
  %workgroup_size = config.get @hrx2.tuning.pointwise.workgroup_size : index
  %ne01 = index.mul %ne0, %ne1 : index
  %ne012 = index.mul %ne01, %ne2 : index
  %total = index.mul %ne012, %ne3 : index
  %rounding = index.add %workgroup_size, %minus_one : index
  %rounded = index.add %total, %rounding : index
  %workgroups = index.div %rounded, %workgroup_size : index
  kernel.launch.config workgroups(%workgroups, %unit, %unit) workgroup_size(%workgroup_size, %unit, %unit) : index
} launch(%src0: buffer, %dst: buffer) {
  %base = index.constant 0 : offset
$preamble  %ne0 = config.get @hrx2.shape.pointwise.ne0 : index
  %ne1 = config.get @hrx2.shape.pointwise.ne1 : index
  %ne2 = config.get @hrx2.shape.pointwise.ne2 : index
  %ne3 = config.get @hrx2.shape.pointwise.ne3 : index
  %src0_nb0 = config.get @hrx2.stride.pointwise.src0_nb0 : index
  %src0_nb1 = config.get @hrx2.stride.pointwise.src0_nb1 : index
  %src0_nb2 = config.get @hrx2.stride.pointwise.src0_nb2 : index
  %src0_nb3 = config.get @hrx2.stride.pointwise.src0_nb3 : index
  %workgroup_size = config.get @hrx2.tuning.pointwise.workgroup_size : index
  %ne0_bounded = index.assume %ne0 [range(%ne0, 1, 1073741824)] : index
  %ne1_bounded = index.assume %ne1 [range(%ne1, 1, 1073741824)] : index
  %ne2_bounded = index.assume %ne2 [range(%ne2, 1, 1073741824)] : index
  %ne3_bounded = index.assume %ne3 [range(%ne3, 1, 1073741824)] : index
  %workgroup0 = kernel.workgroup.id<x> : index
  %lane0 = kernel.workitem.id<x> : index
  %workgroup = index.assume %workgroup0 [range(%workgroup0, 0, 4194303)] : index
  %lane = index.assume %lane0 [range(%lane0, 0, 1023)] : index
  %linear_base0 = index.mul %workgroup, %workgroup_size : index
  %linear_base = index.assume %linear_base0 [range(%linear_base0, 0, 1073741823)] : index
  %linear0 = index.add %linear_base, %lane : index
  %linear = index.assume %linear0 [range(%linear0, 0, 1073741823)] : index
  %ne01 = index.mul %ne0_bounded, %ne1_bounded : index
  %ne012 = index.mul %ne01, %ne2_bounded : index
  %total0 = index.mul %ne012, %ne3_bounded : index
  %total = index.assume %total0 [range(%total0, 1, 1073741824)] : index
  %in_bounds = index.cmp ult, %linear, %total : index

  %src0_global = buffer.assume.memory_space<global> %src0 : buffer
  %dst_global = buffer.assume.memory_space<global> %dst : buffer
  %src0_noalias, %dst_noalias = buffer.assume.noalias %src0_global, %dst_global : buffer, buffer
  %src0_view = buffer.view %src0_noalias[%base] : buffer -> view<1073741824x${src_type}, #dense>
  %dst_view = buffer.view %dst_noalias[%base] : buffer -> view<1073741824x${dst_type}, #dense>

  scf.if %in_bounds {
    %i0 = index.rem %linear, %ne0_bounded : index
    %q0 = index.div %linear, %ne0_bounded : index
    %i1 = index.rem %q0, %ne1_bounded : index
    %q1 = index.div %q0, %ne1_bounded : index
    %i2 = index.rem %q1, %ne2_bounded : index
    %i3 = index.div %q1, %ne2_bounded : index

    %src0_o0 = index.mul %i0, %src0_nb0 : index
    %src0_o1 = index.mul %i1, %src0_nb1 : index
    %src0_o2 = index.mul %i2, %src0_nb2 : index
    %src0_o3 = index.mul %i3, %src0_nb3 : index
    %src0_o01 = index.add %src0_o0, %src0_o1 : index
    %src0_o23 = index.add %src0_o2, %src0_o3 : index
    %src0_idx0 = index.add %src0_o01, %src0_o23 : index
    %src0_idx = index.assume %src0_idx0 [range(%src0_idx0, 0, 1073741823)] : index

$body
  }
  kernel.return
}
"""
)

_VARIANT_KERNEL = {
    CONTIGUOUS: (_CONTIGUOUS_CONFIG, _CONTIGUOUS_EXPORT, "%linear"),
    NON_CONTIGUOUS: (_NON_CONTIGUOUS_CONFIG, _NON_CONTIGUOUS_EXPORT, "%src0_idx"),
}


def _kernel_symbol(op: str, dt: str, variant: str) -> str:
    return f"{op}_{dt}" if variant == CONTIGUOUS else f"{op}_{dt}_non_contiguous_4d"


def _render_kernel(op: str, variant: str) -> str:
    config, export, src_index = _VARIANT_KERNEL[variant]
    spec = UNARY_OPS[op]
    exports = []
    for dt, _DT in DTYPES:
        sym = _kernel_symbol(op, dt, variant)
        src_type, dst_type = ("f32", "f32") if dt == "f32" else ("f16", "i16")
        body = _f32_body(spec.compute, src_index) if dt == "f32" else _f16_body(spec.compute, src_index)
        exports.append(
            export.substitute(
                sym=sym, src_type=src_type, dst_type=dst_type, body=body, preamble=spec.preamble
            )
        )
    # File order matches the checked-in files: f32 export then f16 export.
    exports = [exports[1], exports[0]]  # DTYPES is (f16, f32); emit f32 first, f16 second
    return config + "\n" + "\n".join(exports)


# --- route templates (built as dicts; json.dumps(indent=2) round-trips the checked-in files) ---------

def _route_dict(op: str, dt: str, DT: str, variant: str) -> dict:
    if variant == CONTIGUOUS:
        return {
            "id": f"{op}_{dt}_contiguous_4d",
            "family": f"{op}_{dt}",
            "kernel": {
                "source_id": f"pointwise_{dt}",
                "path": f"{op}/contiguous_4d.loom",
                "root_symbol": f"@{op}_{dt}",
                "export_name": f"{op}_{dt}",
            },
            "tensors": {
                "src0": {"dtype": DT, "dimensions": "src0_dimensions", "strides": "src0_strides"},
                "dst": {"dtype": DT, "dimensions": "dst_dimensions", "strides": "dst_strides"},
            },
            "values": [
                {"name": "contiguous_strides", "contiguous_strides": "dst_dimensions"},
                {"name": "leading_dimensions", "head": {"source": "dst_dimensions", "take": 1}},
                {"name": "trailing_dimensions", "tail": {"source": "dst_dimensions", "drop": 1}},
                {"name": "flattened_trailing_dimensions", "product": "trailing_dimensions"},
                {"name": "total_size", "product": "dst_dimensions"},
            ],
            "constraints": [
                {"name": "total_size", "min": 1, "max": 1073741824},
                {"name": "dst_dimensions", "rank_min": 2, "rank_max": 4},
                {"equals": ["src0_dimensions", "dst_dimensions"]},
                {"equals": ["contiguous_strides", "src0_strides", "dst_strides"]},
            ],
            "launch": {"workgroup_size": [256, 1, 1]},
            "config": {
                "bindings": [
                    {"key": "@hrx2.shape.pointwise.d0", "source": "value.leading_dimensions.0"},
                    {"key": "@hrx2.shape.pointwise.d1", "source": "value.flattened_trailing_dimensions"},
                    {"key": "@hrx2.tuning.pointwise.workgroup_size", "value": "256"},
                ]
            },
        }
    sym = f"{op}_{dt}_non_contiguous_4d"
    return {
        "id": sym,
        "family": f"{op}_{dt}",
        "kernel": {
            "source_id": f"pointwise_{dt}",
            "path": f"{op}/non_contiguous_4d.loom",
            "root_symbol": f"@{sym}",
            "export_name": sym,
        },
        "tensors": {
            "src0": {"dtype": DT, "dimensions": "src0_dimensions", "strides": "src0_strides"},
            "dst": {"dtype": DT, "dimensions": "dst_dimensions", "strides": "dst_strides"},
        },
        "values": [
            {"name": "contiguous_strides", "contiguous_strides": "dst_dimensions"},
            {"name": "total_size", "product": "dst_dimensions"},
        ],
        "constraints": [
            {"name": "dst_dimensions", "length": 4},
            {"equals": ["src0_dimensions", "dst_dimensions"]},
            {"equals": ["contiguous_strides", "dst_strides"]},
        ],
        "launch": {"workgroup_size": [256, 1, 1]},
        "config": {
            "bindings": [
                {"key": "@hrx2.shape.pointwise.ne0", "source": "tensor.dst.dimensions.d0.size"},
                {"key": "@hrx2.shape.pointwise.ne1", "source": "tensor.dst.dimensions.d1.size"},
                {"key": "@hrx2.shape.pointwise.ne2", "source": "tensor.dst.dimensions.d2.size"},
                {"key": "@hrx2.shape.pointwise.ne3", "source": "tensor.dst.dimensions.d3.size"},
                {"key": "@hrx2.stride.pointwise.src0_nb0", "source": "tensor.src0.dimensions.d0.stride"},
                {"key": "@hrx2.stride.pointwise.src0_nb1", "source": "tensor.src0.dimensions.d1.stride"},
                {"key": "@hrx2.stride.pointwise.src0_nb2", "source": "tensor.src0.dimensions.d2.stride"},
                {"key": "@hrx2.stride.pointwise.src0_nb3", "source": "tensor.src0.dimensions.d3.stride"},
                {"key": "@hrx2.tuning.pointwise.workgroup_size", "value": "256"},
            ]
        },
    }


def _route_relpath(op: str, dt: str, variant: str) -> str:
    name = f"{op}_{dt}" if variant == CONTIGUOUS else f"{op}_{dt}_non_contiguous_4d"
    return f"{op}/{name}.json"


# --- public API (mirrors generators/copy.py) --------------------------------------------------------

def render_kernel_artifacts() -> dict[Path, str]:
    """Kernel .loom contents keyed by path relative to ``kernels/v2``."""
    out: dict[Path, str] = {}
    for op, spec in UNARY_OPS.items():
        for variant in spec.variants:
            out[Path(op) / f"{variant}.loom"] = _render_kernel(op, variant)
    return out


def render_catalog_artifacts() -> dict[Path, str]:
    """Route .json contents keyed by path relative to ``catalog/v2``."""
    out: dict[Path, str] = {}
    for op, spec in UNARY_OPS.items():
        for variant in spec.variants:
            for dt, DT in DTYPES:
                out[Path(_route_relpath(op, dt, variant))] = (
                    json.dumps(_route_dict(op, dt, DT, variant), indent=2) + "\n"
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
    return (Path(__file__).resolve(),)
