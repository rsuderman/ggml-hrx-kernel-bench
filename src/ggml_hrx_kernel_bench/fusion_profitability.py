from __future__ import annotations

from collections import OrderedDict, defaultdict
from typing import Any

from .reporting import timing_ns


SCHEMA = "ggml_hrx_kernel_bench.fusion_profitability.v1"
DEFAULT_WIN_THRESHOLD = 0.075

FUSION_COMPONENTS: dict[str, tuple[str, ...]] = {
    "add_rms_norm_mul_f32": ("add_f32", "rms_norm_f32", "mul_f32"),
    "cont_set_rows_f32": ("cont_f32", "set_rows_f32"),
    "mul_mat_f16_f32_batched_cont": ("mul_mat_f16_f32_batched", "cont_f32"),
    "mul_mat_q4_k_swiglu_f32": ("mul_mat_q4_k_f32", "swiglu_f32"),
    "rms_norm_mul_f32": ("rms_norm_f32", "mul_f32"),
    "rms_norm_mul_quantize_q8_1_f32": ("rms_norm_f32", "mul_f32", "quantize_q8_1_f32"),
    "rope_set_rows_f32": ("rope_f32", "set_rows_f32"),
    "softmax_kqv_f32_f16": ("soft_max_f32", "mul_mat_f16_f32_batched"),
}


def analyze_fusion_profitability(
    catalog_rows: list[dict[str, Any]],
    *,
    win_threshold: float = DEFAULT_WIN_THRESHOLD,
) -> OrderedDict[str, Any]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in catalog_rows:
        candidate = row.get("candidate") if isinstance(row.get("candidate"), dict) else {}
        by_family[str(candidate.get("family") or "unknown")].append(row)

    evaluated: list[OrderedDict[str, Any]] = []
    accepted: list[OrderedDict[str, Any]] = []
    for family, components in sorted(FUSION_COMPONENTS.items()):
        for row in by_family.get(family, []):
            fused_time = timing_ns(_catalog_row_as_ledger_row(row))
            component_times = [_best_time(by_family.get(component, [])) for component in components]
            missing = [
                component
                for component, value in zip(components, component_times, strict=True)
                if value is None
            ]
            baseline = None if missing else sum(float(value) for value in component_times if value is not None)
            speedup = None
            profitable = False
            if fused_time is not None and baseline:
                speedup = (baseline - fused_time) / baseline
                profitable = speedup >= win_threshold
            result = OrderedDict(
                [
                    ("family", family),
                    ("candidate_id", (row.get("candidate") or {}).get("candidate_id")),
                    ("route_id", (row.get("candidate") or {}).get("route_id")),
                    ("shape", (row.get("candidate") or {}).get("shape") or {}),
                    ("components", list(components)),
                    ("fused_timing_ns", fused_time),
                    ("baseline_timing_ns", baseline),
                    ("speedup", speedup),
                    ("win_threshold", win_threshold),
                    ("status", "accepted" if profitable else _status(fused_time, baseline, missing)),
                    ("missing_baseline_components", missing),
                ]
            )
            evaluated.append(result)
            if profitable:
                accepted.append(result)

    return OrderedDict(
        [
            ("schema", SCHEMA),
            ("summary", OrderedDict([("evaluated_count", len(evaluated)), ("accepted_count", len(accepted))])),
            ("accepted", accepted),
            ("evaluated", evaluated),
        ]
    )


def _best_time(rows: list[dict[str, Any]]) -> float | None:
    values = [
        timing_ns(_catalog_row_as_ledger_row(row))
        for row in rows
        if row.get("catalog_ready")
    ]
    valid = [float(value) for value in values if value is not None]
    return min(valid) if valid else None


def _catalog_row_as_ledger_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate": row.get("candidate"),
        "benchmark": row.get("benchmark"),
        "status": "ran" if row.get("benchmark") else "compiled",
    }


def _status(fused_time: float | None, baseline: float | None, missing: list[str]) -> str:
    if fused_time is None:
        return "missing_fusion_timing"
    if missing:
        return "missing_baseline_components"
    if baseline is None:
        return "missing_baseline_timing"
    return "below_profit_threshold"
