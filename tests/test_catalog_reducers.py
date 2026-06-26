from __future__ import annotations

from ggml_hrx_kernel_bench.fusion_profitability import analyze_fusion_profitability
from ggml_hrx_kernel_bench.route_reducer import reduce_routes


def _catalog_row(family: str, candidate_id: str, timing_ns: float | None, *, ready: bool = True) -> dict:
    benchmark = None
    if timing_ns is not None:
        benchmark = {"summary": {"operation_timing_ns": {"mean": timing_ns}}}
    return {
        "candidate_id": candidate_id,
        "catalog_ready": ready,
        "candidate": {
            "candidate_id": candidate_id,
            "family": family,
            "route_id": candidate_id + "_route",
            "source_id": family,
            "root_symbol": "@" + family,
            "shape": {"ncols": 128, "nrows": 1},
            "route": {"shape_domain": {"ncols_min": 1, "ncols_max": 4096}},
            "schedule": {"scenario": "decode-1", "source": "llm-atlas"},
        },
        "compile": {"target_artifact_bytes": 64, "report_summary": {"emission_code_byte_count": 32}},
        "benchmark": benchmark,
        "rejection_reasons": [] if ready else ["not_compiled"],
    }


def test_route_reducer_accepts_ready_rows_and_reports_empty_families() -> None:
    reduced = reduce_routes(
        [
            _catalog_row("copy_f32_f16", "ready", 100.0),
            _catalog_row("rms_norm_f32", "not_ready", None, ready=False),
        ]
    )

    assert reduced["summary"]["accepted_count"] == 1
    assert reduced["accepted"][0]["candidate_id"] == "ready"
    assert reduced["rejected"][0]["family"] == "rms_norm_f32"
    assert reduced["rejected"][0]["reason"] == "no_catalog_ready_candidates"


def test_fusion_profitability_requires_decomposed_baseline() -> None:
    missing = analyze_fusion_profitability([_catalog_row("rms_norm_mul_f32", "fused", 80.0)])

    assert missing["evaluated"][0]["status"] == "missing_baseline_components"
    assert missing["accepted"] == []


def test_fusion_profitability_accepts_winning_fusion() -> None:
    report = analyze_fusion_profitability(
        [
            _catalog_row("rms_norm_mul_f32", "fused", 80.0),
            _catalog_row("rms_norm_f32", "rms", 60.0),
            _catalog_row("mul_f32", "mul", 50.0),
        ]
    )

    assert report["summary"]["accepted_count"] == 1
    assert report["accepted"][0]["speedup"] > 0.075
