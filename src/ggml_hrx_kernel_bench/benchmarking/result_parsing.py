from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from .common import load_jsonl_objects as _load_jsonl_objects

def _throughput_summary(
    benchmark_rows: Sequence[dict[str, Any]],
    flop_estimate: dict[str, Any],
) -> dict[str, Any] | None:
    estimated_flops = flop_estimate.get("estimated_flops")
    if not isinstance(estimated_flops, int) or estimated_flops <= 0:
        return None
    throughputs: list[dict[str, Any]] = []
    for row in benchmark_rows:
        result = row.get("benchmark_result")
        if not isinstance(result, dict):
            continue
        measurement = result.get("measurement")
        if not isinstance(measurement, dict):
            continue
        operation_timing = measurement.get("operation_timing_ns")
        if not isinstance(operation_timing, dict):
            continue
        timing: dict[str, Any] = {}
        for key in ("mean", "p50", "p90", "min", "max"):
            value = operation_timing.get(key)
            if isinstance(value, int | float) and value > 0:
                timing[f"flops_per_second_from_{key}_ns"] = estimated_flops * 1_000_000_000.0 / float(value)
        if timing:
            timing["benchmark"] = result.get("benchmark")
            timing["case"] = result.get("case")
            timing["estimated_flops_per_operation"] = estimated_flops
            throughputs.append(timing)
    if not throughputs:
        return None
    return {
        "unit": "flops/second",
        "basis": "operation_timing_ns",
        "rows": throughputs,
    }


def _first_number(*values: object) -> int | float | None:
    for value in values:
        if isinstance(value, int | float):
            return value
    return None


def _recursive_number_by_key(value: object, key: str) -> int | float | None:
    if isinstance(value, dict):
        found = value.get(key)
        if isinstance(found, int | float):
            return found
        for child in value.values():
            result = _recursive_number_by_key(child, key)
            if result is not None:
                return result
    elif isinstance(value, list):
        for child in value:
            result = _recursive_number_by_key(child, key)
            if result is not None:
                return result
    return None


def _operation_timing_summary(benchmark_rows: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    for row in benchmark_rows:
        result = row.get("benchmark_result")
        if not isinstance(result, dict):
            continue
        measurement = result.get("measurement")
        if not isinstance(measurement, dict):
            continue
        operation_timing = measurement.get("operation_timing_ns")
        if isinstance(operation_timing, dict):
            return operation_timing
    return None


def _compile_summary_from_report(report: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(report, dict):
        return None
    target_resources = report.get("target_resources")
    scalar_resources = target_resources.get("scalar") if isinstance(target_resources, dict) else None
    vector_resources = target_resources.get("vector") if isinstance(target_resources, dict) else None
    scalar_final = scalar_resources.get("final") if isinstance(scalar_resources, dict) else None
    vector_final = vector_resources.get("final") if isinstance(vector_resources, dict) else None
    summary = {
        "target_key": report.get("target_key"),
        "artifact_size": _first_number(report.get("artifact_size")),
        "instruction_count": _first_number(report.get("instruction_count")),
        "code_byte_count": _first_number(report.get("code_byte_count")),
        "local_memory_bytes": _first_number(report.get("local_memory_bytes")),
        "private_memory_bytes": _first_number(report.get("private_memory_bytes")),
        "sgpr_count": _first_number(scalar_final.get("register_count") if isinstance(scalar_final, dict) else None),
        "vgpr_count": _first_number(vector_final.get("register_count") if isinstance(vector_final, dict) else None),
        "allocation_spill_count": _first_number(report.get("allocation_spill_count")),
        "materialized_spill_storage_count": _first_number(report.get("allocation_materialized_spill_storage_count")),
        "materialized_spill_store_count": _first_number(report.get("allocation_materialized_spill_store_count")),
        "materialized_reload_count": _first_number(report.get("allocation_materialized_reload_count")),
        "lds_instruction_count": _recursive_number_by_key(report, "local_memory_count"),
        "barrier_instruction_count": _recursive_number_by_key(report, "barrier_count"),
    }
    return {key: value for key, value in summary.items() if value is not None}


def _compile_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    for row in rows:
        if row.get("row") == "benchmark":
            result = row.get("benchmark_result")
            if isinstance(result, dict):
                summary = _compile_summary_from_report(result.get("compile_report"))
                if summary:
                    return summary
        if row.get("row") == "compile":
            summary = _compile_summary_from_report(row.get("compile_report"))
            if summary:
                return summary
    return None


def _benchmark_result_summary(path: Path, *, flop_estimate: dict[str, Any] | None = None) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "missing_output"}
    rows = _load_jsonl_objects(path)
    benchmark_rows = [row for row in rows if row.get("row") == "benchmark"]
    summary_rows = [row for row in rows if row.get("row") == "summary"]
    state = "ok" if benchmark_rows else "no_benchmark_rows"
    for row in benchmark_rows:
        result = row.get("benchmark_result")
        if isinstance(result, dict) and result.get("state") not in {None, "ok"}:
            state = str(result.get("state"))
    summary = {
        "status": state,
        "row_count": len(rows),
        "benchmark_row_count": len(benchmark_rows),
        "summary": summary_rows[-1].get("summary") if summary_rows else None,
    }
    operation_timing = _operation_timing_summary(benchmark_rows)
    if operation_timing is not None:
        summary["operation_timing_ns"] = operation_timing
    compile_summary = _compile_summary(rows)
    if compile_summary is not None:
        summary["compile_summary"] = compile_summary
    if flop_estimate is not None:
        throughput = _throughput_summary(benchmark_rows, flop_estimate)
        if throughput is not None:
            summary["throughput"] = throughput
    return summary
