from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from .kernel_test_config import expect, load_config, validate_config
from .kernel_test_config_runtime import case_result, execute_case, select_cases


RESULT_SCHEMA = "ggml_hrx_kernel_bench.kernel_benchmark_results.v1"
COMPARISON_SCHEMA = "ggml_hrx_kernel_bench.kernel_benchmark_comparison.v1"
MANIFEST_SCHEMA = "ggml_hrx_kernel_bench.materialized_benchmarks.v1"


def timed_values(results: list[dict[str, Any]]) -> list[float]:
    values: list[float] = []
    for result in results:
        value = result.get("operation_timing_ns")
        if isinstance(value, dict):
            mean_value = value.get("mean")
            if isinstance(mean_value, (int, float)):
                values.append(float(mean_value))
        elif isinstance(value, (int, float)):
            values.append(float(value))
    return values


def dispatch_values(results: list[dict[str, Any]]) -> list[float]:
    values: list[float] = []
    for result in results:
        value = result.get("mean_physical_dispatch_duration_ns")
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    timings = timed_values(results)
    dispatch_timings = dispatch_values(results)
    failed = [result for result in results if result.get("status") != "ran"]
    summary: dict[str, Any] = {
        "case_count": len(results),
        "passed_case_count": len(results) - len(failed),
        "failed_case_count": len(failed),
    }
    if timings:
        summary.update(
            {
                "min_operation_timing_ns": min(timings),
                "max_operation_timing_ns": max(timings),
                "mean_operation_timing_ns": mean(timings),
            }
        )
    if dispatch_timings:
        summary.update(
            {
                "min_mean_physical_dispatch_duration_ns": min(dispatch_timings),
                "max_mean_physical_dispatch_duration_ns": max(dispatch_timings),
                "mean_physical_dispatch_duration_ns": mean(dispatch_timings),
            }
        )
    return summary


def public_case_result(result: dict[str, Any]) -> dict[str, Any]:
    public: dict[str, Any] = {
        "case_id": result.get("case_id"),
        "values": result.get("values"),
        "status": result.get("status"),
    }
    if "operation_timing_ns" in result:
        public["operation_timing_ns"] = result.get("operation_timing_ns")
    if "mean_physical_dispatch_duration_ns" in result:
        public["mean_physical_dispatch_duration_ns"] = result.get(
            "mean_physical_dispatch_duration_ns"
        )
    if "physical_dispatches_per_logical_operation" in result:
        public["physical_dispatches_per_logical_operation"] = result.get(
            "physical_dispatches_per_logical_operation"
        )
    if result.get("status") != "ran":
        public["error"] = result.get("error")
        public["message"] = result.get("message")
    return public


def benchmark_config_payload(
    config_path: Path,
    *,
    case_selectors: list[str] | None,
    tool_dir: str | None,
    target: str,
    rocm_path: str | None,
    iterations: int,
    warmup_iterations: int,
    max_batches: int,
    artifact_dir: Path,
    require_tool: Any,
) -> dict[str, Any]:
    config_data = load_config(config_path)
    validate_config(config_data)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    raw_results: list[dict[str, Any]] = []
    for current_case_id, current_case_values in select_cases(config_data, case_selectors):
        case_output_dir = artifact_dir / current_case_id
        try:
            candidate, row, summary = execute_case(
                config_data=config_data,
                current_case_id=current_case_id,
                current_case_values=current_case_values,
                tool_dir=tool_dir,
                target=target,
                rocm_path=rocm_path,
                iterations=iterations,
                warmup_iterations=warmup_iterations,
                max_batches=max_batches,
                output_dir=case_output_dir,
                require_tool=require_tool,
            )
            result = case_result(
                candidate=candidate,
                current_case_id=current_case_id,
                current_case_values=current_case_values,
                row=row,
                summary=summary,
                output_dir=case_output_dir,
            )
        except Exception as exc:
            result = {
                "case_id": current_case_id,
                "values": list(current_case_values),
                "status": "tool_error",
                "error": type(exc).__name__,
                "message": str(exc),
            }
        raw_results.append(result)

    public_results = [public_case_result(result) for result in raw_results]
    return {
        "schema": RESULT_SCHEMA,
        "target": target,
        "summary": summarize_results(public_results),
        "kernel": config_data["kernel"],
        "params": list(config_data["params"]),
        "route_id": config_data.get("route_id"),
        "iterations": iterations,
        "warmup_iterations": warmup_iterations,
        "max_batches": max_batches,
        "cases": public_results,
    }


def write_benchmark_payload(payload: dict[str, Any], result_output: Path) -> None:
    result_output.parent.mkdir(parents=True, exist_ok=True)
    result_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_result_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    expect(isinstance(data, dict), f"benchmark result is not a JSON object: {path}")
    return data


def timing_mean(case: dict[str, Any]) -> float | None:
    timing = case.get("operation_timing_ns")
    if isinstance(timing, dict):
        value = timing.get("mean")
        if isinstance(value, (int, float)):
            return float(value)
    if isinstance(timing, (int, float)):
        return float(timing)
    return None


def dispatch_mean(case: dict[str, Any]) -> float | None:
    value = case.get("mean_physical_dispatch_duration_ns")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def delta_percent(baseline: float | None, candidate: float | None) -> float | None:
    if baseline is None or candidate is None or baseline == 0:
        return None
    return ((candidate - baseline) / baseline) * 100.0


def case_map(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cases = data.get("cases")
    if not isinstance(cases, list):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for case in cases:
        if not isinstance(case, dict):
            continue
        case_id = case.get("case_id")
        if isinstance(case_id, str) and case_id:
            out[case_id] = case
    return out


def summary_value(summary: dict[str, Any], key: str) -> float | None:
    value = summary.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def comparison_summary(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    baseline_summary = (
        baseline.get("summary") if isinstance(baseline.get("summary"), dict) else {}
    )
    candidate_summary = (
        candidate.get("summary") if isinstance(candidate.get("summary"), dict) else {}
    )
    baseline_op = summary_value(baseline_summary, "mean_operation_timing_ns")
    candidate_op = summary_value(candidate_summary, "mean_operation_timing_ns")
    baseline_dispatch = summary_value(
        baseline_summary, "mean_physical_dispatch_duration_ns"
    )
    candidate_dispatch = summary_value(
        candidate_summary, "mean_physical_dispatch_duration_ns"
    )
    return {
        "baseline_case_count": baseline_summary.get("case_count"),
        "candidate_case_count": candidate_summary.get("case_count"),
        "baseline_mean_operation_timing_ns": baseline_op,
        "candidate_mean_operation_timing_ns": candidate_op,
        "mean_operation_timing_delta_ns": (
            candidate_op - baseline_op
            if baseline_op is not None and candidate_op is not None
            else None
        ),
        "mean_operation_timing_delta_pct": delta_percent(baseline_op, candidate_op),
        "baseline_mean_physical_dispatch_duration_ns": baseline_dispatch,
        "candidate_mean_physical_dispatch_duration_ns": candidate_dispatch,
        "mean_physical_dispatch_duration_delta_ns": (
            candidate_dispatch - baseline_dispatch
            if baseline_dispatch is not None and candidate_dispatch is not None
            else None
        ),
        "mean_physical_dispatch_duration_delta_pct": delta_percent(
            baseline_dispatch, candidate_dispatch
        ),
    }


def compare_cases(baseline: dict[str, Any], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    baseline_cases = case_map(baseline)
    candidate_cases = case_map(candidate)
    ordered_case_ids = sorted(set(baseline_cases) | set(candidate_cases))
    rows: list[dict[str, Any]] = []
    for current_case_id in ordered_case_ids:
        baseline_case = baseline_cases.get(current_case_id, {})
        candidate_case = candidate_cases.get(current_case_id, {})
        baseline_op = timing_mean(baseline_case)
        candidate_op = timing_mean(candidate_case)
        baseline_dispatch = dispatch_mean(baseline_case)
        candidate_dispatch = dispatch_mean(candidate_case)
        rows.append(
            {
                "case_id": current_case_id,
                "status": {
                    "baseline": baseline_case.get("status"),
                    "candidate": candidate_case.get("status"),
                },
                "values": candidate_case.get("values") or baseline_case.get("values"),
                "operation_timing_ns": {
                    "baseline_mean": baseline_op,
                    "candidate_mean": candidate_op,
                    "delta_ns": (
                        candidate_op - baseline_op
                        if baseline_op is not None and candidate_op is not None
                        else None
                    ),
                    "delta_pct": delta_percent(baseline_op, candidate_op),
                },
                "mean_physical_dispatch_duration_ns": {
                    "baseline": baseline_dispatch,
                    "candidate": candidate_dispatch,
                    "delta_ns": (
                        candidate_dispatch - baseline_dispatch
                        if baseline_dispatch is not None
                        and candidate_dispatch is not None
                        else None
                    ),
                    "delta_pct": delta_percent(baseline_dispatch, candidate_dispatch),
                },
            }
        )
    return rows


def compare_results_payload(baseline_path: Path, candidate_path: Path) -> dict[str, Any]:
    baseline = load_result_json(baseline_path)
    candidate = load_result_json(candidate_path)
    return {
        "schema": COMPARISON_SCHEMA,
        "kernel": candidate.get("kernel") or baseline.get("kernel"),
        "target": candidate.get("target") or baseline.get("target"),
        "summary": comparison_summary(baseline, candidate),
        "baseline": str(baseline_path),
        "candidate": str(candidate_path),
        "cases": compare_cases(baseline, candidate),
    }


def fmt_float(value: float | None, digits: int = 2) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def fmt_pct(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:+.2f}%"


def trend_marker(delta_pct: float | None) -> str:
    if delta_pct is None:
        return "="
    if delta_pct > 0:
        return "REGRESSION"
    if delta_pct < 0:
        return "IMPROVEMENT"
    return "UNCHANGED"


def comparison_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    cases = payload.get("cases") if isinstance(payload.get("cases"), list) else []
    lines = [
        f"# Benchmark Comparison: {payload.get('kernel', 'unknown')}",
        "",
        f"- Target: `{payload.get('target', '')}`",
        f"- Baseline: `{payload.get('baseline', '')}`",
        f"- Candidate: `{payload.get('candidate', '')}`",
        (
            f"- Mean op delta: `{fmt_float(summary.get('mean_operation_timing_delta_ns'))} ns` "
            f"({fmt_pct(summary.get('mean_operation_timing_delta_pct'))})"
        ),
        (
            f"- Mean dispatch delta: `{fmt_float(summary.get('mean_physical_dispatch_duration_delta_ns'))} ns` "
            f"({fmt_pct(summary.get('mean_physical_dispatch_duration_delta_pct'))})"
        ),
        "",
        "## Cases",
        "",
        "| Case | Status | Baseline mean ns | Candidate mean ns | Delta ns | Delta % | Trend |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for case in cases:
        status = case.get("status") if isinstance(case.get("status"), dict) else {}
        op = (
            case.get("operation_timing_ns")
            if isinstance(case.get("operation_timing_ns"), dict)
            else {}
        )
        current_delta_pct = (
            op.get("delta_pct") if isinstance(op.get("delta_pct"), (int, float)) else None
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    str(case.get("case_id", "")),
                    f"{status.get('baseline', '')} -> {status.get('candidate', '')}",
                    fmt_float(op.get("baseline_mean"), 1),
                    fmt_float(op.get("candidate_mean"), 1),
                    fmt_float(op.get("delta_ns"), 1),
                    fmt_pct(current_delta_pct),
                    trend_marker(current_delta_pct),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def write_comparison_outputs(
    payload: dict[str, Any],
    *,
    output_json: Path | None = None,
    markdown_output: Path | None = None,
) -> None:
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        if markdown_output is None:
            markdown_output = output_json.with_suffix(".md")
    if markdown_output is not None:
        markdown_output.parent.mkdir(parents=True, exist_ok=True)
        markdown_output.write_text(
            comparison_markdown(payload) + "\n", encoding="utf-8"
        )


def slug_for_path(path: Path) -> str:
    stem = path.stem
    return stem.removesuffix(".test")


def collect_config_paths(inputs: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for raw in inputs:
        path = Path(raw).resolve()
        if path.is_file():
            data = load_config(path)
            validate_config(data)
            paths.append(path)
            continue
        if path.is_dir():
            for candidate in sorted(path.rglob("*.json")):
                try:
                    data = load_config(candidate)
                    validate_config(data)
                except Exception:
                    continue
                paths.append(candidate)
            continue
        raise RuntimeError(f"input path does not exist: {path}")

    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        deduped.append(path)
    expect(deduped, "no benchmark configs were discovered")

    slugs = [slug_for_path(path) for path in deduped]
    duplicates = sorted({slug for slug in slugs if slugs.count(slug) > 1})
    expect(not duplicates, f"duplicate config stems are not supported: {duplicates}")
    return deduped


def materialize_benchmark_set(
    config_paths: list[Path],
    *,
    tool_dir: str | None,
    target: str,
    rocm_path: str | None,
    iterations: int,
    warmup_iterations: int,
    max_batches: int,
    artifact_root: Path,
    results_dir: Path,
    compare_results_dir: Path | None,
    comparison_dir: Path,
    require_tool: Any,
) -> tuple[dict[str, Any], bool]:
    manifest_results: list[dict[str, Any]] = []
    manifest_comparisons: list[dict[str, Any]] = []
    had_failures = False

    for config_path in config_paths:
        slug = slug_for_path(config_path)
        artifact_dir = artifact_root / slug
        result_output = results_dir / f"{slug}.json"
        payload = benchmark_config_payload(
            config_path,
            case_selectors=None,
            tool_dir=tool_dir,
            target=target,
            rocm_path=rocm_path,
            iterations=iterations,
            warmup_iterations=warmup_iterations,
            max_batches=max_batches,
            artifact_dir=artifact_dir,
            require_tool=require_tool,
        )
        write_benchmark_payload(payload, result_output)
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        failed_case_count = int(summary.get("failed_case_count") or 0)
        had_failures = had_failures or failed_case_count > 0
        result_record: dict[str, Any] = {
            "config_path": str(config_path),
            "artifact_dir": str(artifact_dir),
            "result_output": str(result_output),
            "kernel": payload.get("kernel"),
            "failed_case_count": failed_case_count,
        }
        manifest_results.append(result_record)

        if compare_results_dir is None:
            continue
        baseline_path = compare_results_dir / f"{slug}.json"
        if not baseline_path.is_file():
            result_record["baseline_result"] = None
            continue
        comparison_output = comparison_dir / f"{slug}.compare.json"
        comparison_payload = compare_results_payload(baseline_path, result_output)
        write_comparison_outputs(comparison_payload, output_json=comparison_output)
        manifest_comparisons.append(
            {
                "kernel": comparison_payload.get("kernel"),
                "baseline": str(baseline_path),
                "candidate": str(result_output),
                "comparison_json": str(comparison_output),
                "comparison_markdown": str(comparison_output.with_suffix(".md")),
            }
        )
        result_record["baseline_result"] = str(baseline_path)

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "target": target,
        "results_dir": str(results_dir),
        "artifact_root": str(artifact_root),
        "comparison_dir": str(comparison_dir),
        "compare_results_dir": str(compare_results_dir) if compare_results_dir else None,
        "results": manifest_results,
        "comparisons": manifest_comparisons,
    }
    return manifest, had_failures
