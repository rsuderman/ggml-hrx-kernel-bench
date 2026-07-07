from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ...cli import run_candidate_row
from ...config import BenchConfig, ToolPaths
from ...reporting import correctness_ok
from ..models import ExecutedCase, RuntimeCaseRequest
from .candidates import candidate_from_shape
from .layout import decode_shape
from .matching import materialize_route_tensors, route_accepts_tensors
from .query import RouteCatalog, select_route


def shape_for_case(config: dict[str, Any], values: list[int]) -> dict[str, int]:
    return decode_shape(list(config["params"]), values)


def build_bench_config(
    *,
    tool_dir: str | None,
    target: str,
    rocm_path: str | None,
    output_dir: Path,
    require_tool: Any,
) -> BenchConfig:
    return BenchConfig(
        output_dir=output_dir,
        target=target,
        tools=ToolPaths(
            loom_link=Path(require_tool("loom-link", tool_dir=tool_dir)),
            iree_benchmark_loom=Path(require_tool("iree-benchmark-loom", tool_dir=tool_dir)),
        ),
        rocm_path=Path(rocm_path).resolve() if rocm_path else None,
    )


def build_run_args(
    *,
    output_dir: Path,
    target: str,
    rocm_path: str | None,
    iterations: int,
    warmup_iterations: int,
    max_batches: int,
) -> argparse.Namespace:
    return argparse.Namespace(
        output_dir=output_dir,
        target=target,
        rocm_path=Path(rocm_path).resolve() if rocm_path else None,
        iterations=iterations,
        warmup_iterations=warmup_iterations,
        max_batches=max_batches,
    )


def execute_case(request: RuntimeCaseRequest, *, catalog: RouteCatalog, kernel_dir: Path) -> ExecutedCase:
    route = select_route(
        catalog,
        family=str(request.config_data["kernel"]),
        route_id=request.config_data.get("route_id"),
    )
    shape = shape_for_case(request.config_data, request.current_case_values)
    tensors = materialize_route_tensors(route, shape)
    if not route_accepts_tensors(route, tensors):
        raise RuntimeError(
            f"v2 route {route.id!r} does not accept shape {json.dumps(shape, sort_keys=True)}"
        )
    candidate = candidate_from_shape(kernel_dir=kernel_dir, route=route, shape=shape)
    if candidate.status != "planned":
        raise RuntimeError(candidate.message or f"candidate {candidate.id} is not runnable")
    request.output_dir.mkdir(parents=True, exist_ok=True)
    bench_config = build_bench_config(
        tool_dir=request.tool_dir,
        target=request.target,
        rocm_path=request.rocm_path,
        output_dir=request.output_dir,
        require_tool=request.require_tool,
    )
    run_args = build_run_args(
        output_dir=request.output_dir,
        target=request.target,
        rocm_path=request.rocm_path,
        iterations=request.iterations,
        warmup_iterations=request.warmup_iterations,
        max_batches=request.max_batches,
    )
    row = run_candidate_row(run_args, bench_config, candidate, sanitizer="none")
    summary = (row.get("benchmark") or {}).get("summary") or {}
    return ExecutedCase(
        candidate=candidate,
        row=row,
        summary=summary,
        current_case_id=request.current_case_id,
        current_case_values=list(request.current_case_values),
        output_dir=request.output_dir,
    )


def case_result(execution: ExecutedCase) -> dict[str, object]:
    benchmark = execution.row.get("benchmark") or {}
    return {
        "case_id": execution.current_case_id,
        "values": list(execution.current_case_values),
        "shape": dict(execution.candidate.shape),
        "candidate_id": execution.candidate.id,
        "status": execution.row.get("status"),
        "correctness": execution.summary.get("correctness"),
        "correctness_ok": correctness_ok(execution.summary.get("correctness")),
        "operation_timing_ns": execution.summary.get("operation_timing_ns"),
        "mean_physical_dispatch_duration_ns": execution.summary.get(
            "mean_physical_dispatch_duration_ns"
        ),
        "physical_dispatches_per_logical_operation": execution.summary.get(
            "physical_dispatches_per_logical_operation"
        ),
        "failure": execution.summary.get("failure"),
        "results_path": benchmark.get("results_path"),
        "artifact_bundle_dir": benchmark.get("artifact_bundle_dir"),
        "output_dir": str(execution.output_dir),
    }
