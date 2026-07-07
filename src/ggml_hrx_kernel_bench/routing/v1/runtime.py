from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ...cli import run_candidate_test_row
from ...config import BenchConfig, ToolPaths
from .family_specs import normalize_shape
from ...kernel_test_config import expect
from .._config import resolve_kernel_dir, resolve_routing_dir
from ...reporting import correctness_ok
from .routes import (
    Candidate,
    build_config,
    iter_routes,
    load_sources_by_id,
    route_launch,
    stable_id,
)


def select_route(
    catalog_dir: Path, *, family: str, route_id: str | None = None
) -> dict[str, Any]:
    matches = [
        route
        for route in iter_routes(catalog_dir)
        if str(route.get("family") or route.get("source_id") or "") == family
    ]
    expect(matches, f"no route found for family={family}")
    if route_id is not None:
        route_matches = [
            route for route in matches if str(route.get("id") or "") == route_id
        ]
        expect(route_matches, f"no route found for family={family} route_id={route_id}")
        expect(
            len(route_matches) == 1,
            (
                f"expected exactly one route for family={family} route_id={route_id}, "
                f"found {len(route_matches)}"
            ),
        )
        return route_matches[0]
    expect(
        len(matches) == 1,
        f"minimal config requires exactly one route for {family}, found {len(matches)}",
    )
    return matches[0]


def shape_for_case(config: dict[str, Any], values: list[int]) -> dict[str, int]:
    params = list(config["params"])
    expect(len(params) == len(values), "params and case values must have the same length")
    return normalize_shape(dict(zip(params, values, strict=True)))


def build_candidate(
    *,
    kernel_dir: Path,
    routing_dir: Path,
    config_data: dict[str, Any],
    current_case_id: str,
    current_case_values: list[int],
) -> Candidate:
    family = str(config_data["kernel"])
    route = select_route(routing_dir, family=family, route_id=config_data.get("route_id"))
    shape = shape_for_case(config_data, current_case_values)
    config_bindings, values, missing = build_config(route, shape)
    expect(not missing, f"missing shape/config bindings: {missing}")
    sources = load_sources_by_id(kernel_dir, routing_dir)
    source_id = str(route.get("source_id") or family)
    source = sources.get(source_id)
    expect(source is not None, f"kernel source is not available for source_id={source_id}")
    candidate_id = (
        f"{family}_{current_case_id}_{stable_id(shape, config_bindings, length=8)}"
    )
    return Candidate(
        id=candidate_id,
        family=family,
        op=str(route.get("op") or ""),
        source_id=source_id,
        source_path=source.path,
        root_symbol=str(route.get("root_symbol") or ""),
        export_name=route.get("export_name"),
        route_id=str(route.get("id") or ""),
        route=route,
        shape=shape,
        values=values,
        config=config_bindings,
        dispatch=route_launch(route, shape),
        supports=dict(route.get("supports") or {}),
        coverage="route_backed",
    )


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
            iree_test_loom=Path(require_tool("iree-test-loom", tool_dir=tool_dir)),
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


def execute_case(
    *,
    kernel_dir: Path | None,
    routing_dir: Path | None,
    config_data: dict[str, Any],
    current_case_id: str,
    current_case_values: list[int],
    tool_dir: str | None,
    target: str,
    rocm_path: str | None,
    iterations: int,
    warmup_iterations: int,
    max_batches: int,
    output_dir: Path,
    require_tool: Any,
) -> tuple[Candidate, dict[str, Any], dict[str, Any]]:
    resolved_kernel_dir = resolve_kernel_dir("v1", kernel_dir)
    resolved_routing_dir = resolve_routing_dir("v1", routing_dir)
    candidate = build_candidate(
        kernel_dir=resolved_kernel_dir,
        routing_dir=resolved_routing_dir,
        config_data=config_data,
        current_case_id=current_case_id,
        current_case_values=current_case_values,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    bench_config = build_bench_config(
        tool_dir=tool_dir,
        target=target,
        rocm_path=rocm_path,
        output_dir=output_dir,
        require_tool=require_tool,
    )
    run_args = build_run_args(
        output_dir=output_dir,
        target=target,
        rocm_path=rocm_path,
        iterations=iterations,
        warmup_iterations=warmup_iterations,
        max_batches=max_batches,
    )
    row = run_candidate_test_row(run_args, bench_config, candidate, sanitizer="none")
    summary = (row.get("test") or {}).get("summary") or {}
    return candidate, row, summary


def case_result(
    *,
    candidate: Candidate,
    current_case_id: str,
    current_case_values: list[int],
    row: dict[str, Any],
    summary: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    test = row.get("test") or {}
    return {
        "case_id": current_case_id,
        "values": list(current_case_values),
        "shape": dict(candidate.shape),
        "candidate_id": candidate.id,
        "status": row.get("status"),
        "correctness": summary.get("correctness"),
        "correctness_ok": correctness_ok(summary.get("correctness")),
        "operation_timing_ns": summary.get("operation_timing_ns"),
        "mean_physical_dispatch_duration_ns": summary.get(
            "mean_physical_dispatch_duration_ns"
        ),
        "physical_dispatches_per_logical_operation": summary.get(
            "physical_dispatches_per_logical_operation"
        ),
        "failure": summary.get("failure"),
        "results_path": test.get("results_path"),
        "stderr_path": test.get("stderr_path"),
        "stdout_path": test.get("stdout_path"),
        "artifact_bundle_dir": None,
        "output_dir": str(output_dir),
    }
