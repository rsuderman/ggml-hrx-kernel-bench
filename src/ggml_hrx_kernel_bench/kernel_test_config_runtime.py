from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .cli import run_candidate_row
from .config import BenchConfig, ToolPaths
from .family_specs import normalize_shape
from .hrx2 import (
    Candidate,
    PROJECT_ROOT,
    build_config,
    iter_routes,
    load_sources_by_id,
    route_launch,
    stable_id,
)
from .kernel_test_config import expect, load_config
from .reporting import correctness_ok


ROOT = PROJECT_ROOT


def case_id(params: list[str], values: list[int]) -> str:
    return "_".join(f"{name}{value}" for name, value in zip(params, values, strict=True))


def list_cases(config: dict[str, Any]) -> list[tuple[str, list[int]]]:
    params = list(config["params"])
    return [(case_id(params, list(values)), list(values)) for values in config["cases"]]


def select_case(config: dict[str, Any], selector: str) -> tuple[str, list[int]]:
    cases = list_cases(config)
    if selector.isdigit():
        index = int(selector)
        expect(0 <= index < len(cases), f"case index out of range: {index}")
        return cases[index]
    for current_case_id, values in cases:
        if current_case_id == selector:
            return current_case_id, values
    raise RuntimeError(f"case not found in config: {selector}")


def select_cases(
    config: dict[str, Any], selectors: list[str] | None
) -> list[tuple[str, list[int]]]:
    if not selectors:
        return list_cases(config)
    return [select_case(config, selector) for selector in selectors]


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
    config_data: dict[str, Any],
    current_case_id: str,
    current_case_values: list[int],
) -> Candidate:
    family = str(config_data["kernel"])
    catalog_dir = ROOT / "catalog" / "hrx2"
    kernel_dir = ROOT / "kernels" / "hrx2"
    route = select_route(catalog_dir, family=family, route_id=config_data.get("route_id"))
    shape = shape_for_case(config_data, current_case_values)
    config_bindings, values, missing = build_config(route, shape)
    expect(not missing, f"missing shape/config bindings: {missing}")
    sources = load_sources_by_id(kernel_dir, catalog_dir)
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
            iree_benchmark_loom=Path(
                require_tool("iree-benchmark-loom", tool_dir=tool_dir)
            ),
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
    candidate = build_candidate(config_data, current_case_id, current_case_values)
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
    row = run_candidate_row(run_args, bench_config, candidate, sanitizer="none")
    summary = (row.get("benchmark") or {}).get("summary") or {}
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
    benchmark = row.get("benchmark") or {}
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
        "results_path": benchmark.get("results_path"),
        "artifact_bundle_dir": benchmark.get("artifact_bundle_dir"),
        "output_dir": str(output_dir),
    }
