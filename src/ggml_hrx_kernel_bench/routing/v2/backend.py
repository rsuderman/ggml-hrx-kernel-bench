from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...cli import run_candidate_row
from ...config import BenchConfig, ToolPaths
from ...import_models import (
    ImportedCase,
    ImportedSuite,
    MappingStatus,
    ResolvedBenchmarkCase,
    UnmappedCase,
    UnmappedReason,
)
from ...reporting import correctness_ok
from ..case_selection import select_case as shared_select_case
from ..case_selection import select_cases as shared_select_cases
from ..models import (
    Candidate,
    CandidateQuery,
    ExecutedCase,
    ExportRequest,
    RoutingContext,
    RoutingExportResult,
    RuntimeCaseRequest,
)
from .descriptors import (
    V2Route,
    build_manifest,
    default_shape_for_route,
    iter_routes,
    load_routes,
    route_accepts_dtype,
    route_accepts_shape,
    route_dispatch,
    route_supports,
    source_path_for_route,
    stable_id,
)


def _normalize_shape(shape: dict[str, int]) -> dict[str, int]:
    normalized = {str(key): int(value) for key, value in shape.items()}
    if "ncols" in normalized and "cols" not in normalized:
        normalized["cols"] = normalized["ncols"]
    if "cols" in normalized and "ncols" not in normalized:
        normalized["ncols"] = normalized["cols"]
    if "nrows" in normalized and "rows" not in normalized:
        normalized["rows"] = normalized["nrows"]
    if "rows" in normalized and "nrows" not in normalized:
        normalized["nrows"] = normalized["rows"]
    return normalized


def _resolve_shape_source(source: str, shape: dict[str, int]) -> int | None:
    if not source.startswith("shape."):
        return None
    key = source.removeprefix("shape.")
    aliases: tuple[str, ...]
    if key == "ncols":
        aliases = ("ncols", "cols")
    elif key == "cols":
        aliases = ("cols", "ncols")
    elif key == "nrows":
        aliases = ("nrows", "rows")
    elif key == "rows":
        aliases = ("rows", "nrows")
    else:
        aliases = (key,)
    for alias in aliases:
        if alias in shape:
            return int(shape[alias])
    return None


def _build_config(
    route: V2Route, shape: dict[str, int]
) -> tuple[dict[str, str], dict[str, int | str], list[str]]:
    config: dict[str, str] = {}
    values: dict[str, int | str] = dict(shape)
    missing: list[str] = []
    for binding in route.bindings:
        key = str(binding["key"])
        if "source" in binding:
            source = str(binding["source"])
            value = _resolve_shape_source(source, shape)
            if value is None:
                missing.append(source)
                continue
            config[key] = str(value)
            values[source] = value
        else:
            config[key] = str(binding["value"])
    return config, values, missing


def _shape_for_case(config: dict[str, Any], values: list[int]) -> dict[str, int]:
    params = list(config["params"])
    if len(params) != len(values):
        raise RuntimeError("params and case values must have the same length")
    shape = _normalize_shape(dict(zip(params, values, strict=True)))
    if "ncols" in shape and "cols" in shape and int(shape["ncols"]) != int(shape["cols"]):
        raise RuntimeError("ncols and cols must match for v2 contiguous pointwise routes")
    if "nrows" in shape and "rows" in shape and int(shape["nrows"]) != int(shape["rows"]):
        raise RuntimeError("nrows and rows must match for v2 contiguous pointwise routes")
    return shape


def _candidate_from_shape(
    *,
    kernel_dir: Path,
    route: V2Route,
    shape: dict[str, int],
    status: str = "planned",
    message: str | None = None,
) -> Candidate:
    config, values, missing = _build_config(route, shape)
    if missing:
        status = "missing_config"
        message = "missing shape/config values: " + ", ".join(missing)
    return Candidate(
        id=f"{route.id}_{stable_id(route.id, shape, config, length=8)}",
        family=route.family,
        op=route.op,
        source_id=route.source_id,
        source_path=source_path_for_route(kernel_dir, route),
        root_symbol=route.root_symbol,
        export_name=route.export_name,
        route_id=route.id,
        route={
            "schema": "ggml_hrx_kernel_bench.routing_route.v2",
            "id": route.id,
            "family": route.family,
            "op": route.op,
            "source_id": route.source_id,
            "root_symbol": route.root_symbol,
            "export_name": route.export_name,
            "layout": route.layout,
            "match": dict(route.match),
            "launch": dict(route.launch),
        },
        shape=shape,
        values=values,
        config=config,
        dispatch=route_dispatch(route, shape),
        supports=route_supports(route),
        schedule=None,
        coverage="route_backed",
        status=status,
        message=message,
    )


def _unmapped_case(
    case: ImportedCase,
    *,
    status: MappingStatus,
    reason: UnmappedReason,
    detail: str | None = None,
    routes: list[V2Route] | None = None,
) -> UnmappedCase:
    candidate_routes = routes or []
    return UnmappedCase(
        imported=case,
        mapping_status=status,
        reason=reason,
        detail=detail,
        candidate_kernel_families=tuple(sorted({route.family for route in candidate_routes})),
        candidate_route_ids=tuple(route.id for route in candidate_routes),
    )


def _lower_contiguous_pointwise_shape(case: ImportedCase) -> dict[str, int]:
    ne = case.normalized_params.get("ne")
    nr = case.normalized_params.get("nr")
    nf = case.normalized_params.get("nf", 0)
    perm1 = case.normalized_params.get("perm1", 0)
    if not isinstance(ne, list) or not isinstance(nr, list):
        raise ValueError("pointwise lowering requires ne and nr arrays")
    if len(ne) != 4 or len(nr) != 4:
        raise ValueError("pointwise lowering requires 4-D extents")
    if any(not isinstance(value, int) for value in (*ne, *nr)):
        raise ValueError("pointwise lowering requires integer extents")
    if not isinstance(nf, int) or not isinstance(perm1, int):
        raise ValueError("pointwise lowering requires integer nf and perm1")
    if nf != 1:
        raise ValueError("contiguous pointwise routing requires nf=1")
    if perm1 != 0:
        raise ValueError("contiguous pointwise routing requires perm1=0")
    if any(int(value) != 1 for value in nr):
        raise ValueError("contiguous pointwise routing requires same-shape inputs")
    shape = {"ncols": int(ne[0]), "nrows": int(ne[1]) * int(ne[2]) * int(ne[3])}
    return _normalize_shape(shape)


def _resolve_route_for_case(
    case: ImportedCase,
    routes: list[V2Route],
) -> tuple[V2Route | None, dict[str, int] | None, UnmappedReason | None, str | None]:
    dtype_matching = [route for route in routes if route_accepts_dtype(route, case.dtype)]
    if not dtype_matching:
        return (
            None,
            None,
            UnmappedReason.NO_DTYPE_MAPPING,
            "matching v2 op mapping exists, but not for this dtype combination",
        )

    lowered: list[tuple[V2Route, dict[str, int]]] = []
    lowering_errors: list[str] = []
    for route in dtype_matching:
        try:
            shape = _lower_contiguous_pointwise_shape(case)
        except ValueError as exc:
            lowering_errors.append(str(exc))
            continue
        lowered.append((route, shape))

    matching = [
        (route, shape)
        for route, shape in lowered
        if route_accepts_shape(route, shape)
    ]
    if not matching:
        if lowered:
            return (
                None,
                None,
                UnmappedReason.NO_ROUTE_MATCH,
                "lowered shape did not satisfy any v2 route",
            )
        detail = lowering_errors[0] if lowering_errors else (
            "matching v2 op mapping exists, but no raw-case lowering is implemented"
        )
        return None, None, UnmappedReason.SHAPE_LOWERING_NOT_IMPLEMENTED, detail
    if len(matching) > 1:
        return None, None, UnmappedReason.AMBIGUOUS_ROUTE_MATCH, None
    route, shape = matching[0]
    return route, shape, None, None


def _routes_by_op(routing_dir: Path) -> dict[str, list[V2Route]]:
    grouped: dict[str, list[V2Route]] = {}
    for route in iter_routes(routing_dir):
        grouped.setdefault(route.op.upper(), []).append(route)
    return grouped


def _resolve_imported_suite(suite: ImportedSuite, *, routing_dir: Path) -> ImportedSuite:
    routes_by_op = _routes_by_op(routing_dir)
    suite.resolved = []
    suite.unmapped = []
    for group in suite.op_groups:
        for case in group.cases:
            op_routes = list(routes_by_op.get(case.op.upper(), ()))
            if not op_routes:
                suite.unmapped.append(
                    _unmapped_case(
                        case,
                        status=MappingStatus.UNMAPPED,
                        reason=UnmappedReason.NO_KERNEL_FAMILY_MAPPING,
                        detail="no v2 route exists for this op",
                    )
                )
                continue
            route, shape, reason, detail = _resolve_route_for_case(case, op_routes)
            if reason is not None:
                status = (
                    MappingStatus.AMBIGUOUS
                    if reason == UnmappedReason.AMBIGUOUS_ROUTE_MATCH
                    else MappingStatus.UNMAPPED
                )
                suite.unmapped.append(
                    _unmapped_case(
                        case,
                        status=status,
                        reason=reason,
                        detail=detail,
                        routes=op_routes,
                    )
                )
                continue
            if route is None or shape is None:
                raise RuntimeError("resolved v2 route is missing shape information")
            suite.resolved.append(
                ResolvedBenchmarkCase(
                    imported=case,
                    kernel_family=route.family,
                    route_id=route.id,
                    params=list(shape.keys()),
                    values=[int(shape[param]) for param in shape],
                )
            )
    return suite


def _select_runtime_route(routing_dir: Path, *, family: str, route_id: str | None) -> V2Route:
    matches = [route for route in iter_routes(routing_dir) if route.family == family]
    if not matches:
        raise RuntimeError(f"no v2 route found for family={family}")
    if route_id is not None:
        route_matches = [route for route in matches if route.id == route_id]
        if not route_matches:
            raise RuntimeError(f"no v2 route found for family={family} route_id={route_id}")
        if len(route_matches) != 1:
            raise RuntimeError(
                f"expected exactly one v2 route for family={family} route_id={route_id}, "
                f"found {len(route_matches)}"
            )
        return route_matches[0]
    if len(matches) != 1:
        raise RuntimeError(f"minimal v2 config requires exactly one route for {family}, found {len(matches)}")
    return matches[0]


def _build_bench_config(
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


def _build_run_args(
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


def _execute_case(request: RuntimeCaseRequest, *, routing_dir: Path) -> ExecutedCase:
    route = _select_runtime_route(
        routing_dir,
        family=str(request.config_data["kernel"]),
        route_id=request.config_data.get("route_id"),
    )
    shape = _shape_for_case(request.config_data, request.current_case_values)
    if not route_accepts_shape(route, shape):
        raise RuntimeError(
            f"v2 route {route.id!r} does not accept shape {json.dumps(shape, sort_keys=True)}"
        )
    kernel_dir = request.kernel_dir or Path.cwd()
    candidate = _candidate_from_shape(kernel_dir=kernel_dir, route=route, shape=shape)
    if candidate.status != "planned":
        raise RuntimeError(candidate.message or f"candidate {candidate.id} is not runnable")
    request.output_dir.mkdir(parents=True, exist_ok=True)
    bench_config = _build_bench_config(
        tool_dir=request.tool_dir,
        target=request.target,
        rocm_path=request.rocm_path,
        output_dir=request.output_dir,
        require_tool=request.require_tool,
    )
    run_args = _build_run_args(
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


def _case_result(execution: ExecutedCase) -> dict[str, object]:
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


@dataclass(frozen=True)
class V2RoutingBackend:
    context: RoutingContext
    version: str = "v2"

    def manifest(self, *, original_root: Path | None = None) -> dict[str, object]:
        return build_manifest(
            kernel_dir=self.context.kernel_dir,
            routing_dir=self.context.routing_dir,
        )

    def candidates(self, query: CandidateQuery) -> list[Candidate]:
        candidates: list[Candidate] = []
        for route in load_routes(self.context.routing_dir):
            if query.families and (
                route.family not in query.families
                and route.source_id not in query.families
                and route.id not in query.families
            ):
                continue
            source_path = source_path_for_route(self.context.kernel_dir, route)
            status = "planned" if source_path.exists() else "missing_source"
            message = None
            if status != "planned":
                message = f"kernel source is not available for source_id={route.source_id}"
            candidates.append(
                _candidate_from_shape(
                    kernel_dir=self.context.kernel_dir,
                    route=route,
                    shape=default_shape_for_route(route),
                    status=status,
                    message=message,
                )
            )
            if query.limit and len(candidates) >= query.limit:
                break
        return candidates

    def export(self, request: ExportRequest) -> RoutingExportResult:
        routes = load_routes(self.context.routing_dir)
        request.output_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = request.output_dir / "routing-export.json"
        payload = {
            "schema": "ggml_hrx_kernel_bench.routing_export_metadata.v2",
            "backend_version": self.version,
            "output_format": "routing-descriptor-v2",
            "target_key": request.target_key,
            "routing_id": request.routing_id,
            "family_count": len({route.family for route in routes}),
            "route_count": len(routes),
            "source_count": len({route.source_id for route in routes}),
            "route_ids": [route.id for route in routes],
        }
        metadata_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return RoutingExportResult(
            backend_version=self.version,
            output_format="routing-descriptor-v2",
            output_dir=request.output_dir,
            target_key=request.target_key,
            written_paths=(metadata_path,),
            metadata={
                "family_count": payload["family_count"],
                "route_count": payload["route_count"],
                "source_count": payload["source_count"],
            },
        )

    def resolve_imported_suite(self, suite: ImportedSuite) -> ImportedSuite:
        return _resolve_imported_suite(suite, routing_dir=self.context.routing_dir)

    def select_case(self, config: dict[str, object], selector: str) -> tuple[str, list[int]]:
        return shared_select_case(config, selector)

    def select_cases(
        self, config: dict[str, object], selectors: list[str] | None
    ) -> list[tuple[str, list[int]]]:
        return shared_select_cases(config, selectors)

    def execute_case(self, request: RuntimeCaseRequest) -> ExecutedCase:
        routing_dir = request.routing_dir or self.context.routing_dir
        return _execute_case(request, routing_dir=routing_dir)

    def case_result(self, execution: ExecutedCase) -> dict[str, object]:
        return _case_result(execution)
