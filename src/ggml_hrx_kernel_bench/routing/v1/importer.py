from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from ...import_models import (
    ImportedCase,
    ImportedSuite,
    MappingStatus,
    ResolvedBenchmarkCase,
    UnmappedCase,
    UnmappedReason,
)
from ...import_route_resolution import resolve_case_routes, route_family
from .routes import iter_routes


def _routes_by_op(catalog_dir) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for route in iter_routes(catalog_dir):
        op = str(route.get("op") or "").upper()
        if op:
            grouped[op].append(route)
    return grouped


def _resolve_route(
    case: ImportedCase,
    routes_by_op: dict[str, list[dict[str, Any]]],
) -> tuple[
    dict[str, Any] | None,
    dict[str, int] | None,
    list[dict[str, Any]],
    UnmappedReason | None,
    str | None,
]:
    op_routes = list(routes_by_op.get(case.op.upper(), ()))
    resolution, candidates, reason, detail = resolve_case_routes(case, op_routes)
    if resolution is None:
        return None, None, candidates, reason, detail
    return resolution.route, resolution.shape, candidates, None, None


def _candidate_kernel_families(routes: Iterable[dict[str, Any]]) -> tuple[str, ...]:
    families = sorted({route_family(route) for route in routes if route_family(route)})
    return tuple(families)


def _unmapped_case(
    case: ImportedCase,
    *,
    status: MappingStatus,
    reason: UnmappedReason,
    detail: str | None = None,
    candidate_kernel_families: tuple[str, ...] = (),
    candidate_route_ids: tuple[str, ...] = (),
) -> UnmappedCase:
    return UnmappedCase(
        imported=case,
        mapping_status=status,
        reason=reason,
        detail=detail,
        candidate_kernel_families=candidate_kernel_families,
        candidate_route_ids=candidate_route_ids,
    )


def resolve_imported_suite(suite: ImportedSuite, *, catalog_dir) -> ImportedSuite:
    routes_by_op = _routes_by_op(catalog_dir)
    resolved: list[ResolvedBenchmarkCase] = []
    unmapped: list[UnmappedCase] = []

    for group in suite.op_groups:
        for case in group.cases:
            op_routes = list(routes_by_op.get(case.op.upper(), ()))
            if not op_routes:
                unmapped.append(
                    _unmapped_case(
                        case,
                        status=MappingStatus.UNMAPPED,
                        reason=UnmappedReason.NO_KERNEL_FAMILY_MAPPING,
                        detail="no catalog route exists for this op",
                    )
                )
                continue

            route, shape, route_candidates, route_reason, route_detail = _resolve_route(
                case,
                routes_by_op,
            )
            if route_reason is not None:
                status = (
                    MappingStatus.AMBIGUOUS
                    if route_reason == UnmappedReason.AMBIGUOUS_ROUTE_MATCH
                    else MappingStatus.UNMAPPED
                )
                unmapped.append(
                    _unmapped_case(
                        case,
                        status=status,
                        reason=route_reason,
                        detail=route_detail
                        or f"could not resolve a unique route for op {case.op}",
                        candidate_kernel_families=_candidate_kernel_families(
                            route_candidates or op_routes
                        ),
                        candidate_route_ids=tuple(
                            str(candidate.get("id") or "")
                            for candidate in route_candidates
                            if candidate.get("id")
                        ),
                    )
                )
                continue

            if shape is None:
                raise RuntimeError("resolved route is missing canonical shape")
            params = list(shape.keys())
            values = [int(shape[param]) for param in params]
            resolved.append(
                ResolvedBenchmarkCase(
                    imported=case,
                    kernel_family=route_family(route),
                    route_id=str(route.get("id") or ""),
                    params=list(params),
                    values=list(values),
                )
            )

    suite.resolved = resolved
    suite.unmapped = unmapped
    return suite
