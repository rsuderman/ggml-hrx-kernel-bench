from __future__ import annotations

from ...import_models import (
    ImportedCase,
    ImportedSuite,
    MappingStatus,
    ResolvedBenchmarkCase,
    UnmappedCase,
    UnmappedReason,
)
from .matching import route_accepts_dtype, route_accepts_tensors
from .models import ConcreteTensor, ConcreteTensorDimension, V2Route
from .query import RouteCatalog, require_route_catalog, routes_for_op
from .shape import normalize_shape


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


def lower_contiguous_pointwise_shape(case: ImportedCase) -> dict[str, int]:
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
    return normalize_shape(shape)


def lower_contiguous_pointwise_tensors(
    case: ImportedCase,
) -> tuple[dict[str, ConcreteTensor], dict[str, int]]:
    shape = lower_contiguous_pointwise_shape(case)
    dtype = str(case.dtype.get("type", "")).upper()
    ncols = int(shape["ncols"])
    nrows = int(shape["nrows"])
    dimensions = (
        ConcreteTensorDimension(name="ncols", size=ncols, stride=1),
        ConcreteTensorDimension(name="nrows", size=nrows, stride=ncols),
    )
    tensors = {
        tensor_name: ConcreteTensor(dtype=dtype, dimensions=dimensions)
        for tensor_name in ("src0", "src1", "dst")
    }
    return tensors, shape


def resolve_route_for_case(
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

    try:
        tensors, shape = lower_contiguous_pointwise_tensors(case)
    except ValueError as exc:
        return None, None, UnmappedReason.SHAPE_LOWERING_NOT_IMPLEMENTED, str(exc)

    matching = [route for route in dtype_matching if route_accepts_tensors(route, tensors)]
    if not matching:
        return (
            None,
            None,
            UnmappedReason.NO_ROUTE_MATCH,
            "lowered tensor descriptors did not satisfy any v2 route",
        )
    if len(matching) > 1:
        return None, None, UnmappedReason.AMBIGUOUS_ROUTE_MATCH, None
    return matching[0], shape, None, None


def resolve_imported_suite(
    suite: ImportedSuite,
    *,
    routing_dir=None,
    catalog: RouteCatalog | None = None,
) -> ImportedSuite:
    resolved_catalog = require_route_catalog(routing_dir=routing_dir, catalog=catalog)
    suite.resolved = []
    suite.unmapped = []
    for group in suite.op_groups:
        op_routes = list(routes_for_op(resolved_catalog, group.op))
        for case in group.cases:
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
            route, shape, reason, detail = resolve_route_for_case(case, op_routes)
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
