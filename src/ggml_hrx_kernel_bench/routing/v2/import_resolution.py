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


def _parse_pointwise_parameters(case: ImportedCase) -> tuple[list[int], list[int], int, int]:
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
    return [int(value) for value in ne], [int(value) for value in nr], int(nf), int(perm1)


def _dimensions_from_extents(extents: list[int]) -> tuple[ConcreteTensorDimension, ...]:
    dimensions: list[ConcreteTensorDimension] = []
    stride = 1
    for index, extent in enumerate(extents):
        dimensions.append(
            ConcreteTensorDimension(name=f"d{index}", size=int(extent), stride=stride)
        )
        stride *= int(extent)
    return tuple(dimensions)


def _fallback_shape_from_extents(extents: list[int]) -> dict[str, int]:
    return normalize_shape(
        {
            "ncols": int(extents[0]),
            "nrows": int(extents[1]) * int(extents[2]) * int(extents[3]),
        }
    )


def lower_contiguous_pointwise_shape(case: ImportedCase) -> dict[str, int]:
    ne, nr, nf, perm1 = _parse_pointwise_parameters(case)
    if nf != 1:
        raise ValueError("contiguous pointwise routing requires nf=1")
    if perm1 != 0:
        raise ValueError("contiguous pointwise routing requires perm1=0")
    if any(int(value) != 1 for value in nr):
        raise ValueError("contiguous pointwise routing requires same-shape inputs")
    return _fallback_shape_from_extents(ne)


def lower_contiguous_pointwise_tensors(
    case: ImportedCase,
) -> tuple[dict[str, ConcreteTensor], dict[str, int]]:
    shape = lower_contiguous_pointwise_shape(case)
    ne, _, _, _ = _parse_pointwise_parameters(case)
    dtype = str(case.dtype.get("type", "")).upper()
    dimensions = _dimensions_from_extents(ne)
    tensors = {
        tensor_name: ConcreteTensor(dtype=dtype, dimensions=dimensions)
        for tensor_name in ("src0", "src1", "dst")
    }
    return tensors, shape


def lower_generic_pointwise_tensors(
    case: ImportedCase,
) -> tuple[dict[str, ConcreteTensor], dict[str, int]]:
    ne, nr, nf, perm1 = _parse_pointwise_parameters(case)
    if nf != 1:
        raise ValueError("generic pointwise routing requires nf=1")
    if perm1 != 0:
        raise ValueError("generic pointwise routing requires perm1=0")
    dst_extents = [int(extent) * int(repeat) for extent, repeat in zip(ne, nr)]
    dtype = str(case.dtype.get("type", "")).upper()
    tensors = {
        "src0": ConcreteTensor(dtype=dtype, dimensions=_dimensions_from_extents(dst_extents)),
        "src1": ConcreteTensor(dtype=dtype, dimensions=_dimensions_from_extents(ne)),
        "dst": ConcreteTensor(dtype=dtype, dimensions=_dimensions_from_extents(dst_extents)),
    }
    return tensors, _fallback_shape_from_extents(dst_extents)


def _parse_unary_parameters(case: ImportedCase) -> tuple[list[int], int]:
    # ggml unary-op params: ne_a = input tensor's 4-D shape [ne0..ne3];
    # v = view flag (0 = contiguous input, 1 = non-contiguous strided view).
    ne = case.normalized_params.get("ne_a")
    v = case.normalized_params.get("v", 0)
    if not isinstance(ne, list):
        raise ValueError("unary lowering requires ne_a array")
    if len(ne) != 4:
        raise ValueError("unary lowering requires 4-D extents")
    if any(not isinstance(value, int) for value in ne):
        raise ValueError("unary lowering requires integer extents")
    if not isinstance(v, int):
        raise ValueError("unary lowering requires integer v")
    return [int(value) for value in ne], int(v)


def lower_contiguous_unary_tensors(
    case: ImportedCase,
) -> tuple[dict[str, ConcreteTensor], dict[str, int]]:
    ne, v = _parse_unary_parameters(case)
    if v != 0:
        raise ValueError("contiguous unary routing requires contiguous input (v=0)")
    dtype = str(case.dtype.get("type", "")).upper()
    dimensions = _dimensions_from_extents(ne)
    tensors = {
        tensor_name: ConcreteTensor(dtype=dtype, dimensions=dimensions)
        for tensor_name in ("src0", "dst")
    }
    return tensors, _fallback_shape_from_extents(ne)


def lower_tensors_for_route(
    case: ImportedCase,
    route: V2Route,
) -> tuple[dict[str, ConcreteTensor], dict[str, int]]:
    if route.op == "ABS":
        return lower_contiguous_unary_tensors(case)
    if route.id == "add_f32_generic_4d":
        return lower_generic_pointwise_tensors(case)
    return lower_contiguous_pointwise_tensors(case)


def shape_for_resolved_route(route: V2Route, tensors: dict[str, ConcreteTensor], fallback_shape: dict[str, int]) -> dict[str, int]:
    if not route.tensors:
        return fallback_shape
    ranked_capture = next(
        (
            descriptor.dimensions_capture
            for descriptor in route.tensors.values()
            for check in route.constraints.checks
            if check.name == descriptor.dimensions_capture and check.length is not None
        ),
        None,
    )
    rank = next(
        (
            int(check.length)
            for check in route.constraints.checks
            if check.name == ranked_capture and check.length is not None
        ),
        None,
    )
    if rank is None:
        return fallback_shape
    if rank == 4:
        tensor_name = "dst" if "dst" in tensors else next(iter(route.tensors))
        first_tensor = tensors[tensor_name]
        return {
            dimension.name: int(dimension.size)
            for dimension in first_tensor.dimensions
        }
    if rank == 2:
        return fallback_shape
    raise ValueError(
        f"v2 route {route.id!r} resolved unsupported rank {rank!r} for capture {ranked_capture!r}"
    )


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

    lowering_errors: list[str] = []
    lowered_any = False
    for route in dtype_matching:
        try:
            tensors, shape = lower_tensors_for_route(case, route)
        except ValueError as exc:
            lowering_errors.append(str(exc))
            continue
        lowered_any = True
        if not route_accepts_tensors(route, tensors):
            continue
        return route, shape_for_resolved_route(route, tensors, shape), None, None
    if lowered_any:
        return (
            None,
            None,
            UnmappedReason.NO_ROUTE_MATCH,
            "lowered tensor descriptors did not satisfy any v2 route",
        )
    return (
        None,
        None,
        UnmappedReason.SHAPE_LOWERING_NOT_IMPLEMENTED,
        lowering_errors[0] if lowering_errors else "pointwise lowering is not implemented for this route set",
    )


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
