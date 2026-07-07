from __future__ import annotations

from ...import_models import (
    ImportedCase,
    ImportedSuite,
    MappingStatus,
    ResolvedBenchmarkCase,
    UnmappedCase,
    UnmappedReason,
)
from .matching import (
    route_accepts_dtype,
    route_accepts_tensors,
    shape_overrides_from_tensors,
    shape_permutations_for_route,
)
from .models import ConcreteTensor, ConcreteTensorDimension, V2Route
from .query import RouteCatalog, require_route_catalog, routes_for_op

POINTWISE_BASE_PERMUTATION = (0, 1, 2, 3)
POINTWISE_SRC1_PERMUTATION = (1, 2, 0, 3)
COPY_BASE_PERMUTATION = (0, 1, 2, 3)
COPY_IDENTITY_PERMUTATION = (0, 0, 0, 0)
COPY_TRANSPOSE_01_PERMUTATION = (1, 0, 2, 3)


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


def _parse_pointwise_permutation(raw: object) -> tuple[int, int, int, int]:
    if not isinstance(raw, list) or len(raw) != 4:
        raise ValueError("pointwise lowering requires a 4-D perm1 permutation list")
    if any(not isinstance(value, int) for value in raw):
        raise ValueError("pointwise lowering requires integer perm1 values")
    permutation = tuple(int(value) for value in raw)
    if tuple(sorted(permutation)) != POINTWISE_BASE_PERMUTATION:
        raise ValueError("pointwise lowering requires perm1 to be a permutation of [0, 1, 2, 3]")
    return permutation


def _parse_pointwise_parameters(
    case: ImportedCase,
) -> tuple[list[int], list[int], int, tuple[int, int, int, int]]:
    ne = case.normalized_params.get("ne")
    nr = case.normalized_params.get("nr")
    nf = case.normalized_params.get("nf", 0)
    perm1 = case.normalized_params.get("perm1", list(POINTWISE_BASE_PERMUTATION))
    if not isinstance(ne, list) or not isinstance(nr, list):
        raise ValueError("pointwise lowering requires ne and nr arrays")
    if len(ne) != 4 or len(nr) != 4:
        raise ValueError("pointwise lowering requires 4-D extents")
    if any(not isinstance(value, int) for value in (*ne, *nr)):
        raise ValueError("pointwise lowering requires integer extents")
    if not isinstance(nf, int):
        raise ValueError("pointwise lowering requires integer nf")
    return [int(value) for value in ne], [int(value) for value in nr], int(nf), _parse_pointwise_permutation(perm1)


def _dimensions_from_extents(extents: list[int]) -> tuple[ConcreteTensorDimension, ...]:
    dimensions: list[ConcreteTensorDimension] = []
    stride = 1
    for index, extent in enumerate(extents):
        dimensions.append(
            ConcreteTensorDimension(name=f"d{index}", size=int(extent), stride=stride)
        )
        stride *= int(extent)
    return tuple(dimensions)


def _dimensions_from_extents_and_strides(
    extents: list[int],
    strides: list[int],
) -> tuple[ConcreteTensorDimension, ...]:
    return tuple(
        ConcreteTensorDimension(name=f"d{index}", size=int(extent), stride=int(stride))
        for index, (extent, stride) in enumerate(zip(extents, strides))
    )


def _permuted_dimensions_from_extents(
    extents: list[int],
    permutation: tuple[int, int, int, int],
) -> tuple[ConcreteTensorDimension, ...]:
    if permutation == POINTWISE_BASE_PERMUTATION:
        return _dimensions_from_extents(extents)
    if permutation != POINTWISE_SRC1_PERMUTATION:
        raise ValueError(f"generic pointwise routing does not support perm1={list(permutation)!r}")
    base_extents = [int(extents[axis]) for axis in permutation]
    base_strides: list[int] = []
    stride = 1
    for extent in base_extents:
        base_strides.append(stride)
        stride *= extent
    logical_strides = [0, 0, 0, 0]
    for base_axis, logical_axis in enumerate(permutation):
        logical_strides[logical_axis] = base_strides[base_axis]
    return _dimensions_from_extents_and_strides(extents, logical_strides)


def _shape_from_extents(extents: list[int]) -> dict[str, int]:
    return {f"d{index}": int(extent) for index, extent in enumerate(extents)}


def _normalize_copy_permutation(
    permutation: tuple[int, int, int, int],
    *,
    name: str,
) -> tuple[int, int, int, int]:
    if permutation == COPY_IDENTITY_PERMUTATION:
        return COPY_BASE_PERMUTATION
    if tuple(sorted(permutation)) != COPY_BASE_PERMUTATION:
        raise ValueError(f"copy lowering requires {name} to be [0, 0, 0, 0] or a permutation of [0, 1, 2, 3]")
    return permutation


def _inverse_permutation(permutation: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    inverse = [0] * len(permutation)
    for index, value in enumerate(permutation):
        inverse[int(value)] = int(index)
    return (inverse[0], inverse[1], inverse[2], inverse[3])


def _chain_permutations(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    return (
        int(second[first[0]]),
        int(second[first[1]]),
        int(second[first[2]]),
        int(second[first[3]]),
    )


def _permuted_contiguous_strides(
    extents: list[int],
    permutation: tuple[int, int, int, int],
) -> list[int]:
    base_extents = [int(extents[axis]) for axis in permutation]
    base_strides: list[int] = []
    stride = 1
    for extent in base_extents:
        base_strides.append(stride)
        stride *= extent
    logical_strides = [0, 0, 0, 0]
    for base_axis, logical_axis in enumerate(permutation):
        logical_strides[logical_axis] = base_strides[base_axis]
    return logical_strides


def _parse_copy_parameters(
    case: ImportedCase,
) -> tuple[list[int], int, tuple[int, int, int, int], tuple[int, int, int, int]]:
    ne = case.normalized_params.get("ne")
    src_transpose = case.normalized_params.get("_src_transpose", 0)
    permute_src = case.normalized_params.get("permute_src", list(COPY_IDENTITY_PERMUTATION))
    permute_dst = case.normalized_params.get("permute_dst", list(COPY_IDENTITY_PERMUTATION))
    if not isinstance(ne, list) or len(ne) != 4:
        raise ValueError("copy lowering requires a 4-D ne extent list")
    if any(not isinstance(value, int) for value in ne):
        raise ValueError("copy lowering requires integer ne extents")
    if not isinstance(src_transpose, int):
        raise ValueError("copy lowering requires integer _src_transpose")
    for name, permutation in (("permute_src", permute_src), ("permute_dst", permute_dst)):
        if not isinstance(permutation, list) or len(permutation) != 4:
            raise ValueError(f"copy lowering requires {name} to be a 4-D permutation list")
        if any(not isinstance(value, int) for value in permutation):
            raise ValueError(f"copy lowering requires integer {name} values")
    return (
        [int(value) for value in ne],
        int(src_transpose),
        tuple(int(value) for value in permute_src),
        tuple(int(value) for value in permute_dst),
    )


def lower_contiguous_pointwise_shape(case: ImportedCase) -> dict[str, int]:
    ne, nr, nf, perm1 = _parse_pointwise_parameters(case)
    if nf != 1:
        raise ValueError("contiguous pointwise routing requires nf=1")
    if perm1 != POINTWISE_BASE_PERMUTATION:
        raise ValueError("contiguous pointwise routing requires perm1=[0, 1, 2, 3]")
    if any(int(value) != 1 for value in nr):
        raise ValueError("contiguous pointwise routing requires same-shape inputs")
    return _shape_from_extents(ne)


def lower_contiguous_pointwise_tensors(
    case: ImportedCase,
) -> tuple[dict[str, ConcreteTensor], dict[str, int]]:
    shape = lower_contiguous_pointwise_shape(case)
    ne, _, _, _ = _parse_pointwise_parameters(case)
    dtype = str(case.dtype.get("type", "")).upper()
    dimensions = _dimensions_from_extents(ne)
    tensors = {
        tensor_name: ConcreteTensor(
            dtype=dtype,
            dimensions=dimensions,
            permutation=POINTWISE_BASE_PERMUTATION,
        )
        for tensor_name in ("src0", "src1", "dst")
    }
    return tensors, shape


def lower_generic_pointwise_tensors(
    case: ImportedCase,
) -> tuple[dict[str, ConcreteTensor], dict[str, int]]:
    ne, nr, nf, perm1 = _parse_pointwise_parameters(case)
    if nf != 1:
        raise ValueError("generic pointwise routing requires nf=1")
    dst_extents = [int(extent) * int(repeat) for extent, repeat in zip(ne, nr)]
    dtype = str(case.dtype.get("type", "")).upper()
    tensors = {
        "src0": ConcreteTensor(
            dtype=dtype,
            dimensions=_dimensions_from_extents(dst_extents),
            permutation=POINTWISE_BASE_PERMUTATION,
        ),
        "src1": ConcreteTensor(
            dtype=dtype,
            dimensions=_permuted_dimensions_from_extents(ne, perm1),
            permutation=perm1,
        ),
        "dst": ConcreteTensor(
            dtype=dtype,
            dimensions=_dimensions_from_extents(dst_extents),
            permutation=POINTWISE_BASE_PERMUTATION,
        ),
    }
    return tensors, _shape_from_extents(dst_extents)


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


def lower_contiguous_copy_tensors(
    case: ImportedCase,
) -> tuple[dict[str, ConcreteTensor], dict[str, int]]:
    ne, src_transpose, permute_src, permute_dst = _parse_copy_parameters(case)
    normalized_src = _normalize_copy_permutation(permute_src, name="permute_src")
    normalized_dst = _normalize_copy_permutation(permute_dst, name="permute_dst")
    if normalized_src != COPY_BASE_PERMUTATION:
        raise ValueError("copy lowering requires permute_src=[0, 0, 0, 0]")
    if normalized_dst != COPY_BASE_PERMUTATION:
        raise ValueError("copy lowering requires permute_dst=[0, 0, 0, 0]")
    if src_transpose != 0:
        raise ValueError("copy lowering requires _src_transpose=0")
    src_dtype = str(case.dtype.get("type_src", "")).upper()
    dst_dtype = str(case.dtype.get("type_dst", "")).upper()
    dimensions = _dimensions_from_extents(ne)
    tensors = {
        "src0": ConcreteTensor(dtype=src_dtype, dimensions=dimensions),
        "dst": ConcreteTensor(dtype=dst_dtype, dimensions=dimensions),
    }
    return tensors, _shape_from_extents(ne)


def lower_transposed_copy_tensors(
    case: ImportedCase,
) -> tuple[dict[str, ConcreteTensor], dict[str, int]]:
    ne, src_transpose, permute_src, permute_dst = _parse_copy_parameters(case)
    normalized_src = _normalize_copy_permutation(permute_src, name="permute_src")
    normalized_dst = _normalize_copy_permutation(permute_dst, name="permute_dst")
    if src_transpose not in {0, 1}:
        raise ValueError("copy lowering requires _src_transpose to be 0 or 1")
    if src_transpose == 1 and normalized_dst != COPY_BASE_PERMUTATION:
        raise ValueError("copy transpose lowering requires permute_dst=[0, 0, 0, 0]")
    src_dtype = str(case.dtype.get("type_src", "")).upper()
    dst_dtype = str(case.dtype.get("type_dst", "")).upper()
    destination_permutation = (
        COPY_TRANSPOSE_01_PERMUTATION if src_transpose == 1 else normalized_dst
    )
    canonical_extents = [int(ne[axis]) for axis in destination_permutation]
    effective_src_permutation = _chain_permutations(
        normalized_src,
        _inverse_permutation(destination_permutation),
    )
    src0_strides = _permuted_contiguous_strides(canonical_extents, effective_src_permutation)
    tensors = {
        "src0": ConcreteTensor(
            dtype=src_dtype,
            dimensions=_dimensions_from_extents_and_strides(canonical_extents, src0_strides),
            permutation=normalized_src,
        ),
        "dst": ConcreteTensor(
            dtype=dst_dtype,
            dimensions=_dimensions_from_extents(canonical_extents),
            permutation=destination_permutation,
        ),
    }
    return tensors, _shape_from_extents(canonical_extents)


def _route_uses_generic_pointwise_lowering(route: V2Route) -> bool:
    has_rank4_dst = False
    has_src0_divides = False
    has_src1_divides = False
    for check in route.constraints.checks:
        if check.name == "dst_dimensions" and check.length == 4:
            has_rank4_dst = True
        if check.divides == ("src0_dimensions", "dst_dimensions"):
            has_src0_divides = True
        if check.divides == ("src1_dimensions", "dst_dimensions"):
            has_src1_divides = True
    return has_rank4_dst and has_src0_divides and has_src1_divides


def _route_uses_contiguous_copy_lowering(route: V2Route) -> bool:
    if route.op != "CPY" or set(route.tensors) != {"src0", "dst"}:
        return False
    src0_dimensions = route.tensors["src0"].dimensions_capture
    src0_strides = route.tensors["src0"].strides_capture
    dst_dimensions = route.tensors["dst"].dimensions_capture
    dst_strides = route.tensors["dst"].strides_capture
    has_total_size = any(
        value.name == "total_size" and value.product == dst_dimensions
        for value in route.values
    )
    has_contiguous_strides = any(
        value.name == "contiguous_strides" and value.contiguous_strides == dst_dimensions
        for value in route.values
    )
    has_equal_dimensions = any(
        set(check.equals) == {src0_dimensions, dst_dimensions}
        for check in route.constraints.checks
    )
    has_equal_strides = any(
        set(check.equals) == {"contiguous_strides", src0_strides, dst_strides}
        for check in route.constraints.checks
    )
    return has_total_size and has_contiguous_strides and has_equal_dimensions and has_equal_strides


def _route_uses_non_contiguous_copy_lowering(route: V2Route) -> bool:
    if route.op != "CPY" or set(route.tensors) != {"src0", "dst"}:
        return False
    src0_dimensions = route.tensors["src0"].dimensions_capture
    dst_dimensions = route.tensors["dst"].dimensions_capture
    dst_strides = route.tensors["dst"].strides_capture
    has_total_size = any(
        value.name == "total_size" and value.product == dst_dimensions
        for value in route.values
    )
    has_contiguous_strides = any(
        value.name == "contiguous_strides" and value.contiguous_strides == dst_dimensions
        for value in route.values
    )
    has_rank4_dst = any(
        check.name == "dst_dimensions" and check.length == 4
        for check in route.constraints.checks
    )
    has_equal_dimensions = any(
        set(check.equals) == {src0_dimensions, dst_dimensions}
        for check in route.constraints.checks
    )
    has_contiguous_dst = any(
        set(check.equals) == {"contiguous_strides", dst_strides}
        for check in route.constraints.checks
    )
    return has_total_size and has_contiguous_strides and has_rank4_dst and has_equal_dimensions and has_contiguous_dst


def lower_tensors_for_route(
    case: ImportedCase,
    route: V2Route,
) -> tuple[dict[str, ConcreteTensor], dict[str, int]]:
    if route.op == "ABS":
        return lower_contiguous_unary_tensors(case)
    if _route_uses_generic_pointwise_lowering(route):
        return lower_generic_pointwise_tensors(case)
    if _route_uses_non_contiguous_copy_lowering(route):
        return lower_transposed_copy_tensors(case)
    if _route_uses_contiguous_copy_lowering(route):
        return lower_contiguous_copy_tensors(case)
    return lower_contiguous_pointwise_tensors(case)


def shape_for_resolved_route(route: V2Route, tensors: dict[str, ConcreteTensor], fallback_shape: dict[str, int]) -> dict[str, int]:
    if not route.tensors:
        return fallback_shape
    tensor_name = "dst" if "dst" in tensors else next(iter(route.tensors))
    return {
        **{
            dimension.name: int(dimension.size)
            for dimension in tensors[tensor_name].dimensions
        },
        **shape_overrides_from_tensors(tensors),
        **shape_permutations_for_route(route, tensors),
    }


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
        lowering_errors[0] if lowering_errors else "shape lowering is not implemented for this route set",
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
