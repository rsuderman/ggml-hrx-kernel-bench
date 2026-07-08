from __future__ import annotations

from ...import_models import (
    ImportedCase,
    ImportedSuite,
    MappingStatus,
    ResolvedBenchmarkCase,
    UnmappedCase,
    UnmappedReason,
)
from .layout import (
    EncodedRouteShape,
    chain_permutations,
    encode_route_shape,
    inverse_permutation,
    permuted_contiguous_strides,
)
from .matching import route_accepts_dtype, route_accepts_tensors
from .models import (
    LOWERING_KIND_COPY_CONTIGUOUS,
    LOWERING_KIND_COPY_NON_CONTIGUOUS_4D,
    ConcreteTensor,
    ConcreteTensorDimension,
    V2Route,
)
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
    if len(ne) < 1:
        raise ValueError("unary lowering requires at least one extent")
    if any(not isinstance(value, int) for value in ne):
        raise ValueError("unary lowering requires integer extents")
    if not isinstance(v, int):
        raise ValueError("unary lowering requires integer v")
    return [int(value) for value in ne], int(v)


def _contiguous_shape_metadata(extents: list[int], *, prefix: str) -> dict[str, int]:
    if not extents:
        raise ValueError("contiguous lowering requires at least one extent")
    trailing_product = 1
    for extent in extents[1:]:
        trailing_product *= int(extent)
    return {
        f"{prefix}.d1": trailing_product,
    }


def _lower_contiguous_extent_tensors(
    case: ImportedCase,
    *,
    extent_key: str,
    dtype_keys: tuple[str, str] = ("type", "type"),
    shape_prefix: str,
) -> tuple[dict[str, ConcreteTensor], dict[str, int]]:
    ne = case.normalized_params.get(extent_key)
    if not isinstance(ne, list) or len(ne) < 1:
        raise ValueError(f"contiguous lowering requires a non-empty {extent_key} extent list")
    if any(not isinstance(value, int) for value in ne):
        raise ValueError(f"contiguous lowering requires integer {extent_key} extents")
    src_dtype = str(case.dtype.get(dtype_keys[0], "")).upper()
    dst_dtype = str(case.dtype.get(dtype_keys[1], "")).upper()
    extents = [int(value) for value in ne]
    tensors = {
        "src0": ConcreteTensor(dtype=src_dtype, dimensions=_dimensions_from_extents(extents)),
        "dst": ConcreteTensor(dtype=dst_dtype, dimensions=_dimensions_from_extents(extents)),
    }
    return tensors, {
        **_shape_from_extents(extents),
        **_contiguous_shape_metadata(extents, prefix=shape_prefix),
    }


def lower_contiguous_unary_tensors(
    case: ImportedCase,
) -> tuple[dict[str, ConcreteTensor], dict[str, int]]:
    ne, v = _parse_unary_parameters(case)
    if v != 0:
        raise ValueError("contiguous unary routing requires contiguous input (v=0)")
    tensors, shape = _lower_contiguous_extent_tensors(
        ImportedCase(
            op=case.op,
            dtype=case.dtype,
            raw_case=case.raw_case,
            normalized_params={"ne_a": ne},
            source_path=case.source_path,
            source_group_index=case.source_group_index,
            source_case_index=case.source_case_index,
        ),
        extent_key="ne_a",
        shape_prefix="pointwise",
    )
    return tensors, shape


def _parse_cont_parameters(case: ImportedCase) -> list[int]:
    ne = case.normalized_params.get("ne")
    use_view_slice = case.normalized_params.get("use_view_slice", 0)
    if not isinstance(ne, list) or len(ne) < 2 or len(ne) > 4:
        raise ValueError("CONT lowering requires a 2-D to 4-D ne extent list")
    if any(not isinstance(value, int) for value in ne):
        raise ValueError("CONT lowering requires integer ne extents")
    if not isinstance(use_view_slice, int):
        raise ValueError("CONT lowering requires integer use_view_slice")
    if use_view_slice != 0:
        raise ValueError("CONT lowering requires contiguous source input (use_view_slice=0)")
    return [int(value) for value in ne]


def lower_contiguous_cont_tensors(
    case: ImportedCase,
) -> tuple[dict[str, ConcreteTensor], dict[str, int]]:
    extents = _parse_cont_parameters(case)
    tensors, shape = _lower_contiguous_extent_tensors(
        ImportedCase(
            op=case.op,
            dtype=case.dtype,
            raw_case=case.raw_case,
            normalized_params={"ne": extents},
            source_path=case.source_path,
            source_group_index=case.source_group_index,
            source_case_index=case.source_case_index,
        ),
        extent_key="ne",
        shape_prefix="cont",
    )
    return tensors, shape


def lower_contiguous_scale_or_clamp_tensors(
    case: ImportedCase,
) -> tuple[dict[str, ConcreteTensor], dict[str, int]]:
    return _lower_contiguous_extent_tensors(case, extent_key="ne", shape_prefix="pointwise")


def lower_rms_norm_tensors(
    case: ImportedCase,
) -> tuple[dict[str, ConcreteTensor], dict[str, int]]:
    ne = case.normalized_params.get("ne")
    eps = case.normalized_params.get("eps", 0.0)
    inplace = case.normalized_params.get("inplace", 0)
    view_mode = case.normalized_params.get("v", 0)
    if not isinstance(ne, list) or len(ne) != 4:
        raise ValueError("RMS_NORM lowering requires a 4-D ne extent list")
    if any(not isinstance(value, int) for value in ne):
        raise ValueError("RMS_NORM lowering requires integer ne extents")
    if isinstance(eps, bool):
        raise ValueError("RMS_NORM lowering requires numeric eps")
    if isinstance(eps, str):
        try:
            eps_value = float(eps)
        except ValueError as exc:
            raise ValueError("RMS_NORM lowering requires numeric eps") from exc
    elif isinstance(eps, (int, float)):
        eps_value = float(eps)
    else:
        raise ValueError("RMS_NORM lowering requires numeric eps")
    if not isinstance(inplace, int):
        raise ValueError("RMS_NORM lowering requires integer inplace")
    if not isinstance(view_mode, int):
        raise ValueError("RMS_NORM lowering requires integer v")
    if view_mode != 0:
        raise ValueError("RMS_NORM v2 routing requires contiguous input (v=0)")
    if inplace != 0:
        raise ValueError("RMS_NORM v2 routing requires inplace=0")
    if eps_value != 0.0:
        raise ValueError("RMS_NORM v2 routing currently requires eps=0.0")
    extents = [int(value) for value in ne]
    dimensions = _dimensions_from_extents(extents)
    dtype = str(case.dtype.get("type", "")).upper()
    tensors = {
        "src0": ConcreteTensor(dtype=dtype, dimensions=dimensions),
        "dst": ConcreteTensor(dtype=dtype, dimensions=dimensions),
    }
    return tensors, _shape_from_extents(extents)


def lower_swiglu_tensors(
    case: ImportedCase,
) -> tuple[dict[str, ConcreteTensor], dict[str, int]]:
    ne = case.normalized_params.get("ne_a")
    split = case.normalized_params.get("split", False)
    swapped = case.normalized_params.get("swapped", 0)
    view_mode = case.normalized_params.get("v", 0)
    if not isinstance(ne, list) or len(ne) != 4:
        raise ValueError("SWIGLU lowering requires a 4-D ne_a extent list")
    if any(not isinstance(value, int) for value in ne):
        raise ValueError("SWIGLU lowering requires integer ne_a extents")
    if not isinstance(split, bool):
        raise ValueError("SWIGLU lowering requires boolean split")
    if not isinstance(swapped, int):
        raise ValueError("SWIGLU lowering requires integer swapped")
    if not isinstance(view_mode, int):
        raise ValueError("SWIGLU lowering requires integer v")
    if view_mode != 0:
        raise ValueError("SWIGLU v2 routing requires contiguous input (v=0)")
    if split:
        raise ValueError("SWIGLU v2 routing currently requires packed input (split=false)")
    if swapped != 0:
        raise ValueError("SWIGLU v2 routing currently requires swapped=0")
    extents = [int(value) for value in ne]
    src0_extents = [int(extents[0]) * 2, *extents[1:]]
    dtype = str(case.dtype.get("type", "")).upper()
    tensors = {
        "src0": ConcreteTensor(
            dtype=dtype,
            dimensions=_dimensions_from_extents(src0_extents),
        ),
        "dst": ConcreteTensor(
            dtype=dtype,
            dimensions=_dimensions_from_extents(extents),
        ),
    }
    return tensors, _shape_from_extents(extents)


def lower_argsort_tensors(
    case: ImportedCase,
) -> tuple[dict[str, ConcreteTensor], dict[str, int]]:
    ne = case.normalized_params.get("ne")
    order = case.normalized_params.get("order", 0)
    if not isinstance(ne, list) or len(ne) != 4:
        raise ValueError("ARGSORT lowering requires a 4-D ne extent list")
    if any(not isinstance(value, int) for value in ne):
        raise ValueError("ARGSORT lowering requires integer ne extents")
    if not isinstance(order, int):
        raise ValueError("ARGSORT lowering requires integer order")
    if order != 0:
        raise ValueError("ARGSORT v2 routing currently requires order=0")
    ncols = int(ne[0])
    nrows = int(ne[1]) * int(ne[2]) * int(ne[3])
    dimensions = _dimensions_from_extents([ncols, nrows])
    tensors = {
        "src0": ConcreteTensor(
            dtype=str(case.dtype.get("type", "")).upper(),
            dimensions=dimensions,
        ),
        "dst": ConcreteTensor(
            dtype="I32",
            dimensions=dimensions,
        ),
    }
    return tensors, _shape_from_extents([ncols, nrows])


def lower_get_rows_tensors(
    case: ImportedCase,
) -> tuple[dict[str, ConcreteTensor], dict[str, int]]:
    embedding_extent_1 = case.normalized_params.get("be1")
    embedding_extent_2 = case.normalized_params.get("be2")
    src0_nrows = case.normalized_params.get("m")
    ncols = case.normalized_params.get("n")
    nrows = case.normalized_params.get("r")
    view_mode = case.normalized_params.get("v", 0)
    for name, value in (
        ("be1", embedding_extent_1),
        ("be2", embedding_extent_2),
        ("m", src0_nrows),
        ("n", ncols),
        ("r", nrows),
        ("v", view_mode),
    ):
        if not isinstance(value, int):
            raise ValueError(f"GET_ROWS lowering requires integer {name}")
    if view_mode != 0:
        raise ValueError("GET_ROWS v2 routing requires contiguous input (v=0)")
    if embedding_extent_1 != 1:
        raise ValueError("GET_ROWS v2 routing currently requires be1=1")
    if embedding_extent_2 != 1:
        raise ValueError("GET_ROWS v2 routing currently requires be2=1")
    dst_extents = [int(ncols), int(nrows)]
    src0_row_count = max(int(src0_nrows), int(nrows), 1)
    tensors = {
        "src0": ConcreteTensor(
            dtype=str(case.dtype.get("type", "")).upper(),
            dimensions=_dimensions_from_extents([int(ncols), src0_row_count]),
        ),
        "src1": ConcreteTensor(
            dtype="I32",
            dimensions=_dimensions_from_extents([1, int(nrows)]),
        ),
        "dst": ConcreteTensor(
            dtype=str(case.dtype.get("type", "")).upper(),
            dimensions=_dimensions_from_extents(dst_extents),
        ),
    }
    return tensors, {
        **_shape_from_extents(dst_extents),
        "get_rows.src0_nrows": src0_row_count,
        "get_rows.idx_row_stride": 1,
    }


def _rope_route_mode_and_freq(route: V2Route) -> tuple[int, int]:
    if route.root_symbol == "@hrx2_rope_normal_f32":
        return 0, 0
    if route.root_symbol == "@hrx2_rope_neox_f32":
        return 2, 0
    if route.root_symbol == "@hrx2_rope_normal_f32_freq":
        return 0, 1
    if route.root_symbol in {"@hrx2_rope_neox_f32_freq", "@hrx2_rope_neox_f32_freq_scale"}:
        return 2, 1
    raise ValueError(f"ROPE v2 routing is not implemented for {route.root_symbol}")


def lower_rope_tensors(
    case: ImportedCase,
    route: V2Route,
) -> tuple[dict[str, ConcreteTensor], dict[str, int]]:
    ne = case.normalized_params.get("ne_a")
    n_dims = case.normalized_params.get("n_dims")
    mode = case.normalized_params.get("mode", 0)
    freq_factor = case.normalized_params.get("ff", 0)
    freq_scale = case.normalized_params.get("fs", 1.0)
    ext_factor = case.normalized_params.get("ef", 0.0)
    attn_factor = case.normalized_params.get("af", 1.0)
    inplace = case.normalized_params.get("inplace", 0)
    view_mode = case.normalized_params.get("v", 0)
    if not isinstance(ne, list) or len(ne) != 4:
        raise ValueError("ROPE lowering requires a 4-D ne_a extent list")
    if any(not isinstance(value, int) for value in ne):
        raise ValueError("ROPE lowering requires integer ne_a extents")
    for name, value in (
        ("n_dims", n_dims),
        ("mode", mode),
        ("ff", freq_factor),
        ("inplace", inplace),
        ("v", view_mode),
    ):
        if not isinstance(value, int):
            raise ValueError(f"ROPE lowering requires integer {name}")
    for name, value in (("fs", freq_scale), ("ef", ext_factor), ("af", attn_factor)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"ROPE lowering requires numeric {name}")
    expected_mode, expected_freq_factor = _rope_route_mode_and_freq(route)
    if view_mode != 0:
        raise ValueError("ROPE v2 routing requires contiguous input (v=0)")
    if inplace != 0:
        raise ValueError("ROPE v2 routing currently requires inplace=0")
    if mode != expected_mode:
        raise ValueError(f"ROPE v2 routing currently requires mode={expected_mode}")
    if freq_factor != expected_freq_factor:
        raise ValueError(f"ROPE v2 routing currently requires ff={expected_freq_factor}")
    # The current v2 normal-mode ROPE kernel already accepts scalar frequency,
    # extension, and attention scaling through its launch ABI, so grouped-YAML
    # scale variants stay on the same route as long as mode/layout match.
    extents = [int(value) for value in ne]
    if route.root_symbol == "@hrx2_rope_neox_f32" and int(n_dims) != extents[0]:
        raise ValueError("ROPE NEOX v2 routing currently requires n_dims == ne_a[0]")
    ncols = extents[0]
    nheads = extents[1]
    ntokens = extents[2] * extents[3]
    dtype = str(case.dtype.get("type", "")).upper()
    tensors = {
        "src0": ConcreteTensor(
            dtype=dtype,
            dimensions=_dimensions_from_extents(extents),
        ),
        "src1": ConcreteTensor(
            dtype="I32",
            dimensions=_dimensions_from_extents([1, 1, ntokens, 1]),
        ),
        "dst": ConcreteTensor(
            dtype=dtype,
            dimensions=_dimensions_from_extents(extents),
        ),
    }
    return tensors, {
        **_shape_from_extents(extents),
        "rope.ncols": ncols,
        "rope.n_dims": int(n_dims),
        "rope.nheads": nheads,
        "rope.ntokens": ntokens,
        "rope.src0_head_stride": ncols,
        "rope.src0_token_stride": ncols * nheads,
        "rope.dst_head_stride": ncols,
        "rope.dst_token_stride": ncols * nheads,
        "rope.pos_token_stride": 1,
    }


def lower_soft_max_tensors(
    case: ImportedCase,
) -> tuple[dict[str, ConcreteTensor], dict[str, int]]:
    ne = case.normalized_params.get("ne")
    mask = case.normalized_params.get("mask", 0)
    sinks = case.normalized_params.get("sinks", 0)
    max_bias = case.normalized_params.get("max_bias", 0.0)
    inplace = case.normalized_params.get("inplace", 0)
    if not isinstance(ne, list) or len(ne) != 4:
        raise ValueError("SOFT_MAX lowering requires a 4-D ne extent list")
    if any(not isinstance(value, int) for value in ne):
        raise ValueError("SOFT_MAX lowering requires integer ne extents")
    for name, value in (("mask", mask), ("sinks", sinks), ("inplace", inplace)):
        if not isinstance(value, int):
            raise ValueError(f"SOFT_MAX lowering requires integer {name}")
    if isinstance(max_bias, bool):
        raise ValueError("SOFT_MAX lowering requires numeric max_bias")
    if isinstance(max_bias, str):
        try:
            max_bias_value = float(max_bias)
        except ValueError as exc:
            raise ValueError("SOFT_MAX lowering requires numeric max_bias") from exc
    elif isinstance(max_bias, (int, float)):
        max_bias_value = float(max_bias)
    else:
        raise ValueError("SOFT_MAX lowering requires numeric max_bias")
    if mask != 0:
        raise ValueError("SOFT_MAX v2 routing currently requires mask=0")
    if sinks != 0:
        raise ValueError("SOFT_MAX v2 routing currently requires sinks=0")
    if inplace != 0:
        raise ValueError("SOFT_MAX v2 routing currently requires inplace=0")
    if max_bias_value != 0.0:
        raise ValueError("SOFT_MAX v2 routing currently requires max_bias=0.0")
    extents = [int(value) for value in ne]
    ncols = extents[0]
    nrows = 1
    for extent in extents[1:]:
        nrows *= int(extent)
    if ncols > 1024:
        raise ValueError("SOFT_MAX v2 routing currently requires ne[0] <= 1024")
    tensors = {
        "src0": ConcreteTensor(
            dtype=str(case.dtype.get("type", "")).upper(),
            dimensions=_dimensions_from_extents([ncols, nrows]),
        ),
        "dst": ConcreteTensor(
            dtype=str(case.dtype.get("type", "")).upper(),
            dimensions=_dimensions_from_extents([ncols, nrows]),
        ),
    }
    return tensors, _shape_from_extents([ncols, nrows])


def lower_sum_rows_tensors(
    case: ImportedCase,
) -> tuple[dict[str, ConcreteTensor], dict[str, int]]:
    ne = case.normalized_params.get("ne")
    permute = case.normalized_params.get("permute", 0)
    slice_mode = case.normalized_params.get("slice", 0)
    if not isinstance(ne, list) or len(ne) != 4:
        raise ValueError("SUM_ROWS lowering requires a 4-D ne extent list")
    if any(not isinstance(value, int) for value in ne):
        raise ValueError("SUM_ROWS lowering requires integer ne extents")
    if not isinstance(permute, int):
        raise ValueError("SUM_ROWS lowering requires integer permute")
    if not isinstance(slice_mode, int):
        raise ValueError("SUM_ROWS lowering requires integer slice")
    if permute != 0:
        raise ValueError("SUM_ROWS v2 routing requires permute=0")
    if slice_mode != 0:
        raise ValueError("SUM_ROWS v2 routing requires slice=0")
    extents = [int(value) for value in ne]
    tensors = {
        "src0": ConcreteTensor(
            dtype=str(case.dtype.get("type", "")).upper(),
            dimensions=_dimensions_from_extents(extents),
        ),
    }
    return tensors, _shape_from_extents(extents)


def lower_set_rows_tensors(
    case: ImportedCase,
) -> tuple[dict[str, ConcreteTensor], dict[str, int]]:
    ne = case.normalized_params.get("ne")
    nr23 = case.normalized_params.get("nr23")
    row_count = case.normalized_params.get("r")
    view_mode = case.normalized_params.get("v", 0)
    if not isinstance(ne, list) or len(ne) != 4:
        raise ValueError("SET_ROWS lowering requires a 4-D ne extent list")
    if not isinstance(nr23, list) or len(nr23) != 2:
        raise ValueError("SET_ROWS lowering requires a 2-D nr23 extent list")
    if any(not isinstance(value, int) for value in (*ne, *nr23)):
        raise ValueError("SET_ROWS lowering requires integer extents")
    if not isinstance(row_count, int):
        raise ValueError("SET_ROWS lowering requires integer r")
    if not isinstance(view_mode, int):
        raise ValueError("SET_ROWS lowering requires integer v")
    if view_mode != 0:
        raise ValueError("SET_ROWS lowering requires contiguous source input (v=0)")
    if row_count <= 0:
        raise ValueError("SET_ROWS lowering requires r >= 1")
    dst_extents = [int(value) for value in ne]
    src0_extents = [int(ne[0]), int(row_count), int(ne[2]), int(ne[3])]
    src1_extents = [int(row_count), int(nr23[0]), int(nr23[1]), 1]
    src1_strides = [1, int(row_count), int(row_count), int(row_count)]
    tensors = {
        "src0": ConcreteTensor(
            dtype=str(case.dtype.get("type", "")).upper(),
            dimensions=_dimensions_from_extents(src0_extents),
        ),
        "src1": ConcreteTensor(
            dtype=str(case.dtype.get("type_idx", "")).upper(),
            dimensions=_dimensions_from_extents_and_strides(src1_extents, src1_strides),
        ),
        "dst": ConcreteTensor(
            dtype=str(case.dtype.get("type", "")).upper(),
            dimensions=_dimensions_from_extents(dst_extents),
        ),
    }
    return tensors, _shape_from_extents(dst_extents)


def _rope_set_rows_route_properties(route: V2Route) -> tuple[int, int]:
    if route.root_symbol == "@hrx2_rope_normal_f32_freq_set_rows_f16":
        return 0, 128
    if route.root_symbol == "@hrx2_rope_neox_f32_freq_set_rows_f16":
        return 2, 96
    raise ValueError(f"ROPE_SET_ROWS v2 routing is not implemented for {route.root_symbol}")


def lower_rope_set_rows_tensors(
    case: ImportedCase,
    route: V2Route,
) -> tuple[dict[str, ConcreteTensor], dict[str, int]]:
    ne = case.normalized_params.get("ne_a")
    mode = case.normalized_params.get("mode")
    if not isinstance(ne, list) or len(ne) != 4:
        raise ValueError("ROPE_SET_ROWS lowering requires a 4-D ne_a extent list")
    if any(not isinstance(value, int) for value in ne):
        raise ValueError("ROPE_SET_ROWS lowering requires integer ne_a extents")
    if not isinstance(mode, int):
        raise ValueError("ROPE_SET_ROWS lowering requires integer mode")
    expected_mode, n_dims = _rope_set_rows_route_properties(route)
    if mode != expected_mode:
        raise ValueError(f"ROPE_SET_ROWS v2 routing currently requires mode={expected_mode}")
    extents = [int(value) for value in ne]
    if extents[3] != 1:
        raise ValueError("ROPE_SET_ROWS v2 routing currently requires ne_a[3]=1")
    ncols = extents[0]
    nheads = extents[1]
    ntokens = extents[2]
    dst_rows = max(ntokens + 2, 4)
    src0_extents = [ncols, nheads, ntokens, 1]
    pos_extents = [ntokens, 1, 1, 1]
    freq_extents = [max(n_dims // 2, 1), 1, 1, 1]
    idx_extents = [ntokens, 1, 1, 1]
    dst_extents = [ncols * nheads, dst_rows, 1, 1]
    tensors = {
        "src0": ConcreteTensor(
            dtype="F32",
            dimensions=_dimensions_from_extents(src0_extents),
        ),
        "pos": ConcreteTensor(
            dtype="I32",
            dimensions=_dimensions_from_extents(pos_extents),
        ),
        "freq": ConcreteTensor(
            dtype="F32",
            dimensions=_dimensions_from_extents(freq_extents),
        ),
        "src1": ConcreteTensor(
            dtype=str(case.dtype.get("type_idx", "")).upper(),
            dimensions=_dimensions_from_extents(idx_extents),
        ),
        "dst": ConcreteTensor(
            dtype="F16",
            dimensions=_dimensions_from_extents(dst_extents),
        ),
    }
    return tensors, {
        "rope.ncols": ncols,
        "rope.n_dims": n_dims,
        "rope.nheads": nheads,
        "rope.ntokens": ntokens,
        "rope.src0_head_stride": ncols,
        "rope.src0_token_stride": ncols * nheads,
        "rope.pos_token_stride": 1,
        "set_rows.ne1": dst_rows,
        "set_rows.ne11": 1,
        "set_rows.ne12": 1,
    }


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


def lower_non_contiguous_copy_tensors(
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
    effective_src_permutation = chain_permutations(
        normalized_src,
        inverse_permutation(destination_permutation),
    )
    if effective_src_permutation is None:
        raise ValueError("copy lowering could not chain source and destination permutations")
    src0_strides = permuted_contiguous_strides(tuple(canonical_extents), effective_src_permutation)
    if src0_strides is None:
        raise ValueError("copy lowering could not derive permuted contiguous strides")
    tensors = {
        "src0": ConcreteTensor(
            dtype=src_dtype,
            dimensions=_dimensions_from_extents_and_strides(canonical_extents, list(src0_strides)),
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
        value.name == "total_size" and value.operation_kind == "product" and value.sources == (dst_dimensions,)
        for value in route.values
    )
    has_contiguous_strides = any(
        value.name == "contiguous_strides"
        and value.operation_kind == "contiguous_strides"
        and value.sources == (dst_dimensions,)
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
        value.name == "total_size" and value.operation_kind == "product" and value.sources == (dst_dimensions,)
        for value in route.values
    )
    has_contiguous_strides = any(
        value.name == "contiguous_strides"
        and value.operation_kind == "contiguous_strides"
        and value.sources == (dst_dimensions,)
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
    if route.op in {"ABS", "EXP", "NEG", "RELU"}:
        return lower_contiguous_unary_tensors(case)
    if route.op == "ARGSORT":
        return lower_argsort_tensors(case)
    if route.op in {"CLAMP", "SCALE", "SQR", "SQRT"}:
        return lower_contiguous_scale_or_clamp_tensors(case)
    if route.op == "CONT":
        return lower_contiguous_cont_tensors(case)
    if route.op == "GET_ROWS":
        return lower_get_rows_tensors(case)
    if route.op == "ROPE":
        return lower_rope_tensors(case, route)
    if route.op == "RMS_NORM":
        return lower_rms_norm_tensors(case)
    if route.op == "SOFT_MAX":
        return lower_soft_max_tensors(case)
    if route.op == "SWIGLU":
        return lower_swiglu_tensors(case)
    if route.op == "SUM_ROWS":
        return lower_sum_rows_tensors(case)
    if route.op == "SET_ROWS":
        return lower_set_rows_tensors(case)
    if route.op == "ROPE_SET_ROWS":
        return lower_rope_set_rows_tensors(case, route)
    if route.lowering_kind == LOWERING_KIND_COPY_CONTIGUOUS:
        return lower_contiguous_copy_tensors(case)
    if route.lowering_kind == LOWERING_KIND_COPY_NON_CONTIGUOUS_4D:
        return lower_non_contiguous_copy_tensors(case)
    if _route_uses_generic_pointwise_lowering(route):
        return lower_generic_pointwise_tensors(case)
    if _route_uses_non_contiguous_copy_lowering(route):
        return lower_non_contiguous_copy_tensors(case)
    if _route_uses_contiguous_copy_lowering(route):
        return lower_contiguous_copy_tensors(case)
    return lower_contiguous_pointwise_tensors(case)


def shape_for_resolved_route(
    route: V2Route,
    tensors: dict[str, ConcreteTensor],
    fallback_shape: dict[str, int],
) -> EncodedRouteShape:
    if not route.tensors:
        return EncodedRouteShape(items=tuple((str(name), int(value)) for name, value in fallback_shape.items()))
    encoded = list(encode_route_shape(route, tensors).items)
    present = {name for name, _ in encoded}
    for name, value in fallback_shape.items():
        if str(name) in present:
            continue
        encoded.append((str(name), int(value)))
    return EncodedRouteShape(items=tuple(encoded))


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
        encoded_shape = shape_for_resolved_route(route, tensors, shape)
        return route, encoded_shape.as_dict(), None, None
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
                    values=[int(value) for value in shape.values()],
                )
            )
    return suite
