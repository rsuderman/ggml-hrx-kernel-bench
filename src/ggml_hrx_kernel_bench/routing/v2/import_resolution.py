from __future__ import annotations

import re
from dataclasses import dataclass

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
    contiguous_strides,
    encode_route_shape,
    inverse_permutation,
    permuted_contiguous_strides,
)
from .matching import route_accepts_dtype
from .models import (
    ConcreteTensor,
    ConcreteTensorDimension,
    V2Route,
)
from .query import RouteCatalog, require_route_catalog, routes_for_op
from .selection import RouteMatchQuery, RouteSelector

POINTWISE_BASE_PERMUTATION = (0, 1, 2, 3)
POINTWISE_SRC1_PERMUTATION = (1, 2, 0, 3)
COPY_BASE_PERMUTATION = (0, 1, 2, 3)
COPY_IDENTITY_PERMUTATION = (0, 0, 0, 0)
COPY_TRANSPOSE_01_PERMUTATION = (1, 0, 2, 3)
_TENSOR_SOURCE_RE = re.compile(
    r"^\s*(?P<dtype>[A-Za-z0-9_]+)\[(?P<extents>[0-9,\s]+)\](?:nb\[(?P<byte_strides>[0-9,\s]+)\])?\s*$"
)
_SCALAR_DTYPE_BYTES = {
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "F64": 8,
    "I8": 1,
    "I16": 2,
    "I32": 4,
    "I64": 8,
    "U8": 1,
    "U16": 2,
    "U32": 4,
    "U64": 8,
}
_IMPORT_QUERY_ONLY_OPS = {
    "ADD_RMS_NORM",
    "ADD",
    "ARGSORT",
    "CLAMP",
    "CONT",
    "CPY",
    "DIV",
    "FLASH_ATTN_EXT",
    "GET_ROWS",
    "MUL_MAT",
    "MUL_MAT_ID",
    "MUL",
    "QUANTIZE",
    "RMS_NORM",
    "ROPE",
    "ROPE_SCALE",
    "ROPE_SET_ROWS",
    "SCALE",
    "SET_ROWS",
    "SOFT_MAX",
    "SQR",
    "SQRT",
    "SUB",
    "SWIGLU",
    "SUM_ROWS",
}


@dataclass(frozen=True)
class TensorSourceDescriptor:
    dtype: str
    extents: tuple[int, ...]
    byte_strides: tuple[int, ...] | None = None


@dataclass(frozen=True)
class ImportedRouteQuery:
    tensors: dict[str, ConcreteTensor]
    fallback_shape: dict[str, int]
    route_ids: tuple[str, ...] | None = None


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


def _parse_int_tuple(raw: str) -> tuple[int, ...]:
    values: list[int] = []
    for item in raw.split(","):
        text = item.strip()
        if not text:
            raise ValueError("empty tensor descriptor integer field")
        values.append(int(text))
    return tuple(values)


def _split_tensor_sources(raw: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(raw):
        if char == "[":
            depth += 1
            continue
        if char == "]":
            depth -= 1
            if depth < 0:
                raise ValueError("unbalanced tensor source descriptor")
            continue
        if char == "," and depth == 0:
            parts.append(raw[start:index].strip())
            start = index + 1
    if depth != 0:
        raise ValueError("unbalanced tensor source descriptor")
    tail = raw[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _parse_tensor_sources(raw: object) -> list[TensorSourceDescriptor]:
    if not isinstance(raw, str) or not raw.strip():
        return []
    descriptors: list[TensorSourceDescriptor] = []
    for part in _split_tensor_sources(raw):
        match = _TENSOR_SOURCE_RE.match(part)
        if match is None:
            return []
        byte_strides = match.group("byte_strides")
        descriptors.append(
            TensorSourceDescriptor(
                dtype=match.group("dtype"),
                extents=_parse_int_tuple(match.group("extents")),
                byte_strides=_parse_int_tuple(byte_strides) if byte_strides else None,
            )
        )
    return descriptors


def _flatten_4d_to_2d_extents(extents: list[int] | tuple[int, ...]) -> tuple[int, int]:
    if len(extents) != 4:
        raise ValueError("2-D projection requires 4-D extents")
    return int(extents[0]), int(extents[1]) * int(extents[2]) * int(extents[3])


def _element_strides_for_source(source: TensorSourceDescriptor) -> tuple[int, ...]:
    if source.byte_strides is None:
        return contiguous_strides(tuple(int(extent) for extent in source.extents))
    dtype = str(source.dtype).upper()
    dtype_bytes = _SCALAR_DTYPE_BYTES.get(dtype)
    if dtype_bytes is None:
        raise ValueError(f"source descriptor uses byte strides for non-scalar dtype {dtype!r}")
    if len(source.byte_strides) != len(source.extents):
        raise ValueError("source descriptor byte stride rank must match extent rank")
    element_strides: list[int] = []
    for byte_stride in source.byte_strides:
        if int(byte_stride) % dtype_bytes != 0:
            raise ValueError("source descriptor byte strides must be divisible by dtype size")
        element_strides.append(int(byte_stride) // dtype_bytes)
    return tuple(element_strides)


def _dtype_from_case(case: ImportedCase, *keys: str, default: str = "") -> str:
    for key in keys:
        value = case.dtype.get(key)
        if value is not None:
            return str(value).upper()
    return default.upper()


def _tensor_from_extents(
    *,
    dtype: str,
    extents: tuple[int, ...],
    strides: tuple[int, ...] | None = None,
) -> ConcreteTensor:
    sizes = [int(extent) for extent in extents]
    tensor_strides = list(strides or contiguous_strides(tuple(sizes)))
    return ConcreteTensor(
        dtype=str(dtype).upper(),
        dimensions=_dimensions_from_extents_and_strides(sizes, tensor_strides),
    )


def _source_tensor(source: TensorSourceDescriptor) -> ConcreteTensor:
    return _tensor_from_extents(
        dtype=source.dtype,
        extents=tuple(int(extent) for extent in source.extents),
        strides=_element_strides_for_source(source),
    )


def _source_tensors_from_descriptors(sources: list[TensorSourceDescriptor]) -> dict[str, ConcreteTensor]:
    return {
        f"src{index}": _source_tensor(source)
        for index, source in enumerate(sources)
    }


def _dst_extents_from_case(case: ImportedCase) -> tuple[int, ...]:
    dst_extents = case.normalized_params.get("ne")
    if not isinstance(dst_extents, list) or any(not isinstance(value, int) for value in dst_extents):
        raise ValueError("source-described routing requires integer ne extent list")
    return tuple(int(value) for value in dst_extents)


def _literal_source_import_query(case: ImportedCase) -> ImportedRouteQuery | None:
    sources = _parse_tensor_sources(case.normalized_params.get("sources"))
    if not sources:
        return None
    dst_extents = _dst_extents_from_case(case)
    dst_dtype = _dtype_from_case(case, "type_dst", "type", default=sources[0].dtype)
    tensors = _source_tensors_from_descriptors(sources)
    tensors["dst"] = _tensor_from_extents(dtype=dst_dtype, extents=dst_extents)
    return ImportedRouteQuery(tensors=tensors, fallback_shape=_shape_from_extents(list(dst_extents)))


def _flattened_2d_broadcast_strides(
    source_extents: tuple[int, ...],
    dst_extents: tuple[int, int],
) -> tuple[int, int]:
    source_d0, source_d1 = _flatten_4d_to_2d_extents(source_extents)
    dst_d0, dst_d1 = dst_extents
    if source_d0 not in {1, dst_d0} and dst_d0 % source_d0 != 0:
        raise ValueError("source-described 2-D projection requires source d0 to divide dst d0")
    if source_d1 not in {1, dst_d1} and dst_d1 % source_d1 != 0:
        raise ValueError("source-described 2-D projection requires source d1 to divide dst d1")
    return (
        0 if source_d0 == 1 and dst_d0 != 1 else 1,
        0 if source_d1 == 1 and dst_d1 != 1 else source_d0,
    )


def _flattened_2d_source_tensor(
    source: TensorSourceDescriptor,
    dst_extents: tuple[int, int],
) -> ConcreteTensor:
    if source.byte_strides is not None:
        raise ValueError("source-described 2-D projection does not support explicit byte strides")
    flattened_extents = _flatten_4d_to_2d_extents(source.extents)
    return _tensor_from_extents(
        dtype=source.dtype,
        extents=flattened_extents,
        strides=_flattened_2d_broadcast_strides(source.extents, dst_extents),
    )


def _flattened_2d_source_import_query(case: ImportedCase) -> ImportedRouteQuery | None:
    sources = _parse_tensor_sources(case.normalized_params.get("sources"))
    if not sources:
        return None
    dst_extents = _dst_extents_from_case(case)
    if len(dst_extents) != 4 or any(len(source.extents) != 4 for source in sources):
        return None
    flattened_dst_extents = _flatten_4d_to_2d_extents(dst_extents)
    dst_dtype = _dtype_from_case(case, "type_dst", "type", default=sources[0].dtype)
    tensors = {
        f"src{index}": _flattened_2d_source_tensor(source, flattened_dst_extents)
        for index, source in enumerate(sources)
    }
    tensors["dst"] = _tensor_from_extents(dtype=dst_dtype, extents=flattened_dst_extents)
    return ImportedRouteQuery(
        tensors=tensors,
        fallback_shape=_shape_from_extents(list(flattened_dst_extents)),
    )


def source_described_import_queries(case: ImportedCase) -> tuple[ImportedRouteQuery, ...]:
    queries: list[ImportedRouteQuery] = []
    for builder in (_flattened_2d_source_import_query, _literal_source_import_query):
        query = builder(case)
        if query is not None:
            queries.append(query)
    return tuple(queries)


def yaml_import_queries(case: ImportedCase) -> tuple[ImportedRouteQuery, ...]:
    queries: list[ImportedRouteQuery] = []
    get_rows_query = _llamacpp_get_rows_import_query(case)
    if get_rows_query is not None:
        queries.append(get_rows_query)
    mul_mat_query = _llamacpp_mul_mat_import_query(case)
    if mul_mat_query is not None:
        queries.append(mul_mat_query)
    mul_mat_id_query = _llamacpp_mul_mat_id_import_query(case)
    if mul_mat_id_query is not None:
        queries.append(mul_mat_id_query)
    swiglu_query = _llamacpp_swiglu_import_query(case)
    if swiglu_query is not None:
        queries.append(swiglu_query)
    contiguous_ne_query = _llamacpp_contiguous_ne_import_query(case)
    if contiguous_ne_query is not None:
        queries.append(contiguous_ne_query)
    cont_query = _llamacpp_cont_import_query(case)
    if cont_query is not None:
        queries.append(cont_query)
    copy_query = _llamacpp_copy_import_query(case)
    if copy_query is not None:
        queries.append(copy_query)
    argsort_query = _llamacpp_argsort_import_query(case)
    if argsort_query is not None:
        queries.append(argsort_query)
    rms_norm_query = _llamacpp_rms_norm_import_query(case)
    if rms_norm_query is not None:
        queries.append(rms_norm_query)
    sum_rows_query = _llamacpp_sum_rows_import_query(case)
    if sum_rows_query is not None:
        queries.append(sum_rows_query)
    quantize_query = _llamacpp_quantize_import_query(case)
    if quantize_query is not None:
        queries.append(quantize_query)
    add_rms_norm_query = _llamacpp_add_rms_norm_import_query(case)
    if add_rms_norm_query is not None:
        queries.append(add_rms_norm_query)
    rms_norm_mul_query = _llamacpp_rms_norm_mul_import_query(case)
    if rms_norm_mul_query is not None:
        queries.append(rms_norm_mul_query)
    pointwise_query = _llamacpp_pointwise_import_query(case)
    if pointwise_query is not None:
        queries.append(pointwise_query)
    set_rows_query = _llamacpp_set_rows_import_query(case)
    if set_rows_query is not None:
        queries.append(set_rows_query)
    rope_set_rows_query = _llamacpp_rope_set_rows_import_query(case)
    if rope_set_rows_query is not None:
        queries.append(rope_set_rows_query)
    rope_query = _llamacpp_rope_import_query(case)
    if rope_query is not None:
        queries.append(rope_query)
    flash_attn_ext_query = _llamacpp_flash_attn_ext_import_query(case)
    if flash_attn_ext_query is not None:
        queries.append(flash_attn_ext_query)
    unary_query = _llamacpp_unary_import_query(case)
    if unary_query is not None:
        queries.append(unary_query)
    soft_max_query = _llamacpp_soft_max_import_query(case)
    if soft_max_query is not None:
        queries.append(soft_max_query)
    try:
        queries.extend(source_described_import_queries(case))
    except ValueError:
        if not queries:
            raise
    return tuple(queries)


def _llamacpp_get_rows_import_query(case: ImportedCase) -> ImportedRouteQuery | None:
    if case.op != "GET_ROWS":
        return None
    sources = _parse_tensor_sources(case.normalized_params.get("sources"))
    if sources:
        return _llamacpp_get_rows_source_import_query(case, sources)
    if case.normalized_params.get("moe_weights") == 1:
        return _llamacpp_get_rows_moe_import_query(case)
    if "m" in case.normalized_params or "n" in case.normalized_params or "r" in case.normalized_params:
        return _llamacpp_get_rows_embedding_import_query(case)
    return None


def _llamacpp_get_rows_source_import_query(
    case: ImportedCase,
    sources: list[TensorSourceDescriptor],
) -> ImportedRouteQuery | None:
    if len(sources) != 2:
        return None
    dst_extents = _dst_extents_from_case(case)
    if len(dst_extents) != 4 or len(sources[0].extents) != 4 or len(sources[1].extents) != 4:
        return None
    ncols, nrows = _flatten_4d_to_2d_extents(dst_extents)
    src0_ncols, src0_nrows = _flatten_4d_to_2d_extents(sources[0].extents)
    if src0_ncols != ncols:
        raise ValueError("GET_ROWS YAML import requires source and destination column counts to match")
    src1_rows = int(sources[1].extents[0])
    if src1_rows != nrows:
        raise ValueError("GET_ROWS YAML import requires index count to match destination rows")
    src0_dtype = str(sources[0].dtype).upper()
    src1_dtype = str(sources[1].dtype).upper()
    dst_dtype = _dtype_from_case(case, "type_dst", "type", default="F32")
    tensors = {
        "src0": _tensor_from_extents(dtype=src0_dtype, extents=(ncols, src0_nrows)),
        "src1": _tensor_from_extents(dtype=src1_dtype, extents=(1, nrows)),
        "dst": _tensor_from_extents(dtype=dst_dtype, extents=(ncols, nrows)),
    }
    return ImportedRouteQuery(
        tensors=tensors,
        fallback_shape={
            **_shape_from_extents([ncols, nrows]),
            "get_rows.src0_nrows": int(src0_nrows),
            "get_rows.idx_row_stride": 1,
        },
    )


def _llamacpp_get_rows_embedding_import_query(case: ImportedCase) -> ImportedRouteQuery:
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
    ncols_value = int(ncols)
    nrows_value = int(nrows)
    src0_row_count = max(int(src0_nrows), nrows_value, 1)
    src0_dtype = str(case.dtype.get("type", "")).upper()
    dst_dtype = "F32" if src0_dtype in {"Q4_K", "Q5_K", "Q6_K", "Q8_0"} else src0_dtype
    tensors = {
        "src0": _tensor_from_extents(dtype=src0_dtype, extents=(ncols_value, src0_row_count)),
        "src1": _tensor_from_extents(dtype="I32", extents=(1, nrows_value)),
        "dst": _tensor_from_extents(dtype=dst_dtype, extents=(ncols_value, nrows_value)),
    }
    return ImportedRouteQuery(
        tensors=tensors,
        fallback_shape={
            **_shape_from_extents([ncols_value, nrows_value]),
            "get_rows.src0_nrows": src0_row_count,
            "get_rows.idx_row_stride": 1,
        },
    )


def _llamacpp_get_rows_moe_import_query(case: ImportedCase) -> ImportedRouteQuery:
    nexperts = case.normalized_params.get("nexperts")
    nselected = case.normalized_params.get("nselected")
    ntokens = case.normalized_params.get("ntokens")
    src0_token_stride = case.normalized_params.get("src0_token_stride", nexperts)
    idx_token_stride = case.normalized_params.get("idx_token_stride", nselected)
    dst_token_stride = case.normalized_params.get("dst_token_stride", nselected)
    for name, value in (
        ("nexperts", nexperts),
        ("nselected", nselected),
        ("ntokens", ntokens),
        ("src0_token_stride", src0_token_stride),
        ("idx_token_stride", idx_token_stride),
        ("dst_token_stride", dst_token_stride),
    ):
        if not isinstance(value, int):
            raise ValueError(f"GET_ROWS MoE lowering requires integer {name}")
    if str(case.dtype.get("type", "")).upper() != "F32":
        raise ValueError("GET_ROWS MoE lowering requires f32 weights")
    if src0_token_stride < nexperts:
        raise ValueError("GET_ROWS MoE lowering requires src0_token_stride >= nexperts")
    if idx_token_stride < nselected:
        raise ValueError("GET_ROWS MoE lowering requires idx_token_stride >= nselected")
    if dst_token_stride < nselected:
        raise ValueError("GET_ROWS MoE lowering requires dst_token_stride >= nselected")
    tensors = {
        "src0": _tensor_from_extents(
            dtype="F32",
            extents=(int(nexperts), int(ntokens)),
            strides=(1, int(src0_token_stride)),
        ),
        "src1": _tensor_from_extents(
            dtype="I32",
            extents=(int(nselected), int(ntokens)),
            strides=(1, int(idx_token_stride)),
        ),
        "dst": _tensor_from_extents(
            dtype="F32",
            extents=(int(nselected), int(ntokens)),
            strides=(1, int(dst_token_stride)),
        ),
    }
    return ImportedRouteQuery(
        tensors=tensors,
        fallback_shape={
            **_shape_from_extents([int(nselected), int(ntokens)]),
            "get_rows_moe.nexperts": int(nexperts),
            "get_rows_moe.nselected": int(nselected),
            "get_rows_moe.ntokens": int(ntokens),
            "get_rows_moe.src0_token_stride": int(src0_token_stride),
            "get_rows_moe.idx_token_stride": int(idx_token_stride),
            "get_rows_moe.dst_token_stride": int(dst_token_stride),
        },
    )


def _llamacpp_mul_mat_import_query(case: ImportedCase) -> ImportedRouteQuery | None:
    if case.op != "MUL_MAT":
        return None
    sources = _parse_tensor_sources(case.normalized_params.get("sources"))
    if sources:
        if len(sources) != 2:
            return None
        return _llamacpp_mul_mat_source_import_query(case, sources)
    if "bs" in case.normalized_params:
        return _llamacpp_mul_mat_shape_import_query(case)
    return None


def _llamacpp_mul_mat_source_import_query(
    case: ImportedCase,
    sources: list[TensorSourceDescriptor],
) -> ImportedRouteQuery | None:
    dst_extents = _dst_extents_from_case(case)
    if len(dst_extents) != 4 or len(sources[0].extents) != 4 or len(sources[1].extents) != 4:
        return None
    k_extent, row_extent = _flatten_4d_to_2d_extents(sources[0].extents)
    rhs_k_extent, col_extent = _flatten_4d_to_2d_extents(sources[1].extents)
    dst_rows, dst_cols = _flatten_4d_to_2d_extents(dst_extents)
    if rhs_k_extent != k_extent:
        raise ValueError("MUL_MAT YAML import requires lhs and rhs k dimensions to match")
    if dst_rows != row_extent:
        raise ValueError("MUL_MAT YAML import requires destination rows to match lhs rows")
    if dst_cols != col_extent:
        raise ValueError("MUL_MAT YAML import requires destination columns to match rhs columns")
    src0_dtype = str(sources[0].dtype).upper()
    src1_dtype = str(sources[1].dtype).upper()
    dst_dtype = _dtype_from_case(case, "type_dst", "type", default="F32")
    tensors = {
        "src0": _tensor_from_extents(dtype=src0_dtype, extents=(k_extent, row_extent)),
        "src1": _tensor_from_extents(dtype=src1_dtype, extents=(k_extent, col_extent)),
        "dst": _tensor_from_extents(dtype=dst_dtype, extents=(row_extent, col_extent)),
    }
    return ImportedRouteQuery(
        tensors=tensors,
        fallback_shape={
            **_shape_from_extents([row_extent, col_extent]),
            "src0_d0": int(k_extent),
            "src0_d1": int(row_extent),
            "src1_d0": int(k_extent),
            "k": int(k_extent),
            "rows": int(row_extent),
            "cols": int(col_extent),
        },
    )


def _llamacpp_mul_mat_shape_import_query(case: ImportedCase) -> ImportedRouteQuery:
    bs = case.normalized_params.get("bs")
    k = case.normalized_params.get("k")
    rows = case.normalized_params.get("m")
    cols = case.normalized_params.get("n")
    nr = case.normalized_params.get("nr")
    output_count = case.normalized_params.get("o", 1)
    permutation = case.normalized_params.get("per", list(POINTWISE_BASE_PERMUTATION))
    if not isinstance(bs, list) or len(bs) != 2:
        raise ValueError("MUL_MAT lowering requires a 2-D bs extent list")
    if not isinstance(nr, list) or len(nr) != 2:
        raise ValueError("MUL_MAT lowering requires a 2-D nr extent list")
    if not all(isinstance(value, int) for value in (*bs, *nr)):
        raise ValueError("MUL_MAT lowering requires integer bs and nr extents")
    for name, value in (("k", k), ("m", rows), ("n", cols), ("o", output_count)):
        if not isinstance(value, int):
            raise ValueError(f"MUL_MAT lowering requires integer {name}")
    if not isinstance(permutation, list) or len(permutation) != 4:
        raise ValueError("MUL_MAT lowering requires a 4-D per permutation list")
    if any(not isinstance(value, int) for value in permutation):
        raise ValueError("MUL_MAT lowering requires integer per values")
    batch_extents = [int(value) for value in bs]
    if [int(value) for value in nr] != [1, 1]:
        raise ValueError("MUL_MAT v2 routing currently requires nr=[1, 1]")
    if int(output_count) != 1:
        raise ValueError("MUL_MAT v2 routing currently requires o=1")
    if [int(value) for value in permutation] != list(POINTWISE_BASE_PERMUTATION):
        raise ValueError("MUL_MAT v2 routing currently requires per=[0, 1, 2, 3]")
    k_extent = int(k)
    row_extent = int(rows)
    col_extent = int(cols)
    if batch_extents != [1, 1]:
        if (
            str(case.dtype.get("type_a", "")).lower() != "f16"
            or str(case.dtype.get("type_b", "")).lower() != "f32"
        ):
            raise ValueError("MUL_MAT v2 routing currently requires bs=[1, 1]")
        src1_extents = (k_extent, col_extent, *batch_extents)
        dst_extents = (row_extent, col_extent, *batch_extents)
        src1_stride_ne2 = k_extent * col_extent
        src1_stride_ne3 = src1_stride_ne2 * batch_extents[0]
        dst_stride_ne2 = row_extent * col_extent
        dst_stride_ne3 = dst_stride_ne2 * batch_extents[0]
        tensors = {
            "src0": _tensor_from_extents(dtype="F16", extents=(k_extent, row_extent, 1, 1)),
            "src1": _tensor_from_extents(dtype="F32", extents=src1_extents),
            "dst": _tensor_from_extents(dtype="F32", extents=dst_extents),
        }
        return ImportedRouteQuery(
            tensors=tensors,
            fallback_shape={
                **_shape_from_extents(list(dst_extents)),
                "src0_d0": k_extent,
                "src0_d1": row_extent,
                "src1_d0": k_extent,
                "src1_d2_stride": src1_stride_ne2,
                "src1_d3_stride": src1_stride_ne3,
                "dst_d2_stride": dst_stride_ne2,
                "dst_d3_stride": dst_stride_ne3,
                "k": k_extent,
                "rows": row_extent,
                "cols": col_extent,
            },
        )
    type_a = str(case.dtype.get("type_a", "")).upper()
    type_b = str(case.dtype.get("type_b", "")).upper()
    tensors = {
        "src0": _tensor_from_extents(dtype=type_a, extents=(k_extent, row_extent)),
        "src1": _tensor_from_extents(dtype=type_b, extents=(k_extent, col_extent)),
        "dst": _tensor_from_extents(dtype="F32", extents=(row_extent, col_extent)),
    }
    return ImportedRouteQuery(
        tensors=tensors,
        fallback_shape={
            "k": k_extent,
            "rows": row_extent,
            "cols": col_extent,
        },
    )


def _llamacpp_mul_mat_id_import_query(case: ImportedCase) -> ImportedRouteQuery | None:
    if case.op != "MUL_MAT_ID":
        return None
    type_a = str(case.dtype.get("type_a", "")).upper()
    type_b = str(case.dtype.get("type_b", "")).upper()
    if type_a not in {"Q4_K", "Q5_K", "Q6_K"} or type_b != "F32":
        raise ValueError("MUL_MAT_ID q4_k/q5_k/q6_k routing requires type_a=q4_K, q5_K, or q6_K and type_b=f32")
    batch = case.normalized_params.get("b", 0)
    k = case.normalized_params.get("k")
    rows = case.normalized_params.get("m")
    ntokens = case.normalized_params.get("n")
    nexperts = case.normalized_params.get("n_mats")
    nselected = case.normalized_params.get("n_used")
    for name, value in (
        ("b", batch),
        ("k", k),
        ("m", rows),
        ("n", ntokens),
        ("n_mats", nexperts),
        ("n_used", nselected),
    ):
        if not isinstance(value, int):
            raise ValueError(f"MUL_MAT_ID q4_k/q5_k/q6_k lowering requires integer {name}")
    if int(batch) != 0:
        raise ValueError("MUL_MAT_ID q4_k/q5_k/q6_k v2 routing currently requires b=0")
    k_extent = int(k)
    row_extent = int(rows)
    token_extent = int(ntokens)
    expert_extent = int(nexperts)
    selected_extent = int(nselected)
    if token_extent != 1:
        raise ValueError("MUL_MAT_ID q4_k/q5_k/q6_k v2 routing currently requires n=1")
    if selected_extent > 2:
        raise ValueError("MUL_MAT_ID q4_k/q5_k/q6_k v2 routing currently requires n_used<=2")
    src1_selected_stride = k_extent
    src1_token_stride = k_extent * selected_extent
    idx_token_stride = selected_extent
    dst_token_stride = row_extent * selected_extent
    tensors = {
        "src0": _tensor_from_extents(dtype=type_a, extents=(k_extent, row_extent, expert_extent)),
        "src1": _tensor_from_extents(
            dtype="F32",
            extents=(k_extent, selected_extent, token_extent),
            strides=(1, src1_selected_stride, src1_token_stride),
        ),
        "src2": _tensor_from_extents(
            dtype="I32",
            extents=(selected_extent, token_extent, 1),
            strides=(1, idx_token_stride, idx_token_stride * token_extent),
        ),
        "dst": _tensor_from_extents(
            dtype="F32",
            extents=(row_extent, selected_extent, token_extent),
            strides=(1, row_extent, dst_token_stride),
        ),
    }
    return ImportedRouteQuery(
        tensors=tensors,
        fallback_shape={
            "k": k_extent,
            "rows": row_extent,
            "nexperts": expert_extent,
            "nselected": selected_extent,
            "ntokens": token_extent,
            "src1_selected_stride": src1_selected_stride,
            "src1_token_stride": src1_token_stride,
            "idx_token_stride": idx_token_stride,
            "dst_token_stride": dst_token_stride,
        },
    )


def _llamacpp_swiglu_import_query(case: ImportedCase) -> ImportedRouteQuery | None:
    if case.op != "SWIGLU":
        return None
    sources = _parse_tensor_sources(case.normalized_params.get("sources"))
    if sources:
        return _llamacpp_swiglu_source_import_query(case, sources)
    if "ne_a" not in case.normalized_params:
        return None
    return _llamacpp_swiglu_ne_a_import_query(case)


def _llamacpp_swiglu_source_import_query(
    case: ImportedCase,
    sources: list[TensorSourceDescriptor],
) -> ImportedRouteQuery | None:
    if len(sources) != 2:
        return None
    dst_extents = _dst_extents_from_case(case)
    if len(dst_extents) != 4 or any(len(source.extents) != 4 for source in sources):
        return None
    if any(source.byte_strides is not None for source in sources):
        raise ValueError("SWIGLU YAML import currently requires contiguous split sources")
    op_params = case.normalized_params.get("op_params")
    if op_params != ["0:2"]:
        raise ValueError("SWIGLU YAML import currently requires op_params=['0:2']")
    if sources[0].extents != sources[1].extents:
        raise ValueError("SWIGLU YAML import requires matching split source extents")
    if sources[0].extents != dst_extents:
        raise ValueError("SWIGLU YAML import requires source extents to match destination extents")
    src_dtype = str(sources[0].dtype).upper()
    src1_dtype = str(sources[1].dtype).upper()
    dst_dtype = _dtype_from_case(case, "type_dst", "type", default=src_dtype)
    if src_dtype != "F32" or src1_dtype != "F32" or dst_dtype != "F32":
        raise ValueError("SWIGLU YAML import currently requires F32 split sources and destination")
    return _packed_swiglu_import_query(dtype=dst_dtype, dst_extents=dst_extents)


def _llamacpp_swiglu_ne_a_import_query(case: ImportedCase) -> ImportedRouteQuery:
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
    return _packed_swiglu_import_query(
        dtype=str(case.dtype.get("type", "")).upper(),
        dst_extents=tuple(int(value) for value in ne),
    )


def _packed_swiglu_import_query(
    *,
    dtype: str,
    dst_extents: tuple[int, ...],
) -> ImportedRouteQuery:
    src0_extents = (int(dst_extents[0]) * 2, *dst_extents[1:])
    tensors = {
        "src0": _tensor_from_extents(dtype=dtype, extents=src0_extents),
        "dst": _tensor_from_extents(dtype=dtype, extents=dst_extents),
    }
    return ImportedRouteQuery(
        tensors=tensors,
        fallback_shape={
            **_shape_from_extents(list(dst_extents)),
            "src0_d0": int(src0_extents[0]),
        },
    )


def _llamacpp_contiguous_ne_import_query(case: ImportedCase) -> ImportedRouteQuery | None:
    if case.op not in {"CLAMP", "SCALE", "SQR", "SQRT"}:
        return None
    ne = case.normalized_params.get("ne")
    if not isinstance(ne, list) or len(ne) < 1:
        raise ValueError("contiguous YAML import requires a non-empty ne extent list")
    if any(not isinstance(value, int) for value in ne):
        raise ValueError("contiguous YAML import requires integer ne extents")
    extents = tuple(int(value) for value in ne)
    dtype = str(case.dtype.get("type", "")).upper()
    tensors = {
        "src0": _tensor_from_extents(dtype=dtype, extents=extents),
        "dst": _tensor_from_extents(dtype=dtype, extents=extents),
    }
    return ImportedRouteQuery(
        tensors=tensors,
        fallback_shape={
            **_shape_from_extents(list(extents)),
            **_contiguous_shape_metadata(list(extents), prefix="pointwise"),
        },
    )


def _llamacpp_cont_import_query(case: ImportedCase) -> ImportedRouteQuery | None:
    if case.op != "CONT":
        return None
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
    extents = tuple(int(value) for value in ne)
    dtype = str(case.dtype.get("type", "")).upper()
    tensors = {
        "src0": _tensor_from_extents(dtype=dtype, extents=extents),
        "dst": _tensor_from_extents(dtype=dtype, extents=extents),
    }
    return ImportedRouteQuery(
        tensors=tensors,
        fallback_shape={
            **_shape_from_extents(list(extents)),
            **_contiguous_shape_metadata(list(extents), prefix="cont"),
        },
    )


def _llamacpp_copy_import_query(case: ImportedCase) -> ImportedRouteQuery | None:
    if case.op != "CPY":
        return None
    ne, src_transpose, permute_src, permute_dst = _parse_copy_parameters(case)
    normalized_src = _normalize_copy_permutation(permute_src, name="permute_src")
    normalized_dst = _normalize_copy_permutation(permute_dst, name="permute_dst")
    src_dtype = str(case.dtype.get("type_src", "")).upper()
    dst_dtype = str(case.dtype.get("type_dst", "")).upper()
    if src_transpose == 0 and normalized_src == COPY_BASE_PERMUTATION and normalized_dst == COPY_BASE_PERMUTATION:
        tensors = {
            "src0": _tensor_from_extents(dtype=src_dtype, extents=tuple(ne)),
            "dst": _tensor_from_extents(dtype=dst_dtype, extents=tuple(ne)),
        }
        return ImportedRouteQuery(tensors=tensors, fallback_shape=_shape_from_extents(ne))
    if src_transpose not in {0, 1}:
        raise ValueError("copy lowering requires _src_transpose to be 0 or 1")
    if src_transpose == 1 and normalized_dst != COPY_BASE_PERMUTATION:
        raise ValueError("copy transpose lowering requires permute_dst=[0, 0, 0, 0]")
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
    return ImportedRouteQuery(tensors=tensors, fallback_shape=_shape_from_extents(canonical_extents))


def _llamacpp_argsort_import_query(case: ImportedCase) -> ImportedRouteQuery | None:
    if case.op != "ARGSORT":
        return None
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
    ncols, nrows = _flatten_4d_to_2d_extents(tuple(int(value) for value in ne))
    tensors = {
        "src0": _tensor_from_extents(dtype=str(case.dtype.get("type", "")).upper(), extents=(ncols, nrows)),
        "dst": _tensor_from_extents(dtype="I32", extents=(ncols, nrows)),
    }
    return ImportedRouteQuery(tensors=tensors, fallback_shape=_shape_from_extents([ncols, nrows]))


def _llamacpp_rms_norm_import_query(case: ImportedCase) -> ImportedRouteQuery | None:
    if case.op != "RMS_NORM":
        return None
    ne = case.normalized_params.get("ne")
    inplace = case.normalized_params.get("inplace", 0)
    view_mode = case.normalized_params.get("v", 0)
    if not isinstance(ne, list) or len(ne) != 4:
        raise ValueError("RMS_NORM lowering requires a 4-D ne extent list")
    if any(not isinstance(value, int) for value in ne):
        raise ValueError("RMS_NORM lowering requires integer ne extents")
    eps_value = _parse_numeric_eps(case, op_name="RMS_NORM")
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
    extents = tuple(int(value) for value in ne)
    dtype = str(case.dtype.get("type", "")).upper()
    tensors = {
        "src0": _tensor_from_extents(dtype=dtype, extents=extents),
        "dst": _tensor_from_extents(dtype=dtype, extents=extents),
    }
    return ImportedRouteQuery(tensors=tensors, fallback_shape=_shape_from_extents(list(extents)))


def _llamacpp_sum_rows_import_query(case: ImportedCase) -> ImportedRouteQuery | None:
    if case.op != "SUM_ROWS":
        return None
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
    extents = tuple(int(value) for value in ne)
    tensors = {
        "src0": _tensor_from_extents(dtype=str(case.dtype.get("type", "")).upper(), extents=extents),
    }
    return ImportedRouteQuery(tensors=tensors, fallback_shape=_shape_from_extents(list(extents)))


def _llamacpp_quantize_import_query(case: ImportedCase) -> ImportedRouteQuery | None:
    if case.op != "QUANTIZE":
        return None
    dst_dtype = str(case.dtype.get("type_dst", "")).upper()
    if dst_dtype == "Q8_1":
        return _llamacpp_quantize_q8_1_import_query(case)
    if dst_dtype == "Q8_1_X4":
        return _llamacpp_rms_norm_mul_quantize_import_query(case)
    return None


def _llamacpp_quantize_q8_1_import_query(case: ImportedCase) -> ImportedRouteQuery:
    ne = case.normalized_params.get("ne")
    if not isinstance(ne, list) or len(ne) != 4:
        raise ValueError("quantize_q8_1_f32 lowering requires a 4-D ne extent list")
    if any(not isinstance(value, int) for value in ne):
        raise ValueError("quantize_q8_1_f32 lowering requires integer ne extents")
    if str(case.dtype.get("type_src", case.dtype.get("type", ""))).upper() != "F32":
        raise ValueError("quantize_q8_1_f32 routing requires type_src=f32")
    if str(case.dtype.get("type_dst", "")).upper() != "Q8_1":
        raise ValueError("quantize_q8_1_f32 routing requires type_dst=q8_1")
    ncols = int(ne[0])
    nrows = int(ne[1]) * int(ne[2]) * int(ne[3])
    if ncols % 32 != 0:
        raise ValueError("quantize_q8_1_f32 routing currently requires ne[0] to be a multiple of 32")
    tensors = {
        "src0": _tensor_from_extents(dtype="F32", extents=(ncols, nrows)),
        "dst": _tensor_from_extents(dtype="Q8_1", extents=(ncols, nrows)),
    }
    return ImportedRouteQuery(
        tensors=tensors,
        fallback_shape={
            **_shape_from_extents([ncols, nrows]),
            "ncols": ncols,
            "nrows": nrows,
            "q8_1.blocks": (ncols + 31) // 32,
            "q8_1.ne1": int(ne[1]),
            "q8_1.z_count": int(ne[2]) * int(ne[3]),
        },
    )


def _llamacpp_rms_norm_mul_quantize_import_query(case: ImportedCase) -> ImportedRouteQuery:
    ne = case.normalized_params.get("ne")
    if not isinstance(ne, list) or len(ne) != 4:
        raise ValueError("rms_norm_mul_quantize_q8_1_f32 lowering requires a 4-D ne extent list")
    if any(not isinstance(value, int) for value in ne):
        raise ValueError("rms_norm_mul_quantize_q8_1_f32 lowering requires integer ne extents")
    if str(case.dtype.get("type_src", case.dtype.get("type", ""))).upper() != "F32":
        raise ValueError("rms_norm_mul_quantize_q8_1_f32 routing requires type_src=f32")
    if str(case.dtype.get("type_weight", "f32")).upper() != "F32":
        raise ValueError("rms_norm_mul_quantize_q8_1_f32 routing requires type_weight=f32")
    if str(case.dtype.get("type_dst", "")).upper() != "Q8_1_X4":
        raise ValueError("rms_norm_mul_quantize_q8_1_f32 routing requires type_dst=q8_1_x4")
    if _parse_numeric_eps(case, op_name="rms_norm_mul_quantize_q8_1_f32") != 0.0:
        raise ValueError("rms_norm_mul_quantize_q8_1_f32 routing currently requires eps=0.0")
    ncols = int(ne[0])
    nrows = int(ne[1]) * int(ne[2]) * int(ne[3])
    tensors = {
        "src0": _tensor_from_extents(dtype="F32", extents=(ncols, nrows)),
        "weight": _tensor_from_extents(dtype="F32", extents=(ncols, 1), strides=(1, 0)),
        "dst": _tensor_from_extents(dtype="Q8_1_X4", extents=(ncols, nrows)),
    }
    return ImportedRouteQuery(tensors=tensors, fallback_shape={"ncols": ncols, "nrows": nrows})


def _llamacpp_add_rms_norm_import_query(case: ImportedCase) -> ImportedRouteQuery | None:
    if case.op != "ADD_RMS_NORM":
        return None
    ne = case.normalized_params.get("ne")
    broadcast = case.normalized_params.get("broadcast", 0)
    if not isinstance(ne, list) or len(ne) != 4:
        raise ValueError("ADD_RMS_NORM lowering requires a 4-D ne extent list")
    if any(not isinstance(value, int) for value in ne):
        raise ValueError("ADD_RMS_NORM lowering requires integer ne extents")
    eps_value = _parse_numeric_eps(case, op_name="ADD_RMS_NORM")
    if not isinstance(broadcast, int):
        raise ValueError("ADD_RMS_NORM lowering requires integer broadcast")
    if str(case.dtype.get("type", "")).upper() != "F32":
        raise ValueError("add_rms_norm_mul_f32 routing requires type=f32")
    if int(broadcast) != 0:
        raise ValueError("add_rms_norm_mul_f32 routing currently requires broadcast=0")
    if eps_value != 0.0:
        raise ValueError("add_rms_norm_mul_f32 routing currently requires eps=0.0")
    ncols = int(ne[0])
    nrows = int(ne[1]) * int(ne[2]) * int(ne[3])
    row_extents = (ncols, nrows)
    tensors = {
        "src0": _tensor_from_extents(dtype="F32", extents=row_extents),
        "src1": _tensor_from_extents(dtype="F32", extents=row_extents),
        "add_dst": _tensor_from_extents(dtype="F32", extents=row_extents),
        "weight": _tensor_from_extents(dtype="F32", extents=(ncols, 1), strides=(1, 0)),
        "dst": _tensor_from_extents(dtype="F32", extents=row_extents),
    }
    return ImportedRouteQuery(tensors=tensors, fallback_shape={"ncols": ncols, "nrows": nrows})


def _llamacpp_rms_norm_mul_import_query(case: ImportedCase) -> ImportedRouteQuery | None:
    if case.op != "MUL":
        return None
    ne = case.normalized_params.get("ne")
    nr = case.normalized_params.get("nr")
    nf = case.normalized_params.get("nf")
    perm1 = case.normalized_params.get("perm1", list(POINTWISE_BASE_PERMUTATION))
    if not isinstance(ne, list) or len(ne) != 4 or any(not isinstance(value, int) for value in ne):
        return None
    if not isinstance(nr, list) or len(nr) != 4 or any(not isinstance(value, int) for value in nr):
        return None
    if not isinstance(nf, int):
        return None
    if not isinstance(perm1, list) or len(perm1) != 4 or any(not isinstance(value, int) for value in perm1):
        return None
    if int(nf) != int(ne[0]):
        return None
    if int(ne[0]) == 1:
        return None
    dtype = str(case.dtype.get("type", "")).upper()
    if dtype != "F32":
        return None
    if tuple(int(value) for value in perm1) != POINTWISE_BASE_PERMUTATION:
        return None
    if any(int(value) != 1 for value in nr):
        return None
    ncols = int(ne[0])
    nrows = int(ne[1]) * int(ne[2]) * int(ne[3])
    tensors = {
        "src0": _tensor_from_extents(dtype="F32", extents=(ncols, nrows)),
        "src1": _tensor_from_extents(dtype="F32", extents=(ncols, 1), strides=(1, 0)),
        "dst": _tensor_from_extents(dtype="F32", extents=(ncols, nrows)),
    }
    return ImportedRouteQuery(
        tensors=tensors,
        fallback_shape={"ncols": ncols, "nrows": nrows},
        route_ids=("rms_norm_mul_f32_n16_r60_vector_tail",),
    )


def _llamacpp_pointwise_import_query(case: ImportedCase) -> ImportedRouteQuery | None:
    if case.op not in {"ADD", "DIV", "MUL", "SUB"} or "sources" in case.normalized_params:
        return None
    if not {"ne", "nr"}.issubset(case.normalized_params):
        return None
    ne, nr, nf, perm1 = _parse_pointwise_parameters(case)
    if nf != 1:
        return None
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
    return ImportedRouteQuery(tensors=tensors, fallback_shape=_shape_from_extents(dst_extents))


def _llamacpp_set_rows_import_query(case: ImportedCase) -> ImportedRouteQuery | None:
    if case.op != "SET_ROWS" or "sources" in case.normalized_params:
        return None
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
    dst_extents = tuple(int(value) for value in ne)
    src0_extents = (int(ne[0]), int(row_count), int(ne[2]), int(ne[3]))
    src1_extents = (int(row_count), int(nr23[0]), int(nr23[1]), 1)
    src1_strides = (1, int(row_count), int(row_count), int(row_count))
    src_dtype = _dtype_from_case(case, "type_src", "type")
    dst_dtype = _dtype_from_case(case, "type_dst", "type")
    tensors = {
        "src0": _tensor_from_extents(dtype=src_dtype, extents=src0_extents),
        "src1": _tensor_from_extents(
            dtype=str(case.dtype.get("type_idx", "")).upper(),
            extents=src1_extents,
            strides=src1_strides,
        ),
        "dst": _tensor_from_extents(dtype=dst_dtype, extents=dst_extents),
    }
    return ImportedRouteQuery(tensors=tensors, fallback_shape=_shape_from_extents(list(dst_extents)))


def _llamacpp_rope_set_rows_import_query(case: ImportedCase) -> ImportedRouteQuery | None:
    if case.op != "ROPE_SET_ROWS":
        return None
    if str(case.dtype.get("type", "")).upper() != "F16":
        return None
    if str(case.dtype.get("type_idx", "")).upper() != "I64":
        return None
    ne = case.normalized_params.get("ne_a")
    mode = case.normalized_params.get("mode")
    if not isinstance(ne, list) or len(ne) != 4:
        raise ValueError("ROPE_SET_ROWS lowering requires a 4-D ne_a extent list")
    if any(not isinstance(value, int) for value in ne):
        raise ValueError("ROPE_SET_ROWS lowering requires integer ne_a extents")
    if not isinstance(mode, int):
        raise ValueError("ROPE_SET_ROWS lowering requires integer mode")
    if mode != 0:
        raise ValueError("ROPE_SET_ROWS v2 routing currently requires mode=0")
    extents = [int(value) for value in ne]
    if extents[3] != 1:
        raise ValueError("ROPE_SET_ROWS v2 routing currently requires ne_a[3]=1")
    ncols = extents[0]
    nheads = extents[1]
    ntokens = extents[2]
    n_dims = 128
    dst_rows = max(ntokens + 2, 4)
    tensors = {
        "src0": _tensor_from_extents(dtype="F32", extents=(ncols, nheads, ntokens, 1)),
        "pos": _tensor_from_extents(dtype="I32", extents=(ntokens, 1, 1, 1)),
        "freq": _tensor_from_extents(dtype="F32", extents=(max(n_dims // 2, 1), 1, 1, 1)),
        "src1": _tensor_from_extents(dtype="I64", extents=(ntokens, 1, 1, 1)),
        "dst": _tensor_from_extents(dtype="F16", extents=(ncols * nheads, dst_rows, 1, 1)),
    }
    return ImportedRouteQuery(
        tensors=tensors,
        fallback_shape={
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
        },
    )


def _llamacpp_rope_import_query(case: ImportedCase) -> ImportedRouteQuery | None:
    if case.op not in {"ROPE", "ROPE_SCALE"}:
        return None
    src_dtype = _dtype_from_case(case, "type_src", "type")
    dst_dtype = _dtype_from_case(case, "type_dst", "type", default=src_dtype)
    idx_dtype = _dtype_from_case(case, "type_idx", default="I32")
    if src_dtype != "F32" or dst_dtype != "F32" or idx_dtype != "I32":
        return None
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
    if view_mode != 0:
        raise ValueError("ROPE v2 routing requires contiguous input (v=0)")
    if inplace != 0:
        raise ValueError("ROPE v2 routing currently requires inplace=0")
    if case.op == "ROPE" and int(mode) not in {0, 2}:
        raise ValueError("ROPE v2 routing currently requires mode=0")
    expected_mode = 2 if case.op == "ROPE_SCALE" else int(mode)
    expected_freq_factor = 1 if case.op == "ROPE_SCALE" else 0
    if int(mode) != expected_mode:
        raise ValueError(f"ROPE v2 routing currently requires mode={expected_mode}")
    if int(freq_factor) != expected_freq_factor:
        raise ValueError(f"ROPE v2 routing currently requires ff={expected_freq_factor}")
    if case.op == "ROPE_SCALE" and float(ext_factor) != 0.0:
        raise ValueError("ROPE_SCALE v2 routing currently requires ef=0.0")
    extents = tuple(int(value) for value in ne)
    if case.op == "ROPE" and int(mode) == 2 and int(n_dims) != extents[0]:
        raise ValueError("ROPE NEOX v2 routing currently requires n_dims == ne_a[0]")
    route_ids = (
        ("rope_scale_f32_neox_freq_n128_d96_h24_t1_contiguous_4d",)
        if case.op == "ROPE_SCALE"
        else (
            ("rope_f32_normal_n128_h32_t2_contiguous_4d",)
            if int(mode) == 0
            else ("rope_neox_f32_n64_h128_t2_contiguous_4d",)
        )
    )
    ncols = extents[0]
    nheads = extents[1]
    ntokens = extents[2] * extents[3]
    tensors = {
        "src0": _tensor_from_extents(dtype="F32", extents=extents),
        "src1": _tensor_from_extents(dtype="I32", extents=(1, 1, ntokens, 1)),
        "dst": _tensor_from_extents(dtype="F32", extents=extents),
    }
    if case.op == "ROPE_SCALE":
        tensors["src2"] = _tensor_from_extents(
            dtype="F32",
            extents=(max(int(n_dims) // 2, 1), 1, 1, 1),
        )
    return ImportedRouteQuery(
        tensors=tensors,
        fallback_shape={
            **_shape_from_extents(list(extents)),
            "rope.ncols": ncols,
            "rope.n_dims": int(n_dims),
            "rope.nheads": nheads,
            "rope.ntokens": ntokens,
            "rope.src0_head_stride": ncols,
            "rope.src0_token_stride": ncols * nheads,
            "rope.dst_head_stride": ncols,
            "rope.dst_token_stride": ncols * nheads,
            "rope.pos_token_stride": 1,
        },
        route_ids=route_ids,
    )


def _llamacpp_flash_attn_ext_import_query(case: ImportedCase) -> ImportedRouteQuery | None:
    if case.op != "FLASH_ATTN_EXT":
        return None
    kv_dtype = _dtype_from_case(case, "type_KV", default="F16")
    if kv_dtype != "F16":
        raise ValueError("FLASH_ATTN_EXT v2 routing currently requires type_KV=f16")
    kv = case.normalized_params.get("kv")
    hsk = case.normalized_params.get("hsk")
    hsv = case.normalized_params.get("hsv")
    nb = case.normalized_params.get("nb")
    nh = case.normalized_params.get("nh")
    nr23 = case.normalized_params.get("nr23")
    permute = case.normalized_params.get("permute")
    mask = case.normalized_params.get("mask")
    sinks = case.normalized_params.get("sinks", 0)
    logit_softcap = case.normalized_params.get("logit_softcap", 0.0)
    max_bias = case.normalized_params.get("max_bias", 0.0)
    if not all(isinstance(value, int) for value in (kv, hsk, hsv, nb, nh, mask, sinks)):
        raise ValueError("FLASH_ATTN_EXT lowering requires integer kv/hsk/hsv/nb/nh/mask/sinks")
    if not isinstance(nr23, list) or len(nr23) != 2 or any(not isinstance(value, int) for value in nr23):
        raise ValueError("FLASH_ATTN_EXT lowering requires a 2-D integer nr23 extent list")
    if not isinstance(permute, list) or len(permute) != 4 or any(not isinstance(value, int) for value in permute):
        raise ValueError("FLASH_ATTN_EXT lowering requires a 4-D integer permute list")
    if isinstance(logit_softcap, bool) or not isinstance(logit_softcap, (int, float)):
        raise ValueError("FLASH_ATTN_EXT lowering requires numeric logit_softcap")
    if isinstance(max_bias, bool) or not isinstance(max_bias, (int, float)):
        raise ValueError("FLASH_ATTN_EXT lowering requires numeric max_bias")
    kv_value = int(kv)
    if int(hsk) != 128:
        raise ValueError("FLASH_ATTN_EXT v2 routing currently requires hsk=128")
    if int(hsv) != 128:
        raise ValueError("FLASH_ATTN_EXT v2 routing currently requires hsv=128")
    if int(nb) != 1:
        raise ValueError("FLASH_ATTN_EXT v2 routing currently requires nb=1")
    if nr23 != [4, 1]:
        raise ValueError("FLASH_ATTN_EXT v2 routing currently requires nr23=[4, 1]")
    if tuple(int(value) for value in permute) != POINTWISE_BASE_PERMUTATION:
        raise ValueError("FLASH_ATTN_EXT v2 routing currently requires permute=[0, 1, 2, 3]")
    if int(mask) != 1:
        raise ValueError("FLASH_ATTN_EXT v2 routing currently requires mask=1")
    if int(sinks) != 0:
        raise ValueError("FLASH_ATTN_EXT v2 routing currently requires sinks=0")
    if float(logit_softcap) != 0.0:
        raise ValueError("FLASH_ATTN_EXT v2 routing currently requires logit_softcap=0.0")
    if float(max_bias) != 0.0:
        raise ValueError("FLASH_ATTN_EXT v2 routing currently requires max_bias=0.0")
    fixed_decode_routes = {
        256: "softmax_kqv_f32_f16_decode_kv256_d128_h24_hkv8_wg64_rows2",
        512: "softmax_kqv_f32_f16_decode_kv512_d128_h24_hkv8_wg256_rows128",
        768: "softmax_kqv_f32_f16_decode_kv768_d128_h24_hkv8_wg256_rows128",
    }
    if int(nh) == 8:
        if kv_value < 512 or kv_value > 4096 or kv_value % 512 != 0:
            raise ValueError(
                "FLASH_ATTN_EXT v2 routing currently requires kv to be a multiple of 512 between 512 and 4096"
            )
        route_id = "softmax_kqv_f32_f16_masked_identity_kv512_4096_d128_h8_wg256_row1"
        cols = 8
    elif int(nh) == 24:
        route_id = fixed_decode_routes.get(kv_value)
        if route_id is None:
            raise ValueError("FLASH_ATTN_EXT v2 routing currently requires kv=256, kv=512, or kv=768")
        cols = 24
    else:
        raise ValueError("FLASH_ATTN_EXT v2 routing currently requires nh=8")
    rows = 128
    nheads_kv = 8
    src1_cols = rows * nheads_kv
    tensors = {
        "src0": _tensor_from_extents(dtype="F32", extents=(kv_value, cols)),
        "mask": _tensor_from_extents(dtype="F32", extents=(kv_value, 1)),
        "src1": _tensor_from_extents(dtype="F16", extents=(kv_value, src1_cols)),
        "dst": _tensor_from_extents(dtype="F32", extents=(rows, cols)),
    }
    return ImportedRouteQuery(
        tensors=tensors,
        fallback_shape={
            "d0": cols,
            "d1": rows,
            "k": kv_value,
            "rows": rows,
            "cols": cols,
            "nheads_kv": nheads_kv,
            "src0_d0": kv_value,
            "src0_d1": cols,
            "mask_d0": kv_value,
            "mask_d1": 1,
            "src1_d0": kv_value,
            "src1_d1": src1_cols,
            "dst_d0": rows,
            "dst_d1": cols,
        },
        route_ids=(route_id,),
    )


def _llamacpp_unary_import_query(case: ImportedCase) -> ImportedRouteQuery | None:
    if case.op not in {"ABS", "EXP", "NEG", "RELU"} or "ne_a" not in case.normalized_params:
        return None
    ne, v = _parse_unary_parameters(case)
    dtype = str(case.dtype.get("type", "")).upper()
    dimensions = tuple(int(extent) for extent in ne)
    if v == 1:
        if len(dimensions) != 4:
            raise ValueError("llama.cpp unary v=1 import requires 4-D ne_a")
        src0_strides = _llamacpp_unary_view_strides(dimensions)
    else:
        src0_strides = contiguous_strides(dimensions)
    tensors = {
        "src0": _tensor_from_extents(dtype=dtype, extents=dimensions, strides=src0_strides),
        "dst": _tensor_from_extents(dtype=dtype, extents=dimensions),
    }
    return ImportedRouteQuery(tensors=tensors, fallback_shape=_shape_from_extents(list(dimensions)))


def _llamacpp_unary_view_strides(dimensions: tuple[int, ...]) -> tuple[int, ...]:
    # ggml test_unary v=1 uses a deterministic inflated parent view.
    view_inflation = (3, 2, 5, 4)
    strides = [1]
    stride = 1
    for index in range(1, len(dimensions)):
        stride *= view_inflation[index - 1] * int(dimensions[index - 1])
        strides.append(stride)
    return tuple(strides)


def _llamacpp_soft_max_import_query(case: ImportedCase) -> ImportedRouteQuery | None:
    if case.op != "SOFT_MAX":
        return None
    ne = case.normalized_params.get("ne")
    nr23 = case.normalized_params.get("nr23")
    mask = case.normalized_params.get("mask", 0)
    sinks = case.normalized_params.get("sinks", 0)
    max_bias = case.normalized_params.get("max_bias", 0.0)
    inplace = case.normalized_params.get("inplace", 0)
    if not isinstance(ne, list) or len(ne) != 4:
        raise ValueError("SOFT_MAX lowering requires a 4-D ne extent list")
    if not isinstance(nr23, list) or len(nr23) != 2:
        raise ValueError("SOFT_MAX lowering requires a 2-D nr23 extent list")
    if any(not isinstance(value, int) for value in ne):
        raise ValueError("SOFT_MAX lowering requires integer ne extents")
    if any(not isinstance(value, int) for value in nr23):
        raise ValueError("SOFT_MAX lowering requires integer nr23 extents")
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
    if int(mask) == 1 and nr23 != [1, 1]:
        raise ValueError("SOFT_MAX masked v2 routing currently requires nr23=[1, 1]")
    if int(mask) not in {0, 1}:
        raise ValueError("SOFT_MAX v2 routing currently requires mask=0 or mask=1")
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
    dtype = str(case.dtype.get("type", "")).upper()
    tensors = {
        "src0": _tensor_from_extents(dtype=dtype, extents=(ncols, nrows)),
        "dst": _tensor_from_extents(dtype=dtype, extents=(ncols, nrows)),
    }
    if int(mask) == 1:
        tensors["mask"] = _tensor_from_extents(dtype="F32", extents=(ncols, nrows))
    return ImportedRouteQuery(tensors=tensors, fallback_shape=_shape_from_extents([ncols, nrows]))


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


def _parse_numeric_eps(case: ImportedCase, *, op_name: str) -> float:
    eps = case.normalized_params.get("eps", 0.0)
    if isinstance(eps, bool):
        raise ValueError(f"{op_name} lowering requires numeric eps")
    if isinstance(eps, str):
        try:
            return float(eps)
        except ValueError as exc:
            raise ValueError(f"{op_name} lowering requires numeric eps") from exc
    if isinstance(eps, (int, float)):
        return float(eps)
    raise ValueError(f"{op_name} lowering requires numeric eps")


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


def shape_for_resolved_route(
    route: V2Route,
    tensors: dict[str, ConcreteTensor],
    fallback_shape: dict[str, int],
) -> EncodedRouteShape:
    if not route.tensors:
        return EncodedRouteShape(items=tuple((str(name), int(value)) for name, value in fallback_shape.items()))
    encoded = dict(encode_route_shape(route, tensors).items)
    encoded.update(_shape_binding_defaults(route, tensors, encoded))
    for name, value in fallback_shape.items():
        encoded[str(name)] = int(value)
    return EncodedRouteShape(items=tuple(encoded.items()))


def _shape_binding_defaults(
    route: V2Route,
    tensors: dict[str, ConcreteTensor],
    existing: dict[str, int],
) -> dict[str, int]:
    defaults: dict[str, int] = {}
    for binding in route.bindings:
        if binding.source is None or not binding.source.startswith("shape."):
            continue
        key = binding.source.removeprefix("shape.")
        if key in existing or key in defaults:
            continue
        resolved = _resolve_shape_binding_default(key, tensors)
        if resolved is not None:
            defaults[key] = resolved
    return defaults


def _resolve_shape_binding_default(
    key: str,
    tensors: dict[str, ConcreteTensor],
) -> int | None:
    if key.endswith("_stride"):
        base_key = key.removesuffix("_stride")
        split_at = base_key.rfind("_")
        if split_at <= 0:
            return None
        tensor_name = base_key[:split_at]
        dimension_name = base_key[split_at + 1 :]
        tensor = tensors.get(tensor_name)
        if tensor is None:
            return None
        for dimension in tensor.dimensions:
            if dimension.name == dimension_name:
                return int(dimension.stride)
        return None
    split_at = key.rfind("_")
    if split_at <= 0:
        return None
    tensor_name = key[:split_at]
    dimension_name = key[split_at + 1 :]
    tensor = tensors.get(tensor_name)
    if tensor is None:
        return None
    for dimension in tensor.dimensions:
        if dimension.name == dimension_name:
            return int(dimension.size)
    return None


def resolve_route_for_case(
    case: ImportedCase,
    routes: list[V2Route],
    *,
    selector: RouteSelector,
) -> tuple[V2Route | None, dict[str, int] | None, UnmappedReason | None, str | None]:
    yaml_query_errors: list[str] = []
    try:
        yaml_queries = yaml_import_queries(case)
    except ValueError as exc:
        yaml_queries = ()
        yaml_query_errors.append(str(exc))
    if yaml_queries:
        route_order = {route.id: index for index, route in enumerate(routes)}
        matched_queries: list[tuple[int, int, str]] = []
        for query_index, query in enumerate(yaml_queries):
            match = selector.select(
                case.op,
                RouteMatchQuery(
                    tensors=query.tensors,
                    allowed_route_ids=query.route_ids,
                ),
            )
            if match is None:
                continue
            matched_queries.append(
                (route_order[match.route_id], query_index, match.route_id)
            )
        if matched_queries:
            # Preserve the original route-major, query-minor selection order
            # while presenting one query at a time to the selector.
            _, query_index, route_id = min(matched_queries)
            route = {route.id: route for route in routes}[route_id]
            query = yaml_queries[query_index]
            encoded_shape = shape_for_resolved_route(route, query.tensors, query.fallback_shape)
            return route, encoded_shape.as_dict(), None, None
        yaml_query_errors.append("YAML import tensor query did not satisfy any v2 route")

    dtype_matching = [route for route in routes if route_accepts_dtype(route, case.dtype)]
    if not dtype_matching:
        return (
            None,
            None,
            UnmappedReason.NO_DTYPE_MAPPING,
            "matching v2 op mapping exists, but not for this dtype combination",
        )
    if case.op in _IMPORT_QUERY_ONLY_OPS and yaml_queries:
        return (
            None,
            None,
            UnmappedReason.NO_ROUTE_MATCH,
            "YAML import tensor query did not satisfy any v2 route",
        )
    if case.op in _IMPORT_QUERY_ONLY_OPS and yaml_query_errors:
        return (
            None,
            None,
            UnmappedReason.SHAPE_LOWERING_NOT_IMPLEMENTED,
            yaml_query_errors[0],
        )
    if case.op in _IMPORT_QUERY_ONLY_OPS:
        return (
            None,
            None,
            UnmappedReason.SHAPE_LOWERING_NOT_IMPLEMENTED,
            "YAML import tensor query is not implemented for this case",
        )

    return (
        None,
        None,
        UnmappedReason.SHAPE_LOWERING_NOT_IMPLEMENTED,
        yaml_query_errors[0]
        if yaml_query_errors
        else "YAML import tensor query is not implemented for this case",
    )


def resolve_imported_suite(
    suite: ImportedSuite,
    *,
    routing_dir=None,
    catalog: RouteCatalog | None = None,
    selector: RouteSelector,
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
            route, shape, reason, detail = resolve_route_for_case(
                case,
                op_routes,
                selector=selector,
            )
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
