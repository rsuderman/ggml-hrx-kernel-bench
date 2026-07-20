from __future__ import annotations

from collections.abc import Mapping
from dataclasses import MISSING, fields

import pytest

from ggml_hrx_kernel_bench.materialized_assets import materialize_asset_root
from ggml_hrx_kernel_bench.routing.v2.matching import route_accepts_tensors
from ggml_hrx_kernel_bench.routing.v2.models import (
    ConcreteTensor,
    ConcreteTensorDimension,
    V2Route,
)
from ggml_hrx_kernel_bench.routing.v2.query import RouteCatalog, load_route_catalog


REPRESENTATIVE_ROUTE_IDS = frozenset(
    {
        "clamp_f32_contiguous_4d",
        "copy_f32_f32_non_contiguous_4d",
        "get_rows_q8_0_f32_embedding_rows_descriptor_4d",
        "mul_mat_q8_0_f32_contiguous_4d",
        "mul_mat_f16_f32_tiled_batched_4d",
        "argsort_f32_i32_n128_r1_desc_wg128",
    }
)

EXPECTED_CATALOG_FEATURES = frozenset(
    {
        "constraint:divides",
        "constraint:equals",
        "constraint:exact_rank",
        "constraint:indexed_bounds",
        "constraint:indexed_multiple_of",
        "constraint:rank_range",
        "constraint:scalar_bounds",
        "constraint:scalar_multiple_of",
        "tensor:permutation_capture",
        "tensor:wildcard_dtype",
        "value:chain_permutations",
        "value:contiguous_strides",
        "value:element",
        "value:empty",
        "value:head",
        "value:inverse_permutation",
        "value:permuted_contiguous_strides",
        "value:product",
        "value:tail",
    }
)

# These contain dataclass field names, not catalog feature labels. Keep them in
# sync with the explicit feature checks in _route_features. An active field not
# listed here is intentionally surfaced as a new, unexpected catalog feature.
HANDLED_CONSTRAINT_FIELDS = frozenset(
    {
        "name",
        "length",
        "rank_min",
        "rank_max",
        "index",
        "min",
        "max",
        "multiple_of",
        "iota",
        "equals",
        "divides",
    }
)

HANDLED_TENSOR_DESCRIPTOR_FIELDS = frozenset(
    {
        "dtype",
        "dimensions_capture",
        "strides_capture",
        "permutation_capture",
    }
)


def _tensor(
    *,
    dtype: str,
    sizes: tuple[int, ...],
    strides: tuple[int, ...],
    permutation: tuple[int, ...] | None = None,
) -> ConcreteTensor:
    if len(sizes) != len(strides):
        raise ValueError("tensor sizes and strides must have the same length")
    return ConcreteTensor(
        dtype=dtype,
        dimensions=tuple(
            ConcreteTensorDimension(name=f"d{index}", size=size, stride=stride)
            for index, (size, stride) in enumerate(zip(sizes, strides, strict=True))
        ),
        permutation=permutation,
    )


def _replace_tensor(
    tensors: Mapping[str, ConcreteTensor],
    name: str,
    tensor: ConcreteTensor,
) -> dict[str, ConcreteTensor]:
    return {**tensors, name: tensor}


def _active_dataclass_fields(value: object) -> set[str]:
    """Return fields that are required or differ from their dataclass defaults."""
    active: set[str] = set()
    for definition in fields(value):
        current = getattr(value, definition.name)
        if definition.default is not MISSING:
            default = definition.default
        elif definition.default_factory is not MISSING:
            default = definition.default_factory()
        else:
            active.add(definition.name)
            continue
        if current != default:
            active.add(definition.name)
    return active


def _route_features(route: V2Route) -> frozenset[str]:
    """Inventory matcher features exercised by one real catalog route.

    HANDLED_* fields must stay synchronized with the explicit descriptor and
    constraint checks below. Fields absent from those sets flow into the
    inventory so the catalog-feature guard fails until their semantics have a
    representative real-route test.
    """
    features = {f"value:{value.operation_kind}" for value in route.values}
    if not route.values:
        features.add("value:empty")
    for descriptor in route.tensors.values():
        if descriptor.permutation_capture is not None:
            features.add("tensor:permutation_capture")
        if descriptor.dtype is None:
            features.add("tensor:wildcard_dtype")
        features.update(
            f"tensor:{field_name}"
            for field_name in _active_dataclass_fields(descriptor)
            - HANDLED_TENSOR_DESCRIPTOR_FIELDS
        )

    for check in route.constraints.checks:
        features.update(
            f"constraint:{field_name}"
            for field_name in _active_dataclass_fields(check) - HANDLED_CONSTRAINT_FIELDS
        )
        if check.equals:
            features.add("constraint:equals")
            continue
        if check.divides:
            features.add("constraint:divides")
            continue
        if check.iota:
            features.add("constraint:iota")
            continue
        if check.length is not None:
            features.add("constraint:exact_rank")
            continue
        if check.rank_min is not None or check.rank_max is not None:
            features.add("constraint:rank_range")
            continue
        if check.index is not None:
            if check.min is not None or check.max is not None:
                features.add("constraint:indexed_bounds")
            if check.multiple_of is not None:
                features.add("constraint:indexed_multiple_of")
            if check.min is None and check.max is None and check.multiple_of is None:
                features.add("constraint:indexed")
            continue
        if check.min is not None or check.max is not None:
            features.add("constraint:scalar_bounds")
        if check.multiple_of is not None:
            features.add("constraint:scalar_multiple_of")
        if check.min is None and check.max is None and check.multiple_of is None:
            features.add("constraint:scalar")
    return frozenset(features)


@pytest.fixture(scope="module")
def real_v2_catalog(tmp_path_factory: pytest.TempPathFactory) -> RouteCatalog:
    asset_root = materialize_asset_root(
        tmp_path_factory.mktemp("routing-v2-matching-catalog") / "assets",
        force=True,
    )
    catalog = load_route_catalog(asset_root / "catalog" / "v2")
    missing_routes = REPRESENTATIVE_ROUTE_IDS - set(catalog.routes_by_id)
    assert not missing_routes, f"materialized catalog is missing routes: {sorted(missing_routes)}"
    return catalog


CLAMP_4D = {
    "src0": _tensor(
        dtype="F32",
        sizes=(10, 5, 4, 3),
        strides=(1, 10, 50, 200),
    ),
    "dst": _tensor(
        dtype="F32",
        sizes=(10, 5, 4, 3),
        strides=(1, 10, 50, 200),
    ),
}

ADD_ROW_BROADCAST_2D = {
    "src0": _tensor(dtype="F32", sizes=(128, 64), strides=(1, 128)),
    "src1": _tensor(dtype="F32", sizes=(128, 1), strides=(1, 0)),
    "dst": _tensor(dtype="F32", sizes=(128, 64), strides=(1, 128)),
}

ADD_GENERIC_4D = {
    "src0": _tensor(
        dtype="F32",
        sizes=(4, 5, 6, 7),
        strides=(3, 29, 211, 1703),
    ),
    "src1": _tensor(
        dtype="F32",
        sizes=(2, 5, 3, 7),
        strides=(1, 2, 10, 30),
    ),
    "dst": _tensor(
        dtype="F32",
        sizes=(4, 5, 6, 7),
        strides=(11, 47, 263, 1499),
    ),
}

ADD_GENERIC_2D_NO_VALUES = {
    "src0": _tensor(dtype="F32", sizes=(128, 64), strides=(7, 999)),
    "src1": _tensor(dtype="F32", sizes=(32, 1), strides=(1, 500)),
    "dst": _tensor(dtype="F32", sizes=(128, 64), strides=(13, 777)),
}

COPY_NONIDENTITY_4D = {
    "src0": _tensor(
        dtype="F32",
        sizes=(2, 3, 4, 5),
        strides=(1, 40, 10, 2),
        permutation=(0, 3, 1, 2),
    ),
    "dst": _tensor(
        dtype="F32",
        sizes=(2, 3, 4, 5),
        strides=(1, 2, 6, 24),
        permutation=(0, 2, 1, 3),
    ),
}

COPY_IMPLICIT_IDENTITY_4D = {
    "src0": _tensor(
        dtype="F32",
        sizes=(2, 3, 4, 5),
        strides=(1, 2, 6, 24),
    ),
    "dst": _tensor(
        dtype="F32",
        sizes=(2, 3, 4, 5),
        strides=(1, 2, 6, 24),
    ),
}

GET_ROWS_Q8_WIDTH_MIN = {
    "src0": _tensor(
        dtype="Q8_0",
        sizes=(32, 7, 1, 1),
        strides=(1, 32, 224, 224),
    ),
    "src1": _tensor(
        dtype="I32",
        sizes=(3, 1, 1, 1),
        strides=(1, 3, 3, 3),
    ),
    "dst": _tensor(
        dtype="F32",
        sizes=(32, 3, 1, 1),
        strides=(1, 32, 96, 96),
    ),
}

GET_ROWS_Q8_WIDTH_MAX = {
    "src0": _tensor(
        dtype="Q8_0",
        sizes=(65536, 2, 1, 1),
        strides=(1, 65536, 131072, 131072),
    ),
    "src1": _tensor(
        dtype="I32",
        sizes=(1, 1, 1, 1),
        strides=(1, 1, 1, 1),
    ),
    "dst": _tensor(
        dtype="F32",
        sizes=(65536, 1, 1, 1),
        strides=(1, 65536, 65536, 65536),
    ),
}

MUL_MAT_Q8_K_MIN = {
    "src0": _tensor(
        dtype="Q8_0",
        sizes=(32, 4, 1, 1),
        strides=(1, 32, 128, 128),
    ),
    "src1": _tensor(
        dtype="F32",
        sizes=(32, 3, 1, 1),
        strides=(1, 32, 96, 96),
    ),
    "dst": _tensor(
        dtype="F32",
        sizes=(4, 3, 1, 1),
        strides=(1, 4, 12, 12),
    ),
}

MUL_MAT_Q8_K_MAX = {
    "src0": _tensor(
        dtype="Q8_0",
        sizes=(32768, 2, 1, 1),
        strides=(1, 32768, 65536, 65536),
    ),
    "src1": _tensor(
        dtype="F32",
        sizes=(32768, 3, 1, 1),
        strides=(1, 32768, 98304, 98304),
    ),
    "dst": _tensor(
        dtype="F32",
        sizes=(2, 3, 1, 1),
        strides=(1, 2, 6, 6),
    ),
}

MUL_MAT_F16_YAML_CASE_33 = {
    "src0": _tensor(
        dtype="F16",
        sizes=(128, 1056, 1, 3),
        strides=(1, 128, 135168, 135168),
        permutation=(0, 2, 1, 3),
    ),
    "src1": _tensor(
        dtype="F32",
        sizes=(128, 1, 1, 3),
        strides=(1, 128, 128, 128),
        permutation=(0, 2, 1, 3),
    ),
    "dst": _tensor(
        dtype="F32",
        sizes=(1056, 1, 1, 3),
        strides=(1, 1056, 1056, 1056),
    ),
}

ARGSORT_WILDCARD_DST = {
    "src0": _tensor(dtype="F32", sizes=(128, 1), strides=(1, 128)),
    "dst": _tensor(dtype="BF16", sizes=(128, 1), strides=(1, 128)),
}


POSITIVE_CASES = (
    pytest.param(
        "clamp_f32_contiguous_4d",
        "inclusive rank-4 upper boundary with an unrelated tensor role",
        {
            **CLAMP_4D,
            "unrelated": _tensor(dtype="I32", sizes=(1,), strides=(1,)),
        },
        id="clamp-rank-max-contiguous",
    ),
    pytest.param(
        "clamp_f32_contiguous_4d",
        "inclusive rank-2 lower boundary",
        {
            "src0": _tensor(dtype="F32", sizes=(10, 5), strides=(1, 10)),
            "dst": _tensor(dtype="F32", sizes=(10, 5), strides=(1, 10)),
        },
        id="clamp-rank-min-contiguous",
    ),
    pytest.param(
        "copy_f32_f32_non_contiguous_4d",
        "implicit identity permutations",
        COPY_IMPLICIT_IDENTITY_4D,
        id="copy-implicit-identity-permutations",
    ),
    pytest.param(
        "copy_f32_f32_non_contiguous_4d",
        "explicit nonidentity permutations and derived strides",
        COPY_NONIDENTITY_4D,
        id="copy-nonidentity-permutations",
    ),
    pytest.param(
        "get_rows_q8_0_f32_embedding_rows_descriptor_4d",
        "inclusive indexed width and multiple lower boundary",
        GET_ROWS_Q8_WIDTH_MIN,
        id="get-rows-indexed-width-min",
    ),
    pytest.param(
        "get_rows_q8_0_f32_embedding_rows_descriptor_4d",
        "inclusive indexed width and multiple upper boundary",
        GET_ROWS_Q8_WIDTH_MAX,
        id="get-rows-indexed-width-max",
    ),
    pytest.param(
        "mul_mat_q8_0_f32_contiguous_4d",
        "inclusive scalar K and multiple lower boundary",
        MUL_MAT_Q8_K_MIN,
        id="mul-mat-q8-scalar-k-min",
    ),
    pytest.param(
        "mul_mat_q8_0_f32_contiguous_4d",
        "inclusive scalar K and multiple upper boundary",
        MUL_MAT_Q8_K_MAX,
        id="mul-mat-q8-scalar-k-max",
    ),
    pytest.param(
        "mul_mat_f16_f32_tiled_batched_4d",
        "transposed batched MUL_MAT[33] layout",
        MUL_MAT_F16_YAML_CASE_33,
        id="mul-mat-f16-yaml-case-33",
    ),
    pytest.param(
        "argsort_f32_i32_n128_r1_desc_wg128",
        "wildcard destination dtype",
        ARGSORT_WILDCARD_DST,
        id="argsort-wildcard-dst-dtype",
    ),
)


NEGATIVE_CASES = (
    pytest.param(
        "clamp_f32_contiguous_4d",
        "missing required dst tensor",
        {"src0": CLAMP_4D["src0"]},
        id="clamp-missing-dst",
    ),
    pytest.param(
        "argsort_f32_i32_n128_r1_desc_wg128",
        "fixed src0 dtype mismatch",
        _replace_tensor(
            ARGSORT_WILDCARD_DST,
            "src0",
            _tensor(dtype="F16", sizes=(128, 1), strides=(1, 128)),
        ),
        id="argsort-fixed-src0-dtype-mismatch",
    ),
    pytest.param(
        "clamp_f32_contiguous_4d",
        "fixed dst dtype mismatch",
        _replace_tensor(
            CLAMP_4D,
            "dst",
            _tensor(
                dtype="F16",
                sizes=(10, 5, 4, 3),
                strides=(1, 10, 50, 200),
            ),
        ),
        id="clamp-fixed-dst-dtype-mismatch",
    ),
    pytest.param(
        "clamp_f32_contiguous_4d",
        "rank below supported range",
        {
            "src0": _tensor(dtype="F32", sizes=(10,), strides=(1,)),
            "dst": _tensor(dtype="F32", sizes=(10,), strides=(1,)),
        },
        id="clamp-rank-below-range",
    ),
    pytest.param(
        "clamp_f32_contiguous_4d",
        "rank above supported range",
        {
            "src0": _tensor(
                dtype="F32",
                sizes=(2, 2, 2, 2, 2),
                strides=(1, 2, 4, 8, 16),
            ),
            "dst": _tensor(
                dtype="F32",
                sizes=(2, 2, 2, 2, 2),
                strides=(1, 2, 4, 8, 16),
            ),
        },
        id="clamp-rank-above-range",
    ),
    pytest.param(
        "clamp_f32_contiguous_4d",
        "cross-tensor dimension equality mismatch",
        _replace_tensor(
            CLAMP_4D,
            "src0",
            _tensor(
                dtype="F32",
                sizes=(11, 5, 4, 3),
                strides=(1, 10, 50, 200),
            ),
        ),
        id="clamp-cross-tensor-dimensions-mismatch",
    ),
    pytest.param(
        "get_rows_q8_0_f32_embedding_rows_descriptor_4d",
        "derived width equality mismatch",
        _replace_tensor(
            GET_ROWS_Q8_WIDTH_MIN,
            "dst",
            _tensor(
                dtype="F32",
                sizes=(64, 3, 1, 1),
                strides=(1, 64, 192, 192),
            ),
        ),
        id="get-rows-derived-width-mismatch",
    ),
    pytest.param(
        "clamp_f32_contiguous_4d",
        "noncontiguous source and destination layouts",
        {
            "src0": _tensor(
                dtype="F32",
                sizes=(10, 5, 4, 3),
                strides=(1, 10, 51, 200),
            ),
            "dst": _tensor(
                dtype="F32",
                sizes=(10, 5, 4, 3),
                strides=(1, 10, 51, 200),
            ),
        },
        id="clamp-noncontiguous-layout",
    ),
    pytest.param(
        "get_rows_q8_0_f32_embedding_rows_descriptor_4d",
        "indexed width above maximum",
        {
            "src0": _tensor(
                dtype="Q8_0",
                sizes=(65568, 2, 1, 1),
                strides=(1, 65568, 131136, 131136),
            ),
            "src1": GET_ROWS_Q8_WIDTH_MAX["src1"],
            "dst": _tensor(
                dtype="F32",
                sizes=(65568, 1, 1, 1),
                strides=(1, 65568, 65568, 65568),
            ),
        },
        id="get-rows-indexed-width-above-max",
    ),
    pytest.param(
        "get_rows_q8_0_f32_embedding_rows_descriptor_4d",
        "indexed width inside bounds but not divisible",
        {
            "src0": _tensor(
                dtype="Q8_0",
                sizes=(33, 7, 1, 1),
                strides=(1, 33, 231, 231),
            ),
            "src1": GET_ROWS_Q8_WIDTH_MIN["src1"],
            "dst": _tensor(
                dtype="F32",
                sizes=(33, 3, 1, 1),
                strides=(1, 33, 99, 99),
            ),
        },
        id="get-rows-indexed-width-not-multiple",
    ),
    pytest.param(
        "mul_mat_q8_0_f32_contiguous_4d",
        "scalar K below minimum",
        {
            "src0": _tensor(dtype="Q8_0", sizes=(0, 4, 1, 1), strides=(1, 0, 0, 0)),
            "src1": _tensor(dtype="F32", sizes=(0, 3, 1, 1), strides=(1, 0, 0, 0)),
            "dst": MUL_MAT_Q8_K_MIN["dst"],
        },
        id="mul-mat-q8-scalar-k-below-min",
    ),
    pytest.param(
        "mul_mat_q8_0_f32_contiguous_4d",
        "scalar K above maximum",
        {
            "src0": _tensor(
                dtype="Q8_0",
                sizes=(32800, 2, 1, 1),
                strides=(1, 32800, 65600, 65600),
            ),
            "src1": _tensor(
                dtype="F32",
                sizes=(32800, 3, 1, 1),
                strides=(1, 32800, 98400, 98400),
            ),
            "dst": MUL_MAT_Q8_K_MAX["dst"],
        },
        id="mul-mat-q8-scalar-k-above-max",
    ),
    pytest.param(
        "mul_mat_q8_0_f32_contiguous_4d",
        "scalar K inside bounds but not divisible",
        {
            "src0": _tensor(
                dtype="Q8_0",
                sizes=(33, 4, 1, 1),
                strides=(1, 33, 132, 132),
            ),
            "src1": _tensor(
                dtype="F32",
                sizes=(33, 3, 1, 1),
                strides=(1, 33, 99, 99),
            ),
            "dst": MUL_MAT_Q8_K_MIN["dst"],
        },
        id="mul-mat-q8-scalar-k-not-multiple",
    ),
    pytest.param(
        "mul_mat_q8_0_f32_contiguous_4d",
        "derived K equality mismatch",
        _replace_tensor(
            MUL_MAT_Q8_K_MIN,
            "src1",
            _tensor(
                dtype="F32",
                sizes=(64, 3, 1, 1),
                strides=(1, 64, 192, 192),
            ),
        ),
        id="mul-mat-q8-derived-k-mismatch",
    ),
    pytest.param(
        "copy_f32_f32_non_contiguous_4d",
        "short source permutation",
        _replace_tensor(
            COPY_NONIDENTITY_4D,
            "src0",
            _tensor(
                dtype="F32",
                sizes=(2, 3, 4, 5),
                strides=(1, 40, 10, 2),
                permutation=(0, 3, 1),
            ),
        ),
        id="copy-short-source-permutation",
    ),
    pytest.param(
        "copy_f32_f32_non_contiguous_4d",
        "duplicate destination permutation axis",
        _replace_tensor(
            COPY_NONIDENTITY_4D,
            "dst",
            _tensor(
                dtype="F32",
                sizes=(2, 3, 4, 5),
                strides=(1, 2, 6, 24),
                permutation=(0, 0, 1, 3),
            ),
        ),
        id="copy-duplicate-destination-permutation",
    ),
    pytest.param(
        "copy_f32_f32_non_contiguous_4d",
        "out-of-range source permutation axis",
        _replace_tensor(
            COPY_NONIDENTITY_4D,
            "src0",
            _tensor(
                dtype="F32",
                sizes=(2, 3, 4, 5),
                strides=(1, 40, 10, 2),
                permutation=(0, 4, 1, 2),
            ),
        ),
        id="copy-out-of-range-source-permutation",
    ),
    pytest.param(
        "copy_f32_f32_non_contiguous_4d",
        "correct permutations with incorrect derived source stride",
        _replace_tensor(
            COPY_NONIDENTITY_4D,
            "src0",
            _tensor(
                dtype="F32",
                sizes=(2, 3, 4, 5),
                strides=(1, 41, 10, 2),
                permutation=(0, 3, 1, 2),
            ),
        ),
        id="copy-correct-permutations-wrong-derived-stride",
    ),
    pytest.param(
        "get_rows_q8_0_f32_embedding_rows_descriptor_4d",
        "too-short destination for element resolution",
        _replace_tensor(
            GET_ROWS_Q8_WIDTH_MIN,
            "dst",
            _tensor(dtype="F32", sizes=(32,), strides=(1,)),
        ),
        id="get-rows-short-element-source",
    ),
)


@pytest.mark.parametrize(("route_id", "property_name", "tensors"), POSITIVE_CASES)
def test_real_catalog_route_accepts_valid_tensor_metadata(
    real_v2_catalog: RouteCatalog,
    route_id: str,
    property_name: str,
    tensors: Mapping[str, ConcreteTensor],
) -> None:
    route = real_v2_catalog.routes_by_id[route_id]
    assert route_accepts_tensors(route, tensors), (
        f"route {route_id!r} rejected valid tensor metadata for {property_name}"
    )


@pytest.mark.parametrize(("route_id", "property_name", "tensors"), NEGATIVE_CASES)
def test_real_catalog_route_rejects_invalid_tensor_metadata(
    real_v2_catalog: RouteCatalog,
    route_id: str,
    property_name: str,
    tensors: Mapping[str, ConcreteTensor],
) -> None:
    route = real_v2_catalog.routes_by_id[route_id]
    assert not route_accepts_tensors(route, tensors), (
        f"route {route_id!r} accepted invalid tensor metadata for {property_name}"
    )
