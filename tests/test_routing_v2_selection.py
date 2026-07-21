from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping

import pytest

import ggml_hrx_kernel_bench.routing.v2.selection as selection_module
from ggml_hrx_kernel_bench.routing.v2.models import (
    ConcreteTensor,
    ConcreteTensorDimension,
    ConstraintCheck,
    RouteConstraints,
    SyntheticTensorDescriptor,
    TensorDescriptor,
    V2Route,
)
from ggml_hrx_kernel_bench.routing.v2.query import RouteCatalog
from ggml_hrx_kernel_bench.routing.v2.selection import (
    ROUTE_QUERY_SCHEMA,
    RouteQuery,
    materialize_route_query_tensors,
    route_query_from_json,
    route_query_to_json,
    select_route_query,
)


def _tensor(
    *,
    dtype: str = "F32",
    sizes: tuple[int, ...] = (4,),
    strides: tuple[int, ...] = (1,),
) -> ConcreteTensor:
    return ConcreteTensor(
        dtype=dtype,
        dimensions=tuple(
            ConcreteTensorDimension(name=f"d{index}", size=size, stride=stride)
            for index, (size, stride) in enumerate(zip(sizes, strides, strict=True))
        ),
    )


def _route(
    route_id: str,
    *,
    op: str = "TEST",
    tensors: Mapping[str, TensorDescriptor] | None = None,
    constraints: tuple[ConstraintCheck, ...] = (),
    attributes: Mapping[str, Any] | None = None,
    synthetic_tensors: Mapping[str, SyntheticTensorDescriptor] | None = None,
) -> V2Route:
    return V2Route(
        id=route_id,
        family="test",
        op=op,
        source_id="test",
        kernel_path=f"test/{route_id}.loom",
        root_symbol=f"@{route_id}",
        export_name=route_id,
        tensors=(
            tensors
            if tensors is not None
            else {
                "src0": TensorDescriptor(
                    dtype="F32",
                    dimensions_capture="dimensions",
                    strides_capture="strides",
                )
            }
        ),
        values=(),
        constraints=RouteConstraints(checks=constraints),
        launch={},
        bindings=(),
        attributes={} if attributes is None else attributes,
        synthetic_tensors={} if synthetic_tensors is None else synthetic_tensors,
    )


def _catalog(*routes: V2Route) -> RouteCatalog:
    by_op: dict[str, list[V2Route]] = defaultdict(list)
    by_family: dict[str, list[V2Route]] = defaultdict(list)
    for route in routes:
        by_op[route.op].append(route)
        by_family[route.family].append(route)
    return RouteCatalog(
        routes=routes,
        routes_by_op={op: tuple(op_routes) for op, op_routes in by_op.items()},
        routes_by_family={family: tuple(family_routes) for family, family_routes in by_family.items()},
        routes_by_id={route.id: route for route in routes},
    )


def test_route_query_json_round_trip_matches_native_query_shape() -> None:
    payload = {
        "op": "ADD",
        "tensors": {
            "src0": {
                "dtype": "F32",
                "dimensions": [4, 2],
                "strides": [1, 8],
                # The native parser accepts a permutation whose length differs from the rank.
                "permutation": [1, 0, 2],
            }
        },
        "attributes": {
            "nothing": None,
            "enabled": True,
            "minimum": -(1 << 63),
            "maximum": (1 << 63) - 1,
            "ratio": 1.25,
            "name": "example",
            "items": [False, 3, {"nested": "value"}],
        },
    }

    query = RouteQuery.from_json(payload)

    assert ROUTE_QUERY_SCHEMA == "ggml_hrx_kernel_bench.route_query.v1"
    assert query.operation == "ADD"
    assert query.tensors["src0"].permutation == (1, 0, 2)
    assert query.to_json() == payload
    assert route_query_to_json(route_query_from_json(payload)) == payload


def test_route_query_json_treats_missing_attributes_and_null_permutation_as_absent() -> None:
    query = route_query_from_json(
        {
            "op": "EMPTY",
            "tensors": {
                "src0": {
                    "dtype": "",
                    "dimensions": [],
                    "strides": [],
                    "permutation": None,
                }
            },
        }
    )

    assert query.attributes == {}
    assert query.tensors["src0"].permutation is None
    assert route_query_to_json(query) == {
        "op": "EMPTY",
        "tensors": {"src0": {"dtype": "", "dimensions": [], "strides": []}},
        "attributes": {},
    }


@pytest.mark.parametrize(
    ("payload", "message"),
    (
        ([], "route query must be an object"),
        ({"op": "ADD", "tensors": {}, "allowed_route_ids": []}, "unknown field"),
        ({"op": "ADD", "tensors": {}, "unexpected": True}, "unknown field"),
        ({"op": 7, "tensors": {}}, "field 'op' must be a string"),
        ({"op": "ADD", "tensors": [], "attributes": {}}, "field 'tensors' must be an object"),
        ({"op": "ADD", "tensors": {}, "attributes": None}, "field 'attributes' must be an object"),
        (
            {
                "op": "ADD",
                "tensors": {
                    "src0": {"dtype": "F32", "dimensions": [True], "strides": [1]}
                },
            },
            "must be a signed 64-bit integer",
        ),
        (
            {
                "op": "ADD",
                "tensors": {
                    "src0": {
                        "dtype": "F32",
                        "dimensions": [1 << 63],
                        "strides": [1],
                    }
                },
            },
            "outside the signed 64-bit integer range",
        ),
        (
            {
                "op": "ADD",
                "tensors": {
                    "src0": {"dtype": "F32", "dimensions": [1], "strides": []}
                },
            },
            "dimensions and strides must have equal length",
        ),
        (
            {"op": "ADD", "tensors": {}, "attributes": {"value": 1 << 63}},
            "outside the signed 64-bit integer range",
        ),
        (
            {"op": "ADD", "tensors": {}, "attributes": {"value": float("inf")}},
            "must be a finite floating-point number",
        ),
    ),
)
def test_route_query_json_rejects_non_native_query_values(
    payload: Any,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        route_query_from_json(payload)


def test_select_route_query_uses_first_catalog_match_before_later_exact_rank() -> None:
    range_route = _route(
        "range",
        constraints=(ConstraintCheck(name="dimensions", rank_min=1, rank_max=4),),
    )
    exact_route = _route(
        "exact",
        constraints=(ConstraintCheck(name="dimensions", length=1),),
    )
    query = RouteQuery(operation="test", tensors={"src0": _tensor()}, attributes={})

    selection = select_route_query(_catalog(range_route, exact_route), query)

    assert selection.status == "matched"
    assert selection.route_ids == ("range",)
    assert selection.candidate_route_ids == ("range",)


def test_select_route_query_resolves_overlapping_routes_to_first_catalog_match() -> None:
    first = _route("first", constraints=(ConstraintCheck(name="dimensions", length=1),))
    second = _route("second", constraints=(ConstraintCheck(name="dimensions", length=1),))

    selection = select_route_query(
        _catalog(first, second),
        RouteQuery(operation="TEST", tensors={"src0": _tensor()}, attributes={}),
    )

    assert selection.status == "matched"
    assert selection.route_ids == ("first",)
    assert selection.candidate_route_ids == ("first",)


def test_select_route_query_does_not_evaluate_routes_after_first_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _route("first")
    second = _route("second")
    evaluated_route_ids: list[str] = []

    def record_attribute_evaluation(route: V2Route, attributes: Mapping[str, Any]) -> bool:
        evaluated_route_ids.append(route.id)
        return True

    monkeypatch.setattr(
        selection_module,
        "route_accepts_attributes",
        record_attribute_evaluation,
    )

    selection = select_route_query(
        _catalog(first, second),
        RouteQuery(operation="TEST", tensors={"src0": _tensor()}, attributes={}),
    )

    assert selection.route_ids == ("first",)
    assert evaluated_route_ids == ["first"]


def test_materialize_route_query_tensors_copies_dimensions_and_builds_contiguous_strides() -> None:
    route = _route(
        "synthetic",
        tensors={
            "src0": TensorDescriptor("F32", "src_dimensions", "src_strides"),
            "mask": TensorDescriptor("F16", "mask_dimensions", "mask_strides"),
        },
        synthetic_tensors={
            "mask": SyntheticTensorDescriptor(
                dtype="F16",
                dimensions_source="src_dimensions",
            )
        },
    )
    query = RouteQuery(
        operation="TEST",
        tensors={"src0": _tensor(sizes=(2, 3, 4), strides=(12, 4, 1))},
        attributes={"mode": 1},
    )
    captured_tensors = dict(query.tensors)
    captured_attributes = dict(query.attributes)

    materialized = materialize_route_query_tensors(route, query)

    assert materialized is not None
    assert materialized["mask"].dtype == "F16"
    assert tuple(dimension.size for dimension in materialized["mask"].dimensions) == (2, 3, 4)
    assert tuple(dimension.stride for dimension in materialized["mask"].dimensions) == (1, 2, 6)
    assert dict(query.tensors) == captured_tensors
    assert dict(query.attributes) == captured_attributes
    assert "mask" not in query.tensors


def test_materialize_route_query_tensors_assembles_constants_and_indexed_captures() -> None:
    route = _route(
        "synthetic",
        tensors={
            "src0": TensorDescriptor("F32", "src_dimensions", "src_strides"),
            "dst": TensorDescriptor("F32", "dst_dimensions", "dst_strides"),
            "mask": TensorDescriptor("I32", "mask_dimensions", "mask_strides"),
        },
        synthetic_tensors={
            "mask": SyntheticTensorDescriptor(
                dtype="I32",
                dimensions_source=(
                    2,
                    {"source": "src_dimensions", "index": 1},
                    {"source": "dst_dimensions", "index": 0},
                ),
            )
        },
    )
    query = RouteQuery(
        operation="TEST",
        tensors={
            "src0": _tensor(sizes=(5, 7), strides=(1, 5)),
            "dst": _tensor(sizes=(11, 13), strides=(1, 11)),
        },
    )

    materialized = materialize_route_query_tensors(route, query)

    assert materialized is not None
    assert tuple(dimension.size for dimension in materialized["mask"].dimensions) == (2, 7, 11)
    assert tuple(dimension.stride for dimension in materialized["mask"].dimensions) == (1, 2, 14)


def test_select_route_query_materializes_synthetic_tensors_and_routes_attributes() -> None:
    route = _route(
        "synthetic",
        tensors={
            "src0": TensorDescriptor("F32", "src_dimensions", "src_strides"),
            "dst": TensorDescriptor("F32", "dst_dimensions", "dst_strides"),
            "mask": TensorDescriptor("F32", "mask_dimensions", "mask_strides"),
        },
        constraints=(
            ConstraintCheck(equals=("src_dimensions", "dst_dimensions", "mask_dimensions")),
            ConstraintCheck(name="attribute.mode", value=1, has_value=True),
        ),
        attributes={"mode": {"type": "i32"}},
        synthetic_tensors={
            "mask": SyntheticTensorDescriptor(dtype="F32", dimensions_source="dst_dimensions")
        },
    )
    catalog = _catalog(route)
    tensors = {"src0": _tensor(), "dst": _tensor()}

    matched = select_route_query(
        catalog,
        RouteQuery(
            operation="TEST",
            tensors=tensors,
            attributes={"mode": 1, "ignored_extra_attribute": True},
        ),
    )
    unmatched = select_route_query(
        catalog,
        RouteQuery(operation="TEST", tensors=tensors, attributes={"mode": 2}),
    )

    assert matched.status == "matched"
    assert matched.route_ids == ("synthetic",)
    assert unmatched.status == "unmatched"
    assert unmatched.route_ids == ()
    assert unmatched.candidate_route_ids == ()


def test_synthetic_tensor_dimensions_do_not_capture_route_attributes() -> None:
    route = _route(
        "synthetic",
        tensors={
            "src0": TensorDescriptor("F32", "src_dimensions", "src_strides"),
            "mask": TensorDescriptor("F32", "mask_dimensions", "mask_strides"),
        },
        attributes={"shape": {"type": "i32[]"}},
        synthetic_tensors={
            "mask": SyntheticTensorDescriptor(
                dtype="F32",
                dimensions_source="attribute.shape",
            )
        },
    )
    query = RouteQuery(
        operation="TEST",
        tensors={"src0": _tensor()},
        attributes={"shape": [2, 3]},
    )

    assert materialize_route_query_tensors(route, query) is None
    assert select_route_query(_catalog(route), query).status == "unmatched"
    assert query.attributes == {"shape": [2, 3]}
    assert set(query.tensors) == {"src0"}


def test_select_route_query_reports_unknown_operation_and_empty_tensors_as_unmatched() -> None:
    selection = select_route_query(
        _catalog(_route("only-route")),
        RouteQuery(operation="UNKNOWN", tensors={}, attributes={}),
    )

    assert selection.status == "unmatched"
    assert selection.route_ids == ()
    assert selection.candidate_route_ids == ()
