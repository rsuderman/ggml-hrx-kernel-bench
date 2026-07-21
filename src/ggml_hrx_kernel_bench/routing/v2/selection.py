from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

from .layout import contiguous_strides
from .matching import (
    route_accepts_attributes,
    route_accepts_tensors,
    route_captures,
)
from .models import ConcreteTensor, ConcreteTensorDimension, V2Route
from .query import RouteCatalog, routes_for_op


ROUTE_QUERY_SCHEMA = "ggml_hrx_kernel_bench.route_query.v1"

RouteSelectionStatus = Literal["matched", "unmatched"]

_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1


@dataclass(frozen=True)
class RouteQuery:
    operation: str
    tensors: Mapping[str, ConcreteTensor]
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tensors", MappingProxyType(dict(self.tensors)))
        object.__setattr__(self, "attributes", MappingProxyType(dict(self.attributes)))

    @classmethod
    def from_json(cls, payload: Any) -> RouteQuery:
        return route_query_from_json(payload)

    def to_json(self) -> dict[str, Any]:
        return route_query_to_json(self)


@dataclass(frozen=True)
class RouteSelection:
    status: RouteSelectionStatus
    route_ids: tuple[str, ...]
    candidate_route_ids: tuple[str, ...]


def _expect_object_fields(
    payload: Any,
    *,
    path: str,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must be an object")
    non_string_keys = [key for key in payload if not isinstance(key, str)]
    if non_string_keys:
        raise ValueError(f"{path} keys must be strings")
    keys = set(payload)
    unknown = keys - required - optional
    if unknown:
        raise ValueError(f"{path} contains unknown field {sorted(unknown)[0]!r}")
    missing = required - keys
    if missing:
        raise ValueError(f"{path} is missing required field {sorted(missing)[0]!r}")
    return payload


def _parse_int64_array(value: Any, *, path: str, field_name: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{path} field {field_name!r} must be an array")
    parsed: list[int] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(
                f"{path} field {field_name!r} element {index} must be a signed 64-bit integer"
            )
        if item < _INT64_MIN or item > _INT64_MAX:
            raise ValueError(
                f"{path} field {field_name!r} element {index} is outside the signed 64-bit "
                "integer range"
            )
        parsed.append(int(item))
    return tuple(parsed)


def _parse_tensor(value: Any, *, role: str) -> ConcreteTensor:
    path = f"input tensor {role!r}"
    tensor = _expect_object_fields(
        value,
        path=path,
        required=frozenset({"dtype", "dimensions", "strides"}),
        optional=frozenset({"permutation"}),
    )
    dtype = tensor["dtype"]
    if not isinstance(dtype, str):
        raise ValueError(f"{path} field 'dtype' must be a string")
    dimensions = _parse_int64_array(
        tensor["dimensions"],
        path=path,
        field_name="dimensions",
    )
    strides = _parse_int64_array(
        tensor["strides"],
        path=path,
        field_name="strides",
    )
    if len(dimensions) != len(strides):
        raise ValueError(f"{path} dimensions and strides must have equal length")
    raw_permutation = tensor.get("permutation")
    permutation = (
        None
        if raw_permutation is None
        else _parse_int64_array(raw_permutation, path=path, field_name="permutation")
    )
    return ConcreteTensor(
        dtype=dtype,
        dimensions=tuple(
            ConcreteTensorDimension(name=f"d{index}", size=size, stride=strides[index])
            for index, size in enumerate(dimensions)
        ),
        permutation=permutation,
    )


def _parse_attribute_value(value: Any, *, path: str) -> Any:
    if value is None or isinstance(value, bool | str):
        return value
    if isinstance(value, int):
        if value < _INT64_MIN or value > _INT64_MAX:
            raise ValueError(f"{path} is outside the signed 64-bit integer range")
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must be a finite floating-point number")
        return float(value)
    if isinstance(value, list):
        return [
            _parse_attribute_value(item, path=f"{path} element {index}")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        non_string_keys = [key for key in value if not isinstance(key, str)]
        if non_string_keys:
            raise ValueError(f"{path} object keys must be strings")
        return {
            key: _parse_attribute_value(item, path=f"{path} field {key!r}")
            for key, item in value.items()
        }
    raise ValueError(f"{path} has an unsupported JSON type")


def route_query_from_json(payload: Any) -> RouteQuery:
    root = _expect_object_fields(
        payload,
        path="route query",
        required=frozenset({"op", "tensors"}),
        optional=frozenset({"attributes"}),
    )
    operation = root["op"]
    if not isinstance(operation, str):
        raise ValueError("route query field 'op' must be a string")
    raw_tensors = root["tensors"]
    if not isinstance(raw_tensors, Mapping):
        raise ValueError("route query field 'tensors' must be an object")
    if any(not isinstance(role, str) for role in raw_tensors):
        raise ValueError("route query tensor names must be strings")
    tensors = {
        role: _parse_tensor(tensor, role=role)
        for role, tensor in raw_tensors.items()
    }
    raw_attributes = root.get("attributes", {})
    if not isinstance(raw_attributes, Mapping):
        raise ValueError("route query field 'attributes' must be an object")
    if any(not isinstance(name, str) for name in raw_attributes):
        raise ValueError("route query attribute names must be strings")
    attributes = {
        name: _parse_attribute_value(value, path=f"input attribute {name!r}")
        for name, value in raw_attributes.items()
    }
    return RouteQuery(operation=operation, tensors=tensors, attributes=attributes)


def _serialize_int64(value: Any, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be a signed 64-bit integer")
    if value < _INT64_MIN or value > _INT64_MAX:
        raise ValueError(f"{path} is outside the signed 64-bit integer range")
    return int(value)


def route_query_to_json(query: RouteQuery) -> dict[str, Any]:
    if not isinstance(query.operation, str):
        raise ValueError("route query operation must be a string")
    tensors: dict[str, dict[str, Any]] = {}
    for role, tensor in query.tensors.items():
        if not isinstance(role, str):
            raise ValueError("route query tensor names must be strings")
        if not isinstance(tensor, ConcreteTensor):
            raise ValueError(f"route query tensor {role!r} must be a ConcreteTensor")
        if not isinstance(tensor.dtype, str):
            raise ValueError(f"route query tensor {role!r} dtype must be a string")
        serialized: dict[str, Any] = {
            "dtype": tensor.dtype,
            "dimensions": [
                _serialize_int64(dimension.size, path=f"input tensor {role!r} dimension {index}")
                for index, dimension in enumerate(tensor.dimensions)
            ],
            "strides": [
                _serialize_int64(dimension.stride, path=f"input tensor {role!r} stride {index}")
                for index, dimension in enumerate(tensor.dimensions)
            ],
        }
        if tensor.permutation is not None:
            serialized["permutation"] = [
                _serialize_int64(axis, path=f"input tensor {role!r} permutation {index}")
                for index, axis in enumerate(tensor.permutation)
            ]
        tensors[role] = serialized
    if any(not isinstance(name, str) for name in query.attributes):
        raise ValueError("route query attribute names must be strings")
    attributes = {
        name: _parse_attribute_value(value, path=f"input attribute {name!r}")
        for name, value in query.attributes.items()
    }
    return {
        "op": query.operation,
        "tensors": tensors,
        "attributes": attributes,
    }


def _synthetic_dimension_sizes(
    dimensions_source: Any,
    captures: Mapping[str, Any],
) -> tuple[int, ...] | None:
    if isinstance(dimensions_source, str):
        dimensions = captures.get(dimensions_source)
        if not isinstance(dimensions, tuple):
            return None
        return tuple(int(value) for value in dimensions)
    if not isinstance(dimensions_source, tuple):
        return None
    sizes: list[int] = []
    for entry in dimensions_source:
        if isinstance(entry, int) and not isinstance(entry, bool):
            sizes.append(int(entry))
            continue
        if not isinstance(entry, Mapping):
            return None
        source = entry.get("source")
        index = entry.get("index")
        if not isinstance(source, str) or not isinstance(index, int) or isinstance(index, bool):
            return None
        source_dimensions = captures.get(source)
        if not isinstance(source_dimensions, tuple) or index < 0 or index >= len(source_dimensions):
            return None
        sizes.append(int(source_dimensions[index]))
    return tuple(sizes)


def _route_tensors_for_query(
    route: V2Route,
    tensors: Mapping[str, ConcreteTensor],
) -> dict[str, ConcreteTensor] | None:
    route_tensors = dict(tensors)
    if not route.synthetic_tensors:
        return route_tensors
    real_tensor_names = set(route.tensors) - set(route.synthetic_tensors)
    captures = route_captures(route, route_tensors, tensor_names=real_tensor_names)
    if captures is None:
        return None
    for tensor_name, synthetic in route.synthetic_tensors.items():
        sizes = _synthetic_dimension_sizes(synthetic.dimensions_source, captures)
        if sizes is None:
            return None
        strides = contiguous_strides(sizes)
        route_tensors[tensor_name] = ConcreteTensor(
            dtype=synthetic.dtype,
            dimensions=tuple(
                ConcreteTensorDimension(
                    name=f"d{index}",
                    size=sizes[index],
                    stride=strides[index],
                )
                for index in range(len(sizes))
            ),
        )
    return route_tensors


def materialize_route_query_tensors(
    route: V2Route,
    query: RouteQuery,
) -> dict[str, ConcreteTensor] | None:
    return _route_tensors_for_query(route, query.tensors)


def select_route_query(catalog: RouteCatalog, query: RouteQuery) -> RouteSelection:
    tensor_names = set(query.tensors)
    for route in routes_for_op(catalog, query.operation):
        required_tensor_names = set(route.tensors) - set(route.synthetic_tensors)
        if required_tensor_names != tensor_names or not route_accepts_attributes(
            route,
            query.attributes,
        ):
            continue
        route_tensors = materialize_route_query_tensors(route, query)
        if route_tensors is None or not route_accepts_tensors(
            route,
            route_tensors,
            query.attributes,
        ):
            continue
        return RouteSelection(
            status="matched",
            route_ids=(route.id,),
            candidate_route_ids=(route.id,),
        )
    return RouteSelection(
        status="unmatched",
        route_ids=(),
        candidate_route_ids=(),
    )
