from __future__ import annotations

from typing import Any, Mapping

from .models import (
    ConcreteTensor,
    ConcreteTensorDimension,
    RouteConstraints,
    StrideDescriptor,
    TensorDescriptor,
    TensorDimensionDescriptor,
    V2Route,
)
from .shape import normalize_shape


def _normalize_dtype(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text.upper()


def route_accepts_dtype(route: V2Route, dtype: Mapping[str, Any]) -> bool:
    if "type" in dtype:
        actual = _normalize_dtype(dtype["type"])
        return all(descriptor.dtype in {None, actual} for descriptor in route.tensors.values())
    for tensor_name, keys in (
        ("src0", ("type_src0", "type_src")),
        ("src1", ("type_src1", "type_src")),
        ("dst", ("type_dst",)),
    ):
        descriptor = route.tensors.get(tensor_name)
        expected = None if descriptor is None else descriptor.dtype
        if expected is None:
            continue
        provided = next((dtype[key] for key in keys if key in dtype), None)
        if provided is not None and _normalize_dtype(provided) != expected:
            return False
    return True


def _bounds_accept(lower: int | None, upper: int | None, value: int) -> bool:
    if lower is not None and value < lower:
        return False
    if upper is not None and value > upper:
        return False
    return True


def _dimension_sizes(tensor: ConcreteTensor) -> dict[str, int]:
    return {dimension.name: int(dimension.size) for dimension in tensor.dimensions}


def _evaluate_stride(descriptor: StrideDescriptor | None, sizes: Mapping[str, int]) -> int | None:
    if descriptor is None:
        return None
    if descriptor.value is not None:
        return int(descriptor.value)
    if descriptor.dimension is not None:
        value = sizes.get(descriptor.dimension)
        return None if value is None else int(value)
    if descriptor.product:
        value = 1
        for name in descriptor.product:
            size = sizes.get(name)
            if size is None:
                return None
            value *= int(size)
        return value
    return None


def tensor_accepts_descriptor(
    descriptor: TensorDescriptor,
    constraints: RouteConstraints,
    tensor: ConcreteTensor,
) -> bool:
    if descriptor.dtype is not None and _normalize_dtype(tensor.dtype) != descriptor.dtype:
        return False
    if len(descriptor.dimensions) != len(tensor.dimensions):
        return False
    if len(descriptor.stride_ids) != len(descriptor.dimensions):
        return False
    sizes = _dimension_sizes(tensor)
    for expected, stride_id, actual in zip(
        descriptor.dimensions,
        descriptor.stride_ids,
        tensor.dimensions,
        strict=True,
    ):
        if expected.name != actual.name:
            return False
        bounds = constraints.sizes.get(expected.name)
        if bounds is not None and not _bounds_accept(bounds.min, bounds.max, int(actual.size)):
            return False
        expected_stride = _evaluate_stride(constraints.strides.get(stride_id.name), sizes)
        if expected_stride is not None and int(actual.stride) != expected_stride:
            return False
    return True


def route_accepts_tensors(route: V2Route, tensors: Mapping[str, ConcreteTensor]) -> bool:
    for tensor_name, descriptor in route.tensors.items():
        tensor = tensors.get(tensor_name)
        if tensor is None or not tensor_accepts_descriptor(descriptor, route.constraints, tensor):
            return False
    return True


def shape_from_tensors(tensors: Mapping[str, ConcreteTensor]) -> dict[str, int]:
    shape: dict[str, int] = {}
    for tensor in tensors.values():
        for dimension in tensor.dimensions:
            shape.setdefault(dimension.name, int(dimension.size))
    return normalize_shape(shape)


def _default_dimension_size(
    dimension: TensorDimensionDescriptor,
    constraints: RouteConstraints,
) -> int:
    bounds = constraints.sizes.get(dimension.name)
    if bounds is not None and bounds.min is not None:
        return int(bounds.min)
    return 1


def default_shape_for_route(route: V2Route) -> dict[str, int]:
    shape: dict[str, int] = {}
    for descriptor in route.tensors.values():
        for dimension in descriptor.dimensions:
            current = shape.get(dimension.name)
            default = _default_dimension_size(dimension, route.constraints)
            shape[dimension.name] = default if current is None else max(current, default)
    return normalize_shape(shape)


def materialize_route_tensors(route: V2Route, sizes: Mapping[str, int]) -> dict[str, ConcreteTensor]:
    materialized: dict[str, ConcreteTensor] = {}
    normalized_sizes = {str(name): int(value) for name, value in sizes.items()}
    for tensor_name, descriptor in route.tensors.items():
        tensor_dimensions: list[ConcreteTensorDimension] = []
        local_sizes: dict[str, int] = {}
        for dimension in descriptor.dimensions:
            if dimension.name not in normalized_sizes:
                raise KeyError(dimension.name)
            local_sizes[dimension.name] = int(normalized_sizes[dimension.name])
        if len(descriptor.stride_ids) != len(descriptor.dimensions):
            raise RuntimeError(
                f"v2 tensor {tensor_name!r} has mismatched dimension and stride identifiers"
            )
        for dimension, stride_id in zip(descriptor.dimensions, descriptor.stride_ids, strict=True):
            stride = _evaluate_stride(route.constraints.strides.get(stride_id.name), local_sizes)
            tensor_dimensions.append(
                ConcreteTensorDimension(
                    name=dimension.name,
                    size=local_sizes[dimension.name],
                    stride=0 if stride is None else stride,
                )
            )
        materialized[tensor_name] = ConcreteTensor(
            dtype=descriptor.dtype or "",
            dimensions=tuple(tensor_dimensions),
        )
    return materialized


def value_from_tensor_source(
    source: str,
    tensors: Mapping[str, ConcreteTensor],
) -> int | str | None:
    parts = source.split(".")
    if len(parts) == 3 and parts[0] == "tensor" and parts[2] == "dtype":
        tensor = tensors.get(parts[1])
        return None if tensor is None else tensor.dtype
    if len(parts) != 5 or parts[0] != "tensor" or parts[2] != "dimensions":
        return None
    tensor = tensors.get(parts[1])
    if tensor is None:
        return None
    dimension_name = parts[3]
    field = parts[4]
    for dimension in tensor.dimensions:
        if dimension.name != dimension_name:
            continue
        if field == "size":
            return int(dimension.size)
        if field == "stride":
            return int(dimension.stride)
        return None
    return None


def _ceil_div(lhs: int, rhs: int) -> int:
    return (lhs + rhs - 1) // rhs


def route_dispatch(route: V2Route, shape: Mapping[str, int]) -> dict[str, Any]:
    normalized_shape = normalize_shape(shape)
    rows_per_workgroup = int(route.launch.get("rows_per_workgroup", 1) or 1)
    cols_per_workgroup = int(route.launch.get("cols_per_workgroup", 1) or 1)
    nrows = int(normalized_shape.get("nrows", normalized_shape.get("rows", 1)))
    ncols = int(normalized_shape.get("ncols", normalized_shape.get("cols", 1)))
    return {
        "workgroup_count": [
            _ceil_div(nrows, rows_per_workgroup),
            _ceil_div(ncols, cols_per_workgroup),
            1,
        ],
        "workgroup_size": list(route.launch.get("workgroup_size", [None, None, None])),
        "rows_per_workgroup": rows_per_workgroup,
        "cols_per_workgroup": cols_per_workgroup,
        "metadata_source": "route_descriptor_v2",
        "has_static_dispatch_workgroup_count": False,
        "has_static_workgroup_size": bool(route.launch.get("workgroup_size")),
    }
