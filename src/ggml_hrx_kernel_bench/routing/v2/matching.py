from __future__ import annotations

from typing import Any, Mapping

from .models import (
    ConcreteTensor,
    ConcreteTensorDimension,
    ConstraintCheck,
    RouteConstraints,
    TensorDescriptor,
    V2Route,
    ValueDefinition,
)
from .shape import normalize_shape


CapturedValue = tuple[int, ...] | int


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


def _multiple_of_accept(divisor: int | None, value: int) -> bool:
    if divisor is None:
        return True
    return value % divisor == 0


def _contiguous_strides(dimensions: tuple[int, ...]) -> tuple[int, ...]:
    stride = 1
    strides: list[int] = []
    for size in dimensions:
        strides.append(stride)
        stride *= int(size)
    return tuple(strides)


def _product(dimensions: tuple[int, ...]) -> int:
    total = 1
    for size in dimensions:
        total *= int(size)
    return total


def _store_capture(
    captures: dict[str, CapturedValue],
    name: str,
    values: CapturedValue,
) -> bool:
    current = captures.get(name)
    if current is None:
        captures[name] = values
        return True
    return current == values


def _capture_tensor(descriptor: TensorDescriptor, tensor: ConcreteTensor, captures: dict[str, CapturedValue]) -> bool:
    if descriptor.dtype is not None and _normalize_dtype(tensor.dtype) != descriptor.dtype:
        return False
    dimensions = tuple(int(dimension.size) for dimension in tensor.dimensions)
    strides = tuple(int(dimension.stride) for dimension in tensor.dimensions)
    if not _store_capture(captures, descriptor.dimensions_capture, dimensions):
        return False
    if not _store_capture(captures, descriptor.strides_capture, strides):
        return False
    return True


def _resolve_values(
    definitions: tuple[ValueDefinition, ...],
    captures: Mapping[str, CapturedValue],
) -> dict[str, CapturedValue] | None:
    resolved: dict[str, CapturedValue] = {}
    for definition in definitions:
        if definition.contiguous_strides is not None:
            source = captures.get(definition.contiguous_strides)
            if not isinstance(source, tuple):
                return None
            resolved[definition.name] = _contiguous_strides(source)
            continue
        if definition.product is not None:
            source = captures.get(definition.product)
            if not isinstance(source, tuple):
                return None
            resolved[definition.name] = _product(source)
            continue
        if definition.name:
            return None
    return resolved


def _constraint_accepts(check: ConstraintCheck, values: Mapping[str, CapturedValue]) -> bool:
    if check.equals:
        first = values.get(check.equals[0])
        if first is None:
            return False
        return all(values.get(name) == first for name in check.equals[1:])
    if check.name is None:
        return False
    captured = values.get(check.name)
    if captured is None:
        return False
    if check.length is not None:
        if not isinstance(captured, tuple):
            return False
        return len(captured) == check.length
    if check.index is not None:
        if not isinstance(captured, tuple):
            return False
        if check.index < 0 or check.index >= len(captured):
            return False
        value = int(captured[check.index])
        return _bounds_accept(check.min, check.max, value) and _multiple_of_accept(check.multiple_of, value)
    if isinstance(captured, tuple):
        return False
    value = int(captured)
    return _bounds_accept(check.min, check.max, value) and _multiple_of_accept(check.multiple_of, value)


def constraints_accept(route_constraints: RouteConstraints, values: Mapping[str, CapturedValue]) -> bool:
    return all(_constraint_accepts(check, values) for check in route_constraints.checks)


def tensor_accepts_descriptor(
    descriptor: TensorDescriptor,
    constraints: RouteConstraints,
    tensor: ConcreteTensor,
    *,
    computed_values: tuple[ValueDefinition, ...] = (),
) -> bool:
    captures: dict[str, CapturedValue] = {}
    if not _capture_tensor(descriptor, tensor, captures):
        return False
    resolved = _resolve_values(computed_values, captures)
    if resolved is None:
        return False
    return constraints_accept(constraints, {**captures, **resolved})


def route_accepts_tensors(route: V2Route, tensors: Mapping[str, ConcreteTensor]) -> bool:
    captures: dict[str, CapturedValue] = {}
    for tensor_name, descriptor in route.tensors.items():
        tensor = tensors.get(tensor_name)
        if tensor is None or not _capture_tensor(descriptor, tensor, captures):
            return False
    resolved = _resolve_values(route.values, captures)
    if resolved is None:
        return False
    return constraints_accept(route.constraints, {**captures, **resolved})


def route_values(route: V2Route, tensors: Mapping[str, ConcreteTensor]) -> dict[str, CapturedValue] | None:
    captures: dict[str, CapturedValue] = {}
    for tensor_name, descriptor in route.tensors.items():
        tensor = tensors.get(tensor_name)
        if tensor is None or not _capture_tensor(descriptor, tensor, captures):
            return None
    resolved = _resolve_values(route.values, captures)
    if resolved is None:
        return None
    return {**captures, **resolved}


def shape_from_tensors(tensors: Mapping[str, ConcreteTensor]) -> dict[str, int]:
    shape: dict[str, int] = {}
    for tensor in tensors.values():
        for dimension in tensor.dimensions:
            shape.setdefault(dimension.name, int(dimension.size))
    return normalize_shape(shape)


def _constraint_for_capture(
    route: V2Route,
    capture_name: str,
) -> tuple[int | None, dict[int, tuple[int | None, int | None]]]:
    rank: int | None = None
    bounds: dict[int, tuple[int | None, int | None]] = {}
    for check in route.constraints.checks:
        if check.name != capture_name:
            continue
        if check.length is not None:
            rank = check.length
            continue
        if check.index is not None:
            bounds[check.index] = (check.min, check.max)
    return rank, bounds


def _shape_capture_for_route(route: V2Route) -> tuple[str, int, dict[int, tuple[int | None, int | None]]]:
    for descriptor in route.tensors.values():
        rank, bounds = _constraint_for_capture(route, descriptor.dimensions_capture)
        if rank is not None:
            return descriptor.dimensions_capture, rank, bounds
    raise RuntimeError(f"v2 route {route.id!r} does not constrain tensor rank")


def _scalar_constraint_min(route: V2Route, capture_name: str) -> int | None:
    minimum: int | None = None
    for check in route.constraints.checks:
        if check.name != capture_name or check.index is not None or check.length is not None:
            continue
        if check.min is None:
            continue
        minimum = check.min if minimum is None else max(minimum, check.min)
    return minimum


def default_shape_for_route(route: V2Route) -> dict[str, int]:
    if not route.tensors:
        return {}
    try:
        _, rank, bounds = _shape_capture_for_route(route)
    except RuntimeError:
        total_size_min = _scalar_constraint_min(route, "total_size")
        default_cols = total_size_min if total_size_min is not None else int(route.launch.get("workgroup_size", [1])[0] or 1)
        return normalize_shape({"ncols": default_cols, "nrows": 1})
    if rank != 2:
        raise RuntimeError(f"v2 route {route.id!r} requires unsupported rank {rank!r} for default shape")
    ncols_min = bounds.get(0, (None, None))[0]
    nrows_min = bounds.get(1, (None, None))[0]
    return normalize_shape(
        {
            "ncols": 1 if ncols_min is None else int(ncols_min),
            "nrows": 1 if nrows_min is None else int(nrows_min),
        }
    )


def materialize_route_tensors(route: V2Route, sizes: Mapping[str, int]) -> dict[str, ConcreteTensor]:
    try:
        _, rank, _ = _shape_capture_for_route(route)
    except RuntimeError:
        rank = 2
    if rank != 2:
        raise RuntimeError(f"v2 route {route.id!r} requires unsupported rank {rank!r} for materialization")
    normalized_sizes = normalize_shape({str(name): int(value) for name, value in sizes.items()})
    if "ncols" not in normalized_sizes or "nrows" not in normalized_sizes:
        raise KeyError("ncols/nrows")
    dimensions = (
        ConcreteTensorDimension(name="ncols", size=int(normalized_sizes["ncols"]), stride=1),
        ConcreteTensorDimension(
            name="nrows",
            size=int(normalized_sizes["nrows"]),
            stride=int(normalized_sizes["ncols"]),
        ),
    )
    return {
        tensor_name: ConcreteTensor(dtype=descriptor.dtype or "", dimensions=dimensions)
        for tensor_name, descriptor in route.tensors.items()
    }


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


def value_from_route_source(
    source: str,
    values: Mapping[str, CapturedValue],
) -> int | tuple[int, ...] | None:
    if not source.startswith("value."):
        return None
    return values.get(source.removeprefix("value."))


def _ceil_div(lhs: int, rhs: int) -> int:
    return (lhs + rhs - 1) // rhs


def route_dispatch(
    route: V2Route,
    shape: Mapping[str, int],
    *,
    values: Mapping[str, CapturedValue] | None = None,
) -> dict[str, Any]:
    normalized_shape = normalize_shape(shape)
    workgroup_size = list(route.launch.get("workgroup_size", [None, None, None]))
    lane_count = int(workgroup_size[0] or 1)
    total_size = None
    if values is not None:
        resolved = values.get("total_size")
        if isinstance(resolved, int):
            total_size = resolved
    if total_size is not None:
        workgroup_count = [_ceil_div(total_size, lane_count), 1, 1]
    else:
        rows_per_workgroup = int(route.launch.get("rows_per_workgroup", 1) or 1)
        cols_per_workgroup = int(route.launch.get("cols_per_workgroup", 1) or 1)
        nrows = int(normalized_shape.get("nrows", normalized_shape.get("rows", 1)))
        ncols = int(normalized_shape.get("ncols", normalized_shape.get("cols", 1)))
        workgroup_count = [
            _ceil_div(nrows, rows_per_workgroup),
            _ceil_div(ncols, cols_per_workgroup),
            1,
        ]
    return {
        "workgroup_count": workgroup_count,
        "workgroup_size": workgroup_size,
        "rows_per_workgroup": int(route.launch.get("rows_per_workgroup", 1) or 1),
        "cols_per_workgroup": int(route.launch.get("cols_per_workgroup", 1) or 1),
        "metadata_source": "route_descriptor_v2",
        "has_static_dispatch_workgroup_count": False,
        "has_static_workgroup_size": bool(route.launch.get("workgroup_size")),
    }
