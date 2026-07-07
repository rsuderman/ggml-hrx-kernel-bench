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


def _inverse_permutation(permutation: tuple[int, ...]) -> tuple[int, ...] | None:
    if tuple(sorted(permutation)) != tuple(range(len(permutation))):
        return None
    inverse = [0] * len(permutation)
    for index, value in enumerate(permutation):
        inverse[int(value)] = int(index)
    return tuple(inverse)


def _chain_permutations(first: tuple[int, ...], second: tuple[int, ...]) -> tuple[int, ...] | None:
    if len(first) != len(second):
        return None
    if tuple(sorted(first)) != tuple(range(len(first))):
        return None
    if tuple(sorted(second)) != tuple(range(len(second))):
        return None
    return tuple(int(second[index]) for index in first)


def _permuted_contiguous_strides(
    dimensions: tuple[int, ...],
    permutation: tuple[int, ...],
) -> tuple[int, ...] | None:
    if len(dimensions) != len(permutation):
        return None
    if tuple(sorted(permutation)) != tuple(range(len(permutation))):
        return None
    base_extents = [int(dimensions[axis]) for axis in permutation]
    base_strides = _contiguous_strides(tuple(base_extents))
    logical_strides = [0] * len(permutation)
    for base_axis, logical_axis in enumerate(permutation):
        logical_strides[int(logical_axis)] = int(base_strides[base_axis])
    return tuple(logical_strides)


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
    permutation = tensor.permutation
    if permutation is None:
        permutation = tuple(range(len(dimensions)))
    elif len(permutation) != len(dimensions):
        return False
    if not _store_capture(captures, descriptor.dimensions_capture, dimensions):
        return False
    if not _store_capture(captures, descriptor.strides_capture, strides):
        return False
    if descriptor.permutation_capture is not None and not _store_capture(
        captures,
        descriptor.permutation_capture,
        tuple(int(axis) for axis in permutation),
    ):
        return False
    return True


def _resolve_values(
    definitions: tuple[ValueDefinition, ...],
    captures: Mapping[str, CapturedValue],
) -> dict[str, CapturedValue] | None:
    resolved: dict[str, CapturedValue] = {}

    def lookup(name: str) -> CapturedValue | None:
        if name in resolved:
            return resolved[name]
        return captures.get(name)

    for definition in definitions:
        if definition.contiguous_strides is not None:
            source = lookup(definition.contiguous_strides)
            if not isinstance(source, tuple):
                return None
            resolved[definition.name] = _contiguous_strides(source)
            continue
        if definition.product is not None:
            source = lookup(definition.product)
            if not isinstance(source, tuple):
                return None
            resolved[definition.name] = _product(source)
            continue
        if definition.inverse_permutation is not None:
            source = lookup(definition.inverse_permutation)
            if not isinstance(source, tuple):
                return None
            inverse = _inverse_permutation(source)
            if inverse is None:
                return None
            resolved[definition.name] = inverse
            continue
        if definition.chain_permutations is not None:
            first = lookup(definition.chain_permutations[0])
            second = lookup(definition.chain_permutations[1])
            if not isinstance(first, tuple) or not isinstance(second, tuple):
                return None
            chained = _chain_permutations(first, second)
            if chained is None:
                return None
            resolved[definition.name] = chained
            continue
        if definition.permuted_contiguous_strides_dimensions is not None:
            dimensions = lookup(definition.permuted_contiguous_strides_dimensions)
            permutation = lookup(definition.permuted_contiguous_strides_permutation or "")
            if not isinstance(dimensions, tuple) or not isinstance(permutation, tuple):
                return None
            strides = _permuted_contiguous_strides(dimensions, permutation)
            if strides is None:
                return None
            resolved[definition.name] = strides
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
    if check.divides:
        first = values.get(check.divides[0])
        if not isinstance(first, tuple):
            return False
        for name in check.divides[1:]:
            current = values.get(name)
            if not isinstance(current, tuple) or len(current) != len(first):
                return False
            for divisor, value in zip(first, current):
                if divisor <= 0 or value % divisor != 0:
                    return False
        return True
    if check.name is None:
        return False
    captured = values.get(check.name)
    if captured is None:
        return False
    if check.iota:
        if not isinstance(captured, tuple):
            return False
        return captured == tuple(range(len(captured)))
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
    return shape


def shape_overrides_from_tensors(tensors: Mapping[str, ConcreteTensor]) -> dict[str, int]:
    shape = shape_from_tensors(tensors)
    overrides: dict[str, int] = {}
    for tensor_name, tensor in tensors.items():
        sizes = [int(dimension.size) for dimension in tensor.dimensions]
        default_strides = _contiguous_strides(tuple(sizes))
        for index, dimension in enumerate(tensor.dimensions):
            base_key = f"{tensor_name}_{dimension.name}"
            if int(dimension.size) != int(shape[dimension.name]):
                overrides[base_key] = int(dimension.size)
            if int(dimension.stride) != int(default_strides[index]):
                overrides[f"{base_key}_stride"] = int(dimension.stride)
    return overrides


def shape_permutations_for_route(
    route: V2Route,
    tensors: Mapping[str, ConcreteTensor],
) -> dict[str, int]:
    overrides: dict[str, int] = {}
    for tensor_name, descriptor in route.tensors.items():
        if descriptor.permutation_capture is None:
            continue
        tensor = tensors.get(tensor_name)
        if tensor is None or tensor.permutation is None:
            continue
        if tensor.permutation == tuple(range(len(tensor.permutation))):
            continue
        for index, axis in enumerate(tensor.permutation):
            overrides[f"{tensor_name}_perm{index}"] = int(axis)
    return overrides


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


def _rank_dimension_names(rank: int) -> tuple[str, ...]:
    return tuple(f"d{index}" for index in range(rank))


def default_shape_for_route(route: V2Route) -> dict[str, int]:
    if not route.tensors:
        return {}
    try:
        _, rank, bounds = _shape_capture_for_route(route)
    except RuntimeError:
        total_size_min = _scalar_constraint_min(route, "total_size")
        default_cols = total_size_min if total_size_min is not None else int(route.launch.get("workgroup_size", [1])[0] or 1)
        return {"d0": default_cols, "d1": 1}
    if rank <= 0:
        raise RuntimeError(f"v2 route {route.id!r} requires unsupported rank {rank!r} for default shape")
    return {
        name: 1 if bounds.get(index, (None, None))[0] is None else int(bounds[index][0])
        for index, name in enumerate(_rank_dimension_names(rank))
    }


def materialize_route_tensors(route: V2Route, sizes: Mapping[str, int]) -> dict[str, ConcreteTensor]:
    normalized_sizes = {str(name): int(value) for name, value in sizes.items()}
    try:
        _, rank, _ = _shape_capture_for_route(route)
    except RuntimeError:
        if "d0" not in normalized_sizes:
            raise KeyError("d0")
        ranked_dimension_names = sorted(
            (
                (int(name[1:]), name)
                for name in normalized_sizes
                if name.startswith("d") and name[1:].isdigit()
            ),
            key=lambda item: item[0],
        )
        if not ranked_dimension_names:
            raise KeyError("d0")
        dimension_names = tuple(name for _, name in ranked_dimension_names)
        if tuple(index for index, _ in ranked_dimension_names) != tuple(range(len(dimension_names))):
            raise KeyError(",".join(f"d{index}" for index in range(len(dimension_names))))
        default_sizes = [int(normalized_sizes[name]) for name in dimension_names]
        default_strides = _contiguous_strides(tuple(default_sizes))
        dimensions = tuple(
            ConcreteTensorDimension(name=name, size=default_sizes[index], stride=default_strides[index])
            for index, name in enumerate(dimension_names)
        )
        return {
            tensor_name: ConcreteTensor(dtype=descriptor.dtype or "", dimensions=dimensions)
            for tensor_name, descriptor in route.tensors.items()
        }
    if rank <= 0:
        raise RuntimeError(f"v2 route {route.id!r} requires unsupported rank {rank!r} for materialization")
    dimension_names = _rank_dimension_names(rank)
    missing = [name for name in dimension_names if name not in normalized_sizes]
    if missing:
        raise KeyError(",".join(missing))
    tensors: dict[str, ConcreteTensor] = {}
    for tensor_name, descriptor in route.tensors.items():
        sizes_for_tensor = [
            int(normalized_sizes.get(f"{tensor_name}_{dimension_name}", normalized_sizes[dimension_name]))
            for dimension_name in dimension_names
        ]
        default_strides = _contiguous_strides(tuple(sizes_for_tensor))
        dimensions: list[ConcreteTensorDimension] = []
        for index, dimension_name in enumerate(dimension_names):
            stride = int(normalized_sizes.get(f"{tensor_name}_{dimension_name}_stride", default_strides[index]))
            dimensions.append(
                ConcreteTensorDimension(
                    name=dimension_name,
                    size=sizes_for_tensor[index],
                    stride=stride,
                )
            )
        permutation = None
        if descriptor.permutation_capture is not None:
            permutation_keys = [f"{tensor_name}_perm{index}" for index in range(rank)]
            present_keys = [key for key in permutation_keys if key in normalized_sizes]
            if present_keys:
                missing_permutation_keys = [key for key in permutation_keys if key not in normalized_sizes]
                if missing_permutation_keys:
                    raise KeyError(",".join(missing_permutation_keys))
                permutation = tuple(int(normalized_sizes[key]) for key in permutation_keys)
        tensors[tensor_name] = ConcreteTensor(
            dtype=descriptor.dtype or "",
            dimensions=tuple(dimensions),
            permutation=permutation,
        )
    return tensors


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
    path = source.removeprefix("value.")
    parts = path.split(".")
    if len(parts) == 1:
        return values.get(parts[0])
    if len(parts) == 2 and parts[1].isdigit():
        value = values.get(parts[0])
        index = int(parts[1])
        if not isinstance(value, tuple) or index < 0 or index >= len(value):
            return None
        return int(value[index])
    return None


def _ceil_div(lhs: int, rhs: int) -> int:
    return (lhs + rhs - 1) // rhs


def route_dispatch(
    route: V2Route,
    shape: Mapping[str, int],
    *,
    values: Mapping[str, CapturedValue] | None = None,
) -> dict[str, Any]:
    normalized_shape = {str(name): int(value) for name, value in shape.items()}
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
        ncols = int(normalized_shape.get("d0", 1))
        nrows = int(normalized_shape.get("d1", 1))
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
