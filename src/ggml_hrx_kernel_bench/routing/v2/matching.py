from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping

from .layout import (
    chain_permutations,
    contiguous_strides,
    inverse_permutation,
    permuted_contiguous_strides,
)
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

_ROUTE_SELECTOR_ENV_VAR = "GGML_HRX_V2_ROUTE_SELECTOR"
_ROUTE_SELECTOR_TIMEOUT_SECONDS = 10.0
_DEFAULT_ROUTE_SELECTOR = (
    Path(__file__).resolve().parents[4]
    / "build"
    / "tools"
    / "v2-route-selector"
    / "ggml-hrx-v2-route-selector"
)


def _normalize_dtype(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text.upper()


def route_accepts_dtype(route: V2Route, dtype: Mapping[str, Any]) -> bool:
    if route.op == "GET_ROWS" and "type" in dtype:
        src_descriptor = route.tensors.get("src0")
        src_expected = None if src_descriptor is None else src_descriptor.dtype
        if src_expected is not None and _normalize_dtype(dtype["type"]) != src_expected:
            return False
        idx_descriptor = route.tensors.get("src1")
        idx_expected = None if idx_descriptor is None else idx_descriptor.dtype
        return "type_idx" not in dtype or idx_expected is None or _normalize_dtype(dtype["type_idx"]) == idx_expected
    for tensor_name, keys in (
        ("src0", ("type_src0", "type_src", "type")),
        ("src1", ("type_src1", "type_idx", "type_src")),
        ("dst", ("type_dst", "type")),
    ):
        descriptor = route.tensors.get(tensor_name)
        expected = None if descriptor is None else descriptor.dtype
        if expected is None:
            continue
        provided = next((dtype[key] for key in keys if key in dtype), None)
        if provided is not None and _normalize_dtype(provided) != expected:
            return False
    return True


def _attribute_values_equal(lhs: Any, rhs: Any) -> bool:
    if isinstance(lhs, tuple) and isinstance(rhs, list | tuple):
        return len(lhs) == len(rhs) and all(
            _attribute_values_equal(left, right) for left, right in zip(lhs, rhs, strict=True)
        )
    if isinstance(lhs, list) and isinstance(rhs, list | tuple):
        return len(lhs) == len(rhs) and all(
            _attribute_values_equal(left, right) for left, right in zip(lhs, rhs, strict=True)
        )
    if isinstance(lhs, Mapping) and isinstance(rhs, Mapping):
        return set(lhs) == set(rhs) and all(_attribute_values_equal(lhs[key], rhs[key]) for key in lhs)
    return lhs == rhs


def route_accepts_attributes(route: V2Route, attributes: Mapping[str, Any]) -> bool:
    for key, expected in route.attributes.items():
        if key not in attributes:
            return False
        if not _attribute_values_equal(expected, attributes[key]):
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


def _product(dimensions: tuple[int, ...]) -> int:
    total = 1
    for size in dimensions:
        total *= int(size)
    return total


def _rank_accepts(
    rank: int,
    *,
    exact: int | None = None,
    minimum: int | None = None,
    maximum: int | None = None,
) -> bool:
    if exact is not None and rank != exact:
        return False
    if minimum is not None and rank < minimum:
        return False
    if maximum is not None and rank > maximum:
        return False
    return True


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
        if definition.operation_kind == "contiguous_strides":
            source = lookup(definition.sources[0])
            if not isinstance(source, tuple):
                return None
            resolved[definition.name] = contiguous_strides(source)
            continue
        if definition.operation_kind == "product":
            source = lookup(definition.sources[0])
            if not isinstance(source, tuple):
                return None
            resolved[definition.name] = _product(source)
            continue
        if definition.operation_kind == "inverse_permutation":
            source = lookup(definition.sources[0])
            if not isinstance(source, tuple):
                return None
            inverse = inverse_permutation(source)
            if inverse is None:
                return None
            resolved[definition.name] = inverse
            continue
        if definition.operation_kind == "element":
            source = lookup(definition.sources[0])
            index = definition.parameters[0]
            if not isinstance(source, tuple) or index >= len(source):
                return None
            resolved[definition.name] = int(source[index])
            continue
        if definition.operation_kind == "head":
            source = lookup(definition.sources[0])
            if not isinstance(source, tuple):
                return None
            resolved[definition.name] = tuple(int(value) for value in source[: definition.parameters[0]])
            continue
        if definition.operation_kind == "tail":
            source = lookup(definition.sources[0])
            if not isinstance(source, tuple):
                return None
            resolved[definition.name] = tuple(int(value) for value in source[definition.parameters[0] :])
            continue
        if definition.operation_kind == "chain_permutations":
            first = lookup(definition.sources[0])
            second = lookup(definition.sources[1])
            if not isinstance(first, tuple) or not isinstance(second, tuple):
                return None
            chained = chain_permutations(first, second)
            if chained is None:
                return None
            resolved[definition.name] = chained
            continue
        if definition.operation_kind == "permuted_contiguous_strides":
            dimensions = lookup(definition.sources[0])
            permutation = lookup(definition.sources[1])
            if not isinstance(dimensions, tuple) or not isinstance(permutation, tuple):
                return None
            strides = permuted_contiguous_strides(dimensions, permutation)
            if strides is None:
                return None
            resolved[definition.name] = strides
            continue
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
    if check.rank_min is not None or check.rank_max is not None:
        if not isinstance(captured, tuple):
            return False
        return _rank_accepts(len(captured), minimum=check.rank_min, maximum=check.rank_max)
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


def route_captures(
    route: V2Route,
    tensors: Mapping[str, ConcreteTensor],
    *,
    tensor_names: set[str] | None = None,
) -> dict[str, CapturedValue] | None:
    captures: dict[str, CapturedValue] = {}
    names = route.tensors.keys() if tensor_names is None else tensor_names
    for tensor_name in names:
        descriptor = route.tensors.get(tensor_name)
        tensor = tensors.get(tensor_name)
        if descriptor is None or tensor is None or not _capture_tensor(descriptor, tensor, captures):
            return None
    return captures


def route_accepts_tensors(route: V2Route, tensors: Mapping[str, ConcreteTensor]) -> bool:
    selector_override = os.environ.get(_ROUTE_SELECTOR_ENV_VAR)
    selector_path = Path(selector_override) if selector_override else _DEFAULT_ROUTE_SELECTOR

    tensor_payload: dict[str, dict[str, object]] = {}
    for role in route.tensors:
        tensor = tensors.get(role)
        if tensor is None:
            continue
        serialized: dict[str, object] = {
            "dtype": str(tensor.dtype),
            "dimensions": [int(dimension.size) for dimension in tensor.dimensions],
            "strides": [int(dimension.stride) for dimension in tensor.dimensions],
        }
        if tensor.permutation is not None:
            serialized["permutation"] = [int(axis) for axis in tensor.permutation]
        tensor_payload[role] = serialized

    payload = {
        "op": route.op,
        "tensors": tensor_payload,
        "allowed_route_ids": [route.id],
    }
    command = [
        str(selector_path),
        "--input",
        "-",
        "--expect-route",
        route.id,
    ]
    try:
        result = subprocess.run(
            command,
            input=json.dumps(payload, separators=(",", ":")),
            text=True,
            capture_output=True,
            check=False,
            timeout=_ROUTE_SELECTOR_TIMEOUT_SECONDS,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"v2 route selector executable not found: {selector_path}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"v2 route selector timed out while evaluating route {route.id!r}: {selector_path}"
        ) from exc
    except (OSError, UnicodeError) as exc:
        raise RuntimeError(
            f"failed to launch v2 route selector for route {route.id!r} at {selector_path}: {exc}"
        ) from exc

    expected_no_match = f"error: NO_MATCH: no route matched operation {route.op!r}\n"
    if result.returncode == 1 and not result.stdout and result.stderr == expected_no_match:
        return False

    expected_output = f"{route.id}\n"
    if result.returncode == 0 and result.stdout == expected_output and not result.stderr:
        return True

    raise RuntimeError(
        f"v2 route selector returned an unexpected result for route {route.id!r}: "
        f"exit_code={result.returncode}, stdout={result.stdout!r}, stderr={result.stderr!r}"
    )


def route_values(route: V2Route, tensors: Mapping[str, ConcreteTensor]) -> dict[str, CapturedValue] | None:
    captures = route_captures(route, tensors)
    if captures is None:
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
        default_strides = contiguous_strides(tuple(sizes))
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
) -> tuple[int | None, int | None, int | None, dict[int, tuple[int | None, int | None]]]:
    exact_rank: int | None = None
    rank_min: int | None = None
    rank_max: int | None = None
    bounds: dict[int, tuple[int | None, int | None]] = {}
    for check in route.constraints.checks:
        if check.name != capture_name:
            continue
        if check.length is not None:
            exact_rank = check.length
            continue
        if check.rank_min is not None or check.rank_max is not None:
            if check.rank_min is not None:
                rank_min = check.rank_min if rank_min is None else max(rank_min, check.rank_min)
            if check.rank_max is not None:
                rank_max = check.rank_max if rank_max is None else min(rank_max, check.rank_max)
            continue
        if check.index is not None:
            bounds[check.index] = (check.min, check.max)
    return exact_rank, rank_min, rank_max, bounds


def _shape_capture_for_route(
    route: V2Route,
) -> tuple[str, int | None, int | None, int | None, dict[int, tuple[int | None, int | None]]]:
    for descriptor in route.tensors.values():
        exact_rank, rank_min, rank_max, bounds = _constraint_for_capture(route, descriptor.dimensions_capture)
        if exact_rank is not None or rank_min is not None or rank_max is not None:
            return descriptor.dimensions_capture, exact_rank, rank_min, rank_max, bounds
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
        _, exact_rank, rank_min, rank_max, bounds = _shape_capture_for_route(route)
    except RuntimeError:
        total_size_min = _scalar_constraint_min(route, "total_size")
        default_cols = total_size_min if total_size_min is not None else int(route.launch.get("workgroup_size", [1])[0] or 1)
        return {"d0": default_cols, "d1": 1}
    rank = exact_rank if exact_rank is not None else rank_min if rank_min is not None else rank_max
    if rank <= 0:
        raise RuntimeError(f"v2 route {route.id!r} requires unsupported rank {rank!r} for default shape")
    return {
        name: 1 if bounds.get(index, (None, None))[0] is None else int(bounds[index][0])
        for index, name in enumerate(_rank_dimension_names(rank))
    }


def materialize_route_tensors(route: V2Route, sizes: Mapping[str, int]) -> dict[str, ConcreteTensor]:
    normalized_sizes = {str(name): int(value) for name, value in sizes.items()}
    try:
        _, exact_rank, rank_min, rank_max, _ = _shape_capture_for_route(route)
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
        default_strides = contiguous_strides(tuple(default_sizes))
        dimensions = tuple(
            ConcreteTensorDimension(name=name, size=default_sizes[index], stride=default_strides[index])
            for index, name in enumerate(dimension_names)
        )
        return {
            tensor_name: ConcreteTensor(dtype=descriptor.dtype or "", dimensions=dimensions)
            for tensor_name, descriptor in route.tensors.items()
        }
    rank = exact_rank
    if rank is None:
        ranked_dimension_names = sorted(
            (
                (int(name[1:]), name)
                for name in normalized_sizes
                if name.startswith("d") and name[1:].isdigit()
            ),
            key=lambda item: item[0],
        )
        if ranked_dimension_names:
            dimension_names = tuple(name for _, name in ranked_dimension_names)
            if tuple(index for index, _ in ranked_dimension_names) != tuple(range(len(dimension_names))):
                raise KeyError(",".join(f"d{index}" for index in range(len(dimension_names))))
            inferred_rank = len(dimension_names)
            if not _rank_accepts(inferred_rank, minimum=rank_min, maximum=rank_max):
                raise RuntimeError(f"v2 route {route.id!r} rank {inferred_rank!r} does not satisfy supported rank range")
            rank = inferred_rank
        else:
            rank = rank_min if rank_min is not None else rank_max
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
        default_strides = contiguous_strides(tuple(sizes_for_tensor))
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


def _total_size_workgroup_count(total_size: int, lane_count: int, route: V2Route) -> list[int]:
    flat_workgroups = _ceil_div(total_size, lane_count)
    max_workgroups_x = route.launch.get("max_workgroups_x")
    if isinstance(max_workgroups_x, int) and max_workgroups_x > 0 and flat_workgroups > max_workgroups_x:
        workgroups_y = _ceil_div(flat_workgroups, max_workgroups_x)
        return [_ceil_div(flat_workgroups, workgroups_y), workgroups_y, 1]
    return [flat_workgroups, 1, 1]


def route_dispatch(
    route: V2Route,
    shape: Mapping[str, int],
    *,
    values: Mapping[str, CapturedValue] | None = None,
) -> dict[str, Any]:
    normalized_shape = {str(name): int(value) for name, value in shape.items()}
    workgroup_size = list(route.launch.get("workgroup_size", [None, None, None]))
    lane_count = int(workgroup_size[0] or 1)
    workgroup_count_source = route.launch.get("workgroup_count_source")
    if isinstance(workgroup_count_source, str) and values is not None:
        resolved = value_from_route_source(workgroup_count_source, values)
        if isinstance(resolved, int):
            return {
                "workgroup_count": [resolved, 1, 1],
                "workgroup_size": workgroup_size,
                "rows_per_workgroup": int(route.launch.get("rows_per_workgroup", 1) or 1),
                "cols_per_workgroup": int(route.launch.get("cols_per_workgroup", 1) or 1),
                "metadata_source": "route_descriptor_v2",
                "has_static_dispatch_workgroup_count": False,
                "has_static_workgroup_size": bool(route.launch.get("workgroup_size")),
            }
    total_size = None
    if values is not None:
        resolved = values.get("total_size")
        if isinstance(resolved, int):
            total_size = resolved
    if total_size is not None:
        workgroup_count = _total_size_workgroup_count(total_size, lane_count, route)
    elif route.family.startswith("mul_mat"):
        rows_per_workgroup = int(route.launch.get("rows_per_workgroup", 1) or 1)
        cols_per_workgroup = int(route.launch.get("cols_per_workgroup", 1) or 1)
        rows = int(normalized_shape.get("rows", normalized_shape.get("d0", 1)))
        cols = int(normalized_shape.get("cols", normalized_shape.get("d1", 1)))
        outer = cols * int(normalized_shape.get("d2", 1)) * int(normalized_shape.get("d3", 1))
        workgroup_count = [
            _ceil_div(rows, rows_per_workgroup),
            _ceil_div(outer, cols_per_workgroup),
            1,
        ]
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
