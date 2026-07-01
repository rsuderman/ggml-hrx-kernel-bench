from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


ROUTER_FILENAME = "router.json"


@dataclass(frozen=True)
class StrideDescriptor:
    value: int | None = None
    dimension: str | None = None
    product: tuple[str, ...] = ()


@dataclass(frozen=True)
class DimensionBounds:
    min: int | None
    max: int | None


@dataclass(frozen=True)
class ConstraintCheck:
    name: str
    min: int | None = None
    max: int | None = None
    value: int | None = None
    dimension: str | None = None
    product: tuple[str, ...] = ()


@dataclass(frozen=True)
class RouteConstraints:
    sizes: dict[str, DimensionBounds]
    strides: dict[str, StrideDescriptor]
    checks: tuple[ConstraintCheck, ...] = ()


@dataclass(frozen=True)
class TensorDimensionDescriptor:
    name: str


@dataclass(frozen=True)
class TensorStrideIdentifier:
    name: str


@dataclass(frozen=True)
class TensorDescriptor:
    dtype: str | None
    dimensions: tuple[TensorDimensionDescriptor, ...]
    strides: tuple[TensorStrideIdentifier, ...]


@dataclass(frozen=True)
class ConcreteTensorDimension:
    name: str
    size: int
    stride: int


@dataclass(frozen=True)
class ConcreteTensor:
    dtype: str
    dimensions: tuple[ConcreteTensorDimension, ...]


@dataclass(frozen=True)
class V2Route:
    id: str
    family: str
    op: str
    source_id: str
    kernel_path: str
    root_symbol: str
    export_name: str | None
    tensors: dict[str, TensorDescriptor]
    constraints: RouteConstraints
    launch: dict[str, Any]
    bindings: tuple[dict[str, str], ...]


def stable_id(*parts: Any, length: int = 10) -> str:
    text = json.dumps(parts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _descriptor_path(routing_dir: Path) -> Path:
    return routing_dir / ROUTER_FILENAME


def _route_file_path(routing_dir: Path, relative_path: str) -> Path:
    return routing_dir / relative_path


def _normalize_op(value: Any) -> str:
    return str(value).strip().upper()


def _normalize_dtype(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text.upper()


def _parse_stride(
    path: Path,
    route_index: int,
    tensor_name: str,
    dimension_name: str,
    raw: Any,
) -> StrideDescriptor | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise RuntimeError(
            "v2 stride descriptor must be a JSON object for "
            f"route {route_index} tensor {tensor_name!r} dimension {dimension_name!r}: {path}"
        )
    has_value = "value" in raw
    has_dimension = "dimension" in raw
    has_product = "product" in raw
    active = int(has_value) + int(has_dimension) + int(has_product)
    if active != 1:
        raise RuntimeError(
            "v2 stride descriptor must define exactly one of value, dimension, or product for "
            f"route {route_index} tensor {tensor_name!r} dimension {dimension_name!r}: {path}"
        )
    if has_value:
        return StrideDescriptor(value=int(raw["value"]))
    if has_dimension:
        return StrideDescriptor(dimension=str(raw["dimension"]))
    product = raw["product"]
    if not isinstance(product, list) or not product:
        raise RuntimeError(
            "v2 stride descriptor product must be a non-empty JSON array for "
            f"route {route_index} tensor {tensor_name!r} dimension {dimension_name!r}: {path}"
        )
    return StrideDescriptor(product=tuple(str(entry) for entry in product))


def _parse_dimension_names(
    path: Path,
    route_index: int,
    tensor_name: str,
    raw: Any,
) -> tuple[TensorDimensionDescriptor, ...]:
    if not isinstance(raw, list) or not raw:
        raise RuntimeError(
            "v2 tensor descriptor must contain a non-empty dimensions array for "
            f"route {route_index} tensor {tensor_name!r}: {path}"
        )
    dimensions: list[TensorDimensionDescriptor] = []
    seen_names: set[str] = set()
    for raw_dimension in raw:
        if isinstance(raw_dimension, dict):
            name = str(raw_dimension["name"])
        else:
            name = str(raw_dimension)
        if name in seen_names:
            raise RuntimeError(
                "v2 tensor dimension names must be unique for "
                f"route {route_index} tensor {tensor_name!r}: {path}"
            )
        seen_names.add(name)
        dimensions.append(TensorDimensionDescriptor(name=name))
    return tuple(dimensions)


def _parse_stride_names(
    path: Path,
    route_index: int,
    tensor_name: str,
    raw: Any,
    dimension_count: int,
) -> tuple[TensorStrideIdentifier, ...]:
    if not isinstance(raw, list) or not raw:
        raise RuntimeError(
            "v2 tensor strides must be a non-empty JSON array for "
            f"route {route_index} tensor {tensor_name!r}: {path}"
        )
    if len(raw) != dimension_count:
        raise RuntimeError(
            "v2 tensor strides must align one-to-one with dimensions for "
            f"route {route_index} tensor {tensor_name!r}: {path}"
        )
    strides: list[TensorStrideIdentifier] = []
    seen_names: set[str] = set()
    for raw_stride in raw:
        if isinstance(raw_stride, dict):
            name = str(raw_stride["name"])
        else:
            name = str(raw_stride)
        if name in seen_names:
            raise RuntimeError(
                "v2 tensor stride identifiers must be unique for "
                f"route {route_index} tensor {tensor_name!r}: {path}"
            )
        seen_names.add(name)
        strides.append(TensorStrideIdentifier(name=name))
    return tuple(strides)


def _parse_tensor_descriptor(
    path: Path,
    route_index: Any,
    tensor_name: str,
    raw: Any,
) -> TensorDescriptor:
    if not isinstance(raw, dict):
        raise RuntimeError(
            f"v2 tensor descriptor must be a JSON object for route {route_index} tensor {tensor_name!r}: {path}"
        )
    dimensions = _parse_dimension_names(path, route_index, tensor_name, raw.get("dimensions"))
    return TensorDescriptor(
        dtype=_normalize_dtype(raw.get("dtype")),
        dimensions=dimensions,
        strides=_parse_stride_names(path, route_index, tensor_name, raw.get("strides"), len(dimensions)),
    )


def _parse_tensors(path: Path, route_index: Any, raw: Any) -> dict[str, TensorDescriptor]:
    if not isinstance(raw, dict) or not raw:
        raise RuntimeError(
            f"v2 route {route_index} must contain a non-empty tensors object: {path}"
        )
    return {
        str(name): _parse_tensor_descriptor(path, route_index, str(name), descriptor)
        for name, descriptor in raw.items()
    }


def _parse_size_constraint_check(
    path: Path,
    route_index: Any,
    raw: Any,
    declared_dimensions: set[str],
) -> tuple[str, DimensionBounds, ConstraintCheck]:
    if not isinstance(raw, dict):
        raise RuntimeError(f"v2 route size constraint must be a JSON object: {path}")
    dimension_name = str(raw["name"])
    if dimension_name not in declared_dimensions:
        raise RuntimeError(
            f"v2 route {route_index} size constraints must reference declared dimensions: {path}"
        )
    bounds = DimensionBounds(
        min=None if raw.get("min") is None else int(raw["min"]),
        max=None if raw.get("max") is None else int(raw["max"]),
    )
    return (
        dimension_name,
        bounds,
        ConstraintCheck(
            name=dimension_name,
            min=bounds.min,
            max=bounds.max,
        ),
    )


def _parse_stride_constraint_check(
    path: Path,
    route_index: Any,
    raw: Any,
    referenced_stride_names: set[str],
    declared_dimensions: set[str],
) -> tuple[str, StrideDescriptor, ConstraintCheck]:
    if not isinstance(raw, dict):
        raise RuntimeError(f"v2 route stride constraint must be a JSON object: {path}")
    stride_name = str(raw["name"])
    if stride_name not in referenced_stride_names:
        raise RuntimeError(
            f"v2 route {route_index} stride constraints reference unknown stride identifier {stride_name!r}: {path}"
        )
    descriptor = _parse_stride(path, route_index, "route", stride_name, raw)
    if descriptor is None:
        raise RuntimeError(f"v2 route {route_index} stride constraint {stride_name!r} is empty: {path}")
    if descriptor.dimension is not None and descriptor.dimension not in declared_dimensions:
        raise RuntimeError(
            f"v2 route {route_index} stride {stride_name!r} references unknown dimension {descriptor.dimension!r}: {path}"
        )
    if descriptor.product and any(name not in declared_dimensions for name in descriptor.product):
        raise RuntimeError(
            f"v2 route {route_index} stride {stride_name!r} references unknown product dimensions: {path}"
        )
    return (
        stride_name,
        descriptor,
        ConstraintCheck(
            name=stride_name,
            value=descriptor.value,
            dimension=descriptor.dimension,
            product=descriptor.product,
        ),
    )


def _parse_constraints(
    path: Path,
    route_index: Any,
    raw: Any,
    tensors: Mapping[str, TensorDescriptor],
) -> RouteConstraints:
    if raw is None:
        raw = []
    if not isinstance(raw, list):
        raise RuntimeError(f"v2 route constraints must be a JSON array: {path}")
    declared_dimensions = {
        dimension.name
        for descriptor in tensors.values()
        for dimension in descriptor.dimensions
    }
    referenced_strides = {
        stride.name
        for descriptor in tensors.values()
        for stride in descriptor.strides
    }
    sizes: dict[str, DimensionBounds] = {}
    strides: dict[str, StrideDescriptor] = {}
    checks: list[ConstraintCheck] = []
    seen_size_names: set[str] = set()
    seen_stride_names: set[str] = set()
    for entry in raw:
        if not isinstance(entry, dict):
            raise RuntimeError(f"v2 route constraints entries must be JSON objects: {path}")
        has_size_fields = any(field in entry for field in ("min", "max"))
        has_stride_fields = any(field in entry for field in ("value", "dimension", "product"))
        if has_size_fields and has_stride_fields:
            raise RuntimeError(
                f"v2 route {route_index} constraint {entry.get('name')!r} mixes size and stride fields: {path}"
            )
        if has_size_fields:
            name, bounds, check = _parse_size_constraint_check(
                path,
                route_index,
                entry,
                declared_dimensions,
            )
            if name in seen_size_names:
                raise RuntimeError(
                    f"v2 route {route_index} repeats size constraint {name!r}: {path}"
                )
            seen_size_names.add(name)
            sizes[name] = bounds
            checks.append(check)
            continue
        if has_stride_fields:
            name, descriptor, check = _parse_stride_constraint_check(
                path,
                route_index,
                entry,
                referenced_strides,
                declared_dimensions,
            )
            if name in seen_stride_names:
                raise RuntimeError(
                    f"v2 route {route_index} repeats stride constraint {name!r}: {path}"
                )
            seen_stride_names.add(name)
            strides[name] = descriptor
            checks.append(check)
            continue
        raise RuntimeError(
            f"v2 route {route_index} constraint {entry.get('name')!r} must define size or stride fields: {path}"
        )
    missing = sorted(referenced_strides - seen_stride_names)
    if missing:
        raise RuntimeError(
            f"v2 route {route_index} is missing stride constraints for identifiers {missing}: {path}"
        )
    return RouteConstraints(
        sizes=sizes,
        strides=strides,
        checks=tuple(checks),
    )
def _parse_route_entry(path: Path, route_index: Any, op: str, raw: Any) -> V2Route:
    if not isinstance(raw, dict):
        raise RuntimeError(f"v2 route entry {route_index} must be a JSON object: {path}")
    kernel = raw.get("kernel") or {}
    launch = raw.get("launch") or {}
    config = raw.get("config") or {}
    bindings = config.get("bindings") or []
    tensors = _parse_tensors(path, route_index, raw.get("tensors"))
    return V2Route(
        id=str(raw["id"]),
        family=str(raw["family"]),
        op=op,
        source_id=str(kernel["source_id"]),
        kernel_path=str(kernel["path"]),
        root_symbol=str(kernel["root_symbol"]),
        export_name=(
            None if kernel.get("export_name") is None else str(kernel["export_name"])
        ),
        tensors=tensors,
        constraints=_parse_constraints(path, route_index, raw.get("constraints"), tensors),
        launch=dict(launch),
        bindings=tuple(dict(binding) for binding in bindings),
    )


def _load_router_index(routing_dir: Path) -> dict[str, tuple[str, ...]]:
    path = _descriptor_path(routing_dir)
    if not path.exists():
        return {}
    data = _load_json(path)
    if not isinstance(data, dict):
        raise RuntimeError(f"v2 routing descriptor must be a JSON object: {path}")
    raw_routes = data.get("routes")
    if not isinstance(raw_routes, dict) or not raw_routes:
        raise RuntimeError(f"v2 routing descriptor routes must be a non-empty JSON object: {path}")
    index: dict[str, tuple[str, ...]] = {}
    for raw_op, raw_files in raw_routes.items():
        op = _normalize_op(raw_op)
        if not isinstance(raw_files, list) or not raw_files:
            raise RuntimeError(
                f"v2 routing descriptor routes[{raw_op!r}] must be a non-empty JSON array: {path}"
            )
        index[op] = tuple(str(raw_file) for raw_file in raw_files)
    return index


def load_routes_for_op(routing_dir: Path, op: str) -> list[V2Route]:
    normalized_op = _normalize_op(op)
    routes: list[V2Route] = []
    for route_file_name in _load_router_index(routing_dir).get(normalized_op, ()):
        route_file = _route_file_path(routing_dir, route_file_name)
        routes.append(
            _parse_route_entry(route_file, f"{normalized_op}:{route_file_name}", normalized_op, _load_json(route_file))
        )
    return routes


def load_routes(routing_dir: Path) -> list[V2Route]:
    routes: list[V2Route] = []
    for op, route_files in _load_router_index(routing_dir).items():
        for route_file_name in route_files:
            route_file = _route_file_path(routing_dir, route_file_name)
            routes.append(
                _parse_route_entry(route_file, f"{op}:{route_file_name}", op, _load_json(route_file))
            )
    return routes


def iter_routes(routing_dir: Path):
    yield from load_routes(routing_dir)


def source_path_for_route(kernel_dir: Path, route: V2Route) -> Path:
    return kernel_dir / route.kernel_path


def _tensor_dtype(route: V2Route, tensor_name: str) -> str | None:
    descriptor = route.tensors.get(tensor_name)
    return None if descriptor is None else descriptor.dtype


def tensor_descriptors_json(route: V2Route) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for tensor_name, descriptor in route.tensors.items():
        payload[tensor_name] = {
            "dtype": descriptor.dtype,
            "dimensions": [dimension.name for dimension in descriptor.dimensions],
            "strides": [stride.name for stride in descriptor.strides],
        }
    return payload


def tensor_constraints_json(route: V2Route) -> dict[str, Any]:
    return [
        {
            key: value
            for key, value in (
                ("name", check.name),
                ("min", check.min),
                ("max", check.max),
                ("value", check.value),
                ("dimension", check.dimension),
                ("product", list(check.product) if check.product else None),
            )
            if value is not None
        }
        for check in route.constraints.checks
    ]


def route_supports(route: V2Route) -> dict[str, Any]:
    return {
        "src0_type": _tensor_dtype(route, "src0"),
        "src1_type": _tensor_dtype(route, "src1"),
        "dst_type": _tensor_dtype(route, "dst"),
        "tensor_orders": {
            tensor_name: [dimension.name for dimension in descriptor.dimensions]
            for tensor_name, descriptor in route.tensors.items()
        },
    }


def route_accepts_dtype(route: V2Route, dtype: Mapping[str, Any]) -> bool:
    if "type" in dtype:
        actual = _normalize_dtype(dtype["type"])
        return all(descriptor.dtype in {None, actual} for descriptor in route.tensors.values())
    for tensor_name, keys in (
        ("src0", ("type_src0", "type_src")),
        ("src1", ("type_src1", "type_src")),
        ("dst", ("type_dst",)),
    ):
        expected = _tensor_dtype(route, tensor_name)
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
    if len(descriptor.strides) != len(descriptor.dimensions):
        return False
    sizes = _dimension_sizes(tensor)
    for expected, stride_id, actual in zip(
        descriptor.dimensions,
        descriptor.strides,
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


def _shape_aliases(shape: Mapping[str, int]) -> dict[str, int]:
    normalized = {str(key): int(value) for key, value in shape.items()}
    if "ncols" in normalized and "cols" not in normalized:
        normalized["cols"] = normalized["ncols"]
    if "cols" in normalized and "ncols" not in normalized:
        normalized["ncols"] = normalized["cols"]
    if "nrows" in normalized and "rows" not in normalized:
        normalized["rows"] = normalized["nrows"]
    if "rows" in normalized and "nrows" not in normalized:
        normalized["nrows"] = normalized["rows"]
    return normalized


def shape_from_tensors(tensors: Mapping[str, ConcreteTensor]) -> dict[str, int]:
    shape: dict[str, int] = {}
    for tensor in tensors.values():
        for dimension in tensor.dimensions:
            shape.setdefault(dimension.name, int(dimension.size))
    return _shape_aliases(shape)


def _default_dimension_size(dimension: TensorDimensionDescriptor, constraints: RouteConstraints) -> int:
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
    return _shape_aliases(shape)


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
        if len(descriptor.strides) != len(descriptor.dimensions):
            raise RuntimeError(
                f"v2 tensor {tensor_name!r} has mismatched dimension and stride identifiers"
            )
        for dimension, stride_id in zip(descriptor.dimensions, descriptor.strides, strict=True):
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
    rows_per_workgroup = int(route.launch.get("rows_per_workgroup", 1) or 1)
    cols_per_workgroup = int(route.launch.get("cols_per_workgroup", 1) or 1)
    nrows = int(shape.get("nrows", shape.get("rows", 1)))
    ncols = int(shape.get("ncols", shape.get("cols", 1)))
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


def build_manifest(*, kernel_dir: Path, routing_dir: Path) -> dict[str, object]:
    routes = load_routes(routing_dir)
    routes_by_kernel: dict[str, list[V2Route]] = {}
    for route in routes:
        routes_by_kernel.setdefault(route.kernel_path, []).append(route)
    entries = []
    kernel_files = sorted(path.name for path in kernel_dir.glob("*.loom"))
    for kernel_name in kernel_files:
        matching = routes_by_kernel.get(kernel_name, [])
        entries.append(
            {
                "path": str(kernel_dir / kernel_name),
                "source_ids": sorted({route.source_id for route in matching}),
                "route_count": len(matching),
                "coverage": "route_backed" if matching else "unrouted",
            }
        )
    referenced_kernel_files = sorted(routes_by_kernel)
    return {
        "schema": "ggml_hrx_kernel_bench.routing_manifest.v2",
        "routing_version": "v2",
        "kernel_count": len(kernel_files),
        "catalog_source_count": len({route.source_id for route in routes}),
        "route_count": len(routes),
        "entries": entries,
        "source_ids_without_routes": [],
        "route_source_ids_without_source_entry": [],
        "kernel_files_without_source_entry": sorted(set(kernel_files) - set(referenced_kernel_files)),
        "source_entries_without_kernel_file": sorted(
            kernel_name
            for kernel_name in referenced_kernel_files
            if not (kernel_dir / kernel_name).exists()
        ),
    }
