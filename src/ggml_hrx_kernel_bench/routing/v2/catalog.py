from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .models import (
    ConstraintCheck,
    DimensionBounds,
    RouteConstraints,
    StrideDescriptor,
    TensorDescriptor,
    TensorDimensionDescriptor,
    TensorStrideIdentifier,
    V2Route,
)


ROUTER_FILENAME = "router.json"


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
    route_index: Any,
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
    route_index: Any,
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
    route_index: Any,
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
        stride_ids=_parse_stride_names(path, route_index, tensor_name, raw.get("strides"), len(dimensions)),
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
            identifier=dimension_name,
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
            identifier=stride_name,
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
        for stride in descriptor.stride_ids
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


def load_route_index(routing_dir: Path) -> dict[str, tuple[str, ...]]:
    return _load_router_index(routing_dir)


def load_route_file(routing_dir: Path, *, op: str, route_file_name: str) -> V2Route:
    normalized_op = _normalize_op(op)
    route_file = _route_file_path(routing_dir, route_file_name)
    return _parse_route_entry(
        route_file,
        f"{normalized_op}:{route_file_name}",
        normalized_op,
        _load_json(route_file),
    )
