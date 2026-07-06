from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import (
    ConstraintCheck,
    RouteConstraints,
    TensorDescriptor,
    V2Route,
    ValueDefinition,
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


def _parse_capture_name(
    path: Path,
    route_index: Any,
    tensor_name: str,
    field_name: str,
    raw: Any,
) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise RuntimeError(
            f"v2 tensor {field_name} must be a non-empty string capture for route {route_index} tensor {tensor_name!r}: {path}"
        )
    return raw.strip()


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
    return TensorDescriptor(
        dtype=_normalize_dtype(raw.get("dtype")),
        dimensions_capture=_parse_capture_name(path, route_index, tensor_name, "dimensions", raw.get("dimensions")),
        strides_capture=_parse_capture_name(path, route_index, tensor_name, "strides", raw.get("strides")),
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


def _parse_value_definition(path: Path, route_index: Any, raw: Any) -> ValueDefinition:
    if not isinstance(raw, dict):
        raise RuntimeError(f"v2 route value definition must be a JSON object: {path}")
    name = str(raw.get("name") or "").strip()
    if not name:
        raise RuntimeError(f"v2 route {route_index} value definitions require name: {path}")
    contiguous_strides = raw.get("contiguous_strides")
    product = raw.get("product")
    operations = sum(value is not None for value in (contiguous_strides, product))
    if operations != 1:
        raise RuntimeError(
            f"v2 route {route_index} value definition {name!r} must define exactly one supported computation: {path}"
        )
    if contiguous_strides is not None and (not isinstance(contiguous_strides, str) or not contiguous_strides.strip()):
        raise RuntimeError(
            f"v2 route {route_index} value definition {name!r} contiguous_strides must reference a capture name: {path}"
        )
    if product is not None and (not isinstance(product, str) or not product.strip()):
        raise RuntimeError(
            f"v2 route {route_index} value definition {name!r} product must reference a capture name: {path}"
        )
    return ValueDefinition(
        name=name,
        contiguous_strides=None if contiguous_strides is None else contiguous_strides.strip(),
        product=None if product is None else product.strip(),
    )


def _parse_values(path: Path, route_index: Any, raw: Any) -> tuple[ValueDefinition, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise RuntimeError(f"v2 route values must be a JSON array: {path}")
    values = tuple(_parse_value_definition(path, route_index, entry) for entry in raw)
    names = [value.name for value in values]
    if len(names) != len(set(names)):
        raise RuntimeError(f"v2 route {route_index} repeats value names: {path}")
    return values


def _parse_equals_constraint(path: Path, route_index: Any, raw: Any) -> ConstraintCheck:
    names = raw.get("equals")
    if not isinstance(names, list) or len(names) < 2:
        raise RuntimeError(
            f"v2 route {route_index} equals constraints must be arrays with at least two names: {path}"
        )
    return ConstraintCheck(equals=tuple(str(name) for name in names))


def _parse_divides_constraint(path: Path, route_index: Any, raw: Any) -> ConstraintCheck:
    names = raw.get("divides")
    if not isinstance(names, list) or len(names) < 2:
        raise RuntimeError(
            f"v2 route {route_index} divides constraints must be arrays with at least two names: {path}"
        )
    return ConstraintCheck(divides=tuple(str(name) for name in names))


def _parse_capture_constraint(path: Path, route_index: Any, raw: Any) -> ConstraintCheck:
    name = str(raw.get("name") or "").strip()
    if not name:
        raise RuntimeError(f"v2 route {route_index} constraints require name: {path}")
    length = raw.get("length")
    index = raw.get("index")
    lower = raw.get("min")
    upper = raw.get("max")
    multiple_of = raw.get("multiple_of")
    if length is not None:
        if index is not None or lower is not None or upper is not None or multiple_of is not None:
            raise RuntimeError(
                f"v2 route {route_index} length constraints cannot mix index/min/max/multiple_of fields: {path}"
            )
        return ConstraintCheck(name=name, length=int(length))
    if lower is None and upper is None and multiple_of is None:
        raise RuntimeError(
            f"v2 route {route_index} constraint {name!r} must define length, bounds, or multiple_of: {path}"
        )
    return ConstraintCheck(
        name=name,
        index=None if index is None else int(index),
        min=None if lower is None else int(lower),
        max=None if upper is None else int(upper),
        multiple_of=None if multiple_of is None else int(multiple_of),
    )


def _parse_constraints(path: Path, route_index: Any, raw: Any) -> RouteConstraints:
    if raw is None:
        return RouteConstraints()
    if not isinstance(raw, list):
        raise RuntimeError(f"v2 route constraints must be a JSON array: {path}")
    checks: list[ConstraintCheck] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise RuntimeError(f"v2 route constraints entries must be JSON objects: {path}")
        if "equals" in entry:
            checks.append(_parse_equals_constraint(path, route_index, entry))
            continue
        if "divides" in entry:
            checks.append(_parse_divides_constraint(path, route_index, entry))
            continue
        checks.append(_parse_capture_constraint(path, route_index, entry))
    return RouteConstraints(checks=tuple(checks))


def _parse_route_entry(path: Path, route_index: Any, op: str, raw: Any) -> V2Route:
    if not isinstance(raw, dict):
        raise RuntimeError(f"v2 route entry {route_index} must be a JSON object: {path}")
    kernel = raw.get("kernel") or {}
    launch = raw.get("launch") or {}
    config = raw.get("config") or {}
    bindings = config.get("bindings") or []
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
        tensors=_parse_tensors(path, route_index, raw.get("tensors")),
        values=_parse_values(path, route_index, raw.get("values")),
        constraints=_parse_constraints(path, route_index, raw.get("constraints")),
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
