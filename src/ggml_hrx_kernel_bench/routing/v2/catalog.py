from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import (
    BindingDefinition,
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
        permutation_capture=(
            None
            if raw.get("permutation") is None
            else _parse_capture_name(path, route_index, tensor_name, "permutation", raw.get("permutation"))
        ),
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
    inverse_permutation = raw.get("inverse_permutation")
    head = raw.get("head")
    tail = raw.get("tail")
    chain_permutations = raw.get("chain_permutations")
    permuted_contiguous_strides = raw.get("permuted_contiguous_strides")
    operations = sum(
        value is not None
        for value in (
            contiguous_strides,
            product,
            inverse_permutation,
            head,
            tail,
            chain_permutations,
            permuted_contiguous_strides,
        )
    )
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
    if inverse_permutation is not None and (
        not isinstance(inverse_permutation, str) or not inverse_permutation.strip()
    ):
        raise RuntimeError(
            f"v2 route {route_index} value definition {name!r} inverse_permutation must reference a capture name: {path}"
        )
    for operation_name, operation_value in (("head", head), ("tail", tail)):
        if operation_value is None:
            continue
        if not isinstance(operation_value, dict):
            raise RuntimeError(
                f"v2 route {route_index} value definition {name!r} {operation_name} must be a JSON object: {path}"
            )
        source = operation_value.get("source")
        amount_key = "take" if operation_name == "head" else "drop"
        amount = operation_value.get(amount_key)
        if not isinstance(source, str) or not source.strip():
            raise RuntimeError(
                f"v2 route {route_index} value definition {name!r} {operation_name}.source must reference a capture name: {path}"
            )
        if not isinstance(amount, int) or amount < 0:
            raise RuntimeError(
                f"v2 route {route_index} value definition {name!r} {operation_name}.{amount_key} must be a non-negative integer: {path}"
            )
    if chain_permutations is not None:
        if (
            not isinstance(chain_permutations, list)
            or len(chain_permutations) != 2
            or any(not isinstance(entry, str) or not entry.strip() for entry in chain_permutations)
        ):
            raise RuntimeError(
                f"v2 route {route_index} value definition {name!r} chain_permutations must reference exactly two capture names: {path}"
            )
    if permuted_contiguous_strides is not None:
        if not isinstance(permuted_contiguous_strides, dict):
            raise RuntimeError(
                f"v2 route {route_index} value definition {name!r} permuted_contiguous_strides must be a JSON object: {path}"
            )
        dimensions = permuted_contiguous_strides.get("dimensions")
        permutation = permuted_contiguous_strides.get("permutation")
        if not isinstance(dimensions, str) or not dimensions.strip():
            raise RuntimeError(
                f"v2 route {route_index} value definition {name!r} permuted_contiguous_strides.dimensions must reference a capture name: {path}"
            )
        if not isinstance(permutation, str) or not permutation.strip():
            raise RuntimeError(
                f"v2 route {route_index} value definition {name!r} permuted_contiguous_strides.permutation must reference a capture name: {path}"
            )
    if contiguous_strides is not None:
        return ValueDefinition(
            name=name,
            operation_kind="contiguous_strides",
            sources=(contiguous_strides.strip(),),
        )
    if product is not None:
        return ValueDefinition(
            name=name,
            operation_kind="product",
            sources=(product.strip(),),
        )
    if inverse_permutation is not None:
        return ValueDefinition(
            name=name,
            operation_kind="inverse_permutation",
            sources=(inverse_permutation.strip(),),
        )
    if head is not None:
        return ValueDefinition(
            name=name,
            operation_kind="head",
            sources=(str(head["source"]).strip(),),
            parameters=(int(head["take"]),),
        )
    if tail is not None:
        return ValueDefinition(
            name=name,
            operation_kind="tail",
            sources=(str(tail["source"]).strip(),),
            parameters=(int(tail["drop"]),),
        )
    if chain_permutations is not None:
        return ValueDefinition(
            name=name,
            operation_kind="chain_permutations",
            sources=(str(chain_permutations[0]).strip(), str(chain_permutations[1]).strip()),
        )
    if permuted_contiguous_strides is not None:
        return ValueDefinition(
            name=name,
            operation_kind="permuted_contiguous_strides",
            sources=(
                str(permuted_contiguous_strides["dimensions"]).strip(),
                str(permuted_contiguous_strides["permutation"]).strip(),
            ),
        )
    raise AssertionError(f"v2 route {route_index} value definition {name!r} did not resolve to an operation: {path}")


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
    rank_min = raw.get("rank_min")
    rank_max = raw.get("rank_max")
    index = raw.get("index")
    lower = raw.get("min")
    upper = raw.get("max")
    multiple_of = raw.get("multiple_of")
    iota = raw.get("iota")
    if rank_min is not None or rank_max is not None:
        if length is not None or index is not None or lower is not None or upper is not None or multiple_of is not None or iota is not None:
            raise RuntimeError(
                f"v2 route {route_index} rank constraints cannot mix length/index/min/max/multiple_of/iota fields: {path}"
            )
        normalized_rank_min = None if rank_min is None else int(rank_min)
        normalized_rank_max = None if rank_max is None else int(rank_max)
        if normalized_rank_min is not None and normalized_rank_min <= 0:
            raise RuntimeError(f"v2 route {route_index} rank_min must be positive: {path}")
        if normalized_rank_max is not None and normalized_rank_max <= 0:
            raise RuntimeError(f"v2 route {route_index} rank_max must be positive: {path}")
        if (
            normalized_rank_min is not None
            and normalized_rank_max is not None
            and normalized_rank_min > normalized_rank_max
        ):
            raise RuntimeError(f"v2 route {route_index} rank_min must be <= rank_max: {path}")
        return ConstraintCheck(name=name, rank_min=normalized_rank_min, rank_max=normalized_rank_max)
    if length is not None:
        if index is not None or lower is not None or upper is not None or multiple_of is not None or iota is not None:
            raise RuntimeError(
                f"v2 route {route_index} length constraints cannot mix index/min/max/multiple_of/iota fields: {path}"
            )
        return ConstraintCheck(name=name, length=int(length))
    if iota is not None:
        if index is not None or lower is not None or upper is not None or multiple_of is not None:
            raise RuntimeError(
                f"v2 route {route_index} iota constraints cannot mix index/min/max/multiple_of fields: {path}"
            )
        if not isinstance(iota, bool) or not iota:
            raise RuntimeError(f"v2 route {route_index} iota constraints must set iota=true: {path}")
        return ConstraintCheck(name=name, iota=True)
    if lower is None and upper is None and multiple_of is None:
        raise RuntimeError(
            f"v2 route {route_index} constraint {name!r} must define length, iota, bounds, or multiple_of: {path}"
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


def _parse_non_empty_string(path: Path, context: str, raw: Any) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise RuntimeError(f"{context} must be a non-empty string: {path}")
    return raw.strip()


def _parse_bindings(path: Path, route_index: Any, raw: Any) -> tuple[BindingDefinition, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise RuntimeError(f"v2 route {route_index} config.bindings must be a JSON array: {path}")
    bindings: list[BindingDefinition] = []
    for binding_index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise RuntimeError(
                f"v2 route {route_index} config.bindings[{binding_index}] must be a JSON object: {path}"
            )
        extra_keys = set(entry) - {"key", "source", "value"}
        if extra_keys:
            raise RuntimeError(
                f"v2 route {route_index} config.bindings[{binding_index}] has unsupported keys {sorted(extra_keys)!r}: {path}"
            )
        key = _parse_non_empty_string(
            path,
            f"v2 route {route_index} config.bindings[{binding_index}].key",
            entry.get("key"),
        )
        source = entry.get("source")
        value = entry.get("value")
        if source is not None and value is not None:
            raise RuntimeError(
                f"v2 route {route_index} config.bindings[{binding_index}] must not define both source and value: {path}"
            )
        if source is None and value is None:
            raise RuntimeError(
                f"v2 route {route_index} config.bindings[{binding_index}] must define source or value: {path}"
            )
        bindings.append(
            BindingDefinition(
                key=key,
                source=None if source is None else _parse_non_empty_string(
                    path,
                    f"v2 route {route_index} config.bindings[{binding_index}].source",
                    source,
                ),
                value=None if value is None else _parse_non_empty_string(
                    path,
                    f"v2 route {route_index} config.bindings[{binding_index}].value",
                    value,
                ),
            )
        )
    return tuple(bindings)


def _parse_kernel(path: Path, route_index: Any, raw: Any) -> dict[str, str | None]:
    if not isinstance(raw, dict):
        raise RuntimeError(f"v2 route {route_index} kernel must be a JSON object: {path}")
    extra_keys = set(raw) - {"source_id", "path", "root_symbol", "export_name"}
    if extra_keys:
        raise RuntimeError(f"v2 route {route_index} kernel has unsupported keys {sorted(extra_keys)!r}: {path}")
    return {
        "source_id": _parse_non_empty_string(path, f"v2 route {route_index} kernel.source_id", raw.get("source_id")),
        "path": _parse_non_empty_string(path, f"v2 route {route_index} kernel.path", raw.get("path")),
        "root_symbol": _parse_non_empty_string(
            path,
            f"v2 route {route_index} kernel.root_symbol",
            raw.get("root_symbol"),
        ),
        "export_name": (
            None
            if raw.get("export_name") is None
            else _parse_non_empty_string(path, f"v2 route {route_index} kernel.export_name", raw.get("export_name"))
        ),
    }


def _parse_route_entry(path: Path, route_index: Any, op: str, raw: Any) -> V2Route:
    if not isinstance(raw, dict):
        raise RuntimeError(f"v2 route entry {route_index} must be a JSON object: {path}")
    kernel = _parse_kernel(path, route_index, raw.get("kernel"))
    launch = raw.get("launch") or {}
    config = raw.get("config") or {}
    route_id = _parse_non_empty_string(path, f"v2 route {route_index} id", raw.get("id"))
    family = _parse_non_empty_string(path, f"v2 route {route_index} family", raw.get("family"))
    if not isinstance(launch, dict):
        raise RuntimeError(f"v2 route {route_index} launch must be a JSON object: {path}")
    if not isinstance(config, dict):
        raise RuntimeError(f"v2 route {route_index} config must be a JSON object: {path}")
    extra_keys = set(raw) - {
        "bindings",
        "config",
        "constraints",
        "family",
        "id",
        "kernel",
        "launch",
        "tensors",
        "values",
    }
    if extra_keys:
        raise RuntimeError(f"v2 route {route_index} has unsupported keys {sorted(extra_keys)!r}: {path}")
    return V2Route(
        id=route_id,
        family=family,
        op=op,
        source_id=str(kernel["source_id"]),
        kernel_path=str(kernel["path"]),
        root_symbol=str(kernel["root_symbol"]),
        export_name=None if kernel.get("export_name") is None else str(kernel["export_name"]),
        tensors=_parse_tensors(path, route_index, raw.get("tensors")),
        values=_parse_values(path, route_index, raw.get("values")),
        constraints=_parse_constraints(path, route_index, raw.get("constraints")),
        launch=dict(launch),
        bindings=_parse_bindings(path, route_index, config.get("bindings")),
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
