from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from .routing.v2.layout import EncodedRouteShape, contiguous_strides, encode_route_shape
from .routing.v2.matching import (
    route_accepts_attributes,
    route_accepts_tensors,
    route_captures,
    route_values,
)
from .routing.v2.models import ConcreteTensor, ConcreteTensorDimension, V2Route
from .routing.v2.query import load_route_catalog, routes_for_op


YAML_ROUTE_IMPORT_SCHEMA = "ggml_hrx_kernel_bench.yaml_route_import.v1"
YAML_SURFACE_SCHEMA = "ggml_hrx_kernel_bench.yaml_surface.v1"
IMPORT_TEST_COVERAGE_SCHEMA = "ggml_hrx_kernel_bench.import_test_coverage.v1"
GENERATED_KERNEL_TESTS_SCHEMA = "ggml_hrx_kernel_bench.generated_kernel_tests.v1"
ROUTE_EXECUTION_ABI_SCHEMA = "ggml_hrx_kernel_bench.route_execution_abi.v1"


def _normalize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _normalize_value(current)
            for key, current in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, list):
        return [_normalize_value(current) for current in value]
    return value


def _json_key(value: Any) -> str:
    return json.dumps(_normalize_value(value), sort_keys=True, separators=(",", ":"))


def _histogram(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def _safe_name(name: str) -> str:
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in name)


def _params_signature(params_key: tuple[str, ...]) -> str:
    payload = json.dumps(list(params_key), separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _config_filename(
    kernel_family: str,
    route_id: str | None,
    params_key: tuple[str, ...],
    *,
    require_params_suffix: bool,
) -> str:
    parts = [_safe_name(kernel_family)]
    if route_id:
        parts.append(_safe_name(route_id))
    if require_params_suffix:
        parts.append(_params_signature(params_key))
    return ".".join(parts) + ".json"


def _dtype(value: Any) -> str:
    return str(value).strip().upper()


@dataclass(frozen=True)
class YamlTensor:
    role: str
    dtype: str
    shape: tuple[int, ...]
    storage_shape: tuple[int, ...] | None
    offset: str | int | None
    permutation: tuple[int, ...] | None
    raw: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "role": self.role,
            "dtype": self.dtype,
            "shape": list(self.shape),
        }
        if self.storage_shape is not None:
            payload["storage_shape"] = list(self.storage_shape)
        if self.offset is not None:
            payload["offset"] = self.offset
        if self.permutation is not None:
            payload["permutation"] = list(self.permutation)
        return payload


@dataclass(frozen=True)
class YamlCase:
    source_path: str
    op: str
    case_index: int
    inputs: tuple[YamlTensor, ...]
    destinations: tuple[YamlTensor, ...]
    attributes: dict[str, Any]
    raw_case: dict[str, Any]

    @property
    def source_id(self) -> str:
        return f"{self.source_path}:{self.op}[{self.case_index}]"

    def to_json(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "op": self.op,
            "case_index": self.case_index,
            "inputs": [tensor.to_json() for tensor in self.inputs],
            "destinations": [tensor.to_json() for tensor in self.destinations],
            "attributes": self.attributes,
        }


@dataclass(frozen=True)
class InvalidYamlCase:
    source_path: str
    op: str
    case_index: int
    reason: str
    raw_case: Any

    def to_json(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "op": self.op,
            "case_index": self.case_index,
            "reason": self.reason,
            "raw_case": _normalize_value(self.raw_case),
        }


def _expect_int_tuple(value: Any, *, field: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list")
    values: list[int] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(f"{field}[{index}] must be an integer")
        if item < 0:
            raise ValueError(f"{field}[{index}] must be non-negative")
        values.append(int(item))
    return tuple(values)


def _optional_permutation(raw: dict[str, Any]) -> tuple[int, ...] | None:
    value = raw.get("permutation")
    if value is None:
        value = raw.get("transposition")
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        raise ValueError("permutation/transposition must be a non-empty list")
    values: list[int] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(f"permutation/transposition[{index}] must be an integer")
        values.append(int(item))
    return tuple(values)


def _parse_tensor(raw: Any, *, role: str) -> YamlTensor:
    if not isinstance(raw, dict):
        raise ValueError(f"{role} tensor must be a mapping")
    dtype = raw.get("dtype")
    if not isinstance(dtype, str) or not dtype.strip():
        raise ValueError(f"{role}.dtype must be a non-empty string")
    shape = _expect_int_tuple(raw.get("shape"), field=f"{role}.shape")
    storage_shape = None
    if "storage_shape" in raw:
        storage_shape = _expect_int_tuple(raw.get("storage_shape"), field=f"{role}.storage_shape")
        if len(storage_shape) < len(shape):
            raise ValueError(f"{role}.storage_shape rank must be >= shape rank")
    offset = raw.get("offset")
    if (
        offset is not None
        and not isinstance(offset, str)
        and (isinstance(offset, bool) or not isinstance(offset, int))
    ):
        raise ValueError(f"{role}.offset must be a string or integer")
    permutation = _optional_permutation(raw)
    if permutation is not None and len(permutation) != len(shape):
        raise ValueError(f"{role}.permutation rank must match shape rank")
    return YamlTensor(
        role=role,
        dtype=_dtype(dtype),
        shape=shape,
        storage_shape=storage_shape,
        offset=offset,
        permutation=permutation,
        raw=_normalize_value(raw),
    )


def _parse_case(source_path: Path, op: str, case_index: int, raw_case: Any) -> YamlCase:
    if not isinstance(raw_case, dict):
        raise ValueError("case must be a mapping")
    raw_inputs = raw_case.get("inputs", [])
    raw_destinations = raw_case.get("destinations", [])
    if not isinstance(raw_inputs, list):
        raise ValueError("inputs must be a list")
    if not isinstance(raw_destinations, list):
        raise ValueError("destinations must be a list")
    attributes = raw_case.get("attributes", {})
    if attributes is None:
        attributes = {}
    if not isinstance(attributes, dict):
        raise ValueError("attributes must be a mapping")
    inputs = tuple(
        _parse_tensor(tensor, role=f"src{index}")
        for index, tensor in enumerate(raw_inputs)
    )
    destinations = tuple(
        _parse_tensor(tensor, role="dst" if index == 0 else f"dst{index}")
        for index, tensor in enumerate(raw_destinations)
    )
    return YamlCase(
        source_path=str(source_path),
        op=op,
        case_index=case_index,
        inputs=inputs,
        destinations=destinations,
        attributes=_normalize_value(attributes),
        raw_case=_normalize_value(raw_case),
    )


def load_yaml_cases(yaml_paths: Iterable[Path]) -> tuple[list[YamlCase], list[InvalidYamlCase]]:
    cases: list[YamlCase] = []
    invalid: list[InvalidYamlCase] = []
    for yaml_path in yaml_paths:
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("ops"), dict):
            invalid.append(
                InvalidYamlCase(
                    source_path=str(yaml_path),
                    op="<suite>",
                    case_index=0,
                    reason="YAML must contain an ops mapping",
                    raw_case=data,
                )
            )
            continue
        for raw_op, raw_cases in data["ops"].items():
            op = str(raw_op).strip().upper()
            if not isinstance(raw_cases, list):
                invalid.append(
                    InvalidYamlCase(
                        source_path=str(yaml_path),
                        op=op,
                        case_index=0,
                        reason=f"ops.{op} must be a list",
                        raw_case=raw_cases,
                    )
                )
                continue
            for case_index, raw_case in enumerate(raw_cases):
                try:
                    cases.append(_parse_case(yaml_path, op, case_index, raw_case))
                except ValueError as exc:
                    invalid.append(
                        InvalidYamlCase(
                            source_path=str(yaml_path),
                            op=op,
                            case_index=case_index,
                            reason=str(exc),
                            raw_case=raw_case,
                        )
                    )
    return cases, invalid


def _shape_family(shape: tuple[int, ...]) -> str:
    return "x".join(str(value) for value in shape)


def _tensor_signature(tensor: YamlTensor) -> dict[str, Any]:
    return {
        "role": tensor.role,
        "dtype": tensor.dtype,
        "rank": len(tensor.shape),
        "shape": list(tensor.shape),
        "storage_shape": None if tensor.storage_shape is None else list(tensor.storage_shape),
        "offset": tensor.offset,
        "permutation": None if tensor.permutation is None else list(tensor.permutation),
    }


def _case_signature(case: YamlCase) -> dict[str, Any]:
    return {
        "inputs": [_tensor_signature(tensor) for tensor in case.inputs],
        "destinations": [_tensor_signature(tensor) for tensor in case.destinations],
        "attributes": case.attributes,
    }


def _attribute_key_paths(value: Any, *, prefix: str = "") -> list[str]:
    if isinstance(value, dict):
        keys: list[str] = []
        for key, inner in sorted(value.items()):
            current = f"{prefix}.{key}" if prefix else str(key)
            keys.append(current)
            keys.extend(_attribute_key_paths(inner, prefix=current))
        return keys
    if isinstance(value, list):
        keys = []
        for index, inner in enumerate(value):
            keys.extend(_attribute_key_paths(inner, prefix=f"{prefix}[]"))
        return keys
    return []


def _surface_for_op(
    op: str,
    cases: list[YamlCase],
    invalid_cases: list[InvalidYamlCase],
    route_count: int,
) -> dict[str, Any]:
    source_counts: Counter[str] = Counter()
    input_counts: Counter[str] = Counter()
    destination_counts: Counter[str] = Counter()
    dtype_combinations: Counter[str] = Counter()
    rank_histogram: Counter[str] = Counter()
    shape_families: Counter[str] = Counter()
    storage_families: Counter[str] = Counter()
    offset_classes: Counter[str] = Counter()
    permutation_families: Counter[str] = Counter()
    attribute_keys: Counter[str] = Counter()
    attribute_values: Counter[str] = Counter()
    signature_counts: Counter[str] = Counter()
    query_ready_count = 0

    for case in cases:
        source_counts[case.source_path] += 1
        input_counts[str(len(case.inputs))] += 1
        destination_counts[str(len(case.destinations))] += 1
        dtypes = {
            "inputs": [tensor.dtype for tensor in case.inputs],
            "destinations": [tensor.dtype for tensor in case.destinations],
        }
        dtype_combinations[_json_key(dtypes)] += 1
        for tensor in (*case.inputs, *case.destinations):
            rank_histogram[str(len(tensor.shape))] += 1
            shape_families[_shape_family(tensor.shape)] += 1
            if tensor.storage_shape is not None:
                storage_families[_shape_family(tensor.storage_shape)] += 1
            if tensor.offset is not None:
                offset_classes[str(tensor.offset)] += 1
            if tensor.permutation is not None:
                permutation_families[_json_key(tensor.permutation)] += 1
        for key in _attribute_key_paths(case.attributes):
            attribute_keys[key] += 1
        if case.attributes:
            attribute_values[_json_key(case.attributes)] += 1
        signature_counts[_json_key(_case_signature(case))] += 1
        if case.inputs and case.destinations:
            query_ready_count += 1

    return {
        "schema": YAML_SURFACE_SCHEMA,
        "op": op,
        "case_count": len(cases),
        "invalid_case_count": len(invalid_cases),
        "route_count": route_count,
        "query_ready_case_count": query_ready_count,
        "source_counts": _histogram(source_counts),
        "input_count_histogram": _histogram(input_counts),
        "destination_count_histogram": _histogram(destination_counts),
        "dtype_combinations": _histogram(dtype_combinations),
        "rank_histogram": _histogram(rank_histogram),
        "shape_families": _histogram(shape_families),
        "storage_shape_families": _histogram(storage_families),
        "offset_classes": _histogram(offset_classes),
        "permutation_families": _histogram(permutation_families),
        "attribute_keys": _histogram(attribute_keys),
        "attribute_values": _histogram(attribute_values),
        "unique_configuration_count": len(signature_counts),
        "repeated_configurations": {
            key: count for key, count in sorted(signature_counts.items()) if count > 1
        },
        "invalid_cases": [case.to_json() for case in invalid_cases],
    }


def _concrete_tensor(tensor: YamlTensor) -> ConcreteTensor:
    stride_source = tensor.storage_shape if tensor.storage_shape is not None else tensor.shape
    strides = contiguous_strides(stride_source)[: len(tensor.shape)]
    dimensions = tuple(
        ConcreteTensorDimension(name=f"d{index}", size=size, stride=strides[index])
        for index, size in enumerate(tensor.shape)
    )
    return ConcreteTensor(
        dtype=tensor.dtype,
        dimensions=dimensions,
        permutation=tensor.permutation,
    )


def _query_tensors(case: YamlCase) -> dict[str, ConcreteTensor]:
    tensors = {
        tensor.role: _concrete_tensor(tensor)
        for tensor in case.inputs
    }
    tensors.update(
        {
            tensor.role: _concrete_tensor(tensor)
            for tensor in case.destinations
        }
    )
    return tensors


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


def _route_tensors_for_case(route: V2Route, case: YamlCase) -> dict[str, ConcreteTensor] | None:
    tensors = _query_tensors(case)
    if not route.synthetic_tensors:
        return tensors
    real_tensor_names = set(route.tensors) - set(route.synthetic_tensors)
    captures = route_captures(route, tensors, tensor_names=real_tensor_names)
    if captures is None:
        return None
    merged = dict(tensors)
    for tensor_name, synthetic in route.synthetic_tensors.items():
        sizes = _synthetic_dimension_sizes(synthetic.dimensions_source, captures)
        if sizes is None:
            return None
        strides = contiguous_strides(sizes)
        merged[tensor_name] = ConcreteTensor(
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
    return merged


def _shape_binding_defaults(
    route: V2Route,
    tensors: dict[str, ConcreteTensor],
    existing: dict[str, int],
) -> dict[str, int]:
    defaults: dict[str, int] = {}
    for binding in route.bindings:
        if binding.source is None or not binding.source.startswith("shape."):
            continue
        key = binding.source.removeprefix("shape.")
        if key in existing or key in defaults:
            continue
        resolved = _resolve_shape_binding_default(key, tensors)
        if resolved is not None:
            defaults[key] = resolved
    return defaults


def _shape_binding_value_defaults(
    route: V2Route,
    tensors: dict[str, ConcreteTensor],
    existing: dict[str, int],
) -> dict[str, int]:
    values = route_values(route, tensors)
    if values is None:
        return {}
    defaults: dict[str, int] = {}
    for binding in route.bindings:
        if binding.source is None or not binding.source.startswith("shape."):
            continue
        key = binding.source.removeprefix("shape.")
        if key in existing or key in defaults:
            continue
        value = values.get(key)
        if isinstance(value, int):
            defaults[key] = int(value)
    return defaults


def _resolve_shape_binding_default(
    key: str,
    tensors: dict[str, ConcreteTensor],
) -> int | None:
    if key.endswith("_stride"):
        base_key = key.removesuffix("_stride")
        split_at = base_key.rfind("_")
        if split_at <= 0:
            return None
        tensor_name = base_key[:split_at]
        dimension_name = base_key[split_at + 1 :]
        tensor = tensors.get(tensor_name)
        if tensor is None:
            return None
        for dimension in tensor.dimensions:
            if dimension.name == dimension_name:
                return int(dimension.stride)
        return None
    split_at = key.rfind("_")
    if split_at <= 0:
        return None
    tensor_name = key[:split_at]
    dimension_name = key[split_at + 1 :]
    tensor = tensors.get(tensor_name)
    if tensor is None:
        return None
    for dimension in tensor.dimensions:
        if dimension.name == dimension_name:
            return int(dimension.size)
    return None


def _shape_for_matched_route(route: V2Route, tensors: dict[str, ConcreteTensor]) -> EncodedRouteShape:
    encoded = dict(encode_route_shape(route, tensors).items)
    encoded.update(_shape_binding_value_defaults(route, tensors, encoded))
    encoded.update(_shape_binding_defaults(route, tensors, encoded))
    return EncodedRouteShape(items=tuple(encoded.items()))


def _route_specificity_score(route: Any, tensors: dict[str, ConcreteTensor]) -> int:
    dimension_ranks: dict[str, int] = {}
    for tensor_name, descriptor in route.tensors.items():
        tensor = tensors.get(tensor_name)
        if tensor is None:
            continue
        dimension_ranks[descriptor.dimensions_capture] = len(tensor.dimensions)

    score = 0
    has_exact_rank_match = False
    has_rank_range_match = False
    for check in route.constraints.checks:
        if check.length is not None and check.name in dimension_ranks:
            if dimension_ranks[check.name] == int(check.length):
                has_exact_rank_match = True
            continue
        if (check.rank_min is not None or check.rank_max is not None) and check.name in dimension_ranks:
            rank = dimension_ranks[check.name]
            if (check.rank_min is None or rank >= int(check.rank_min)) and (
                check.rank_max is None or rank <= int(check.rank_max)
            ):
                has_rank_range_match = True
            continue
        if check.equals:
            score += 100
            continue
        if check.divides:
            score += 10
            continue
        if check.name is not None:
            score += 1
    if has_exact_rank_match:
        score += 500
    if has_rank_range_match:
        score += 250
    score += 25 * len(route.attributes)
    return score


def _select_preferred_routes(
    routes: list[Any],
    tensors: dict[str, ConcreteTensor],
) -> list[Any]:
    if len(routes) <= 1:
        return routes
    scored = [(route, _route_specificity_score(route, tensors)) for route in routes]
    best_score = max(score for _, score in scored)
    return [route for route, score in scored if score == best_score]


def _match_case(case: YamlCase, routes: list[Any]) -> dict[str, Any]:
    tensors = _query_tensors(case)
    tensor_names = set(tensors)
    candidate_matches = []
    route_tensors_by_id: dict[str, dict[str, ConcreteTensor]] = {}
    for route in routes:
        required_tensor_names = set(route.tensors) - set(route.synthetic_tensors)
        if required_tensor_names != tensor_names or not route_accepts_attributes(route, case.attributes):
            continue
        route_tensors = _route_tensors_for_case(route, case)
        if route_tensors is None or not route_accepts_tensors(route, route_tensors):
            continue
        candidate_matches.append(route)
        route_tensors_by_id[route.id] = route_tensors
    matches = _select_preferred_routes(
        candidate_matches,
        route_tensors_by_id[candidate_matches[0].id] if candidate_matches else tensors,
    )
    if len(matches) == 1:
        status = "matched"
    elif matches:
        status = "ambiguous"
    else:
        status = "unmatched"
    return {
        "status": status,
        "source_id": case.source_id,
        "source_path": case.source_path,
        "op": case.op,
        "case_index": case.case_index,
        "surface_signature": _json_key(_case_signature(case)),
        "tensor_names": sorted(tensor_names),
        "candidate_route_count": len(routes),
        "candidate_matched_route_ids": [route.id for route in candidate_matches],
        "matched_route_ids": [route.id for route in matches],
        "case": case.to_json(),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_ordered_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _surface_markdown(surface: dict[str, Any]) -> str:
    lines = [
        f"# YAML Surface: {surface['op']}",
        "",
        f"- Cases: `{surface['case_count']}`",
        f"- Invalid cases: `{surface['invalid_case_count']}`",
        f"- Query-ready cases: `{surface['query_ready_case_count']}`",
        f"- V2 route count: `{surface['route_count']}`",
        f"- Unique configurations: `{surface['unique_configuration_count']}`",
        "",
    ]
    for title, key in (
        ("Input Counts", "input_count_histogram"),
        ("Destination Counts", "destination_count_histogram"),
        ("Ranks", "rank_histogram"),
        ("DTypes", "dtype_combinations"),
        ("Storage Shapes", "storage_shape_families"),
        ("Offsets", "offset_classes"),
        ("Permutations", "permutation_families"),
        ("Attributes", "attribute_keys"),
    ):
        lines.extend([f"## {title}", ""])
        values = surface.get(key, {})
        if not values:
            lines.append("No entries.")
        else:
            lines.extend(["| Value | Count |", "| --- | ---: |"])
            for value, count in values.items():
                lines.append(f"| `{value}` | {count} |")
        lines.append("")
    return "\n".join(lines)


def _summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# YAML Route Import Summary",
        "",
        f"- YAML files: `{len(summary['yaml_paths'])}`",
        f"- Operations: `{summary['operation_count']}`",
        f"- Cases: `{summary['case_count']}`",
        f"- Invalid cases: `{summary['invalid_case_count']}`",
        f"- Matched: `{summary['matched_case_count']}`",
        f"- Ambiguous: `{summary['ambiguous_case_count']}`",
        f"- Unmatched: `{summary['unmatched_case_count']}`",
        f"- Generated benchmark configs: `{summary['generated_config_count']}`",
        "",
        "| Op | Cases | Invalid | Routes | Matched | Ambiguous | Unmatched | Configs |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["operations"]:
        lines.append(
            f"| `{row['op']}` | {row['case_count']} | {row['invalid_case_count']} | "
            f"{row['route_count']} | {row['matched_case_count']} | "
            f"{row['ambiguous_case_count']} | {row['unmatched_case_count']} | "
            f"{row['generated_config_count']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _import_coverage_payload(
    *,
    operation_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    coverage_rows = [
        {
            "op": row["op"],
            "pass_case_count": row["matched_case_count"],
            "fail_case_count": row["case_count"] + row["invalid_case_count"] - row["matched_case_count"],
        }
        for row in sorted(operation_rows, key=lambda current: current["op"])
    ]
    return {
        "schema": IMPORT_TEST_COVERAGE_SCHEMA,
        "operation_count": len(coverage_rows),
        "total_pass_case_count": sum(row["pass_case_count"] for row in coverage_rows),
        "total_fail_case_count": sum(row["fail_case_count"] for row in coverage_rows),
        "operations": coverage_rows,
    }


def _generated_kernel_test_entries(config_paths: list[Path], *, op: str | None = None) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in sorted(config_paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        entry = {
            "config_path": str(path),
            "config_name": path.name,
            "kernel": str(payload.get("kernel") or ""),
            "case_count": len(payload.get("cases", [])),
        }
        route_id = payload.get("route_id")
        if route_id:
            entry["route_id"] = str(route_id)
        if op:
            entry["op"] = op
        entries.append(entry)
    return entries


def _write_generated_kernel_tests_json(
    *,
    yaml_paths: list[Path],
    config_paths: list[Path],
    path: Path,
    op: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": GENERATED_KERNEL_TESTS_SCHEMA,
        "entry_count": len(config_paths),
        "entries": _generated_kernel_test_entries(config_paths, op=op),
    }
    if len(yaml_paths) == 1:
        payload["source_path"] = str(yaml_paths[0])
    else:
        payload["source_paths"] = [str(yaml_path) for yaml_path in yaml_paths]
    if op is not None:
        payload["op"] = op
    _write_ordered_json(path, payload)
    return payload


def _runtime_dtype(dtype: str | None) -> str:
    return str(dtype or "").strip().lower()


def _role_sort_key(role: str) -> tuple[int, int, str]:
    if role.startswith("src") and role[3:].isdigit():
        return (0, int(role[3:]), role)
    if role == "dst":
        return (1, 0, role)
    if role.startswith("dst") and role[3:].isdigit():
        return (1, int(role[3:]), role)
    return (2, 0, role)


def _execution_abi_for_route(route: V2Route) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    position = 0
    for role in sorted(route.tensors, key=_role_sort_key):
        tensor = route.tensors[role]
        kind = "input" if role.startswith("src") else "output"
        entry: dict[str, Any] = {
            "position": position,
            "role": role,
            "kind": kind,
            "dtype": _runtime_dtype(tensor.dtype),
            "fixture": role,
        }
        if kind == "output":
            entry["fixture"] = f"{role}_init" if role != "dst" else "dst_init"
            entry["expect"] = {
                "fixture": "expected" if role == "dst" else f"{role}_expected",
                "mode": "close",
            }
        entries.append(entry)
        position += 1
    return {
        "schema": ROUTE_EXECUTION_ABI_SCHEMA,
        "route_id": route.id,
        "entries": entries,
    }


def _emit_compact_configs(
    *,
    yaml_paths: list[Path],
    op: str,
    cases: list[YamlCase],
    rows: list[dict[str, Any]],
    routes_by_id: Mapping[str, V2Route],
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[str, str, tuple[str, ...]], list[list[int]]] = defaultdict(list)
    seen_cases: dict[tuple[str, str, tuple[str, ...]], set[tuple[int, ...]]] = defaultdict(set)

    for case, row in zip(cases, rows, strict=True):
        if row["status"] != "matched":
            continue
        route_id = row["matched_route_ids"][0]
        route = routes_by_id[route_id]
        route_tensors = _route_tensors_for_case(route, case)
        if route_tensors is None:
            raise RuntimeError(f"matched route {route.id!r} could not materialize route tensors for {case.source_id}")
        shape = _shape_for_matched_route(route, route_tensors)
        params_key = tuple(shape.params)
        values_key = tuple(shape.values)
        group_key = (route.family, route.id, params_key)
        if values_key in seen_cases[group_key]:
            continue
        seen_cases[group_key].add(values_key)
        grouped[group_key].append(shape.values)

    base_key_counts = Counter((kernel_family, route_id) for kernel_family, route_id, _ in grouped)
    emitted: list[Path] = []
    seen_paths: set[Path] = set()
    for (kernel_family, route_id, params_key), cases_for_config in sorted(
        grouped.items(),
        key=lambda item: (item[0][0], item[0][1], item[0][2]),
    ):
        payload: dict[str, Any] = {
            "kernel": kernel_family,
            "params": list(params_key),
            "cases": cases_for_config,
            "route_id": route_id,
            "execution_abi": _execution_abi_for_route(routes_by_id[route_id]),
        }
        filename = _config_filename(
            kernel_family,
            route_id,
            params_key,
            require_params_suffix=base_key_counts[(kernel_family, route_id)] > 1,
        )
        path = output_dir / filename
        if path in seen_paths:
            raise RuntimeError(f"generated config path collision for {path}")
        seen_paths.add(path)
        _write_ordered_json(path, payload)
        emitted.append(path)

    _write_generated_kernel_tests_json(
        yaml_paths=yaml_paths,
        config_paths=emitted,
        path=output_dir.parent / "generated-kernel-tests.json",
        op=op,
    )
    return emitted


def _iteration_log(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Iteration Log",
            "",
            "### 1: Descriptor YAML surface and route report",
            "",
            "- Did:",
            "  - Parsed descriptor YAML inputs.",
            "  - Materialized YAML surface reports.",
            "  - Ran generic v2 route descriptor matching.",
            "- Evidence:",
            "  - `yaml-surface-summary.json`",
            "  - `route-import-summary.json`",
            "  - `route-matches.json`",
            "  - `route-unmatched.json`",
            "- Issues:",
            f"  - Invalid descriptor cases: `{summary['invalid_case_count']}`",
            f"  - Unmatched cases: `{summary['unmatched_case_count']}`",
            f"  - Ambiguous cases: `{summary['ambiguous_case_count']}`",
            "- Classification:",
            "  - `follow_up`" if summary["invalid_case_count"] or summary["unmatched_case_count"] or summary["ambiguous_case_count"] else "  - `none`",
            "- Next:",
            "  - Inspect the first operation surface report and select the first narrow support slice.",
            "",
        ]
    )


def _op_iteration_log(op: str, surface: dict[str, Any], counts: Counter[str]) -> str:
    return "\n".join(
        [
            f"# Iteration Log: {op}",
            "",
            "### 1: Surface inventory and generic route match",
            "",
            "- Did:",
            f"  - Materialized `{op}` YAML surface report.",
            f"  - Ran generic route matching for `{op}`.",
            "- Evidence:",
            "  - `yaml-surface.json`",
            "  - `route-import-summary.json`",
            "  - `route-matches.json`",
            "  - `route-unmatched.json`",
            "- Issues:",
            f"  - Invalid descriptor cases: `{surface['invalid_case_count']}`",
            f"  - Unmatched cases: `{counts.get('unmatched', 0)}`",
            f"  - Ambiguous cases: `{counts.get('ambiguous', 0)}`",
            "- Classification:",
            "  - `follow_up`" if surface["invalid_case_count"] or counts.get("unmatched", 0) or counts.get("ambiguous", 0) else "  - `none`",
            "- Next:",
            "  - Select a narrow support slice from this operation surface report.",
            "",
        ]
    )


def materialize_yaml_route_import(
    yaml_paths: list[Path],
    *,
    output_dir: Path,
    routing_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cases, invalid_cases = load_yaml_cases(yaml_paths)
    catalog = load_route_catalog(routing_dir)
    cases_by_op: dict[str, list[YamlCase]] = defaultdict(list)
    invalid_by_op: dict[str, list[InvalidYamlCase]] = defaultdict(list)
    for case in cases:
        cases_by_op[case.op].append(case)
    for invalid in invalid_cases:
        invalid_by_op[invalid.op].append(invalid)
    all_ops = sorted(set(cases_by_op) | set(invalid_by_op))

    all_matches: list[dict[str, Any]] = []
    all_unmatched: list[dict[str, Any]] = []
    all_generated_config_paths: list[Path] = []
    operation_rows: list[dict[str, Any]] = []
    surfaces: dict[str, dict[str, Any]] = {}

    for op in all_ops:
        op_cases = cases_by_op.get(op, [])
        op_invalid = invalid_by_op.get(op, [])
        op_routes = list(routes_for_op(catalog, op))
        surface = _surface_for_op(op, op_cases, op_invalid, len(op_routes))
        surfaces[op] = surface
        rows = [_match_case(case, op_routes) for case in op_cases]
        counts = Counter(row["status"] for row in rows)
        matches = [row for row in rows if row["status"] == "matched"]
        non_matches = [row for row in rows if row["status"] != "matched"]
        all_matches.extend(matches)
        all_unmatched.extend(non_matches)
        op_dir = output_dir / "ops" / op
        generated_config_paths = _emit_compact_configs(
            yaml_paths=yaml_paths,
            op=op,
            cases=op_cases,
            rows=rows,
            routes_by_id=catalog.routes_by_id,
            output_dir=op_dir / "generated-import-configs",
        )
        all_generated_config_paths.extend(generated_config_paths)
        _write_json(op_dir / "yaml-surface.json", surface)
        _write_text(op_dir / "yaml-surface.md", _surface_markdown(surface))
        op_summary = {
            "schema": YAML_ROUTE_IMPORT_SCHEMA,
            "op": op,
            "case_count": len(op_cases),
            "invalid_case_count": len(op_invalid),
            "route_count": len(op_routes),
            "matched_case_count": counts.get("matched", 0),
            "ambiguous_case_count": counts.get("ambiguous", 0),
            "unmatched_case_count": counts.get("unmatched", 0),
            "generated_config_count": len(generated_config_paths),
            "generated_kernel_tests_path": str(op_dir / "generated-kernel-tests.json"),
            "generated_config_paths": [str(path) for path in generated_config_paths],
        }
        _write_json(op_dir / "route-import-summary.json", op_summary)
        _write_ordered_json(
            op_dir / "import-coverage.json",
            _import_coverage_payload(operation_rows=[op_summary]),
        )
        _write_json(op_dir / "route-matches.json", {"schema": YAML_ROUTE_IMPORT_SCHEMA, "rows": matches})
        _write_json(op_dir / "route-unmatched.json", {"schema": YAML_ROUTE_IMPORT_SCHEMA, "rows": non_matches})
        _write_text(op_dir / "iteration-log.md", _op_iteration_log(op, surface, counts))
        operation_rows.append(op_summary)

    summary = {
        "schema": YAML_ROUTE_IMPORT_SCHEMA,
        "yaml_paths": [str(path) for path in yaml_paths],
        "routing_dir": str(routing_dir),
        "operation_count": len(all_ops),
        "case_count": len(cases),
        "invalid_case_count": len(invalid_cases),
        "matched_case_count": len(all_matches),
        "ambiguous_case_count": sum(1 for row in all_unmatched if row["status"] == "ambiguous"),
        "unmatched_case_count": sum(1 for row in all_unmatched if row["status"] == "unmatched"),
        "generated_config_count": len(all_generated_config_paths),
        "generated_kernel_tests_path": str(output_dir / "generated-kernel-tests.json"),
        "generated_config_paths": [str(path) for path in all_generated_config_paths],
        "operations": sorted(operation_rows, key=lambda row: row["op"]),
    }
    surface_summary = {
        "schema": YAML_SURFACE_SCHEMA,
        "yaml_paths": [str(path) for path in yaml_paths],
        "operation_count": len(all_ops),
        "case_count": len(cases),
        "invalid_case_count": len(invalid_cases),
        "operations": [
            {
                "op": op,
                "case_count": surfaces[op]["case_count"],
                "invalid_case_count": surfaces[op]["invalid_case_count"],
                "query_ready_case_count": surfaces[op]["query_ready_case_count"],
                "route_count": surfaces[op]["route_count"],
                "unique_configuration_count": surfaces[op]["unique_configuration_count"],
            }
            for op in all_ops
        ],
    }
    _write_json(output_dir / "yaml-surface-summary.json", surface_summary)
    _write_text(output_dir / "yaml-surface-summary.md", _summary_markdown(summary))
    _write_json(output_dir / "route-import-summary.json", summary)
    _write_text(output_dir / "route-import-summary.md", _summary_markdown(summary))
    _write_ordered_json(
        output_dir / "import-coverage.json",
        _import_coverage_payload(operation_rows=operation_rows),
    )
    _write_generated_kernel_tests_json(
        yaml_paths=yaml_paths,
        config_paths=all_generated_config_paths,
        path=output_dir / "generated-kernel-tests.json",
    )
    _write_json(output_dir / "route-matches.json", {"schema": YAML_ROUTE_IMPORT_SCHEMA, "rows": all_matches})
    _write_json(output_dir / "route-unmatched.json", {"schema": YAML_ROUTE_IMPORT_SCHEMA, "rows": all_unmatched})
    _write_text(output_dir / "iteration-log.md", _iteration_log(summary))
    return summary
