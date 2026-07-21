from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

from .route_query_config import ROUTE_QUERY_IMPORT_SCHEMA, materialize_route_query_configs
from .routing.v2.layout import contiguous_strides
from .routing.v2.models import ConcreteTensor, ConcreteTensorDimension
from .routing.v2.query import RouteCatalog, load_route_catalog, routes_for_op
from .routing.v2.selection import (
    RouteQuery,
    select_route_query,
)


YAML_ROUTE_IMPORT_SCHEMA = "ggml_hrx_kernel_bench.yaml_route_import.v1"
YAML_SURFACE_SCHEMA = "ggml_hrx_kernel_bench.yaml_surface.v1"
IMPORT_TEST_COVERAGE_SCHEMA = "ggml_hrx_kernel_bench.import_test_coverage.v1"


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


def _route_query_for_case(case: YamlCase) -> RouteQuery:
    return RouteQuery(
        operation=case.op,
        tensors=_query_tensors(case),
        attributes=case.attributes,
    )


def _match_case(case: YamlCase, catalog: RouteCatalog) -> dict[str, Any]:
    query = _route_query_for_case(case)
    selection = select_route_query(catalog, query)
    routes = routes_for_op(catalog, case.op)
    return {
        "status": selection.status,
        "source_id": case.source_id,
        "source_path": case.source_path,
        "op": case.op,
        "case_index": case.case_index,
        "surface_signature": _json_key(_case_signature(case)),
        "tensor_names": sorted(query.tensors),
        "candidate_route_count": len(routes),
        "candidate_matched_route_ids": list(selection.candidate_route_ids),
        "matched_route_ids": list(selection.route_ids),
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
        f"- Generated import configs: `{summary['generated_config_count']}`",
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


def materialize_yaml_route_queries(
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
    matched_queries: list[RouteQuery] = []
    operation_rows: list[dict[str, Any]] = []
    surfaces: dict[str, dict[str, Any]] = {}

    for op in all_ops:
        op_cases = cases_by_op.get(op, [])
        op_invalid = invalid_by_op.get(op, [])
        op_routes = list(routes_for_op(catalog, op))
        surface = _surface_for_op(op, op_cases, op_invalid, len(op_routes))
        surfaces[op] = surface
        rows = [_match_case(case, catalog) for case in op_cases]
        counts = Counter(row["status"] for row in rows)
        matches = [row for row in rows if row["status"] == "matched"]
        non_matches = [row for row in rows if row["status"] != "matched"]
        all_matches.extend(matches)
        all_unmatched.extend(non_matches)
        matched_queries.extend(
            _route_query_for_case(case)
            for case, row in zip(op_cases, rows, strict=True)
            if row["status"] == "matched"
        )
        op_dir = output_dir / "ops" / op
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
            "generated_config_count": 0,
            "generated_kernel_tests_path": str(op_dir / "generated-kernel-tests.json"),
            "generated_config_paths": [],
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
        "generated_config_count": 0,
        "generated_kernel_tests_path": str(output_dir / "generated-kernel-tests.json"),
        "generated_config_paths": [],
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
    _write_json(output_dir / "route-matches.json", {"schema": YAML_ROUTE_IMPORT_SCHEMA, "rows": all_matches})
    _write_json(output_dir / "route-unmatched.json", {"schema": YAML_ROUTE_IMPORT_SCHEMA, "rows": all_unmatched})
    _write_text(output_dir / "iteration-log.md", _iteration_log(summary))

    query_path = output_dir / "route-queries.jsonl"
    query_path.write_text(
        "".join(
            json.dumps(query.to_json(), separators=(",", ":")) + "\n"
            for query in matched_queries
        ),
        encoding="utf-8",
    )
    metadata = {
        "schema": ROUTE_QUERY_IMPORT_SCHEMA,
        "yaml_paths": [str(path) for path in yaml_paths],
        "routing_dir": str(routing_dir),
        "query_path": str(query_path),
        "query_count": len(matched_queries),
        "operation_count": len(operation_rows),
        "operations": [
            {
                "op": row["op"],
                "case_count": row["case_count"],
                "invalid_case_count": row["invalid_case_count"],
                "matched_case_count": row["matched_case_count"],
                "ambiguous_case_count": row["ambiguous_case_count"],
                "unmatched_case_count": row["unmatched_case_count"],
            }
            for row in sorted(operation_rows, key=lambda row: row["op"])
        ],
    }
    _write_json(output_dir / "route-query-import.json", metadata)
    return summary


def materialize_yaml_route_import(
    yaml_paths: list[Path],
    *,
    output_dir: Path,
    routing_dir: Path,
) -> dict[str, Any]:
    materialize_yaml_route_queries(
        yaml_paths,
        output_dir=output_dir,
        routing_dir=routing_dir,
    )
    return materialize_route_query_configs(
        output_dir / "route-queries.jsonl",
        metadata_path=output_dir / "route-query-import.json",
        output_dir=output_dir,
        routing_dir=routing_dir,
    )
