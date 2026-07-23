from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from .routing.v2.layout import EncodedRouteShape, encode_route_shape
from .routing.v2.matching import route_values
from .routing.v2.models import ConcreteTensor, V2Route
from .routing.v2.query import RouteCatalog, load_route_catalog
from .routing.v2.selection import (
    RouteQuery,
    materialize_route_query_tensors,
    select_route_query,
)


ROUTE_QUERY_IMPORT_SCHEMA = "ggml_hrx_kernel_bench.route_query_import.v1"
GENERATED_KERNEL_TESTS_SCHEMA = "ggml_hrx_kernel_bench.generated_kernel_tests.v1"
ROUTE_EXECUTION_ABI_SCHEMA = "ggml_hrx_kernel_bench.route_execution_abi.v1"

STATIC_SCALAR_ABI_BY_FAMILY: dict[str, tuple[dict[str, Any], ...]] = {
    "scale_f32": (
        {"role": "scale", "dtype": "f32", "value": 0.625},
        {"role": "bias", "dtype": "f32", "value": -0.125},
    ),
    "clamp_f32": (
        {"role": "min", "dtype": "f32", "value": -0.45},
        {"role": "max", "dtype": "f32", "value": 0.55},
    ),
    "clamp_f16": (
        {"role": "min", "dtype": "f32", "value": -0.45},
        {"role": "max", "dtype": "f32", "value": 0.55},
    ),
    "rope_f32": (
        {"role": "theta_scale", "dtype": "f32", "value": 0.75},
    ),
    "rope_neox_f32": (
        {"role": "theta_scale", "dtype": "f32", "value": 0.75},
    ),
    "rope_f16": (
        {"role": "theta_scale", "dtype": "f32", "value": 0.75},
    ),
    "rope_neox_f16": (
        {"role": "theta_scale", "dtype": "f32", "value": 0.75},
    ),
}
ATTRIBUTE_SCALAR_ABI_BY_FAMILY: dict[str, tuple[dict[str, Any], ...]] = {
    "rms_norm_f32": (
        {"role": "eps", "dtype": "f32", "attribute": "eps"},
    ),
    "soft_max_f32": (
        {"role": "scale", "dtype": "f32", "attribute": "scale", "default": 0.75},
    ),
    "rope_f32": (
        {"role": "freq_scale", "dtype": "f32", "attribute": "freq_scale", "default": 1.1},
        {"role": "attn_factor", "dtype": "f32", "attribute": "attn_factor", "default": 0.9},
    ),
    "rope_neox_f32": (
        {"role": "freq_scale", "dtype": "f32", "attribute": "freq_scale", "default": 1.1},
        {"role": "attn_factor", "dtype": "f32", "attribute": "attn_factor", "default": 0.9},
    ),
    "rope_f16": (
        {"role": "freq_scale", "dtype": "f32", "attribute": "freq_scale", "default": 1.1},
        {"role": "attn_factor", "dtype": "f32", "attribute": "attn_factor", "default": 0.9},
    ),
    "rope_neox_f16": (
        {"role": "freq_scale", "dtype": "f32", "attribute": "freq_scale", "default": 1.1},
        {"role": "attn_factor", "dtype": "f32", "attribute": "attn_factor", "default": 0.9},
    ),
    "flash_attn_ext_f32_f16": (
        {
            "role": "scale",
            "dtype": "f32",
            "attribute": "scale",
            "default": 0.08838834764831843,
        },
    ),
    "softmax_kqv_f32_f16": (
        {"role": "scale", "dtype": "f32", "attribute": "scale", "default": 0.75},
    ),
    "leaky_relu_f32": (
        {"role": "negative_slope", "dtype": "f32", "attribute": "negative_slope", "default": 0.1},
    ),
    "leaky_relu_f16": (
        {"role": "negative_slope", "dtype": "f32", "attribute": "negative_slope", "default": 0.1},
    ),
    "softcap_f32": (
        {"role": "softcap", "dtype": "f32", "attribute": "softcap", "default": 50.0},
    ),
}
FIXTURE_BY_FAMILY_ROLE: dict[tuple[str, str], str] = {
    ("get_rows_f32", "src1"): "indices",
    ("rms_norm_f32", "src0"): "src",
    ("soft_max_f32", "mask"): "mask",
    ("rope_f32", "src1"): "positions",
    ("rope_neox_f32", "src1"): "positions",
    ("rope_f16", "src1"): "positions",
    ("rope_neox_f16", "src1"): "positions",
}
INPUT_ROLES_BY_FAMILY: dict[str, frozenset[str]] = {
    "soft_max_f32": frozenset({"mask"}),
}


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


def _shape_binding_attribute_defaults(
    route: V2Route,
    attributes: Mapping[str, Any] | None,
    existing: dict[str, int],
) -> dict[str, int]:
    if attributes is None:
        return {}
    defaults: dict[str, int] = {}
    for binding in route.bindings:
        if binding.source is None or not binding.source.startswith("attribute."):
            continue
        if not binding.key.startswith("@shape."):
            continue
        shape_key = binding.key.removeprefix("@shape.")
        if shape_key in existing or shape_key in defaults:
            continue
        attribute_key = binding.source.removeprefix("attribute.")
        value = attributes.get(attribute_key)
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        defaults[shape_key] = int(value)
    return defaults


def _shape_binding_value_defaults(
    route: V2Route,
    tensors: dict[str, ConcreteTensor],
    existing: dict[str, int],
    attributes: Mapping[str, Any] | None = None,
) -> dict[str, int]:
    values = route_values(route, tensors, attributes)
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


def _shape_for_matched_route(
    route: V2Route,
    tensors: dict[str, ConcreteTensor],
    attributes: Mapping[str, Any] | None = None,
) -> EncodedRouteShape:
    encoded = dict(encode_route_shape(route, tensors).items)
    encoded.update(_shape_binding_value_defaults(route, tensors, encoded, attributes))
    encoded.update(_shape_binding_attribute_defaults(route, attributes, encoded))
    encoded.update(_shape_binding_defaults(route, tensors, encoded))
    return EncodedRouteShape(items=tuple(encoded.items()))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_ordered_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _generated_kernel_test_entries(
    config_paths: list[Path],
    *,
    op: str | None = None,
) -> list[dict[str, Any]]:
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


def _attribute_scalar_values_for_route(
    route: V2Route,
    attributes: Mapping[str, Any],
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for scalar in ATTRIBUTE_SCALAR_ABI_BY_FAMILY.get(route.family, ()):
        attribute_name = str(scalar["attribute"])
        if attribute_name in attributes:
            values[attribute_name] = attributes[attribute_name]
        elif "default" in scalar:
            values[attribute_name] = scalar["default"]
    return values


def _attribute_scalar_key(route: V2Route, attributes: Mapping[str, Any]) -> str:
    return _json_key(_attribute_scalar_values_for_route(route, attributes))


def _role_sort_key(role: str) -> tuple[int, int, str]:
    if role.startswith("src") and role[3:].isdigit():
        return (0, int(role[3:]), role)
    if role == "mask":
        return (0, 1000, role)
    if role == "dst":
        return (1, 0, role)
    if role.startswith("dst") and role[3:].isdigit():
        return (1, int(role[3:]), role)
    return (2, 0, role)


def _execution_abi_for_route(
    route: V2Route,
    attributes: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if route.family in {"set_rows_f32", "cont_set_rows_f32"}:
        update_role, index_role = (
            ("src1", "src2") if "src2" in route.tensors else ("src0", "src1")
        )
        output_dtype = "f16" if route.family == "cont_set_rows_f32" else "f32"
        return {
            "schema": ROUTE_EXECUTION_ABI_SCHEMA,
            "route_id": route.id,
            "entries": [
                {
                    "position": 0,
                    "role": update_role,
                    "kind": "input",
                    "dtype": "f32",
                    "fixture": "src0",
                },
                {
                    "position": 1,
                    "role": index_role,
                    "kind": "input",
                    "dtype": "i32",
                    "fixture": "indices",
                },
                {
                    "position": 2,
                    "role": "dst",
                    "kind": "output",
                    "dtype": output_dtype,
                    "fixture": "dst_init",
                    "expect": {
                        "fixture": "expected",
                        "mode": "close",
                    },
                },
            ],
        }
    entries: list[dict[str, Any]] = []
    position = 0
    for scalar in STATIC_SCALAR_ABI_BY_FAMILY.get(route.family, ()):
        entries.append(
            {
                "position": position,
                "role": scalar["role"],
                "kind": "scalar",
                "dtype": scalar["dtype"],
                "value": scalar["value"],
            }
        )
        position += 1
    attribute_values = _attribute_scalar_values_for_route(route, attributes or {})
    for scalar in ATTRIBUTE_SCALAR_ABI_BY_FAMILY.get(route.family, ()):
        attribute_name = str(scalar["attribute"])
        if attribute_name not in attribute_values:
            continue
        entries.append(
            {
                "position": position,
                "role": scalar["role"],
                "kind": "scalar",
                "dtype": scalar["dtype"],
                "value": attribute_values[attribute_name],
            }
        )
        position += 1
    for role in sorted(route.tensors, key=_role_sort_key):
        tensor = route.tensors[role]
        kind = (
            "input"
            if role.startswith("src")
            or role in INPUT_ROLES_BY_FAMILY.get(route.family, frozenset())
            else "output"
        )
        entry: dict[str, Any] = {
            "position": position,
            "role": role,
            "kind": kind,
            "dtype": _runtime_dtype(tensor.dtype),
            "fixture": FIXTURE_BY_FAMILY_ROLE.get((route.family, role), role),
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


def _load_json_object(path: Path, *, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"missing {description}: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {description} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must be a JSON object: {path}")
    return payload


def _load_import_metadata(path: Path) -> dict[str, Any]:
    metadata = _load_json_object(path, description="route query import metadata")
    if metadata.get("schema") != ROUTE_QUERY_IMPORT_SCHEMA:
        raise ValueError(
            f"route query import metadata schema must be {ROUTE_QUERY_IMPORT_SCHEMA!r}: {path}"
        )
    if not isinstance(metadata.get("yaml_paths"), list) or any(
        not isinstance(item, str) for item in metadata["yaml_paths"]
    ):
        raise ValueError(f"route query import metadata yaml_paths must be an array of strings: {path}")
    if not isinstance(metadata.get("operations"), list) or any(
        not isinstance(item, dict) or not isinstance(item.get("op"), str)
        for item in metadata["operations"]
    ):
        raise ValueError(f"route query import metadata operations must be an array of objects: {path}")
    query_count = metadata.get("query_count")
    if isinstance(query_count, bool) or not isinstance(query_count, int) or query_count < 0:
        raise ValueError(f"route query import metadata query_count must be non-negative: {path}")
    return metadata


def _read_route_queries_jsonl(path: Path) -> list[tuple[int, RouteQuery]]:
    rows: list[tuple[int, RouteQuery]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise ValueError(f"missing RouteQuery JSONL: {path}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
        try:
            query = RouteQuery.from_json(payload)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
        rows.append((line_number, query))
    return rows


def _canonical_query(query: RouteQuery) -> RouteQuery:
    return RouteQuery(
        operation=query.operation,
        tensors={
            role: query.tensors[role]
            for role in sorted(query.tensors, key=_role_sort_key)
        },
        attributes=query.attributes,
    )


def _clean_generated_outputs(output_dir: Path) -> None:
    ops_dir = output_dir / "ops"
    if ops_dir.is_dir():
        for op_dir in ops_dir.iterdir():
            if not op_dir.is_dir():
                continue
            config_dir = op_dir / "generated-import-configs"
            if config_dir.exists():
                shutil.rmtree(config_dir)
            manifest_path = op_dir / "generated-kernel-tests.json"
            if manifest_path.exists():
                manifest_path.unlink()
    manifest_path = output_dir / "generated-kernel-tests.json"
    if manifest_path.exists():
        manifest_path.unlink()


def _emit_grouped_configs(
    *,
    grouped: Mapping[tuple[str, str, tuple[str, ...], str], list[list[int]]],
    group_attributes: Mapping[
        tuple[str, str, tuple[str, ...], str],
        dict[str, Any],
    ],
    routes_by_id: Mapping[str, V2Route],
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base_key_counts = Counter(
        (kernel_family, route_id)
        for kernel_family, route_id, _, _ in grouped
    )
    emitted: list[Path] = []
    seen_paths: set[Path] = set()
    for (kernel_family, route_id, params_key, attribute_key), cases_for_config in sorted(
        grouped.items(),
        key=lambda item: (item[0][0], item[0][1], item[0][2], item[0][3]),
    ):
        group_key = (kernel_family, route_id, params_key, attribute_key)
        payload: dict[str, Any] = {
            "kernel": kernel_family,
            "params": list(params_key),
            "cases": cases_for_config,
            "route_id": route_id,
            "execution_abi": _execution_abi_for_route(
                routes_by_id[route_id],
                attributes=group_attributes[group_key],
            ),
        }
        filename = _config_filename(
            kernel_family,
            route_id,
            params_key,
            require_params_suffix=base_key_counts[(kernel_family, route_id)] > 1,
        )
        if attribute_key != "{}":
            digest = hashlib.sha1(attribute_key.encode("utf-8")).hexdigest()[:12]
            filename = f"{Path(filename).stem}.{digest}.json"
        path = output_dir / filename
        if path in seen_paths:
            raise RuntimeError(f"generated config path collision for {path}")
        seen_paths.add(path)
        _write_ordered_json(path, payload)
        emitted.append(path)
    return emitted


def _group_route_queries(
    *,
    query_rows: list[tuple[int, RouteQuery]],
    query_path: Path,
    catalog: RouteCatalog,
    operation_names: set[str],
) -> tuple[
    dict[str, dict[tuple[str, str, tuple[str, ...], str], list[list[int]]]],
    dict[str, dict[tuple[str, str, tuple[str, ...], str], dict[str, Any]]],
]:
    grouped_by_op: dict[
        str,
        dict[tuple[str, str, tuple[str, ...], str], list[list[int]]],
    ] = defaultdict(lambda: defaultdict(list))
    attributes_by_op: dict[
        str,
        dict[tuple[str, str, tuple[str, ...], str], dict[str, Any]],
    ] = defaultdict(dict)
    seen_by_op: dict[
        str,
        dict[tuple[str, str, tuple[str, ...], str], set[tuple[int, ...]]],
    ] = defaultdict(lambda: defaultdict(set))

    for line_number, raw_query in query_rows:
        query = _canonical_query(raw_query)
        selection = select_route_query(catalog, query)
        if selection.status != "matched":
            raise RuntimeError(f"{query_path}:{line_number}: RouteQuery does not match a route")
        route_id = selection.route_ids[0]
        route = catalog.routes_by_id[route_id]
        op = route.op
        if op not in operation_names:
            raise RuntimeError(
                f"{query_path}:{line_number}: routed operation {op!r} is absent from import metadata"
            )
        route_tensors = materialize_route_query_tensors(route, query)
        if route_tensors is None:
            raise RuntimeError(
                f"{query_path}:{line_number}: matched route {route.id!r} could not materialize tensors"
            )
        shape = _shape_for_matched_route(route, route_tensors, query.attributes)
        params_key = tuple(shape.params)
        values_key = tuple(shape.values)
        attribute_key = _attribute_scalar_key(route, query.attributes)
        group_key = (route.family, route.id, params_key, attribute_key)
        if values_key in seen_by_op[op][group_key]:
            continue
        seen_by_op[op][group_key].add(values_key)
        attributes_by_op[op][group_key] = _attribute_scalar_values_for_route(
            route,
            query.attributes,
        )
        grouped_by_op[op][group_key].append(shape.values)
    return (
        {op: dict(groups) for op, groups in grouped_by_op.items()},
        {op: dict(groups) for op, groups in attributes_by_op.items()},
    )


def materialize_route_query_configs(
    query_jsonl_path: Path,
    *,
    metadata_path: Path,
    output_dir: Path,
    routing_dir: Path,
) -> dict[str, Any]:
    metadata = _load_import_metadata(metadata_path)
    query_rows = _read_route_queries_jsonl(query_jsonl_path)
    if len(query_rows) != metadata["query_count"]:
        raise ValueError(
            f"{query_jsonl_path}: expected {metadata['query_count']} RouteQuery row(s), "
            f"found {len(query_rows)}"
        )
    metadata_routing_dir = metadata.get("routing_dir")
    if not isinstance(metadata_routing_dir, str):
        raise ValueError(f"route query import metadata routing_dir must be a string: {metadata_path}")
    if Path(metadata_routing_dir).resolve() != routing_dir.resolve():
        raise ValueError(
            f"routing directory does not match import metadata: "
            f"{routing_dir} != {metadata_routing_dir}"
        )

    catalog = load_route_catalog(routing_dir)
    operation_names = {str(row["op"]) for row in metadata["operations"]}
    grouped_by_op, attributes_by_op = _group_route_queries(
        query_rows=query_rows,
        query_path=query_jsonl_path,
        catalog=catalog,
        operation_names=operation_names,
    )
    _clean_generated_outputs(output_dir)

    yaml_paths = [Path(raw_path) for raw_path in metadata["yaml_paths"]]
    summary_path = output_dir / "route-import-summary.json"
    summary = _load_json_object(summary_path, description="route import summary")
    summary_operations = summary.get("operations")
    if not isinstance(summary_operations, list):
        raise ValueError(f"route import summary operations must be an array: {summary_path}")
    summary_by_op = {
        str(row.get("op")): row
        for row in summary_operations
        if isinstance(row, dict) and isinstance(row.get("op"), str)
    }
    if set(summary_by_op) != operation_names:
        raise ValueError("route import summary operations do not match import metadata")

    all_generated_config_paths: list[Path] = []
    final_operation_rows: list[dict[str, Any]] = []
    for op in sorted(operation_names):
        op_dir = output_dir / "ops" / op
        generated_config_paths = _emit_grouped_configs(
            grouped=grouped_by_op.get(op, {}),
            group_attributes=attributes_by_op.get(op, {}),
            routes_by_id=catalog.routes_by_id,
            output_dir=op_dir / "generated-import-configs",
        )
        all_generated_config_paths.extend(generated_config_paths)
        _write_generated_kernel_tests_json(
            yaml_paths=yaml_paths,
            config_paths=generated_config_paths,
            path=op_dir / "generated-kernel-tests.json",
            op=op,
        )
        op_summary = dict(summary_by_op[op])
        op_summary.update(
            {
                "generated_config_count": len(generated_config_paths),
                "generated_kernel_tests_path": str(op_dir / "generated-kernel-tests.json"),
                "generated_config_paths": [str(path) for path in generated_config_paths],
            }
        )
        _write_json(op_dir / "route-import-summary.json", op_summary)
        final_operation_rows.append(op_summary)

    summary.update(
        {
            "generated_config_count": len(all_generated_config_paths),
            "generated_kernel_tests_path": str(output_dir / "generated-kernel-tests.json"),
            "generated_config_paths": [str(path) for path in all_generated_config_paths],
            "operations": final_operation_rows,
        }
    )
    _write_generated_kernel_tests_json(
        yaml_paths=yaml_paths,
        config_paths=all_generated_config_paths,
        path=output_dir / "generated-kernel-tests.json",
    )
    _write_json(summary_path, summary)
    summary_markdown = _summary_markdown(summary)
    _write_text(output_dir / "route-import-summary.md", summary_markdown)
    _write_text(output_dir / "yaml-surface-summary.md", summary_markdown)
    return summary
