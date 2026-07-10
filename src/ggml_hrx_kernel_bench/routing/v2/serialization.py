from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import V2Route


def source_path_for_route(kernel_dir: Path, route: V2Route) -> Path:
    return kernel_dir / route.kernel_path


def tensor_descriptors_json(route: V2Route) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for tensor_name, descriptor in route.tensors.items():
        payload[tensor_name] = {
            key: value
            for key, value in (
                ("dtype", descriptor.dtype),
                ("dimensions", descriptor.dimensions_capture),
                ("strides", descriptor.strides_capture),
                ("permutation", descriptor.permutation_capture),
            )
            if value is not None
        }
    return payload


def tensor_values_json(route: V2Route) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for value in route.values:
        if value.operation_kind == "contiguous_strides":
            payload.append({"name": value.name, "contiguous_strides": value.sources[0]})
            continue
        if value.operation_kind == "product":
            payload.append({"name": value.name, "product": value.sources[0]})
            continue
        if value.operation_kind == "inverse_permutation":
            payload.append({"name": value.name, "inverse_permutation": value.sources[0]})
            continue
        if value.operation_kind == "element":
            payload.append(
                {"name": value.name, "element": {"source": value.sources[0], "index": value.parameters[0]}}
            )
            continue
        if value.operation_kind == "head":
            payload.append(
                {"name": value.name, "head": {"source": value.sources[0], "take": value.parameters[0]}}
            )
            continue
        if value.operation_kind == "tail":
            payload.append(
                {"name": value.name, "tail": {"source": value.sources[0], "drop": value.parameters[0]}}
            )
            continue
        if value.operation_kind == "chain_permutations":
            payload.append({"name": value.name, "chain_permutations": list(value.sources)})
            continue
        if value.operation_kind == "permuted_contiguous_strides":
            payload.append(
                {
                    "name": value.name,
                    "permuted_contiguous_strides": {
                        "dimensions": value.sources[0],
                        "permutation": value.sources[1],
                    },
                }
            )
            continue
        raise AssertionError(f"unsupported value operation kind: {value.operation_kind}")
    return payload


def synthetic_tensors_json(route: V2Route) -> dict[str, Any]:
    return {
        tensor_name: {
            "dtype": descriptor.dtype,
            "dimensions": descriptor.dimensions_source,
        }
        for tensor_name, descriptor in route.synthetic_tensors.items()
    }


def tensor_constraints_json(route: V2Route) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for check in route.constraints.checks:
        if check.equals:
            payload.append({"equals": list(check.equals)})
            continue
        if check.divides:
            payload.append({"divides": list(check.divides)})
            continue
        payload.append(
            {
                key: value
                for key, value in (
                    ("name", check.name),
                    ("length", check.length),
                    ("rank_min", check.rank_min),
                    ("rank_max", check.rank_max),
                    ("index", check.index),
                    ("min", check.min),
                    ("max", check.max),
                    ("multiple_of", check.multiple_of),
                    ("iota", True if check.iota else None),
                )
                if value is not None
            }
        )
    return payload


def route_supports(route: V2Route) -> dict[str, Any]:
    return {
        "src0_type": route.tensors.get("src0").dtype if route.tensors.get("src0") else None,
        "src1_type": route.tensors.get("src1").dtype if route.tensors.get("src1") else None,
        "dst_type": route.tensors.get("dst").dtype if route.tensors.get("dst") else None,
        "tensor_captures": {
            tensor_name: {
                key: value
                for key, value in (
                    ("dimensions", descriptor.dimensions_capture),
                    ("strides", descriptor.strides_capture),
                    ("permutation", descriptor.permutation_capture),
                )
                if value is not None
            }
            for tensor_name, descriptor in route.tensors.items()
        },
    }


def route_summary_json(route: V2Route) -> dict[str, Any]:
    return {
        "schema": "ggml_hrx_kernel_bench.routing_route.v2",
        "id": route.id,
        "family": route.family,
        "op": route.op,
        "source_id": route.source_id,
        "root_symbol": route.root_symbol,
        "export_name": route.export_name,
        "tensors": tensor_descriptors_json(route),
        "synthetic_tensors": synthetic_tensors_json(route),
        "attributes": dict(route.attributes),
        "values": tensor_values_json(route),
        "constraints": tensor_constraints_json(route),
        "launch": dict(route.launch),
    }
