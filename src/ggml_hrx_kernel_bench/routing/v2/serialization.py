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
            "dtype": descriptor.dtype,
            "dimensions": descriptor.dimensions_capture,
            "strides": descriptor.strides_capture,
            "permutation": descriptor.permutation_capture,
        }
    return payload


def tensor_values_json(route: V2Route) -> list[dict[str, Any]]:
    return [
        {
            key: entry
            for key, entry in (
                ("name", value.name),
                ("contiguous_strides", value.contiguous_strides),
                ("product", value.product),
            )
            if entry is not None
        }
        for value in route.values
    ]


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
                "dimensions": descriptor.dimensions_capture,
                "strides": descriptor.strides_capture,
                "permutation": descriptor.permutation_capture,
            }
            for tensor_name, descriptor in route.tensors.items()
        },
    }


def route_json(route: V2Route) -> dict[str, Any]:
    return {
        "schema": "ggml_hrx_kernel_bench.routing_route.v2",
        "id": route.id,
        "family": route.family,
        "op": route.op,
        "source_id": route.source_id,
        "root_symbol": route.root_symbol,
        "export_name": route.export_name,
        "tensors": tensor_descriptors_json(route),
        "values": tensor_values_json(route),
        "constraints": tensor_constraints_json(route),
        "launch": dict(route.launch),
    }
