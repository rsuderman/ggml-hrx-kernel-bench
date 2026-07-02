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
            "dimensions": [dimension.name for dimension in descriptor.dimensions],
            "strides": [stride.name for stride in descriptor.stride_ids],
        }
    return payload


def tensor_constraints_json(route: V2Route) -> list[dict[str, Any]]:
    return [
        {
            key: value
            for key, value in (
                ("name", check.identifier),
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
        "src0_type": route.tensors.get("src0").dtype if route.tensors.get("src0") else None,
        "src1_type": route.tensors.get("src1").dtype if route.tensors.get("src1") else None,
        "dst_type": route.tensors.get("dst").dtype if route.tensors.get("dst") else None,
        "tensor_orders": {
            tensor_name: [dimension.name for dimension in descriptor.dimensions]
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
        "constraints": tensor_constraints_json(route),
        "launch": dict(route.launch),
    }
