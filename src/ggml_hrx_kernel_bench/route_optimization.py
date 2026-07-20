from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .routing.v2.query import load_route_catalog
from .routing.v2.serialization import route_summary_json, source_path_for_route


ROUTE_INVENTORY_SCHEMA = "ggml_hrx_kernel_bench.route_inventory.v1"
DEFAULT_MODEL_IMPORT_DIR = Path(
    "build/tests/models/artifacts/Llama-3.3-8B-Instruct.Q8_0-route-import-v2"
)


def route_inventory_payload(
    *,
    op: str,
    generated_import_dir: Path | None,
    routing_dir: Path,
    kernel_dir: Path,
    target: str,
    repo_root: Path,
) -> dict[str, Any]:
    normalized_op = op.strip().upper()
    if normalized_op != "FLASH_ATTN_EXT":
        raise ValueError("route-inventory currently supports only FLASH_ATTN_EXT")

    op_dir = resolve_op_import_dir(
        generated_import_dir or repo_root / DEFAULT_MODEL_IMPORT_DIR,
        op=normalized_op,
    )
    matches = load_json_object(op_dir / "route-matches.json")
    unmatched = load_json_object(op_dir / "route-unmatched.json")
    summary = load_optional_json_object(op_dir / "route-import-summary.json")
    generated_tests = load_optional_json_object(op_dir / "generated-kernel-tests.json")
    catalog = load_route_catalog(routing_dir)

    rows = matches.get("rows", [])
    if not isinstance(rows, list):
        raise ValueError(f"route matches rows must be an array: {op_dir / 'route-matches.json'}")

    cases = [inventory_case(row, catalog, kernel_dir=kernel_dir) for row in rows]
    route_ids = sorted({route_id for case in cases for route_id in case["matching_route_ids"]})

    return {
        "schema": ROUTE_INVENTORY_SCHEMA,
        "op": normalized_op,
        "target": target,
        "generated_import_dir": str(op_dir.parent),
        "op_import_dir": str(op_dir),
        "routing_dir": str(routing_dir),
        "kernel_dir": str(kernel_dir),
        "summary": summary,
        "generated_kernel_tests": generated_tests,
        "case_count": len(cases),
        "selected_route_ids": sorted(
            {case["selected_route_id"] for case in cases if case["selected_route_id"]}
        ),
        "matching_route_ids": route_ids,
        "unmatched_rows": unmatched.get("rows", []),
        "cases": cases,
    }


def resolve_op_import_dir(generated_import_dir: Path, *, op: str) -> Path:
    candidate = generated_import_dir.resolve()
    if candidate.name == op and (candidate / "route-matches.json").is_file():
        return candidate
    op_dir = candidate / "ops" / op
    if (op_dir / "route-matches.json").is_file():
        return op_dir
    direct_op_dir = candidate / op
    if (direct_op_dir / "route-matches.json").is_file():
        return direct_op_dir
    raise FileNotFoundError(
        f"could not find route-matches.json for {op} under {generated_import_dir}"
    )


def inventory_case(
    row: dict[str, Any],
    catalog: Any,
    *,
    kernel_dir: Path,
) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise ValueError("route match row must be a JSON object")
    matching_route_ids = string_list(row.get("matched_route_ids"))
    selected_route_id = matching_route_ids[0] if matching_route_ids else None
    return {
        "case_index": row.get("case_index"),
        "source_id": row.get("source_id"),
        "source_path": row.get("source_path"),
        "status": row.get("status"),
        "selected_route_id": selected_route_id,
        "matching_route_ids": matching_route_ids,
        "candidate_matched_route_ids": string_list(row.get("candidate_matched_route_ids")),
        "case": row.get("case"),
        "routes": [
            inventory_route(route_id, catalog, kernel_dir=kernel_dir)
            for route_id in matching_route_ids
        ],
    }


def inventory_route(route_id: str, catalog: Any, *, kernel_dir: Path) -> dict[str, Any]:
    route = catalog.routes_by_id.get(route_id)
    if route is None:
        return {
            "route_id": route_id,
            "status": "missing_catalog_route",
        }
    source_path = source_path_for_route(kernel_dir, route)
    return {
        "route_id": route.id,
        "family": route.family,
        "source_id": route.source_id,
        "source_path": str(source_path),
        "source_exists": source_path.is_file(),
        "root_symbol": route.root_symbol,
        "export_name": route.export_name,
        "route": json_value(route_summary_json(route)),
    }


def write_route_inventory(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_json_object(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def load_optional_json_object(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return load_json_object(path)


def string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [entry for entry in value if isinstance(entry, str) and entry]


def json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): json_value(inner) for key, inner in value.items()}
    if isinstance(value, tuple | list):
        return [json_value(entry) for entry in value]
    return value
