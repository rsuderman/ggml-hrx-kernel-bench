from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


ROUTER_FILENAME = "router.json"


@dataclass(frozen=True)
class V2Route:
    id: str
    family: str
    op: str
    source_id: str
    kernel_path: str
    root_symbol: str
    export_name: str | None
    layout: str
    dtypes: dict[str, str]
    match: dict[str, Any]
    launch: dict[str, Any]
    bindings: tuple[dict[str, str], ...]


def stable_id(*parts: Any, length: int = 10) -> str:
    text = json.dumps(parts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _descriptor_path(routing_dir: Path) -> Path:
    return routing_dir / ROUTER_FILENAME


def load_routes(routing_dir: Path) -> list[V2Route]:
    path = _descriptor_path(routing_dir)
    if not path.exists():
        return []
    data = _load_json(path)
    if not isinstance(data, dict):
        raise RuntimeError(f"v2 routing descriptor must be a JSON object: {path}")
    raw_routes = data.get("routes")
    if not isinstance(raw_routes, list):
        raise RuntimeError(f"v2 routing descriptor must contain a routes array: {path}")
    routes: list[V2Route] = []
    for index, raw in enumerate(raw_routes):
        if not isinstance(raw, dict):
            raise RuntimeError(f"v2 route entry {index} must be a JSON object: {path}")
        kernel = raw.get("kernel") or {}
        types = raw.get("types") or {}
        match = raw.get("match") or {}
        launch = raw.get("launch") or {}
        config = raw.get("config") or {}
        bindings = config.get("bindings") or []
        routes.append(
            V2Route(
                id=str(raw["id"]),
                family=str(raw["family"]),
                op=str(raw["op"]),
                source_id=str(kernel["source_id"]),
                kernel_path=str(kernel["path"]),
                root_symbol=str(kernel["root_symbol"]),
                export_name=(
                    None if kernel.get("export_name") is None else str(kernel["export_name"])
                ),
                layout=str(raw.get("layout") or match.get("layout") or ""),
                dtypes={str(key): str(value).upper() for key, value in dict(types).items()},
                match=dict(match),
                launch=dict(launch),
                bindings=tuple(dict(binding) for binding in bindings),
            )
        )
    return routes


def iter_routes(routing_dir: Path):
    yield from load_routes(routing_dir)


def source_path_for_route(kernel_dir: Path, route: V2Route) -> Path:
    return kernel_dir / route.kernel_path


def route_supports(route: V2Route) -> dict[str, Any]:
    return {
        "src0_type": route.dtypes.get("src0"),
        "src1_type": route.dtypes.get("src1"),
        "dst_type": route.dtypes.get("dst"),
        "layout": route.layout,
    }


def route_accepts_dtype(route: V2Route, dtype: Mapping[str, Any]) -> bool:
    if "type" in dtype:
        actual = str(dtype["type"]).upper()
        return all(route.dtypes.get(key) in {None, actual} for key in ("src0", "src1", "dst"))
    return all(
        expected is None or str(dtype.get(name, "")).upper() == expected
        for name, expected in (
            ("type_src", route.dtypes.get("src0")),
            ("type_dst", route.dtypes.get("dst")),
        )
        if name in dtype
    )


def _bounds_accept(bounds: Mapping[str, Any], value: int) -> bool:
    lower = int(bounds.get("min", value))
    upper = int(bounds.get("max", value))
    return lower <= value <= upper


def route_accepts_shape(route: V2Route, shape: Mapping[str, int]) -> bool:
    if str(route.match.get("kind") or "") != "pointwise_contiguous_2d":
        return False
    ncols = int(shape.get("ncols", shape.get("cols", 0)))
    nrows = int(shape.get("nrows", shape.get("rows", 0)))
    cols = int(shape.get("cols", ncols))
    rows = int(shape.get("rows", nrows))
    if ncols != cols or nrows != rows:
        return False
    return _bounds_accept(dict(route.match.get("ncols") or {}), ncols) and _bounds_accept(
        dict(route.match.get("nrows") or {}), nrows
    )


def default_shape_for_route(route: V2Route) -> dict[str, int]:
    ncols = int((route.match.get("ncols") or {}).get("min", 1))
    nrows = int((route.match.get("nrows") or {}).get("min", 1))
    return {
        "ncols": ncols,
        "nrows": nrows,
        "cols": ncols,
        "rows": nrows,
    }


def _ceil_div(lhs: int, rhs: int) -> int:
    return (lhs + rhs - 1) // rhs


def route_dispatch(route: V2Route, shape: Mapping[str, int]) -> dict[str, Any]:
    rows_per_workgroup = int(route.launch.get("rows_per_workgroup", 1) or 1)
    cols_per_workgroup = int(route.launch.get("cols_per_workgroup", 1) or 1)
    nrows = int(shape.get("nrows", shape.get("rows", 1)))
    ncols = int(shape.get("ncols", shape.get("cols", 1)))
    return {
        "workgroup_count": [
            _ceil_div(nrows, rows_per_workgroup),
            _ceil_div(ncols, cols_per_workgroup),
            1,
        ],
        "workgroup_size": list(route.launch.get("workgroup_size", [None, None, None])),
        "rows_per_workgroup": rows_per_workgroup,
        "cols_per_workgroup": cols_per_workgroup,
        "metadata_source": "route_descriptor_v2",
        "has_static_dispatch_workgroup_count": False,
        "has_static_workgroup_size": bool(route.launch.get("workgroup_size")),
    }


def build_manifest(*, kernel_dir: Path, routing_dir: Path) -> dict[str, object]:
    routes = load_routes(routing_dir)
    routes_by_kernel: dict[str, list[V2Route]] = {}
    for route in routes:
        routes_by_kernel.setdefault(route.kernel_path, []).append(route)
    entries = []
    kernel_files = sorted(path.name for path in kernel_dir.glob("*.loom"))
    for kernel_name in kernel_files:
        matching = routes_by_kernel.get(kernel_name, [])
        entries.append(
            {
                "path": str(kernel_dir / kernel_name),
                "source_ids": sorted({route.source_id for route in matching}),
                "route_count": len(matching),
                "coverage": "route_backed" if matching else "unrouted",
            }
        )
    referenced_kernel_files = sorted(routes_by_kernel)
    return {
        "schema": "ggml_hrx_kernel_bench.routing_manifest.v2",
        "routing_version": "v2",
        "kernel_count": len(kernel_files),
        "catalog_source_count": len({route.source_id for route in routes}),
        "route_count": len(routes),
        "entries": entries,
        "source_ids_without_routes": [],
        "route_source_ids_without_source_entry": [],
        "kernel_files_without_source_entry": sorted(set(kernel_files) - set(referenced_kernel_files)),
        "source_entries_without_kernel_file": sorted(
            kernel_name
            for kernel_name in referenced_kernel_files
            if not (kernel_dir / kernel_name).exists()
        ),
    }
