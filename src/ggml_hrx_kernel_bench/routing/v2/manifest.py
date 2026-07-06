from __future__ import annotations

import hashlib
from pathlib import Path

from .query import RouteCatalog, require_route_catalog


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_manifest(
    *,
    kernel_dir: Path,
    routing_dir: Path | None = None,
    catalog: RouteCatalog | None = None,
    original_root: Path | None = None,
) -> dict[str, object]:
    resolved_catalog = require_route_catalog(routing_dir=routing_dir, catalog=catalog)
    routes_by_kernel: dict[str, list] = {}
    for route in resolved_catalog.routes:
        routes_by_kernel.setdefault(route.kernel_path, []).append(route)
    entries = []
    kernel_files = sorted(str(path.relative_to(kernel_dir)) for path in kernel_dir.rglob("*.loom"))
    for kernel_name in kernel_files:
        kernel_path = kernel_dir / kernel_name
        matching = routes_by_kernel.get(kernel_name, [])
        imported_sha256 = _file_sha256(kernel_path)
        original_path = None
        original_sha256 = None
        if original_root is not None:
            candidate = original_root / "kernels" / kernel_name
            if candidate.exists():
                original_path = str(candidate)
                original_sha256 = _file_sha256(candidate)
        entries.append(
            {
                "path": str(kernel_path),
                "imported_sha256": imported_sha256,
                "original_path": original_path,
                "original_sha256": original_sha256,
                "mechanical_rewrites": (
                    ["content differs from original HRX2 source"]
                    if original_sha256 and original_sha256 != imported_sha256
                    else []
                ),
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
        "catalog_source_count": len({route.source_id for route in resolved_catalog.routes}),
        "route_count": len(resolved_catalog.routes),
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
