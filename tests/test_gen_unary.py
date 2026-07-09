from __future__ import annotations

import json
from pathlib import Path

from ggml_hrx_kernel_bench.generators import unary
from ggml_hrx_kernel_bench.materialized_assets import materialize_asset_root


def test_materialized_v2_includes_generated_unary_kernels_and_routes(tmp_path: Path) -> None:
    asset_root = materialize_asset_root(tmp_path / "assets", force=True)
    kernel_dir = asset_root / "kernels" / "v2"
    catalog_dir = asset_root / "catalog" / "v2"

    for relative_path, expected in unary.render_kernel_artifacts().items():
        assert (kernel_dir / relative_path).read_text(encoding="utf-8") == expected, str(relative_path)
    for relative_path, expected in unary.render_catalog_artifacts().items():
        assert (catalog_dir / relative_path).read_text(encoding="utf-8") == expected, str(relative_path)

    routes = json.loads((catalog_dir / "router.json").read_text(encoding="utf-8"))["routes"]
    for op_key, paths in unary.router_route_lists().items():
        assert routes[op_key] == paths, op_key


def test_generated_unary_routes_omit_lowering_metadata() -> None:
    for contents in unary.render_catalog_artifacts().values():
        assert "lowering" not in json.loads(contents)
