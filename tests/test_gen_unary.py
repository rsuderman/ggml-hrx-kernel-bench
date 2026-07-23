from __future__ import annotations

import json
from pathlib import Path

import pytest

from ggml_hrx_kernel_bench.generators import unary
from ggml_hrx_kernel_bench.materialized_assets import materialize_asset_root
from ggml_hrx_kernel_bench.routing.v2.query import load_route_catalog


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


def test_generated_attribute_requirements_are_typed_declarations_with_pins() -> None:
    artifacts = {
        str(path): json.loads(contents)
        for path, contents in unary.render_catalog_artifacts().items()
    }
    leaky_relu = artifacts["leaky_relu/leaky_relu_f32.json"]

    assert leaky_relu["attributes"] == {"negative_slope": {"type": "f32"}}
    assert {"name": "attribute.negative_slope", "value": 0.1} in leaky_relu["constraints"]

    softcap = artifacts["softcap/softcap_f32.json"]
    assert softcap["attributes"] == {"softcap": {"type": "f32"}}
    assert {"name": "attribute.softcap", "value": 50.0} in softcap["constraints"]

    for path, route in artifacts.items():
        for requirement in (route.get("attributes") or {}).values():
            assert isinstance(requirement, dict) and set(requirement) == {"type"}, path


def test_route_catalog_rejects_literal_attribute_requirements(tmp_path: Path) -> None:
    asset_root = materialize_asset_root(tmp_path / "assets", force=True)
    catalog_dir = asset_root / "catalog" / "v2"
    route_path = catalog_dir / "leaky_relu" / "leaky_relu_f32.json"
    route = json.loads(route_path.read_text(encoding="utf-8"))
    route["attributes"]["negative_slope"] = 0.1
    route_path.write_text(json.dumps(route, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="must be a typed declaration"):
        load_route_catalog(catalog_dir)


def test_materialized_v2_routes_include_matching_operation(tmp_path: Path) -> None:
    asset_root = materialize_asset_root(tmp_path / "assets", force=True)
    catalog_dir = asset_root / "catalog" / "v2"
    routes = json.loads((catalog_dir / "router.json").read_text(encoding="utf-8"))["routes"]

    for op, relative_paths in routes.items():
        for relative_path in relative_paths:
            route = json.loads((catalog_dir / relative_path).read_text(encoding="utf-8"))
            assert route["op"] == op, relative_path


def test_route_catalog_rejects_operation_mismatch(tmp_path: Path) -> None:
    asset_root = materialize_asset_root(tmp_path / "assets", force=True)
    catalog_dir = asset_root / "catalog" / "v2"
    route_path = catalog_dir / "clamp" / "clamp_f16.json"
    route = json.loads(route_path.read_text(encoding="utf-8"))
    route["op"] = "SCALE"
    route_path.write_text(json.dumps(route, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="does not match router operation"):
        load_route_catalog(catalog_dir)
