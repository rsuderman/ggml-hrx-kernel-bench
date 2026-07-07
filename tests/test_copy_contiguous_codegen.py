from __future__ import annotations

import json
from pathlib import Path

from ggml_hrx_kernel_bench.generators.copy import (
    generated_catalog_route_paths,
    render_catalog_artifacts,
    render_kernel_artifacts,
    write_catalog_artifacts,
)
from ggml_hrx_kernel_bench.materialized_assets import materialize_asset_root


def test_copy_route_writer_emits_expected_descriptors(tmp_path: Path) -> None:
    catalog_root = tmp_path / "catalog" / "v2"
    written_paths = write_catalog_artifacts(catalog_root)

    assert len(written_paths) == len(render_catalog_artifacts())
    for relative_path, expected_contents in render_catalog_artifacts().items():
        artifact_path = catalog_root / relative_path
        assert artifact_path in written_paths
        assert artifact_path.read_text(encoding="utf-8") == expected_contents, str(relative_path)


def test_generated_copy_descriptors_do_not_serialize_lowering_metadata() -> None:
    for contents in render_catalog_artifacts().values():
        payload = json.loads(contents)
        assert "lowering" not in payload


def test_materialized_v2_kernels_include_generated_contiguous_copy(tmp_path: Path) -> None:
    asset_root = materialize_asset_root(tmp_path / "assets", force=True)
    kernel_dir = asset_root / "kernels" / "v2"

    for relative_path, expected_contents in render_kernel_artifacts().items():
        artifact_path = kernel_dir / relative_path
        assert artifact_path.read_text(encoding="utf-8") == expected_contents, str(relative_path)

    for relative_path, expected_contents in render_catalog_artifacts().items():
        artifact_path = asset_root / "catalog" / "v2" / relative_path
        assert artifact_path.read_text(encoding="utf-8") == expected_contents, str(relative_path)

    router_payload = json.loads(
        (asset_root / "catalog" / "v2" / "router.json").read_text(encoding="utf-8")
    )
    assert router_payload["routes"]["CPY"] == list(generated_catalog_route_paths())
