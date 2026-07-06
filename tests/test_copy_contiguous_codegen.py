from __future__ import annotations

from pathlib import Path

from ggml_hrx_kernel_bench.generators.copy_contiguous import (
    render_catalog_artifacts,
    render_kernel_artifacts,
)
from ggml_hrx_kernel_bench.materialized_assets import materialize_asset_root


def test_checked_in_copy_route_descriptors_match_generator() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    for relative_path, expected_contents in render_catalog_artifacts().items():
        artifact_path = repo_root / "catalog" / "v2" / relative_path
        assert artifact_path.read_text(encoding="utf-8") == expected_contents, str(relative_path)


def test_materialized_v2_kernels_include_generated_contiguous_copy(tmp_path: Path) -> None:
    asset_root = materialize_asset_root(tmp_path / "assets", force=True)
    kernel_dir = asset_root / "kernels" / "v2"

    for relative_path, expected_contents in render_kernel_artifacts().items():
        artifact_path = kernel_dir / relative_path
        assert artifact_path.read_text(encoding="utf-8") == expected_contents, str(relative_path)

    for relative_path, expected_contents in render_catalog_artifacts().items():
        artifact_path = asset_root / "catalog" / "v2" / relative_path
        assert artifact_path.read_text(encoding="utf-8") == expected_contents, str(relative_path)
