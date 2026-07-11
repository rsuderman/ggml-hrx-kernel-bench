from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import ggml_hrx_kernel_bench.materialized_assets as materialized_assets
from ggml_hrx_kernel_bench.generators.copy import (
    generated_artifacts,
    generated_catalog_route_paths,
    generator_input_paths,
    render_catalog_artifacts,
    render_kernel_artifacts,
    write_catalog_artifacts,
)
from ggml_hrx_kernel_bench.materialized_assets import (
    ASSET_ROOT_ENV_VAR,
    configured_asset_root,
    default_asset_root,
    materialize_asset_root,
    write_active_asset_root_metadata,
)


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


def test_generated_copy_artifact_plan_orders_flavors_by_priority() -> None:
    artifacts = generated_artifacts()

    first_non_contiguous = next(index for index, artifact in enumerate(artifacts) if artifact.flavor.name == "non_contiguous")

    assert all(artifact.flavor.name == "contiguous" for artifact in artifacts[:first_non_contiguous])
    assert all(artifact.flavor.name == "non_contiguous" for artifact in artifacts[first_non_contiguous:])


def test_copy_generator_input_paths_cover_templates_and_python_sources() -> None:
    input_paths = generator_input_paths()

    assert any(path.name == "copy.py" for path in input_paths)
    assert any(path.name == "copy_common.py" for path in input_paths)
    assert any(path.name == "contiguous_1d.loom.tmpl" for path in input_paths)
    assert any(path.name == "non_contiguous_4d.json.tmpl" for path in input_paths)


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


def test_materialize_asset_root_refreshes_missing_copied_runtime_asset(tmp_path: Path) -> None:
    asset_root = materialize_asset_root(tmp_path / "assets", force=True)
    copied_kernel_path = asset_root / "kernels" / "v2" / "cont" / "contiguous_4d.loom"
    copied_kernel_path.unlink()

    materialize_asset_root(asset_root, force=False)

    assert copied_kernel_path.is_file()


def test_materialize_asset_root_refreshes_router_missing_generated_cpy_routes(tmp_path: Path) -> None:
    asset_root = materialize_asset_root(tmp_path / "assets", force=True)
    router_path = asset_root / "catalog" / "v2" / "router.json"
    router_payload = json.loads(router_path.read_text(encoding="utf-8"))
    router_payload["routes"].pop("CPY", None)
    router_path.write_text(json.dumps(router_payload, indent=2) + "\n", encoding="utf-8")

    materialize_asset_root(asset_root, force=False)

    refreshed_router_payload = json.loads(router_path.read_text(encoding="utf-8"))
    assert refreshed_router_payload["routes"]["CPY"] == list(generated_catalog_route_paths())


def test_materialize_asset_root_rejects_non_empty_unmanaged_output_root(tmp_path: Path) -> None:
    unmanaged_root = tmp_path / "existing-dir"
    unmanaged_root.mkdir()
    (unmanaged_root / "keep.txt").write_text("keep\n", encoding="utf-8")

    with pytest.raises(ValueError, match="refusing to replace non-empty directory"):
        materialize_asset_root(unmanaged_root, force=True)

    assert (unmanaged_root / "keep.txt").read_text(encoding="utf-8") == "keep\n"


def test_default_asset_root_uses_environment_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    asset_root = materialize_asset_root(tmp_path / "assets", force=True)
    monkeypatch.setenv(ASSET_ROOT_ENV_VAR, str(asset_root))

    assert configured_asset_root() == asset_root
    assert default_asset_root() == asset_root


def test_default_asset_root_falls_back_from_stale_metadata_to_default_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    default_asset_root_path = materialize_asset_root(tmp_path / "default-assets", force=True)
    stale_asset_root = tmp_path / "stale-assets"
    metadata_path = tmp_path / "build" / "generated" / "active-asset-root.json"
    write_active_asset_root_metadata(metadata_path, stale_asset_root)
    monkeypatch.delenv(ASSET_ROOT_ENV_VAR, raising=False)
    monkeypatch.setattr(materialized_assets, "ACTIVE_ASSET_ROOT_METADATA_PATH", metadata_path)
    monkeypatch.setattr(materialized_assets, "DEFAULT_ASSET_ROOT", default_asset_root_path)

    assert configured_asset_root() == stale_asset_root.resolve()
    assert default_asset_root() == default_asset_root_path


def test_default_asset_root_uses_active_asset_root_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset_root = materialize_asset_root(tmp_path / "assets", force=True)
    metadata_path = tmp_path / "build" / "generated" / "active-asset-root.json"
    write_active_asset_root_metadata(metadata_path, asset_root)
    monkeypatch.delenv(ASSET_ROOT_ENV_VAR, raising=False)
    monkeypatch.setattr(materialized_assets, "ACTIVE_ASSET_ROOT_METADATA_PATH", metadata_path)

    assert configured_asset_root() == asset_root
    assert default_asset_root() == asset_root


def test_environment_override_wins_over_active_asset_root_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata_asset_root = materialize_asset_root(tmp_path / "metadata-assets", force=True)
    override_asset_root = materialize_asset_root(tmp_path / "override-assets", force=True)
    metadata_path = tmp_path / "build" / "generated" / "active-asset-root.json"
    write_active_asset_root_metadata(metadata_path, metadata_asset_root)
    monkeypatch.setattr(materialized_assets, "ACTIVE_ASSET_ROOT_METADATA_PATH", metadata_path)
    monkeypatch.setenv(ASSET_ROOT_ENV_VAR, str(override_asset_root))

    assert configured_asset_root() == override_asset_root
    assert default_asset_root() == override_asset_root


def test_default_asset_root_requires_prebuilt_assets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    missing_asset_root = tmp_path / "missing-assets"
    monkeypatch.setenv(ASSET_ROOT_ENV_VAR, str(missing_asset_root))

    with pytest.raises(FileNotFoundError, match="Build the CMake target `runtime-assets`"):
        default_asset_root()


def test_materialize_assets_script_publishes_metadata_for_default_asset_discovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset_root = tmp_path / "script-assets"
    metadata_path = tmp_path / "build" / "generated" / "active-asset-root.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/materialize_assets.py",
            "--output",
            str(asset_root),
            "--metadata-output",
            str(metadata_path),
        ],
        check=True,
        cwd=materialized_assets.PROJECT_ROOT,
    )
    monkeypatch.delenv(ASSET_ROOT_ENV_VAR, raising=False)
    monkeypatch.setattr(materialized_assets, "ACTIVE_ASSET_ROOT_METADATA_PATH", metadata_path)

    assert default_asset_root() == asset_root.resolve()


def test_runtime_assets_cmake_target_publishes_metadata_for_default_asset_discovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_dir = tmp_path / "cmake-build"
    asset_root = tmp_path / "cmake-assets"
    metadata_path = tmp_path / "build" / "generated" / "active-asset-root.json"
    subprocess.run(
        [
            "cmake",
            "-S",
            str(materialized_assets.PROJECT_ROOT),
            "-B",
            str(build_dir),
            "-DGGML_HRX_BUILD_LOOM_TOOLS=OFF",
            f"-DGGML_HRX_ASSET_ROOT={asset_root}",
            f"-DGGML_HRX_ACTIVE_ASSET_ROOT_METADATA_PATH={metadata_path}",
        ],
        check=True,
        cwd=materialized_assets.PROJECT_ROOT,
    )
    subprocess.run(
        [
            "cmake",
            "--build",
            str(build_dir),
            "--target",
            "runtime-assets",
        ],
        check=True,
        cwd=materialized_assets.PROJECT_ROOT,
    )
    monkeypatch.delenv(ASSET_ROOT_ENV_VAR, raising=False)
    monkeypatch.setattr(materialized_assets, "ACTIVE_ASSET_ROOT_METADATA_PATH", metadata_path)

    assert default_asset_root() == asset_root.resolve()


def test_default_asset_root_rejects_generated_only_tree_missing_copied_runtime_assets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    incomplete_asset_root = tmp_path / "generated-only-assets"
    incomplete_asset_root.mkdir(parents=True, exist_ok=True)
    (incomplete_asset_root / ".materialized.stamp").write_text("materialized\n", encoding="utf-8")
    for relative_path, contents in render_kernel_artifacts().items():
        artifact_path = incomplete_asset_root / "kernels" / "v2" / relative_path
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(contents, encoding="utf-8")
    for relative_path, contents in render_catalog_artifacts().items():
        artifact_path = incomplete_asset_root / "catalog" / "v2" / relative_path
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(contents, encoding="utf-8")
    router_path = incomplete_asset_root / "catalog" / "v2" / "router.json"
    router_path.parent.mkdir(parents=True, exist_ok=True)
    router_path.write_text(
        json.dumps({"routes": {"CPY": list(generated_catalog_route_paths())}}, indent=2) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(ASSET_ROOT_ENV_VAR, str(incomplete_asset_root))

    with pytest.raises(FileNotFoundError, match="Build the CMake target `runtime-assets`"):
        default_asset_root()


def test_default_asset_root_rejects_incomplete_materialized_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    incomplete_asset_root = tmp_path / "incomplete-assets"
    incomplete_asset_root.mkdir(parents=True, exist_ok=True)
    (incomplete_asset_root / ".materialized.stamp").write_text("materialized\n", encoding="utf-8")
    monkeypatch.setenv(ASSET_ROOT_ENV_VAR, str(incomplete_asset_root))

    with pytest.raises(FileNotFoundError, match="Build the CMake target `runtime-assets`"):
        default_asset_root()
