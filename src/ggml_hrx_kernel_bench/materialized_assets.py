from __future__ import annotations

import json
import shutil
from pathlib import Path

from .generators.copy_contiguous import (
    generated_catalog_route_paths,
    render_catalog_artifacts,
    render_kernel_artifacts,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASSET_ROOT = PROJECT_ROOT / "build" / "generated" / "assets"
SUPPORTED_VERSIONS = ("v1", "v2")
ASSET_DIR_NAMES: dict[str, str] = {
    "v1": "hrx2",
    "v2": "v2",
}

SOURCE_KERNEL_DIRS: dict[str, Path] = {
    "v1": PROJECT_ROOT / "kernels" / "hrx2",
    "v2": PROJECT_ROOT / "kernels" / "v2",
}
SOURCE_ROUTING_DIRS: dict[str, Path] = {
    "v1": PROJECT_ROOT / "catalog" / "hrx2",
    "v2": PROJECT_ROOT / "catalog" / "v2",
}


def _asset_stamp_path(output_root: Path) -> Path:
    return output_root / ".materialized.stamp"


def _generated_kernel_paths() -> tuple[Path, ...]:
    return tuple(Path("kernels") / "v2" / relative_path for relative_path in render_kernel_artifacts())


def _generated_catalog_paths() -> tuple[Path, ...]:
    return tuple(Path("catalog") / "v2" / relative_path for relative_path in render_catalog_artifacts())


def _v2_router_path(output_root: Path) -> Path:
    return output_root / "catalog" / "v2" / "router.json"


def _materialization_inputs() -> list[Path]:
    inputs: list[Path] = []
    for version in SUPPORTED_VERSIONS:
        inputs.extend(sorted(SOURCE_KERNEL_DIRS[version].rglob("*.loom")))
        inputs.extend(sorted(SOURCE_ROUTING_DIRS[version].rglob("*.json")))
    inputs.extend(
        [
            SOURCE_KERNEL_DIRS["v2"] / "copy" / "contiguous_1d.loom.tmpl",
            SOURCE_KERNEL_DIRS["v2"] / "copy" / "non_contiguous_4d.loom.tmpl",
            SOURCE_ROUTING_DIRS["v2"] / "copy" / "contiguous_1d.json.tmpl",
            SOURCE_ROUTING_DIRS["v2"] / "copy" / "non_contiguous_4d.json.tmpl",
            Path(__file__),
            Path(__file__).resolve().parent / "generators" / "copy_contiguous.py",
        ]
    )
    return inputs


def _needs_refresh(output_root: Path, *, force: bool) -> bool:
    if force:
        return True
    stamp_path = _asset_stamp_path(output_root)
    if not stamp_path.is_file():
        return True
    stamp_mtime = stamp_path.stat().st_mtime
    expected_paths = _generated_kernel_paths() + _generated_catalog_paths()
    if any(not (output_root / relative_path).is_file() for relative_path in expected_paths):
        return True
    return any(path.stat().st_mtime > stamp_mtime for path in _materialization_inputs() if path.exists())


def materialize_asset_root(output_root: Path, *, force: bool = False) -> Path:
    destination_root = output_root.resolve()
    source_roots = {path.resolve() for path in (*SOURCE_KERNEL_DIRS.values(), *SOURCE_ROUTING_DIRS.values())}
    if destination_root in source_roots:
        raise ValueError("refusing to materialize assets in-place")
    if not _needs_refresh(destination_root, force=force):
        return destination_root
    if destination_root.exists():
        shutil.rmtree(destination_root)
    destination_root.mkdir(parents=True, exist_ok=True)

    for version in SUPPORTED_VERSIONS:
        asset_dir_name = _asset_dir_name(version)
        for source_path in sorted(SOURCE_KERNEL_DIRS[version].rglob("*.loom")):
            relative_path = (
                Path("kernels")
                / asset_dir_name
                / source_path.relative_to(SOURCE_KERNEL_DIRS[version])
            )
            destination_path = destination_root / relative_path
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination_path)
        for source_path in sorted(SOURCE_ROUTING_DIRS[version].rglob("*.json")):
            relative_path = (
                Path("catalog")
                / asset_dir_name
                / source_path.relative_to(SOURCE_ROUTING_DIRS[version])
            )
            destination_path = destination_root / relative_path
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination_path)

    for relative_path, contents in render_kernel_artifacts().items():
        destination_path = destination_root / "kernels" / "v2" / relative_path
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        destination_path.write_text(contents, encoding="utf-8")

    for relative_path, contents in render_catalog_artifacts().items():
        destination_path = destination_root / "catalog" / "v2" / relative_path
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        destination_path.write_text(contents, encoding="utf-8")

    _write_materialized_v2_router(destination_root)
    _asset_stamp_path(destination_root).write_text("materialized\n", encoding="utf-8")
    return destination_root


def _write_materialized_v2_router(destination_root: Path) -> None:
    router_path = _v2_router_path(destination_root)
    payload = json.loads(router_path.read_text(encoding="utf-8"))
    routes = payload.get("routes")
    if not isinstance(routes, dict):
        raise RuntimeError(f"invalid v2 router payload at {router_path}")
    routes["CPY"] = list(generated_catalog_route_paths())
    router_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def version_dir_name(version: str) -> str:
    return _asset_dir_name(version)


def version_catalog_dir_name(version: str) -> str:
    return version_dir_name(version)


def _asset_dir_name(version: str) -> str:
    try:
        return ASSET_DIR_NAMES[version]
    except KeyError as exc:
        raise ValueError(f"unsupported routing version: {version}") from exc


def default_asset_root() -> Path:
    return materialize_asset_root(DEFAULT_ASSET_ROOT)


def default_kernel_dir(version: str) -> Path:
    return default_asset_root() / "kernels" / version_dir_name(version)


def default_routing_dir(version: str) -> Path:
    return default_asset_root() / "catalog" / version_catalog_dir_name(version)
