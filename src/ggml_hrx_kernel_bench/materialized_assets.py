from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from .generators.copy import (
    generated_catalog_route_paths,
    generator_input_paths,
    render_catalog_artifacts,
    render_kernel_artifacts,
)
from .generators.cont_copy import (
    generator_input_paths as cont_copy_generator_input_paths,
    render_catalog_artifacts as cont_copy_render_catalog_artifacts,
    router_route_list as cont_copy_router_route_list,
)
from .generators.unary import (
    generator_input_paths as unary_generator_input_paths,
    render_catalog_artifacts as unary_render_catalog_artifacts,
    render_kernel_artifacts as unary_render_kernel_artifacts,
    router_route_lists as unary_router_route_lists,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASSET_ROOT = PROJECT_ROOT / "build" / "generated" / "assets"
ACTIVE_ASSET_ROOT_METADATA_PATH = (
    PROJECT_ROOT / "build" / "generated" / "active-asset-root.json"
)
ASSET_ROOT_ENV_VAR = "GGML_HRX_ASSET_ROOT"
SUPPORTED_VERSIONS = ("v2",)
ASSET_DIR_NAMES: dict[str, str] = {
    "v2": "v2",
}

SOURCE_KERNEL_DIRS: dict[str, Path] = {
    "v2": PROJECT_ROOT / "kernels" / "v2",
}
SOURCE_ROUTING_DIRS: dict[str, Path] = {
    "v2": PROJECT_ROOT / "catalog" / "v2",
}


def _asset_stamp_path(output_root: Path) -> Path:
    return output_root / ".materialized.stamp"


def _load_active_asset_root() -> Path | None:
    metadata_path = ACTIVE_ASSET_ROOT_METADATA_PATH
    if not metadata_path.is_file():
        return None
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid active asset metadata at {metadata_path}: expected JSON object")
    asset_root = payload.get("asset_root")
    if not isinstance(asset_root, str) or not asset_root:
        raise RuntimeError(f"invalid active asset metadata at {metadata_path}: missing asset_root")
    return Path(asset_root).expanduser().resolve()


def write_active_asset_root_metadata(metadata_path: Path, asset_root: Path) -> Path:
    resolved_metadata_path = metadata_path.resolve()
    resolved_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_metadata_path.write_text(
        json.dumps(
            {
                "schema": "ggml_hrx_kernel_bench.active_asset_root.v1",
                "asset_root": str(asset_root.resolve()),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return resolved_metadata_path


def configured_asset_root() -> Path:
    override = os.environ.get(ASSET_ROOT_ENV_VAR)
    if override:
        return Path(override).expanduser().resolve()
    active_asset_root = _load_active_asset_root()
    if active_asset_root is not None:
        return active_asset_root
    return DEFAULT_ASSET_ROOT


def _default_asset_root_candidates() -> tuple[Path, ...]:
    active_asset_root = _load_active_asset_root()
    if active_asset_root is None or active_asset_root == DEFAULT_ASSET_ROOT:
        return (DEFAULT_ASSET_ROOT,)
    return (active_asset_root, DEFAULT_ASSET_ROOT)


def _copied_kernel_paths() -> tuple[Path, ...]:
    paths: list[Path] = []
    for version in SUPPORTED_VERSIONS:
        asset_dir_name = _asset_dir_name(version)
        for source_path in sorted(SOURCE_KERNEL_DIRS[version].rglob("*.loom")):
            paths.append(
                Path("kernels") / asset_dir_name / source_path.relative_to(SOURCE_KERNEL_DIRS[version])
            )
    return tuple(paths)


def _copied_catalog_paths() -> tuple[Path, ...]:
    paths: list[Path] = []
    for version in SUPPORTED_VERSIONS:
        asset_dir_name = _asset_dir_name(version)
        for source_path in sorted(SOURCE_ROUTING_DIRS[version].rglob("*.json")):
            paths.append(
                Path("catalog") / asset_dir_name / source_path.relative_to(SOURCE_ROUTING_DIRS[version])
            )
    return tuple(paths)


def _generated_kernel_paths() -> tuple[Path, ...]:
    return tuple(
        Path("kernels") / "v2" / relative_path
        for relative_path in (*render_kernel_artifacts(), *unary_render_kernel_artifacts())
    )


def _generated_catalog_paths() -> tuple[Path, ...]:
    return tuple(
        Path("catalog") / "v2" / relative_path
        for relative_path in (*render_catalog_artifacts(), *cont_copy_render_catalog_artifacts(), *unary_render_catalog_artifacts())
    )


def _v2_router_path(output_root: Path) -> Path:
    return output_root / "catalog" / "v2" / "router.json"


def _validate_materialized_v2_router(output_root: Path) -> None:
    router_path = _v2_router_path(output_root)
    payload = json.loads(router_path.read_text(encoding="utf-8"))
    routes = payload.get("routes")
    if not isinstance(routes, dict):
        raise RuntimeError(f"invalid v2 router payload at {router_path}")
    if routes.get("CPY") != list(generated_catalog_route_paths()):
        raise FileNotFoundError(
            f"materialized v2 router is missing generated CPY routes at {router_path}"
        )
    if routes.get("CONT") != cont_copy_router_route_list():
        raise FileNotFoundError(
            f"materialized v2 router is missing generated CONT copy routes at {router_path}"
        )
    for op_key, route_paths in unary_router_route_lists().items():
        if routes.get(op_key) != route_paths:
            raise FileNotFoundError(
                f"materialized v2 router is missing generated {op_key} routes at {router_path}"
            )


def _validate_materialized_asset_root(output_root: Path) -> None:
    stamp_path = _asset_stamp_path(output_root)
    if not stamp_path.is_file():
        raise FileNotFoundError(f"missing materialized asset stamp: {stamp_path}")
    expected_paths = (
        *tuple(output_root / relative_path for relative_path in _copied_kernel_paths()),
        *tuple(output_root / relative_path for relative_path in _copied_catalog_paths()),
        *tuple(output_root / relative_path for relative_path in _generated_kernel_paths()),
        *tuple(output_root / relative_path for relative_path in _generated_catalog_paths()),
    )
    missing_paths = [path for path in expected_paths if not path.is_file()]
    if missing_paths:
        missing_text = ", ".join(str(path) for path in missing_paths[:3])
        if len(missing_paths) > 3:
            missing_text += ", ..."
        raise FileNotFoundError(f"materialized asset root is incomplete: missing {missing_text}")
    _validate_materialized_v2_router(output_root)


def _materialization_inputs() -> list[Path]:
    inputs: list[Path] = []
    for version in SUPPORTED_VERSIONS:
        inputs.extend(sorted(SOURCE_KERNEL_DIRS[version].rglob("*.loom")))
        inputs.extend(sorted(SOURCE_ROUTING_DIRS[version].rglob("*.json")))
    inputs.append(Path(__file__))
    inputs.extend(generator_input_paths())
    inputs.extend(cont_copy_generator_input_paths())
    inputs.extend(unary_generator_input_paths())
    return inputs


def _needs_refresh(output_root: Path, *, force: bool) -> bool:
    if force:
        return True
    stamp_path = _asset_stamp_path(output_root)
    if not stamp_path.is_file():
        return True
    try:
        _validate_materialized_asset_root(output_root)
    except FileNotFoundError:
        return True
    stamp_mtime = stamp_path.stat().st_mtime
    return any(path.stat().st_mtime > stamp_mtime for path in _materialization_inputs() if path.exists())


def _validate_output_root_destination(output_root: Path) -> None:
    destination_root = output_root.resolve()
    source_roots = {path.resolve() for path in (*SOURCE_KERNEL_DIRS.values(), *SOURCE_ROUTING_DIRS.values())}
    if destination_root in source_roots:
        raise ValueError("refusing to materialize assets in-place")
    if not destination_root.exists():
        return
    if destination_root.is_file():
        raise ValueError(f"refusing to replace file with materialized assets: {destination_root}")
    stamp_path = _asset_stamp_path(destination_root)
    if stamp_path.is_file():
        return
    if any(destination_root.iterdir()):
        raise ValueError(
            "refusing to replace non-empty directory without materialized asset stamp: "
            f"{destination_root}"
        )


def materialize_asset_root(output_root: Path, *, force: bool = False) -> Path:
    destination_root = output_root.resolve()
    _validate_output_root_destination(destination_root)
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

    for relative_path, contents in cont_copy_render_catalog_artifacts().items():
        destination_path = destination_root / "catalog" / "v2" / relative_path
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        destination_path.write_text(contents, encoding="utf-8")

    for relative_path, contents in unary_render_kernel_artifacts().items():
        destination_path = destination_root / "kernels" / "v2" / relative_path
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        destination_path.write_text(contents, encoding="utf-8")

    for relative_path, contents in unary_render_catalog_artifacts().items():
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
    routes["CONT"] = cont_copy_router_route_list()
    for op_key, route_paths in unary_router_route_lists().items():
        routes[op_key] = route_paths
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
    override = os.environ.get(ASSET_ROOT_ENV_VAR)
    if override:
        asset_root = Path(override).expanduser().resolve()
        try:
            _validate_materialized_asset_root(asset_root)
        except FileNotFoundError:
            raise FileNotFoundError(
                "default runtime assets are not materialized at "
                f"{asset_root}. Build the CMake target `runtime-assets` or set "
                f"{ASSET_ROOT_ENV_VAR} to a materialized asset root."
            ) from None
        return asset_root

    attempted_roots: list[Path] = []
    for asset_root in _default_asset_root_candidates():
        attempted_roots.append(asset_root)
        try:
            _validate_materialized_asset_root(asset_root)
            return asset_root
        except FileNotFoundError:
            continue
    attempted_text = ", ".join(str(path) for path in attempted_roots)
    raise FileNotFoundError(
        "default runtime assets are not materialized. Checked "
        f"{attempted_text}. Build the CMake target `runtime-assets` or set "
        f"{ASSET_ROOT_ENV_VAR} to a materialized asset root."
    )


def default_kernel_dir(version: str) -> Path:
    return default_asset_root() / "kernels" / version_dir_name(version)


def default_routing_dir(version: str) -> Path:
    return default_asset_root() / "catalog" / version_catalog_dir_name(version)
