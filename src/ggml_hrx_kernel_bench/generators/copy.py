from __future__ import annotations

from pathlib import Path

from . import copy_contiguous, copy_non_contiguous


def _merge_artifacts(*artifact_sets: dict[Path, str]) -> dict[Path, str]:
    merged: dict[Path, str] = {}
    for artifacts in artifact_sets:
        for relative_path, contents in artifacts.items():
            if relative_path in merged:
                raise RuntimeError(f"duplicate generated artifact path: {relative_path}")
            merged[relative_path] = contents
    return merged


def render_kernel_artifacts() -> dict[Path, str]:
    return _merge_artifacts(
        copy_contiguous.render_kernel_artifacts(),
        copy_non_contiguous.render_kernel_artifacts(),
    )


def render_catalog_artifacts() -> dict[Path, str]:
    return _merge_artifacts(
        copy_contiguous.render_catalog_artifacts(),
        copy_non_contiguous.render_catalog_artifacts(),
    )


def generated_catalog_route_paths() -> tuple[str, ...]:
    return tuple(relative_path.as_posix() for relative_path in render_catalog_artifacts())


def write_catalog_artifacts(catalog_root: Path) -> tuple[Path, ...]:
    written_paths: list[Path] = []
    for relative_path, contents in render_catalog_artifacts().items():
        path = catalog_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
        written_paths.append(path)
    return tuple(written_paths)
