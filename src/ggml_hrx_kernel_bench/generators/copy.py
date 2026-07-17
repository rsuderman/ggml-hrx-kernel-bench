from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .copy_common import (
    COPY_ROUTE_FLAVORS,
    WORKGROUP_SIZE,
    CopyKernelVariant,
    CopyRouteFlavor,
    load_kernel_template,
    load_route_template,
    render_compute_block,
    repo_root,
    supported_variants,
)


@dataclass(frozen=True)
class GeneratedCopyArtifact:
    variant: CopyKernelVariant
    flavor: CopyRouteFlavor

    @property
    def route_id(self) -> str:
        return self.flavor.route_id_for(self.variant)

    @property
    def kernel_relative_path(self) -> Path:
        return self.flavor.kernel_relative_path_for(self.variant)

    @property
    def catalog_relative_path(self) -> Path:
        return self.flavor.catalog_relative_path_for(self.variant)

    @property
    def root_symbol(self) -> str:
        return self.flavor.root_symbol_for(self.variant)

    @property
    def export_name(self) -> str:
        return self.flavor.export_name_for(self.variant)


def generated_artifacts() -> tuple[GeneratedCopyArtifact, ...]:
    return tuple(
        GeneratedCopyArtifact(variant=variant, flavor=flavor)
        for flavor in COPY_ROUTE_FLAVORS
        for variant in supported_variants()
    )


def generator_input_paths() -> tuple[Path, ...]:
    generator_dir = repo_root() / "src" / "ggml_hrx_kernel_bench" / "generators"
    kernel_template_dir = repo_root() / "kernels" / "v2" / "copy"
    route_template_dir = repo_root() / "catalog" / "v2" / "copy"
    return tuple(sorted(generator_dir.glob("*.py"))) + tuple(sorted(kernel_template_dir.glob("*.tmpl"))) + tuple(
        sorted(route_template_dir.glob("*.tmpl"))
    )


def _render_kernel_artifact(artifact: GeneratedCopyArtifact) -> str:
    return (
        load_kernel_template(artifact.flavor.kernel_template_name)
        .substitute(
            export_name=artifact.export_name,
            root_symbol=artifact.root_symbol,
            src_type=artifact.variant.src.loom_type,
            dst_type=artifact.variant.dst.loom_type,
            compute_block=render_compute_block(
                artifact.variant,
                src_index=artifact.flavor.src_index_name,
                dst_index="%linear",
            ),
        )
        .rstrip()
        + "\n"
    )


def _render_route_json(artifact: GeneratedCopyArtifact) -> str:
    return load_route_template(artifact.flavor.route_template_name).substitute(
        route_id=artifact.route_id,
        family=artifact.variant.family,
        op="CPY",
        kernel_path=artifact.kernel_relative_path.as_posix(),
        root_symbol=artifact.root_symbol,
        export_name=artifact.export_name,
        src_dtype=artifact.variant.src.route_dtype,
        dst_dtype=artifact.variant.dst.route_dtype,
        workgroup_size=WORKGROUP_SIZE,
    )


def render_kernel_artifacts() -> dict[Path, str]:
    return {artifact.kernel_relative_path: _render_kernel_artifact(artifact) for artifact in generated_artifacts()}


def render_catalog_artifacts() -> dict[Path, str]:
    return {artifact.catalog_relative_path: _render_route_json(artifact) for artifact in generated_artifacts()}


def generated_catalog_route_paths() -> tuple[str, ...]:
    return tuple(artifact.catalog_relative_path.as_posix() for artifact in generated_artifacts())


def write_catalog_artifacts(catalog_root: Path) -> tuple[Path, ...]:
    written_paths: list[Path] = []
    for relative_path, contents in render_catalog_artifacts().items():
        path = catalog_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
        written_paths.append(path)
    return tuple(written_paths)
