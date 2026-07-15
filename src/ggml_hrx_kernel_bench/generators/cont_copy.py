from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .copy_common import (
    WORKGROUP_SIZE,
    CopyKernelVariant,
    load_route_template,
    supported_variants,
)
from .utils import load_template, repo_root


@dataclass(frozen=True)
class ContCopyRouteFlavor:
    route_suffix: str
    route_template_name: str
    kernel_route_suffix: str

    def route_id_for(self, variant: CopyKernelVariant) -> str:
        return f"cont_{variant.family}_{self.route_suffix}"

    def kernel_route_id_for(self, variant: CopyKernelVariant) -> str:
        return f"{variant.family}_{self.kernel_route_suffix}"

    def catalog_relative_path_for(self, variant: CopyKernelVariant) -> Path:
        return Path("cont") / f"{self.route_id_for(variant)}.json"

    def kernel_relative_path_for(self, variant: CopyKernelVariant) -> Path:
        return Path("copy") / f"{self.kernel_route_id_for(variant)}.loom"

    def root_symbol_for(self, variant: CopyKernelVariant) -> str:
        return f"@{self.kernel_route_id_for(variant)}"

    def export_name_for(self, variant: CopyKernelVariant) -> str:
        return self.kernel_route_id_for(variant)


NON_CONTIGUOUS_4D = ContCopyRouteFlavor(
    route_suffix="non_contiguous_4d",
    route_template_name="non_contiguous_4d.json.tmpl",
    kernel_route_suffix="non_contiguous_4d",
)
STORAGE_4D = ContCopyRouteFlavor(
    route_suffix="storage_4d",
    route_template_name="copy_storage_4d.json.tmpl",
    kernel_route_suffix="non_contiguous_4d",
)
I32_CONTIGUOUS_1D = ContCopyRouteFlavor(
    route_suffix="contiguous_1d",
    route_template_name="contiguous_1d.json.tmpl",
    kernel_route_suffix="contiguous_1d",
)


@dataclass(frozen=True)
class GeneratedContCopyRoute:
    variant: CopyKernelVariant
    flavor: ContCopyRouteFlavor

    @property
    def route_id(self) -> str:
        return self.flavor.route_id_for(self.variant)

    @property
    def catalog_relative_path(self) -> Path:
        return self.flavor.catalog_relative_path_for(self.variant)

    @property
    def kernel_relative_path(self) -> Path:
        return self.flavor.kernel_relative_path_for(self.variant)

    @property
    def root_symbol(self) -> str:
        return self.flavor.root_symbol_for(self.variant)

    @property
    def export_name(self) -> str:
        return self.flavor.export_name_for(self.variant)


def _same_dtype_variants() -> tuple[CopyKernelVariant, ...]:
    return tuple(variant for variant in supported_variants() if variant.src == variant.dst)


def generated_artifacts() -> tuple[GeneratedContCopyRoute, ...]:
    same_dtype = _same_dtype_variants()
    i32_variants = tuple(variant for variant in same_dtype if variant.src.name == "i32")
    return (
        *(GeneratedContCopyRoute(variant, NON_CONTIGUOUS_4D) for variant in same_dtype),
        *(GeneratedContCopyRoute(variant, STORAGE_4D) for variant in same_dtype),
        *(GeneratedContCopyRoute(variant, I32_CONTIGUOUS_1D) for variant in i32_variants),
    )


def _route_template(flavor: ContCopyRouteFlavor):
    if flavor.route_template_name == "copy_storage_4d.json.tmpl":
        return load_template("catalog", "v2", "cont", flavor.route_template_name)
    return load_route_template(flavor.route_template_name)


def _render_route_json(artifact: GeneratedContCopyRoute) -> str:
    variant = artifact.variant
    return _route_template(artifact.flavor).substitute(
        route_id=artifact.route_id,
        family=variant.family,
        kernel_path=artifact.kernel_relative_path.as_posix(),
        root_symbol=artifact.root_symbol,
        export_name=artifact.export_name,
        src_dtype=variant.src.route_dtype,
        dst_dtype=variant.dst.route_dtype,
        dtype=variant.src.route_dtype,
        workgroup_size=WORKGROUP_SIZE,
    )


def render_catalog_artifacts() -> dict[Path, str]:
    return {artifact.catalog_relative_path: _render_route_json(artifact) for artifact in generated_artifacts()}


def router_route_list() -> list[str]:
    generated_paths = [artifact.catalog_relative_path.as_posix() for artifact in generated_artifacts()]
    return [*generated_paths, "cont/cont_f32_contiguous_4d.json"]


def generator_input_paths() -> tuple[Path, ...]:
    return (
        Path(__file__).resolve(),
        repo_root() / "catalog" / "v2" / "cont" / "copy_storage_4d.json.tmpl",
        repo_root() / "catalog" / "v2" / "copy" / "non_contiguous_4d.json.tmpl",
        repo_root() / "catalog" / "v2" / "copy" / "contiguous_1d.json.tmpl",
    )
