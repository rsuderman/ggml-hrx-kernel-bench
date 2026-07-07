from __future__ import annotations

from pathlib import Path

from .copy_common import (
    WORKGROUP_SIZE,
    CopyKernelVariant,
    load_kernel_template,
    load_route_template,
    render_compute_block,
    supported_variants,
)


def _render_kernel_artifact(variant: CopyKernelVariant) -> str:
    return (
        load_kernel_template("contiguous_1d.loom.tmpl")
        .substitute(
            export_name=variant.export_name,
            root_symbol=variant.root_symbol,
            src_type=variant.src.loom_type,
            dst_type=variant.dst.loom_type,
            compute_block=render_compute_block(variant, src_index="%linear", dst_index="%linear"),
        )
        .rstrip()
        + "\n"
    )


def render_route_json(variant: CopyKernelVariant) -> str:
    return load_route_template("contiguous_1d.json.tmpl").substitute(
        route_id=variant.route_id,
        family=variant.family,
        kernel_path=variant.kernel_path.as_posix(),
        root_symbol=variant.root_symbol,
        export_name=variant.export_name,
        src_dtype=variant.src.route_dtype,
        dst_dtype=variant.dst.route_dtype,
        workgroup_size=WORKGROUP_SIZE,
    )


def render_kernel_artifacts() -> dict[Path, str]:
    return {variant.kernel_path: _render_kernel_artifact(variant) for variant in supported_variants()}


def render_catalog_artifacts() -> dict[Path, str]:
    return {
        variant.route_path.relative_to(Path("catalog") / "v2"): render_route_json(variant)
        for variant in supported_variants()
    }
