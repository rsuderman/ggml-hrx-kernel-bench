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
    compute = render_compute_block(variant, src_index="%src0_idx", dst_index="%linear")
    return (
        load_kernel_template("non_contiguous_4d.loom.tmpl")
        .substitute(
            export_name=variant.non_contiguous_export_name,
            root_symbol=variant.non_contiguous_root_symbol,
            src_type=variant.src.loom_type,
            dst_type=variant.dst.loom_type,
            compute_block=compute,
        )
        .rstrip()
        + "\n"
    )


def render_kernel_artifacts() -> dict[Path, str]:
    return {
        variant.non_contiguous_kernel_path: _render_kernel_artifact(variant)
        for variant in supported_variants()
    }


def render_route_json(variant: CopyKernelVariant) -> str:
    return load_route_template("non_contiguous_4d.json.tmpl").substitute(
        route_id=variant.non_contiguous_route_id,
        family=variant.family,
        kernel_path=variant.non_contiguous_kernel_path.as_posix(),
        root_symbol=variant.non_contiguous_root_symbol,
        export_name=variant.non_contiguous_export_name,
        src_dtype=variant.src.route_dtype,
        dst_dtype=variant.dst.route_dtype,
        workgroup_size=WORKGROUP_SIZE,
    )


def render_catalog_artifacts() -> dict[Path, str]:
    return {
        variant.non_contiguous_route_path.relative_to(Path("catalog") / "v2"): render_route_json(variant)
        for variant in supported_variants()
    }
