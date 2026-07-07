from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from string import Template


WORKGROUP_SIZE = 256
KERNEL_PREAMBLE = (
    "config.decl @hrx2.shape.copy.n : %value: index where [range(%value, 1, 1073741824)]\n"
    "config.decl @hrx2.tuning.copy.workgroup_size : %value: index where [range(%value, 32, 1024), mul(%value, 32)]\n"
)
NON_CONTIGUOUS_KERNEL_PREAMBLE = (
    "config.decl @hrx2.shape.copy4d.ne0 : %value: index where [range(%value, 1, 1073741824)]\n"
    "config.decl @hrx2.shape.copy4d.ne1 : %value: index where [range(%value, 1, 1073741824)]\n"
    "config.decl @hrx2.shape.copy4d.ne2 : %value: index where [range(%value, 1, 1073741824)]\n"
    "config.decl @hrx2.shape.copy4d.ne3 : %value: index where [range(%value, 1, 1073741824)]\n"
    "config.decl @hrx2.stride.copy4d.src0_nb0 : %value: index where [range(%value, 0, 1073741824)]\n"
    "config.decl @hrx2.stride.copy4d.src0_nb1 : %value: index where [range(%value, 0, 1073741824)]\n"
    "config.decl @hrx2.stride.copy4d.src0_nb2 : %value: index where [range(%value, 0, 1073741824)]\n"
    "config.decl @hrx2.stride.copy4d.src0_nb3 : %value: index where [range(%value, 0, 1073741824)]\n"
    "config.decl @hrx2.tuning.copy4d.workgroup_size : %value: index where [range(%value, 32, 1024), mul(%value, 32)]\n"
)
LOAD_WRITE_TEMPLATE = Template(
    "    %value = view.load %src0_view[$src_index] : view<1073741824x$src_type, #dense> -> $src_type\n"
    "    view.store %value, %dst_view[$dst_index] : $dst_type, view<1073741824x$dst_type, #dense>"
)
LOAD_VALUE_TEMPLATE = Template(
    "    %value_$src_type = view.load %src0_view[$src_index] : view<1073741824x$src_type, #dense> -> $src_type"
)
LOAD_F32_VALUE_TEMPLATE = Template(
    "    %value_f32 = view.load %src0_view[$src_index] : view<1073741824x$src_type, #dense> -> $src_type"
)
EXTEND_TO_F32_TEMPLATE = Template(
    "    %value_f32 = scalar.extf %value_$src_type : $src_type to f32"
)
TRUNCATE_FROM_F32_TEMPLATE = Template(
    "    %value = scalar.fptrunc %value_f32 : f32 to $dst_type"
)
STORE_VALUE_TEMPLATE = Template(
    "    view.store $value_name, %dst_view[$dst_index] : $dst_type, view<1073741824x$dst_type, #dense>"
)


@dataclass(frozen=True)
class ScalarDType:
    name: str
    route_dtype: str
    loom_type: str
    domain: str
    precision_rank: int


@dataclass(frozen=True)
class CopyKernelVariant:
    src: ScalarDType
    dst: ScalarDType

    @property
    def family(self) -> str:
        return f"copy_{self.src.name}_{self.dst.name}"

    @property
    def route_id(self) -> str:
        return f"{self.family}_contiguous_1d"

    @property
    def non_contiguous_route_id(self) -> str:
        return f"{self.family}_non_contiguous_4d"

    @property
    def root_symbol(self) -> str:
        return f"@{self.route_id}"

    @property
    def non_contiguous_root_symbol(self) -> str:
        return f"@{self.non_contiguous_route_id}"

    @property
    def export_name(self) -> str:
        return self.route_id

    @property
    def non_contiguous_export_name(self) -> str:
        return self.non_contiguous_route_id

    @property
    def route_path(self) -> Path:
        return Path("catalog") / "v2" / "copy" / f"{self.route_id}.json"

    @property
    def non_contiguous_route_path(self) -> Path:
        return Path("catalog") / "v2" / "copy" / f"{self.non_contiguous_route_id}.json"

    def conversion_kind(self) -> str | None:
        if self.src.domain != self.dst.domain:
            return None
        if self.src.loom_type == self.dst.loom_type:
            return "identity"
        if self.src.domain == "float" and self.dst.domain == "float":
            return "cast_via_f32"
        return None


SCALAR_DTYPES: tuple[ScalarDType, ...] = (
    ScalarDType(name="bf16", route_dtype="BF16", loom_type="bf16", domain="float", precision_rank=16),
    ScalarDType(name="f16", route_dtype="F16", loom_type="f16", domain="float", precision_rank=16),
    ScalarDType(name="f32", route_dtype="F32", loom_type="f32", domain="float", precision_rank=32),
)


def supported_variants() -> tuple[CopyKernelVariant, ...]:
    variants: list[CopyKernelVariant] = []
    for src in SCALAR_DTYPES:
        for dst in SCALAR_DTYPES:
            variant = CopyKernelVariant(src=src, dst=dst)
            if variant.conversion_kind() is not None:
                variants.append(variant)
    variants.sort(key=lambda variant: (0 if variant.src.name == variant.dst.name else 1, variant.src.name, variant.dst.name))
    return tuple(variants)


def _copy_compute_block(
    variant: CopyKernelVariant,
    *,
    src_index: str,
    dst_index: str,
) -> str:
    src_type = variant.src.loom_type
    dst_type = variant.dst.loom_type
    conversion = variant.conversion_kind()
    if conversion is None:
        raise ValueError(f"unsupported contiguous copy conversion: {variant.src.name} -> {variant.dst.name}")
    if conversion == "identity":
        return LOAD_WRITE_TEMPLATE.substitute(
            src_type=src_type,
            dst_type=dst_type,
            src_index=src_index,
            dst_index=dst_index,
        )
    if conversion == "cast_via_f32":
        if src_type == "f32":
            compute_lines = [
                LOAD_F32_VALUE_TEMPLATE.substitute(src_type=src_type, src_index=src_index)
            ]
        else:
            compute_lines = [
                LOAD_VALUE_TEMPLATE.substitute(src_type=src_type, src_index=src_index)
            ]
            compute_lines.append(EXTEND_TO_F32_TEMPLATE.substitute(src_type=src_type))
        if dst_type == "f32":
            compute_lines.append(
                STORE_VALUE_TEMPLATE.substitute(
                    value_name="%value_f32",
                    dst_type=dst_type,
                    dst_index=dst_index,
                )
            )
        else:
            compute_lines.append(TRUNCATE_FROM_F32_TEMPLATE.substitute(dst_type=dst_type))
            compute_lines.append(
                STORE_VALUE_TEMPLATE.substitute(
                    value_name="%value",
                    dst_type=dst_type,
                    dst_index=dst_index,
                )
            )
        return "\n".join(compute_lines)
    raise AssertionError(f"unexpected conversion kind: {conversion}")


def _kernel_body(variant: CopyKernelVariant) -> str:
    src_type = variant.src.loom_type
    dst_type = variant.dst.loom_type
    compute = _copy_compute_block(variant, src_index="%linear", dst_index="%linear")
    return _load_kernel_template().substitute(
        export_name=variant.export_name,
        root_symbol=variant.root_symbol,
        src_type=src_type,
        dst_type=dst_type,
        compute_block=compute,
    ).rstrip()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _template_path() -> Path:
    return _repo_root() / "kernels" / "v2" / "copy" / "contiguous_1d.loom.tmpl"


def _non_contiguous_template_path() -> Path:
    return _repo_root() / "kernels" / "v2" / "copy" / "non_contiguous_4d.loom.tmpl"


def _contiguous_route_template_path() -> Path:
    return _repo_root() / "catalog" / "v2" / "copy" / "contiguous_1d.json.tmpl"


def _non_contiguous_route_template_path() -> Path:
    return _repo_root() / "catalog" / "v2" / "copy" / "non_contiguous_4d.json.tmpl"


def _load_kernel_template() -> Template:
    return Template(_template_path().read_text(encoding="utf-8"))


def _load_non_contiguous_kernel_template() -> Template:
    return Template(_non_contiguous_template_path().read_text(encoding="utf-8"))


def _load_contiguous_route_template() -> Template:
    return Template(_contiguous_route_template_path().read_text(encoding="utf-8"))


def _load_non_contiguous_route_template() -> Template:
    return Template(_non_contiguous_route_template_path().read_text(encoding="utf-8"))


def render_loom() -> str:
    bodies = "\n\n".join(_kernel_body(variant) for variant in supported_variants())
    return f"{KERNEL_PREAMBLE}\n{bodies}\n"


def render_non_contiguous_loom() -> str:
    bodies = []
    for variant in supported_variants():
        compute = _copy_compute_block(variant, src_index="%src0_idx", dst_index="%linear")
        bodies.append(
            _load_non_contiguous_kernel_template().substitute(
                export_name=variant.non_contiguous_export_name,
                root_symbol=variant.non_contiguous_root_symbol,
                src_type=variant.src.loom_type,
                dst_type=variant.dst.loom_type,
                compute_block=compute,
            ).rstrip()
        )
    return f"{NON_CONTIGUOUS_KERNEL_PREAMBLE}\n" + "\n\n".join(bodies) + "\n"


def render_route_json(variant: CopyKernelVariant) -> str:
    return _load_contiguous_route_template().substitute(
        route_id=variant.route_id,
        family=variant.family,
        root_symbol=variant.root_symbol,
        export_name=variant.export_name,
        src_dtype=variant.src.route_dtype,
        dst_dtype=variant.dst.route_dtype,
        workgroup_size=WORKGROUP_SIZE,
    )


def render_non_contiguous_route_json_for_variant(variant: CopyKernelVariant) -> str:
    return _load_non_contiguous_route_template().substitute(
        route_id=variant.non_contiguous_route_id,
        family=variant.family,
        root_symbol=variant.non_contiguous_root_symbol,
        export_name=variant.non_contiguous_export_name,
        src_dtype=variant.src.route_dtype,
        dst_dtype=variant.dst.route_dtype,
        workgroup_size=WORKGROUP_SIZE,
    )


def render_kernel_artifacts() -> dict[Path, str]:
    return {
        Path("copy") / "contiguous_1d.loom": render_loom(),
        Path("copy") / "non_contiguous_4d.loom": render_non_contiguous_loom(),
    }


def render_catalog_artifacts() -> dict[Path, str]:
    artifacts: dict[Path, str] = {}
    for variant in supported_variants():
        artifacts[variant.route_path.relative_to(Path("catalog") / "v2")] = render_route_json(variant)
        artifacts[variant.non_contiguous_route_path.relative_to(Path("catalog") / "v2")] = (
            render_non_contiguous_route_json_for_variant(variant)
        )
    return artifacts


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
