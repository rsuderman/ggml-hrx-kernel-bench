from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from string import Template


WORKGROUP_SIZE = 256
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


@dataclass(frozen=True)
class CopyRouteFlavor:
    name: str
    route_suffix: str
    kernel_template_name: str
    route_template_name: str
    src_index_name: str

    def route_id_for(self, variant: CopyKernelVariant) -> str:
        return f"{variant.family}_{self.route_suffix}"

    def root_symbol_for(self, variant: CopyKernelVariant) -> str:
        return f"@{self.route_id_for(variant)}"

    def export_name_for(self, variant: CopyKernelVariant) -> str:
        return self.route_id_for(variant)

    def kernel_relative_path_for(self, variant: CopyKernelVariant) -> Path:
        return Path("copy") / f"{self.route_id_for(variant)}.loom"

    def catalog_relative_path_for(self, variant: CopyKernelVariant) -> Path:
        return Path("copy") / f"{self.route_id_for(variant)}.json"

# Route order defines router preference. Keep the preferred flavors first.
COPY_ROUTE_FLAVORS: tuple[CopyRouteFlavor, ...] = (
    CopyRouteFlavor(
        name="contiguous",
        route_suffix="contiguous_1d",
        kernel_template_name="contiguous_1d.loom.tmpl",
        route_template_name="contiguous_1d.json.tmpl",
        src_index_name="%linear",
    ),
    CopyRouteFlavor(
        name="non_contiguous",
        route_suffix="non_contiguous_4d",
        kernel_template_name="non_contiguous_4d.loom.tmpl",
        route_template_name="non_contiguous_4d.json.tmpl",
        src_index_name="%src0_idx",
    ),
)


def supported_variants() -> tuple[CopyKernelVariant, ...]:
    variants: list[CopyKernelVariant] = []
    for src in SCALAR_DTYPES:
        for dst in SCALAR_DTYPES:
            variant = CopyKernelVariant(src=src, dst=dst)
            if variant.conversion_kind() is not None:
                variants.append(variant)
    variants.sort(
        key=lambda variant: (
            0 if variant.src.name == variant.dst.name else 1,
            variant.src.name,
            variant.dst.name,
        )
    )
    return tuple(variants)


def render_compute_block(
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


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_kernel_template(name: str) -> Template:
    return Template((repo_root() / "kernels" / "v2" / "copy" / name).read_text(encoding="utf-8"))


def load_route_template(name: str) -> Template:
    return Template((repo_root() / "catalog" / "v2" / "copy" / name).read_text(encoding="utf-8"))
