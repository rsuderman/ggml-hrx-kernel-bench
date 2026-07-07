from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from string import Template


WORKGROUP_SIZE = 256
KERNEL_PREAMBLE = (
    "config.decl @hrx2.shape.copy.n : %value: index where [range(%value, 1, 1073741824)]\n"
    "config.decl @hrx2.tuning.copy.workgroup_size : %value: index where [range(%value, 32, 1024), mul(%value, 32)]\n"
)
LOAD_WRITE_TEMPLATE = Template(
    "    %value = view.load %src0_view[%linear] : view<1073741824x$src_type, #dense> -> $src_type\n"
    "    view.store %value, %dst_view[%linear] : $dst_type, view<1073741824x$dst_type, #dense>"
)
LOAD_VALUE_TEMPLATE = Template(
    "    %value_$src_type = view.load %src0_view[%linear] : view<1073741824x$src_type, #dense> -> $src_type"
)
LOAD_F32_VALUE_TEMPLATE = Template(
    "    %value_f32 = view.load %src0_view[%linear] : view<1073741824x$src_type, #dense> -> $src_type"
)
EXTEND_TO_F32_TEMPLATE = Template(
    "    %value_f32 = scalar.extf %value_$src_type : $src_type to f32"
)
TRUNCATE_FROM_F32_TEMPLATE = Template(
    "    %value = scalar.fptrunc %value_f32 : f32 to $dst_type"
)
STORE_VALUE_TEMPLATE = Template(
    "    view.store $value_name, %dst_view[%linear] : $dst_type, view<1073741824x$dst_type, #dense>"
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
    def root_symbol(self) -> str:
        return f"@{self.route_id}"

    @property
    def export_name(self) -> str:
        return self.route_id

    @property
    def route_path(self) -> Path:
        return Path("catalog") / "v2" / "copy" / f"{self.route_id}.json"

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


def _kernel_body(variant: CopyKernelVariant) -> str:
    src_type = variant.src.loom_type
    dst_type = variant.dst.loom_type
    conversion = variant.conversion_kind()
    if conversion is None:
        raise ValueError(f"unsupported contiguous copy conversion: {variant.src.name} -> {variant.dst.name}")
    if conversion == "identity":
        compute = LOAD_WRITE_TEMPLATE.substitute(src_type=src_type, dst_type=dst_type)
    elif conversion == "cast_via_f32":
        if src_type == "f32":
            compute_lines = [LOAD_F32_VALUE_TEMPLATE.substitute(src_type=src_type)]
        else:
            compute_lines = [LOAD_VALUE_TEMPLATE.substitute(src_type=src_type)]
            compute_lines.append(EXTEND_TO_F32_TEMPLATE.substitute(src_type=src_type))
        if dst_type == "f32":
            compute_lines.append(STORE_VALUE_TEMPLATE.substitute(value_name="%value_f32", dst_type=dst_type))
        else:
            compute_lines.append(TRUNCATE_FROM_F32_TEMPLATE.substitute(dst_type=dst_type))
            compute_lines.append(STORE_VALUE_TEMPLATE.substitute(value_name="%value", dst_type=dst_type))
        compute = "\n".join(compute_lines)
    else:
        raise AssertionError(f"unexpected conversion kind: {conversion}")
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


def _load_kernel_template() -> Template:
    return Template(_template_path().read_text(encoding="utf-8"))


def render_loom() -> str:
    bodies = "\n\n".join(_kernel_body(variant) for variant in supported_variants())
    return f"{KERNEL_PREAMBLE}\n{bodies}\n"


def route_descriptor(variant: CopyKernelVariant) -> dict[str, object]:
    return {
        "id": variant.route_id,
        "family": variant.family,
        "kernel": {
            "source_id": variant.family,
            "path": "copy/contiguous_1d.loom",
            "root_symbol": variant.root_symbol,
            "export_name": variant.export_name,
        },
        "tensors": {
            "src0": {
                "dtype": variant.src.route_dtype,
                "dimensions": "src0_dimensions",
                "strides": "src0_strides",
            },
            "dst": {
                "dtype": variant.dst.route_dtype,
                "dimensions": "dst_dimensions",
                "strides": "dst_strides",
            },
        },
        "values": [
            {
                "name": "contiguous_strides",
                "contiguous_strides": "dst_dimensions",
            },
            {
                "name": "total_size",
                "product": "dst_dimensions",
            },
        ],
        "constraints": [
            {"equals": ["src0_dimensions", "dst_dimensions"]},
            {"equals": ["contiguous_strides", "src0_strides", "dst_strides"]},
        ],
        "launch": {
            "workgroup_size": [WORKGROUP_SIZE, 1, 1],
        },
        "config": {
            "bindings": [
                {
                    "key": "@hrx2.shape.copy.n",
                    "source": "value.total_size",
                },
                {
                    "key": "@hrx2.tuning.copy.workgroup_size",
                    "value": str(WORKGROUP_SIZE),
                },
            ]
        },
    }


def render_route_json(variant: CopyKernelVariant) -> str:
    return json.dumps(route_descriptor(variant), indent=2) + "\n"


def render_kernel_artifacts() -> dict[Path, str]:
    return {
        Path("copy") / "contiguous_1d.loom": render_loom(),
    }


def render_catalog_artifacts() -> dict[Path, str]:
    artifacts: dict[Path, str] = {}
    for variant in supported_variants():
        artifacts[variant.route_path.relative_to(Path("catalog") / "v2")] = render_route_json(variant)
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
