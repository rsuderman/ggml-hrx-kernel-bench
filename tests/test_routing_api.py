from __future__ import annotations

import json
from pathlib import Path

import pytest

from ggml_hrx_kernel_bench.generators.copy import (
    render_catalog_artifacts,
    render_kernel_artifacts,
)
from ggml_hrx_kernel_bench.routing.api import (
    CandidateQuery,
    RuntimeCaseRequest,
    create_router,
)
from ggml_hrx_kernel_bench.yaml_route_import import materialize_yaml_route_import
from ggml_hrx_kernel_bench.routing.v2.candidates import candidate_from_shape
from ggml_hrx_kernel_bench.routing.v2.query import load_route_catalog, routes_for_op


# Unary + copy families are code-generated into the materialized asset tree (not the source catalog),
# so resolve routes against a once-materialized catalog rather than catalog/v2 on disk.
import tempfile as _tempfile

from ggml_hrx_kernel_bench.materialized_assets import materialize_asset_root as _materialize_asset_root

_MATERIALIZED_V2_ASSETS = _materialize_asset_root(
    Path(_tempfile.mkdtemp(prefix="hrx-v2-routing-assets-")) / "assets", force=True
)
ACTUAL_V2_ROUTING_DIR = _MATERIALIZED_V2_ASSETS / "catalog" / "v2"
ACTUAL_V2_KERNEL_DIR = _MATERIALIZED_V2_ASSETS / "kernels" / "v2"


def _softmax_kqv_shape(k: int, *, rows: int = 128, cols: int = 24, nheads_kv: int = 8) -> dict[str, int]:
    return {
        "d0": cols,
        "d1": rows,
        "src0_d0": k,
        "src0_d1": cols,
        "mask_d0": k,
        "mask_d1": 1,
        "src1_d0": k,
        "src1_d1": rows * nheads_kv,
        "dst_d0": rows,
        "dst_d1": cols,
        "k": k,
        "rows": rows,
        "cols": cols,
        "nheads_kv": nheads_kv,
    }


def _write_v2_descriptor(routing_dir: Path) -> None:
    routing_dir.mkdir(parents=True, exist_ok=True)
    (routing_dir / "router.json").write_text(
        json.dumps(
            {
                "schema": "ggml_hrx_kernel_bench.routing_descriptors.v2",
                "routes": {"ADD": ["add/add_f32_contiguous_1d.json", "add/add_f32_generic_4d.json"]},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    add_dir = routing_dir / "add"
    add_dir.mkdir(parents=True, exist_ok=True)
    (add_dir / "add_f32_contiguous_1d.json").write_text(
        json.dumps(
            {
                "id": "add_f32_contiguous_1d",
                "family": "add_f32",
                "kernel": {
                    "source_id": "add_f32",
                    "path": "add/contiguous_1d.loom",
                    "root_symbol": "@hrx2_add_f32_contiguous_1d",
                    "export_name": "hrx2_add_f32_contiguous_1d",
                },
                "tensors": {
                    "src0": {
                        "dtype": "F32",
                        "dimensions": "src0_dimensions",
                        "strides": "src0_strides",
                    },
                    "src1": {
                        "dtype": "F32",
                        "dimensions": "src1_dimensions",
                        "strides": "src1_strides",
                    },
                    "dst": {
                        "dtype": "F32",
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
                    {"equals": ["src0_dimensions", "src1_dimensions", "dst_dimensions"]},
                    {"equals": ["contiguous_strides", "src0_strides", "src1_strides", "dst_strides"]},
                ],
                "launch": {
                    "workgroup_size": [256, 1, 1],
                },
                "config": {
                    "bindings": [
                        {
                            "key": "@hrx2.shape.pointwise.total_size",
                            "source": "value.total_size",
                        },
                        {
                            "key": "@hrx2.tuning.pointwise.workgroup_size",
                            "value": "256",
                        },
                    ]
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (add_dir / "add_f32_generic_4d.json").write_text(
        json.dumps(
            {
                "id": "add_f32_generic_4d",
                "family": "add_f32",
                "kernel": {
                    "source_id": "add_f32",
                    "path": "add/generic_4d.loom",
                    "root_symbol": "@hrx2_add_f32_generic_4d",
                    "export_name": "hrx2_add_f32_generic_4d",
                },
                "tensors": {
                    "src0": {
                        "dtype": "F32",
                        "dimensions": "src0_dimensions",
                        "strides": "src0_strides",
                    },
                    "src1": {
                        "dtype": "F32",
                        "dimensions": "src1_dimensions",
                        "strides": "src1_strides",
                    },
                    "dst": {
                        "dtype": "F32",
                        "dimensions": "dst_dimensions",
                        "strides": "dst_strides",
                    },
                },
                "values": [
                    {
                        "name": "total_size",
                        "product": "dst_dimensions",
                    }
                ],
                "constraints": [
                    {"name": "dst_dimensions", "length": 4},
                    {"divides": ["src0_dimensions", "dst_dimensions"]},
                    {"divides": ["src1_dimensions", "dst_dimensions"]},
                ],
                "launch": {
                    "workgroup_size": [256, 1, 1],
                },
                "config": {
                    "bindings": [
                        {"key": "@hrx2.shape.add4d.ne0", "source": "tensor.dst.dimensions.d0.size"},
                        {"key": "@hrx2.shape.add4d.ne1", "source": "tensor.dst.dimensions.d1.size"},
                        {"key": "@hrx2.shape.add4d.ne2", "source": "tensor.dst.dimensions.d2.size"},
                        {"key": "@hrx2.shape.add4d.ne3", "source": "tensor.dst.dimensions.d3.size"},
                        {"key": "@hrx2.shape.add4d.src0_ne0", "source": "tensor.src0.dimensions.d0.size"},
                        {"key": "@hrx2.shape.add4d.src0_ne1", "source": "tensor.src0.dimensions.d1.size"},
                        {"key": "@hrx2.shape.add4d.src0_ne2", "source": "tensor.src0.dimensions.d2.size"},
                        {"key": "@hrx2.shape.add4d.src0_ne3", "source": "tensor.src0.dimensions.d3.size"},
                        {"key": "@hrx2.shape.add4d.src1_ne0", "source": "tensor.src1.dimensions.d0.size"},
                        {"key": "@hrx2.shape.add4d.src1_ne1", "source": "tensor.src1.dimensions.d1.size"},
                        {"key": "@hrx2.shape.add4d.src1_ne2", "source": "tensor.src1.dimensions.d2.size"},
                        {"key": "@hrx2.shape.add4d.src1_ne3", "source": "tensor.src1.dimensions.d3.size"},
                        {"key": "@hrx2.stride.add4d.src0_nb0", "source": "tensor.src0.dimensions.d0.stride"},
                        {"key": "@hrx2.stride.add4d.src0_nb1", "source": "tensor.src0.dimensions.d1.stride"},
                        {"key": "@hrx2.stride.add4d.src0_nb2", "source": "tensor.src0.dimensions.d2.stride"},
                        {"key": "@hrx2.stride.add4d.src0_nb3", "source": "tensor.src0.dimensions.d3.stride"},
                        {"key": "@hrx2.stride.add4d.src1_nb0", "source": "tensor.src1.dimensions.d0.stride"},
                        {"key": "@hrx2.stride.add4d.src1_nb1", "source": "tensor.src1.dimensions.d1.stride"},
                        {"key": "@hrx2.stride.add4d.src1_nb2", "source": "tensor.src1.dimensions.d2.stride"},
                        {"key": "@hrx2.stride.add4d.src1_nb3", "source": "tensor.src1.dimensions.d3.stride"},
                        {"key": "@hrx2.stride.add4d.dst_nb0", "source": "tensor.dst.dimensions.d0.stride"},
                        {"key": "@hrx2.stride.add4d.dst_nb1", "source": "tensor.dst.dimensions.d1.stride"},
                        {"key": "@hrx2.stride.add4d.dst_nb2", "source": "tensor.dst.dimensions.d2.stride"},
                        {"key": "@hrx2.stride.add4d.dst_nb3", "source": "tensor.dst.dimensions.d3.stride"},
                        {"key": "@hrx2.tuning.add4d.workgroup_size", "value": "256"},
                    ]
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_kernel(kernel_dir: Path) -> None:
    add_dir = kernel_dir / "add"
    add_dir.mkdir(parents=True, exist_ok=True)
    (add_dir / "contiguous_1d.loom").write_text(
        'kernel.def export("hrx2_add_f32_contiguous_1d") @hrx2_add_f32_contiguous_1d\n',
        encoding="utf-8",
    )
    (add_dir / "generic_4d.loom").write_text(
        'kernel.def export("hrx2_add_f32_generic_4d") @hrx2_add_f32_generic_4d\n',
        encoding="utf-8",
    )


def _write_v2_copy_descriptor(routing_dir: Path) -> None:
    routing_dir.mkdir(parents=True, exist_ok=True)
    route_paths = (
        "copy/copy_f32_f32_non_contiguous_4d.json",
        "copy/copy_f32_f16_contiguous_1d.json",
    )
    (routing_dir / "router.json").write_text(
        json.dumps(
            {
                "schema": "ggml_hrx_kernel_bench.routing_descriptors.v2",
                "routes": {"CPY": list(route_paths)},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    artifacts = render_catalog_artifacts()
    for route_path in route_paths:
        relative_path = Path(route_path)
        descriptor_path = routing_dir / relative_path
        descriptor_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor_path.write_text(artifacts[relative_path], encoding="utf-8")


def _write_copy_kernel(kernel_dir: Path) -> None:
    artifacts = render_kernel_artifacts()
    for relative_path in (
        Path("copy") / "copy_f32_f16_contiguous_1d.loom",
        Path("copy") / "copy_f32_f32_non_contiguous_4d.loom",
    ):
        kernel_path = kernel_dir / relative_path
        kernel_path.parent.mkdir(parents=True, exist_ok=True)
        kernel_path.write_text(artifacts[relative_path], encoding="utf-8")


def test_v2_router_returns_no_candidates_without_descriptor(tmp_path: Path) -> None:
    router = create_router(
        version="v2",
        kernel_dir=tmp_path / "kernels",
        routing_dir=tmp_path / "routing",
    )

    assert router.candidates(CandidateQuery()) == []


def test_v2_router_returns_contiguous_add_candidate(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernels"
    routing_dir = tmp_path / "routing"
    _write_kernel(kernel_dir)
    _write_v2_descriptor(routing_dir)
    router = create_router(version="v2", kernel_dir=kernel_dir, routing_dir=routing_dir)

    candidates = router.candidates(CandidateQuery())

    assert len(candidates) == 2
    contiguous = candidates[0]
    generic = candidates[1]
    assert contiguous.family == "add_f32"
    assert contiguous.route_id == "add_f32_contiguous_1d"
    assert contiguous.root_symbol == "@hrx2_add_f32_contiguous_1d"
    assert contiguous.shape == {"d0": 256, "d1": 1}
    assert generic.route_id == "add_f32_generic_4d"
    assert generic.root_symbol == "@hrx2_add_f32_generic_4d"
    assert generic.shape == {"d0": 1, "d1": 1, "d2": 1, "d3": 1}


def test_v2_manifest_includes_original_root_metadata(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernels"
    routing_dir = tmp_path / "routing"
    original_root = tmp_path / "original"
    _write_kernel(kernel_dir)
    _write_v2_descriptor(routing_dir)
    (original_root / "kernels" / "add").mkdir(parents=True, exist_ok=True)
    (original_root / "kernels" / "add" / "contiguous_1d.loom").write_text(
        'kernel.def export("hrx2_add_f32_contiguous_1d") @hrx2_add_f32_contiguous_1d\n',
        encoding="utf-8",
    )
    (original_root / "kernels" / "add" / "generic_4d.loom").write_text(
        'kernel.def export("hrx2_add_f32_generic_4d") @hrx2_add_f32_generic_4d\n',
        encoding="utf-8",
    )

    router = create_router(version="v2", kernel_dir=kernel_dir, routing_dir=routing_dir)

    manifest = router.manifest(original_root=original_root)

    assert manifest["route_count"] == 2
    assert len(manifest["entries"]) == 2
    by_path = {Path(entry["path"]).relative_to(kernel_dir).as_posix(): entry for entry in manifest["entries"]}
    contiguous = by_path["add/contiguous_1d.loom"]
    generic = by_path["add/generic_4d.loom"]
    assert contiguous["original_path"] == str(original_root / "kernels" / "add" / "contiguous_1d.loom")
    assert generic["original_path"] == str(original_root / "kernels" / "add" / "generic_4d.loom")
    assert contiguous["original_sha256"] is not None
    assert generic["original_sha256"] is not None
    assert contiguous["imported_sha256"] is not None
    assert generic["imported_sha256"] is not None
    assert contiguous["mechanical_rewrites"] == []
    assert generic["mechanical_rewrites"] == []

def test_v2_copy_catalog_keeps_generated_descriptors_declarative(tmp_path: Path) -> None:
    routing_dir = tmp_path / "routing"
    _write_v2_copy_descriptor(routing_dir)

    catalog = load_route_catalog(routing_dir)
    by_id = {route.id: route for route in catalog.routes}
    descriptors = [
        json.loads((routing_dir / "copy" / descriptor_name).read_text(encoding="utf-8"))
        for descriptor_name in (
            "copy_f32_f16_contiguous_1d.json",
            "copy_f32_f32_non_contiguous_4d.json",
        )
    ]

    assert by_id["copy_f32_f16_contiguous_1d"]
    assert by_id["copy_f32_f32_non_contiguous_4d"]
    assert all("import" not in descriptor for descriptor in descriptors)
    assert all("lowering" not in descriptor for descriptor in descriptors)

def test_yaml_route_import_accepts_numeric_cont_offsets(tmp_path: Path) -> None:
    yaml_path = tmp_path / "cont.v2.yaml"
    yaml_path.write_text(
        json.dumps(
            {
                "ops": {
                    "CONT": [
                        {
                            "inputs": [{"dtype": "F32", "shape": [2, 1, 3, 5], "offset": 0}],
                            "destinations": [{"dtype": "F32", "shape": [2, 1, 3, 5]}],
                        },
                        {
                            "inputs": [
                                {
                                    "dtype": "F32",
                                    "shape": [1, 1, 4, 1],
                                    "storage_shape": [1, 4, 4, 1],
                                    "offset": 12,
                                }
                            ],
                            "destinations": [{"dtype": "F32", "shape": [1, 4, 4, 1]}],
                        },
                    ]
                }
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "route-import"
    summary = materialize_yaml_route_import(
        [yaml_path],
        output_dir=output_dir,
        routing_dir=ACTUAL_V2_ROUTING_DIR,
    )

    op_summary = next(row for row in summary["operations"] if row["op"] == "CONT")
    assert op_summary["invalid_case_count"] == 0
    assert op_summary["matched_case_count"] == 1
    assert op_summary["unmatched_case_count"] == 1
    route_matches = json.loads((output_dir / "ops" / "CONT" / "route-matches.json").read_text())
    assert route_matches["rows"][0]["case_index"] == 0
    assert route_matches["rows"][0]["matched_route_ids"] == ["cont_f32_contiguous_4d"]
    route_unmatched = json.loads((output_dir / "ops" / "CONT" / "route-unmatched.json").read_text())
    assert route_unmatched["rows"][0]["case_index"] == 1


def test_yaml_route_import_emits_execution_abi_for_add_f32_case(tmp_path: Path) -> None:
    yaml_path = tmp_path / "add.v2.yaml"
    yaml_path.write_text(
        json.dumps(
            {
                "ops": {
                    "ADD": [
                        {
                            "inputs": [
                                {"dtype": "F32", "shape": [4, 1, 1, 1]},
                                {"dtype": "F32", "shape": [4, 1, 1, 1]},
                            ],
                            "destinations": [{"dtype": "F32", "shape": [4, 1, 1, 1]}],
                        }
                    ]
                }
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "route-import"
    summary = materialize_yaml_route_import(
        [yaml_path],
        output_dir=output_dir,
        routing_dir=ACTUAL_V2_ROUTING_DIR,
    )

    op_summary = next(row for row in summary["operations"] if row["op"] == "ADD")
    assert op_summary["matched_case_count"] == 1
    assert op_summary["unmatched_case_count"] == 0
    config = json.loads(Path(summary["generated_config_paths"][0]).read_text())
    assert config["kernel"] == "add_f32"
    assert config["route_id"] == "add_f32_generic_4d"
    assert config["execution_abi"] == {
        "schema": "ggml_hrx_kernel_bench.route_execution_abi.v1",
        "route_id": "add_f32_generic_4d",
        "entries": [
            {
                "position": 0,
                "role": "src0",
                "kind": "input",
                "dtype": "f32",
                "fixture": "src0",
            },
            {
                "position": 1,
                "role": "src1",
                "kind": "input",
                "dtype": "f32",
                "fixture": "src1",
            },
            {
                "position": 2,
                "role": "dst",
                "kind": "output",
                "dtype": "f32",
                "fixture": "dst_init",
                "expect": {
                    "fixture": "expected",
                    "mode": "close",
                },
            },
        ],
    }


def test_yaml_route_import_emits_scalar_execution_abi_for_scale_f32_case(tmp_path: Path) -> None:
    yaml_path = tmp_path / "scale.v2.yaml"
    yaml_path.write_text(
        json.dumps(
            {
                "ops": {
                    "SCALE": [
                        {
                            "inputs": [{"dtype": "F32", "shape": [4, 1, 1, 1]}],
                            "destinations": [{"dtype": "F32", "shape": [4, 1, 1, 1]}],
                        }
                    ]
                }
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "route-import"
    summary = materialize_yaml_route_import(
        [yaml_path],
        output_dir=output_dir,
        routing_dir=ACTUAL_V2_ROUTING_DIR,
    )

    op_summary = next(row for row in summary["operations"] if row["op"] == "SCALE")
    assert op_summary["matched_case_count"] == 1
    assert op_summary["unmatched_case_count"] == 0
    config = json.loads(Path(summary["generated_config_paths"][0]).read_text())
    assert config["kernel"] == "scale_f32"
    assert config["route_id"] == "scale_f32_contiguous_4d"
    assert config["execution_abi"] == {
        "schema": "ggml_hrx_kernel_bench.route_execution_abi.v1",
        "route_id": "scale_f32_contiguous_4d",
        "entries": [
            {
                "position": 0,
                "role": "scale",
                "kind": "scalar",
                "dtype": "f32",
                "value": 0.625,
            },
            {
                "position": 1,
                "role": "bias",
                "kind": "scalar",
                "dtype": "f32",
                "value": -0.125,
            },
            {
                "position": 2,
                "role": "src0",
                "kind": "input",
                "dtype": "f32",
                "fixture": "src0",
            },
            {
                "position": 3,
                "role": "dst",
                "kind": "output",
                "dtype": "f32",
                "fixture": "dst_init",
                "expect": {
                    "fixture": "expected",
                    "mode": "close",
                },
            },
        ],
    }


def test_yaml_route_import_splits_rms_norm_execution_abi_by_eps(tmp_path: Path) -> None:
    yaml_path = tmp_path / "rms-norm.v2.yaml"
    yaml_path.write_text(
        json.dumps(
            {
                "ops": {
                    "RMS_NORM": [
                        {
                            "inputs": [{"dtype": "F32", "shape": [4, 2, 1, 1]}],
                            "destinations": [{"dtype": "F32", "shape": [4, 2, 1, 1]}],
                            "attributes": {"eps": 0.0},
                        },
                        {
                            "inputs": [{"dtype": "F32", "shape": [4, 2, 1, 1]}],
                            "destinations": [{"dtype": "F32", "shape": [4, 2, 1, 1]}],
                            "attributes": {"eps": 0.0001},
                        },
                    ]
                }
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "route-import"
    summary = materialize_yaml_route_import(
        [yaml_path],
        output_dir=output_dir,
        routing_dir=ACTUAL_V2_ROUTING_DIR,
    )

    op_summary = next(row for row in summary["operations"] if row["op"] == "RMS_NORM")
    assert op_summary["matched_case_count"] == 2
    assert op_summary["unmatched_case_count"] == 0
    configs = [json.loads(Path(path).read_text()) for path in summary["generated_config_paths"]]
    assert [config["kernel"] for config in configs] == ["rms_norm_f32", "rms_norm_f32"]
    assert sorted(
        entry["value"]
        for config in configs
        for entry in config["execution_abi"]["entries"]
        if entry["kind"] == "scalar"
    ) == [0.0, 0.0001]
    for config in configs:
        assert config["route_id"] == "rms_norm_f32_contiguous_4d"
        assert config["cases"] == [[4, 2, 1, 1]]
        assert config["execution_abi"]["entries"][1:] == [
            {
                "position": 1,
                "role": "src0",
                "kind": "input",
                "dtype": "f32",
                "fixture": "src",
            },
            {
                "position": 2,
                "role": "dst",
                "kind": "output",
                "dtype": "f32",
                "fixture": "dst_init",
                "expect": {
                    "fixture": "expected",
                    "mode": "close",
                },
            },
        ]


def test_v2_rms_norm_dispatch_uses_flattened_trailing_rows() -> None:
    catalog = load_route_catalog(ACTUAL_V2_ROUTING_DIR)
    route = next(current for current in routes_for_op(catalog, "RMS_NORM"))

    candidate = candidate_from_shape(
        kernel_dir=ACTUAL_V2_KERNEL_DIR,
        route=route,
        shape={"d0": 1025, "d1": 5, "d2": 4, "d3": 3},
    )

    assert candidate.config["@hrx2.shape.rms_norm.nrows"] == "60"
    assert candidate.dispatch["workgroup_count"] == [60, 1, 1]


@pytest.mark.parametrize("route_id", ["soft_max_f32_contiguous_4d", "soft_max_f32_mask_contiguous_4d"])
def test_v2_soft_max_dispatch_uses_flattened_trailing_rows(route_id: str) -> None:
    catalog = load_route_catalog(ACTUAL_V2_ROUTING_DIR)
    route = next(current for current in routes_for_op(catalog, "SOFT_MAX") if current.id == route_id)

    candidate = candidate_from_shape(
        kernel_dir=ACTUAL_V2_KERNEL_DIR,
        route=route,
        shape={"d0": 16, "d1": 2, "d2": 32, "d3": 1},
    )

    assert candidate.config["@hrx2.shape.soft_max.nrows"] == "64"
    assert candidate.dispatch["workgroup_count"] == [64, 1, 1]


def test_v2_default_cont_candidate_derives_rank_polymorphic_shape_bindings() -> None:
    router = create_router(version="v2", kernel_dir=ACTUAL_V2_KERNEL_DIR, routing_dir=ACTUAL_V2_ROUTING_DIR)

    candidate = next(
        current for current in router.candidates(CandidateQuery()) if current.route_id == "cont_f32_contiguous_4d"
    )

    assert candidate.shape == {"d0": 1, "d1": 1}
    assert candidate.config == {
        "@hrx2.shape.cont.d0": "1",
        "@hrx2.shape.cont.d1": "1",
        "@hrx2.shape.cont.ne1": "1",
        "@hrx2.shape.cont.ne2": "1",
        "@hrx2.stride.cont.src_nb1": "1",
        "@hrx2.stride.cont.src_nb2": "1",
        "@hrx2.stride.cont.src_nb3": "1",
        "@hrx2.tuning.cont.workgroup_size": "256",
    }


def test_yaml_route_import_matches_descriptor_set_rows_f32_case(tmp_path: Path) -> None:
    yaml_path = tmp_path / "set_rows.v2.yaml"
    yaml_path.write_text(
        json.dumps(
            {
                "ops": {
                    "SET_ROWS": [
                        {
                            "inputs": [
                                {"dtype": "F32", "shape": [3, 3, 14, 3]},
                                {"dtype": "F32", "shape": [3, 2, 14, 3]},
                                {"dtype": "I64", "shape": [2, 7, 1, 1]},
                            ],
                            "destinations": [{"dtype": "F32", "shape": [3, 3, 14, 3]}],
                        }
                    ]
                }
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "route-import"
    summary = materialize_yaml_route_import(
        [yaml_path],
        output_dir=output_dir,
        routing_dir=ACTUAL_V2_ROUTING_DIR,
    )

    op_summary = next(row for row in summary["operations"] if row["op"] == "SET_ROWS")
    assert op_summary["matched_case_count"] == 1
    assert op_summary["unmatched_case_count"] == 0
    route_matches = json.loads((output_dir / "ops" / "SET_ROWS" / "route-matches.json").read_text())
    assert route_matches["rows"][0]["matched_route_ids"] == ["set_rows_f32_f32_descriptor_4d"]
    config = json.loads(Path(summary["generated_config_paths"][0]).read_text())
    assert config["kernel"] == "set_rows_f32"
    assert config["route_id"] == "set_rows_f32_f32_descriptor_4d"
    assert config["execution_abi"] == {
        "schema": "ggml_hrx_kernel_bench.route_execution_abi.v1",
        "route_id": "set_rows_f32_f32_descriptor_4d",
        "entries": [
            {
                "position": 0,
                "role": "src1",
                "kind": "input",
                "dtype": "f32",
                "fixture": "src0",
            },
            {
                "position": 1,
                "role": "src2",
                "kind": "input",
                "dtype": "i32",
                "fixture": "indices",
            },
            {
                "position": 2,
                "role": "dst",
                "kind": "output",
                "dtype": "f32",
                "fixture": "dst_init",
                "expect": {
                    "fixture": "expected",
                    "mode": "close",
                },
            },
        ],
    }
    shape = dict(zip(config["params"], config["cases"][0], strict=True))
    assert shape == {
        "d0": 3,
        "d1": 3,
        "d2": 14,
        "d3": 3,
        "src1_d1": 2,
        "src2_d0": 2,
        "src2_d1": 7,
        "src2_d2": 1,
        "src2_d3": 1,
    }


def test_yaml_route_import_matches_model_cont_set_rows_f32_f16_case(tmp_path: Path) -> None:
    yaml_path = tmp_path / "set_rows_model.v2.yaml"
    yaml_path.write_text(
        json.dumps(
            {
                "ops": {
                    "SET_ROWS": [
                        {
                            "inputs": [
                                {"dtype": "F32", "shape": [1024, 512, 1, 1]},
                                {"dtype": "I64", "shape": [512, 1, 1, 1]},
                            ],
                            "destinations": [{"dtype": "F16", "shape": [1024, 8192, 1, 1]}],
                        }
                    ]
                }
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "route-import"
    summary = materialize_yaml_route_import(
        [yaml_path],
        output_dir=output_dir,
        routing_dir=ACTUAL_V2_ROUTING_DIR,
    )

    op_summary = next(row for row in summary["operations"] if row["op"] == "SET_ROWS")
    assert op_summary["matched_case_count"] == 1
    assert op_summary["unmatched_case_count"] == 0
    route_matches = json.loads((output_dir / "ops" / "SET_ROWS" / "route-matches.json").read_text())
    assert route_matches["rows"][0]["matched_route_ids"] == [
        "cont_set_rows_f32_f16_n1024_dst8192_contiguous_4d"
    ]
    config = json.loads(Path(summary["generated_config_paths"][0]).read_text())
    assert config["kernel"] == "cont_set_rows_f32"
    assert config["route_id"] == "cont_set_rows_f32_f16_n1024_dst8192_contiguous_4d"
    assert config["execution_abi"]["entries"] == [
        {
            "position": 0,
            "role": "src0",
            "kind": "input",
            "dtype": "f32",
            "fixture": "src0",
        },
        {
            "position": 1,
            "role": "src1",
            "kind": "input",
            "dtype": "i32",
            "fixture": "indices",
        },
        {
            "position": 2,
            "role": "dst",
            "kind": "output",
            "dtype": "f16",
            "fixture": "dst_init",
            "expect": {
                "fixture": "expected",
                "mode": "close",
            },
        },
    ]
    shape = dict(zip(config["params"], config["cases"][0], strict=True))
    assert shape == {
        "d0": 1024,
        "d1": 8192,
        "d2": 1,
        "d3": 1,
        "src0_d1": 512,
        "src1_d0": 512,
        "src1_d1": 1,
    }


def test_yaml_route_import_matches_descriptor_get_rows_f32_case(tmp_path: Path) -> None:
    yaml_path = tmp_path / "get_rows.v2.yaml"
    yaml_path.write_text(
        json.dumps(
            {
                "ops": {
                    "GET_ROWS": [
                        {
                            "inputs": [
                                {"dtype": "F32", "shape": [4096, 512, 1, 1]},
                                {"dtype": "I32", "shape": [512, 1, 1, 1]},
                            ],
                            "destinations": [{"dtype": "F32", "shape": [4096, 512, 1, 1]}],
                        }
                    ]
                }
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "route-import"
    summary = materialize_yaml_route_import(
        [yaml_path],
        output_dir=output_dir,
        routing_dir=ACTUAL_V2_ROUTING_DIR,
    )

    op_summary = next(row for row in summary["operations"] if row["op"] == "GET_ROWS")
    assert op_summary["matched_case_count"] == 1
    assert op_summary["unmatched_case_count"] == 0
    route_matches = json.loads((output_dir / "ops" / "GET_ROWS" / "route-matches.json").read_text())
    assert route_matches["rows"][0]["matched_route_ids"] == ["get_rows_f32_embedding_rows_descriptor_4d"]
    config = json.loads(Path(summary["generated_config_paths"][0]).read_text())
    assert config["kernel"] == "get_rows_f32"
    assert config["route_id"] == "get_rows_f32_embedding_rows_descriptor_4d"
    assert config["execution_abi"] == {
        "schema": "ggml_hrx_kernel_bench.route_execution_abi.v1",
        "route_id": "get_rows_f32_embedding_rows_descriptor_4d",
        "entries": [
            {
                "position": 0,
                "role": "src0",
                "kind": "input",
                "dtype": "f32",
                "fixture": "src0",
            },
            {
                "position": 1,
                "role": "src1",
                "kind": "input",
                "dtype": "i32",
                "fixture": "indices",
            },
            {
                "position": 2,
                "role": "dst",
                "kind": "output",
                "dtype": "f32",
                "fixture": "dst_init",
                "expect": {
                    "fixture": "expected",
                    "mode": "close",
                },
            },
        ],
    }
    shape = dict(zip(config["params"], config["cases"][0], strict=True))
    assert shape == {
        "d0": 4096,
        "d1": 512,
        "d2": 1,
        "d3": 1,
        "src1_d0": 512,
        "src1_d1": 1,
    }


def test_yaml_route_import_matches_descriptor_get_rows_q8_0_case(tmp_path: Path) -> None:
    yaml_path = tmp_path / "get_rows_q8_0.v2.yaml"
    yaml_path.write_text(
        json.dumps(
            {
                "ops": {
                    "GET_ROWS": [
                        {
                            "inputs": [
                                {"dtype": "Q8_0", "shape": [4096, 128256, 1, 1]},
                                {"dtype": "I32", "shape": [512, 1, 1, 1]},
                            ],
                            "destinations": [{"dtype": "F32", "shape": [4096, 512, 1, 1]}],
                        }
                    ]
                }
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "route-import"
    summary = materialize_yaml_route_import(
        [yaml_path],
        output_dir=output_dir,
        routing_dir=ACTUAL_V2_ROUTING_DIR,
    )

    op_summary = next(row for row in summary["operations"] if row["op"] == "GET_ROWS")
    assert op_summary["matched_case_count"] == 1
    assert op_summary["unmatched_case_count"] == 0
    route_matches = json.loads((output_dir / "ops" / "GET_ROWS" / "route-matches.json").read_text())
    assert route_matches["rows"][0]["matched_route_ids"] == ["get_rows_q8_0_f32_embedding_rows_descriptor_4d"]
    config = json.loads(Path(summary["generated_config_paths"][0]).read_text())
    assert config["kernel"] == "get_rows_q8_0_f32"
    assert config["route_id"] == "get_rows_q8_0_f32_embedding_rows_descriptor_4d"
    shape = dict(zip(config["params"], config["cases"][0], strict=True))
    assert shape == {
        "d0": 4096,
        "d1": 512,
        "d2": 1,
        "d3": 1,
        "src0_d1": 128256,
        "src1_d0": 512,
        "src1_d1": 1,
    }


def test_v2_catalog_rejects_binding_with_source_and_value(tmp_path: Path) -> None:
    routing_dir = tmp_path / "routing"
    routing_dir.mkdir(parents=True, exist_ok=True)
    (routing_dir / "router.json").write_text(
        json.dumps(
            {
                "schema": "ggml_hrx_kernel_bench.routing_descriptors.v2",
                "routes": {"CPY": ["copy/bad.json"]},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    copy_dir = routing_dir / "copy"
    copy_dir.mkdir(parents=True, exist_ok=True)
    (copy_dir / "bad.json").write_text(
        json.dumps(
            {
                "id": "copy_bad",
                "family": "copy_bad",
                "kernel": {
                    "source_id": "copy_bad",
                    "path": "copy/copy_bad_contiguous_1d.loom",
                    "root_symbol": "@copy_bad",
                    "export_name": "copy_bad",
                },
                "tensors": {
                    "src0": {"dtype": "F32", "dimensions": "src0_dimensions", "strides": "src0_strides"},
                    "dst": {"dtype": "F32", "dimensions": "dst_dimensions", "strides": "dst_strides"},
                },
                "values": [{"name": "total_size", "product": "dst_dimensions"}],
                "constraints": [{"equals": ["src0_dimensions", "dst_dimensions"]}],
                "launch": {"workgroup_size": [256, 1, 1]},
                "config": {
                    "bindings": [
                        {
                            "key": "@hrx2.shape.copy.n",
                            "source": "value.total_size",
                            "value": "256",
                        }
                    ]
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="must not define both source and value"):
        load_route_catalog(routing_dir)


def test_v2_catalog_objects_are_immutable(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernels"
    routing_dir = tmp_path / "routing"
    _write_kernel(kernel_dir)
    _write_v2_descriptor(routing_dir)

    catalog = load_route_catalog(routing_dir)
    route = catalog.routes[0]

    with pytest.raises(TypeError):
        catalog.routes_by_id["new"] = route  # type: ignore[index]
    with pytest.raises(TypeError):
        route.tensors["extra"] = route.tensors["src0"]  # type: ignore[index]
    with pytest.raises(TypeError):
        route.launch["rows_per_workgroup"] = 2  # type: ignore[index]


def test_v2_candidate_route_payload_uses_capture_lists(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernels"
    routing_dir = tmp_path / "routing"
    _write_kernel(kernel_dir)
    _write_v2_descriptor(routing_dir)
    router = create_router(version="v2", kernel_dir=kernel_dir, routing_dir=routing_dir)

    candidate = router.candidates(CandidateQuery())[0]

    assert candidate.route["tensors"]["src0"]["dimensions"] == "src0_dimensions"
    assert "permutation" not in candidate.route["tensors"]["src0"]
    assert candidate.route["values"] == [
        {"name": "contiguous_strides", "contiguous_strides": "dst_dimensions"},
        {"name": "total_size", "product": "dst_dimensions"},
    ]
    assert candidate.route["constraints"] == [
        {"equals": ["src0_dimensions", "src1_dimensions", "dst_dimensions"]},
        {"equals": ["contiguous_strides", "src0_strides", "src1_strides", "dst_strides"]},
    ]


def test_v2_generic_4d_candidate_binds_dimension_sizes_and_strides(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernels"
    routing_dir = tmp_path / "routing"
    _write_kernel(kernel_dir)
    _write_v2_descriptor(routing_dir)
    router = create_router(version="v2", kernel_dir=kernel_dir, routing_dir=routing_dir)

    generic = next(
        candidate for candidate in router.candidates(CandidateQuery()) if candidate.route_id == "add_f32_generic_4d"
    )

    assert generic.config == {
        "@hrx2.shape.add4d.ne0": "1",
        "@hrx2.shape.add4d.ne1": "1",
        "@hrx2.shape.add4d.ne2": "1",
        "@hrx2.shape.add4d.ne3": "1",
        "@hrx2.shape.add4d.src0_ne0": "1",
        "@hrx2.shape.add4d.src0_ne1": "1",
        "@hrx2.shape.add4d.src0_ne2": "1",
        "@hrx2.shape.add4d.src0_ne3": "1",
        "@hrx2.shape.add4d.src1_ne0": "1",
        "@hrx2.shape.add4d.src1_ne1": "1",
        "@hrx2.shape.add4d.src1_ne2": "1",
        "@hrx2.shape.add4d.src1_ne3": "1",
        "@hrx2.stride.add4d.src0_nb0": "1",
        "@hrx2.stride.add4d.src0_nb1": "1",
        "@hrx2.stride.add4d.src0_nb2": "1",
        "@hrx2.stride.add4d.src0_nb3": "1",
        "@hrx2.stride.add4d.src1_nb0": "1",
        "@hrx2.stride.add4d.src1_nb1": "1",
        "@hrx2.stride.add4d.src1_nb2": "1",
        "@hrx2.stride.add4d.src1_nb3": "1",
        "@hrx2.stride.add4d.dst_nb0": "1",
        "@hrx2.stride.add4d.dst_nb1": "1",
        "@hrx2.stride.add4d.dst_nb2": "1",
        "@hrx2.stride.add4d.dst_nb3": "1",
        "@hrx2.tuning.add4d.workgroup_size": "256",
    }

def test_v2_router_executes_matching_case(tmp_path: Path, monkeypatch) -> None:
    kernel_dir = tmp_path / "kernels"
    routing_dir = tmp_path / "routing"
    output_dir = tmp_path / "out"
    _write_kernel(kernel_dir)
    _write_v2_descriptor(routing_dir)
    router = create_router(version="v2", kernel_dir=kernel_dir, routing_dir=routing_dir)
    config = {
        "kernel": "add_f32",
        "route_id": "add_f32_contiguous_1d",
        "params": ["d0", "d1", "d2", "d3"],
        "cases": [[16, 64, 1, 1]],
    }

    def fake_run_candidate_test_row(args, bench_config, candidate, *, sanitizer):
        assert sanitizer == "none"
        assert candidate.source_path == kernel_dir / "add" / "contiguous_1d.loom"
        assert candidate.root_symbol == "@hrx2_add_f32_contiguous_1d"
        assert candidate.config == {
            "@hrx2.shape.pointwise.total_size": "1024",
            "@hrx2.tuning.pointwise.workgroup_size": "256",
        }
        return {
            "status": "ran",
            "test": {
                "summary": {
                    "correctness": {"state": "ok"},
                    "operation_timing_ns": {"mean": 1.0},
                }
            },
        }

    monkeypatch.setattr(
        "ggml_hrx_kernel_bench.routing.v2.runtime.run_candidate_test_row",
        fake_run_candidate_test_row,
    )
    execution = router.execute_case(
        RuntimeCaseRequest(
            kernel_dir=kernel_dir,
            routing_dir=routing_dir,
            config_data=config,
            current_case_id="d016_d164_d21_d31",
            current_case_values=[16, 64, 1, 1],
            tool_dir=None,
            target="gfx1100",
            rocm_path=None,
            iterations=1,
            warmup_iterations=0,
            max_batches=1,
            output_dir=output_dir,
            require_tool=lambda name, tool_dir=None: "/bin/true",
        )
    )

    result = router.case_result(execution)

    assert result["status"] == "ran"
    assert result["correctness_ok"] is True
    assert result["shape"] == {"d0": 16, "d1": 64, "d2": 1, "d3": 1}

@pytest.mark.parametrize(
    ("src0_dtype", "src0_shape", "src1_shape", "dst_shape", "expected_route_id", "expected_shape"),
    (
        (
            "Q8_0",
            [4096, 1024, 1, 1],
            [4096, 1, 1, 1],
            [1024, 1, 1, 1],
            "mul_mat_q8_0_f32_contiguous_4d",
            {"k": 4096, "rows": 1024, "cols": 1},
        ),
        (
            "Q4_K",
            [256, 16, 1, 1],
            [256, 8, 1, 1],
            [16, 8, 1, 1],
            "mul_mat_q4_k_f32_direct_contiguous_4d",
            {"k": 256, "rows": 16, "cols": 8},
        ),
        (
            "Q5_K",
            [256, 16, 1, 1],
            [256, 1, 1, 1],
            [16, 1, 1, 1],
            "mul_mat_q5_k_f32_dot16_contiguous_cols1_4d",
            {"k": 256, "rows": 16, "cols": 1},
        ),
        (
            "Q6_K",
            [256, 16, 1, 1],
            [256, 8, 1, 1],
            [16, 8, 1, 1],
            "mul_mat_q6_k_f32_direct_contiguous_4d",
            {"k": 256, "rows": 16, "cols": 8},
        ),
        (
            "F32",
            [256, 16, 1, 1],
            [256, 8, 1, 1],
            [16, 8, 1, 1],
            "mul_mat_f32_f32_contiguous_4d",
            {"k": 256, "rows": 16, "cols": 8},
        ),
        (
            "F16",
            [256, 16, 1, 1],
            [256, 8, 1, 1],
            [16, 8, 1, 1],
            "mul_mat_f16_f32_batched_contiguous_4d",
            {"k": 256, "rows": 16, "cols": 8},
        ),
    ),
)
def test_yaml_route_import_matches_rank4_quantized_mul_mat_descriptor(
    tmp_path: Path,
    src0_dtype: str,
    src0_shape: list[int],
    src1_shape: list[int],
    dst_shape: list[int],
    expected_route_id: str,
    expected_shape: dict[str, int],
) -> None:
    yaml_path = tmp_path / "mul_mat.yaml"
    yaml_path.write_text(
        json.dumps(
            {
                "ops": {
                    "MUL_MAT": [
                        {
                            "inputs": [
                                {"dtype": src0_dtype, "shape": src0_shape},
                                {"dtype": "F32", "shape": src1_shape},
                            ],
                            "destinations": [
                                {"dtype": "F32", "shape": dst_shape},
                            ],
                        }
                    ]
                }
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "route-import"
    summary = materialize_yaml_route_import(
        [yaml_path],
        output_dir=output_dir,
        routing_dir=ACTUAL_V2_ROUTING_DIR,
    )

    op_summary = next(row for row in summary["operations"] if row["op"] == "MUL_MAT")
    assert op_summary["matched_case_count"] == 1
    route_matches = json.loads((output_dir / "ops" / "MUL_MAT" / "route-matches.json").read_text())
    assert route_matches["rows"][0]["matched_route_ids"] == [expected_route_id]
    assert summary["generated_config_count"] == 1
    config_path = Path(summary["generated_config_paths"][0])
    config = json.loads(config_path.read_text())
    assert config["route_id"] == expected_route_id
    shape = dict(zip(config["params"], config["cases"][0], strict=True))
    assert shape["k"] == expected_shape["k"]
    assert shape["rows"] == expected_shape["rows"]
    assert shape["cols"] == expected_shape["cols"]


def test_yaml_route_import_matches_unmasked_rank4_soft_max_descriptor(tmp_path: Path) -> None:
    yaml_path = tmp_path / "soft_max.yaml"
    yaml_path.write_text(
        json.dumps(
            {
                "ops": {
                    "SOFT_MAX": [
                        {
                            "inputs": [{"dtype": "F32", "shape": [16, 2, 32, 1]}],
                            "destinations": [{"dtype": "F32", "shape": [16, 2, 32, 1]}],
                            "attributes": {
                                "m_prec": "f16",
                                "mask": 0,
                                "max_bias": 0.0,
                                "nr23": [1, 1],
                                "scale": 0.1,
                                "sinks": 0,
                            },
                        },
                        {
                            "inputs": [{"dtype": "F32", "shape": [16, 2, 32, 1]}],
                            "destinations": [{"dtype": "F32", "shape": [16, 2, 32, 1]}],
                            "attributes": {
                                "m_prec": "f16",
                                "mask": 0,
                                "max_bias": 0.0,
                                "nr23": [1, 1],
                                "scale": 0.1,
                                "sinks": 1,
                            },
                        },
                        {
                            "inputs": [{"dtype": "F32", "shape": [16, 2, 32, 1]}],
                            "destinations": [{"dtype": "F32", "shape": [16, 2, 32, 1]}],
                            "attributes": {
                                "m_prec": "f16",
                                "mask": 1,
                                "max_bias": 0.0,
                                "nr23": [1, 1],
                                "scale": 0.1,
                                "sinks": 0,
                            },
                        },
                    ]
                }
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "route-import"
    summary = materialize_yaml_route_import(
        [yaml_path],
        output_dir=output_dir,
        routing_dir=ACTUAL_V2_ROUTING_DIR,
    )

    op_summary = next(row for row in summary["operations"] if row["op"] == "SOFT_MAX")
    assert op_summary["matched_case_count"] == 2
    assert op_summary["unmatched_case_count"] == 1
    route_matches = json.loads((output_dir / "ops" / "SOFT_MAX" / "route-matches.json").read_text())
    assert route_matches["rows"][0]["matched_route_ids"] == ["soft_max_f32_contiguous_4d"]
    assert route_matches["rows"][1]["matched_route_ids"] == ["soft_max_f32_mask_contiguous_4d"]
    route_unmatched = json.loads((output_dir / "ops" / "SOFT_MAX" / "route-unmatched.json").read_text())
    assert route_unmatched["rows"][0]["case"]["attributes"]["sinks"] == 1
    configs = {
        json.loads(Path(raw_path).read_text())["route_id"]: json.loads(Path(raw_path).read_text())
        for raw_path in summary["generated_config_paths"]
    }
    plain_shape = dict(
        zip(
            configs["soft_max_f32_contiguous_4d"]["params"],
            configs["soft_max_f32_contiguous_4d"]["cases"][0],
            strict=True,
        )
    )
    masked_shape = dict(
        zip(
            configs["soft_max_f32_mask_contiguous_4d"]["params"],
            configs["soft_max_f32_mask_contiguous_4d"]["cases"][0],
            strict=True,
        )
    )
    assert plain_shape == {"d0": 16, "d1": 2, "d2": 32, "d3": 1}
    assert masked_shape == {"d0": 16, "d1": 2, "d2": 32, "d3": 1}
    assert configs["soft_max_f32_contiguous_4d"]["execution_abi"]["entries"] == [
        {
            "position": 0,
            "role": "scale",
            "kind": "scalar",
            "dtype": "f32",
            "value": 0.75,
        },
        {
            "position": 1,
            "role": "src0",
            "kind": "input",
            "dtype": "f32",
            "fixture": "src0",
        },
        {
            "position": 2,
            "role": "dst",
            "kind": "output",
            "dtype": "f32",
            "fixture": "dst_init",
            "expect": {
                "fixture": "expected",
                "mode": "close",
            },
        },
    ]
    assert configs["soft_max_f32_mask_contiguous_4d"]["execution_abi"]["entries"] == [
        {
            "position": 0,
            "role": "scale",
            "kind": "scalar",
            "dtype": "f32",
            "value": 0.75,
        },
        {
            "position": 1,
            "role": "src0",
            "kind": "input",
            "dtype": "f32",
            "fixture": "src0",
        },
        {
            "position": 2,
            "role": "mask",
            "kind": "input",
            "dtype": "f32",
            "fixture": "mask",
        },
        {
            "position": 3,
            "role": "dst",
            "kind": "output",
            "dtype": "f32",
            "fixture": "dst_init",
            "expect": {
                "fixture": "expected",
                "mode": "close",
            },
        },
    ]


def test_yaml_route_import_matches_default_rank4_rope_descriptor(tmp_path: Path) -> None:
    yaml_path = tmp_path / "rope.yaml"
    yaml_path.write_text(
        json.dumps(
            {
                "ops": {
                    "ROPE": [
                        {
                            "inputs": [{"dtype": "F32", "shape": [128, 32, 2, 1]}],
                            "destinations": [{"dtype": "F32", "shape": [128, 32, 2, 1]}],
                            "attributes": {
                                "af": 1.0,
                                "ef": 0.0,
                                "ff": 0,
                                "fs": 1.0,
                                "mode": 0,
                                "n_ctx": 512,
                                "n_dims": 128,
                            },
                        },
                        {
                            "inputs": [
                                {
                                    "dtype": "F32",
                                    "shape": [128, 32, 2, 1],
                                    "storage_shape": [256, 128, 6, 1],
                                }
                            ],
                            "destinations": [{"dtype": "F32", "shape": [128, 32, 2, 1]}],
                            "attributes": {
                                "af": 1.0,
                                "ef": 0.0,
                                "ff": 0,
                                "fs": 1.0,
                                "mode": 0,
                                "n_ctx": 512,
                                "n_dims": 128,
                            },
                        },
                        {
                            "inputs": [{"dtype": "F32", "shape": [128, 64, 2, 1]}],
                            "destinations": [{"dtype": "F32", "shape": [128, 64, 2, 1]}],
                            "attributes": {
                                "af": 1.0,
                                "ef": 0.0,
                                "ff": 0,
                                "fs": 1.0,
                                "mode": 0,
                                "n_ctx": 512,
                                "n_dims": 128,
                            },
                        },
                        {
                            "inputs": [{"dtype": "F32", "shape": [64, 128, 2, 1]}],
                            "destinations": [{"dtype": "F32", "shape": [64, 128, 2, 1]}],
                            "attributes": {
                                "af": 1.0,
                                "ef": 0.0,
                                "ff": 0,
                                "fs": 1.0,
                                "mode": 2,
                                "n_ctx": 512,
                                "n_dims": 64,
                            },
                        },
                        {
                            "inputs": [{"dtype": "F32", "shape": [64, 8, 2, 1]}],
                            "destinations": [{"dtype": "F32", "shape": [64, 8, 2, 1]}],
                            "attributes": {
                                "af": 1.0,
                                "ef": 0.0,
                                "ff": 0,
                                "fs": 1.0,
                                "mode": 2,
                                "n_ctx": 512,
                                "n_dims": 64,
                            },
                        },
                        {
                            "inputs": [
                                {
                                    "dtype": "F32",
                                    "shape": [64, 128, 2, 1],
                                    "storage_shape": [128, 512, 6, 1],
                                }
                            ],
                            "destinations": [{"dtype": "F32", "shape": [64, 128, 2, 1]}],
                            "attributes": {
                                "af": 1.0,
                                "ef": 0.0,
                                "ff": 0,
                                "fs": 1.0,
                                "mode": 2,
                                "n_ctx": 512,
                                "n_dims": 64,
                            },
                        },
                    ]
                }
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "route-import"
    summary = materialize_yaml_route_import(
        [yaml_path],
        output_dir=output_dir,
        routing_dir=ACTUAL_V2_ROUTING_DIR,
    )

    op_summary = next(row for row in summary["operations"] if row["op"] == "ROPE")
    assert op_summary["matched_case_count"] == 6
    assert op_summary["unmatched_case_count"] == 0
    route_matches = json.loads((output_dir / "ops" / "ROPE" / "route-matches.json").read_text())
    assert route_matches["rows"][0]["matched_route_ids"] == ["rope_f32_normal_n128_h32_t2_contiguous_4d"]
    assert route_matches["rows"][1]["matched_route_ids"] == ["rope_f32_normal_n128_h32_t2_contiguous_4d"]
    assert route_matches["rows"][2]["matched_route_ids"] == ["rope_f32_normal_n128_h32_t2_contiguous_4d"]
    assert route_matches["rows"][3]["matched_route_ids"] == ["rope_neox_f32_n64_h128_t2_contiguous_4d"]
    assert route_matches["rows"][4]["matched_route_ids"] == ["rope_neox_f32_n64_h128_t2_contiguous_4d"]
    assert route_matches["rows"][5]["matched_route_ids"] == ["rope_neox_f32_n64_h128_t2_contiguous_4d"]
    route_shapes: dict[str, list[dict[str, int]]] = {}
    route_abis: dict[str, dict[str, object]] = {}
    for raw_path in summary["generated_config_paths"]:
        config = json.loads(Path(raw_path).read_text())
        route_shapes.setdefault(config["route_id"], []).extend(
            dict(zip(config["params"], case_values, strict=True)) for case_values in config["cases"]
        )
        route_abis.setdefault(config["route_id"], config["execution_abi"])
    normal_shapes = route_shapes["rope_f32_normal_n128_h32_t2_contiguous_4d"]
    neox_shapes = route_shapes["rope_neox_f32_n64_h128_t2_contiguous_4d"]
    normal_shape = next(shape for shape in normal_shapes if "src0_d1_stride" not in shape)
    normal_padded_shape = next(shape for shape in normal_shapes if "src0_d1_stride" in shape)
    neox_shape = next(shape for shape in neox_shapes if "src0_d1_stride" not in shape)
    neox_padded_shape = next(shape for shape in neox_shapes if "src0_d1_stride" in shape)
    assert normal_shape["rope.ncols"] == 128
    assert normal_shape["rope.n_dims"] == 128
    assert normal_shape["rope.nheads"] == 32
    assert normal_shape["rope.ntokens"] == 2
    assert normal_shape["rope.src0_head_stride"] == 128
    assert normal_shape["rope.src0_token_stride"] == 4096
    assert normal_shape["rope.dst_head_stride"] == 128
    assert normal_shape["rope.dst_token_stride"] == 4096
    assert normal_shape["rope.pos_token_stride"] == 1
    assert normal_padded_shape["rope.src0_head_stride"] == 256
    assert normal_padded_shape["rope.src0_token_stride"] == 32768
    assert normal_padded_shape["rope.dst_head_stride"] == 128
    assert normal_padded_shape["rope.dst_token_stride"] == 4096
    normal_h64_shape = next(shape for shape in normal_shapes if shape["rope.nheads"] == 64)
    assert normal_h64_shape["rope.ncols"] == 128
    assert normal_h64_shape["rope.n_dims"] == 128
    assert normal_h64_shape["rope.ntokens"] == 2
    assert normal_h64_shape["rope.src0_head_stride"] == 128
    assert normal_h64_shape["rope.src0_token_stride"] == 8192
    assert normal_h64_shape["rope.dst_head_stride"] == 128
    assert normal_h64_shape["rope.dst_token_stride"] == 8192
    assert neox_shape["rope.ncols"] == 64
    assert "rope.n_dims" not in neox_shape
    assert neox_shape["rope.nheads"] == 128
    assert neox_shape["rope.ntokens"] == 2
    assert neox_shape["rope.src0_head_stride"] == 64
    assert neox_shape["rope.src0_token_stride"] == 8192
    assert neox_shape["rope.dst_head_stride"] == 64
    assert neox_shape["rope.dst_token_stride"] == 8192
    assert neox_shape["rope.pos_token_stride"] == 1
    neox_h8_shape = next(shape for shape in neox_shapes if shape["rope.nheads"] == 8)
    assert neox_h8_shape["rope.ncols"] == 64
    assert "rope.n_dims" not in neox_h8_shape
    assert neox_h8_shape["rope.ntokens"] == 2
    assert neox_h8_shape["rope.src0_head_stride"] == 64
    assert neox_h8_shape["rope.src0_token_stride"] == 512
    assert neox_h8_shape["rope.dst_head_stride"] == 64
    assert neox_h8_shape["rope.dst_token_stride"] == 512
    assert neox_padded_shape["rope.src0_head_stride"] == 128
    assert neox_padded_shape["rope.src0_token_stride"] == 65536
    assert neox_padded_shape["rope.dst_head_stride"] == 64
    assert neox_padded_shape["rope.dst_token_stride"] == 8192
    expected_abi = [
        {
            "position": 0,
            "role": "theta_scale",
            "kind": "scalar",
            "dtype": "f32",
            "value": 0.75,
        },
        {
            "position": 1,
            "role": "freq_scale",
            "kind": "scalar",
            "dtype": "f32",
            "value": 1.1,
        },
        {
            "position": 2,
            "role": "attn_factor",
            "kind": "scalar",
            "dtype": "f32",
            "value": 0.9,
        },
        {
            "position": 3,
            "role": "src0",
            "kind": "input",
            "dtype": "f32",
            "fixture": "src0",
        },
        {
            "position": 4,
            "role": "src1",
            "kind": "input",
            "dtype": "i32",
            "fixture": "positions",
        },
        {
            "position": 5,
            "role": "dst",
            "kind": "output",
            "dtype": "f32",
            "fixture": "dst_init",
            "expect": {
                "fixture": "expected",
                "mode": "close",
            },
        },
    ]
    assert route_abis["rope_f32_normal_n128_h32_t2_contiguous_4d"]["entries"] == expected_abi
    assert route_abis["rope_neox_f32_n64_h128_t2_contiguous_4d"]["entries"] == expected_abi

@pytest.mark.parametrize(
    ("route_id", "k", "workgroup_count"),
    (
        ("softmax_kqv_f32_f16_decode_kv256_d128_h24_hkv8_wg64_rows2", 256, [64, 24, 1]),
        ("softmax_kqv_f32_f16_decode_kv512_d128_h24_hkv8_wg256_rows128", 512, [1, 24, 1]),
        ("softmax_kqv_f32_f16_decode_kv768_d128_h24_hkv8_wg256_rows128", 768, [1, 24, 1]),
    ),
)
def test_v2_flash_attn_ext_routes_materialize_softmax_kqv_decode_variants(
    route_id: str,
    k: int,
    workgroup_count: list[int],
) -> None:
    catalog = load_route_catalog(ACTUAL_V2_ROUTING_DIR)
    routes = list(routes_for_op(catalog, "FLASH_ATTN_EXT"))
    route = next(current for current in routes if current.id == route_id)

    candidate = candidate_from_shape(
        kernel_dir=ACTUAL_V2_KERNEL_DIR,
        route=route,
        shape=_softmax_kqv_shape(k),
    )

    assert candidate.family == "softmax_kqv_f32_f16"
    assert candidate.status == "planned"
    assert candidate.dispatch["workgroup_count"] == workgroup_count
    assert candidate.shape["k"] == k
    assert candidate.shape["rows"] == 128
    assert candidate.shape["cols"] == 24
    assert candidate.shape["nheads_kv"] == 8


@pytest.mark.parametrize("kv", (512, 1024, 2048, 4096))
def test_v2_flash_attn_ext_masked_identity_route_materializes_candidates(kv: int) -> None:
    catalog = load_route_catalog(ACTUAL_V2_ROUTING_DIR)
    route = next(
        current
        for current in routes_for_op(catalog, "FLASH_ATTN_EXT")
        if current.id == "softmax_kqv_f32_f16_masked_identity_kv512_4096_d128_h8_wg256_row1"
    )

    candidate = candidate_from_shape(
        kernel_dir=ACTUAL_V2_KERNEL_DIR,
        route=route,
        shape=_softmax_kqv_shape(kv, cols=8),
    )

    assert candidate.family == "softmax_kqv_f32_f16"
    assert candidate.status == "planned"
    assert candidate.dispatch["workgroup_count"] == [128, 8, 1]
    assert candidate.shape["k"] == kv
    assert candidate.shape["rows"] == 128
    assert candidate.shape["cols"] == 8
    assert candidate.shape["nheads_kv"] == 8
