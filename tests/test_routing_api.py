from __future__ import annotations

import json
from pathlib import Path

import pytest

from ggml_hrx_kernel_bench.generators.copy import (
    render_catalog_artifacts,
    render_kernel_artifacts,
)
from ggml_hrx_kernel_bench.import_models import (
    ImportedCase,
    ImportedOpGroup,
    ImportedSuite,
)
from ggml_hrx_kernel_bench.routing.api import (
    CandidateQuery,
    RuntimeCaseRequest,
    create_router,
)
from ggml_hrx_kernel_bench.import_route_resolution import resolve_case_routes
from ggml_hrx_kernel_bench.routing.v2.import_resolution import resolve_imported_suite, resolve_route_for_case
from ggml_hrx_kernel_bench.routing.v2.candidates import candidate_from_shape
from ggml_hrx_kernel_bench.routing.v1.routes import DEFAULT_V1_ROUTING_DIR, iter_routes
from ggml_hrx_kernel_bench.routing.v2.query import load_route_catalog
from ggml_hrx_kernel_bench.routing.v2.models import (
    ConstraintCheck,
    RouteConstraints,
    TensorDescriptor,
    V2Route,
    ValueDefinition,
)
from ggml_hrx_kernel_bench.routing.v2.manifest import build_manifest
from ggml_hrx_kernel_bench.routing.v2.query import load_route_catalog, routes_for_op


ACTUAL_V2_ROUTING_DIR = Path(__file__).resolve().parents[1] / "catalog" / "v2"
ACTUAL_V2_KERNEL_DIR = Path(__file__).resolve().parents[1] / "kernels" / "v2"


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


def test_v2_resolve_copy_route_for_contiguous_case(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernels"
    routing_dir = tmp_path / "routing"
    _write_copy_kernel(kernel_dir)
    _write_v2_copy_descriptor(routing_dir)
    catalog = load_route_catalog(routing_dir)
    case = ImportedCase(
        op="CPY",
        dtype={"type_src": "f32", "type_dst": "f16"},
        raw_case={},
        normalized_params={
            "ne": [16, 4, 2, 2],
            "_src_transpose": 0,
            "permute_src": [0, 0, 0, 0],
            "permute_dst": [0, 0, 0, 0],
        },
        source_path="tests/kernels/data/llamacpp_test.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    route, shape, reason, detail = resolve_route_for_case(case, list(routes_for_op(catalog, "CPY")))

    assert reason is None
    assert detail is None
    assert route is not None
    assert route.id == "copy_f32_f16_contiguous_1d"
    assert shape == {"d0": 16, "d1": 4, "d2": 2, "d3": 2}


def test_v2_resolve_copy_route_for_transposed_f32_case(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernels"
    routing_dir = tmp_path / "routing"
    _write_copy_kernel(kernel_dir)
    _write_v2_copy_descriptor(routing_dir)
    catalog = load_route_catalog(routing_dir)
    case = ImportedCase(
        op="CPY",
        dtype={"type_src": "f32", "type_dst": "f32"},
        raw_case={},
        normalized_params={
            "ne": [256, 4, 3, 1],
            "_src_transpose": 1,
            "permute_src": [0, 0, 0, 0],
            "permute_dst": [0, 0, 0, 0],
        },
        source_path="tests/kernels/data/llamacpp_test.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    route, shape, reason, detail = resolve_route_for_case(case, list(routes_for_op(catalog, "CPY")))

    assert reason is None
    assert detail is None
    assert route is not None
    assert route.id == "copy_f32_f32_non_contiguous_4d"
    assert shape == {
        "d0": 4,
        "d1": 256,
        "d2": 3,
        "d3": 1,
        "dst_perm0": 1,
        "dst_perm1": 0,
        "dst_perm2": 2,
        "dst_perm3": 3,
        "src0_d0_stride": 256,
        "src0_d1_stride": 1,
    }


def test_v2_resolve_copy_route_for_chained_source_and_destination_permutations(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernels"
    routing_dir = tmp_path / "routing"
    _write_copy_kernel(kernel_dir)
    _write_v2_copy_descriptor(routing_dir)
    catalog = load_route_catalog(routing_dir)
    case = ImportedCase(
        op="CPY",
        dtype={"type_src": "f32", "type_dst": "f32"},
        raw_case={},
        normalized_params={
            "ne": [2, 3, 5, 7],
            "_src_transpose": 0,
            "permute_src": [0, 2, 1, 3],
            "permute_dst": [0, 3, 1, 2],
        },
        source_path="tests/kernels/data/llamacpp_test.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    route, shape, reason, detail = resolve_route_for_case(case, list(routes_for_op(catalog, "CPY")))

    assert reason is None
    assert detail is None
    assert route is not None
    assert route.id == "copy_f32_f32_non_contiguous_4d"
    assert shape == {
        "d0": 2,
        "d1": 7,
        "d2": 3,
        "d3": 5,
        "dst_perm0": 0,
        "dst_perm1": 3,
        "dst_perm2": 1,
        "dst_perm3": 2,
        "src0_d1_stride": 30,
        "src0_d2_stride": 10,
        "src0_d3_stride": 2,
        "src0_perm0": 0,
        "src0_perm1": 2,
        "src0_perm2": 1,
        "src0_perm3": 3,
    }


def test_v2_copy_catalog_infers_lowering_kinds_from_generated_descriptors(tmp_path: Path) -> None:
    routing_dir = tmp_path / "routing"
    _write_v2_copy_descriptor(routing_dir)

    catalog = load_route_catalog(routing_dir)
    by_id = {route.id: route for route in catalog.routes}

    assert by_id["copy_f32_f16_contiguous_1d"].lowering_kind == "copy_contiguous"
    assert by_id["copy_f32_f32_non_contiguous_4d"].lowering_kind == "copy_non_contiguous_4d"


def test_v2_resolve_cont_route_for_contiguous_f32_case() -> None:
    catalog = load_route_catalog(ACTUAL_V2_ROUTING_DIR)
    case = ImportedCase(
        op="CONT",
        dtype={"type": "f32"},
        raw_case={},
        normalized_params={
            "ne": [2, 3, 5, 7],
            "use_view_slice": 0,
        },
        source_path="tests/kernels/data/llamacpp_test.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    route, shape, reason, detail = resolve_route_for_case(case, list(routes_for_op(catalog, "CONT")))

    assert reason is None
    assert detail is None
    assert route is not None
    assert route.id == "cont_f32_contiguous_4d"
    assert shape == {"d0": 2, "d1": 3, "d2": 5, "d3": 7, "cont.d1": 105}


def test_v2_resolve_cont_route_for_rank2_f32_case() -> None:
    catalog = load_route_catalog(ACTUAL_V2_ROUTING_DIR)
    case = ImportedCase(
        op="CONT",
        dtype={"type": "f32"},
        raw_case={},
        normalized_params={
            "ne": [7, 5],
            "use_view_slice": 0,
        },
        source_path="tests/kernels/data/llamacpp_test.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    route, shape, reason, detail = resolve_route_for_case(case, list(routes_for_op(catalog, "CONT")))

    assert reason is None
    assert detail is None
    assert route is not None
    assert route.id == "cont_f32_contiguous_4d"
    assert shape == {"d0": 7, "d1": 5, "cont.d1": 5}


def test_v2_resolve_cont_route_for_rank3_f32_case() -> None:
    catalog = load_route_catalog(ACTUAL_V2_ROUTING_DIR)
    case = ImportedCase(
        op="CONT",
        dtype={"type": "f32"},
        raw_case={},
        normalized_params={
            "ne": [4, 3, 2],
            "use_view_slice": 0,
        },
        source_path="tests/kernels/data/llamacpp_test.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    route, shape, reason, detail = resolve_route_for_case(case, list(routes_for_op(catalog, "CONT")))

    assert reason is None
    assert detail is None
    assert route is not None
    assert route.id == "cont_f32_contiguous_4d"
    assert shape == {"d0": 4, "d1": 3, "d2": 2, "cont.d1": 6}


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


@pytest.mark.parametrize(
    ("op", "dtype", "route_id"),
    (
        ("EXP", "f16", "exp_f16_contiguous_4d"),
        ("EXP", "f32", "exp_f32_contiguous_4d"),
        ("NEG", "f16", "neg_f16_contiguous_4d"),
        ("NEG", "f32", "neg_f32_contiguous_4d"),
        ("RELU", "f16", "relu_f16_contiguous_4d"),
        ("RELU", "f32", "relu_f32_contiguous_4d"),
    ),
)
def test_v2_resolve_ne_a_unary_route_for_contiguous_case(op: str, dtype: str, route_id: str) -> None:
    catalog = load_route_catalog(ACTUAL_V2_ROUTING_DIR)
    case = ImportedCase(
        op=op,
        dtype={"type": dtype},
        raw_case={},
        normalized_params={
            "ne_a": [4, 3, 2, 5],
            "v": 0,
        },
        source_path="tests/kernels/data/llamacpp_test.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    route, shape, reason, detail = resolve_route_for_case(case, list(routes_for_op(catalog, op)))

    assert reason is None
    assert detail is None
    assert route is not None
    assert route.id == route_id
    assert shape == {"d0": 4, "d1": 3, "d2": 2, "d3": 5, "pointwise.d1": 30}


def test_v2_relu_view_case_remains_unmapped_without_non_contiguous_unary_route() -> None:
    catalog = load_route_catalog(ACTUAL_V2_ROUTING_DIR)
    case = ImportedCase(
        op="RELU",
        dtype={"type": "f32"},
        raw_case={},
        normalized_params={
            "ne_a": [4, 3, 2, 5],
            "v": 1,
        },
        source_path="tests/kernels/data/llamacpp_test.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    route, shape, reason, detail = resolve_route_for_case(case, list(routes_for_op(catalog, "RELU")))

    assert route is None
    assert shape is None
    assert reason is not None
    assert reason.value == "shape_lowering_not_implemented"
    assert detail == "contiguous unary routing requires contiguous input (v=0)"


def test_v2_resolve_scale_route_for_f32_case() -> None:
    catalog = load_route_catalog(ACTUAL_V2_ROUTING_DIR)
    case = ImportedCase(
        op="SCALE",
        dtype={"type": "f32"},
        raw_case={},
        normalized_params={
            "ne": [10, 10, 10, 10],
            "scale": 2.0,
            "bias": 1.0,
            "inplace": 0,
        },
        source_path="tests/kernels/data/llamacpp_test.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    route, shape, reason, detail = resolve_route_for_case(case, list(routes_for_op(catalog, "SCALE")))

    assert reason is None
    assert detail is None
    assert route is not None
    assert route.id == "scale_f32_contiguous_4d"
    assert shape == {"d0": 10, "d1": 10, "d2": 10, "d3": 10, "pointwise.d1": 1000}


def test_v2_resolve_scale_route_for_rank2_f32_case() -> None:
    catalog = load_route_catalog(ACTUAL_V2_ROUTING_DIR)
    case = ImportedCase(
        op="SCALE",
        dtype={"type": "f32"},
        raw_case={},
        normalized_params={
            "ne": [7, 5],
            "scale": 2.0,
            "bias": 1.0,
            "inplace": 0,
        },
        source_path="tests/kernels/data/llamacpp_test.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    route, shape, reason, detail = resolve_route_for_case(case, list(routes_for_op(catalog, "SCALE")))

    assert reason is None
    assert detail is None
    assert route is not None
    assert route.id == "scale_f32_contiguous_4d"
    assert shape == {"d0": 7, "d1": 5, "pointwise.d1": 5}


def test_v2_resolve_scale_route_for_rank3_f32_case() -> None:
    catalog = load_route_catalog(ACTUAL_V2_ROUTING_DIR)
    case = ImportedCase(
        op="SCALE",
        dtype={"type": "f32"},
        raw_case={},
        normalized_params={
            "ne": [4, 3, 2],
            "scale": 2.0,
            "bias": 1.0,
            "inplace": 0,
        },
        source_path="tests/kernels/data/llamacpp_test.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    route, shape, reason, detail = resolve_route_for_case(case, list(routes_for_op(catalog, "SCALE")))

    assert reason is None
    assert detail is None
    assert route is not None
    assert route.id == "scale_f32_contiguous_4d"
    assert shape == {"d0": 4, "d1": 3, "d2": 2, "pointwise.d1": 6}


@pytest.mark.parametrize(
    ("op", "dtype", "route_id"),
    (
        ("SQR", "f16", "sqr_f16_contiguous_4d"),
        ("SQR", "f32", "sqr_f32_contiguous_4d"),
        ("SQRT", "f16", "sqrt_f16_contiguous_4d"),
        ("SQRT", "f32", "sqrt_f32_contiguous_4d"),
    ),
)
def test_v2_resolve_ne_unary_route_for_rank3_case(op: str, dtype: str, route_id: str) -> None:
    catalog = load_route_catalog(ACTUAL_V2_ROUTING_DIR)
    case = ImportedCase(
        op=op,
        dtype={"type": dtype},
        raw_case={},
        normalized_params={
            "ne": [4, 3, 2],
        },
        source_path="tests/kernels/data/llamacpp_test.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    route, shape, reason, detail = resolve_route_for_case(case, list(routes_for_op(catalog, op)))

    assert reason is None
    assert detail is None
    assert route is not None
    assert route.id == route_id
    assert shape == {"d0": 4, "d1": 3, "d2": 2, "pointwise.d1": 6}


def test_v2_resolve_clamp_route_for_f32_case() -> None:
    catalog = load_route_catalog(ACTUAL_V2_ROUTING_DIR)
    case = ImportedCase(
        op="CLAMP",
        dtype={"type": "f32"},
        raw_case={},
        normalized_params={
            "ne": [7, 1, 5, 3],
            "min": -0.5,
            "max": 0.5,
        },
        source_path="tests/kernels/data/llamacpp_test.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    route, shape, reason, detail = resolve_route_for_case(case, list(routes_for_op(catalog, "CLAMP")))

    assert reason is None
    assert detail is None
    assert route is not None
    assert route.id == "clamp_f32_contiguous_4d"
    assert shape == {"d0": 7, "d1": 1, "d2": 5, "d3": 3, "pointwise.d1": 15}


def test_v2_resolve_clamp_route_for_rank2_f32_case() -> None:
    catalog = load_route_catalog(ACTUAL_V2_ROUTING_DIR)
    case = ImportedCase(
        op="CLAMP",
        dtype={"type": "f32"},
        raw_case={},
        normalized_params={
            "ne": [7, 5],
            "min": -0.5,
            "max": 0.5,
        },
        source_path="tests/kernels/data/llamacpp_test.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    route, shape, reason, detail = resolve_route_for_case(case, list(routes_for_op(catalog, "CLAMP")))

    assert reason is None
    assert detail is None
    assert route is not None
    assert route.id == "clamp_f32_contiguous_4d"
    assert shape == {"d0": 7, "d1": 5, "pointwise.d1": 5}


def test_v2_resolve_clamp_route_for_rank3_f32_case() -> None:
    catalog = load_route_catalog(ACTUAL_V2_ROUTING_DIR)
    case = ImportedCase(
        op="CLAMP",
        dtype={"type": "f32"},
        raw_case={},
        normalized_params={
            "ne": [4, 3, 2],
            "min": -0.5,
            "max": 0.5,
        },
        source_path="tests/kernels/data/llamacpp_test.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    route, shape, reason, detail = resolve_route_for_case(case, list(routes_for_op(catalog, "CLAMP")))

    assert reason is None
    assert detail is None
    assert route is not None
    assert route.id == "clamp_f32_contiguous_4d"
    assert shape == {"d0": 4, "d1": 3, "d2": 2, "pointwise.d1": 6}


def test_v2_resolve_set_rows_route_for_f32_i64_case() -> None:
    catalog = load_route_catalog(ACTUAL_V2_ROUTING_DIR)
    case = ImportedCase(
        op="SET_ROWS",
        dtype={"type": "f32", "type_idx": "i64"},
        raw_case={},
        normalized_params={
            "ne": [256, 5, 7, 3],
            "nr23": [1, 1],
            "r": 1,
            "v": 0,
        },
        source_path="tests/kernels/data/llamacpp_test.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    route, shape, reason, detail = resolve_route_for_case(case, list(routes_for_op(catalog, "SET_ROWS")))

    assert reason is None
    assert detail is None
    assert route is not None
    assert route.id == "set_rows_f32_f32_contiguous_4d"
    assert shape == {
        "d0": 256,
        "d1": 5,
        "d2": 7,
        "d3": 3,
        "src0_d1": 1,
        "src1_d0": 1,
        "src1_d1": 1,
        "src1_d2": 1,
        "src1_d3": 1,
    }


def test_v2_set_rows_i32_indices_remain_unmapped_without_dtype_route() -> None:
    catalog = load_route_catalog(ACTUAL_V2_ROUTING_DIR)
    case = ImportedCase(
        op="SET_ROWS",
        dtype={"type": "f32", "type_idx": "i32"},
        raw_case={},
        normalized_params={
            "ne": [1, 8, 1, 3],
            "nr23": [1, 1],
            "r": 2,
            "v": 0,
        },
        source_path="tests/kernels/data/llamacpp_test.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    route, shape, reason, detail = resolve_route_for_case(case, list(routes_for_op(catalog, "SET_ROWS")))

    assert route is None
    assert shape is None
    assert reason is not None
    assert reason.value == "no_dtype_mapping"
    assert detail == "matching v2 op mapping exists, but not for this dtype combination"


def test_v2_resolve_set_rows_route_preserves_non_contiguous_idx_stride() -> None:
    catalog = load_route_catalog(ACTUAL_V2_ROUTING_DIR)
    case = ImportedCase(
        op="SET_ROWS",
        dtype={"type": "f32", "type_idx": "i64"},
        raw_case={},
        normalized_params={
            "ne": [31, 3, 7, 1],
            "nr23": [2, 3],
            "r": 2,
            "v": 0,
        },
        source_path="tests/kernels/data/llamacpp_test.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    route, shape, reason, detail = resolve_route_for_case(case, list(routes_for_op(catalog, "SET_ROWS")))

    assert reason is None
    assert detail is None
    assert route is not None
    assert route.id == "set_rows_f32_f32_contiguous_4d"
    assert shape is not None
    assert shape["src1_d2_stride"] == 2


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


def test_v2_router_lowers_permuted_rhs_add_case(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernels"
    routing_dir = tmp_path / "routing"
    _write_kernel(kernel_dir)
    _write_v2_descriptor(routing_dir)
    router = create_router(version="v2", kernel_dir=kernel_dir, routing_dir=routing_dir)
    suite = ImportedSuite(
        schema="test",
        source_path="test.yaml",
        op_groups=[
            ImportedOpGroup(
                op="ADD",
                dtype={"type": "f32"},
                source_path="test.yaml",
                cases=(
                    ImportedCase(
                        op="ADD",
                        dtype={"type": "f32"},
                        raw_case={
                            "ne": [10, 5, 4, 3],
                            "nr": [1, 1, 1, 1],
                            "nf": 1,
                            "perm1": [1, 2, 0, 3],
                        },
                        normalized_params={
                            "ne": [10, 5, 4, 3],
                            "nr": [1, 1, 1, 1],
                            "nf": 1,
                            "perm1": [1, 2, 0, 3],
                        },
                        source_path="test.yaml",
                        source_group_index=0,
                        source_case_index=0,
                    ),
                ),
            )
        ],
    )

    resolved = router.resolve_imported_suite(suite)

    assert len(resolved.resolved) == 1
    assert resolved.resolved[0].route_id == "add_f32_generic_4d"
    assert resolved.resolved[0].params == ["d0", "d1", "d2", "d3", "src1_d0_stride", "src1_d1_stride", "src1_d2_stride"]
    assert resolved.resolved[0].values == [10, 5, 4, 3, 20, 1, 5]
    assert resolved.unmapped == []


def test_v2_import_resolution_lowers_permuted_rhs_for_non_add_generic_route() -> None:
    route = V2Route(
        id="mul_f32_generic_4d",
        family="mul_f32",
        op="MUL",
        source_id="mul_f32",
        kernel_path="mul/generic_4d.loom",
        root_symbol="@hrx2_mul_f32_generic_4d",
        export_name="hrx2_mul_f32_generic_4d",
        tensors={
            "src0": TensorDescriptor(dtype="F32", dimensions_capture="src0_dimensions", strides_capture="src0_strides"),
            "src1": TensorDescriptor(dtype="F32", dimensions_capture="src1_dimensions", strides_capture="src1_strides"),
            "dst": TensorDescriptor(dtype="F32", dimensions_capture="dst_dimensions", strides_capture="dst_strides"),
        },
        values=(ValueDefinition(name="total_size", operation_kind="product", sources=("dst_dimensions",)),),
        constraints=RouteConstraints(
            checks=(
                ConstraintCheck(name="dst_dimensions", length=4),
                ConstraintCheck(divides=("src0_dimensions", "dst_dimensions")),
                ConstraintCheck(divides=("src1_dimensions", "dst_dimensions")),
            )
        ),
        launch={"workgroup_size": [256, 1, 1]},
        bindings=(),
    )
    case = ImportedCase(
        op="MUL",
        dtype={"type": "f32"},
        raw_case={
            "ne": [10, 5, 4, 3],
            "nr": [1, 1, 1, 1],
            "nf": 1,
            "perm1": [1, 2, 0, 3],
        },
        normalized_params={
            "ne": [10, 5, 4, 3],
            "nr": [1, 1, 1, 1],
            "nf": 1,
            "perm1": [1, 2, 0, 3],
        },
        source_path="test.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    resolved_route, shape, reason, detail = resolve_route_for_case(case, [route])

    assert resolved_route is route
    assert shape == {
        "d0": 10,
        "d1": 5,
        "d2": 4,
        "d3": 3,
        "src1_d0_stride": 20,
        "src1_d1_stride": 1,
        "src1_d2_stride": 5,
    }
    assert reason is None
    assert detail is None


def test_v2_helpers_require_catalog_or_routing_dir(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="routing_dir or catalog is required"):
        build_manifest(kernel_dir=tmp_path / "kernels")
    with pytest.raises(ValueError, match="routing_dir or catalog is required"):
        resolve_imported_suite(
            ImportedSuite(schema="test", source_path="test.yaml", op_groups=[]),
        )


def test_v2_router_maps_contiguous_and_generic_add_cases(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernels"
    routing_dir = tmp_path / "routing"
    _write_kernel(kernel_dir)
    _write_v2_descriptor(routing_dir)
    router = create_router(version="v2", kernel_dir=kernel_dir, routing_dir=routing_dir)
    suite = ImportedSuite(
        schema="test",
        source_path="test.yaml",
        op_groups=[
            ImportedOpGroup(
                op="ADD",
                dtype={"type": "f32"},
                source_path="test.yaml",
                cases=(
                    ImportedCase(
                        op="ADD",
                        dtype={"type": "f32"},
                        raw_case={"ne": [16, 64, 1, 1], "nr": [1, 1, 1, 1], "nf": 1, "perm1": [0, 1, 2, 3]},
                        normalized_params={"ne": [16, 64, 1, 1], "nr": [1, 1, 1, 1], "nf": 1, "perm1": [0, 1, 2, 3]},
                        source_path="test.yaml",
                        source_group_index=0,
                        source_case_index=0,
                    ),
                    ImportedCase(
                        op="ADD",
                        dtype={"type": "f32"},
                        raw_case={"ne": [10, 5, 1, 1], "nr": [1, 1, 1, 2], "nf": 1, "perm1": [0, 1, 2, 3]},
                        normalized_params={"ne": [10, 5, 1, 1], "nr": [1, 1, 1, 2], "nf": 1, "perm1": [0, 1, 2, 3]},
                        source_path="test.yaml",
                        source_group_index=0,
                        source_case_index=1,
                    ),
                    ImportedCase(
                        op="ADD",
                        dtype={"type": "f32"},
                        raw_case={"ne": [10, 5, 4, 3], "nr": [1, 1, 1, 1], "nf": 1, "perm1": [1, 2, 0, 3]},
                        normalized_params={"ne": [10, 5, 4, 3], "nr": [1, 1, 1, 1], "nf": 1, "perm1": [1, 2, 0, 3]},
                        source_path="test.yaml",
                        source_group_index=0,
                        source_case_index=2,
                    ),
                ),
            )
        ],
    )

    resolved = router.resolve_imported_suite(suite)

    assert len(resolved.resolved) == 3
    assert resolved.resolved[0].kernel_family == "add_f32"
    assert resolved.resolved[0].route_id == "add_f32_contiguous_1d"
    assert resolved.resolved[0].params == ["d0", "d1", "d2", "d3"]
    assert resolved.resolved[0].values == [16, 64, 1, 1]
    assert resolved.resolved[1].route_id == "add_f32_generic_4d"
    assert resolved.resolved[1].params == ["d0", "d1", "d2", "d3", "src1_d3"]
    assert resolved.resolved[1].values == [10, 5, 1, 2, 1]
    assert resolved.resolved[2].route_id == "add_f32_generic_4d"
    assert resolved.resolved[2].params == ["d0", "d1", "d2", "d3", "src1_d0_stride", "src1_d1_stride", "src1_d2_stride"]
    assert resolved.resolved[2].values == [10, 5, 4, 3, 20, 1, 5]
    assert resolved.unmapped == []


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


def test_v1_oversized_contiguous_add_case_falls_back_to_generic_route() -> None:
    case = ImportedCase(
        op="ADD",
        dtype={"type": "f32"},
        raw_case={
            "ne": [1, 102400, 1, 1],
            "nr": [1, 1, 1, 1],
            "nf": 1,
            "perm1": [0, 1, 2, 3],
        },
        normalized_params={
            "ne": [1, 102400, 1, 1],
            "nr": [1, 1, 1, 1],
            "nf": 1,
            "perm1": [0, 1, 2, 3],
        },
        source_path="tests/kernels/data/llamacpp_test.yaml",
        source_group_index=0,
        source_case_index=0,
    )
    add_routes = [
        route for route in iter_routes(DEFAULT_V1_ROUTING_DIR) if str(route.get("op") or "") == "ADD"
    ]

    resolution, _, reason, detail = resolve_case_routes(case, add_routes)

    assert reason is None
    assert detail is None
    assert resolution is not None
    assert resolution.route["id"] == "add_f32_generic_wg256"
    assert resolution.shape == {"ncols": 1, "nrows": 102400, "cols": 1, "rows": 102400}


def test_v2_oversized_contiguous_pointwise_case_becomes_unmapped() -> None:
    case = ImportedCase(
        op="ADD",
        dtype={"type": "f16"},
        raw_case={
            "ne": [64, 262144, 1, 1],
            "nr": [1, 1, 1, 1],
            "nf": 1,
            "perm1": [0, 1, 2, 3],
        },
        normalized_params={
            "ne": [64, 262144, 1, 1],
            "nr": [1, 1, 1, 1],
            "nf": 1,
            "perm1": [0, 1, 2, 3],
        },
        source_path="tests/kernels/data/llamacpp_test.yaml",
        source_group_index=0,
        source_case_index=0,
    )
    catalog = load_route_catalog(Path("catalog/v2"))
    add_routes = list(routes_for_op(catalog, "ADD"))

    resolved_route, shape, reason, detail = resolve_route_for_case(case, add_routes)

    assert resolved_route is None
    assert shape is None
    assert reason is not None
    assert reason.value == "no_route_match"
    assert detail == "lowered tensor descriptors did not satisfy any v2 route"


def test_v2_sum_rows_route_resolves_for_contiguous_case() -> None:
    case = ImportedCase(
        op="SUM_ROWS",
        dtype={"type": "f32"},
        raw_case={
            "ne": [33, 256, 1, 1],
            "permute": 0,
            "slice": 0,
        },
        normalized_params={
            "ne": [33, 256, 1, 1],
            "permute": 0,
            "slice": 0,
        },
        source_path="tests/kernels/data/llamacpp_test.yaml",
        source_group_index=0,
        source_case_index=0,
    )
    catalog = load_route_catalog(Path("catalog/v2"))
    sum_rows_routes = list(routes_for_op(catalog, "SUM_ROWS"))

    resolved_route, shape, reason, detail = resolve_route_for_case(case, sum_rows_routes)

    assert resolved_route is not None
    assert resolved_route.id == "sum_rows_f32_contiguous_4d"
    assert shape is not None
    assert shape["d0"] == 33
    assert shape["d1"] == 256
    assert reason is None
    assert detail is None


def test_v2_sum_rows_permuted_case_stays_unmapped() -> None:
    case = ImportedCase(
        op="SUM_ROWS",
        dtype={"type": "f32"},
        raw_case={
            "ne": [11, 5, 6, 3],
            "permute": 1,
            "slice": 0,
        },
        normalized_params={
            "ne": [11, 5, 6, 3],
            "permute": 1,
            "slice": 0,
        },
        source_path="tests/kernels/data/llamacpp_test.yaml",
        source_group_index=0,
        source_case_index=0,
    )
    catalog = load_route_catalog(Path("catalog/v2"))
    sum_rows_routes = list(routes_for_op(catalog, "SUM_ROWS"))

    resolved_route, shape, reason, detail = resolve_route_for_case(case, sum_rows_routes)

    assert resolved_route is None
    assert shape is None
    assert reason is not None
    assert reason.value == "shape_lowering_not_implemented"
    assert detail == "SUM_ROWS v2 routing requires permute=0"


def test_v2_rms_norm_route_resolves_for_contiguous_eps0_case() -> None:
    catalog = load_route_catalog(ACTUAL_V2_ROUTING_DIR)
    case = ImportedCase(
        op="RMS_NORM",
        dtype={"type": "f32"},
        raw_case={},
        normalized_params={
            "eps": 0.0,
            "inplace": 0,
            "ne": [1025, 5, 4, 3],
            "v": 0,
        },
        source_path="tests/kernels/data/llamacpp_test.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    rms_norm_routes = list(routes_for_op(catalog, "RMS_NORM"))

    resolved_route, shape, reason, detail = resolve_route_for_case(case, rms_norm_routes)

    assert reason is None
    assert detail is None
    assert resolved_route is not None
    assert resolved_route.id == "rms_norm_f32_contiguous_4d"
    assert shape == {"d0": 1025, "d1": 5, "d2": 4, "d3": 3}


def test_v2_rms_norm_nonzero_eps_case_stays_unmapped() -> None:
    catalog = load_route_catalog(ACTUAL_V2_ROUTING_DIR)
    case = ImportedCase(
        op="RMS_NORM",
        dtype={"type": "f32"},
        raw_case={},
        normalized_params={
            "eps": 1.0e-4,
            "inplace": 0,
            "ne": [64, 5, 4, 3],
            "v": 0,
        },
        source_path="tests/kernels/data/llamacpp_test.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    rms_norm_routes = list(routes_for_op(catalog, "RMS_NORM"))

    resolved_route, shape, reason, detail = resolve_route_for_case(case, rms_norm_routes)

    assert resolved_route is None
    assert shape is None
    assert reason is not None
    assert reason.value == "shape_lowering_not_implemented"
    assert detail == "RMS_NORM v2 routing currently requires eps=0.0"


def test_v2_swiglu_route_resolves_for_packed_contiguous_case() -> None:
    catalog = load_route_catalog(ACTUAL_V2_ROUTING_DIR)
    case = ImportedCase(
        op="SWIGLU",
        dtype={"type": "f32"},
        raw_case={},
        normalized_params={
            "ne_a": [128, 2, 2, 2],
            "swapped": 0,
            "v": 0,
        },
        source_path="tests/kernels/data/llamacpp_test.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    swiglu_routes = list(routes_for_op(catalog, "SWIGLU"))

    resolved_route, shape, reason, detail = resolve_route_for_case(case, swiglu_routes)

    assert reason is None
    assert detail is None
    assert resolved_route is not None
    assert resolved_route.id == "swiglu_f32_packed_contiguous_4d"
    assert shape == {"d0": 128, "d1": 2, "d2": 2, "d3": 2, "src0_d0": 256}


def test_v2_swiglu_split_case_stays_unmapped_without_split_route() -> None:
    catalog = load_route_catalog(ACTUAL_V2_ROUTING_DIR)
    case = ImportedCase(
        op="SWIGLU",
        dtype={"type": "f32"},
        raw_case={},
        normalized_params={
            "ne_a": [128, 2, 2, 2],
            "split": True,
            "v": 0,
        },
        source_path="tests/kernels/data/llamacpp_test.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    swiglu_routes = list(routes_for_op(catalog, "SWIGLU"))

    resolved_route, shape, reason, detail = resolve_route_for_case(case, swiglu_routes)

    assert resolved_route is None
    assert shape is None
    assert reason is not None
    assert reason.value == "shape_lowering_not_implemented"
    assert detail == "SWIGLU v2 routing currently requires packed input (split=false)"


def test_v2_swiglu_swapped_case_stays_unmapped() -> None:
    catalog = load_route_catalog(ACTUAL_V2_ROUTING_DIR)
    case = ImportedCase(
        op="SWIGLU",
        dtype={"type": "f32"},
        raw_case={},
        normalized_params={
            "ne_a": [5, 7, 11, 13],
            "swapped": 1,
            "v": 0,
        },
        source_path="tests/kernels/data/llamacpp_test.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    swiglu_routes = list(routes_for_op(catalog, "SWIGLU"))

    resolved_route, shape, reason, detail = resolve_route_for_case(case, swiglu_routes)

    assert resolved_route is None
    assert shape is None
    assert reason is not None
    assert reason.value == "shape_lowering_not_implemented"
    assert detail == "SWIGLU v2 routing currently requires swapped=0"


def test_v2_get_rows_route_resolves_for_base_f32_case() -> None:
    catalog = load_route_catalog(ACTUAL_V2_ROUTING_DIR)
    case = ImportedCase(
        op="GET_ROWS",
        dtype={"type": "f32"},
        raw_case={},
        normalized_params={
            "be1": 1,
            "be2": 1,
            "m": 5,
            "n": 256,
            "r": 4,
            "v": 0,
        },
        source_path="tests/kernels/data/llamacpp_test.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    get_rows_routes = list(routes_for_op(catalog, "GET_ROWS"))

    resolved_route, shape, reason, detail = resolve_route_for_case(case, get_rows_routes)

    assert reason is None
    assert detail is None
    assert resolved_route is not None
    assert resolved_route.id == "get_rows_f32_embedding_rows_2d"
    assert shape == {
        "d0": 256,
        "d1": 4,
        "src0_d1": 5,
        "src1_d0": 1,
        "get_rows.src0_nrows": 5,
        "get_rows.idx_row_stride": 1,
    }


def test_v2_get_rows_view_case_stays_unmapped() -> None:
    catalog = load_route_catalog(ACTUAL_V2_ROUTING_DIR)
    case = ImportedCase(
        op="GET_ROWS",
        dtype={"type": "f32"},
        raw_case={},
        normalized_params={
            "be1": 1,
            "be2": 1,
            "m": 5,
            "n": 256,
            "r": 4,
            "v": 1,
        },
        source_path="tests/kernels/data/llamacpp_test.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    get_rows_routes = list(routes_for_op(catalog, "GET_ROWS"))

    resolved_route, shape, reason, detail = resolve_route_for_case(case, get_rows_routes)

    assert resolved_route is None
    assert shape is None
    assert reason is not None
    assert reason.value == "shape_lowering_not_implemented"
    assert detail == "GET_ROWS v2 routing requires contiguous input (v=0)"


def test_v2_get_rows_non_unit_be1_case_stays_unmapped() -> None:
    catalog = load_route_catalog(ACTUAL_V2_ROUTING_DIR)
    case = ImportedCase(
        op="GET_ROWS",
        dtype={"type": "f32"},
        raw_case={},
        normalized_params={
            "be1": 7,
            "be2": 1,
            "m": 5,
            "n": 256,
            "r": 4,
            "v": 0,
        },
        source_path="tests/kernels/data/llamacpp_test.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    get_rows_routes = list(routes_for_op(catalog, "GET_ROWS"))

    resolved_route, shape, reason, detail = resolve_route_for_case(case, get_rows_routes)

    assert resolved_route is None
    assert shape is None
    assert reason is not None
    assert reason.value == "shape_lowering_not_implemented"
    assert detail == "GET_ROWS v2 routing currently requires be1=1"


def test_v2_argsort_route_resolves_for_base_f32_case() -> None:
    catalog = load_route_catalog(ACTUAL_V2_ROUTING_DIR)
    case = ImportedCase(
        op="ARGSORT",
        dtype={"type": "f32"},
        raw_case={},
        normalized_params={
            "ne": [128, 1, 1, 1],
            "order": 0,
        },
        source_path="tests/kernels/data/llamacpp_test.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    argsort_routes = list(routes_for_op(catalog, "ARGSORT"))

    resolved_route, shape, reason, detail = resolve_route_for_case(case, argsort_routes)

    assert reason is None
    assert detail is None
    assert resolved_route is not None
    assert resolved_route.id == "argsort_f32_i32_n128_r1_desc_wg128"
    assert shape == {"d0": 128, "d1": 1}


def test_v2_argsort_ascending_case_stays_unmapped() -> None:
    catalog = load_route_catalog(ACTUAL_V2_ROUTING_DIR)
    case = ImportedCase(
        op="ARGSORT",
        dtype={"type": "f32"},
        raw_case={},
        normalized_params={
            "ne": [128, 1, 1, 1],
            "order": 1,
        },
        source_path="tests/kernels/data/llamacpp_test.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    argsort_routes = list(routes_for_op(catalog, "ARGSORT"))

    resolved_route, shape, reason, detail = resolve_route_for_case(case, argsort_routes)

    assert resolved_route is None
    assert shape is None
    assert reason is not None
    assert reason.value == "shape_lowering_not_implemented"
    assert detail == "ARGSORT v2 routing currently requires order=0"


def test_v2_argsort_non_route_shape_case_stays_unmapped() -> None:
    catalog = load_route_catalog(ACTUAL_V2_ROUTING_DIR)
    case = ImportedCase(
        op="ARGSORT",
        dtype={"type": "f32"},
        raw_case={},
        normalized_params={
            "ne": [127, 1, 1, 1],
            "order": 0,
        },
        source_path="tests/kernels/data/llamacpp_test.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    argsort_routes = list(routes_for_op(catalog, "ARGSORT"))

    resolved_route, shape, reason, detail = resolve_route_for_case(case, argsort_routes)

    assert resolved_route is None
    assert shape is None
    assert reason is not None
    assert reason.value == "no_route_match"
    assert detail == "lowered tensor descriptors did not satisfy any v2 route"


def test_v2_mul_mat_route_resolves_for_small_contiguous_f32_case() -> None:
    catalog = load_route_catalog(ACTUAL_V2_ROUTING_DIR)
    case = ImportedCase(
        op="MUL_MAT",
        dtype={"type_a": "f32", "type_b": "f32"},
        raw_case={},
        normalized_params={
            "bs": [1, 1],
            "k": 256,
            "k_v": 0,
            "m": 16,
            "n": 8,
            "nr": [1, 1],
            "o": 1,
            "per": [0, 1, 2, 3],
        },
        source_path="tests/kernels/data/llamacpp_test.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    routes = list(routes_for_op(catalog, "MUL_MAT"))

    resolved_route, shape, reason, detail = resolve_route_for_case(case, routes)

    assert reason is None
    assert detail is None
    assert resolved_route is not None
    assert resolved_route.id == "mul_mat_f32_f32_contiguous_small_2d"
    assert shape == {
        "d0": 16,
        "d1": 8,
        "src0_d0": 256,
        "src0_d1": 16,
        "src1_d0": 256,
        "k": 256,
        "rows": 16,
        "cols": 8,
    }


def test_v2_mul_mat_route_resolves_for_logits_cols1_f32_case() -> None:
    catalog = load_route_catalog(ACTUAL_V2_ROUTING_DIR)
    case = ImportedCase(
        op="MUL_MAT",
        dtype={"type_a": "f32", "type_b": "f32"},
        raw_case={},
        normalized_params={
            "bs": [1, 1],
            "k": 1057,
            "k_v": 2113,
            "m": 129,
            "n": 1,
            "nr": [1, 1],
            "o": 1,
            "per": [0, 1, 2, 3],
        },
        source_path="tests/kernels/data/llamacpp_test.yaml",
        source_group_index=4,
        source_case_index=6,
    )

    routes = list(routes_for_op(catalog, "MUL_MAT"))

    resolved_route, shape, reason, detail = resolve_route_for_case(case, routes)

    assert reason is None
    assert detail is None
    assert resolved_route is not None
    assert resolved_route.id == "mul_mat_f32_f32_contiguous_logits_cols1_2d"
    assert shape == {
        "d0": 129,
        "d1": 1,
        "src0_d0": 1057,
        "src0_d1": 129,
        "src1_d0": 1057,
        "k": 1057,
        "rows": 129,
        "cols": 1,
    }


def test_v2_mul_mat_route_resolves_for_logits_cols1_f16_batched_case() -> None:
    catalog = load_route_catalog(ACTUAL_V2_ROUTING_DIR)
    case = ImportedCase(
        op="MUL_MAT",
        dtype={"type_a": "f16", "type_b": "f32"},
        raw_case={},
        normalized_params={
            "bs": [1, 1],
            "k": 1057,
            "k_v": 2113,
            "m": 129,
            "n": 1,
            "nr": [1, 1],
            "o": 1,
            "per": [0, 1, 2, 3],
        },
        source_path="tests/kernels/data/llamacpp_test.yaml",
        source_group_index=2,
        source_case_index=6,
    )

    routes = list(routes_for_op(catalog, "MUL_MAT"))

    resolved_route, shape, reason, detail = resolve_route_for_case(case, routes)

    assert reason is None
    assert detail is None
    assert resolved_route is not None
    assert resolved_route.id == "mul_mat_f16_f32_batched_logits_cols1_2d"
    assert shape == {
        "d0": 129,
        "d1": 1,
        "src0_d0": 1057,
        "src0_d1": 129,
        "src1_d0": 1057,
        "k": 1057,
        "rows": 129,
        "cols": 1,
    }

    candidate = candidate_from_shape(
        kernel_dir=ACTUAL_V2_KERNEL_DIR,
        route=resolved_route,
        shape=shape,
    )
    assert candidate.config["@hrx2.shape.mul_mat_f16.k"] == "1057"
    assert candidate.config["@hrx2.shape.mul_mat_f16.rows"] == "129"
    assert candidate.config["@hrx2.shape.mul_mat_f16.cols"] == "1"
    assert candidate.config["@hrx2.shape.mul_mat_f16.src0_stride_row"] == "1057"
    assert candidate.config["@hrx2.shape.mul_mat_f16.src1_stride_col"] == "1057"
    assert candidate.config["@hrx2.shape.mul_mat_f16.dst_stride_col"] == "129"
    assert candidate.config["@hrx2.tuning.mul_mat_f16.workgroup_size"] == "256"


def test_v2_mul_mat_route_resolves_for_q8_0_case() -> None:
    catalog = load_route_catalog(ACTUAL_V2_ROUTING_DIR)
    case = ImportedCase(
        op="MUL_MAT",
        dtype={"type_a": "q8_0", "type_b": "f32"},
        raw_case={},
        normalized_params={
            "bs": [1, 1],
            "k": 256,
            "k_v": 0,
            "m": 1,
            "n": 64,
            "nr": [1, 1],
            "o": 1,
            "per": [0, 1, 2, 3],
        },
        source_path="tests/kernels/data/llamacpp_test.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    routes = list(routes_for_op(catalog, "MUL_MAT"))

    resolved_route, shape, reason, detail = resolve_route_for_case(case, routes)

    assert reason is None
    assert detail is None
    assert resolved_route is not None
    assert resolved_route.id == "mul_mat_q8_0_f32_contiguous_2d"
    assert shape == {
        "d0": 1,
        "d1": 64,
        "src0_d0": 256,
        "src0_d1": 1,
        "src1_d0": 256,
        "k": 256,
        "rows": 1,
        "cols": 64,
    }


def test_v2_mul_mat_route_resolves_for_q5_dot16_case() -> None:
    catalog = load_route_catalog(ACTUAL_V2_ROUTING_DIR)
    case = ImportedCase(
        op="MUL_MAT",
        dtype={"type_a": "q5_k", "type_b": "f32"},
        raw_case={},
        normalized_params={
            "bs": [1, 1],
            "k": 256,
            "k_v": 0,
            "m": 16,
            "n": 1,
            "nr": [1, 1],
            "o": 1,
            "per": [0, 1, 2, 3],
        },
        source_path="tests/kernels/data/llamacpp_test.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    routes = list(routes_for_op(catalog, "MUL_MAT"))

    resolved_route, shape, reason, detail = resolve_route_for_case(case, routes)

    assert reason is None
    assert detail is None
    assert resolved_route is not None
    assert resolved_route.id == "mul_mat_q5_k_f32_dot16_contiguous_cols1_2d"
    assert shape == {
        "d0": 16,
        "d1": 1,
        "src0_d0": 256,
        "src0_d1": 16,
        "src1_d0": 256,
        "k": 256,
        "rows": 16,
        "cols": 1,
    }


def test_v2_mul_mat_route_prefers_q6_rows2_for_cols1_case() -> None:
    catalog = load_route_catalog(ACTUAL_V2_ROUTING_DIR)
    case = ImportedCase(
        op="MUL_MAT",
        dtype={"type_a": "q6_k", "type_b": "f32"},
        raw_case={},
        normalized_params={
            "bs": [1, 1],
            "k": 256,
            "k_v": 0,
            "m": 16,
            "n": 1,
            "nr": [1, 1],
            "o": 1,
            "per": [0, 1, 2, 3],
        },
        source_path="tests/kernels/data/llamacpp_test.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    routes = list(routes_for_op(catalog, "MUL_MAT"))

    resolved_route, shape, reason, detail = resolve_route_for_case(case, routes)

    assert reason is None
    assert detail is None
    assert resolved_route is not None
    assert resolved_route.id == "mul_mat_q6_k_f32_rows2_contiguous_cols1_2d"
    assert shape == {
        "d0": 16,
        "d1": 1,
        "src0_d0": 256,
        "src0_d1": 16,
        "src1_d0": 256,
        "k": 256,
        "rows": 16,
        "cols": 1,
    }


def test_v2_mul_mat_route_resolves_for_q6_direct_case() -> None:
    catalog = load_route_catalog(ACTUAL_V2_ROUTING_DIR)
    case = ImportedCase(
        op="MUL_MAT",
        dtype={"type_a": "q6_k", "type_b": "f32"},
        raw_case={},
        normalized_params={
            "bs": [1, 1],
            "k": 256,
            "k_v": 0,
            "m": 16,
            "n": 8,
            "nr": [1, 1],
            "o": 1,
            "per": [0, 1, 2, 3],
        },
        source_path="tests/kernels/data/llamacpp_test.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    routes = list(routes_for_op(catalog, "MUL_MAT"))

    resolved_route, shape, reason, detail = resolve_route_for_case(case, routes)

    assert reason is None
    assert detail is None
    assert resolved_route is not None
    assert resolved_route.id == "mul_mat_q6_k_f32_direct_contiguous_2d"
    assert shape == {
        "d0": 16,
        "d1": 8,
        "src0_d0": 256,
        "src0_d1": 16,
        "src1_d0": 256,
        "k": 256,
        "rows": 16,
        "cols": 8,
    }


def test_v2_mul_mat_broadcast_case_stays_unmapped() -> None:
    catalog = load_route_catalog(ACTUAL_V2_ROUTING_DIR)
    case = ImportedCase(
        op="MUL_MAT",
        dtype={"type_a": "f32", "type_b": "f32"},
        raw_case={},
        normalized_params={
            "bs": [1, 1],
            "k": 1056,
            "k_v": 2112,
            "m": 128,
            "n": 1,
            "nr": [4, 1],
            "o": 1,
            "per": [0, 1, 2, 3],
        },
        source_path="tests/kernels/data/llamacpp_test.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    routes = list(routes_for_op(catalog, "MUL_MAT"))

    resolved_route, shape, reason, detail = resolve_route_for_case(case, routes)

    assert resolved_route is None
    assert shape is None
    assert reason is not None
    assert reason.value == "shape_lowering_not_implemented"
    assert detail == "MUL_MAT v2 routing currently requires nr=[1, 1]"


def test_v2_mul_mat_permuted_case_stays_unmapped() -> None:
    catalog = load_route_catalog(ACTUAL_V2_ROUTING_DIR)
    case = ImportedCase(
        op="MUL_MAT",
        dtype={"type_a": "f32", "type_b": "f32"},
        raw_case={},
        normalized_params={
            "bs": [1, 1],
            "k": 128,
            "k_v": 0,
            "m": 1056,
            "n": 1,
            "nr": [1, 1],
            "o": 1,
            "per": [0, 2, 1, 3],
        },
        source_path="tests/kernels/data/llamacpp_test.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    routes = list(routes_for_op(catalog, "MUL_MAT"))

    resolved_route, shape, reason, detail = resolve_route_for_case(case, routes)

    assert resolved_route is None
    assert shape is None
    assert reason is not None
    assert reason.value == "shape_lowering_not_implemented"
    assert detail == "MUL_MAT v2 routing currently requires per=[0, 1, 2, 3]"


def test_v2_rope_route_resolves_for_plain_f32_case() -> None:
    catalog = load_route_catalog(ACTUAL_V2_ROUTING_DIR)
    case = ImportedCase(
        op="ROPE",
        dtype={"type": "f32"},
        raw_case={},
        normalized_params={
            "af": 1.0,
            "ef": 0.0,
            "ff": 0,
            "fs": 1.0,
            "inplace": 0,
            "mode": 0,
            "n_ctx": 512,
            "n_dims": 128,
            "ne_a": [128, 32, 2, 1],
            "v": 0,
        },
        source_path="tests/kernels/data/llamacpp_test.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    rope_routes = list(routes_for_op(catalog, "ROPE"))

    resolved_route, shape, reason, detail = resolve_route_for_case(case, rope_routes)

    assert reason is None
    assert detail is None
    assert resolved_route is not None
    assert resolved_route.id == "rope_f32_normal_n128_h32_t2_contiguous_4d"
    assert shape == {
        "d0": 128,
        "d1": 32,
        "d2": 2,
        "d3": 1,
        "src1_d0": 1,
        "src1_d1": 1,
        "rope.ncols": 128,
        "rope.n_dims": 128,
        "rope.nheads": 32,
        "rope.ntokens": 2,
        "rope.src0_head_stride": 128,
        "rope.src0_token_stride": 4096,
        "rope.dst_head_stride": 128,
        "rope.dst_token_stride": 4096,
        "rope.pos_token_stride": 1,
    }


def test_v2_rope_route_resolves_for_scaled_plain_f32_case() -> None:
    catalog = load_route_catalog(ACTUAL_V2_ROUTING_DIR)
    case = ImportedCase(
        op="ROPE",
        dtype={"type": "f32"},
        raw_case={},
        normalized_params={
            "af": 1.4245,
            "ef": 0.7465,
            "ff": 0,
            "fs": 1.4245,
            "inplace": 0,
            "mode": 0,
            "n_ctx": 512,
            "n_dims": 128,
            "ne_a": [128, 32, 2, 1],
            "v": 0,
        },
        source_path="tests/kernels/data/llamacpp_test.yaml",
        source_group_index=0,
        source_case_index=152,
    )

    rope_routes = list(routes_for_op(catalog, "ROPE"))

    resolved_route, shape, reason, detail = resolve_route_for_case(case, rope_routes)

    assert reason is None
    assert detail is None
    assert resolved_route is not None
    assert resolved_route.id == "rope_f32_normal_n128_h32_t2_contiguous_4d"
    assert shape == {
        "d0": 128,
        "d1": 32,
        "d2": 2,
        "d3": 1,
        "src1_d0": 1,
        "src1_d1": 1,
        "rope.ncols": 128,
        "rope.n_dims": 128,
        "rope.nheads": 32,
        "rope.ntokens": 2,
        "rope.src0_head_stride": 128,
        "rope.src0_token_stride": 4096,
        "rope.dst_head_stride": 128,
        "rope.dst_token_stride": 4096,
        "rope.pos_token_stride": 1,
    }


def test_v2_rope_view_case_stays_unmapped() -> None:
    catalog = load_route_catalog(ACTUAL_V2_ROUTING_DIR)
    case = ImportedCase(
        op="ROPE",
        dtype={"type": "f32"},
        raw_case={},
        normalized_params={
            "af": 1.0,
            "ef": 0.0,
            "ff": 0,
            "fs": 1.0,
            "inplace": 0,
            "mode": 0,
            "n_ctx": 512,
            "n_dims": 128,
            "ne_a": [128, 32, 2, 1],
            "v": 1,
        },
        source_path="tests/kernels/data/llamacpp_test.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    rope_routes = list(routes_for_op(catalog, "ROPE"))

    resolved_route, shape, reason, detail = resolve_route_for_case(case, rope_routes)

    assert resolved_route is None
    assert shape is None
    assert reason is not None
    assert reason.value == "shape_lowering_not_implemented"
    assert detail == "ROPE v2 routing requires contiguous input (v=0)"


def test_v2_rope_neox_route_resolves_for_plain_f32_case() -> None:
    catalog = load_route_catalog(ACTUAL_V2_ROUTING_DIR)
    case = ImportedCase(
        op="ROPE",
        dtype={"type": "f32"},
        raw_case={},
        normalized_params={
            "af": 1.0,
            "ef": 0.0,
            "ff": 0,
            "fs": 1.0,
            "inplace": 0,
            "mode": 2,
            "n_ctx": 512,
            "n_dims": 64,
            "ne_a": [64, 128, 2, 1],
            "v": 0,
        },
        source_path="tests/kernels/data/llamacpp_test.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    rope_routes = list(routes_for_op(catalog, "ROPE"))

    resolved_route, shape, reason, detail = resolve_route_for_case(case, rope_routes)

    assert reason is None
    assert detail is None
    assert resolved_route is not None
    assert resolved_route.id == "rope_neox_f32_n64_h128_t2_contiguous_4d"
    assert shape == {
        "d0": 64,
        "d1": 128,
        "d2": 2,
        "d3": 1,
        "src1_d0": 1,
        "src1_d1": 1,
        "rope.ncols": 64,
        "rope.n_dims": 64,
        "rope.nheads": 128,
        "rope.ntokens": 2,
        "rope.src0_head_stride": 64,
        "rope.src0_token_stride": 8192,
        "rope.dst_head_stride": 64,
        "rope.dst_token_stride": 8192,
        "rope.pos_token_stride": 1,
    }


def test_v2_rope_neox_partial_dims_case_stays_unmapped() -> None:
    catalog = load_route_catalog(ACTUAL_V2_ROUTING_DIR)
    case = ImportedCase(
        op="ROPE",
        dtype={"type": "f32"},
        raw_case={},
        normalized_params={
            "af": 1.0,
            "ef": 0.0,
            "ff": 0,
            "fs": 1.0,
            "inplace": 0,
            "mode": 2,
            "n_ctx": 512,
            "n_dims": 32,
            "ne_a": [80, 32, 2, 1],
            "v": 0,
        },
        source_path="tests/kernels/data/llamacpp_test.yaml",
        source_group_index=0,
        source_case_index=20,
    )

    rope_routes = [
        route
        for route in routes_for_op(catalog, "ROPE")
        if route.id == "rope_neox_f32_n64_h128_t2_contiguous_4d"
    ]

    resolved_route, shape, reason, detail = resolve_route_for_case(case, rope_routes)

    assert resolved_route is None
    assert shape is None
    assert reason is not None
    assert reason.value == "shape_lowering_not_implemented"
    assert detail == "ROPE NEOX v2 routing currently requires n_dims == ne_a[0]"


def test_v2_rope_set_rows_route_resolves_for_f16_mode0_case() -> None:
    catalog = load_route_catalog(ACTUAL_V2_ROUTING_DIR)
    case = ImportedCase(
        op="ROPE_SET_ROWS",
        dtype={"type": "f16", "type_idx": "i64"},
        raw_case={},
        normalized_params={
            "mode": 0,
            "ne_a": [128, 32, 1, 1],
        },
        source_path="tests/kernels/data/llamacpp_test.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    routes = list(routes_for_op(catalog, "ROPE_SET_ROWS"))
    resolved_route, shape, reason, detail = resolve_route_for_case(case, routes)

    assert reason is None
    assert detail is None
    assert resolved_route is not None
    assert resolved_route.id == "rope_set_rows_f16_normal_n128_h32_t1_contiguous_4d"
    assert shape is not None
    assert shape["d0"] == 4096
    assert shape["d1"] == 4
    assert shape["d2"] == 1
    assert shape["d3"] == 1
    assert shape["src0_d0"] == 128
    assert shape["src0_d1"] == 32
    assert shape["rope.ncols"] == 128
    assert shape["rope.n_dims"] == 128
    assert shape["rope.nheads"] == 32
    assert shape["rope.ntokens"] == 1
    assert shape["set_rows.ne1"] == 4


def test_v2_rope_set_rows_f32_case_stays_unmapped() -> None:
    catalog = load_route_catalog(ACTUAL_V2_ROUTING_DIR)
    case = ImportedCase(
        op="ROPE_SET_ROWS",
        dtype={"type": "f32", "type_idx": "i64"},
        raw_case={},
        normalized_params={
            "mode": 0,
            "ne_a": [128, 32, 8, 1],
        },
        source_path="tests/kernels/data/llamacpp_test.yaml",
        source_group_index=1,
        source_case_index=4,
    )

    routes = list(routes_for_op(catalog, "ROPE_SET_ROWS"))
    resolved_route, shape, reason, detail = resolve_route_for_case(case, routes)

    assert resolved_route is None
    assert shape is None
    assert reason is not None
    assert reason.value == "no_dtype_mapping"
    assert detail == "matching v2 op mapping exists, but not for this dtype combination"


def test_v2_rope_set_rows_batch_case_stays_unmapped() -> None:
    catalog = load_route_catalog(ACTUAL_V2_ROUTING_DIR)
    case = ImportedCase(
        op="ROPE_SET_ROWS",
        dtype={"type": "f16", "type_idx": "i64"},
        raw_case={},
        normalized_params={
            "mode": 0,
            "ne_a": [128, 32, 8, 3],
        },
        source_path="tests/kernels/data/llamacpp_test.yaml",
        source_group_index=0,
        source_case_index=5,
    )

    routes = list(routes_for_op(catalog, "ROPE_SET_ROWS"))
    resolved_route, shape, reason, detail = resolve_route_for_case(case, routes)

    assert resolved_route is None
    assert shape is None
    assert reason is not None
    assert reason.value == "shape_lowering_not_implemented"
    assert detail == "ROPE_SET_ROWS v2 routing currently requires ne_a[3]=1"


def test_v2_rope_set_rows_multi_token_case_stays_unmapped() -> None:
    catalog = load_route_catalog(ACTUAL_V2_ROUTING_DIR)
    case = ImportedCase(
        op="ROPE_SET_ROWS",
        dtype={"type": "f16", "type_idx": "i64"},
        raw_case={},
        normalized_params={
            "mode": 0,
            "ne_a": [128, 32, 8, 1],
        },
        source_path="tests/kernels/data/llamacpp_test.yaml",
        source_group_index=0,
        source_case_index=4,
    )

    routes = list(routes_for_op(catalog, "ROPE_SET_ROWS"))
    resolved_route, shape, reason, detail = resolve_route_for_case(case, routes)

    assert resolved_route is None
    assert shape is None
    assert reason is not None
    assert reason.value == "no_route_match"


def test_v2_soft_max_route_resolves_for_plain_f32_case() -> None:
    catalog = load_route_catalog(ACTUAL_V2_ROUTING_DIR)
    case = ImportedCase(
        op="SOFT_MAX",
        dtype={"type": "f32"},
        raw_case={},
        normalized_params={
            "inplace": 0,
            "m_prec": "f32",
            "mask": 0,
            "max_bias": 0.0,
            "ne": [16, 16, 1, 1],
            "nr23": [1, 1],
            "scale": 1.0,
            "sinks": 0,
        },
        source_path="tests/kernels/data/llamacpp_test.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    soft_max_routes = list(routes_for_op(catalog, "SOFT_MAX"))

    resolved_route, shape, reason, detail = resolve_route_for_case(case, soft_max_routes)

    assert reason is None
    assert detail is None
    assert resolved_route is not None
    assert resolved_route.id == "soft_max_f32_contiguous_2d"
    assert shape == {"d0": 16, "d1": 16}


def test_v2_soft_max_masked_case_stays_unmapped() -> None:
    catalog = load_route_catalog(ACTUAL_V2_ROUTING_DIR)
    case = ImportedCase(
        op="SOFT_MAX",
        dtype={"type": "f32"},
        raw_case={},
        normalized_params={
            "inplace": 0,
            "m_prec": "f16",
            "mask": 1,
            "max_bias": 0.0,
            "ne": [1024, 16, 1, 1],
            "nr23": [1, 1],
            "scale": 1.0,
            "sinks": 0,
        },
        source_path="tests/kernels/data/llamacpp_test.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    soft_max_routes = list(routes_for_op(catalog, "SOFT_MAX"))

    resolved_route, shape, reason, detail = resolve_route_for_case(case, soft_max_routes)

    assert resolved_route is None
    assert shape is None
    assert reason is not None
    assert reason.value == "shape_lowering_not_implemented"
    assert detail == "SOFT_MAX v2 routing currently requires mask=0"


def test_v2_soft_max_large_ncols_case_stays_unmapped() -> None:
    catalog = load_route_catalog(ACTUAL_V2_ROUTING_DIR)
    case = ImportedCase(
        op="SOFT_MAX",
        dtype={"type": "f32"},
        raw_case={},
        normalized_params={
            "inplace": 0,
            "m_prec": "f32",
            "mask": 0,
            "max_bias": 0.0,
            "ne": [200000, 1, 1, 1],
            "nr23": [1, 1],
            "scale": 1.0,
            "sinks": 0,
        },
        source_path="tests/kernels/data/llamacpp_test.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    soft_max_routes = list(routes_for_op(catalog, "SOFT_MAX"))

    resolved_route, shape, reason, detail = resolve_route_for_case(case, soft_max_routes)

    assert resolved_route is None
    assert shape is None
    assert reason is not None
    assert reason.value == "shape_lowering_not_implemented"
    assert detail == "SOFT_MAX v2 routing currently requires ne[0] <= 1024"
