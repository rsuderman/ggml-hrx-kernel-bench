from __future__ import annotations

import json
from pathlib import Path

import pytest

import ggml_hrx_kernel_bench.routing.v2.backend as v2_backend
from ggml_hrx_kernel_bench.generators.copy import (
    render_catalog_artifacts,
    render_kernel_artifacts,
)
from ggml_hrx_kernel_bench.import_models import (
    ImportedCase,
    ImportedOpGroup,
    ImportedSuite,
    ResolvedBenchmarkCase,
    UnmappedCase,
)
from ggml_hrx_kernel_bench.routing.api import (
    CandidateQuery,
    RoutingBackend,
    RuntimeCaseRequest,
    create_router,
)
from ggml_hrx_kernel_bench.yaml_route_import import materialize_yaml_route_import
from ggml_hrx_kernel_bench.import_route_resolution import resolve_case_routes
from ggml_hrx_kernel_bench.routing.v2.import_resolution import (
    resolve_imported_suite,
    resolve_route_for_case,
)
from ggml_hrx_kernel_bench.routing.v2.candidates import candidate_from_shape
from ggml_hrx_kernel_bench.routing.v1.routes import DEFAULT_V1_ROUTING_DIR, iter_routes
from ggml_hrx_kernel_bench.routing.v2.models import (
    ConstraintCheck,
    RouteConstraints,
    TensorDescriptor,
    V2Route,
    ValueDefinition,
)
from ggml_hrx_kernel_bench.routing.v2.manifest import build_manifest
from ggml_hrx_kernel_bench.routing.v2.matching import materialize_route_tensors, route_accepts_tensors
from ggml_hrx_kernel_bench.routing.v2.query import RouteCatalog, load_route_catalog, routes_for_op
from ggml_hrx_kernel_bench.routing.v2.selection import (
    PythonRouteSelector,
    RouteMatch,
    RouteMatchQuery,
)


# Unary + copy families are code-generated into the materialized asset tree (not the source catalog),
# so resolve routes against a once-materialized catalog rather than catalog/v2 on disk.
import tempfile as _tempfile

from ggml_hrx_kernel_bench.materialized_assets import materialize_asset_root as _materialize_asset_root

_MATERIALIZED_V2_ASSETS = _materialize_asset_root(
    Path(_tempfile.mkdtemp(prefix="hrx-v2-routing-assets-")) / "assets", force=True
)
ACTUAL_V2_ROUTING_DIR = _MATERIALIZED_V2_ASSETS / "catalog" / "v2"
ACTUAL_V2_KERNEL_DIR = _MATERIALIZED_V2_ASSETS / "kernels" / "v2"


def _python_router(*, kernel_dir: Path, routing_dir: Path) -> RoutingBackend:
    return create_router(
        version="v2",
        kernel_dir=kernel_dir,
        routing_dir=routing_dir,
        v2_selector_mode="python",
    )


ACTUAL_V2_ROUTER = _python_router(
    kernel_dir=ACTUAL_V2_KERNEL_DIR,
    routing_dir=ACTUAL_V2_ROUTING_DIR,
)
SOURCE_V2_ROUTER = _python_router(
    kernel_dir=Path("kernels/v2"),
    routing_dir=Path("catalog/v2"),
)


def _resolve_one_case(
    router: RoutingBackend,
    case: ImportedCase,
) -> ResolvedBenchmarkCase | UnmappedCase:
    suite = ImportedSuite(
        schema="test",
        source_path=case.source_path,
        op_groups=[
            ImportedOpGroup(
                op=case.op,
                dtype=case.dtype,
                source_path=case.source_path,
                cases=(case,),
            )
        ],
    )

    result = router.resolve_imported_suite(suite)

    rows: list[ResolvedBenchmarkCase | UnmappedCase] = [*result.resolved, *result.unmapped]
    assert len(rows) == 1
    return rows[0]


def _resolved_shape(result: ResolvedBenchmarkCase) -> dict[str, int]:
    return dict(zip(result.params, result.values, strict=True))


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


def test_v2_resolve_copy_route_for_contiguous_case(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernels"
    routing_dir = tmp_path / "routing"
    _write_copy_kernel(kernel_dir)
    _write_v2_copy_descriptor(routing_dir)
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
        source_path="tests/kernels/data/llamacpp_test.v2.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    result = _resolve_one_case(
        _python_router(kernel_dir=kernel_dir, routing_dir=routing_dir),
        case,
    )

    assert isinstance(result, ResolvedBenchmarkCase)
    assert result.route_id == "copy_f32_f16_contiguous_1d"
    assert _resolved_shape(result) == {"d0": 16, "d1": 4, "d2": 2, "d3": 2}


def test_v2_resolve_copy_route_for_transposed_f32_case(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernels"
    routing_dir = tmp_path / "routing"
    _write_copy_kernel(kernel_dir)
    _write_v2_copy_descriptor(routing_dir)
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
        source_path="tests/kernels/data/llamacpp_test.v2.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    result = _resolve_one_case(
        _python_router(kernel_dir=kernel_dir, routing_dir=routing_dir),
        case,
    )

    assert isinstance(result, ResolvedBenchmarkCase)
    assert result.route_id == "copy_f32_f32_non_contiguous_4d"
    assert _resolved_shape(result) == {
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
        source_path="tests/kernels/data/llamacpp_test.v2.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    result = _resolve_one_case(
        _python_router(kernel_dir=kernel_dir, routing_dir=routing_dir),
        case,
    )

    assert isinstance(result, ResolvedBenchmarkCase)
    assert result.route_id == "copy_f32_f32_non_contiguous_4d"
    assert _resolved_shape(result) == {
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


def test_v2_resolve_cont_route_for_contiguous_f32_case() -> None:
    case = ImportedCase(
        op="CONT",
        dtype={"type": "f32"},
        raw_case={},
        normalized_params={
            "ne": [2, 3, 5, 7],
            "use_view_slice": 0,
        },
        source_path="tests/kernels/data/llamacpp_test.v2.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, ResolvedBenchmarkCase)
    assert result.route_id == "cont_f32_contiguous_4d"
    assert _resolved_shape(result) == {"d0": 2, "d1": 3, "d2": 5, "d3": 7, "cont.d1": 105}


def test_v2_resolve_cont_route_for_rank2_f32_case() -> None:
    case = ImportedCase(
        op="CONT",
        dtype={"type": "f32"},
        raw_case={},
        normalized_params={
            "ne": [7, 5],
            "use_view_slice": 0,
        },
        source_path="tests/kernels/data/llamacpp_test.v2.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, ResolvedBenchmarkCase)
    assert result.route_id == "cont_f32_contiguous_4d"
    assert _resolved_shape(result) == {"d0": 7, "d1": 5, "cont.d1": 5}


def test_v2_resolve_cont_route_for_rank3_f32_case() -> None:
    case = ImportedCase(
        op="CONT",
        dtype={"type": "f32"},
        raw_case={},
        normalized_params={
            "ne": [4, 3, 2],
            "use_view_slice": 0,
        },
        source_path="tests/kernels/data/llamacpp_test.v2.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, ResolvedBenchmarkCase)
    assert result.route_id == "cont_f32_contiguous_4d"
    assert _resolved_shape(result) == {"d0": 4, "d1": 3, "d2": 2, "cont.d1": 6}


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
    case = ImportedCase(
        op=op,
        dtype={"type": dtype},
        raw_case={},
        normalized_params={
            "ne_a": [4, 3, 2, 5],
            "v": 0,
        },
        source_path="tests/kernels/data/llamacpp_test.v2.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, ResolvedBenchmarkCase)
    assert result.route_id == route_id
    assert _resolved_shape(result) == {"d0": 4, "d1": 3, "d2": 2, "d3": 5}


@pytest.mark.parametrize("op", ("EXP", "NEG", "RELU"))
def test_v2_unary_view_case_maps_to_non_contiguous_route(op: str) -> None:
    # ne_a=[4,3,2,5] v=1 -> ggml view src0 strides [1, 3*4, 6*4*3, 30*4*3*2] = [1,12,72,720].
    # The generated non-contiguous unary routes accept the strided src0 the import query emits.
    case = ImportedCase(
        op=op,
        dtype={"type": "f32"},
        raw_case={},
        normalized_params={
            "ne_a": [4, 3, 2, 5],
            "v": 1,
        },
        source_path="tests/kernels/data/llamacpp_test.v2.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, ResolvedBenchmarkCase)
    assert result.route_id == f"{op.lower()}_f32_non_contiguous_4d"
    shape = _resolved_shape(result)
    assert shape["d0"] == 4 and shape["d1"] == 3 and shape["d2"] == 2 and shape["d3"] == 5
    assert shape["src0_d1_stride"] == 12
    assert shape["src0_d2_stride"] == 72
    assert shape["src0_d3_stride"] == 720


@pytest.mark.parametrize(
    ("dtype", "route_id"),
    (
        ("f16", "abs_f16_contiguous_4d"),
        ("f32", "abs_f32_contiguous_4d"),
    ),
)
def test_v2_resolve_abs_contiguous_case(dtype: str, route_id: str) -> None:
    case = ImportedCase(
        op="ABS",
        dtype={"type": dtype},
        raw_case={},
        normalized_params={"ne_a": [5, 7, 11, 13], "v": 0},
        source_path="tests/kernels/data/llamacpp_test.v2.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, ResolvedBenchmarkCase)
    assert result.route_id == route_id
    assert _resolved_shape(result) == {"d0": 5, "d1": 7, "d2": 11, "d3": 13}


@pytest.mark.parametrize(
    ("dtype", "route_id"),
    (
        ("f16", "abs_f16_non_contiguous_4d"),
        ("f32", "abs_f32_non_contiguous_4d"),
    ),
)
def test_v2_resolve_abs_view_case_encodes_strided_src0(dtype: str, route_id: str) -> None:
    # ggml test_unary v=1 view: parent inflated [3, 2, 5, 4] per dim; the ne_a view
    # keeps element strides [1, 3*ne0, 6*ne0*ne1, 30*ne0*ne1*ne2].
    case = ImportedCase(
        op="ABS",
        dtype={"type": dtype},
        raw_case={},
        normalized_params={"ne_a": [5, 7, 11, 13], "v": 1},
        source_path="tests/kernels/data/llamacpp_test.v2.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, ResolvedBenchmarkCase)
    assert result.route_id == route_id
    shape = _resolved_shape(result)
    assert shape["d0"] == 5 and shape["d1"] == 7 and shape["d2"] == 11 and shape["d3"] == 13
    assert shape["src0_d1_stride"] == 15
    assert shape["src0_d2_stride"] == 210
    assert shape["src0_d3_stride"] == 11550


def test_v2_abs_view_shape_round_trips_through_materialize() -> None:
    # The serialized shape must rehydrate into the same strided src0 and still be accepted
    # by the route (the runtime hydration + validation path).
    catalog = load_route_catalog(ACTUAL_V2_ROUTING_DIR)
    case = ImportedCase(
        op="ABS",
        dtype={"type": "f16"},
        raw_case={},
        normalized_params={"ne_a": [5, 7, 11, 13], "v": 1},
        source_path="tests/kernels/data/llamacpp_test.v2.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)
    assert isinstance(result, ResolvedBenchmarkCase)
    assert result.route_id is not None
    route = catalog.routes_by_id[result.route_id]
    shape = _resolved_shape(result)

    tensors = materialize_route_tensors(route, shape)
    assert route_accepts_tensors(route, tensors) is True
    src0_strides = tuple(dim.stride for dim in tensors["src0"].dimensions)
    assert src0_strides == (1, 15, 210, 11550)
    dst_strides = tuple(dim.stride for dim in tensors["dst"].dimensions)
    assert dst_strides == (1, 5, 35, 385)


def test_v2_resolve_scale_route_for_f32_case() -> None:
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
        source_path="tests/kernels/data/llamacpp_test.v2.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, ResolvedBenchmarkCase)
    assert result.route_id == "scale_f32_contiguous_4d"
    assert _resolved_shape(result) == {
        "d0": 10,
        "d1": 10,
        "d2": 10,
        "d3": 10,
        "pointwise.d1": 1000,
    }


def test_v2_resolve_scale_route_for_rank2_f32_case() -> None:
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
        source_path="tests/kernels/data/llamacpp_test.v2.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, ResolvedBenchmarkCase)
    assert result.route_id == "scale_f32_contiguous_4d"
    assert _resolved_shape(result) == {"d0": 7, "d1": 5, "pointwise.d1": 5}


def test_v2_resolve_scale_route_for_rank3_f32_case() -> None:
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
        source_path="tests/kernels/data/llamacpp_test.v2.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, ResolvedBenchmarkCase)
    assert result.route_id == "scale_f32_contiguous_4d"
    assert _resolved_shape(result) == {"d0": 4, "d1": 3, "d2": 2, "pointwise.d1": 6}


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
    case = ImportedCase(
        op=op,
        dtype={"type": dtype},
        raw_case={},
        normalized_params={
            "ne": [4, 3, 2],
        },
        source_path="tests/kernels/data/llamacpp_test.v2.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, ResolvedBenchmarkCase)
    assert result.route_id == route_id
    assert _resolved_shape(result) == {"d0": 4, "d1": 3, "d2": 2, "pointwise.d1": 6}


def test_v2_resolve_clamp_route_for_f32_case() -> None:
    case = ImportedCase(
        op="CLAMP",
        dtype={"type": "f32"},
        raw_case={},
        normalized_params={
            "ne": [7, 1, 5, 3],
            "min": -0.5,
            "max": 0.5,
        },
        source_path="tests/kernels/data/llamacpp_test.v2.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, ResolvedBenchmarkCase)
    assert result.route_id == "clamp_f32_contiguous_4d"
    assert _resolved_shape(result) == {"d0": 7, "d1": 1, "d2": 5, "d3": 3, "pointwise.d1": 15}


def test_v2_resolve_clamp_route_for_rank2_f32_case() -> None:
    case = ImportedCase(
        op="CLAMP",
        dtype={"type": "f32"},
        raw_case={},
        normalized_params={
            "ne": [7, 5],
            "min": -0.5,
            "max": 0.5,
        },
        source_path="tests/kernels/data/llamacpp_test.v2.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, ResolvedBenchmarkCase)
    assert result.route_id == "clamp_f32_contiguous_4d"
    assert _resolved_shape(result) == {"d0": 7, "d1": 5, "pointwise.d1": 5}


def test_v2_resolve_clamp_route_for_rank3_f32_case() -> None:
    case = ImportedCase(
        op="CLAMP",
        dtype={"type": "f32"},
        raw_case={},
        normalized_params={
            "ne": [4, 3, 2],
            "min": -0.5,
            "max": 0.5,
        },
        source_path="tests/kernels/data/llamacpp_test.v2.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, ResolvedBenchmarkCase)
    assert result.route_id == "clamp_f32_contiguous_4d"
    assert _resolved_shape(result) == {"d0": 4, "d1": 3, "d2": 2, "pointwise.d1": 6}


def test_v2_resolve_set_rows_route_for_f32_i64_case() -> None:
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
        source_path="tests/kernels/data/llamacpp_test.v2.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, ResolvedBenchmarkCase)
    assert result.route_id == "set_rows_f32_f32_contiguous_4d"
    assert _resolved_shape(result) == {
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
        source_path="tests/kernels/data/llamacpp_test.v2.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, UnmappedCase)
    assert result.reason.value == "no_dtype_mapping"
    assert result.detail == "matching v2 op mapping exists, but not for this dtype combination"


def test_v2_resolve_cont_set_rows_route_for_f32_i64_f16_case() -> None:
    case = ImportedCase(
        op="SET_ROWS",
        dtype={"type_src": "f32", "type_idx": "i64", "type_dst": "f16"},
        raw_case={},
        normalized_params={
            "ne": [128, 4, 1, 1],
            "nr23": [1, 1],
            "r": 2,
            "v": 0,
        },
        source_path="tests/kernels/data/additional_test.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, ResolvedBenchmarkCase)
    assert result.route_id == "cont_set_rows_f32_f16_n128_r2_dst4_contiguous_4d"
    assert _resolved_shape(result) == {
        "d0": 128,
        "d1": 4,
        "d2": 1,
        "d3": 1,
        "src0_d1": 2,
        "src1_d0": 2,
        "src1_d1": 1,
    }


def test_v2_resolve_set_rows_route_preserves_non_contiguous_idx_stride() -> None:
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
        source_path="tests/kernels/data/llamacpp_test.v2.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, ResolvedBenchmarkCase)
    assert result.route_id == "set_rows_f32_f32_contiguous_4d"
    shape = _resolved_shape(result)
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
    router = _python_router(kernel_dir=kernel_dir, routing_dir=routing_dir)
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

    resolved_route, shape, reason, detail = resolve_route_for_case(
        case,
        [route],
        selector=PythonRouteSelector((route,)),
    )

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


def test_v2_mul_route_resolves_rms_norm_mul_f32_fused_case() -> None:
    case = ImportedCase(
        op="MUL",
        dtype={"type": "f32"},
        raw_case={},
        normalized_params={
            "ne": [16, 5, 4, 3],
            "nf": 16,
            "nr": [1, 1, 1, 1],
            "perm1": [0, 1, 2, 3],
        },
        source_path="tests/kernels/data/llamacpp_test.v2.yaml",
        source_group_index=1,
        source_case_index=42,
    )
    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, ResolvedBenchmarkCase)
    assert result.route_id == "rms_norm_mul_f32_n16_r60_vector_tail"
    assert _resolved_shape(result) == {
        "d0": 16,
        "d1": 60,
        "src1_d1": 1,
        "src1_d1_stride": 0,
        "ncols": 16,
        "nrows": 60,
    }


def test_v2_model_style_add_resolves_to_generic_2d_route() -> None:
    case = ImportedCase(
        op="ADD",
        dtype={
            "type": "f32",
            "type_dst": "f32",
            "type_src": "f32",
            "type_src0": "f32",
            "type_src1": "f32",
        },
        raw_case={},
        normalized_params={
            "ne": [4096, 512, 1, 1],
            "sources": "f32[4096,512,1,1],f32[4096,512,1,1]",
        },
        source_path="tests/models/data/Llama-3.3-8B-Instruct.Q8_0.v2.yaml",
        source_group_index=0,
        source_case_index=1,
    )
    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, ResolvedBenchmarkCase)
    assert result.route_id == "add_f32_generic_2d"
    assert _resolved_shape(result) == {
        "d0": 4096,
        "d1": 512,
        "src0_d1_stride": 4096,
        "src1_d0": 4096,
        "src1_d1_stride": 4096,
    }


def test_v2_model_style_mul_broadcast_resolves_to_generic_2d_route() -> None:
    case = ImportedCase(
        op="MUL",
        dtype={
            "type": "f32",
            "type_dst": "f32",
            "type_src": "f32",
            "type_src0": "f32",
            "type_src1": "f32",
        },
        raw_case={},
        normalized_params={
            "ne": [4096, 512, 1, 1],
            "sources": "f32[4096,512,1,1],f32[4096,1,1,1]",
        },
        source_path="tests/models/data/Llama-3.3-8B-Instruct.Q8_0.v2.yaml",
        source_group_index=0,
        source_case_index=1,
    )
    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, ResolvedBenchmarkCase)
    assert result.route_id == "mul_f32_generic_2d"
    assert _resolved_shape(result) == {
        "d0": 4096,
        "d1": 512,
        "src1_d1": 1,
        "src1_d1_stride": 0,
        "src0_d1_stride": 4096,
        "src1_d0": 4096,
    }


def test_v2_add_rms_norm_route_resolves_fused_mul_case() -> None:
    case = ImportedCase(
        op="ADD_RMS_NORM",
        dtype={"type": "f32"},
        raw_case={},
        normalized_params={
            "broadcast": 0,
            "eps": 0.0,
            "ne": [64, 5, 4, 3],
        },
        source_path="tests/kernels/data/llamacpp_test.v2.yaml",
        source_group_index=0,
        source_case_index=1,
    )
    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, ResolvedBenchmarkCase)
    assert result.route_id == "add_rms_norm_mul_f32_n64_r60_vector_tail"
    assert _resolved_shape(result) == {
        "d0": 64,
        "d1": 60,
        "weight_d1": 1,
        "weight_d1_stride": 0,
        "ncols": 64,
        "nrows": 60,
    }


def test_v2_add_rms_norm_nonzero_eps_case_stays_unmapped() -> None:
    case = ImportedCase(
        op="ADD_RMS_NORM",
        dtype={"type": "f32"},
        raw_case={},
        normalized_params={
            "broadcast": 0,
            "eps": 1.0e-4,
            "ne": [64, 5, 4, 3],
        },
        source_path="tests/kernels/data/llamacpp_test.v2.yaml",
        source_group_index=0,
        source_case_index=3,
    )
    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, UnmappedCase)
    assert result.reason.value == "shape_lowering_not_implemented"
    assert result.detail == "add_rms_norm_mul_f32 routing currently requires eps=0.0"


def test_v2_quantize_route_resolves_rms_norm_mul_quantize_q8_1_x4_case() -> None:
    case = ImportedCase(
        op="QUANTIZE",
        dtype={"type_src": "f32", "type_weight": "f32", "type_dst": "q8_1_x4"},
        raw_case={},
        normalized_params={
            "eps": 0.0,
            "ne": [3072, 1, 1, 1],
        },
        source_path="tests/kernels/data/additional_test.yaml",
        source_group_index=0,
        source_case_index=0,
    )
    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, ResolvedBenchmarkCase)
    assert result.route_id == "rms_norm_mul_quantize_q8_1_f32_n3072_r1_x4_recompute"
    assert _resolved_shape(result) == {
        "d0": 3072,
        "d1": 1,
        "weight_d1_stride": 0,
        "ncols": 3072,
        "nrows": 1,
    }


def test_v2_quantize_route_resolves_standalone_q8_1_case() -> None:
    case = ImportedCase(
        op="QUANTIZE",
        dtype={"type_src": "f32", "type_dst": "q8_1"},
        raw_case={},
        normalized_params={
            "ne": [256, 1, 1, 1],
        },
        source_path="tests/kernels/data/additional_test.yaml",
        source_group_index=1,
        source_case_index=0,
    )
    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, ResolvedBenchmarkCase)
    assert result.route_id == "quantize_q8_1_f32_contiguous_n256_r1"
    assert _resolved_shape(result) == {
        "d0": 256,
        "d1": 1,
        "ncols": 256,
        "nrows": 1,
        "q8_1.blocks": 8,
        "q8_1.ne1": 1,
        "q8_1.z_count": 1,
    }


def test_v2_quantize_route_rejects_unsupported_destination() -> None:
    case = ImportedCase(
        op="QUANTIZE",
        dtype={"type_src": "f32", "type_dst": "q8_0"},
        raw_case={},
        normalized_params={
            "eps": 0.0,
            "ne": [3072, 1, 1, 1],
        },
        source_path="tests/kernels/data/additional_test.yaml",
        source_group_index=0,
        source_case_index=0,
    )
    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, UnmappedCase)
    assert result.reason.value == "no_dtype_mapping"
    assert result.detail == "matching v2 op mapping exists, but not for this dtype combination"


def test_v2_helpers_require_catalog_or_routing_dir(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="routing_dir or catalog is required"):
        build_manifest(kernel_dir=tmp_path / "kernels")
    with pytest.raises(ValueError, match="routing_dir or catalog is required"):
        resolve_imported_suite(
            ImportedSuite(schema="test", source_path="test.yaml", op_groups=[]),
            selector=PythonRouteSelector(()),
        )


def test_v2_public_import_resolution_uses_configured_selector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory_modes: list[str | None] = []
    select_calls: list[tuple[str, RouteMatchQuery]] = []

    class RecordingSelector:
        def __init__(self, catalog: RouteCatalog):
            self._delegate = PythonRouteSelector(catalog)

        def select(self, op: str, query: RouteMatchQuery) -> RouteMatch | None:
            select_calls.append((op, query))
            return self._delegate.select(op, query)

    def recording_selector_factory(
        catalog: RouteCatalog,
        *,
        mode: str | None = None,
    ) -> RecordingSelector:
        factory_modes.append(mode)
        return RecordingSelector(catalog)

    monkeypatch.setattr(v2_backend, "create_route_selector", recording_selector_factory)
    router = _python_router(
        kernel_dir=ACTUAL_V2_KERNEL_DIR,
        routing_dir=ACTUAL_V2_ROUTING_DIR,
    )
    case = ImportedCase(
        op="ABS",
        dtype={"type": "f32"},
        raw_case={},
        normalized_params={"ne_a": [5, 7, 11, 13], "v": 0},
        source_path="tests/kernels/data/llamacpp_test.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    result = _resolve_one_case(router, case)

    assert isinstance(result, ResolvedBenchmarkCase)
    assert result.route_id == "abs_f32_contiguous_4d"
    assert factory_modes == ["python"]
    assert len(select_calls) == 1
    op, query = select_calls[0]
    assert op == "ABS"
    assert set(query.tensors) == {"src0", "dst"}
    assert query.allowed_route_ids is None


def test_v2_router_maps_contiguous_and_generic_add_cases(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernels"
    routing_dir = tmp_path / "routing"
    _write_kernel(kernel_dir)
    _write_v2_descriptor(routing_dir)
    router = _python_router(kernel_dir=kernel_dir, routing_dir=routing_dir)
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
    router = _python_router(kernel_dir=kernel_dir, routing_dir=routing_dir)
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
        source_path="tests/kernels/data/llamacpp_test.v2.yaml",
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


def test_v1_copy_route_uses_role_specific_dtypes() -> None:
    case = ImportedCase(
        op="CPY",
        dtype={"type": "f16", "type_dst": "f16", "type_src": "f32"},
        raw_case={
            "_src_transpose": 0,
            "ne": [1, 2, 3, 4],
            "permute_dst": [0, 0, 0, 0],
            "permute_src": [0, 0, 0, 0],
        },
        normalized_params={
            "_src_transpose": 0,
            "ne": [1, 2, 3, 4],
            "permute_dst": [0, 0, 0, 0],
            "permute_src": [0, 0, 0, 0],
        },
        source_path="tests/kernels/data/llamacpp_test.v2.yaml",
        source_group_index=0,
        source_case_index=0,
    )
    copy_routes = [
        route for route in iter_routes(DEFAULT_V1_ROUTING_DIR) if str(route.get("op") or "") == "CPY"
    ]

    resolution, _, reason, detail = resolve_case_routes(case, copy_routes)

    assert reason is None
    assert detail is None
    assert resolution is not None
    assert resolution.route["id"] == "copy_f32_f16_generic_wg256"
    assert resolution.shape == {"ncols": 24, "nrows": 1, "cols": 24, "rows": 1}


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
        source_path="tests/kernels/data/llamacpp_test.v2.yaml",
        source_group_index=0,
        source_case_index=0,
    )
    result = _resolve_one_case(SOURCE_V2_ROUTER, case)

    assert isinstance(result, UnmappedCase)
    assert result.reason.value == "no_route_match"
    assert result.detail == "YAML import tensor query did not satisfy any v2 route"


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
        source_path="tests/kernels/data/llamacpp_test.v2.yaml",
        source_group_index=0,
        source_case_index=0,
    )
    result = _resolve_one_case(SOURCE_V2_ROUTER, case)

    assert isinstance(result, ResolvedBenchmarkCase)
    assert result.route_id == "sum_rows_f32_contiguous_4d"
    shape = _resolved_shape(result)
    assert shape["d0"] == 33
    assert shape["d1"] == 256


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
        source_path="tests/kernels/data/llamacpp_test.v2.yaml",
        source_group_index=0,
        source_case_index=0,
    )
    result = _resolve_one_case(SOURCE_V2_ROUTER, case)

    assert isinstance(result, UnmappedCase)
    assert result.reason.value == "shape_lowering_not_implemented"
    assert result.detail == "SUM_ROWS v2 routing requires permute=0"


def test_v2_rms_norm_route_resolves_for_contiguous_eps0_case() -> None:
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
        source_path="tests/kernels/data/llamacpp_test.v2.yaml",
        source_group_index=0,
        source_case_index=0,
    )
    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, ResolvedBenchmarkCase)
    assert result.route_id == "rms_norm_f32_contiguous_4d"
    assert _resolved_shape(result) == {"d0": 1025, "d1": 5, "d2": 4, "d3": 3}


def test_v2_rms_norm_nonzero_eps_case_stays_unmapped() -> None:
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
        source_path="tests/kernels/data/llamacpp_test.v2.yaml",
        source_group_index=0,
        source_case_index=0,
    )
    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, UnmappedCase)
    assert result.reason.value == "shape_lowering_not_implemented"
    assert result.detail == "RMS_NORM v2 routing currently requires eps=0.0"


def test_v2_swiglu_route_resolves_for_packed_contiguous_case() -> None:
    case = ImportedCase(
        op="SWIGLU",
        dtype={"type": "f32"},
        raw_case={},
        normalized_params={
            "ne_a": [128, 2, 2, 2],
            "swapped": 0,
            "v": 0,
        },
        source_path="tests/kernels/data/llamacpp_test.v2.yaml",
        source_group_index=0,
        source_case_index=0,
    )
    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, ResolvedBenchmarkCase)
    assert result.route_id == "swiglu_f32_packed_contiguous_4d"
    assert _resolved_shape(result) == {"d0": 128, "d1": 2, "d2": 2, "d3": 2, "src0_d0": 256}


@pytest.mark.parametrize("rows", [1, 512])
def test_v2_model_style_swiglu_split_sources_pack_into_existing_route(rows: int) -> None:
    case = ImportedCase(
        op="SWIGLU",
        dtype={
            "type": "f32",
            "type_dst": "f32",
            "type_src": "f32",
            "type_src0": "f32",
            "type_src1": "f32",
        },
        raw_case={},
        normalized_params={
            "ne": [14336, rows, 1, 1],
            "op_params": ["0:2"],
            "sources": f"f32[14336,{rows},1,1],f32[14336,{rows},1,1]",
        },
        source_path="tests/models/data/Llama-3.3-8B-Instruct.Q8_0.v2.yaml",
        source_group_index=0,
        source_case_index=0,
    )
    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, ResolvedBenchmarkCase)
    assert result.route_id == "swiglu_f32_packed_contiguous_4d"
    assert _resolved_shape(result) == {
        "d0": 14336,
        "d1": rows,
        "d2": 1,
        "d3": 1,
        "src0_d0": 28672,
    }


def test_v2_model_style_swiglu_unknown_op_params_stays_unmapped() -> None:
    case = ImportedCase(
        op="SWIGLU",
        dtype={
            "type": "f32",
            "type_dst": "f32",
            "type_src": "f32",
            "type_src0": "f32",
            "type_src1": "f32",
        },
        raw_case={},
        normalized_params={
            "ne": [14336, 1, 1, 1],
            "op_params": ["1:2"],
            "sources": "f32[14336,1,1,1],f32[14336,1,1,1]",
        },
        source_path="tests/models/data/Llama-3.3-8B-Instruct.Q8_0.v2.yaml",
        source_group_index=0,
        source_case_index=0,
    )
    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, UnmappedCase)
    assert result.reason.value == "shape_lowering_not_implemented"
    assert result.detail == "SWIGLU YAML import currently requires op_params=['0:2']"


def test_v2_swiglu_split_case_stays_unmapped_without_split_route() -> None:
    case = ImportedCase(
        op="SWIGLU",
        dtype={"type": "f32"},
        raw_case={},
        normalized_params={
            "ne_a": [128, 2, 2, 2],
            "split": True,
            "v": 0,
        },
        source_path="tests/kernels/data/llamacpp_test.v2.yaml",
        source_group_index=0,
        source_case_index=0,
    )
    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, UnmappedCase)
    assert result.reason.value == "shape_lowering_not_implemented"
    assert result.detail == "SWIGLU v2 routing currently requires packed input (split=false)"


def test_v2_swiglu_swapped_case_stays_unmapped() -> None:
    case = ImportedCase(
        op="SWIGLU",
        dtype={"type": "f32"},
        raw_case={},
        normalized_params={
            "ne_a": [5, 7, 11, 13],
            "swapped": 1,
            "v": 0,
        },
        source_path="tests/kernels/data/llamacpp_test.v2.yaml",
        source_group_index=0,
        source_case_index=0,
    )
    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, UnmappedCase)
    assert result.reason.value == "shape_lowering_not_implemented"
    assert result.detail == "SWIGLU v2 routing currently requires swapped=0"


def test_v2_get_rows_route_resolves_for_base_f32_case() -> None:
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
        source_path="tests/kernels/data/llamacpp_test.v2.yaml",
        source_group_index=0,
        source_case_index=0,
    )
    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, ResolvedBenchmarkCase)
    assert result.route_id == "get_rows_f32_embedding_rows_2d"
    assert _resolved_shape(result) == {
        "d0": 256,
        "d1": 4,
        "src0_d1": 5,
        "src1_d0": 1,
        "get_rows.src0_nrows": 5,
        "get_rows.idx_row_stride": 1,
    }


def test_v2_get_rows_route_resolves_for_q8_0_case() -> None:
    case = ImportedCase(
        op="GET_ROWS",
        dtype={
            "type": "q8_0",
            "type_dst": "q8_0",
            "type_idx": "i32",
            "type_src": "q8_0",
        },
        raw_case={},
        normalized_params={
            "be1": 1,
            "be2": 1,
            "m": 5,
            "n": 256,
            "r": 4,
            "v": 0,
        },
        source_path="tests/kernels/data/llamacpp_test.v2.yaml",
        source_group_index=0,
        source_case_index=0,
    )
    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, ResolvedBenchmarkCase)
    assert result.route_id == "get_rows_q8_0_f32_embedding_rows_2d"
    assert _resolved_shape(result) == {
        "d0": 256,
        "d1": 4,
        "src0_d1": 5,
        "src1_d0": 1,
        "get_rows.src0_nrows": 5,
        "get_rows.idx_row_stride": 1,
    }


def test_v2_model_style_get_rows_f32_uses_source_table_row_count() -> None:
    case = ImportedCase(
        op="GET_ROWS",
        dtype={
            "type": "f32",
            "type_dst": "f32",
            "type_idx": "i32",
            "type_src": "f32",
            "type_src0": "f32",
            "type_src1": "i32",
        },
        raw_case={},
        normalized_params={
            "ne": [4096, 512, 1, 1],
            "sources": "f32[4096,512,1,1],i32[512,1,1,1]",
        },
        source_path="tests/models/data/Llama-3.3-8B-Instruct.Q8_0.v2.yaml",
        source_group_index=0,
        source_case_index=1,
    )

    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, ResolvedBenchmarkCase)
    assert result.route_id == "get_rows_f32_embedding_rows_2d"
    assert _resolved_shape(result) == {
        "d0": 4096,
        "d1": 512,
        "src1_d0": 1,
        "get_rows.src0_nrows": 512,
        "get_rows.idx_row_stride": 1,
    }


def test_v2_model_style_get_rows_q8_0_preserves_embedding_table_rows() -> None:
    case = ImportedCase(
        op="GET_ROWS",
        dtype={
            "type": "q8_0",
            "type_dst": "f32",
            "type_idx": "i32",
            "type_src": "q8_0",
            "type_src0": "q8_0",
            "type_src1": "i32",
        },
        raw_case={},
        normalized_params={
            "ne": [4096, 512, 1, 1],
            "sources": "q8_0[4096,128256,1,1],i32[512,1,1,1]",
        },
        source_path="tests/models/data/Llama-3.3-8B-Instruct.Q8_0.v2.yaml",
        source_group_index=1,
        source_case_index=1,
    )

    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, ResolvedBenchmarkCase)
    assert result.route_id == "get_rows_q8_0_f32_embedding_rows_2d"
    assert _resolved_shape(result) == {
        "d0": 4096,
        "d1": 512,
        "src0_d1": 128256,
        "src1_d0": 1,
        "get_rows.src0_nrows": 128256,
        "get_rows.idx_row_stride": 1,
    }


def test_v2_get_rows_route_resolves_for_q4_k_case() -> None:
    case = ImportedCase(
        op="GET_ROWS",
        dtype={"type": "q4_K"},
        raw_case={},
        normalized_params={
            "be1": 1,
            "be2": 1,
            "m": 5,
            "n": 256,
            "r": 4,
            "v": 0,
        },
        source_path="tests/kernels/data/llamacpp_test.v2.yaml",
        source_group_index=0,
        source_case_index=0,
    )
    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, ResolvedBenchmarkCase)
    assert result.route_id == "get_rows_q4_k_f32_embedding_rows_2d"
    assert _resolved_shape(result) == {
        "d0": 256,
        "d1": 4,
        "src0_d1": 5,
        "src1_d0": 1,
        "get_rows.src0_nrows": 5,
        "get_rows.idx_row_stride": 1,
    }


def test_v2_get_rows_route_resolves_for_q5_k_case() -> None:
    case = ImportedCase(
        op="GET_ROWS",
        dtype={"type": "q5_K"},
        raw_case={},
        normalized_params={
            "be1": 1,
            "be2": 1,
            "m": 5,
            "n": 256,
            "r": 4,
            "v": 0,
        },
        source_path="tests/kernels/data/llamacpp_test.v2.yaml",
        source_group_index=0,
        source_case_index=0,
    )
    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, ResolvedBenchmarkCase)
    assert result.route_id == "get_rows_q5_k_f32_embedding_rows_2d"
    assert _resolved_shape(result) == {
        "d0": 256,
        "d1": 4,
        "src0_d1": 5,
        "src1_d0": 1,
        "get_rows.src0_nrows": 5,
        "get_rows.idx_row_stride": 1,
    }


def test_v2_get_rows_route_resolves_for_q6_k_case() -> None:
    case = ImportedCase(
        op="GET_ROWS",
        dtype={"type": "q6_K"},
        raw_case={},
        normalized_params={
            "be1": 1,
            "be2": 1,
            "m": 5,
            "n": 256,
            "r": 4,
            "v": 0,
        },
        source_path="tests/kernels/data/llamacpp_test.v2.yaml",
        source_group_index=0,
        source_case_index=0,
    )
    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, ResolvedBenchmarkCase)
    assert result.route_id == "get_rows_q6_k_f32_embedding_rows_2d"
    assert _resolved_shape(result) == {
        "d0": 256,
        "d1": 4,
        "src0_d1": 5,
        "src1_d0": 1,
        "get_rows.src0_nrows": 5,
        "get_rows.idx_row_stride": 1,
    }


def test_v2_get_rows_route_resolves_for_moe_weights_case() -> None:
    case = ImportedCase(
        op="GET_ROWS",
        dtype={"type": "f32"},
        raw_case={},
        normalized_params={
            "moe_weights": 1,
            "nexperts": 128,
            "nselected": 8,
            "ntokens": 16,
            "src0_token_stride": 128,
            "idx_token_stride": 8,
            "dst_token_stride": 8,
        },
        source_path="tests/kernels/data/additional_test.yaml",
        source_group_index=0,
        source_case_index=0,
    )
    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, ResolvedBenchmarkCase)
    assert result.route_id == "get_rows_moe_weights_f32_topk_view_2d"
    assert _resolved_shape(result) == {
        "d0": 8,
        "d1": 16,
        "src0_d0": 128,
        "get_rows_moe.nexperts": 128,
        "get_rows_moe.nselected": 8,
        "get_rows_moe.ntokens": 16,
        "get_rows_moe.src0_token_stride": 128,
        "get_rows_moe.idx_token_stride": 8,
        "get_rows_moe.dst_token_stride": 8,
    }


def test_v2_get_rows_view_case_stays_unmapped() -> None:
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
        source_path="tests/kernels/data/llamacpp_test.v2.yaml",
        source_group_index=0,
        source_case_index=0,
    )
    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, UnmappedCase)
    assert result.reason.value == "shape_lowering_not_implemented"
    assert result.detail == "GET_ROWS v2 routing requires contiguous input (v=0)"


def test_v2_get_rows_non_unit_be1_case_stays_unmapped() -> None:
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
        source_path="tests/kernels/data/llamacpp_test.v2.yaml",
        source_group_index=0,
        source_case_index=0,
    )
    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, UnmappedCase)
    assert result.reason.value == "shape_lowering_not_implemented"
    assert result.detail == "GET_ROWS v2 routing currently requires be1=1"


def test_v2_argsort_route_resolves_for_base_f32_case() -> None:
    case = ImportedCase(
        op="ARGSORT",
        dtype={"type": "f32"},
        raw_case={},
        normalized_params={
            "ne": [128, 1, 1, 1],
            "order": 0,
        },
        source_path="tests/kernels/data/llamacpp_test.v2.yaml",
        source_group_index=0,
        source_case_index=0,
    )
    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, ResolvedBenchmarkCase)
    assert result.route_id == "argsort_f32_i32_n128_r1_desc_wg128"
    assert _resolved_shape(result) == {"d0": 128, "d1": 1}


def test_v2_argsort_ascending_case_stays_unmapped() -> None:
    case = ImportedCase(
        op="ARGSORT",
        dtype={"type": "f32"},
        raw_case={},
        normalized_params={
            "ne": [128, 1, 1, 1],
            "order": 1,
        },
        source_path="tests/kernels/data/llamacpp_test.v2.yaml",
        source_group_index=0,
        source_case_index=0,
    )
    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, UnmappedCase)
    assert result.reason.value == "shape_lowering_not_implemented"
    assert result.detail == "ARGSORT v2 routing currently requires order=0"


def test_v2_argsort_non_route_shape_case_stays_unmapped() -> None:
    case = ImportedCase(
        op="ARGSORT",
        dtype={"type": "f32"},
        raw_case={},
        normalized_params={
            "ne": [127, 1, 1, 1],
            "order": 0,
        },
        source_path="tests/kernels/data/llamacpp_test.v2.yaml",
        source_group_index=0,
        source_case_index=0,
    )
    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, UnmappedCase)
    assert result.reason.value == "no_route_match"
    assert result.detail == "YAML import tensor query did not satisfy any v2 route"


def test_v2_mul_mat_route_resolves_for_small_contiguous_f32_case() -> None:
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
        source_path="tests/kernels/data/llamacpp_test.v2.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, ResolvedBenchmarkCase)
    assert result.route_id == "mul_mat_f32_f32_contiguous_small_2d"
    assert _resolved_shape(result) == {
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
        source_path="tests/kernels/data/llamacpp_test.v2.yaml",
        source_group_index=4,
        source_case_index=6,
    )

    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, ResolvedBenchmarkCase)
    assert result.route_id == "mul_mat_f32_f32_contiguous_logits_cols1_2d"
    assert _resolved_shape(result) == {
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
        source_path="tests/kernels/data/llamacpp_test.v2.yaml",
        source_group_index=2,
        source_case_index=6,
    )

    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, ResolvedBenchmarkCase)
    assert result.route_id == "mul_mat_f16_f32_batched_logits_cols1_2d"
    shape = _resolved_shape(result)
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
        route=catalog.routes_by_id[result.route_id],
        shape=shape,
    )
    assert candidate.config["@hrx2.shape.mul_mat_f16.k"] == "1057"
    assert candidate.config["@hrx2.shape.mul_mat_f16.rows"] == "129"
    assert candidate.config["@hrx2.shape.mul_mat_f16.cols"] == "1"
    assert candidate.config["@hrx2.shape.mul_mat_f16.src0_stride_row"] == "1057"
    assert candidate.config["@hrx2.shape.mul_mat_f16.src1_stride_col"] == "1057"
    assert candidate.config["@hrx2.shape.mul_mat_f16.dst_stride_col"] == "129"
    assert candidate.config["@hrx2.tuning.mul_mat_f16.workgroup_size"] == "256"


def test_v2_mul_mat_route_resolves_for_batched_logits_cols1_f16_case() -> None:
    catalog = load_route_catalog(ACTUAL_V2_ROUTING_DIR)
    case = ImportedCase(
        op="MUL_MAT",
        dtype={"type_a": "f16", "type_b": "f32"},
        raw_case={},
        normalized_params={
            "bs": [2, 3],
            "k": 1056,
            "k_v": 2112,
            "m": 128,
            "n": 1,
            "nr": [1, 1],
            "o": 1,
            "per": [0, 1, 2, 3],
        },
        source_path="tests/kernels/data/llamacpp_test.v2.yaml",
        source_group_index=2,
        source_case_index=40,
    )

    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, ResolvedBenchmarkCase)
    assert result.route_id == "mul_mat_f16_f32_batched_logits_cols1_4d"
    shape = _resolved_shape(result)
    assert shape == {
        "d0": 128,
        "d1": 1,
        "d2": 2,
        "d3": 3,
        "dst_d2_stride": 128,
        "dst_d3_stride": 256,
        "src0_d0": 1056,
        "src0_d1": 128,
        "src0_d2": 1,
        "src0_d3": 1,
        "src1_d0": 1056,
        "src1_d2_stride": 1056,
        "src1_d3_stride": 2112,
        "k": 1056,
        "rows": 128,
        "cols": 1,
    }

    candidate = candidate_from_shape(
        kernel_dir=ACTUAL_V2_KERNEL_DIR,
        route=catalog.routes_by_id[result.route_id],
        shape=shape,
    )
    assert candidate.config["@hrx2.shape.mul_mat_f16.dst_ne2"] == "2"
    assert candidate.config["@hrx2.shape.mul_mat_f16.dst_ne3"] == "3"
    assert candidate.config["@hrx2.shape.mul_mat_f16.src1_stride_col"] == "1056"
    assert candidate.config["@hrx2.shape.mul_mat_f16.src1_stride_ne2"] == "1056"
    assert candidate.config["@hrx2.shape.mul_mat_f16.src1_stride_ne3"] == "2112"
    assert candidate.config["@hrx2.shape.mul_mat_f16.dst_stride_col"] == "128"
    assert candidate.config["@hrx2.shape.mul_mat_f16.dst_stride_ne2"] == "128"
    assert candidate.config["@hrx2.shape.mul_mat_f16.dst_stride_ne3"] == "256"


def test_v2_mul_mat_route_resolves_for_q8_0_case() -> None:
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
        source_path="tests/kernels/data/llamacpp_test.v2.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, ResolvedBenchmarkCase)
    assert result.route_id == "mul_mat_q8_0_f32_contiguous_2d"
    assert _resolved_shape(result) == {
        "d0": 1,
        "d1": 64,
        "src0_d0": 256,
        "src0_d1": 1,
        "src1_d0": 256,
        "k": 256,
        "rows": 1,
        "cols": 64,
    }


@pytest.mark.parametrize(
    ("name", "ne", "sources", "expected_shape"),
    (
        (
            "Qcur-0",
            [4096, 1, 1, 1],
            "q8_0[4096,4096,1,1],f32[4096,1,1,1]",
            {
                "d0": 4096,
                "d1": 1,
                "src0_d0": 4096,
                "src0_d1": 4096,
                "src1_d0": 4096,
                "k": 4096,
                "rows": 4096,
                "cols": 1,
            },
        ),
        (
            "ffn_gate-0",
            [14336, 512, 1, 1],
            "q8_0[4096,14336,1,1],f32[4096,512,1,1]",
            {
                "d0": 14336,
                "d1": 512,
                "src0_d0": 4096,
                "src0_d1": 14336,
                "src1_d0": 4096,
                "k": 4096,
                "rows": 14336,
                "cols": 512,
            },
        ),
    ),
)
def test_v2_model_style_mul_mat_q8_0_resolves_to_contiguous_2d_route(
    name: str,
    ne: list[int],
    sources: str,
    expected_shape: dict[str, int],
) -> None:
    case = ImportedCase(
        op="MUL_MAT",
        dtype={
            "type": "f32",
            "type_a": "q8_0",
            "type_b": "f32",
            "type_dst": "f32",
            "type_src": "q8_0",
            "type_src0": "q8_0",
            "type_src1": "f32",
        },
        raw_case={},
        normalized_params={
            "name": name,
            "ne": ne,
            "op_params": [],
            "sources": sources,
        },
        source_path="tests/models/data/Llama-3.3-8B-Instruct.Q8_0.v2.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, ResolvedBenchmarkCase)
    assert result.route_id == "mul_mat_q8_0_f32_contiguous_2d"
    assert _resolved_shape(result) == expected_shape


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
    assert plain_shape["ncols"] == 16
    assert plain_shape["nrows"] == 64
    assert masked_shape["ncols"] == 16
    assert masked_shape["nrows"] == 64


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
    for raw_path in summary["generated_config_paths"]:
        config = json.loads(Path(raw_path).read_text())
        route_shapes.setdefault(config["route_id"], []).extend(
            dict(zip(config["params"], case_values, strict=True)) for case_values in config["cases"]
        )
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


def test_v2_mul_mat_route_resolves_for_q4_k_direct_case() -> None:
    case = ImportedCase(
        op="MUL_MAT",
        dtype={"type_a": "q4_k", "type_b": "f32"},
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
        source_path="tests/kernels/data/llamacpp_test.v2.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, ResolvedBenchmarkCase)
    assert result.route_id == "mul_mat_q4_k_f32_direct_contiguous_2d"
    assert _resolved_shape(result) == {
        "d0": 16,
        "d1": 8,
        "src0_d0": 256,
        "src0_d1": 16,
        "src1_d0": 256,
        "k": 256,
        "rows": 16,
        "cols": 8,
    }


def test_v2_mul_mat_id_route_resolves_for_q4_k_case() -> None:
    case = ImportedCase(
        op="MUL_MAT_ID",
        dtype={"type_a": "q4_K", "type_b": "f32"},
        raw_case={},
        normalized_params={
            "b": 0,
            "k": 256,
            "m": 512,
            "n": 1,
            "n_mats": 4,
            "n_used": 2,
        },
        source_path="tests/kernels/data/llamacpp_test.v2.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, ResolvedBenchmarkCase)
    assert result.route_id == "mul_mat_id_q4_k_f32_expert_planes_3d"
    assert _resolved_shape(result) == {
        "d0": 512,
        "d1": 2,
        "d2": 1,
        "src0_d0": 256,
        "src0_d1": 512,
        "src0_d2": 4,
        "src1_d0": 256,
        "src2_d0": 2,
        "src2_d1": 1,
        "k": 256,
        "rows": 512,
        "nexperts": 4,
        "nselected": 2,
        "ntokens": 1,
        "src1_selected_stride": 256,
        "src1_token_stride": 512,
        "idx_token_stride": 2,
        "dst_token_stride": 1024,
    }


def test_v2_mul_mat_id_route_resolves_for_q5_k_case() -> None:
    case = ImportedCase(
        op="MUL_MAT_ID",
        dtype={"type_a": "q5_K", "type_b": "f32"},
        raw_case={},
        normalized_params={
            "b": 0,
            "k": 256,
            "m": 512,
            "n": 1,
            "n_mats": 4,
            "n_used": 2,
        },
        source_path="tests/kernels/data/llamacpp_test.v2.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, ResolvedBenchmarkCase)
    assert result.route_id == "mul_mat_id_q5_k_f32_expert_planes_3d"
    assert _resolved_shape(result) == {
        "d0": 512,
        "d1": 2,
        "d2": 1,
        "src0_d0": 256,
        "src0_d1": 512,
        "src0_d2": 4,
        "src1_d0": 256,
        "src2_d0": 2,
        "src2_d1": 1,
        "k": 256,
        "rows": 512,
        "nexperts": 4,
        "nselected": 2,
        "ntokens": 1,
        "src1_selected_stride": 256,
        "src1_token_stride": 512,
        "idx_token_stride": 2,
        "dst_token_stride": 1024,
    }


def test_v2_mul_mat_id_route_resolves_for_q6_k_case() -> None:
    case = ImportedCase(
        op="MUL_MAT_ID",
        dtype={"type_a": "q6_K", "type_b": "f32"},
        raw_case={},
        normalized_params={
            "b": 0,
            "k": 256,
            "m": 512,
            "n": 1,
            "n_mats": 4,
            "n_used": 2,
        },
        source_path="tests/kernels/data/llamacpp_test.v2.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, ResolvedBenchmarkCase)
    assert result.route_id == "mul_mat_id_q6_k_f32_expert_planes_3d"
    assert _resolved_shape(result) == {
        "d0": 512,
        "d1": 2,
        "d2": 1,
        "src0_d0": 256,
        "src0_d1": 512,
        "src0_d2": 4,
        "src1_d0": 256,
        "src2_d0": 2,
        "src2_d1": 1,
        "k": 256,
        "rows": 512,
        "nexperts": 4,
        "nselected": 2,
        "ntokens": 1,
        "src1_selected_stride": 256,
        "src1_token_stride": 512,
        "idx_token_stride": 2,
        "dst_token_stride": 1024,
    }


def test_v2_mul_mat_route_resolves_for_q5_dot16_case() -> None:
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
        source_path="tests/kernels/data/llamacpp_test.v2.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, ResolvedBenchmarkCase)
    assert result.route_id == "mul_mat_q5_k_f32_dot16_contiguous_cols1_2d"
    assert _resolved_shape(result) == {
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
        source_path="tests/kernels/data/llamacpp_test.v2.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, ResolvedBenchmarkCase)
    assert result.route_id == "mul_mat_q6_k_f32_rows2_contiguous_cols1_2d"
    assert _resolved_shape(result) == {
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
        source_path="tests/kernels/data/llamacpp_test.v2.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, ResolvedBenchmarkCase)
    assert result.route_id == "mul_mat_q6_k_f32_direct_contiguous_2d"
    assert _resolved_shape(result) == {
        "d0": 16,
        "d1": 8,
        "src0_d0": 256,
        "src0_d1": 16,
        "src1_d0": 256,
        "k": 256,
        "rows": 16,
        "cols": 8,
    }


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


@pytest.mark.parametrize(
    ("route_id", "kv"),
    (
        ("softmax_kqv_f32_f16_decode_kv256_d128_h24_hkv8_wg64_rows2", 256),
        ("softmax_kqv_f32_f16_decode_kv512_d128_h24_hkv8_wg256_rows128", 512),
        ("softmax_kqv_f32_f16_decode_kv768_d128_h24_hkv8_wg256_rows128", 768),
    ),
)
def test_v2_flash_attn_ext_fixed_decode_routes_resolve_grouped_yaml_cases(
    route_id: str,
    kv: int,
) -> None:
    case = ImportedCase(
        op="FLASH_ATTN_EXT",
        dtype={"type_KV": "f16"},
        raw_case={},
        normalized_params={
            "hsk": 128,
            "hsv": 128,
            "kv": kv,
            "logit_softcap": 0.0,
            "mask": 1,
            "max_bias": 0.0,
            "nb": 1,
            "nh": 24,
            "nr23": [4, 1],
            "permute": [0, 1, 2, 3],
            "prec": "f32",
            "sinks": 0,
        },
        source_path="tests/kernels/data/additional_test.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, ResolvedBenchmarkCase)
    assert result.route_id == route_id
    assert _resolved_shape(result) == {
        "d0": 24,
        "d1": 128,
        "src0_d0": kv,
        "src0_d1": 24,
        "mask_d0": kv,
        "mask_d1": 1,
        "src1_d0": kv,
        "src1_d1": 1024,
        "dst_d0": 128,
        "dst_d1": 24,
        "k": kv,
        "rows": 128,
        "cols": 24,
        "nheads_kv": 8,
    }


def test_v2_flash_attn_ext_route_leaves_maskless_grouped_yaml_case_unmapped() -> None:
    case = ImportedCase(
        op="FLASH_ATTN_EXT",
        dtype={"type_KV": "f16"},
        raw_case={},
        normalized_params={
            "hsk": 128,
            "hsv": 128,
            "kv": 4096,
            "nb": 1,
            "nh": 8,
            "nr23": [4, 1],
            "permute": [0, 1, 2, 3],
            "mask": 0,
            "sinks": 0,
        },
        source_path="tests/kernels/data/llamacpp_test.v2.yaml",
        source_group_index=1,
        source_case_index=3,
    )

    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, UnmappedCase)
    assert result.reason.value == "shape_lowering_not_implemented"
    assert result.detail == "FLASH_ATTN_EXT v2 routing currently requires mask=1"


@pytest.mark.parametrize("kv", (512, 1024, 2048, 4096))
def test_v2_flash_attn_ext_masked_identity_route_resolves_grouped_yaml_family(kv: int) -> None:
    case = ImportedCase(
        op="FLASH_ATTN_EXT",
        dtype={"type_KV": "f16"},
        raw_case={},
        normalized_params={
            "hsk": 128,
            "hsv": 128,
            "kv": kv,
            "logit_softcap": 0.0,
            "mask": 1,
            "max_bias": 0.0,
            "nb": 1,
            "nh": 8,
            "nr23": [4, 1],
            "permute": [0, 1, 2, 3],
            "prec": "f32",
            "sinks": 0,
        },
        source_path="tests/kernels/data/llamacpp_test.v2.yaml",
        source_group_index=1,
        source_case_index=0,
    )

    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, ResolvedBenchmarkCase)
    assert result.route_id == "softmax_kqv_f32_f16_masked_identity_kv512_4096_d128_h8_wg256_row1"
    assert _resolved_shape(result) == {
        "d0": 8,
        "d1": 128,
        "src0_d0": kv,
        "src0_d1": 8,
        "mask_d0": kv,
        "mask_d1": 1,
        "src1_d0": kv,
        "src1_d1": 1024,
        "dst_d0": 128,
        "dst_d1": 8,
        "k": kv,
        "rows": 128,
        "cols": 8,
        "nheads_kv": 8,
    }


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


def test_v2_mul_mat_broadcast_case_stays_unmapped() -> None:
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
        source_path="tests/kernels/data/llamacpp_test.v2.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, UnmappedCase)
    assert result.reason.value == "shape_lowering_not_implemented"
    assert result.detail == "MUL_MAT v2 routing currently requires nr=[1, 1]"


def test_v2_mul_mat_permuted_case_stays_unmapped() -> None:
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
        source_path="tests/kernels/data/llamacpp_test.v2.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, UnmappedCase)
    assert result.reason.value == "shape_lowering_not_implemented"
    assert result.detail == "MUL_MAT v2 routing currently requires per=[0, 1, 2, 3]"


def test_v2_rope_route_resolves_for_plain_f32_case() -> None:
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
        source_path="tests/kernels/data/llamacpp_test.v2.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, ResolvedBenchmarkCase)
    assert result.route_id == "rope_f32_normal_n128_h32_t2_contiguous_4d"
    assert _resolved_shape(result) == {
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
        source_path="tests/kernels/data/llamacpp_test.v2.yaml",
        source_group_index=0,
        source_case_index=152,
    )

    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, ResolvedBenchmarkCase)
    assert result.route_id == "rope_f32_normal_n128_h32_t2_contiguous_4d"
    assert _resolved_shape(result) == {
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
        source_path="tests/kernels/data/llamacpp_test.v2.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, UnmappedCase)
    assert result.reason.value == "shape_lowering_not_implemented"
    assert result.detail == "ROPE v2 routing requires contiguous input (v=0)"


def test_v2_rope_neox_route_resolves_for_plain_f32_case() -> None:
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
        source_path="tests/kernels/data/llamacpp_test.v2.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, ResolvedBenchmarkCase)
    assert result.route_id == "rope_neox_f32_n64_h128_t2_contiguous_4d"
    assert _resolved_shape(result) == {
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
        source_path="tests/kernels/data/llamacpp_test.v2.yaml",
        source_group_index=0,
        source_case_index=20,
    )

    rope_routes = [
        route
        for route in routes_for_op(catalog, "ROPE")
        if route.id == "rope_neox_f32_n64_h128_t2_contiguous_4d"
    ]

    resolved_route, shape, reason, detail = resolve_route_for_case(
        case,
        rope_routes,
        selector=PythonRouteSelector(rope_routes),
    )

    assert resolved_route is None
    assert shape is None
    assert reason is not None
    assert reason.value == "shape_lowering_not_implemented"
    assert detail == "ROPE NEOX v2 routing currently requires n_dims == ne_a[0]"


def test_v2_rope_scale_route_resolves_for_neox_freq_f32_case() -> None:
    case = ImportedCase(
        op="ROPE_SCALE",
        dtype={"type": "f32"},
        raw_case={},
        normalized_params={
            "af": 1.0,
            "ef": 0.0,
            "ff": 1,
            "fs": 1.0,
            "inplace": 0,
            "mode": 2,
            "n_ctx": 512,
            "n_dims": 96,
            "ne_a": [128, 24, 1, 1],
            "v": 0,
        },
        source_path="tests/kernels/data/additional_test.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, ResolvedBenchmarkCase)
    assert result.route_id == "rope_scale_f32_neox_freq_n128_d96_h24_t1_contiguous_4d"
    assert _resolved_shape(result) == {
        "d0": 128,
        "d1": 24,
        "d2": 1,
        "d3": 1,
        "src1_d0": 1,
        "src1_d1": 1,
        "src2_d0": 48,
        "src2_d1": 1,
        "rope.ncols": 128,
        "rope.n_dims": 96,
        "rope.nheads": 24,
        "rope.ntokens": 1,
        "rope.src0_head_stride": 128,
        "rope.src0_token_stride": 3072,
        "rope.dst_head_stride": 128,
        "rope.dst_token_stride": 3072,
        "rope.pos_token_stride": 1,
    }


def test_v2_rope_scale_nonzero_ext_factor_stays_unmapped() -> None:
    case = ImportedCase(
        op="ROPE_SCALE",
        dtype={"type": "f32"},
        raw_case={},
        normalized_params={
            "af": 1.0,
            "ef": 1.0,
            "ff": 1,
            "fs": 1.0,
            "inplace": 0,
            "mode": 2,
            "n_ctx": 512,
            "n_dims": 96,
            "ne_a": [128, 24, 1, 1],
            "v": 0,
        },
        source_path="tests/kernels/data/additional_test.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, UnmappedCase)
    assert result.reason.value == "shape_lowering_not_implemented"
    assert result.detail == "ROPE_SCALE v2 routing currently requires ef=0.0"


def test_v2_rope_set_rows_route_resolves_for_f16_mode0_case() -> None:
    case = ImportedCase(
        op="ROPE_SET_ROWS",
        dtype={"type": "f16", "type_idx": "i64"},
        raw_case={},
        normalized_params={
            "mode": 0,
            "ne_a": [128, 32, 1, 1],
        },
        source_path="tests/kernels/data/llamacpp_test.v2.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, ResolvedBenchmarkCase)
    assert result.route_id == "rope_set_rows_f16_normal_n128_h32_t1_contiguous_4d"
    shape = _resolved_shape(result)
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
    case = ImportedCase(
        op="ROPE_SET_ROWS",
        dtype={"type": "f32", "type_idx": "i64"},
        raw_case={},
        normalized_params={
            "mode": 0,
            "ne_a": [128, 32, 8, 1],
        },
        source_path="tests/kernels/data/llamacpp_test.v2.yaml",
        source_group_index=1,
        source_case_index=4,
    )

    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, UnmappedCase)
    assert result.reason.value == "no_dtype_mapping"
    assert result.detail == "matching v2 op mapping exists, but not for this dtype combination"


def test_v2_rope_set_rows_batch_case_stays_unmapped() -> None:
    case = ImportedCase(
        op="ROPE_SET_ROWS",
        dtype={"type": "f16", "type_idx": "i64"},
        raw_case={},
        normalized_params={
            "mode": 0,
            "ne_a": [128, 32, 8, 3],
        },
        source_path="tests/kernels/data/llamacpp_test.v2.yaml",
        source_group_index=0,
        source_case_index=5,
    )

    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, UnmappedCase)
    assert result.reason.value == "shape_lowering_not_implemented"
    assert result.detail == "ROPE_SET_ROWS v2 routing currently requires ne_a[3]=1"


def test_v2_rope_set_rows_multi_token_case_stays_unmapped() -> None:
    case = ImportedCase(
        op="ROPE_SET_ROWS",
        dtype={"type": "f16", "type_idx": "i64"},
        raw_case={},
        normalized_params={
            "mode": 0,
            "ne_a": [128, 32, 8, 1],
        },
        source_path="tests/kernels/data/llamacpp_test.v2.yaml",
        source_group_index=0,
        source_case_index=4,
    )

    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, UnmappedCase)
    assert result.reason.value == "no_route_match"


def test_v2_soft_max_route_resolves_for_plain_f32_case() -> None:
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
        source_path="tests/kernels/data/llamacpp_test.v2.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, ResolvedBenchmarkCase)
    assert result.route_id == "soft_max_f32_contiguous_2d"
    assert _resolved_shape(result) == {"d0": 16, "d1": 16}


def test_v2_soft_max_masked_case_stays_unmapped() -> None:
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
        source_path="tests/kernels/data/llamacpp_test.v2.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, ResolvedBenchmarkCase)
    assert result.route_id == "soft_max_f32_mask_contiguous_2d"
    assert _resolved_shape(result) == {"d0": 1024, "d1": 16}


def test_v2_soft_max_masked_broadcast_case_stays_unmapped() -> None:
    case = ImportedCase(
        op="SOFT_MAX",
        dtype={"type": "f32"},
        raw_case={},
        normalized_params={
            "inplace": 0,
            "m_prec": "f16",
            "mask": 1,
            "max_bias": 0.0,
            "ne": [15, 15, 1, 1],
            "nr23": [2, 3],
            "scale": 1.0,
            "sinks": 0,
        },
        source_path="tests/kernels/data/llamacpp_test.v2.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, UnmappedCase)
    assert result.reason.value == "shape_lowering_not_implemented"
    assert result.detail == "SOFT_MAX masked v2 routing currently requires nr23=[1, 1]"


def test_v2_soft_max_large_ncols_case_stays_unmapped() -> None:
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
        source_path="tests/kernels/data/llamacpp_test.v2.yaml",
        source_group_index=0,
        source_case_index=0,
    )

    result = _resolve_one_case(ACTUAL_V2_ROUTER, case)

    assert isinstance(result, UnmappedCase)
    assert result.reason.value == "shape_lowering_not_implemented"
    assert result.detail == "SOFT_MAX v2 routing currently requires ne[0] <= 1024"
