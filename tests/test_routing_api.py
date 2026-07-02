from __future__ import annotations

import json
from pathlib import Path

import pytest

from ggml_hrx_kernel_bench.import_models import (
    ImportedCase,
    ImportedOpGroup,
    ImportedSuite,
    UnmappedReason,
)
from ggml_hrx_kernel_bench.routing.api import (
    CandidateQuery,
    RuntimeCaseRequest,
    create_router,
)
from ggml_hrx_kernel_bench.routing.v2.import_resolution import resolve_imported_suite
from ggml_hrx_kernel_bench.routing.v2.manifest import build_manifest
from ggml_hrx_kernel_bench.routing.v2.query import load_route_catalog


def _write_v2_descriptor(routing_dir: Path) -> None:
    routing_dir.mkdir(parents=True, exist_ok=True)
    (routing_dir / "router.json").write_text(
        json.dumps(
            {
                "schema": "ggml_hrx_kernel_bench.routing_descriptors.v2",
                "routes": {"ADD": ["add_f32.json"]},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (routing_dir / "add_f32.json").write_text(
        json.dumps(
            {
                "id": "add_f32_contiguous_1d",
                "family": "add_f32",
                "kernel": {
                    "source_id": "add_f32",
                    "path": "add_f32.loom",
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


def _write_kernel(kernel_dir: Path) -> None:
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "add_f32.loom").write_text(
        'kernel.def export("hrx2_add_f32_contiguous_1d") @hrx2_add_f32_contiguous_1d\n',
        encoding="utf-8",
    )


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

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.family == "add_f32"
    assert candidate.route_id == "add_f32_contiguous_1d"
    assert candidate.root_symbol == "@hrx2_add_f32_contiguous_1d"
    assert candidate.shape == {"ncols": 256, "nrows": 1, "cols": 256, "rows": 1}


def test_v2_manifest_includes_original_root_metadata(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernels"
    routing_dir = tmp_path / "routing"
    original_root = tmp_path / "original"
    _write_kernel(kernel_dir)
    _write_v2_descriptor(routing_dir)
    (original_root / "kernels").mkdir(parents=True, exist_ok=True)
    (original_root / "kernels" / "add_f32.loom").write_text(
        'kernel.def export("hrx2_add_f32_contiguous_1d") @hrx2_add_f32_contiguous_1d\n',
        encoding="utf-8",
    )

    router = create_router(version="v2", kernel_dir=kernel_dir, routing_dir=routing_dir)

    manifest = router.manifest(original_root=original_root)

    assert manifest["route_count"] == 1
    assert len(manifest["entries"]) == 1
    entry = manifest["entries"][0]
    assert entry["original_path"] == str(original_root / "kernels" / "add_f32.loom")
    assert entry["original_sha256"] is not None
    assert entry["imported_sha256"] is not None
    assert entry["mechanical_rewrites"] == []


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
    assert candidate.route["values"] == [
        {"name": "contiguous_strides", "contiguous_strides": "dst_dimensions"},
        {"name": "total_size", "product": "dst_dimensions"},
    ]
    assert candidate.route["constraints"] == [
        {"equals": ["src0_dimensions", "src1_dimensions", "dst_dimensions"]},
        {"equals": ["contiguous_strides", "src0_strides", "src1_strides", "dst_strides"]},
    ]


def test_v2_helpers_require_catalog_or_routing_dir(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="routing_dir or catalog is required"):
        build_manifest(kernel_dir=tmp_path / "kernels")
    with pytest.raises(ValueError, match="routing_dir or catalog is required"):
        resolve_imported_suite(
            ImportedSuite(schema="test", source_path="test.yaml", op_groups=[]),
        )


def test_v2_router_maps_only_matching_contiguous_add_cases(tmp_path: Path) -> None:
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
                        raw_case={"ne": [16, 64, 1, 1], "nr": [1, 1, 1, 1], "nf": 1, "perm1": 0},
                        normalized_params={"ne": [16, 64, 1, 1], "nr": [1, 1, 1, 1], "nf": 1, "perm1": 0},
                        source_path="test.yaml",
                        source_group_index=0,
                        source_case_index=0,
                    ),
                    ImportedCase(
                        op="ADD",
                        dtype={"type": "f32"},
                        raw_case={"ne": [10, 5, 1, 1], "nr": [1, 1, 1, 2], "nf": 1, "perm1": 0},
                        normalized_params={"ne": [10, 5, 1, 1], "nr": [1, 1, 1, 2], "nf": 1, "perm1": 0},
                        source_path="test.yaml",
                        source_group_index=0,
                        source_case_index=1,
                    ),
                ),
            )
        ],
    )

    resolved = router.resolve_imported_suite(suite)

    assert len(resolved.resolved) == 1
    assert resolved.resolved[0].kernel_family == "add_f32"
    assert resolved.resolved[0].route_id == "add_f32_contiguous_1d"
    assert resolved.resolved[0].params == ["ncols", "nrows", "cols", "rows"]
    assert resolved.resolved[0].values == [16, 64, 16, 64]
    assert len(resolved.unmapped) == 1
    assert resolved.unmapped[0].reason == UnmappedReason.SHAPE_LOWERING_NOT_IMPLEMENTED


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
        "params": ["ncols", "nrows", "cols", "rows"],
        "cases": [[16, 64, 16, 64]],
    }

    def fake_run_candidate_row(args, bench_config, candidate, *, sanitizer):
        assert sanitizer == "none"
        assert candidate.source_path == kernel_dir / "add_f32.loom"
        assert candidate.root_symbol == "@hrx2_add_f32_contiguous_1d"
        assert candidate.config == {
            "@hrx2.shape.pointwise.total_size": "1024",
            "@hrx2.tuning.pointwise.workgroup_size": "256",
        }
        return {
            "status": "ran",
            "benchmark": {
                "summary": {
                    "correctness": {"state": "ok"},
                    "operation_timing_ns": {"mean": 1.0},
                }
            },
        }

    monkeypatch.setattr(
        "ggml_hrx_kernel_bench.routing.v2.runtime.run_candidate_row",
        fake_run_candidate_row,
    )
    execution = router.execute_case(
        RuntimeCaseRequest(
            kernel_dir=kernel_dir,
            routing_dir=routing_dir,
            config_data=config,
            current_case_id="ncols16_nrows64_cols16_rows64",
            current_case_values=[16, 64, 16, 64],
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
    assert result["shape"] == {"ncols": 16, "nrows": 64, "cols": 16, "rows": 64}
