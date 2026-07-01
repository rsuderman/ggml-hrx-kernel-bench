from __future__ import annotations

import json
from pathlib import Path

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
                "id": "add_f32_contiguous_2d",
                "family": "add_f32",
                "kernel": {
                    "source_id": "add_pointwise_f32",
                    "path": "add_pointwise_f32.loom",
                    "root_symbol": "@hrx2_add_f32_contiguous_2d",
                    "export_name": "hrx2_add_f32_contiguous_2d",
                },
                "tensors": {
                    "src0": {
                        "dtype": "F32",
                        "dimensions": ["ncols", "nrows"],
                        "strides": ["src0_stride_ncols", "src0_stride_nrows"],
                    },
                    "src1": {
                        "dtype": "F32",
                        "dimensions": ["ncols", "nrows"],
                        "strides": ["src1_stride_ncols", "src1_stride_nrows"],
                    },
                    "dst": {
                        "dtype": "F32",
                        "dimensions": ["ncols", "nrows"],
                        "strides": ["dst_stride_ncols", "dst_stride_nrows"],
                    },
                },
                "constraints": [
                    {"name": "ncols", "min": 1, "max": 65536},
                    {"name": "nrows", "min": 64, "max": 1048576},
                    {"name": "src0_stride_ncols", "value": 1},
                    {"name": "src0_stride_nrows", "dimension": "ncols"},
                    {"name": "src1_stride_ncols", "value": 1},
                    {"name": "src1_stride_nrows", "dimension": "ncols"},
                    {"name": "dst_stride_ncols", "value": 1},
                    {"name": "dst_stride_nrows", "dimension": "ncols"},
                ],
                "launch": {
                    "workgroup_size": [256, 1, 1],
                    "rows_per_workgroup": 1,
                    "cols_per_workgroup": 256,
                },
                "config": {
                    "bindings": [
                        {
                            "key": "@hrx2.shape.pointwise.ncols",
                            "source": "tensor.dst.dimensions.ncols.size",
                        },
                        {
                            "key": "@hrx2.shape.pointwise.nrows",
                            "source": "tensor.dst.dimensions.nrows.size",
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
    (kernel_dir / "add_pointwise_f32.loom").write_text(
        'kernel.def export("hrx2_add_f32_contiguous_2d") @hrx2_add_f32_contiguous_2d\n',
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
    assert candidate.route_id == "add_f32_contiguous_2d"
    assert candidate.root_symbol == "@hrx2_add_f32_contiguous_2d"
    assert candidate.shape == {"ncols": 1, "nrows": 64, "cols": 1, "rows": 64}


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
                        raw_case={"ne": [10, 100, 1, 1], "nr": [1, 1, 1, 1], "nf": 1, "perm1": 0},
                        normalized_params={"ne": [10, 100, 1, 1], "nr": [1, 1, 1, 1], "nf": 1, "perm1": 0},
                        source_path="test.yaml",
                        source_group_index=0,
                        source_case_index=0,
                    ),
                    ImportedCase(
                        op="ADD",
                        dtype={"type": "f32"},
                        raw_case={"ne": [10, 5, 1, 1], "nr": [1, 1, 1, 1], "nf": 1, "perm1": 0},
                        normalized_params={"ne": [10, 5, 1, 1], "nr": [1, 1, 1, 1], "nf": 1, "perm1": 0},
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
    assert resolved.resolved[0].route_id == "add_f32_contiguous_2d"
    assert resolved.resolved[0].params == ["ncols", "nrows", "cols", "rows"]
    assert resolved.resolved[0].values == [10, 100, 10, 100]
    assert len(resolved.unmapped) == 1
    assert resolved.unmapped[0].reason == UnmappedReason.NO_ROUTE_MATCH


def test_v2_router_executes_matching_case(tmp_path: Path, monkeypatch) -> None:
    kernel_dir = tmp_path / "kernels"
    routing_dir = tmp_path / "routing"
    output_dir = tmp_path / "out"
    _write_kernel(kernel_dir)
    _write_v2_descriptor(routing_dir)
    router = create_router(version="v2", kernel_dir=kernel_dir, routing_dir=routing_dir)
    config = {
        "kernel": "add_f32",
        "route_id": "add_f32_contiguous_2d",
        "params": ["ncols", "nrows", "cols", "rows"],
        "cases": [[10, 100, 10, 100]],
    }

    def fake_run_candidate_row(args, bench_config, candidate, *, sanitizer):
        assert sanitizer == "none"
        assert candidate.source_path == kernel_dir / "add_pointwise_f32.loom"
        assert candidate.root_symbol == "@hrx2_add_f32_contiguous_2d"
        assert candidate.config == {
            "@hrx2.shape.pointwise.ncols": "10",
            "@hrx2.shape.pointwise.nrows": "100",
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
        "ggml_hrx_kernel_bench.routing.v2.backend.run_candidate_row",
        fake_run_candidate_row,
    )
    execution = router.execute_case(
        RuntimeCaseRequest(
            kernel_dir=kernel_dir,
            routing_dir=routing_dir,
            config_data=config,
            current_case_id="ncols10_nrows100_cols10_rows100",
            current_case_values=[10, 100, 10, 100],
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
    assert result["shape"] == {"ncols": 10, "nrows": 100, "cols": 10, "rows": 100}
