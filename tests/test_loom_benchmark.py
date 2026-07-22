from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ggml_hrx_kernel_bench.benchmarking import collect, common, compare, discovery, materialize, result_parsing, workbench  # noqa: E402


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _descriptor(
    *,
    kernel: Path,
    route_id: str = "add_f32_contiguous_4d",
    case_id: str = "d04",
) -> dict[str, object]:
    return {
        "schema": "ggml_hrx_kernel_bench.loom_execution_descriptor.v1",
        "kernel": str(kernel),
        "root": "@add_f32",
        "target": "gfx1100",
        "workgroup_count": [1, 1, 1],
        "configs": {
            "@shape.pointwise.ne0": "4",
            "@tuning.pointwise.workgroup_size": "256",
        },
        "scalars": [],
        "bindings": [
            {
                "position": 0,
                "kind": "input",
                "dtype": "f32",
                "name": "src0",
                "path": "fixtures/src0.npy",
            },
            {
                "position": 1,
                "kind": "input",
                "dtype": "f32",
                "name": "src1",
                "path": "fixtures/src1.npy",
            },
            {
                "position": 2,
                "kind": "output",
                "dtype": "f32",
                "name": "dst",
                "path": "fixtures/dst_init.npy",
                "expect": {
                    "mode": "close",
                    "path": "fixtures/expected.npy",
                    "atol": 1e-6,
                    "rtol": 1e-6,
                },
            },
        ],
        "metadata": {
            "source": "generated-kernel-tests",
            "case_id": case_id,
            "case_values": [4],
            "shape": {"d0": 4},
            "route_id": route_id,
            "candidate_id": f"{route_id}_candidate",
            "element_counts": {
                "src0": 4,
                "src1": 4,
                "dst": 4,
            },
            "oracle_array_element_counts": {
                "src0": 4,
                "src1": 4,
                "dst_init": 4,
                "expected": 4,
            },
        },
    }


def _write_prepared_case(
    tmp_path: Path,
    *,
    op: str = "ADD",
    route_id: str = "add_f32_contiguous_4d",
    case_id: str = "d04",
) -> tuple[Path, Path, Path]:
    repo_root = tmp_path / "repo"
    asset_root = repo_root / "build" / "generated" / "assets"
    kernel = asset_root / "kernels" / "v2" / "add" / "f32.loom"
    kernel.parent.mkdir(parents=True, exist_ok=True)
    kernel.write_text("kernel.def @add_f32() {} launch(%src0: buffer) { kernel.return }\n", encoding="utf-8")

    descriptor_root = tmp_path / "descriptors" / op
    for fixture_name in ("src0.npy", "src1.npy", "dst_init.npy", "expected.npy"):
        fixture_path = descriptor_root / "fixtures" / fixture_name
        fixture_path.parent.mkdir(parents=True, exist_ok=True)
        fixture_path.write_bytes(fixture_name.encode("utf-8"))
    descriptor_path = descriptor_root / "case.json"
    _write_json(
        descriptor_path,
        _descriptor(kernel=kernel, route_id=route_id, case_id=case_id),
    )

    prepare_root = tmp_path / "prepare"
    _write_json(
        prepare_root / op / "loom-execution-runs.json",
        {
            "schema": "ggml_hrx_kernel_bench.loom_execution_runs.v1",
            "descriptor_manifest_path": str(descriptor_root / "loom-execution-descriptors.json"),
            "execute": False,
            "entry_count": 1,
            "prepared_count": 1,
            "executed_count": 0,
            "passed_count": 0,
            "failed_count": 0,
            "skipped_count": 0,
            "entries": [
                {
                    "descriptor_path": str(descriptor_path),
                    "case_id": case_id,
                    "case_values": [4],
                    "kernel": "add_f32",
                    "route_id": route_id,
                    "status": "prepared",
                }
            ],
        },
    )
    return prepare_root, repo_root, asset_root


def test_discovers_buckets_from_prepared_manifest(tmp_path: Path) -> None:
    prepare_root, repo_root, asset_root = _write_prepared_case(tmp_path)

    cases = discovery.discover_cases(
        prepare_root=prepare_root,
        repo_root=repo_root,
        asset_root=asset_root,
    )
    buckets = discovery.bucket_cases(cases)

    assert len(buckets) == 1
    bucket = buckets[0]
    assert bucket.op == "ADD"
    assert bucket.route_id == "add_f32_contiguous_4d"
    assert bucket.normalized_kernel_source == "asset:kernels/v2/add/f32.loom"
    assert len(bucket.implementation_id) == 16
    assert len(bucket.cases) == 1


def test_bucket_dedupes_equal_descriptor_execution_digest(tmp_path: Path) -> None:
    prepare_root, repo_root, asset_root = _write_prepared_case(tmp_path)
    manifest_path = prepare_root / "ADD" / "loom-execution-runs.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["entries"].append(dict(manifest["entries"][0]))
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    cases = discovery.discover_cases(
        prepare_root=prepare_root,
        repo_root=repo_root,
        asset_root=asset_root,
    )

    assert len(cases) == 2
    assert len(discovery.bucket_cases(cases)[0].cases) == 1
    assert len(discovery.bucket_cases(cases, dedupe=False)[0].cases) == 2


def test_estimates_mul_mat_flops_from_descriptor_configs(tmp_path: Path) -> None:
    kernel = tmp_path / "mul_mat.loom"
    descriptor = _descriptor(kernel=kernel, route_id="mul_mat_f16_f32_tiled_batched_4d")
    descriptor["configs"] = {
        "@shape.mul_mat_f16.rows": "16",
        "@shape.mul_mat_f16.cols": "16",
        "@shape.mul_mat_f16.k": "4",
        "@shape.mul_mat_f16.dst_ne2": "3",
        "@shape.mul_mat_f16.dst_ne3": "2",
    }

    estimate = discovery._estimate_mul_mat_flops(descriptor)

    assert estimate["status"] == "estimated"
    assert estimate["estimated_flops"] == 12288
    assert estimate["formula"] == "2 * rows * cols * k * dst_ne2 * dst_ne3"


def test_benchmark_result_summary_includes_flops_per_second(tmp_path: Path) -> None:
    result_path = tmp_path / "benchmark-results.jsonl"
    rows = [
        {
            "row": "benchmark",
            "benchmark_result": {
                "benchmark": "bench_mul_mat",
                "case": "case_mul_mat",
                "state": "ok",
                "measurement": {
                    "operation_timing_ns": {
                        "mean": 20_000,
                        "p50": 10_000,
                    }
                },
            },
        }
    ]
    result_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    summary = result_parsing._benchmark_result_summary(
        result_path,
        flop_estimate={"estimated_flops": 1_000},
    )

    throughput = summary["throughput"]["rows"][0]
    assert throughput["estimated_flops_per_operation"] == 1_000
    assert throughput["flops_per_second_from_mean_ns"] == 50_000_000.0
    assert throughput["flops_per_second_from_p50_ns"] == 100_000_000.0
    assert summary["operation_timing_ns"]["mean"] == 20_000


def test_benchmark_result_summary_rejects_empty_benchmark_output(tmp_path: Path) -> None:
    result_path = tmp_path / "benchmark-results.jsonl"
    result_path.write_text(json.dumps({"row": "summary", "summary": {}}) + "\n", encoding="utf-8")

    summary = result_parsing._benchmark_result_summary(result_path)

    assert summary["status"] == "no_benchmark_rows"
    assert summary["benchmark_row_count"] == 0


def test_benchmark_result_summary_extracts_compile_summary(tmp_path: Path) -> None:
    result_path = tmp_path / "benchmark-results.jsonl"
    row = {
        "row": "compile",
        "compile_report": {
            "target_key": "gfx1100",
            "artifact_size": 4096,
            "instruction_count": 123,
            "local_memory_bytes": 64,
            "private_memory_bytes": 0,
            "allocation_spill_count": 0,
            "target_resources": {
                "scalar": {"final": {"register_count": 20}},
                "vector": {"final": {"register_count": 32}},
            },
            "static_instruction_mix": {
                "local_memory_count": 3,
                "barrier_count": 1,
            },
        },
    }
    result_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    summary = result_parsing._benchmark_result_summary(result_path)

    assert summary["compile_summary"] == {
        "allocation_spill_count": 0,
        "artifact_size": 4096,
        "barrier_instruction_count": 1,
        "instruction_count": 123,
        "lds_instruction_count": 3,
        "local_memory_bytes": 64,
        "private_memory_bytes": 0,
        "sgpr_count": 20,
        "target_key": "gfx1100",
        "vgpr_count": 32,
    }


def _write_tool_benchmark_result(path: Path, *, mean_ns: float, state: str = "ok") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "row": "benchmark",
        "benchmark_result": {
            "state": state,
            "measurement": {
                "operation_timing_ns": {
                    "mean": mean_ns,
                    "p50": mean_ns,
                    "p90": mean_ns,
                }
            },
        },
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def _write_wrapper_result(
    path: Path,
    *,
    case_means: dict[str, float],
    candidate_name: str,
    status: str = "pass",
    compile_summary: dict[str, float] | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for case_id, mean_ns in case_means.items():
        output = path.parent / f"{candidate_name}-{case_id}.jsonl"
        _write_tool_benchmark_result(output, mean_ns=mean_ns)
        lines.append(
            json.dumps(
                {
                    "schema": common.RESULT_SCHEMA,
                    "candidate_name": candidate_name,
                    "case_id": case_id,
                    "op": "MUL_MAT",
                    "route_id": "mul_mat_f16_f32_tiled_batched_4d",
                    "status": status,
                    "estimated_flops": 1_000,
                    "shape_bucket": {
                        "k": 256 if case_id.endswith("large") else 4,
                        "layout_kind": "contiguous_or_unknown",
                        "flop_bucket": "1k-10k",
                    },
                    "benchmark_output_path": str(output),
                    "compile_summary": compile_summary or {},
                },
                sort_keys=True,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_compare_result_sets_reports_ratios_and_groups(tmp_path: Path) -> None:
    baseline = _write_wrapper_result(
        tmp_path / "baseline" / "results.jsonl",
        case_means={"case_small": 100.0, "case_large": 200.0},
        candidate_name="baseline",
    )
    candidate = _write_wrapper_result(
        tmp_path / "candidate" / "results.jsonl",
        case_means={"case_small": 90.0, "case_large": 240.0},
        candidate_name="candidate",
    )

    result = compare.compare_result_sets(
        baseline_path=baseline,
        candidates=[("candidate", candidate)],
        threshold=0.05,
        group_by=["k"],
    )

    candidate_summary = result["candidates"][0]
    assert candidate_summary["wins"] == 1
    assert candidate_summary["losses"] == 1
    assert candidate_summary["neutral"] == 0
    assert candidate_summary["groups"]["k=4"]["geomean_time_ratio"] == 0.9
    assert candidate_summary["groups"]["k=256"]["geomean_time_ratio"] == 1.2


def test_compare_command_regression_gate_returns_failure(tmp_path: Path) -> None:
    baseline = _write_wrapper_result(
        tmp_path / "baseline" / "results.jsonl",
        case_means={"case_small": 100.0},
        candidate_name="baseline",
    )
    candidate = _write_wrapper_result(
        tmp_path / "candidate" / "results.jsonl",
        case_means={"case_small": 120.0},
        candidate_name="candidate",
    )
    args = SimpleNamespace(
        baseline=baseline,
        candidate=[f"candidate={candidate}"],
        threshold=0.02,
        group_by=[],
        json=True,
        fail_on_geomean_regression=0.05,
        fail_on_case_regression=None,
        fail_on_correctness_failure=False,
    )

    assert compare.command_compare(args) == 1


def test_compare_command_applies_policy_and_writes_output(tmp_path: Path) -> None:
    baseline = _write_wrapper_result(
        tmp_path / "baseline" / "results.jsonl",
        case_means={"case_small": 100.0},
        candidate_name="baseline",
    )
    candidate = _write_wrapper_result(
        tmp_path / "candidate" / "results.jsonl",
        case_means={"case_small": 120.0},
        candidate_name="candidate",
    )
    policy = tmp_path / "policy.json"
    _write_json(policy, {"max_geomean_regression": 0.05, "threshold": 0.02})
    output = tmp_path / "compare.json"

    status = compare.command_compare(
        SimpleNamespace(
            baseline=baseline,
            candidate=[f"candidate={candidate}"],
            threshold=0.10,
            group_by=[],
            json=False,
            output=output,
            policy=policy,
            fail_on_geomean_regression=None,
            fail_on_case_regression=None,
            fail_on_correctness_failure=False,
        )
    )

    compare_payload = json.loads(output.read_text(encoding="utf-8"))
    assert status == 1
    assert compare_payload["threshold"] == 0.02
    assert compare_payload["acceptance"]["passed"] is False
    assert compare_payload["acceptance"]["failures"][0]["kind"] == "geomean_regression"


def test_writes_transient_descriptor_workbench(tmp_path: Path, monkeypatch) -> None:
    prepare_root, repo_root, asset_root = _write_prepared_case(tmp_path)
    case = discovery.discover_cases(
        prepare_root=prepare_root,
        repo_root=repo_root,
        asset_root=asset_root,
    )[0]

    def fake_run(command, *, check, text, stdout, stderr):
        output = Path(next(part.split("=", 1)[1] for part in command if str(part).startswith("--output=")))
        output.write_text("linked source\n", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(workbench.subprocess, "run", fake_run)

    workbench_path, bench_symbol, metadata = workbench._write_descriptor_workbench(
        case=case,
        run_dir=tmp_path / "run",
        repo_root=repo_root,
        loom_link="loom-link",
    )

    text = workbench_path.read_text(encoding="utf-8")
    assert "linked source" in text
    assert "check.case public @case_add_f32_contiguous_4d_" in text
    assert "func.call @add_f32(%src0, %src1, %dst)" in text
    assert f"check.benchmark<@case_add_f32_contiguous_4d_{case.execution_digest[:12]}> {bench_symbol}" in text
    assert (tmp_path / "run" / "fixtures" / "src0.npy").read_bytes() == b"src0.npy"
    assert metadata["workbench_path"] == str(workbench_path)
    assert metadata["descriptor_kernel_source"] == case.kernel_source
    assert metadata["effective_kernel_source"] == case.kernel_source
    assert metadata["source_override_used"] is False


def test_generate_scripts_materializes_catalog_tree(tmp_path: Path, monkeypatch) -> None:
    prepare_root, repo_root, asset_root = _write_prepared_case(tmp_path)
    output_root = tmp_path / "benchmarks"
    stale_route_dir = output_root / "catalog" / "v2" / "ADD" / "stale_route"
    _write_json(
        output_root / "catalog" / "v2" / "index.json",
        {
            "schema": common.SCRIPT_INDEX_SCHEMA,
            "routes": [{"manifest_path": str(stale_route_dir / "manifest.json")}],
        },
    )
    (stale_route_dir / "cases" / "stale_case").mkdir(parents=True)
    (stale_route_dir / "run.sh").write_text("# stale\n", encoding="utf-8")
    (stale_route_dir / "runs").mkdir()
    (stale_route_dir / "runs" / "keep.txt").write_text("old result\n", encoding="utf-8")

    def fake_workbench(*, case, run_dir, repo_root, loom_link, kernel_source_override=None):
        run_dir.mkdir(parents=True, exist_ok=True)
        workbench = run_dir / "benchmark.loom"
        workbench.write_text("check.benchmark<@case> @bench\n", encoding="utf-8")
        return (
            workbench,
            "@bench_add",
            {
                "descriptor_kernel_source": case.kernel_source,
                "descriptor_kernel_source_hash": "descriptor-hash",
                "effective_kernel_source": str(kernel_source_override or case.kernel_source),
                "effective_kernel_source_hash": "effective-hash",
                "source_override_used": kernel_source_override is not None,
            },
        )

    monkeypatch.setattr(materialize, "_write_descriptor_workbench", fake_workbench)

    status = materialize.command_generate_scripts(
        SimpleNamespace(
            prepare_root=prepare_root,
            repo_root=repo_root,
            asset_root=asset_root,
            op=None,
            route_id=None,
            implementation_id=None,
            kernel_source=None,
            root=None,
            case_id=None,
            no_dedupe=False,
            output_root=output_root,
            tool_dir=None,
            benchmark_runner="/tools/iree-benchmark-loom",
            benchmark_device="amdgpu",
            benchmark_measure="dispatch_complete",
            loom_link="/tools/loom-link",
            python_executable=sys.executable,
        )
    )

    route_dir = output_root / "catalog" / "v2" / "ADD" / "add_f32_contiguous_4d"
    route_manifest = json.loads((route_dir / "manifest.json").read_text(encoding="utf-8"))
    case_manifest_path = Path(route_manifest["cases"][0]["manifest_path"])
    case_script = (case_manifest_path.parent / "run.sh").read_text(encoding="utf-8")
    route_script = (route_dir / "run.sh").read_text(encoding="utf-8")
    collect_script = (route_dir / "collect.sh").read_text(encoding="utf-8")

    assert status == 0
    assert (output_root / "catalog" / "v2" / "index.json").is_file()
    assert (route_dir / "run.sh").stat().st_mode & 0o111
    assert (route_dir / "collect.sh").stat().st_mode & 0o111
    assert case_manifest_path.is_file()
    assert "iree-benchmark-loom" in case_script
    assert "--device=$BENCHMARK_DEVICE" in case_script
    assert "BENCHMARK_DEVICE=amdgpu" in case_script
    assert "--measure=$BENCHMARK_MEASURE" in case_script
    assert "BENCHMARK_MEASURE=dispatch_complete" in case_script
    assert "while [[ $# -gt 0 ]]" not in case_script
    assert "while [[ $# -gt 0 ]]" not in route_script
    assert "while [[ $# -gt 0 ]]" not in collect_script
    assert "ggml_hrx_kernel_bench.benchmarking.collect" in collect_script
    assert "PYTHONPATH=\"$REPO_ROOT/src:$PYTHONPATH\"" in collect_script
    assert "loom-bench-collect" not in collect_script
    assert "--kernel-source" not in case_script
    assert "--kernel-source" not in route_script
    assert "prepare_case" not in case_script
    assert '"$@"' in case_script
    assert 'bash "$CASE_SCRIPT" "$RUN_DIR/cases/$CASE_NAME" "$@"' in route_script
    assert route_manifest["schema"] == common.SCRIPT_ROUTE_MANIFEST_SCHEMA
    assert route_manifest["case_count"] == 1
    assert route_manifest["defaults"]["benchmark_runner"] == "/tools/iree-benchmark-loom"
    assert route_manifest["defaults"]["benchmark_device"] == "amdgpu"
    assert route_manifest["defaults"]["benchmark_measure"] == "dispatch_complete"
    case_manifest = json.loads(case_manifest_path.read_text(encoding="utf-8"))
    assert case_manifest["defaults"]["benchmark_device"] == "amdgpu"
    assert case_manifest["defaults"]["benchmark_measure"] == "dispatch_complete"
    assert not (stale_route_dir / "run.sh").exists()
    assert not (stale_route_dir / "cases").exists()
    assert (stale_route_dir / "runs" / "keep.txt").is_file()


def test_generate_scripts_bakes_candidate_kernel_source(tmp_path: Path, monkeypatch) -> None:
    prepare_root, repo_root, asset_root = _write_prepared_case(tmp_path)
    candidate_source = tmp_path / "candidate.loom"
    candidate_source.write_text("kernel.def @add_f32() {} launch(%src0: buffer) { kernel.return }\n", encoding="utf-8")
    output_root = tmp_path / "benchmarks"

    def fake_workbench(*, case, run_dir, repo_root, loom_link, kernel_source_override=None):
        run_dir.mkdir(parents=True, exist_ok=True)
        workbench = run_dir / "benchmark.loom"
        workbench.write_text("check.benchmark<@case> @bench\n", encoding="utf-8")
        return (
            workbench,
            "@bench_add",
            {
                "descriptor_kernel_source": case.kernel_source,
                "descriptor_kernel_source_hash": "descriptor-hash",
                "effective_kernel_source": str(kernel_source_override or case.kernel_source),
                "effective_kernel_source_hash": "effective-hash",
                "source_override_used": kernel_source_override is not None,
            },
        )

    monkeypatch.setattr(materialize, "_write_descriptor_workbench", fake_workbench)

    assert (
        materialize.command_generate_scripts(
            SimpleNamespace(
                prepare_root=prepare_root,
                repo_root=repo_root,
                asset_root=asset_root,
                op="ADD",
                route_id=None,
                implementation_id=None,
                kernel_source=candidate_source,
                root=None,
                case_id=None,
                no_dedupe=False,
                output_root=output_root,
                tool_dir=None,
                benchmark_runner="/tools/iree-benchmark-loom",
                benchmark_device="amdgpu",
                benchmark_measure="dispatch_complete",
                loom_link="/tools/loom-link",
                python_executable=sys.executable,
            )
        )
        == 0
    )

    route_dir = output_root / "catalog" / "v2" / "ADD" / "add_f32_contiguous_4d"
    route_manifest = json.loads((route_dir / "manifest.json").read_text(encoding="utf-8"))
    case_manifest = json.loads(Path(route_manifest["cases"][0]["manifest_path"]).read_text(encoding="utf-8"))

    assert case_manifest["preparation"]["source_override_used"] is True
    assert case_manifest["preparation"]["effective_kernel_source"] == str(candidate_source.resolve())
    assert "--kernel-source" not in (route_dir / "run.sh").read_text(encoding="utf-8")


def test_collect_generated_route_run_writes_results_and_summary(tmp_path: Path, monkeypatch) -> None:
    prepare_root, repo_root, asset_root = _write_prepared_case(
        tmp_path,
        op="MUL_MAT",
        route_id="mul_mat_f16_f32_tiled_batched_4d",
        case_id="mul_mat_small",
    )
    descriptor_path = prepare_root.parent / "descriptors" / "MUL_MAT" / "case.json"
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    descriptor["configs"] = {
        "@shape.mul_mat_f16.rows": "16",
        "@shape.mul_mat_f16.cols": "16",
        "@shape.mul_mat_f16.k": "4",
    }
    descriptor_path.write_text(json.dumps(descriptor, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_root = tmp_path / "benchmarks"

    def fake_workbench(*, case, run_dir, repo_root, loom_link, kernel_source_override=None):
        run_dir.mkdir(parents=True, exist_ok=True)
        workbench = run_dir / "benchmark.loom"
        workbench.write_text("check.benchmark<@case> @bench\n", encoding="utf-8")
        return (
            workbench,
            "@bench_mul_mat",
            {
                "descriptor_kernel_source": case.kernel_source,
                "descriptor_kernel_source_hash": "descriptor-hash",
                "effective_kernel_source": str(kernel_source_override or case.kernel_source),
                "effective_kernel_source_hash": "effective-hash",
                "source_override_used": kernel_source_override is not None,
            },
        )

    monkeypatch.setattr(materialize, "_write_descriptor_workbench", fake_workbench)
    assert (
        materialize.command_generate_scripts(
            SimpleNamespace(
                prepare_root=prepare_root,
                repo_root=repo_root,
                asset_root=asset_root,
                op="MUL_MAT",
                route_id=None,
                implementation_id=None,
                kernel_source=None,
                root=None,
                case_id=None,
                no_dedupe=False,
                output_root=output_root,
                tool_dir=None,
                benchmark_runner="/tools/iree-benchmark-loom",
                benchmark_device="amdgpu",
                benchmark_measure="dispatch_complete",
                loom_link="/tools/loom-link",
                python_executable=sys.executable,
            )
        )
        == 0
    )

    route_dir = output_root / "catalog" / "v2" / "MUL_MAT" / "mul_mat_f16_f32_tiled_batched_4d"
    route_manifest = json.loads((route_dir / "manifest.json").read_text(encoding="utf-8"))
    case_entry = route_manifest["cases"][0]
    run_dir = route_dir / "runs" / "manual"
    run_case_dir = run_dir / "cases" / case_entry["run_case_dir_name"]
    _write_tool_benchmark_result(run_case_dir / "benchmark-results.jsonl", mean_ns=20_000)
    (run_case_dir / "returncode.txt").write_text("0\n", encoding="utf-8")
    (run_case_dir / "command.txt").write_text("/tools/iree-benchmark-loom\nbenchmark.loom\n", encoding="utf-8")

    status = collect.command_collect(
        SimpleNamespace(
            manifest=route_dir / "manifest.json",
            run_dir=run_dir,
            output=run_dir / "summary.json",
            results=run_dir / "results.jsonl",
            markdown=run_dir / "summary.md",
            database=None,
            run_name=None,
            candidate_name="baseline",
        )
    )

    rows = [json.loads(line) for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    markdown = (run_dir / "summary.md").read_text(encoding="utf-8")

    assert status == 0
    assert rows[0]["schema"] == common.RESULT_SCHEMA
    assert rows[0]["estimated_flops"] == 2048
    assert rows[0]["status"] == "pass"
    assert summary["passed_count"] == 1
    assert summary["total_estimated_flops"] == 2048
    assert "| `mul_mat_small` | `pass` |" in markdown
