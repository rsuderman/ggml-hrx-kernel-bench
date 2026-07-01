from __future__ import annotations

import argparse
import json
from pathlib import Path

import bootstrap  # noqa: F401

from ggml_hrx_kernel_bench.kernel_test_config import load_config
from ggml_hrx_kernel_bench.routing.api import DEFAULT_KERNEL_DIR, RuntimeCaseRequest, create_router, default_routing_dir, supported_routing_versions
from ggml_hrx_kernel_bench.required_tools import require_tool


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _safe_name(value: str) -> str:
    return ''.join(ch.lower() if ch.isalnum() else '-' for ch in value).strip('-') or 'generated'


def _runtime_environment_blocked(result: dict) -> bool:
    if result.get("status") != "run_failed":
        return False
    results_path = result.get("results_path")
    if not isinstance(results_path, str) or not results_path:
        return False
    run_dir = Path(results_path).resolve().parent
    stderr_path = run_dir / "benchmark.stderr.txt"
    if not stderr_path.is_file():
        return False
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
    markers = (
        "HSA_STATUS_ERROR_OUT_OF_RESOURCES",
        "creating driver for device 'amdgpu'",
        "/dev/kfd",
    )
    return any(marker in stderr for marker in markers)


def _load_manifest(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _expect(isinstance(payload, dict), "manifest must be a JSON object")
    entries = payload.get("entries")
    _expect(isinstance(entries, list), "manifest entries must be a list")
    return payload


def _manifest_entry_label(index: int, entry: dict, config_path: Path) -> str:
    kernel = str(entry.get("kernel") or "")
    route_id = str(entry.get("route_id") or "")
    parts = [f"[{index + 1}]"]
    if kernel:
        parts.append(kernel)
    if route_id:
        parts.append(f"route={route_id}")
    parts.append(config_path.name)
    return " ".join(parts)


def _print_running_case(
    *,
    index: int,
    total: int,
    entry: dict,
    config_path: Path,
    case_id: str,
    case_values: list[int],
) -> None:
    label = _manifest_entry_label(index, entry, config_path)
    print(
        f"Running {index + 1}/{total}: {label} case={case_id} values={case_values}",
        flush=True,
    )


def _print_runtime_blocked(*, manifest_path: Path, result: dict) -> None:
    print("Generated kernel runtime test skipped: runtime environment unavailable.")
    print(f"Manifest: {manifest_path}")
    print(f"Config: {result.get('config_path')}")
    print(f"Case: {result.get('case_id')}")
    print(f"Results: {result.get('results_path')}")
    print(
        json.dumps(
            {
                "manifest_path": str(manifest_path),
                "result": result,
                "status": "skipped_runtime_unavailable",
            },
            indent=2,
            sort_keys=True,
        )
    )


def _print_failure(
    *,
    manifest_path: Path,
    result: dict,
    reason: str,
) -> None:
    print("Generated kernel runtime test failed.")
    print(f"Reason: {reason}")
    print(f"Manifest: {manifest_path}")
    print(f"Config: {result.get('config_path')}")
    print(f"Case: {result.get('case_id')}")
    print(f"Candidate: {result.get('candidate_id')}")
    print(f"Results: {result.get('results_path')}")
    artifact_bundle_dir = result.get("artifact_bundle_dir")
    if artifact_bundle_dir:
        print(f"Artifacts: {artifact_bundle_dir}")
    failure = result.get("failure")
    correctness = result.get("correctness")
    if failure:
        print(f"Failure detail: {failure}")
    if correctness is not None:
        print(f"Correctness: {correctness}")
    print("Result:")
    print(json.dumps(result, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run generated kernel tests listed in a manifest.")
    parser.add_argument("manifest_path")
    parser.add_argument("--case-selector", default="0")
    parser.add_argument("--tool-dir", help="optional directory containing loom-link and iree-benchmark-loom")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target", default="gfx1100")
    parser.add_argument("--rocm-path")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--warmup-iterations", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=1)
    parser.add_argument(
        "--routing-version",
        choices=supported_routing_versions(),
        default="v1",
    )
    parser.add_argument("--routing-dir", type=Path)
    parser.add_argument("--kernel-dir", type=Path, default=DEFAULT_KERNEL_DIR)
    args = parser.parse_args()
    if args.routing_dir is None:
        args.routing_dir = default_routing_dir(args.routing_version)

    manifest_path = Path(args.manifest_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = _load_manifest(manifest_path)
    total_entries = len(payload["entries"])
    router = create_router(
        version=args.routing_version,
        kernel_dir=args.kernel_dir.resolve(),
        routing_dir=args.routing_dir.resolve(),
    )

    results = []
    for index, entry in enumerate(payload["entries"]):
        _expect(isinstance(entry, dict), f"entries[{index}] must be an object")
        config_path = Path(str(entry.get("config_path") or "")).resolve()
        _expect(config_path.is_file(), f"missing generated config {config_path}")
        config_data = load_config(config_path)
        current_case_id, current_case_values = router.select_case(
            config_data, args.case_selector
        )
        _print_running_case(
            index=index,
            total=total_entries,
            entry=entry,
            config_path=config_path,
            case_id=current_case_id,
            case_values=current_case_values,
        )
        case_output_dir = output_dir / f"{index:03d}-{_safe_name(config_path.stem)}"
        execution = router.execute_case(
            RuntimeCaseRequest(
                kernel_dir=args.kernel_dir.resolve(),
                routing_dir=args.routing_dir.resolve(),
                config_data=config_data,
                current_case_id=current_case_id,
                current_case_values=current_case_values,
                tool_dir=args.tool_dir,
                target=args.target,
                rocm_path=args.rocm_path,
                iterations=args.iterations,
                warmup_iterations=args.warmup_iterations,
                max_batches=args.max_batches,
                output_dir=case_output_dir,
                require_tool=require_tool,
            )
        )
        result = router.case_result(execution)
        result["config_path"] = str(config_path)
        results.append(result)
        if _runtime_environment_blocked(result):
            _print_runtime_blocked(manifest_path=manifest_path, result=result)
            return 125
        if result["status"] != "ran":
            _print_failure(
                manifest_path=manifest_path,
                result=result,
                reason="kernel run did not complete successfully",
            )
            raise RuntimeError("kernel run failed")
        if not result["correctness_ok"]:
            _print_failure(
                manifest_path=manifest_path,
                result=result,
                reason="kernel correctness check failed",
            )
            raise RuntimeError("kernel correctness check failed")

    print(json.dumps({"manifest_path": str(manifest_path), "result_count": len(results), "results": results}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
