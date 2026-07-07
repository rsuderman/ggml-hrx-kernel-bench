from __future__ import annotations

import argparse
import json
from pathlib import Path

import bootstrap  # noqa: F401

from ggml_hrx_kernel_bench.kernel_test_config import load_config
from ggml_hrx_kernel_bench.routing.api import RuntimeCaseRequest, create_router, supported_routing_versions
from ggml_hrx_kernel_bench.required_tools import require_tool


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _safe_name(value: str) -> str:
    return ''.join(ch.lower() if ch.isalnum() else '-' for ch in value).strip('-') or 'generated'


def _runtime_environment_blocked(result: dict) -> bool:
    if result.get("status") != "run_failed":
        return False
    stderr_path = result.get("stderr_path")
    if not isinstance(stderr_path, str) or not stderr_path:
        return False
    stderr_file = Path(stderr_path).resolve()
    if not stderr_file.is_file():
        return False
    stderr = stderr_file.read_text(encoding="utf-8", errors="replace")
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
    case_index: int,
    total_cases: int,
    entry_index: int,
    entry: dict,
    config_path: Path,
    case_id: str,
    case_values: list[int],
) -> None:
    label = _manifest_entry_label(entry_index, entry, config_path)
    print(
        f"Running {case_index + 1}/{total_cases}: {label} case={case_id} values={case_values}",
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
    parser.add_argument(
        "--tool-dir",
        help="optional PATH-style search list containing loom-link and iree-test-loom",
    )
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
    parser.add_argument("--kernel-dir", type=Path)
    args = parser.parse_args()

    manifest_path = Path(args.manifest_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = _load_manifest(manifest_path)
    router = create_router(
        version=args.routing_version,
        kernel_dir=args.kernel_dir.resolve() if args.kernel_dir else None,
        routing_dir=args.routing_dir.resolve() if args.routing_dir else None,
    )

    manifest_entries: list[tuple[int, dict, Path, dict, list[tuple[str, list[int]]]]] = []
    for index, entry in enumerate(payload["entries"]):
        _expect(isinstance(entry, dict), f"entries[{index}] must be an object")
        config_path = Path(str(entry.get("config_path") or "")).resolve()
        _expect(config_path.is_file(), f"missing generated config {config_path}")
        config_data = load_config(config_path)
        selected_cases = router.select_cases(config_data, None)
        manifest_entries.append((index, entry, config_path, config_data, selected_cases))

    total_cases = sum(len(selected_cases) for _, _, _, _, selected_cases in manifest_entries)
    case_index = 0

    for entry_index, entry, config_path, config_data, selected_cases in manifest_entries:
        config_output_dir = output_dir / f"{entry_index:03d}-{_safe_name(config_path.stem)}"
        for current_case_id, current_case_values in selected_cases:
            _print_running_case(
                case_index=case_index,
                total_cases=total_cases,
                entry_index=entry_index,
                entry=entry,
                config_path=config_path,
                case_id=current_case_id,
                case_values=current_case_values,
            )
            case_output_dir = config_output_dir / f"{case_index:03d}-{_safe_name(current_case_id)}"
            execution = router.execute_case(
                RuntimeCaseRequest(
                    kernel_dir=router.context.kernel_dir,
                    routing_dir=router.context.routing_dir,
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
            case_index += 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
