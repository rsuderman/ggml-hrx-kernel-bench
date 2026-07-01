from __future__ import annotations

import argparse
import json
from pathlib import Path

import bootstrap  # noqa: F401

from ggml_hrx_kernel_bench.kernel_test_config import load_config
from ggml_hrx_kernel_bench.routing.api import RuntimeCaseRequest, create_router, supported_routing_versions
from ggml_hrx_kernel_bench.required_tools import require_tool


def _resolve_config_path(raw_path: str) -> Path:
    path = Path(raw_path).resolve()
    if path.is_dir():
        matches = sorted(candidate for candidate in path.glob("*.json") if candidate.is_file())
        if len(matches) != 1:
            raise RuntimeError(f"expected exactly one config JSON in {path}, saw {matches}")
        return matches[0]
    return path


def _print_running_case(
    *,
    config_path: Path,
    config_data: dict,
    current_case_id: str,
    current_case_values: list[int],
) -> None:
    kernel = str(config_data.get("kernel") or "")
    route_id = str(config_data.get("route_id") or "")
    parts = ["Running kernel test:"]
    if kernel:
        parts.append(kernel)
    if route_id:
        parts.append(f"route={route_id}")
    parts.append(config_path.name)
    parts.append(f"case={current_case_id}")
    parts.append(f"values={current_case_values}")
    print(" ".join(parts), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a kernel correctness test from a kernel-test-config file."
    )
    parser.add_argument("config_path")
    parser.add_argument("case_selector")
    parser.add_argument(
        "--tool-dir",
        help="optional directory containing loom-link and iree-benchmark-loom",
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

    config_path = _resolve_config_path(args.config_path)
    config_data = load_config(config_path)
    router = create_router(
        version=args.routing_version,
        kernel_dir=args.kernel_dir.resolve() if args.kernel_dir else None,
        routing_dir=args.routing_dir.resolve() if args.routing_dir else None,
    )
    current_case_id, current_case_values = router.select_case(
        config_data, args.case_selector
    )
    _print_running_case(
        config_path=config_path,
        config_data=config_data,
        current_case_id=current_case_id,
        current_case_values=current_case_values,
    )
    output_dir = Path(args.output_dir).resolve()
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
            output_dir=output_dir,
            require_tool=require_tool,
        )
    )
    result = router.case_result(execution)
    if result["status"] != "ran":
        raise RuntimeError(f"kernel run failed: {json.dumps(result, sort_keys=True)}")
    if not result["correctness_ok"]:
        raise RuntimeError(
            f"kernel correctness check failed: {json.dumps(result, sort_keys=True)}"
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
