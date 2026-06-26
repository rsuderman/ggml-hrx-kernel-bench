from __future__ import annotations

import argparse
import json
from pathlib import Path

import bootstrap  # noqa: F401

from ggml_hrx_kernel_bench.kernel_test_config import load_config
from ggml_hrx_kernel_bench.kernel_test_config_runtime import (
    case_result,
    execute_case,
    select_case,
)
from ggml_hrx_kernel_bench.required_tools import require_tool


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
    args = parser.parse_args()

    config_path = Path(args.config_path).resolve()
    config_data = load_config(config_path)
    current_case_id, current_case_values = select_case(config_data, args.case_selector)
    output_dir = Path(args.output_dir).resolve()
    candidate, row, summary = execute_case(
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
    result = case_result(
        candidate=candidate,
        current_case_id=current_case_id,
        current_case_values=current_case_values,
        row=row,
        summary=summary,
        output_dir=output_dir,
    )
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
