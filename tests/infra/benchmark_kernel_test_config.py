from __future__ import annotations

import argparse
import json
from pathlib import Path

from bootstrap import ROOT

from ggml_hrx_kernel_bench.benchmarking import (
    benchmark_config_payload,
    write_benchmark_payload,
)
from ggml_hrx_kernel_bench.kernel_test_config import load_config
from ggml_hrx_kernel_bench.required_tools import require_tool


def _default_artifact_dir(kernel: str) -> Path:
    return ROOT / "benchmark-artifacts" / kernel


def _default_result_output(kernel: str) -> Path:
    return ROOT / "benchmark-results" / f"{kernel}.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark all or selected cases from a kernel test config file."
    )
    parser.add_argument("config_path")
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        dest="cases",
        help="case selector or numeric case index; may be repeated",
    )
    parser.add_argument(
        "--tool-dir",
        help="optional directory containing loom-link and iree-benchmark-loom",
    )
    parser.add_argument(
        "--artifact-dir",
        help="directory where per-case benchmark artifacts are written; defaults to benchmark-artifacts/<kernel>",
    )
    parser.add_argument(
        "--output-dir",
        dest="artifact_dir_legacy",
        help="deprecated alias for --artifact-dir",
    )
    parser.add_argument(
        "--result-output",
        help="path to write the benchmark summary JSON; defaults to benchmark-results/<kernel>.json",
    )
    parser.add_argument("--target", default="gfx1100")
    parser.add_argument("--rocm-path")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup-iterations", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=10)
    args = parser.parse_args()

    config_path = Path(args.config_path).resolve()
    config_data = load_config(config_path)
    kernel = str(config_data["kernel"])
    artifact_dir_raw = args.artifact_dir or args.artifact_dir_legacy
    artifact_dir = (
        Path(artifact_dir_raw).resolve()
        if artifact_dir_raw
        else _default_artifact_dir(kernel)
    )
    payload = benchmark_config_payload(
        config_path,
        case_selectors=args.cases or None,
        tool_dir=args.tool_dir,
        target=args.target,
        rocm_path=args.rocm_path,
        iterations=args.iterations,
        warmup_iterations=args.warmup_iterations,
        max_batches=args.max_batches,
        artifact_dir=artifact_dir,
        require_tool=require_tool,
    )
    result_output = (
        Path(args.result_output).resolve()
        if args.result_output
        else _default_result_output(kernel)
    )
    write_benchmark_payload(payload, result_output)
    print(
        json.dumps(
            {
                "artifact_dir": str(artifact_dir),
                "result_output": str(result_output),
                **payload["summary"],
            },
            sort_keys=True,
        )
    )
    return 0 if payload["summary"]["failed_case_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
