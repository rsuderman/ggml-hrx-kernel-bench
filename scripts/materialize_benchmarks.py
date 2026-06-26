from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ggml_hrx_kernel_bench.benchmarking import (  # noqa: E402
    collect_config_paths,
    materialize_benchmark_set,
)
from ggml_hrx_kernel_bench.required_tools import require_tool  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Materialize benchmark results for one or more kernel benchmark configs."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="benchmark config file(s) or directories containing configs",
    )
    parser.add_argument(
        "--tool-dir",
        help="optional directory containing loom-link and iree-benchmark-loom",
    )
    parser.add_argument(
        "--artifact-root",
        help="directory where per-kernel benchmark artifacts are written",
    )
    parser.add_argument(
        "--results-dir",
        help="directory where benchmark result JSON files are written",
    )
    parser.add_argument(
        "--compare-results-dir",
        help="optional directory of existing benchmark result JSON files to compare against",
    )
    parser.add_argument(
        "--comparison-dir",
        help="directory where comparison JSON/markdown files are written",
    )
    parser.add_argument("--target", default="gfx1100")
    parser.add_argument("--rocm-path")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup-iterations", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=10)
    args = parser.parse_args()

    config_paths = collect_config_paths(args.inputs)
    artifact_root = (
        Path(args.artifact_root).resolve()
        if args.artifact_root
        else ROOT / "benchmark-artifacts" / "materialized"
    )
    results_dir = (
        Path(args.results_dir).resolve()
        if args.results_dir
        else ROOT / "benchmark-results" / "materialized"
    )
    comparison_dir = (
        Path(args.comparison_dir).resolve()
        if args.comparison_dir
        else results_dir / "comparisons"
    )
    compare_results_dir = (
        Path(args.compare_results_dir).resolve()
        if args.compare_results_dir
        else None
    )

    manifest, had_failures = materialize_benchmark_set(
        config_paths,
        tool_dir=args.tool_dir,
        target=args.target,
        rocm_path=args.rocm_path,
        iterations=args.iterations,
        warmup_iterations=args.warmup_iterations,
        max_batches=args.max_batches,
        artifact_root=artifact_root,
        results_dir=results_dir,
        compare_results_dir=compare_results_dir,
        comparison_dir=comparison_dir,
        require_tool=require_tool,
    )
    manifest_path = results_dir / "benchmark-manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest_path": str(manifest_path),
                "result_count": len(manifest["results"]),
                "comparison_count": len(manifest["comparisons"]),
            },
            sort_keys=True,
        )
    )
    return 1 if had_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
