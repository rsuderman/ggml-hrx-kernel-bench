from __future__ import annotations

import argparse
import os
from pathlib import Path

import bootstrap  # noqa: F401

from check_yaml_route_import import validate_yaml_route_import
from ggml_hrx_kernel_bench.route_query_config import materialize_route_query_configs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate grouped route configs from matched RouteQuery JSONL."
    )
    parser.add_argument("output_dir")
    parser.add_argument("--route-queries", type=Path, required=True)
    parser.add_argument("--import-metadata", type=Path, required=True)
    parser.add_argument("--routing-dir", type=Path, required=True)
    parser.add_argument("--expected-coverage", type=Path)
    parser.add_argument(
        "--tool-dir",
        help=(
            "optional PATH-style search list containing loom-link, loom-compile, "
            "ggml-hrx-run-loom, iree-test-loom, and iree-benchmark-loom"
        ),
    )
    args = parser.parse_args()

    if args.tool_dir:
        os.environ["PATH"] = f"{args.tool_dir}:{os.environ.get('PATH', '')}"

    output_dir = Path(args.output_dir).resolve()
    summary = materialize_route_query_configs(
        args.route_queries.resolve(),
        metadata_path=args.import_metadata.resolve(),
        output_dir=output_dir,
        routing_dir=args.routing_dir.resolve(),
    )
    expected_coverage_path = (
        args.expected_coverage.resolve() if args.expected_coverage is not None else None
    )
    validate_yaml_route_import(output_dir, summary, expected_coverage_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
