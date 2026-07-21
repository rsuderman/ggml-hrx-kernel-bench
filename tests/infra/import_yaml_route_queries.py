from __future__ import annotations

import argparse
import os
from pathlib import Path

import bootstrap  # noqa: F401

from ggml_hrx_kernel_bench.yaml_route_import import materialize_yaml_route_queries


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import descriptor YAML cases that match a route as RouteQuery JSONL."
    )
    parser.add_argument("output_dir")
    parser.add_argument("--yaml", action="append", required=True, help="descriptor YAML input path")
    parser.add_argument("--routing-dir", type=Path, required=True)
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

    materialize_yaml_route_queries(
        [Path(raw_path).resolve() for raw_path in args.yaml],
        output_dir=Path(args.output_dir).resolve(),
        routing_dir=args.routing_dir.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
