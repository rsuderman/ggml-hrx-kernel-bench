from __future__ import annotations

import argparse

import bootstrap  # noqa: F401

from ggml_hrx_kernel_bench.required_tools import REQUIRED_TOOL_NAMES, require_tool


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate that a required external tool is available."
    )
    parser.add_argument("tool_name", choices=REQUIRED_TOOL_NAMES)
    parser.add_argument(
        "--tool-dir",
        help="optional PATH-style search list containing loom-link, loom-compile, iree-run-loom, iree-test-loom, and iree-benchmark-loom",
    )
    args = parser.parse_args()
    path = require_tool(args.tool_name, tool_dir=args.tool_dir)
    print(f"{args.tool_name}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
