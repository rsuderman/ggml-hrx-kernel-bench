from __future__ import annotations

import argparse
import json
from pathlib import Path

import bootstrap  # noqa: F401

from ggml_hrx_kernel_bench.loom_execution_descriptor import run_execution_descriptor_manifest
from ggml_hrx_kernel_bench.required_tools import (
    require_iree_run_loom_expected_buffer_tolerance,
    require_tool,
    resolve_tool,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare or execute compact Loom execution descriptors listed in a descriptor manifest."
    )
    parser.add_argument("manifest_path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runner", default="ggml-hrx-run-loom-simple")
    parser.add_argument("--tool-dir", help="optional PATH-style search list containing loom-link and iree-run-loom")
    parser.add_argument("--loom-link", type=Path)
    parser.add_argument("--iree-run-loom", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--limit", type=int)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="do not print the prepare/execute run manifest")
    args = parser.parse_args()

    loom_link = args.loom_link
    iree_run_loom = args.iree_run_loom
    if loom_link is None:
        resolved = resolve_tool("loom-link", tool_dir=args.tool_dir)
        loom_link = Path(resolved) if resolved else None
    if iree_run_loom is None:
        resolved = (
            require_tool("iree-run-loom", tool_dir=args.tool_dir)
            if args.execute
            else resolve_tool("iree-run-loom", tool_dir=args.tool_dir)
        )
        iree_run_loom = Path(resolved) if resolved else None
    if iree_run_loom is not None:
        require_iree_run_loom_expected_buffer_tolerance(tool_path=iree_run_loom)

    manifest = run_execution_descriptor_manifest(
        manifest_path=args.manifest_path,
        output_dir=args.output_dir,
        runner=args.runner,
        loom_link=loom_link,
        iree_run_loom=iree_run_loom,
        repo_root=args.repo_root,
        execute=args.execute,
        limit=args.limit,
    )
    if not args.quiet:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["failed_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
