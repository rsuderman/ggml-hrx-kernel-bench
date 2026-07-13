from __future__ import annotations

import argparse
import json
from pathlib import Path

import bootstrap  # noqa: F401

from ggml_hrx_kernel_bench.harness_inventory import (
    build_harness_inventory,
    inventory_to_markdown,
    write_inventory_json,
    write_inventory_markdown,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report descriptor-vs-legacy runtime harness registration and descriptor coverage by op."
    )
    parser.add_argument("--name", required=True)
    parser.add_argument("--route-import-dir", type=Path, required=True)
    parser.add_argument("--descriptor-output-dir", type=Path, required=True)
    parser.add_argument("--descriptor-tests-cmake", type=Path)
    parser.add_argument("--legacy-runtime-tests-cmake", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args()

    inventory = build_harness_inventory(
        name=args.name,
        route_import_dir=args.route_import_dir,
        descriptor_output_dir=args.descriptor_output_dir,
        descriptor_tests_cmake=args.descriptor_tests_cmake,
        legacy_runtime_tests_cmake=args.legacy_runtime_tests_cmake,
    )
    if args.output_json:
        write_inventory_json(inventory, args.output_json)
    if args.output_md:
        write_inventory_markdown(inventory, args.output_md)
    if not args.output_json and not args.output_md:
        print(json.dumps(inventory, indent=2, sort_keys=True))
    elif args.output_md and not args.output_json:
        print(inventory_to_markdown(inventory), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
