from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from bootstrap import CATALOG_DIR, ROOT

from ggml_hrx_kernel_bench.grouped_yaml_import import materialize_grouped_yaml
from ggml_hrx_kernel_bench.routing.api import supported_routing_versions


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import grouped llama.cpp workload YAML into compact benchmark configs and unmapped backlog artifacts."
    )
    parser.add_argument("yaml_path")
    parser.add_argument(
        "--output-dir",
        help="output directory for import-coverage.json, generated-kernel-tests.json, imported-workload.json, unmapped.json, import-summary.md, and generated-import-configs/",
    )
    parser.add_argument(
        "--split-by-op",
        action="store_true",
        help="emit one import bundle per operation under <output-dir>/ops/<op>/",
    )
    parser.add_argument(
        "--tool-dir",
        help="optional PATH-style search list containing loom-link, loom-compile, iree-test-loom, and iree-benchmark-loom",
    )
    parser.add_argument(
        "--routing-version",
        choices=supported_routing_versions(),
        default="v1",
    )
    parser.add_argument("--routing-dir", type=Path, default=CATALOG_DIR)
    args = parser.parse_args()

    if args.tool_dir:
        os.environ["PATH"] = f"{args.tool_dir}:{os.environ.get('PATH', '')}"

    yaml_path = Path(args.yaml_path).resolve()
    default_name = f"{yaml_path.stem}-by-op" if args.split_by_op else yaml_path.stem
    output_dir = Path(args.output_dir).resolve() if args.output_dir else ROOT / "import-results" / default_name
    payload = materialize_grouped_yaml(
        yaml_path,
        output_dir=output_dir,
        routing_version=args.routing_version,
        routing_dir=args.routing_dir.resolve(),
        split_by_op=args.split_by_op,
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
