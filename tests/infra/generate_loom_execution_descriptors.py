from __future__ import annotations

import argparse
import json
from pathlib import Path

import bootstrap  # noqa: F401

from ggml_hrx_kernel_bench.loom_execution_descriptor import write_generated_execution_descriptors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate compact Loom execution descriptors from generated kernel tests."
    )
    parser.add_argument("manifest_path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--kernel-dir", type=Path, default=Path("kernels/v2"))
    parser.add_argument("--routing-dir", type=Path, default=Path("catalog/v2"))
    parser.add_argument("--target", default="gfx1100")
    parser.add_argument("--max-elements", type=int, default=65536)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--kernel", action="append", help="only emit descriptors for this generated kernel family; may repeat")
    parser.add_argument("--route-id", action="append", help="only emit descriptors for this route id; may repeat")
    parser.add_argument(
        "--case-index",
        action="append",
        type=int,
        dest="case_indices",
        metavar="INDEX",
        help="zero-based case index to emit from each selected config; may repeat",
    )
    parser.add_argument("--quiet", action="store_true", help="do not print the generated descriptor manifest")
    args = parser.parse_args()

    manifest = write_generated_execution_descriptors(
        manifest_path=args.manifest_path.resolve(),
        output_dir=args.output_dir.resolve(),
        kernel_dir=args.kernel_dir,
        routing_dir=args.routing_dir,
        target=args.target,
        max_elements=args.max_elements,
        limit=args.limit,
        kernels=set(args.kernel) if args.kernel else None,
        route_ids=set(args.route_id) if args.route_id else None,
        case_indices=args.case_indices,
    )
    if not args.quiet:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
