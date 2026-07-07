from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ggml_hrx_kernel_bench.materialized_assets import (
    materialize_asset_root,
    write_active_asset_root_metadata,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize the runtime asset tree.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    asset_root = materialize_asset_root(args.output, force=args.force)
    if args.metadata_output is not None:
        write_active_asset_root_metadata(args.metadata_output, asset_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
