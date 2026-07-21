from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    source_root = Path(__file__).resolve().parents[2] / "src"
    sys.path.insert(0, str(source_root))

    from ggml_hrx_kernel_bench.benchmarking.materialize import main as materialize_main

    return materialize_main()


if __name__ == "__main__":
    raise SystemExit(main())
