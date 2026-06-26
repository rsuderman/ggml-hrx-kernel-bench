from __future__ import annotations

import argparse
from pathlib import Path

import bootstrap  # noqa: F401

from ggml_hrx_kernel_bench.benchmarking import (
    compare_results_payload,
    comparison_markdown,
    write_comparison_outputs,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two benchmark result JSON files.")
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--output", help="path to write comparison JSON")
    parser.add_argument(
        "--markdown-output",
        help="path to write human-readable markdown comparison",
    )
    args = parser.parse_args()

    payload = compare_results_payload(
        Path(args.baseline).resolve(),
        Path(args.candidate).resolve(),
    )
    output = Path(args.output).resolve() if args.output else None
    markdown_output = Path(args.markdown_output).resolve() if args.markdown_output else None
    write_comparison_outputs(
        payload,
        output_json=output,
        markdown_output=markdown_output,
    )
    print(comparison_markdown(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
