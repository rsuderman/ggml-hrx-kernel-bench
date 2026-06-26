from __future__ import annotations

import argparse
from pathlib import Path

import bootstrap  # noqa: F401

from ggml_hrx_kernel_bench.kernel_test_config import load_config, validate_config


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a kernel test config JSON file.")
    parser.add_argument("config_path")
    parser.add_argument(
        "--expect-case",
        help="comma-separated integer case values, for example 1,16384",
    )
    args = parser.parse_args()

    path = Path(args.config_path).resolve()
    data = load_config(path)
    validate_config(data)

    if args.expect_case:
        expected_case = [int(part) for part in args.expect_case.split(",") if part]
        cases = data.get("cases")
        _expect(isinstance(cases, list), "cases must be a list")
        _expect(expected_case in cases, f"expected case {expected_case} in config, saw {cases}")
        print(f"validated {path} with expected case {expected_case}")
        return 0

    print(f"validated {path} with {len(data['cases'])} case(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
