from __future__ import annotations

import argparse
from pathlib import Path

import bootstrap  # noqa: F401

from ggml_hrx_kernel_bench.kernel_test_config import load_config, validate_config


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _resolve_config_path(raw_path: str) -> Path:
    path = Path(raw_path).resolve()
    if path.is_dir():
        matches = sorted(candidate for candidate in path.glob("*.json") if candidate.is_file())
        _expect(len(matches) == 1, f"expected exactly one config JSON in {path}, saw {matches}")
        return matches[0]
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a kernel test config JSON file.")
    parser.add_argument("config_path")
    parser.add_argument(
        "--expect-case",
        help="comma-separated integer case values, for example 1,16384",
    )
    parser.add_argument(
        "--expect-route-id",
        help="exact route_id expected in the config",
    )
    args = parser.parse_args()

    path = _resolve_config_path(args.config_path)
    data = load_config(path)
    validate_config(data)

    if args.expect_case:
        expected_case = [int(part) for part in args.expect_case.split(",") if part]
        cases = data.get("cases")
        _expect(isinstance(cases, list), "cases must be a list")
        _expect(expected_case in cases, f"expected case {expected_case} in config, saw {cases}")

    if args.expect_route_id:
        route_id = data.get("route_id")
        _expect(route_id == args.expect_route_id, f"expected route_id {args.expect_route_id!r}, saw {route_id!r}")

    details: list[str] = []
    if args.expect_case:
        details.append(f"expected case {expected_case}")
    if args.expect_route_id:
        details.append(f"route_id {args.expect_route_id}")
    if details:
        print(f"validated {path} with {' and '.join(details)}")
        return 0

    print(f"validated {path} with {len(data['cases'])} case(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
