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
    args = parser.parse_args()

    path = _resolve_config_path(args.config_path)
    data = load_config(path)
    validate_config(data)
    print(f"validated {path} with {len(data['cases'])} case(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
