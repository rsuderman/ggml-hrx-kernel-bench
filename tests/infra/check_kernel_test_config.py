from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_TOP_LEVEL_KEYS = {"kernel", "params", "cases"}
ALLOWED_TOP_LEVEL_KEYS = REQUIRED_TOP_LEVEL_KEYS


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _validate_case(case: object, *, context: str, param_count: int) -> None:
    _expect(isinstance(case, list), f"{context} must be an array")
    _expect(len(case) == param_count, f"{context} must have exactly {param_count} value(s)")
    for index, value in enumerate(case):
        _expect(isinstance(value, int) and value > 0, f"{context}[{index}] must be a positive integer")


def validate_config(data: object) -> None:
    _expect(isinstance(data, dict), "config must be a JSON object")
    extra_keys = set(data) - ALLOWED_TOP_LEVEL_KEYS
    _expect(not extra_keys, f"unexpected top-level keys: {sorted(extra_keys)}")
    missing_keys = REQUIRED_TOP_LEVEL_KEYS - set(data)
    _expect(not missing_keys, f"missing top-level keys: {sorted(missing_keys)}")
    _expect(isinstance(data["kernel"], str) and data["kernel"], "kernel must be a non-empty string")
    _expect(isinstance(data["params"], list) and data["params"], "params must be a non-empty array")
    _expect(isinstance(data["cases"], list) and data["cases"], "cases must be a non-empty array")

    seen_params: set[str] = set()
    for index, name in enumerate(data["params"]):
        _expect(isinstance(name, str) and name, f"params[{index}] must be a non-empty string")
        _expect(name not in seen_params, f"params[{index}] duplicates {name!r}")
        seen_params.add(name)

    param_count = len(data["params"])
    for index, case in enumerate(data["cases"]):
        _validate_case(case, context=f"cases[{index}]", param_count=param_count)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a kernel test config JSON file.")
    parser.add_argument("config_path")
    args = parser.parse_args()

    path = Path(args.config_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    validate_config(data)
    print(f"validated {path} with {len(data['cases'])} case(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
