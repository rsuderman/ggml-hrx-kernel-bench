from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED_TOP_LEVEL_KEYS = {"kernel", "params", "cases"}
OPTIONAL_TOP_LEVEL_KEYS = {"route_id"}
ALLOWED_TOP_LEVEL_KEYS = REQUIRED_TOP_LEVEL_KEYS | OPTIONAL_TOP_LEVEL_KEYS


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_config(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    expect(isinstance(data, dict), "config must be a JSON object")
    return data


def validate_case(case: object, *, context: str, param_count: int) -> None:
    expect(isinstance(case, list), f"{context} must be an array")
    expect(len(case) == param_count, f"{context} must have exactly {param_count} value(s)")
    for index, value in enumerate(case):
        expect(
            isinstance(value, int) and value >= 0,
            f"{context}[{index}] must be a non-negative integer",
        )


def validate_config(data: object) -> None:
    expect(isinstance(data, dict), "config must be a JSON object")
    extra_keys = set(data) - ALLOWED_TOP_LEVEL_KEYS
    expect(not extra_keys, f"unexpected top-level keys: {sorted(extra_keys)}")
    missing_keys = REQUIRED_TOP_LEVEL_KEYS - set(data)
    expect(not missing_keys, f"missing top-level keys: {sorted(missing_keys)}")
    expect(isinstance(data["kernel"], str) and data["kernel"], "kernel must be a non-empty string")
    if "route_id" in data:
        expect(isinstance(data["route_id"], str) and data["route_id"], "route_id must be a non-empty string")
    expect(isinstance(data["params"], list) and data["params"], "params must be a non-empty array")
    expect(isinstance(data["cases"], list) and data["cases"], "cases must be a non-empty array")

    seen_params: set[str] = set()
    for index, name in enumerate(data["params"]):
        expect(isinstance(name, str) and name, f"params[{index}] must be a non-empty string")
        expect(name not in seen_params, f"params[{index}] duplicates {name!r}")
        seen_params.add(name)

    param_count = len(data["params"])
    for index, case in enumerate(data["cases"]):
        validate_case(case, context=f"cases[{index}]", param_count=param_count)
