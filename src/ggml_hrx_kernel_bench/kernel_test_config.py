from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED_TOP_LEVEL_KEYS = {"kernel", "params", "cases"}
ROUTE_EXECUTION_ABI_SCHEMA = "ggml_hrx_kernel_bench.route_execution_abi.v1"
OPTIONAL_TOP_LEVEL_KEYS = {"route_id", "execution_abi"}
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
    if "execution_abi" in data:
        validate_execution_abi(data["execution_abi"])
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


def validate_execution_abi(data: object) -> None:
    expect(isinstance(data, dict), "execution_abi must be a JSON object")
    expect(data.get("schema") == ROUTE_EXECUTION_ABI_SCHEMA, f"execution_abi.schema must be {ROUTE_EXECUTION_ABI_SCHEMA!r}")
    if "route_id" in data:
        expect(
            isinstance(data["route_id"], str) and data["route_id"],
            "execution_abi.route_id must be a non-empty string",
        )
    entries = data.get("entries")
    expect(isinstance(entries, list) and entries, "execution_abi.entries must be a non-empty array")
    seen_positions: set[int] = set()
    for index, entry in enumerate(entries):
        expect(isinstance(entry, dict), f"execution_abi.entries[{index}] must be an object")
        position = entry.get("position")
        expect(
            isinstance(position, int) and position >= 0 and not isinstance(position, bool),
            f"execution_abi.entries[{index}].position must be a non-negative integer",
        )
        expect(position not in seen_positions, f"execution_abi.entries[{index}].position duplicates {position}")
        seen_positions.add(position)
        role = entry.get("role")
        expect(isinstance(role, str) and role, f"execution_abi.entries[{index}].role must be a non-empty string")
        kind = entry.get("kind")
        expect(kind in ("input", "output", "scalar"), f"execution_abi.entries[{index}].kind must be input, output, or scalar")
        dtype = entry.get("dtype")
        expect(isinstance(dtype, str) and dtype, f"execution_abi.entries[{index}].dtype must be a non-empty string")
        if kind == "scalar":
            value = entry.get("value")
            expect(
                isinstance(value, (int, float, str)) and not isinstance(value, bool) and str(value),
                f"execution_abi.entries[{index}].value must be a scalar value",
            )
        else:
            fixture = entry.get("fixture")
            expect(
                isinstance(fixture, str) and fixture,
                f"execution_abi.entries[{index}].fixture must be a non-empty string",
            )
        if kind == "output":
            expected = entry.get("expect")
            expect(isinstance(expected, dict), f"execution_abi.entries[{index}].expect must be an object")
            expect(expected.get("mode") == "close", f"execution_abi.entries[{index}].expect.mode must be close")
            expected_fixture = expected.get("fixture")
            expect(
                isinstance(expected_fixture, str) and expected_fixture,
                f"execution_abi.entries[{index}].expect.fixture must be a non-empty string",
            )
