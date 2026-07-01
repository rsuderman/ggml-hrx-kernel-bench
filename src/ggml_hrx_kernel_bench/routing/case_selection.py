from __future__ import annotations

from typing import Any

from ..kernel_test_config import expect


def case_id(params: list[str], values: list[int]) -> str:
    return "_".join(f"{name}{value}" for name, value in zip(params, values, strict=True))


def list_cases(config: dict[str, Any]) -> list[tuple[str, list[int]]]:
    params = list(config["params"])
    return [(case_id(params, list(values)), list(values)) for values in config["cases"]]


def select_case(config: dict[str, Any], selector: str) -> tuple[str, list[int]]:
    cases = list_cases(config)
    if selector.isdigit():
        index = int(selector)
        expect(0 <= index < len(cases), f"case index out of range: {index}")
        return cases[index]
    for current_case_id, values in cases:
        if current_case_id == selector:
            return current_case_id, values
    raise RuntimeError(f"case not found in config: {selector}")


def select_cases(
    config: dict[str, Any], selectors: list[str] | None
) -> list[tuple[str, list[int]]]:
    if not selectors:
        return list_cases(config)
    return [select_case(config, selector) for selector in selectors]
