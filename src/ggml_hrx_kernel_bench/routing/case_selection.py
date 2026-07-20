from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..kernel_test_config import expect


def case_id(params: list[str], values: list[int]) -> str:
    return "_".join(f"{name}{value}" for name, value in zip(params, values, strict=True))


def list_cases(config: dict[str, Any]) -> list[tuple[str, list[int]]]:
    return [
        (current_case_id, values)
        for _, current_case_id, values in list_indexed_cases(config)
    ]


def list_indexed_cases(config: dict[str, Any]) -> list[tuple[int, str, list[int]]]:
    params = list(config["params"])
    return [
        (index, case_id(params, list(values)), list(values))
        for index, values in enumerate(config["cases"])
    ]


def select_case(config: dict[str, Any], index: int) -> tuple[str, list[int]]:
    _, current_case_id, values = select_indexed_case(config, index)
    return current_case_id, values


def select_indexed_case(config: dict[str, Any], index: int) -> tuple[int, str, list[int]]:
    cases = list_indexed_cases(config)
    expect(
        isinstance(index, int) and not isinstance(index, bool),
        f"case index must be an integer: {index!r}",
    )
    expect(0 <= index < len(cases), f"case index out of range: {index}")
    return cases[index]


def select_cases(
    config: dict[str, Any], indices: Sequence[int] | None
) -> list[tuple[str, list[int]]]:
    return [
        (current_case_id, values)
        for _, current_case_id, values in select_indexed_cases(config, indices)
    ]


def select_indexed_cases(
    config: dict[str, Any], indices: Sequence[int] | None
) -> list[tuple[int, str, list[int]]]:
    if not indices:
        return list_indexed_cases(config)
    return [select_indexed_case(config, index) for index in indices]
