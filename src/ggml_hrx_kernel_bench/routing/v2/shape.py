from __future__ import annotations

from typing import Mapping


def aliases_for_dimension(name: str) -> tuple[str, ...]:
    if name == "ncols":
        return ("ncols", "cols")
    if name == "cols":
        return ("cols", "ncols")
    if name == "nrows":
        return ("nrows", "rows")
    if name == "rows":
        return ("rows", "nrows")
    return (name,)


def normalize_shape(shape: Mapping[str, int]) -> dict[str, int]:
    normalized = {str(key): int(value) for key, value in shape.items()}
    if "ncols" in normalized and "cols" not in normalized:
        normalized["cols"] = normalized["ncols"]
    if "cols" in normalized and "ncols" not in normalized:
        normalized["ncols"] = normalized["cols"]
    if "nrows" in normalized and "rows" not in normalized:
        normalized["rows"] = normalized["nrows"]
    if "rows" in normalized and "nrows" not in normalized:
        normalized["nrows"] = normalized["rows"]
    return normalized


def validate_shape_aliases(shape: Mapping[str, int], *, context: str) -> None:
    normalized = normalize_shape(shape)
    if "ncols" in normalized and "cols" in normalized and normalized["ncols"] != normalized["cols"]:
        raise RuntimeError(f"ncols and cols must match for {context}")
    if "nrows" in normalized and "rows" in normalized and normalized["nrows"] != normalized["rows"]:
        raise RuntimeError(f"nrows and rows must match for {context}")


def resolve_shape_source(source: str, shape: Mapping[str, int]) -> int | None:
    if not source.startswith("shape."):
        return None
    key = source.removeprefix("shape.")
    normalized = normalize_shape(shape)
    for alias in aliases_for_dimension(key):
        if alias in normalized:
            return int(normalized[alias])
    return None
