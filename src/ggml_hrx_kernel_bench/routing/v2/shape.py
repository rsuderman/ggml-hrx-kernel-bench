from __future__ import annotations

from typing import Mapping


def resolve_shape_source(source: str, shape: Mapping[str, int]) -> int | None:
    if not source.startswith("shape."):
        return None
    key = source.removeprefix("shape.")
    value = shape.get(key)
    return None if value is None else int(value)
