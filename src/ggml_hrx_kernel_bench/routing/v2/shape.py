from __future__ import annotations

from typing import Mapping


def _ranked_extents(shape: Mapping[str, int]) -> tuple[int, ...] | None:
    ranked: list[tuple[int, int]] = []
    for name, value in shape.items():
        key = str(name)
        if not key.startswith("d") or not key[1:].isdigit():
            continue
        ranked.append((int(key[1:]), int(value)))
    if not ranked:
        return None
    ranked.sort()
    if tuple(index for index, _ in ranked) != tuple(range(len(ranked))):
        return None
    return tuple(value for _, value in ranked)


def _contiguous_strides(extents: tuple[int, ...]) -> tuple[int, ...]:
    stride = 1
    strides: list[int] = []
    for extent in extents:
        strides.append(stride)
        stride *= int(extent)
    return tuple(strides)


def _derived_cont_shape_source(key: str, shape: Mapping[str, int]) -> int | None:
    extents = _ranked_extents(shape)
    if extents is None or len(extents) < 2 or len(extents) > 4:
        return None
    padded_extents = tuple(extents) + (1,) * (4 - len(extents))
    padded_strides = _contiguous_strides(padded_extents)
    derived = {
        "cont.d1": int(padded_extents[1] * padded_extents[2] * padded_extents[3]),
        "cont.ne1": int(padded_extents[1]),
        "cont.ne2": int(padded_extents[2]),
        "cont.src_nb1": int(padded_strides[1]),
        "cont.src_nb2": int(padded_strides[2]),
        "cont.src_nb3": int(padded_strides[3]),
    }
    return derived.get(key)


def resolve_shape_source(source: str, shape: Mapping[str, int]) -> int | None:
    if not source.startswith("shape."):
        return None
    key = source.removeprefix("shape.")
    value = shape.get(key)
    if value is not None:
        return int(value)
    derived = _derived_cont_shape_source(key, shape)
    return None if derived is None else int(derived)
