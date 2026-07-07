from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .models import ConcreteTensor, V2Route


def contiguous_strides(dimensions: tuple[int, ...]) -> tuple[int, ...]:
    stride = 1
    strides: list[int] = []
    for size in dimensions:
        strides.append(stride)
        stride *= int(size)
    return tuple(strides)


def inverse_permutation(permutation: tuple[int, ...]) -> tuple[int, ...] | None:
    if tuple(sorted(permutation)) != tuple(range(len(permutation))):
        return None
    inverse = [0] * len(permutation)
    for index, value in enumerate(permutation):
        inverse[int(value)] = int(index)
    return tuple(inverse)


def chain_permutations(first: tuple[int, ...], second: tuple[int, ...]) -> tuple[int, ...] | None:
    if len(first) != len(second):
        return None
    if tuple(sorted(first)) != tuple(range(len(first))):
        return None
    if tuple(sorted(second)) != tuple(range(len(second))):
        return None
    return tuple(int(second[index]) for index in first)


def permuted_contiguous_strides(
    dimensions: tuple[int, ...],
    permutation: tuple[int, ...],
) -> tuple[int, ...] | None:
    if len(dimensions) != len(permutation):
        return None
    if tuple(sorted(permutation)) != tuple(range(len(permutation))):
        return None
    base_extents = [int(dimensions[axis]) for axis in permutation]
    base_strides = contiguous_strides(tuple(base_extents))
    logical_strides = [0] * len(permutation)
    for base_axis, logical_axis in enumerate(permutation):
        logical_strides[int(logical_axis)] = int(base_strides[base_axis])
    return tuple(logical_strides)


@dataclass(frozen=True)
class EncodedRouteShape:
    items: tuple[tuple[str, int], ...]

    @property
    def params(self) -> list[str]:
        return [name for name, _ in self.items]

    @property
    def values(self) -> list[int]:
        return [value for _, value in self.items]

    def as_dict(self) -> dict[str, int]:
        return {name: value for name, value in self.items}


def decode_shape(params: list[str], values: list[int]) -> dict[str, int]:
    if len(params) != len(values):
        raise RuntimeError("params and case values must have the same length")
    decoded: dict[str, int] = {}
    for name, value in zip(params, values, strict=True):
        key = str(name)
        if key in decoded:
            raise RuntimeError(f"duplicate shape parameter: {key}")
        decoded[key] = int(value)
    return decoded


def encode_route_shape(route: V2Route, tensors: Mapping[str, ConcreteTensor]) -> EncodedRouteShape:
    if not route.tensors:
        return EncodedRouteShape(items=())
    anchor_name = "dst" if "dst" in tensors else next(iter(route.tensors))
    anchor = tensors[anchor_name]
    items: list[tuple[str, int]] = [
        (dimension.name, int(dimension.size))
        for dimension in anchor.dimensions
    ]
    anchor_sizes = {dimension.name: int(dimension.size) for dimension in anchor.dimensions}
    for tensor_name, tensor in tensors.items():
        sizes = [int(dimension.size) for dimension in tensor.dimensions]
        default_strides = contiguous_strides(tuple(sizes))
        for index, dimension in enumerate(tensor.dimensions):
            base_key = f"{tensor_name}_{dimension.name}"
            if int(dimension.size) != int(anchor_sizes[dimension.name]):
                items.append((base_key, int(dimension.size)))
            if int(dimension.stride) != int(default_strides[index]):
                items.append((f"{base_key}_stride", int(dimension.stride)))
        descriptor = route.tensors.get(tensor_name)
        if descriptor is None or descriptor.permutation_capture is None or tensor.permutation is None:
            continue
        permutation = tuple(int(axis) for axis in tensor.permutation)
        if permutation == tuple(range(len(permutation))):
            continue
        for index, axis in enumerate(permutation):
            items.append((f"{tensor_name}_perm{index}", int(axis)))
    return EncodedRouteShape(items=tuple(items))
