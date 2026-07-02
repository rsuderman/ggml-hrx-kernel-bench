from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping


def _freeze_value(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({str(key): _freeze_value(inner) for key, inner in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_value(entry) for entry in value)
    if isinstance(value, tuple):
        return tuple(_freeze_value(entry) for entry in value)
    return value


def _freeze_mapping(mapping: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType({str(key): _freeze_value(value) for key, value in mapping.items()})


@dataclass(frozen=True)
class StrideDescriptor:
    value: int | None = None
    dimension: str | None = None
    product: tuple[str, ...] = ()


@dataclass(frozen=True)
class DimensionBounds:
    min: int | None
    max: int | None


@dataclass(frozen=True)
class ConstraintCheck:
    identifier: str
    min: int | None = None
    max: int | None = None
    value: int | None = None
    dimension: str | None = None
    product: tuple[str, ...] = ()


@dataclass(frozen=True)
class RouteConstraints:
    sizes: Mapping[str, DimensionBounds]
    strides: Mapping[str, StrideDescriptor]
    checks: tuple[ConstraintCheck, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "sizes", _freeze_mapping(self.sizes))
        object.__setattr__(self, "strides", _freeze_mapping(self.strides))


@dataclass(frozen=True)
class TensorDimensionDescriptor:
    name: str


@dataclass(frozen=True)
class TensorStrideIdentifier:
    name: str


@dataclass(frozen=True)
class TensorDescriptor:
    dtype: str | None
    dimensions: tuple[TensorDimensionDescriptor, ...]
    stride_ids: tuple[TensorStrideIdentifier, ...]


@dataclass(frozen=True)
class ConcreteTensorDimension:
    name: str
    size: int
    stride: int


@dataclass(frozen=True)
class ConcreteTensor:
    dtype: str
    dimensions: tuple[ConcreteTensorDimension, ...]


@dataclass(frozen=True)
class V2Route:
    id: str
    family: str
    op: str
    source_id: str
    kernel_path: str
    root_symbol: str
    export_name: str | None
    tensors: Mapping[str, TensorDescriptor]
    constraints: RouteConstraints
    launch: Mapping[str, Any]
    bindings: tuple[Mapping[str, str], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "tensors", _freeze_mapping(self.tensors))
        object.__setattr__(self, "launch", _freeze_mapping(self.launch))
        object.__setattr__(
            self,
            "bindings",
            tuple(_freeze_mapping(binding) for binding in self.bindings),
        )


def stable_id(*parts: Any, length: int = 10) -> str:
    text = json.dumps(parts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]
