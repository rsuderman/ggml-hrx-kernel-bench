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
class ConstraintCheck:
    name: str | None = None
    length: int | None = None
    index: int | None = None
    min: int | None = None
    max: int | None = None
    multiple_of: int | None = None
    equals: tuple[str, ...] = ()


@dataclass(frozen=True)
class RouteConstraints:
    checks: tuple[ConstraintCheck, ...] = ()


@dataclass(frozen=True)
class ValueDefinition:
    name: str
    contiguous_strides: str | None = None
    product: str | None = None


@dataclass(frozen=True)
class TensorDescriptor:
    dtype: str | None
    dimensions_capture: str
    strides_capture: str


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
    values: tuple[ValueDefinition, ...]
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
