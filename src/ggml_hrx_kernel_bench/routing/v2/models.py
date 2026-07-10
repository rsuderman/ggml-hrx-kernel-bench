from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

VALUE_OPERATION_SPEC: dict[str, tuple[int, int]] = {
    "contiguous_strides": (1, 0),
    "product": (1, 0),
    "inverse_permutation": (1, 0),
    "head": (1, 1),
    "tail": (1, 1),
    "chain_permutations": (2, 0),
    "permuted_contiguous_strides": (2, 0),
}

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
    rank_min: int | None = None
    rank_max: int | None = None
    index: int | None = None
    min: int | None = None
    max: int | None = None
    multiple_of: int | None = None
    iota: bool = False
    equals: tuple[str, ...] = ()
    divides: tuple[str, ...] = ()


@dataclass(frozen=True)
class RouteConstraints:
    checks: tuple[ConstraintCheck, ...] = ()


@dataclass(frozen=True)
class ValueDefinition:
    name: str
    operation_kind: str
    sources: tuple[str, ...]
    parameters: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        operation_spec = VALUE_OPERATION_SPEC.get(self.operation_kind)
        if operation_spec is None:
            raise ValueError(f"unsupported value operation kind: {self.operation_kind!r}")
        expected_sources, expected_parameters = operation_spec
        if len(self.sources) != expected_sources:
            raise ValueError(
                f"value operation {self.operation_kind!r} requires {expected_sources} source(s), got {len(self.sources)}"
            )
        if len(self.parameters) != expected_parameters:
            raise ValueError(
                f"value operation {self.operation_kind!r} requires {expected_parameters} parameter(s), got {len(self.parameters)}"
            )


@dataclass(frozen=True)
class BindingDefinition:
    key: str
    source: str | None = None
    value: str | None = None

    def __post_init__(self) -> None:
        if bool(self.source) == bool(self.value):
            raise ValueError("bindings require exactly one of source or value")


@dataclass(frozen=True)
class TensorDescriptor:
    dtype: str | None
    dimensions_capture: str
    strides_capture: str
    permutation_capture: str | None = None


@dataclass(frozen=True)
class ConcreteTensorDimension:
    name: str
    size: int
    stride: int


@dataclass(frozen=True)
class ConcreteTensor:
    dtype: str
    dimensions: tuple[ConcreteTensorDimension, ...]
    permutation: tuple[int, ...] | None = None


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
    bindings: tuple[BindingDefinition, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "tensors", _freeze_mapping(self.tensors))
        object.__setattr__(self, "launch", _freeze_mapping(self.launch))


def stable_id(*parts: Any, length: int = 10) -> str:
    text = json.dumps(parts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]
