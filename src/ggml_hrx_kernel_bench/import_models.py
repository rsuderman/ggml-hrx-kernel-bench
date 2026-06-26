from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class MappingStatus(StrEnum):
    MAPPED = "mapped"
    UNMAPPED = "unmapped"
    AMBIGUOUS = "ambiguous"


class UnmappedReason(StrEnum):
    NO_KERNEL_FAMILY_MAPPING = "no_kernel_family_mapping"
    NO_DTYPE_MAPPING = "no_dtype_mapping"
    SHAPE_LOWERING_NOT_IMPLEMENTED = "shape_lowering_not_implemented"
    NO_ROUTE_MATCH = "no_route_match"
    AMBIGUOUS_ROUTE_MATCH = "ambiguous_route_match"


@dataclass(frozen=True)
class ImportedCase:
    op: str
    dtype: dict[str, Any]
    raw_case: dict[str, Any]
    normalized_params: dict[str, Any]
    source_path: str
    source_group_index: int
    source_case_index: int

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResolvedBenchmarkCase:
    imported: ImportedCase
    kernel_family: str
    route_id: str | None
    params: list[str]
    values: list[int]
    mapping_status: MappingStatus = MappingStatus.MAPPED

    def to_json(self) -> dict[str, Any]:
        return {
            "mapping_status": self.mapping_status.value,
            "imported": self.imported.to_json(),
            "kernel_family": self.kernel_family,
            "route_id": self.route_id,
            "params": list(self.params),
            "values": list(self.values),
        }


@dataclass(frozen=True)
class UnmappedCase:
    imported: ImportedCase
    mapping_status: MappingStatus
    reason: UnmappedReason
    detail: str | None = None
    candidate_kernel_families: tuple[str, ...] = ()
    candidate_route_ids: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "mapping_status": self.mapping_status.value,
            "reason": self.reason.value,
            "detail": self.detail,
            "candidate_kernel_families": list(self.candidate_kernel_families),
            "candidate_route_ids": list(self.candidate_route_ids),
            "imported": self.imported.to_json(),
        }


@dataclass(frozen=True)
class ImportedOpGroup:
    op: str
    dtype: dict[str, Any]
    source_path: str
    cases: tuple[ImportedCase, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "dtype": dict(self.dtype),
            "source_path": self.source_path,
            "cases": [case.to_json() for case in self.cases],
        }


@dataclass
class ImportedSuite:
    schema: str
    source_path: str
    op_groups: list[ImportedOpGroup] = field(default_factory=list)
    resolved: list[ResolvedBenchmarkCase] = field(default_factory=list)
    unmapped: list[UnmappedCase] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "source_path": self.source_path,
            "op_groups": [group.to_json() for group in self.op_groups],
            "resolved": [row.to_json() for row in self.resolved],
            "unmapped": [row.to_json() for row in self.unmapped],
        }
