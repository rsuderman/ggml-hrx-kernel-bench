from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..observed_shapes import ObservedShapeCatalog
from ..specs import file_sha256


@dataclass(frozen=True)
class Candidate:
    id: str
    family: str
    op: str
    source_id: str
    source_path: Path
    root_symbol: str
    export_name: str | None
    route_id: str | None
    route: dict[str, Any] | None
    shape: dict[str, int]
    values: dict[str, int | str]
    config: dict[str, str]
    dispatch: dict[str, Any]
    supports: dict[str, Any]
    coverage: str
    schedule: dict[str, Any] | None = None
    status: str = "planned"
    message: str | None = None

    def to_ledger(self) -> dict[str, Any]:
        return {
            "candidate_id": self.id,
            "family": self.family,
            "op": self.op,
            "source_id": self.source_id,
            "source_path": str(self.source_path),
            "source_sha256": file_sha256(self.source_path),
            "root_symbol": self.root_symbol,
            "export_name": self.export_name,
            "route_id": self.route_id,
            "shape": self.shape,
            "values": self.values,
            "config_bindings": self.config,
            "dispatch": self.dispatch,
            "supports": self.supports,
            "schedule": self.schedule,
            "coverage": self.coverage,
            "status": self.status,
            "message": self.message,
        }


@dataclass(frozen=True)
class RoutingContext:
    kernel_dir: Path
    routing_dir: Path
    observed_shapes: ObservedShapeCatalog | None = None


@dataclass(frozen=True)
class CandidateQuery:
    families: set[str] | None = None
    limit: int | None = None
    sweep: str = "minimal"
    include_source_only: bool = False


@dataclass(frozen=True)
class ExportRequest:
    output_dir: Path
    target_key: str
    families: set[str] | None = None
    routing_id: str | None = None
    sweep: str = "minimal"


@dataclass(frozen=True)
class RoutingExportResult:
    backend_version: str
    output_format: str
    output_dir: Path
    target_key: str
    written_paths: tuple[Path, ...]
    metadata: dict[str, Any]

    def to_ledger(self) -> dict[str, Any]:
        return {
            "schema": "ggml_hrx_kernel_bench.routing_export.v1",
            "backend_version": self.backend_version,
            "output_format": self.output_format,
            "output_dir": str(self.output_dir),
            "target_key": self.target_key,
            "written_paths": [str(path) for path in self.written_paths],
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class RuntimeCaseRequest:
    kernel_dir: Path | None
    routing_dir: Path | None
    config_data: dict[str, Any]
    current_case_id: str
    current_case_values: list[int]
    tool_dir: str | None
    target: str
    rocm_path: str | None
    iterations: int
    warmup_iterations: int
    max_batches: int
    output_dir: Path
    require_tool: Any


@dataclass(frozen=True)
class ExecutedCase:
    candidate: Candidate
    row: dict[str, Any]
    summary: dict[str, Any]
    current_case_id: str
    current_case_values: list[int]
    output_dir: Path


__all__ = [
    "Candidate",
    "CandidateQuery",
    "ExecutedCase",
    "ExportRequest",
    "RoutingExportResult",
    "RoutingContext",
    "RuntimeCaseRequest",
]
