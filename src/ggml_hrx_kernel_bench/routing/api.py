from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from ._config import DEFAULT_ROUTING_VERSION, build_routing_context, supported_routing_versions
from .models import (
    Candidate,
    CandidateQuery,
    ExecutedCase,
    ExportRequest,
    RoutingContext,
    RoutingExportResult,
    RuntimeCaseRequest,
)


class RoutingBackend(Protocol):
    version: str

    def manifest(self, *, original_root: Path | None = None) -> dict[str, Any]: ...

    def candidates(self, query: CandidateQuery) -> list[Candidate]: ...

    def export(self, request: ExportRequest) -> RoutingExportResult: ...

    def select_case(
        self, config: dict[str, Any], selector: str
    ) -> tuple[str, list[int]]: ...

    def select_cases(
        self, config: dict[str, Any], selectors: list[str] | None
    ) -> list[tuple[str, list[int]]]: ...

    def execute_case(self, request: RuntimeCaseRequest) -> ExecutedCase: ...

    def case_result(self, execution: ExecutedCase) -> dict[str, Any]: ...


def create_router(
    *,
    version: str = DEFAULT_ROUTING_VERSION,
    kernel_dir: Path | None = None,
    routing_dir: Path | None = None,
) -> RoutingBackend:
    if version == "v2":
        from .v2.backend import V2RoutingBackend

        return V2RoutingBackend(
            build_routing_context(
                version=version,
                kernel_dir=kernel_dir,
                routing_dir=routing_dir,
            )
        )
    raise ValueError(f"unsupported routing version: {version}")


__all__ = [
    "Candidate",
    "CandidateQuery",
    "DEFAULT_ROUTING_VERSION",
    "ExecutedCase",
    "ExportRequest",
    "RoutingBackend",
    "RoutingContext",
    "RoutingExportResult",
    "RuntimeCaseRequest",
    "create_router",
    "supported_routing_versions",
]
