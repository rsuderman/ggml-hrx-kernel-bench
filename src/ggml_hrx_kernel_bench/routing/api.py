from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from .models import (
    Candidate,
    CandidateQuery,
    ExecutedCase,
    ExportRequest,
    ImportedSuite,
    RoutingContext,
    RoutingExportResult,
    RuntimeCaseRequest,
)
from .v1.routes import DEFAULT_V1_KERNEL_DIR, DEFAULT_V1_ROUTING_DIR

DEFAULT_ROUTING_VERSION = "v1"
DEFAULT_KERNEL_DIR = DEFAULT_V1_KERNEL_DIR
DEFAULT_ROUTING_DIRS: dict[str, Path] = {
    "v1": DEFAULT_V1_ROUTING_DIR,
    "v2": DEFAULT_V1_ROUTING_DIR.parent / "v2",
}


def default_routing_dir(version: str) -> Path:
    try:
        return DEFAULT_ROUTING_DIRS[version]
    except KeyError as exc:
        raise ValueError(f"unsupported routing version: {version}") from exc


def supported_routing_versions() -> tuple[str, ...]:
    return ("v1", "v2")


class RoutingBackend(Protocol):
    version: str

    def manifest(self, *, original_root: Path | None = None) -> dict[str, Any]: ...

    def candidates(self, query: CandidateQuery) -> list[Candidate]: ...

    def export(self, request: ExportRequest) -> RoutingExportResult: ...

    def resolve_imported_suite(self, suite: ImportedSuite) -> ImportedSuite: ...

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
    kernel_dir: Path = DEFAULT_KERNEL_DIR,
    routing_dir: Path | None = None,
    observed_shapes=None,
) -> RoutingBackend:
    if version == "v1":
        from .v1.backend import V1RoutingBackend

        return V1RoutingBackend(
            RoutingContext(
                kernel_dir=kernel_dir,
                routing_dir=routing_dir or default_routing_dir(version),
                observed_shapes=observed_shapes,
            )
        )
    if version == "v2":
        from .v2.backend import V2RoutingBackend

        return V2RoutingBackend(
            RoutingContext(
                kernel_dir=kernel_dir,
                routing_dir=routing_dir or default_routing_dir(version),
                observed_shapes=observed_shapes,
            )
        )
    raise ValueError(f"unsupported routing version: {version}")


__all__ = [
    "Candidate",
    "CandidateQuery",
    "DEFAULT_KERNEL_DIR",
    "DEFAULT_ROUTING_DIRS",
    "DEFAULT_ROUTING_VERSION",
    "ExecutedCase",
    "ExportRequest",
    "RoutingBackend",
    "RoutingContext",
    "RoutingExportResult",
    "RuntimeCaseRequest",
    "create_router",
    "default_routing_dir",
    "supported_routing_versions",
]
