from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from ..case_selection import select_case as shared_select_case
from ..case_selection import select_cases as shared_select_cases
from ..models import (
    Candidate,
    CandidateQuery,
    ExecutedCase,
    ExportRequest,
    RoutingContext,
    RoutingExportResult,
    RuntimeCaseRequest,
)
from .candidates import list_candidates
from .manifest import build_manifest
from .query import RouteCatalog, load_route_catalog
from .runtime import case_result as runtime_case_result
from .runtime import execute_case as runtime_execute_case
from .selection import RouteSelector, create_route_selector


@dataclass(frozen=True)
class V2RoutingBackend:
    context: RoutingContext
    version: str = "v2"

    @cached_property
    def catalog(self) -> RouteCatalog:
        return load_route_catalog(self.context.routing_dir)

    @cached_property
    def selector(self) -> RouteSelector:
        return create_route_selector(self.catalog)

    def manifest(self, *, original_root: Path | None = None) -> dict[str, object]:
        return build_manifest(
            kernel_dir=self.context.kernel_dir,
            catalog=self.catalog,
            original_root=original_root,
        )

    def candidates(self, query: CandidateQuery) -> list[Candidate]:
        return list_candidates(kernel_dir=self.context.kernel_dir, catalog=self.catalog, query=query)

    def export(self, request: ExportRequest) -> RoutingExportResult:
        routes = self.catalog.routes
        request.output_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = request.output_dir / "routing-export.json"
        payload = {
            "schema": "ggml_hrx_kernel_bench.routing_export_metadata.v2",
            "backend_version": self.version,
            "output_format": "routing-descriptor-v2",
            "target_key": request.target_key,
            "routing_id": request.routing_id,
            "family_count": len({route.family for route in routes}),
            "route_count": len(routes),
            "source_count": len({route.source_id for route in routes}),
            "route_ids": [route.id for route in routes],
        }
        metadata_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return RoutingExportResult(
            backend_version=self.version,
            output_format="routing-descriptor-v2",
            output_dir=request.output_dir,
            target_key=request.target_key,
            written_paths=(metadata_path,),
            metadata={
                "family_count": payload["family_count"],
                "route_count": payload["route_count"],
                "source_count": payload["source_count"],
            },
        )

    def select_case(self, config: dict[str, object], selector: str) -> tuple[str, list[int]]:
        return shared_select_case(config, selector)

    def select_cases(
        self, config: dict[str, object], selectors: list[str] | None
    ) -> list[tuple[str, list[int]]]:
        return shared_select_cases(config, selectors)

    def execute_case(self, request: RuntimeCaseRequest) -> ExecutedCase:
        kernel_dir = request.kernel_dir or self.context.kernel_dir
        routing_dir = request.routing_dir or self.context.routing_dir
        if routing_dir == self.context.routing_dir:
            catalog = self.catalog
            selector = self.selector
        else:
            catalog = load_route_catalog(routing_dir)
            selector = create_route_selector(catalog)
        return runtime_execute_case(
            request,
            catalog=catalog,
            kernel_dir=kernel_dir,
            selector=selector,
        )

    def case_result(self, execution: ExecutedCase) -> dict[str, object]:
        return runtime_case_result(execution)
