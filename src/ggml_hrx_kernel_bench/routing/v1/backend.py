from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..models import (
    Candidate,
    CandidateQuery,
    ExecutedCase,
    ExportRequest,
    ImportedSuite,
    RoutingContext,
    RoutingExportResult,
    RuntimeCaseRequest,
)
from ..case_selection import select_case as shared_select_case
from ..case_selection import select_cases as shared_select_cases
from .export import export_llama_catalog
from .importer import resolve_imported_suite
from .routes import all_candidates, build_manifest
from .runtime import case_result as runtime_case_result
from .runtime import execute_case as runtime_execute_case


@dataclass(frozen=True)
class V1RoutingBackend:
    context: RoutingContext
    version: str = "v1"

    def manifest(self, *, original_root: Path | None = None) -> dict[str, Any]:
        return build_manifest(
            self.context.kernel_dir,
            self.context.routing_dir,
            original_root=original_root,
        )

    def candidates(self, query: CandidateQuery) -> list[Candidate]:
        return all_candidates(
            self.context.kernel_dir,
            self.context.routing_dir,
            families=query.families,
            limit=query.limit,
            sweep=query.sweep,
            observed_shapes=self.context.observed_shapes,
            include_source_only=query.include_source_only,
        )

    def export(self, request: ExportRequest) -> RoutingExportResult:
        result = export_llama_catalog(
            output_dir=request.output_dir,
            kernel_dir=self.context.kernel_dir,
            catalog_dir=self.context.routing_dir,
            target_key=request.target_key,
            families=request.families,
            catalog_id=request.routing_id,
            sweep=request.sweep,
        )
        return RoutingExportResult(
            backend_version=self.version,
            output_format="llama-catalog-v1",
            output_dir=result.output_dir,
            target_key=result.target_key,
            written_paths=result.written_paths,
            metadata={
                "family_count": result.family_count,
                "source_count": result.source_count,
                "route_count": result.route_count,
                "test_case_count": result.test_case_count,
            },
        )

    def resolve_imported_suite(self, suite: ImportedSuite) -> ImportedSuite:
        return resolve_imported_suite(suite, catalog_dir=self.context.routing_dir)

    def select_case(
        self, config: dict[str, Any], selector: str
    ) -> tuple[str, list[int]]:
        return shared_select_case(config, selector)

    def select_cases(
        self, config: dict[str, Any], selectors: list[str] | None
    ) -> list[tuple[str, list[int]]]:
        return shared_select_cases(config, selectors)

    def execute_case(self, request: RuntimeCaseRequest) -> ExecutedCase:
        kernel_dir = request.kernel_dir or self.context.kernel_dir
        routing_dir = request.routing_dir or self.context.routing_dir
        candidate, row, summary = runtime_execute_case(
            kernel_dir=kernel_dir,
            routing_dir=routing_dir,
            config_data=request.config_data,
            current_case_id=request.current_case_id,
            current_case_values=request.current_case_values,
            tool_dir=request.tool_dir,
            target=request.target,
            rocm_path=request.rocm_path,
            iterations=request.iterations,
            warmup_iterations=request.warmup_iterations,
            max_batches=request.max_batches,
            output_dir=request.output_dir,
            require_tool=request.require_tool,
        )
        return ExecutedCase(
            candidate=candidate,
            row=row,
            summary=summary,
            current_case_id=request.current_case_id,
            current_case_values=list(request.current_case_values),
            output_dir=request.output_dir,
        )

    def case_result(self, execution: ExecutedCase) -> dict[str, Any]:
        return runtime_case_result(
            candidate=execution.candidate,
            current_case_id=execution.current_case_id,
            current_case_values=execution.current_case_values,
            row=execution.row,
            summary=execution.summary,
            output_dir=execution.output_dir,
        )
