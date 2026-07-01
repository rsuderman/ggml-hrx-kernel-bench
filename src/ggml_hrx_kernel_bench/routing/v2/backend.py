from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ...import_models import ImportedSuite, MappingStatus, UnmappedCase, UnmappedReason
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


@dataclass(frozen=True)
class V2RoutingBackend:
    context: RoutingContext
    version: str = "v2"

    def manifest(self, *, original_root: Path | None = None) -> dict[str, object]:
        kernel_files = sorted(path.name for path in self.context.kernel_dir.glob("*.loom"))
        return {
            "schema": "ggml_hrx_kernel_bench.routing_manifest.v2",
            "routing_version": self.version,
            "kernel_count": len(kernel_files),
            "catalog_source_count": 0,
            "route_count": 0,
            "entries": [
                {
                    "path": str(path),
                    "source_ids": [],
                    "route_count": 0,
                    "coverage": "unrouted",
                }
                for path in kernel_files
            ],
            "source_ids_without_routes": [],
            "route_source_ids_without_source_entry": [],
            "kernel_files_without_source_entry": kernel_files,
            "source_entries_without_kernel_file": [],
        }

    def candidates(self, query: CandidateQuery) -> list[Candidate]:
        return []

    def export(self, request: ExportRequest) -> RoutingExportResult:
        request.output_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = request.output_dir / "routing-export.json"
        payload = {
            "schema": "ggml_hrx_kernel_bench.routing_export_metadata.v1",
            "backend_version": self.version,
            "output_format": "empty-routing-v2",
            "target_key": request.target_key,
            "routing_id": request.routing_id,
            "family_count": 0,
            "route_count": 0,
            "source_count": 0,
        }
        metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return RoutingExportResult(
            backend_version=self.version,
            output_format="empty-routing-v2",
            output_dir=request.output_dir,
            target_key=request.target_key,
            written_paths=(metadata_path,),
            metadata={
                "family_count": 0,
                "route_count": 0,
                "source_count": 0,
            },
        )

    def resolve_imported_suite(self, suite: ImportedSuite) -> ImportedSuite:
        suite.resolved = []
        suite.unmapped = [
            UnmappedCase(
                imported=case,
                mapping_status=MappingStatus.UNMAPPED,
                reason=UnmappedReason.NO_KERNEL_FAMILY_MAPPING,
                detail="routing backend v2 returned no matching routes",
            )
            for group in suite.op_groups
            for case in group.cases
        ]
        return suite

    def select_case(self, config: dict[str, object], selector: str) -> tuple[str, list[int]]:
        return shared_select_case(config, selector)

    def select_cases(
        self, config: dict[str, object], selectors: list[str] | None
    ) -> list[tuple[str, list[int]]]:
        return shared_select_cases(config, selectors)

    def execute_case(self, request: RuntimeCaseRequest) -> ExecutedCase:
        raise RuntimeError(
            f"routing backend {self.version} returned no matching route for kernel "
            f"{request.config_data.get('kernel')!r}"
        )

    def case_result(self, execution: ExecutedCase) -> dict[str, object]:
        return {
            "case_id": execution.current_case_id,
            "values": list(execution.current_case_values),
            "candidate_id": execution.candidate.id,
            "status": execution.row.get("status"),
        }
