from __future__ import annotations

from pathlib import Path

from ggml_hrx_kernel_bench.import_models import ImportedCase, ImportedOpGroup, ImportedSuite, UnmappedReason
from ggml_hrx_kernel_bench.routing.api import CandidateQuery, create_router


def test_v2_router_returns_no_candidates(tmp_path: Path) -> None:
    router = create_router(version="v2", kernel_dir=tmp_path)

    assert router.candidates(CandidateQuery()) == []


def test_v2_router_marks_imports_unmapped(tmp_path: Path) -> None:
    router = create_router(version="v2", kernel_dir=tmp_path)
    suite = ImportedSuite(
        schema="test",
        source_path="test.yaml",
        op_groups=[
            ImportedOpGroup(
                op="ADD",
                dtype={},
                source_path="test.yaml",
                cases=(
                    ImportedCase(
                        op="ADD",
                        dtype={},
                        raw_case={"ne": [1, 1, 1, 1]},
                        normalized_params={"ne": [1, 1, 1, 1]},
                        source_path="test.yaml",
                        source_group_index=0,
                        source_case_index=0,
                    ),
                ),
            )
        ],
    )

    resolved = router.resolve_imported_suite(suite)

    assert resolved.resolved == []
    assert len(resolved.unmapped) == 1
    assert resolved.unmapped[0].reason == UnmappedReason.NO_KERNEL_FAMILY_MAPPING
