from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ggml_hrx_kernel_bench.harness_inventory import (  # noqa: E402
    build_harness_inventory,
    inventory_to_markdown,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_build_harness_inventory_reports_descriptor_and_legacy_state(tmp_path: Path) -> None:
    route_import_dir = tmp_path / "route-import"
    descriptor_output_dir = tmp_path / "descriptors"
    _write_json(
        route_import_dir / "import-coverage.json",
        {
            "schema": "ggml_hrx_kernel_bench.import_test_coverage.v1",
            "operation_count": 2,
            "total_pass_case_count": 5,
            "total_fail_case_count": 7,
            "operations": [
                {"op": "EXP", "pass_case_count": 4, "fail_case_count": 0},
                {"op": "ADD", "pass_case_count": 1, "fail_case_count": 7},
            ],
        },
    )
    _write_json(
        descriptor_output_dir / "EXP" / "loom-execution-descriptors.json",
        {
            "schema": "ggml_hrx_kernel_bench.loom_execution_descriptors.v1",
            "entry_count": 6,
            "emitted_count": 4,
            "skipped_count": 1,
            "unsupported_count": 1,
            "filtered_count": 0,
            "entries": [],
        },
    )
    descriptor_cmake = tmp_path / "descriptor.cmake"
    descriptor_cmake.write_text(
        "\n".join(
            [
                "add_test(NAME kernel-descriptor-generate-demo-EXP-generated)",
                "add_test(NAME kernel-descriptor-execute-demo-EXP-generated)",
                "add_test(NAME kernel-descriptor-generate-demo-ADD-generated)",
            ]
        ),
        encoding="utf-8",
    )
    legacy_cmake = tmp_path / "legacy.cmake"
    legacy_cmake.write_text(
        "add_test(NAME kernel-run-demo-ADD-generated)\n",
        encoding="utf-8",
    )

    inventory = build_harness_inventory(
        name="demo",
        route_import_dir=route_import_dir,
        descriptor_output_dir=descriptor_output_dir,
        descriptor_tests_cmake=descriptor_cmake,
        legacy_runtime_tests_cmake=legacy_cmake,
    )

    rows = {row["op"]: row for row in inventory["rows"]}
    assert rows["EXP"]["route_import_matched_count"] == 4
    assert rows["EXP"]["descriptor_emitted_count"] == 4
    assert rows["EXP"]["descriptor_skipped_count"] == 1
    assert rows["EXP"]["descriptor_unsupported_count"] == 1
    assert rows["EXP"]["descriptor_generate_registered"] is True
    assert rows["EXP"]["descriptor_execute_registered"] is True
    assert rows["EXP"]["legacy_runtime_registered"] is False
    assert rows["EXP"]["descriptor_hsa_status"] == "enabled"

    assert rows["ADD"]["route_import_unmatched_count"] == 7
    assert rows["ADD"]["descriptor_emitted_count"] is None
    assert rows["ADD"]["descriptor_execute_registered"] is False
    assert rows["ADD"]["legacy_runtime_registered"] is True
    assert rows["ADD"]["descriptor_hsa_status"] == "gated"

    markdown = inventory_to_markdown(inventory)
    assert "| EXP | 4 | 0 | 4 | 1 | 1 | generate/execute | no | enabled |" in markdown
    assert "| ADD | 1 | 7 | - | - | - | generate | yes | gated |" in markdown
