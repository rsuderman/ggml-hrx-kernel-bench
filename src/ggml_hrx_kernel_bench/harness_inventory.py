from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SCHEMA = "ggml_hrx_kernel_bench.harness_inventory.v1"


@dataclass(frozen=True)
class HarnessInventoryRow:
    op: str
    route_import_matched_count: int
    route_import_unmatched_count: int
    descriptor_emitted_count: int | None
    descriptor_skipped_count: int | None
    descriptor_unsupported_count: int | None
    descriptor_filtered_count: int | None
    descriptor_generate_registered: bool
    descriptor_execute_registered: bool
    legacy_runtime_registered: bool
    descriptor_hsa_status: str


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def _operation_coverage_rows(route_import_dir: Path) -> list[dict[str, Any]]:
    coverage_path = route_import_dir / "import-coverage.json"
    coverage = _load_json(coverage_path)
    if coverage.get("schema") != "ggml_hrx_kernel_bench.import_test_coverage.v1":
        raise RuntimeError(f"unsupported import coverage schema in {coverage_path}")
    operations = coverage.get("operations")
    if not isinstance(operations, list):
        raise RuntimeError(f"{coverage_path} must contain an operations list")
    rows: list[dict[str, Any]] = []
    for index, operation in enumerate(operations):
        if not isinstance(operation, dict):
            raise RuntimeError(f"{coverage_path} operations[{index}] must be an object")
        op = operation.get("op")
        if not isinstance(op, str) or not op:
            raise RuntimeError(f"{coverage_path} operations[{index}].op must be a non-empty string")
        rows.append(operation)
    return sorted(rows, key=lambda row: str(row["op"]))


def _descriptor_counts(manifest_path: Path) -> tuple[int, int, int, int] | None:
    if not manifest_path.is_file():
        return None
    manifest = _load_json(manifest_path)
    if manifest.get("schema") != "ggml_hrx_kernel_bench.loom_execution_descriptors.v1":
        raise RuntimeError(f"unsupported descriptor manifest schema in {manifest_path}")
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise RuntimeError(f"{manifest_path} must contain an entries list")
    emitted = int(manifest.get("emitted_count", sum(1 for entry in entries if entry.get("status") == "emitted")))
    skipped = int(manifest.get("skipped_count", sum(1 for entry in entries if entry.get("status") == "skipped")))
    unsupported = int(
        manifest.get("unsupported_count", sum(1 for entry in entries if entry.get("status") == "unsupported"))
    )
    filtered = int(manifest.get("filtered_count", 0))
    return emitted, skipped, unsupported, filtered


def _read_optional_text(path: Path | None) -> str:
    if path is None or not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")


def _contains_test(cmake_text: str, name: str) -> bool:
    return name in cmake_text


def build_harness_inventory(
    *,
    name: str,
    route_import_dir: Path,
    descriptor_output_dir: Path,
    descriptor_tests_cmake: Path | None,
    legacy_runtime_tests_cmake: Path | None,
) -> dict[str, Any]:
    descriptor_cmake = _read_optional_text(descriptor_tests_cmake)
    legacy_cmake = _read_optional_text(legacy_runtime_tests_cmake)
    rows: list[HarnessInventoryRow] = []
    for operation in _operation_coverage_rows(route_import_dir):
        op = str(operation["op"])
        op_safe_name = _safe_name(op)
        descriptor_manifest_path = descriptor_output_dir / op_safe_name / "loom-execution-descriptors.json"
        counts = _descriptor_counts(descriptor_manifest_path)
        generate_registered = _contains_test(
            descriptor_cmake,
            f"kernel-descriptor-generate-{name}-{op_safe_name}-generated",
        )
        execute_registered = _contains_test(
            descriptor_cmake,
            f"kernel-descriptor-execute-{name}-{op_safe_name}-generated",
        )
        legacy_registered = _contains_test(
            legacy_cmake,
            f"kernel-run-{name}-{op_safe_name}-generated",
        )
        if execute_registered:
            hsa_status = "enabled"
        elif generate_registered:
            hsa_status = "gated"
        else:
            hsa_status = "not_registered"
        rows.append(
            HarnessInventoryRow(
                op=op,
                route_import_matched_count=int(operation.get("pass_case_count", 0)),
                route_import_unmatched_count=int(operation.get("fail_case_count", 0)),
                descriptor_emitted_count=counts[0] if counts is not None else None,
                descriptor_skipped_count=counts[1] if counts is not None else None,
                descriptor_unsupported_count=counts[2] if counts is not None else None,
                descriptor_filtered_count=counts[3] if counts is not None else None,
                descriptor_generate_registered=generate_registered,
                descriptor_execute_registered=execute_registered,
                legacy_runtime_registered=legacy_registered,
                descriptor_hsa_status=hsa_status,
            )
        )
    return {
        "schema": SCHEMA,
        "name": name,
        "route_import_dir": str(route_import_dir),
        "descriptor_output_dir": str(descriptor_output_dir),
        "descriptor_tests_cmake": str(descriptor_tests_cmake) if descriptor_tests_cmake else None,
        "legacy_runtime_tests_cmake": str(legacy_runtime_tests_cmake) if legacy_runtime_tests_cmake else None,
        "operation_count": len(rows),
        "rows": [asdict(row) for row in rows],
    }


def write_inventory_json(inventory: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def inventory_to_markdown(inventory: dict[str, Any]) -> str:
    rows = inventory["rows"]
    lines = [
        f"# Harness inventory: {inventory['name']}",
        "",
        "| op | route matched | route unmatched | descriptor emitted | descriptor skipped | descriptor unsupported | descriptor tests | legacy runtime | HSA execute |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in rows:
        descriptor_tests = "/".join(
            part
            for part, registered in (
                ("generate", row["descriptor_generate_registered"]),
                ("execute", row["descriptor_execute_registered"]),
            )
            if registered
        )
        descriptor_tests = descriptor_tests or "none"
        descriptor_emitted = "-" if row["descriptor_emitted_count"] is None else str(row["descriptor_emitted_count"])
        descriptor_skipped = "-" if row["descriptor_skipped_count"] is None else str(row["descriptor_skipped_count"])
        descriptor_unsupported = (
            "-" if row["descriptor_unsupported_count"] is None else str(row["descriptor_unsupported_count"])
        )
        lines.append(
            "| {op} | {matched} | {unmatched} | {emitted} | {skipped} | {unsupported} | {tests} | {legacy} | {hsa} |".format(
                op=row["op"],
                matched=row["route_import_matched_count"],
                unmatched=row["route_import_unmatched_count"],
                emitted=descriptor_emitted,
                skipped=descriptor_skipped,
                unsupported=descriptor_unsupported,
                tests=descriptor_tests,
                legacy="yes" if row["legacy_runtime_registered"] else "no",
                hsa=row["descriptor_hsa_status"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def write_inventory_markdown(inventory: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(inventory_to_markdown(inventory), encoding="utf-8")
