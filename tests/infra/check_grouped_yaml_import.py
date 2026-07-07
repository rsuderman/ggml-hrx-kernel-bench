from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from bootstrap import CATALOG_DIR

from ggml_hrx_kernel_bench.grouped_yaml_import import materialize_grouped_yaml
from ggml_hrx_kernel_bench.routing.api import supported_routing_versions


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _load_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    _expect(isinstance(data, dict), f"expected JSON object: {path}")
    return data


def _expect_subset(actual: object, expected: object, *, context: str) -> None:
    if isinstance(expected, dict):
        _expect(isinstance(actual, dict), f"{context} must be an object")
        for key, expected_value in expected.items():
            _expect(key in actual, f"missing key {context}.{key}")
            _expect_subset(actual[key], expected_value, context=f"{context}.{key}")
        return
    if isinstance(expected, list):
        _expect(isinstance(actual, list), f"{context} must be a list")
        _expect(
            len(actual) == len(expected),
            f"expected {context} to have {len(expected)} item(s), saw {len(actual)}",
        )
        for index, expected_value in enumerate(expected):
            _expect_subset(actual[index], expected_value, context=f"{context}[{index}]")
        return
    _expect(actual == expected, f"expected {context}={expected!r}, saw {actual!r}")


def _formatted_json(payload: object) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


def _validate_generated_kernel_tests(path: Path, generated_config_paths: list[str]) -> None:
    _expect(path.is_file(), "missing generated-kernel-tests.json")
    payload = _load_json(path)
    entries = payload.get("entries", [])
    _expect(isinstance(entries, list), "generated kernel test entries must be a list")
    manifest_paths = [str(entry.get("config_path")) for entry in entries]
    _expect(
        len(set(manifest_paths)) == len(manifest_paths),
        "generated-kernel-tests.json contains duplicate config paths",
    )
    _expect(
        sorted(manifest_paths) == sorted(str(Path(raw_path)) for raw_path in generated_config_paths),
        "generated-kernel-tests.json does not match generated config paths",
    )


def _validate_bundle_artifacts(payload: dict) -> None:
    import_coverage_path = Path(str(payload["import_coverage_path"]))
    imported_workload_path = Path(str(payload["imported_workload_path"]))
    unmapped_path = Path(str(payload["unmapped_path"]))
    summary_markdown_path = Path(str(payload["summary_markdown_path"]))
    generated_kernel_tests_path = Path(str(payload["generated_kernel_tests_path"]))
    _expect(import_coverage_path.is_file(), "missing import-coverage.json")
    _expect(imported_workload_path.is_file(), "missing imported-workload.json")
    _expect(unmapped_path.is_file(), "missing unmapped.json")
    _expect(summary_markdown_path.is_file(), "missing import-summary.md")

    generated_config_paths = payload.get("generated_config_paths", [])
    _expect(isinstance(generated_config_paths, list), "generated_config_paths must be a list")
    _expect(
        len(set(str(Path(raw_path)) for raw_path in generated_config_paths)) == len(generated_config_paths),
        "generated_config_paths contains duplicates",
    )
    mapped_case_count = int(payload.get("mapped_case_count", 0))
    if mapped_case_count > 0:
        _expect(generated_config_paths, "expected generated configs for mapped cases")
    if mapped_case_count == 0:
        _expect(not generated_config_paths, "expected no generated configs when nothing mapped")

    for raw_path in generated_config_paths:
        config_path = Path(str(raw_path))
        _expect(config_path.is_file(), f"missing generated config {config_path}")

    _validate_generated_kernel_tests(generated_kernel_tests_path, generated_config_paths)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run and validate grouped YAML import outputs.")
    parser.add_argument("yaml_path")
    parser.add_argument("output_dir")
    parser.add_argument("--expected-coverage", required=True)
    parser.add_argument(
        "--tool-dir",
        help="optional PATH-style search list containing loom-link, loom-compile, iree-test-loom, and iree-benchmark-loom",
    )
    parser.add_argument("--split-by-op", action="store_true")
    parser.add_argument(
        "--routing-version",
        choices=supported_routing_versions(),
        default="v1",
    )
    parser.add_argument("--routing-dir", type=Path, default=CATALOG_DIR)
    args = parser.parse_args()

    if args.tool_dir:
        os.environ["PATH"] = f"{args.tool_dir}:{os.environ.get('PATH', '')}"

    yaml_path = Path(args.yaml_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    expected_coverage_path = Path(args.expected_coverage).resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)

    payload = materialize_grouped_yaml(
        yaml_path,
        output_dir=output_dir,
        routing_version=args.routing_version,
        routing_dir=args.routing_dir.resolve(),
        split_by_op=args.split_by_op,
    )

    if args.split_by_op:
        operation_index_path = Path(str(payload["operation_index_path"]))
        _expect(operation_index_path.is_file(), "missing operation-index.json")
        generated_kernel_tests_path = Path(str(payload["generated_kernel_tests_path"]))
        _validate_generated_kernel_tests(
            generated_kernel_tests_path,
            [
                raw_path
                for op_payload in payload.get("operations", {}).values()
                for raw_path in op_payload.get("generated_config_paths", [])
            ],
        )
        operations = payload.get("operations")
        _expect(isinstance(operations, dict), "operations must be an object")
        for op_payload in operations.values():
            _expect(isinstance(op_payload, dict), "operation payload must be an object")
            _validate_bundle_artifacts(op_payload)
    else:
        _validate_bundle_artifacts(payload)

    actual_coverage_path = Path(str(payload["import_coverage_path"]))
    actual_coverage = _load_json(actual_coverage_path)
    expected_coverage = _load_json(expected_coverage_path)
    try:
        _expect_subset(actual_coverage, expected_coverage, context="coverage")
    except RuntimeError as exc:
        print("Grouped YAML import coverage validation failed.")
        print("")
        print(f"Error: {exc}")
        print("")
        print("Actual coverage:")
        print(_formatted_json(actual_coverage))
        print("")
        print("Expected coverage subset:")
        print(_formatted_json(expected_coverage))
        raise

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
