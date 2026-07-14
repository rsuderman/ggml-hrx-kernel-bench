from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import bootstrap  # noqa: F401

from ggml_hrx_kernel_bench.kernel_test_config import validate_config
from ggml_hrx_kernel_bench.yaml_route_import import materialize_yaml_route_import


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


def _validate_import_coverage(output_dir: Path, summary: dict) -> None:
    import_coverage_path = output_dir / "import-coverage.json"
    _expect(import_coverage_path.is_file(), f"missing {import_coverage_path}")
    coverage = _load_json(import_coverage_path)
    _expect(
        coverage.get("schema") == "ggml_hrx_kernel_bench.import_test_coverage.v1",
        "unexpected import coverage schema",
    )
    coverage_operations = coverage.get("operations")
    summary_operations = summary.get("operations")
    _expect(isinstance(coverage_operations, list), "coverage operations must be a list")
    _expect(isinstance(summary_operations, list), "summary operations must be a list")
    _expect(
        len(coverage_operations) == len(summary_operations),
        "coverage operation count does not match route summary",
    )

    summary_by_op = {row["op"]: row for row in summary_operations}
    total_pass = 0
    total_fail = 0
    for row in coverage_operations:
        _expect(isinstance(row, dict), "coverage operation row must be an object")
        op = str(row.get("op"))
        _expect(op in summary_by_op, f"coverage has unknown op {op}")
        summary_row = summary_by_op[op]
        expected_pass = int(summary_row["matched_case_count"])
        expected_fail = (
            int(summary_row["case_count"])
            + int(summary_row["invalid_case_count"])
            - expected_pass
        )
        _expect(row.get("pass_case_count") == expected_pass, f"{op} pass count mismatch")
        _expect(row.get("fail_case_count") == expected_fail, f"{op} fail count mismatch")
        total_pass += expected_pass
        total_fail += expected_fail

        op_coverage_path = output_dir / "ops" / op / "import-coverage.json"
        _expect(op_coverage_path.is_file(), f"missing {op_coverage_path}")
        op_coverage = _load_json(op_coverage_path)
        _expect(
            op_coverage.get("operations") == [row],
            f"{op} per-op import coverage does not match top-level row",
        )

    _expect(coverage.get("operation_count") == len(coverage_operations), "operation count mismatch")
    _expect(coverage.get("total_pass_case_count") == total_pass, "total pass count mismatch")
    _expect(coverage.get("total_fail_case_count") == total_fail, "total fail count mismatch")


def _validate_expected_import_coverage(output_dir: Path, expected_coverage_path: Path) -> None:
    actual_coverage_path = output_dir / "import-coverage.json"
    actual_coverage = _load_json(actual_coverage_path)
    expected_coverage = _load_json(expected_coverage_path)
    try:
        _expect_subset(actual_coverage, expected_coverage, context="coverage")
    except RuntimeError as exc:
        print("YAML route import coverage validation failed.")
        print("")
        print(f"Error: {exc}")
        print("")
        print("Actual coverage:")
        print(_formatted_json(actual_coverage))
        print("")
        print("Expected coverage subset:")
        print(_formatted_json(expected_coverage))
        raise


def _validate_generated_kernel_tests(path: Path, generated_config_paths: list[str]) -> None:
    _expect(path.is_file(), f"missing generated kernel tests manifest {path}")
    payload = _load_json(path)
    _expect(
        payload.get("schema") == "ggml_hrx_kernel_bench.generated_kernel_tests.v1",
        f"unexpected generated kernel tests schema in {path}",
    )
    entries = payload.get("entries")
    _expect(isinstance(entries, list), f"{path} entries must be a list")
    manifest_paths = [str(entry.get("config_path")) for entry in entries]
    _expect(
        len(set(manifest_paths)) == len(manifest_paths),
        f"{path} contains duplicate config paths",
    )
    expected_paths = [str(Path(raw_path)) for raw_path in generated_config_paths]
    _expect(
        sorted(manifest_paths) == sorted(expected_paths),
        f"{path} does not match generated config paths",
    )
    _expect(payload.get("entry_count") == len(entries), f"{path} entry_count mismatch")


def _validate_generated_configs(output_dir: Path, summary: dict) -> None:
    generated_config_paths = summary.get("generated_config_paths", [])
    _expect(isinstance(generated_config_paths, list), "generated_config_paths must be a list")
    _expect(
        len(set(str(Path(raw_path)) for raw_path in generated_config_paths)) == len(generated_config_paths),
        "generated_config_paths contains duplicates",
    )
    for raw_path in generated_config_paths:
        config_path = Path(str(raw_path))
        _expect(config_path.is_file(), f"missing generated config {config_path}")
        validate_config(_load_json(config_path))
    _expect(
        summary.get("generated_config_count") == len(generated_config_paths),
        "generated_config_count mismatch",
    )
    _validate_generated_kernel_tests(output_dir / "generated-kernel-tests.json", generated_config_paths)

    operations = summary.get("operations")
    _expect(isinstance(operations, list), "summary operations must be a list")
    for row in operations:
        _expect(isinstance(row, dict), "operation summary row must be an object")
        op = str(row.get("op"))
        op_config_paths = row.get("generated_config_paths", [])
        _expect(isinstance(op_config_paths, list), f"{op} generated_config_paths must be a list")
        _expect(
            row.get("generated_config_count") == len(op_config_paths),
            f"{op} generated_config_count mismatch",
        )
        for raw_path in op_config_paths:
            config_path = Path(str(raw_path))
            _expect(config_path.is_file(), f"missing generated config {config_path}")
            validate_config(_load_json(config_path))
        _validate_generated_kernel_tests(
            output_dir / "ops" / op / "generated-kernel-tests.json",
            op_config_paths,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize descriptor YAML route import reports.")
    parser.add_argument("output_dir")
    parser.add_argument("--yaml", action="append", required=True, help="descriptor YAML input path")
    parser.add_argument("--routing-dir", type=Path, required=True)
    parser.add_argument("--expected-coverage", type=Path, required=True)
    parser.add_argument(
        "--tool-dir",
        help="optional PATH-style search list containing loom-link, loom-compile, ggml-hrx-run-loom, iree-test-loom, and iree-benchmark-loom",
    )
    args = parser.parse_args()

    if args.tool_dir:
        os.environ["PATH"] = f"{args.tool_dir}:{os.environ.get('PATH', '')}"

    output_dir = Path(args.output_dir).resolve()
    summary = materialize_yaml_route_import(
        [Path(raw_path).resolve() for raw_path in args.yaml],
        output_dir=output_dir,
        routing_dir=args.routing_dir.resolve(),
    )
    _validate_import_coverage(output_dir, summary)
    _validate_expected_import_coverage(output_dir, args.expected_coverage.resolve())
    _validate_generated_configs(output_dir, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
