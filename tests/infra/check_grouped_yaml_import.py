from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from bootstrap import CATALOG_DIR

from ggml_hrx_kernel_bench.grouped_yaml_import import materialize_grouped_yaml


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _load_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    _expect(isinstance(data, dict), f"expected JSON object: {path}")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Run and validate grouped YAML import outputs.")
    parser.add_argument("yaml_path")
    parser.add_argument("output_dir")
    parser.add_argument("--expect-mapped-cases", type=int, required=True)
    parser.add_argument("--expect-unmapped-cases", type=int, required=True)
    parser.add_argument("--expect-config-count", type=int, required=True)
    parser.add_argument("--expect-unmapped-reason", action="append", default=[])
    parser.add_argument("--expect-kernel-case", action="append", default=[])
    args = parser.parse_args()

    yaml_path = Path(args.yaml_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)

    payload = materialize_grouped_yaml(
        yaml_path,
        output_dir=output_dir,
        catalog_dir=CATALOG_DIR,
        split_by_op=False,
    )

    _expect(
        payload["mapped_case_count"] == args.expect_mapped_cases,
        f"expected {args.expect_mapped_cases} mapped cases, saw {payload['mapped_case_count']}",
    )
    _expect(
        payload["unmapped_case_count"] == args.expect_unmapped_cases,
        f"expected {args.expect_unmapped_cases} unmapped cases, saw {payload['unmapped_case_count']}",
    )
    _expect(
        payload["generated_config_count"] == args.expect_config_count,
        f"expected {args.expect_config_count} generated configs, saw {payload['generated_config_count']}",
    )

    imported_workload_path = Path(str(payload["imported_workload_path"]))
    unmapped_path = Path(str(payload["unmapped_path"]))
    summary_markdown_path = Path(str(payload["summary_markdown_path"]))
    _expect(imported_workload_path.is_file(), "missing imported-workload.json")
    _expect(unmapped_path.is_file(), "missing unmapped.json")
    _expect(summary_markdown_path.is_file(), "missing import-summary.md")

    unmapped_payload = _load_json(unmapped_path)
    rows = unmapped_payload.get("rows")
    _expect(isinstance(rows, list), "unmapped rows must be a list")
    reason_counts = Counter(
        row.get("reason")
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("reason"), str)
    )
    for expectation in args.expect_unmapped_reason:
        reason, raw_count = expectation.split("=", 1)
        _expect(
            reason_counts.get(reason, 0) == int(raw_count),
            f"expected reason {reason} to appear {raw_count} time(s), saw {reason_counts.get(reason, 0)}",
        )

    configs_by_kernel: dict[str, dict] = {}
    for raw_path in payload.get("generated_config_paths", []):
        path = Path(str(raw_path))
        config_payload = _load_json(path)
        kernel = config_payload.get("kernel")
        if isinstance(kernel, str):
            configs_by_kernel[kernel] = config_payload

    for expectation in args.expect_kernel_case:
        kernel, raw_values = expectation.split(":", 1)
        config_payload = configs_by_kernel.get(kernel)
        _expect(config_payload is not None, f"missing generated config for {kernel}")
        expected_values = [int(part) for part in raw_values.split(",") if part]
        cases = config_payload.get("cases")
        _expect(isinstance(cases, list), f"cases must be a list in {kernel} config")
        _expect(expected_values in cases, f"expected case {expected_values} in {kernel} config, saw {cases}")

    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
