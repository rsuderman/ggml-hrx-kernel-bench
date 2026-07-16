from __future__ import annotations

import json
from pathlib import Path


EXPECTED_TOP_LEVEL_KEYS = [
    "schema",
    "operation_count",
    "total_pass_case_count",
    "total_fail_case_count",
    "operations",
]

EXPECTED_OPERATION_KEYS = [
    "op",
    "pass_case_count",
    "fail_case_count",
]


def test_expected_import_coverage_fixtures_preserve_canonical_key_order() -> None:
    fixture_paths = [
        Path("tests/kernels/data/llamacpp.import-coverage.json"),
        Path("tests/models/data/llama-8b-q8.import-coverage.json"),
    ]

    for fixture_path in fixture_paths:
        payload = json.loads(fixture_path.read_text(encoding="utf-8"))
        assert list(payload.keys()) == EXPECTED_TOP_LEVEL_KEYS
        assert payload["operations"], f"{fixture_path} must contain operation coverage rows"
        for row in payload["operations"]:
            assert list(row.keys()) == EXPECTED_OPERATION_KEYS
