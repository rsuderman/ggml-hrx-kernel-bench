from __future__ import annotations

import json
from pathlib import Path

from ggml_hrx_kernel_bench.grouped_yaml_import import (
    emit_compact_configs,
    write_generated_kernel_tests_json,
)
from ggml_hrx_kernel_bench.import_models import ImportedCase, ImportedSuite, ResolvedBenchmarkCase


def _resolved_case(
    *,
    params: list[str],
    values: list[int],
) -> ResolvedBenchmarkCase:
    return ResolvedBenchmarkCase(
        imported=ImportedCase(
            op="ADD",
            dtype={"input": "f16"},
            raw_case={"shape": values},
            normalized_params={"shape": values},
            source_path="test.yaml",
            source_group_index=0,
            source_case_index=0,
        ),
        kernel_family="add_f16",
        route_id="add_f16_generic_4d",
        params=params,
        values=values,
    )


def test_emit_compact_configs_keeps_distinct_param_schemas_separate(tmp_path: Path) -> None:
    suite = ImportedSuite(
        schema="test",
        source_path="test.yaml",
        resolved=[
            _resolved_case(params=["nrows", "ncols"], values=[1, 128]),
            _resolved_case(params=["batch", "nrows", "ncols"], values=[4, 1, 128]),
            _resolved_case(params=["nrows", "ncols"], values=[2, 256]),
        ],
    )

    config_paths = emit_compact_configs(suite, tmp_path)

    assert len(config_paths) == 2
    assert len({path.name for path in config_paths}) == 2
    assert all(path.name.startswith("add_f16.add_f16_generic_4d.") for path in config_paths)

    payloads = {
        tuple(json.loads(path.read_text(encoding="utf-8"))["params"]): json.loads(
            path.read_text(encoding="utf-8")
        )
        for path in config_paths
    }

    assert payloads[("nrows", "ncols")]["cases"] == [[1, 128], [2, 256]]
    assert payloads[("batch", "nrows", "ncols")]["cases"] == [[4, 1, 128]]


def test_generated_kernel_tests_manifest_uses_unique_config_paths(tmp_path: Path) -> None:
    suite = ImportedSuite(
        schema="test",
        source_path="test.yaml",
        resolved=[
            _resolved_case(params=["nrows", "ncols"], values=[1, 128]),
            _resolved_case(params=["batch", "nrows", "ncols"], values=[4, 1, 128]),
        ],
    )
    config_paths = emit_compact_configs(suite, tmp_path / "generated-import-configs")
    manifest_path = tmp_path / "generated-kernel-tests.json"

    payload = write_generated_kernel_tests_json(
        source_path=Path("test.yaml"),
        config_paths=config_paths,
        path=manifest_path,
        op="ADD",
    )

    manifest_paths = [entry["config_path"] for entry in payload["entries"]]
    assert payload["entry_count"] == 2
    assert len(set(manifest_paths)) == len(manifest_paths)
