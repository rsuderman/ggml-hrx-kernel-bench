from __future__ import annotations

import json
from pathlib import Path

import pytest

from ggml_hrx_kernel_bench.materialized_assets import materialize_asset_root
from ggml_hrx_kernel_bench.route_query_config import materialize_route_query_configs
from ggml_hrx_kernel_bench.routing.v2.selection import RouteQuery
from ggml_hrx_kernel_bench.yaml_route_import import (
    materialize_yaml_route_import,
    materialize_yaml_route_queries,
)


@pytest.fixture(scope="module")
def routing_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    asset_root = materialize_asset_root(
        tmp_path_factory.mktemp("yaml-route-import-pipeline-assets") / "assets",
        force=True,
    )
    return asset_root / "catalog" / "v2"


def _rms_norm_case(
    *,
    dtype: str = "F32",
    shape: tuple[int, ...] = (4, 2, 1, 1),
    eps: float = 0.0,
) -> dict[str, object]:
    return {
        "inputs": [{"dtype": dtype, "shape": list(shape)}],
        "destinations": [{"dtype": dtype, "shape": list(shape)}],
        "attributes": {"eps": eps},
    }


def _write_mixed_rms_norm_yaml(path: Path) -> None:
    duplicate = _rms_norm_case(eps=0.0)
    path.write_text(
        json.dumps(
            {
                "ops": {
                    "RMS_NORM": [
                        duplicate,
                        duplicate,
                        _rms_norm_case(shape=(8, 3, 1, 1), eps=0.0),
                        _rms_norm_case(eps=0.0001),
                        _rms_norm_case(dtype="F16"),
                        {
                            "inputs": [{"dtype": "F32", "shape": "not-a-shape"}],
                            "destinations": [
                                {"dtype": "F32", "shape": [4, 2, 1, 1]}
                            ],
                        },
                    ]
                }
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _materialize_queries(
    yaml_path: Path,
    output_dir: Path,
    routing_dir: Path,
) -> tuple[Path, Path]:
    materialize_yaml_route_queries(
        [yaml_path],
        output_dir=output_dir,
        routing_dir=routing_dir,
    )
    return output_dir / "route-queries.jsonl", output_dir / "route-query-import.json"


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_yaml_route_query_stage_writes_bare_matched_queries_and_keeps_reports(
    tmp_path: Path,
    routing_dir: Path,
) -> None:
    yaml_path = tmp_path / "rms-norm.v2.yaml"
    _write_mixed_rms_norm_yaml(yaml_path)
    output_dir = tmp_path / "route-import"

    query_path, metadata_path = _materialize_queries(yaml_path, output_dir, routing_dir)

    query_payloads = [
        json.loads(line)
        for line in query_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(query_payloads) == 4
    assert query_payloads[0] == query_payloads[1]
    assert query_payloads[2]["tensors"]["dst"]["dimensions"] == [8, 3, 1, 1]
    assert query_payloads[3]["attributes"] == {"eps": 0.0001}
    assert all(set(payload) == {"op", "tensors", "attributes"} for payload in query_payloads)
    assert all(RouteQuery.from_json(payload).to_json() == payload for payload in query_payloads)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["schema"] == "ggml_hrx_kernel_bench.route_query_import.v1"
    assert metadata["query_count"] == 4

    summary = json.loads((output_dir / "route-import-summary.json").read_text(encoding="utf-8"))
    assert summary["case_count"] == 5
    assert summary["invalid_case_count"] == 1
    assert summary["matched_case_count"] == 4
    assert summary["unmatched_case_count"] == 1

    matches = json.loads((output_dir / "route-matches.json").read_text(encoding="utf-8"))
    unmatched = json.loads((output_dir / "route-unmatched.json").read_text(encoding="utf-8"))
    surface = json.loads(
        (output_dir / "ops" / "RMS_NORM" / "yaml-surface.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(matches["rows"]) == 4
    assert len(unmatched["rows"]) == 1
    assert unmatched["rows"][0]["case"]["inputs"][0]["dtype"] == "F16"
    assert surface["invalid_case_count"] == 1
    assert surface["invalid_cases"][0]["case_index"] == 5
    assert "shape must be a non-empty list" in surface["invalid_cases"][0]["reason"]


def test_yaml_route_query_stage_orders_records_by_operation(
    tmp_path: Path,
    routing_dir: Path,
) -> None:
    yaml_path = tmp_path / "two-ops.v2.yaml"
    yaml_path.write_text(
        json.dumps(
            {
                "ops": {
                    "SCALE": [_rms_norm_case()],
                    "RMS_NORM": [_rms_norm_case()],
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )

    query_path, _ = _materialize_queries(yaml_path, tmp_path / "route-import", routing_dir)

    assert [
        json.loads(line)["op"]
        for line in query_path.read_text(encoding="utf-8").splitlines()
    ] == ["RMS_NORM", "SCALE"]


def test_route_query_config_stage_groups_deduplicates_and_splits_scalar_attributes(
    tmp_path: Path,
    routing_dir: Path,
) -> None:
    yaml_path = tmp_path / "rms-norm.v2.yaml"
    _write_mixed_rms_norm_yaml(yaml_path)
    output_dir = tmp_path / "route-import"
    query_path, metadata_path = _materialize_queries(yaml_path, output_dir, routing_dir)

    summary = materialize_route_query_configs(
        query_path,
        metadata_path=metadata_path,
        output_dir=output_dir,
        routing_dir=routing_dir,
    )

    assert summary["generated_config_count"] == 2
    op_summary = next(row for row in summary["operations"] if row["op"] == "RMS_NORM")
    assert op_summary["generated_config_count"] == 2
    config_paths = sorted(
        (output_dir / "ops" / "RMS_NORM" / "generated-import-configs").glob("*.json")
    )
    configs = [json.loads(path.read_text(encoding="utf-8")) for path in config_paths]
    assert len(configs) == 2
    assert all(len(path.stem.split(".")) == 4 for path in config_paths)
    assert all(config["route_id"] == "rms_norm_f32_contiguous_4d" for config in configs)
    configs_by_eps = {
        next(
            entry["value"]
            for entry in config["execution_abi"]["entries"]
            if entry["kind"] == "scalar" and entry["role"] == "eps"
        ): config
        for config in configs
    }
    assert configs_by_eps[0.0]["cases"] == [[4, 2, 1, 1], [8, 3, 1, 1]]
    assert configs_by_eps[0.0001]["cases"] == [[4, 2, 1, 1]]
    assert sorted(
        entry["value"]
        for config in configs
        for entry in config["execution_abi"]["entries"]
        if entry["kind"] == "scalar" and entry["role"] == "eps"
    ) == [0.0, 0.0001]
    for summary_name in ("yaml-surface-summary.md", "route-import-summary.md"):
        summary_markdown = (output_dir / summary_name).read_text(encoding="utf-8")
        assert "- Generated benchmark configs: `2`" in summary_markdown


def test_route_query_config_stage_accepts_empty_jsonl_and_keeps_zero_match_op(
    tmp_path: Path,
    routing_dir: Path,
) -> None:
    yaml_path = tmp_path / "rms-norm.v2.yaml"
    yaml_path.write_text(
        json.dumps({"ops": {"RMS_NORM": [_rms_norm_case(dtype="F16")]}}) + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "route-import"
    query_path, metadata_path = _materialize_queries(yaml_path, output_dir, routing_dir)

    assert not query_path.read_text(encoding="utf-8").strip()
    summary = materialize_route_query_configs(
        query_path,
        metadata_path=metadata_path,
        output_dir=output_dir,
        routing_dir=routing_dir,
    )

    assert summary["generated_config_count"] == 0
    assert len(summary["operations"]) == 1
    op_summary = summary["operations"][0]
    assert op_summary["op"] == "RMS_NORM"
    assert op_summary["matched_case_count"] == 0
    assert op_summary["unmatched_case_count"] == 1
    assert op_summary["generated_config_count"] == 0
    assert op_summary["generated_config_paths"] == []
    manifest = json.loads(
        (output_dir / "ops" / "RMS_NORM" / "generated-kernel-tests.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["entry_count"] == 0
    assert manifest["entries"] == []


@pytest.mark.parametrize(
    "replacement",
    (
        "{",
        json.dumps({"op": 1, "tensors": {}, "attributes": {}}),
        "unmatched",
    ),
    ids=("malformed-json", "invalid-route-query", "unmatched-route-query"),
)
def test_route_query_config_stage_reports_jsonl_path_and_line(
    tmp_path: Path,
    routing_dir: Path,
    replacement: str,
) -> None:
    yaml_path = tmp_path / "rms-norm.v2.yaml"
    yaml_path.write_text(
        json.dumps({"ops": {"RMS_NORM": [_rms_norm_case()]}}) + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "route-import"
    query_path, metadata_path = _materialize_queries(yaml_path, output_dir, routing_dir)
    if replacement == "unmatched":
        payload = json.loads(query_path.read_text(encoding="utf-8"))
        payload["op"] = "NO_SUCH_OPERATION"
        replacement = json.dumps(payload)
    query_path.write_text(f"\n{replacement}\n", encoding="utf-8")

    with pytest.raises((ValueError, RuntimeError)) as exc_info:
        materialize_route_query_configs(
            query_path,
            metadata_path=metadata_path,
            output_dir=output_dir,
            routing_dir=routing_dir,
        )

    assert f"{query_path}:2:" in str(exc_info.value)


def test_yaml_route_import_facade_matches_explicit_two_stage_pipeline(
    tmp_path: Path,
    routing_dir: Path,
) -> None:
    yaml_path = tmp_path / "rms-norm.v2.yaml"
    _write_mixed_rms_norm_yaml(yaml_path)
    output_dir = tmp_path / "route-import"
    query_path, metadata_path = _materialize_queries(yaml_path, output_dir, routing_dir)
    explicit_summary = materialize_route_query_configs(
        query_path,
        metadata_path=metadata_path,
        output_dir=output_dir,
        routing_dir=routing_dir,
    )
    explicit_tree = _tree_bytes(output_dir)

    facade_summary = materialize_yaml_route_import(
        [yaml_path],
        output_dir=output_dir,
        routing_dir=routing_dir,
    )

    assert facade_summary == explicit_summary
    assert _tree_bytes(output_dir) == explicit_tree
