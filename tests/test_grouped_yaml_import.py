from __future__ import annotations

import json
from pathlib import Path

import yaml

from ggml_hrx_kernel_bench.grouped_yaml_import import (
    emit_compact_configs,
    load_grouped_yaml_suite,
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


def test_model_style_cpy_sources_derive_role_dtypes(tmp_path: Path) -> None:
    yaml_path = tmp_path / "model.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "ops": {
                    "CPY": [
                        {
                            "dtype": {"type": "f16"},
                            "cases": [
                                {
                                    "name": "attn_inp_kq_mask",
                                    "ne": [8192, 512, 1, 1],
                                    "op_params": [],
                                    "sources": "f32[8192,512,1,1],f16[8192,512,1,1]",
                                }
                            ],
                        }
                    ]
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    suite = load_grouped_yaml_suite(yaml_path)
    case = suite.op_groups[0].cases[0]

    assert case.dtype == {
        "type": "f16",
        "type_dst": "f16",
        "type_src": "f32",
        "type_src0": "f32",
        "type_src1": "f16",
    }
    assert case.normalized_params["sources"] == "f32[8192,512,1,1],f16[8192,512,1,1]"


def test_model_style_mul_mat_sources_derive_operand_aliases(tmp_path: Path) -> None:
    yaml_path = tmp_path / "model.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "ops": {
                    "MUL_MAT": [
                        {
                            "dtype": {"type_dst": "f32", "type_src": "q8_0"},
                            "cases": [
                                {
                                    "name": "Qcur-0",
                                    "ne": [4096, 512, 1, 1],
                                    "op_params": [],
                                    "sources": "q8_0[4096,4096,1,1],f32[4096,512,1,1]",
                                }
                            ],
                        }
                    ]
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    suite = load_grouped_yaml_suite(yaml_path)
    case = suite.op_groups[0].cases[0]

    assert case.dtype == {
        "type": "f32",
        "type_a": "q8_0",
        "type_b": "f32",
        "type_dst": "f32",
        "type_src": "q8_0",
        "type_src0": "q8_0",
        "type_src1": "f32",
    }


def test_model_style_get_rows_sources_derive_index_dtype(tmp_path: Path) -> None:
    yaml_path = tmp_path / "model.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "ops": {
                    "GET_ROWS": [
                        {
                            "dtype": {"type_dst": "f32", "type_src": "q8_0"},
                            "cases": [
                                {
                                    "name": "embd",
                                    "ne": [4096, 512, 1, 1],
                                    "op_params": [],
                                    "sources": "q8_0[4096,128256,1,1],i32[512,1,1,1]",
                                }
                            ],
                        }
                    ]
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    suite = load_grouped_yaml_suite(yaml_path)
    case = suite.op_groups[0].cases[0]

    assert case.dtype == {
        "type": "q8_0",
        "type_dst": "f32",
        "type_idx": "i32",
        "type_src": "q8_0",
        "type_src0": "q8_0",
        "type_src1": "i32",
    }


def test_legacy_rope_dtype_defaults_to_i32_index_dtype(tmp_path: Path) -> None:
    yaml_path = tmp_path / "legacy.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "ops": {
                    "ROPE": [
                        {
                            "dtype": {"type_dst": "f32", "type_src": "f32"},
                            "cases": [
                                {
                                    "ne_a": [128, 32, 2, 1],
                                    "n_dims": 128,
                                    "mode": 0,
                                    "ff": 0,
                                    "fs": 1.0,
                                    "ef": 0.0,
                                    "af": 1.0,
                                    "inplace": 0,
                                    "v": 0,
                                }
                            ],
                        }
                    ]
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    suite = load_grouped_yaml_suite(yaml_path)
    case = suite.op_groups[0].cases[0]

    assert case.dtype == {
        "type": "f32",
        "type_dst": "f32",
        "type_idx": "i32",
        "type_src": "f32",
    }
