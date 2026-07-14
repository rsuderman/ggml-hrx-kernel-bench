from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ggml_hrx_kernel_bench.loom_execution_descriptor import (  # noqa: E402
    DESCRIPTOR_MANIFEST_SCHEMA,
    ROUTE_EXECUTION_ABI_SCHEMA,
    SCHEMA,
    descriptor_from_generated_case,
    execute_prepared,
    prepare_execution,
    run_execution_descriptor_manifest,
    validate_descriptor,
    write_generated_execution_descriptors,
)
from ggml_hrx_kernel_bench.materialized_assets import materialize_asset_root  # noqa: E402


def _descriptor() -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "kernel": "kernels/v2/add/contiguous_1d.loom",
        "root": "@add_f32_contiguous_1d",
        "target": "gfx1100",
        "configs": {
            "hrx2.shape.pointwise.total_size": 4,
            "hrx2.tuning.pointwise.workgroup_size": 32,
        },
        "bindings": [
            {
                "name": "src0",
                "position": 0,
                "kind": "input",
                "dtype": "f32",
                "values": [1.0, 2.0, 3.0, 4.0],
            },
            {
                "name": "src1",
                "position": 1,
                "kind": "input",
                "dtype": "f32",
                "values": [5.0, 6.0, 7.0, 8.0],
            },
            {
                "name": "dst",
                "position": 2,
                "kind": "output",
                "dtype": "f32",
                "values": [0.0, 0.0, 0.0, 0.0],
                "expect": {
                    "mode": "close",
                    "values": [6.0, 8.0, 10.0, 12.0],
                    "atol": 1e-5,
                    "rtol": 1e-5,
                },
            },
        ],
    }


def _write_descriptor(tmp_path: Path, data: dict[str, object] | None = None) -> Path:
    path = tmp_path / "descriptor.json"
    path.write_text(json.dumps(data or _descriptor()) + "\n", encoding="utf-8")
    return path


def _binary_f32_execution_abi(route_id: str) -> dict[str, object]:
    return {
        "schema": ROUTE_EXECUTION_ABI_SCHEMA,
        "route_id": route_id,
        "entries": [
            {
                "position": 0,
                "role": "src0",
                "kind": "input",
                "dtype": "f32",
                "fixture": "src0",
            },
            {
                "position": 1,
                "role": "src1",
                "kind": "input",
                "dtype": "f32",
                "fixture": "src1",
            },
            {
                "position": 2,
                "role": "dst",
                "kind": "output",
                "dtype": "f32",
                "fixture": "dst_init",
                "expect": {
                    "fixture": "expected",
                    "mode": "close",
                },
            },
        ],
    }


def _binary_f16_execution_abi(route_id: str) -> dict[str, object]:
    return {
        "schema": ROUTE_EXECUTION_ABI_SCHEMA,
        "route_id": route_id,
        "entries": [
            {
                "position": 0,
                "role": "src0",
                "kind": "input",
                "dtype": "f16",
                "fixture": "src0",
            },
            {
                "position": 1,
                "role": "src1",
                "kind": "input",
                "dtype": "f16",
                "fixture": "src1",
            },
            {
                "position": 2,
                "role": "dst",
                "kind": "output",
                "dtype": "f16",
                "fixture": "dst_init",
                "expect": {
                    "fixture": "expected",
                    "mode": "close",
                },
            },
        ],
    }


def _get_rows_f32_execution_abi(route_id: str) -> dict[str, object]:
    return {
        "schema": ROUTE_EXECUTION_ABI_SCHEMA,
        "route_id": route_id,
        "entries": [
            {
                "position": 0,
                "role": "src0",
                "kind": "input",
                "dtype": "f32",
                "fixture": "src0",
            },
            {
                "position": 1,
                "role": "src1",
                "kind": "input",
                "dtype": "i32",
                "fixture": "indices",
            },
            {
                "position": 2,
                "role": "dst",
                "kind": "output",
                "dtype": "f32",
                "fixture": "dst_init",
                "expect": {
                    "fixture": "expected",
                    "mode": "close",
                },
            },
        ],
    }


def _set_rows_f32_execution_abi(route_id: str) -> dict[str, object]:
    return {
        "schema": ROUTE_EXECUTION_ABI_SCHEMA,
        "route_id": route_id,
        "entries": [
            {
                "position": 0,
                "role": "src1",
                "kind": "input",
                "dtype": "f32",
                "fixture": "src0",
            },
            {
                "position": 1,
                "role": "src2",
                "kind": "input",
                "dtype": "i32",
                "fixture": "indices",
            },
            {
                "position": 2,
                "role": "dst",
                "kind": "output",
                "dtype": "f32",
                "fixture": "dst_init",
                "expect": {
                    "fixture": "expected",
                    "mode": "close",
                },
            },
        ],
    }


def _unary_f32_execution_abi(route_id: str) -> dict[str, object]:
    return {
        "schema": ROUTE_EXECUTION_ABI_SCHEMA,
        "route_id": route_id,
        "entries": [
            {
                "position": 0,
                "role": "src0",
                "kind": "input",
                "dtype": "f32",
                "fixture": "src0",
            },
            {
                "position": 1,
                "role": "dst",
                "kind": "output",
                "dtype": "f32",
                "fixture": "dst_init",
                "expect": {
                    "fixture": "expected",
                    "mode": "close",
                },
            },
        ],
    }


def _unary_f16_execution_abi(route_id: str) -> dict[str, object]:
    return {
        "schema": ROUTE_EXECUTION_ABI_SCHEMA,
        "route_id": route_id,
        "entries": [
            {
                "position": 0,
                "role": "src0",
                "kind": "input",
                "dtype": "f16",
                "fixture": "src0",
            },
            {
                "position": 1,
                "role": "dst",
                "kind": "output",
                "dtype": "f16",
                "fixture": "dst_init",
                "expect": {
                    "fixture": "expected",
                    "mode": "close",
                },
            },
        ],
    }


def _scale_f32_execution_abi(route_id: str) -> dict[str, object]:
    return {
        "schema": ROUTE_EXECUTION_ABI_SCHEMA,
        "route_id": route_id,
        "entries": [
            {
                "position": 0,
                "role": "scale",
                "kind": "scalar",
                "dtype": "f32",
                "value": 0.625,
            },
            {
                "position": 1,
                "role": "bias",
                "kind": "scalar",
                "dtype": "f32",
                "value": -0.125,
            },
            {
                "position": 2,
                "role": "src0",
                "kind": "input",
                "dtype": "f32",
                "fixture": "src0",
            },
            {
                "position": 3,
                "role": "dst",
                "kind": "output",
                "dtype": "f32",
                "fixture": "dst_init",
                "expect": {
                    "fixture": "expected",
                    "mode": "close",
                },
            },
        ],
    }


def _generated_add_config() -> dict[str, object]:
    route_id = "add_f32_generic_4d"
    return {
        "kernel": "add_f32",
        "params": ["d0", "d1", "d2", "d3"],
        "cases": [[4, 1, 1, 1]],
        "route_id": route_id,
        "execution_abi": _binary_f32_execution_abi(route_id),
    }


def _generated_add_f16_config() -> dict[str, object]:
    route_id = "add_f16_generic_4d"
    return {
        "kernel": "add_f16",
        "params": ["d0", "d1", "d2", "d3"],
        "cases": [[4, 1, 1, 1]],
        "route_id": route_id,
        "execution_abi": _binary_f16_execution_abi(route_id),
    }


def _generated_get_rows_f32_config() -> dict[str, object]:
    route_id = "get_rows_f32_embedding_rows_descriptor_4d"
    return {
        "kernel": "get_rows_f32",
        "params": ["d0", "d1", "d2", "d3", "src0_d1", "src1_d0", "src1_d1"],
        "cases": [[1, 2, 1, 1, 8, 2, 1]],
        "route_id": route_id,
        "execution_abi": _get_rows_f32_execution_abi(route_id),
    }


def _generated_set_rows_f32_config() -> dict[str, object]:
    route_id = "set_rows_f32_f32_descriptor_4d"
    return {
        "kernel": "set_rows_f32",
        "params": [
            "d0",
            "d1",
            "d2",
            "d3",
            "src1_d1",
            "src2_d0",
            "src2_d1",
            "src2_d2",
            "src2_d3",
        ],
        "cases": [[3, 3, 14, 3, 2, 2, 7, 1, 1]],
        "route_id": route_id,
        "execution_abi": _set_rows_f32_execution_abi(route_id),
    }


def _generated_cont_set_rows_f32_f16_config() -> dict[str, object]:
    route_id = "cont_set_rows_f32_f16_n1024_dst8192_contiguous_4d"
    return {
        "kernel": "cont_set_rows_f32",
        "params": ["d0", "d1", "d2", "d3", "src0_d1", "src1_d0", "src1_d1", "src1_d2", "src1_d3"],
        "cases": [[1024, 8192, 1, 1, 512, 512, 1, 1, 1]],
        "route_id": route_id,
        "execution_abi": {
            "schema": ROUTE_EXECUTION_ABI_SCHEMA,
            "route_id": route_id,
            "entries": [
                {
                    "position": 0,
                    "role": "src0",
                    "kind": "input",
                    "dtype": "f32",
                    "fixture": "src0",
                },
                {
                    "position": 1,
                    "role": "src1",
                    "kind": "input",
                    "dtype": "i32",
                    "fixture": "indices",
                },
                {
                    "position": 2,
                    "role": "dst",
                    "kind": "output",
                    "dtype": "f16",
                    "fixture": "dst_init",
                    "expect": {
                        "fixture": "expected",
                        "mode": "close",
                    },
                },
            ],
        },
    }


def _generated_scale_config() -> dict[str, object]:
    route_id = "scale_f32_contiguous_4d"
    return {
        "kernel": "scale_f32",
        "params": ["d0", "d1", "d2", "d3"],
        "cases": [[4, 1, 1, 1]],
        "route_id": route_id,
        "execution_abi": _scale_f32_execution_abi(route_id),
    }


def _generated_rms_norm_config(eps: float) -> dict[str, object]:
    route_id = "rms_norm_f32_contiguous_4d"
    return {
        "kernel": "rms_norm_f32",
        "params": ["d0", "d1", "d2", "d3"],
        "cases": [[4, 2, 1, 1]],
        "route_id": route_id,
        "execution_abi": {
            "schema": ROUTE_EXECUTION_ABI_SCHEMA,
            "route_id": route_id,
            "entries": [
                {
                    "position": 0,
                    "role": "eps",
                    "kind": "scalar",
                    "dtype": "f32",
                    "value": eps,
                },
                {
                    "position": 1,
                    "role": "src0",
                    "kind": "input",
                    "dtype": "f32",
                    "fixture": "src",
                },
                {
                    "position": 2,
                    "role": "dst",
                    "kind": "output",
                    "dtype": "f32",
                    "fixture": "dst_init",
                    "expect": {
                        "fixture": "expected",
                        "mode": "close",
                    },
                },
            ],
        },
    }


def _generated_cont_config() -> dict[str, object]:
    route_id = "cont_f32_contiguous_4d"
    return {
        "kernel": "cont_f32",
        "params": ["d0", "d1", "d2", "d3"],
        "cases": [[2, 3, 5, 1]],
        "route_id": route_id,
        "execution_abi": _unary_f32_execution_abi(route_id),
    }


def _copy_execution_abi(route_id: str, *, src_dtype: str, dst_dtype: str) -> dict[str, object]:
    return {
        "schema": ROUTE_EXECUTION_ABI_SCHEMA,
        "route_id": route_id,
        "entries": [
            {
                "position": 0,
                "role": "src0",
                "kind": "input",
                "dtype": src_dtype,
                "fixture": "src0",
            },
            {
                "position": 1,
                "role": "dst",
                "kind": "output",
                "dtype": dst_dtype,
                "fixture": "dst_init",
                "expect": {
                    "fixture": "expected",
                    "mode": "close",
                },
            },
        ],
    }


def _mul_mat_execution_abi(route_id: str, *, src0_dtype: str) -> dict[str, object]:
    return {
        "schema": ROUTE_EXECUTION_ABI_SCHEMA,
        "route_id": route_id,
        "entries": [
            {
                "position": 0,
                "role": "src0",
                "kind": "input",
                "dtype": src0_dtype,
                "fixture": "src0",
            },
            {
                "position": 1,
                "role": "src1",
                "kind": "input",
                "dtype": "f32",
                "fixture": "src1",
            },
            {
                "position": 2,
                "role": "dst",
                "kind": "output",
                "dtype": "f32",
                "fixture": "dst_init",
                "expect": {
                    "fixture": "expected",
                    "mode": "close",
                },
            },
        ],
    }


def _generated_copy_config(src_dtype: str, dst_dtype: str, *, route_suffix: str = "contiguous_1d") -> dict[str, object]:
    route_id = f"copy_{src_dtype}_{dst_dtype}_{route_suffix}"
    return {
        "kernel": f"copy_{src_dtype}_{dst_dtype}",
        "params": ["d0", "d1", "d2", "d3"],
        "cases": [[4, 1, 1, 1]],
        "route_id": route_id,
        "execution_abi": _copy_execution_abi(route_id, src_dtype=src_dtype, dst_dtype=dst_dtype),
    }


def _generated_mul_mat_config(
    family: str,
    route_id: str,
    params: list[str],
    case_values: list[int],
    *,
    src0_dtype: str,
) -> dict[str, object]:
    return {
        "kernel": family,
        "params": params,
        "cases": [case_values],
        "route_id": route_id,
        "execution_abi": _mul_mat_execution_abi(route_id, src0_dtype=src0_dtype),
    }


def _generated_swiglu_config() -> dict[str, object]:
    route_id = "swiglu_f32_packed_contiguous_4d"
    return {
        "kernel": "swiglu_f32",
        "params": ["d0", "d1", "d2", "d3", "src0_d0"],
        "cases": [[4, 2, 1, 1, 8]],
        "route_id": route_id,
        "execution_abi": _unary_f32_execution_abi(route_id),
    }


def _soft_max_f32_execution_abi(route_id: str, *, masked: bool) -> dict[str, object]:
    entries: list[dict[str, object]] = [
        {
            "position": 0,
            "role": "scale",
            "kind": "scalar",
            "dtype": "f32",
            "value": 0.75,
        },
        {
            "position": 1,
            "role": "src0",
            "kind": "input",
            "dtype": "f32",
            "fixture": "src0",
        },
    ]
    if masked:
        entries.append(
            {
                "position": 2,
                "role": "mask",
                "kind": "input",
                "dtype": "f32",
                "fixture": "mask",
            }
        )
    entries.append(
        {
            "position": 3 if masked else 2,
            "role": "dst",
            "kind": "output",
            "dtype": "f32",
            "fixture": "dst_init",
            "expect": {
                "fixture": "expected",
                "mode": "close",
            },
        }
    )
    return {
        "schema": ROUTE_EXECUTION_ABI_SCHEMA,
        "route_id": route_id,
        "entries": entries,
    }


def _generated_soft_max_config(*, masked: bool = False) -> dict[str, object]:
    route_id = "soft_max_f32_mask_contiguous_4d" if masked else "soft_max_f32_contiguous_4d"
    return {
        "kernel": "soft_max_f32",
        "params": ["d0", "d1", "d2", "d3"],
        "cases": [[4, 2, 1, 1]],
        "route_id": route_id,
        "execution_abi": _soft_max_f32_execution_abi(route_id, masked=masked),
    }


def _rope_f32_execution_abi(route_id: str) -> dict[str, object]:
    return {
        "schema": ROUTE_EXECUTION_ABI_SCHEMA,
        "route_id": route_id,
        "entries": [
            {
                "position": 0,
                "role": "theta_scale",
                "kind": "scalar",
                "dtype": "f32",
                "value": 0.75,
            },
            {
                "position": 1,
                "role": "freq_scale",
                "kind": "scalar",
                "dtype": "f32",
                "value": 1.1,
            },
            {
                "position": 2,
                "role": "attn_factor",
                "kind": "scalar",
                "dtype": "f32",
                "value": 0.9,
            },
            {
                "position": 3,
                "role": "src0",
                "kind": "input",
                "dtype": "f32",
                "fixture": "src0",
            },
            {
                "position": 4,
                "role": "src1",
                "kind": "input",
                "dtype": "i32",
                "fixture": "positions",
            },
            {
                "position": 5,
                "role": "dst",
                "kind": "output",
                "dtype": "f32",
                "fixture": "dst_init",
                "expect": {
                    "fixture": "expected",
                    "mode": "close",
                },
            },
        ],
    }


def _generated_rope_config(*, neox: bool = False) -> dict[str, object]:
    route_id = "rope_neox_f32_n64_h128_t2_contiguous_4d" if neox else "rope_f32_normal_n128_h32_t2_contiguous_4d"
    return {
        "kernel": "rope_neox_f32" if neox else "rope_f32",
        "params": [
            "d0",
            "d1",
            "d2",
            "d3",
            "src1_d0",
            "src1_d1",
            "rope.ncols",
            *([] if neox else ["rope.n_dims"]),
            "rope.nheads",
            "rope.ntokens",
            "rope.src0_head_stride",
            "rope.src0_token_stride",
            "rope.dst_head_stride",
            "rope.dst_token_stride",
            "rope.pos_token_stride",
        ],
        "cases": [
            [64, 1, 2, 1, 1, 1, 64, 1, 2, 64, 64, 64, 64, 1]
            if neox
            else [128, 32, 2, 1, 1, 1, 128, 128, 32, 2, 128, 4096, 128, 4096, 1]
        ],
        "route_id": route_id,
        "execution_abi": _rope_f32_execution_abi(route_id),
    }


def _generated_unary_config(kernel: str) -> dict[str, object]:
    route_id = f"{kernel}_contiguous_4d"
    return {
        "kernel": kernel,
        "params": ["d0", "d1", "d2", "d3"],
        "cases": [[4, 1, 1, 1]],
        "route_id": route_id,
        "execution_abi": _unary_f32_execution_abi(route_id),
    }


def _generated_unary_f16_config(kernel: str, *, route_id: str | None = None) -> dict[str, object]:
    route_id = route_id or f"{kernel}_contiguous_4d"
    return {
        "kernel": kernel,
        "params": ["d0", "d1", "d2", "d3"],
        "cases": [[4, 1, 1, 1]],
        "route_id": route_id,
        "execution_abi": _unary_f16_execution_abi(route_id),
    }


def _generated_binary_config(kernel: str) -> dict[str, object]:
    route_id = f"{kernel}_generic_4d"
    return {
        "kernel": kernel,
        "params": ["d0", "d1", "d2", "d3"],
        "cases": [[4, 1, 1, 1]],
        "route_id": route_id,
        "execution_abi": _binary_f32_execution_abi(route_id),
    }


def test_validate_descriptor_rejects_duplicate_positions() -> None:
    data = _descriptor()
    bindings = data["bindings"]
    assert isinstance(bindings, list)
    bindings[1]["position"] = 0

    with pytest.raises(RuntimeError, match="duplicates 0"):
        validate_descriptor(data)


def test_prepare_execution_materializes_inline_f32_fixtures(tmp_path: Path) -> None:
    descriptor_path = _write_descriptor(tmp_path)
    prepared = prepare_execution(
        descriptor_path=descriptor_path,
        fixture_dir=tmp_path / "fixtures",
        output_path=tmp_path / "result.json",
        runner=tmp_path / "runner",
        loom_link=tmp_path / "loom-link",
        iree_run_loom=tmp_path / "iree-run-loom",
        repo_root=Path.cwd(),
        execute_iree_run_loom=True,
    )

    assert np.load(tmp_path / "fixtures" / "src0.npy").tolist() == [1.0, 2.0, 3.0, 4.0]
    assert np.load(tmp_path / "fixtures" / "dst_expected.npy").tolist() == [
        6.0,
        8.0,
        10.0,
        12.0,
    ]
    assert prepared.command[:2] == [str(tmp_path / "runner"), "--kernel"]
    assert "--execute-iree-run-loom-command" in prepared.command
    assert "--loom-link" in prepared.command
    assert "--iree-run-loom" in prepared.command
    assert "--linked-kernel-output" in prepared.command
    assert "--binding" in prepared.command
    assert "--expect" in prepared.command
    assert "2:close:" in " ".join(prepared.command)


def test_prepare_execution_requires_iree_run_loom_for_execution(tmp_path: Path) -> None:
    descriptor_path = _write_descriptor(tmp_path)

    with pytest.raises(RuntimeError, match="execute requires an explicit iree-run-loom path"):
        prepare_execution(
            descriptor_path=descriptor_path,
            fixture_dir=tmp_path / "fixtures",
            output_path=tmp_path / "result.json",
            runner=tmp_path / "runner",
            loom_link=tmp_path / "loom-link",
            iree_run_loom=None,
            repo_root=Path.cwd(),
            execute_iree_run_loom=True,
        )


def test_prepare_execution_uses_descriptor_relative_fixture_paths(tmp_path: Path) -> None:
    np.save(tmp_path / "src.npy", np.asarray([1.0, 2.0], dtype=np.float32), allow_pickle=False)
    np.save(tmp_path / "dst.npy", np.asarray([0.0, 0.0], dtype=np.float32), allow_pickle=False)
    np.save(tmp_path / "expected.npy", np.asarray([1.0, 2.0], dtype=np.float32), allow_pickle=False)
    data = {
        "schema": SCHEMA,
        "kernel": "kernels/v2/copy/contiguous_1d.loom",
        "root": "@copy_f32_f32_contiguous_1d",
        "target": "gfx1100",
        "bindings": [
            {"position": 0, "kind": "input", "dtype": "f32", "path": "src.npy"},
            {
                "position": 1,
                "kind": "output",
                "dtype": "f32",
                "path": "dst.npy",
                "expect": {"mode": "close", "path": "expected.npy", "atol": 0.0, "rtol": 0.0},
            },
        ],
    }
    descriptor_path = _write_descriptor(tmp_path, data)

    prepared = prepare_execution(
        descriptor_path=descriptor_path,
        fixture_dir=tmp_path / "fixtures",
        output_path=tmp_path / "result.json",
        runner="runner",
        loom_link=None,
        iree_run_loom=None,
        repo_root=Path.cwd(),
    )

    command = " ".join(prepared.command)
    assert f"0:input:f32:2:{tmp_path / 'src.npy'}" in command
    assert f"1:output:f32:2:{tmp_path / 'dst.npy'}" in command
    assert f"1:close:{tmp_path / 'expected.npy'}:0.0:0.0" in command


def test_prepare_execution_emits_scalar_flags(tmp_path: Path) -> None:
    np.save(tmp_path / "src.npy", np.asarray([1.0, 2.0], dtype=np.float32), allow_pickle=False)
    np.save(tmp_path / "dst.npy", np.asarray([0.0, 0.0], dtype=np.float32), allow_pickle=False)
    np.save(tmp_path / "expected.npy", np.asarray([0.5, 1.125], dtype=np.float32), allow_pickle=False)
    data = {
        "schema": SCHEMA,
        "kernel": "kernels/v2/scale/contiguous_4d.loom",
        "root": "@scale_f32",
        "target": "gfx1100",
        "scalars": [
            {"position": 0, "name": "scale", "dtype": "f32", "value": 0.625},
            {"position": 1, "name": "bias", "dtype": "f32", "value": -0.125},
        ],
        "bindings": [
            {"position": 2, "kind": "input", "dtype": "f32", "path": "src.npy"},
            {
                "position": 3,
                "kind": "output",
                "dtype": "f32",
                "path": "dst.npy",
                "expect": {"mode": "close", "path": "expected.npy", "atol": 0.0, "rtol": 0.0},
            },
        ],
    }
    descriptor_path = _write_descriptor(tmp_path, data)

    prepared = prepare_execution(
        descriptor_path=descriptor_path,
        fixture_dir=tmp_path / "fixtures",
        output_path=tmp_path / "result.json",
        runner="runner",
        loom_link=None,
        iree_run_loom=None,
        repo_root=Path.cwd(),
    )

    command = " ".join(prepared.command)
    assert "0:f32:0.625" in command
    assert "1:f32:-0.125" in command
    assert f"2:input:f32:2:{tmp_path / 'src.npy'}" in command
    assert f"3:close:{tmp_path / 'expected.npy'}:0.0:0.0" in command


def test_execute_prepared_invokes_runner(tmp_path: Path) -> None:
    descriptor_path = _write_descriptor(tmp_path)
    runner = tmp_path / "fake-runner.py"
    capture = tmp_path / "capture.json"
    runner.write_text(
        "\n".join(
            [
                "import json",
                "import sys",
                f"open({str(capture)!r}, 'w').write(json.dumps(sys.argv[1:]))",
                "print('fake runner called')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    prepared = prepare_execution(
        descriptor_path=descriptor_path,
        fixture_dir=tmp_path / "fixtures",
        output_path=tmp_path / "result.json",
        runner=sys.executable,
        loom_link=None,
        iree_run_loom=None,
        repo_root=Path.cwd(),
    )
    command = list(prepared.command)
    command.insert(1, str(runner))
    prepared = prepared.__class__(
        descriptor_path=prepared.descriptor_path,
        fixture_dir=prepared.fixture_dir,
        output_path=prepared.output_path,
        command=command,
    )

    result = execute_prepared(prepared)

    assert result.returncode == 0
    assert result.stdout == "fake runner called\n"
    captured = json.loads(capture.read_text(encoding="utf-8"))
    assert "--kernel" in captured


def test_infra_script_prints_command_without_execution(tmp_path: Path) -> None:
    descriptor_path = _write_descriptor(tmp_path)
    script = Path("tests/infra/run_loom_execution_descriptor.py")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            str(descriptor_path),
            "--fixture-dir",
            str(tmp_path / "fixtures"),
            "--runner",
            "runner",
            "--repo-root",
            str(Path.cwd()),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["command"][0] == "runner"
    assert "--execute-iree-run-loom-command" not in payload["command"]


def test_descriptor_from_generated_add_case_emits_compact_payload(tmp_path: Path) -> None:
    result = descriptor_from_generated_case(
        config_data=_generated_add_config(),
        case_id="d0-4-d1-1-d2-1-d3-1",
        case_values=[4, 1, 1, 1],
        kernel_dir=Path("kernels/v2"),
        routing_dir=Path("catalog/v2"),
        target="gfx1100",
        max_elements=32,
        oracle_fixture_dir=tmp_path / "oracle-fixtures",
        descriptor_dir=tmp_path,
    )

    assert result.status == "emitted"
    assert result.descriptor is not None
    descriptor = result.descriptor
    assert descriptor["kernel"] == "kernels/v2/add/generic_4d.loom"
    assert descriptor["root"] == "@add_f32_generic_4d"
    assert descriptor["workgroup_count"] == [1, 1, 1]
    assert descriptor["configs"]["@hrx2.shape.add4d.ne0"] == "4"
    assert descriptor["metadata"]["element_counts"] == {"src0": 4, "src1": 4, "dst": 4}
    assert descriptor["metadata"]["execution_abi"]["schema"] == ROUTE_EXECUTION_ABI_SCHEMA
    assert descriptor["metadata"]["oracle"]["status"] == "fixtures_ready"
    assert descriptor["bindings"][0]["path"] == "oracle-fixtures/src0.npy"
    assert descriptor["bindings"][1]["path"] == "oracle-fixtures/src1.npy"
    assert descriptor["bindings"][2]["path"] == "oracle-fixtures/dst_init.npy"
    assert descriptor["bindings"][2]["expect"]["path"] == "oracle-fixtures/expected.npy"
    src0 = np.load(tmp_path / descriptor["bindings"][0]["path"])
    src1 = np.load(tmp_path / descriptor["bindings"][1]["path"])
    expected = np.load(tmp_path / descriptor["bindings"][2]["expect"]["path"])
    assert expected.tolist() == (src0 + src1).astype(np.float32).tolist()


def test_descriptor_from_generated_mul_case_uses_oracle_fixture_paths(tmp_path: Path) -> None:
    result = descriptor_from_generated_case(
        config_data=_generated_binary_config("mul_f32"),
        case_id="d0-4-d1-1-d2-1-d3-1",
        case_values=[4, 1, 1, 1],
        kernel_dir=Path("kernels/v2"),
        routing_dir=Path("catalog/v2"),
        target="gfx1100",
        max_elements=32,
        oracle_fixture_dir=tmp_path / "oracle-fixtures",
        descriptor_dir=tmp_path,
    )

    assert result.status == "emitted"
    assert result.descriptor is not None
    descriptor = result.descriptor
    assert descriptor["kernel"] == "kernels/v2/mul/generic_4d.loom"
    assert descriptor["root"] == "@mul_f32_generic_4d"
    assert descriptor["metadata"]["oracle"]["status"] == "fixtures_ready"
    src0 = np.load(tmp_path / descriptor["bindings"][0]["path"])
    src1 = np.load(tmp_path / descriptor["bindings"][1]["path"])
    expected = np.load(tmp_path / descriptor["bindings"][2]["expect"]["path"])
    assert expected.tolist() == (src0 * src1).astype(np.float32).tolist()


def test_descriptor_from_generated_unary_f32_case_uses_abi_bindings(tmp_path: Path) -> None:
    assets = materialize_asset_root(tmp_path / "assets", force=True)
    result = descriptor_from_generated_case(
        config_data=_generated_unary_config("abs_f32"),
        case_id="d0-4-d1-1-d2-1-d3-1",
        case_values=[4, 1, 1, 1],
        kernel_dir=assets / "kernels" / "v2",
        routing_dir=assets / "catalog" / "v2",
        target="gfx1100",
        max_elements=32,
        oracle_fixture_dir=tmp_path / "oracle-fixtures",
        descriptor_dir=tmp_path,
    )

    assert result.status == "emitted"
    assert result.descriptor is not None
    descriptor = result.descriptor
    assert descriptor["kernel"] == str(assets / "kernels" / "v2" / "abs" / "contiguous_4d.loom")
    assert descriptor["root"] == "@abs_f32"
    assert descriptor["metadata"]["route_id"] == "abs_f32_contiguous_4d"
    assert [binding["position"] for binding in descriptor["bindings"]] == [0, 1]
    assert descriptor["bindings"][0]["name"] == "src0"
    assert descriptor["bindings"][1]["name"] == "dst"
    src0 = np.load(tmp_path / descriptor["bindings"][0]["path"])
    expected = np.load(tmp_path / descriptor["bindings"][1]["expect"]["path"])
    assert expected.tolist() == np.abs(src0).astype(np.float32).tolist()


def test_descriptor_from_generated_scale_f32_case_uses_scalar_abi(tmp_path: Path) -> None:
    assets = materialize_asset_root(tmp_path / "assets", force=True)
    result = descriptor_from_generated_case(
        config_data=_generated_scale_config(),
        case_id="d0-4-d1-1-d2-1-d3-1",
        case_values=[4, 1, 1, 1],
        kernel_dir=assets / "kernels" / "v2",
        routing_dir=assets / "catalog" / "v2",
        target="gfx1100",
        max_elements=32,
        oracle_fixture_dir=tmp_path / "oracle-fixtures",
        descriptor_dir=tmp_path,
    )

    assert result.status == "emitted"
    assert result.descriptor is not None
    descriptor = result.descriptor
    assert descriptor["root"] == "@scale_f32"
    assert descriptor["scalars"] == [
        {"name": "scale", "position": 0, "dtype": "f32", "value": 0.625},
        {"name": "bias", "position": 1, "dtype": "f32", "value": -0.125},
    ]
    assert [binding["position"] for binding in descriptor["bindings"]] == [2, 3]
    src0 = np.load(tmp_path / descriptor["bindings"][0]["path"])
    expected = np.load(tmp_path / descriptor["bindings"][1]["expect"]["path"])
    assert expected.tolist() == (src0 * np.float32(0.625) + np.float32(-0.125)).astype(np.float32).tolist()


def test_descriptor_from_generated_rms_norm_f32_case_uses_eps_scalar_abi(tmp_path: Path) -> None:
    assets = materialize_asset_root(tmp_path / "assets", force=True)
    result = descriptor_from_generated_case(
        config_data=_generated_rms_norm_config(0.0001),
        case_id="d0-4-d1-2-d2-1-d3-1",
        case_values=[4, 2, 1, 1],
        kernel_dir=assets / "kernels" / "v2",
        routing_dir=assets / "catalog" / "v2",
        target="gfx1100",
        max_elements=32,
        oracle_fixture_dir=tmp_path / "oracle-fixtures",
        descriptor_dir=tmp_path,
    )

    assert result.status == "emitted", result.reason
    assert result.descriptor is not None
    descriptor = result.descriptor
    assert descriptor["root"] == "@hrx2_rms_norm_f32"
    assert descriptor["scalars"] == [
        {"name": "eps", "position": 0, "dtype": "f32", "value": 0.0001},
    ]
    assert [binding["name"] for binding in descriptor["bindings"]] == ["src0", "dst"]
    assert [binding["position"] for binding in descriptor["bindings"]] == [1, 2]
    src0 = np.load(tmp_path / descriptor["bindings"][0]["path"]).reshape(2, 4)
    expected = np.load(tmp_path / descriptor["bindings"][1]["expect"]["path"]).reshape(2, 4)
    scale = np.reciprocal(np.sqrt(np.mean(src0 * src0, axis=1, keepdims=True) + np.float32(0.0001)))
    assert np.allclose(expected, (src0 * scale).astype(np.float32), atol=1e-4, rtol=1e-4)


def test_descriptor_from_generated_cont_f32_case_uses_unary_buffer_abi(tmp_path: Path) -> None:
    assets = materialize_asset_root(tmp_path / "assets", force=True)
    result = descriptor_from_generated_case(
        config_data=_generated_cont_config(),
        case_id="d0-2-d1-3-d2-5-d3-1",
        case_values=[2, 3, 5, 1],
        kernel_dir=assets / "kernels" / "v2",
        routing_dir=assets / "catalog" / "v2",
        target="gfx1100",
        max_elements=64,
        oracle_fixture_dir=tmp_path / "oracle-fixtures",
        descriptor_dir=tmp_path,
    )

    assert result.status == "emitted", result.reason
    assert result.descriptor is not None
    descriptor = result.descriptor
    assert descriptor["root"] == "@cont_f32"
    assert [binding["name"] for binding in descriptor["bindings"]] == ["src0", "dst"]
    assert [binding["dtype"] for binding in descriptor["bindings"]] == ["f32", "f32"]
    src0 = np.load(tmp_path / descriptor["bindings"][0]["path"])
    expected = np.load(tmp_path / descriptor["bindings"][1]["expect"]["path"])
    assert expected.tolist() == src0.tolist()

    descriptor_path = _write_descriptor(tmp_path, descriptor)
    prepared = prepare_execution(
        descriptor_path=descriptor_path,
        fixture_dir=tmp_path / "fixtures",
        output_path=tmp_path / "result.json",
        runner="runner",
        loom_link=None,
        iree_run_loom=None,
        repo_root=tmp_path,
    )
    command = prepared.command
    assert f"0:input:f32:30:{tmp_path / descriptor['bindings'][0]['path']}" in command
    assert f"1:output:f32:30:{tmp_path / descriptor['bindings'][1]['path']}" in command


@pytest.mark.parametrize(
    ("src_dtype", "dst_dtype", "src_storage_dtype", "dst_storage_dtype"),
    [
        ("f32", "f32", np.float32, np.float32),
        ("f32", "bf16", np.float32, np.int16),
        ("bf16", "f32", np.int16, np.float32),
        ("f16", "bf16", np.int16, np.int16),
    ],
)
def test_descriptor_from_generated_copy_case_uses_cast_storage_dtypes(
    tmp_path: Path,
    src_dtype: str,
    dst_dtype: str,
    src_storage_dtype: object,
    dst_storage_dtype: object,
) -> None:
    assets = materialize_asset_root(tmp_path / "assets", force=True)
    result = descriptor_from_generated_case(
        config_data=_generated_copy_config(src_dtype, dst_dtype),
        case_id="d0-4-d1-1-d2-1-d3-1",
        case_values=[4, 1, 1, 1],
        kernel_dir=assets / "kernels" / "v2",
        routing_dir=assets / "catalog" / "v2",
        target="gfx1100",
        max_elements=32,
        oracle_fixture_dir=tmp_path / "oracle-fixtures",
        descriptor_dir=tmp_path,
    )

    assert result.status == "emitted", result.reason
    assert result.descriptor is not None
    descriptor = result.descriptor
    assert descriptor["root"] == f"@copy_{src_dtype}_{dst_dtype}_contiguous_1d"
    assert [binding["name"] for binding in descriptor["bindings"]] == ["src0", "dst"]
    assert [binding["dtype"] for binding in descriptor["bindings"]] == [src_dtype, dst_dtype]
    assert descriptor["metadata"]["element_counts"] == {"dst": 4, "src0": 4}
    src0 = np.load(tmp_path / descriptor["bindings"][0]["path"])
    dst_init = np.load(tmp_path / descriptor["bindings"][1]["path"])
    expected = np.load(tmp_path / descriptor["bindings"][1]["expect"]["path"])
    assert src0.dtype == src_storage_dtype
    assert dst_init.dtype == dst_storage_dtype
    assert expected.dtype == dst_storage_dtype

    descriptor_path = _write_descriptor(tmp_path, descriptor)
    prepared = prepare_execution(
        descriptor_path=descriptor_path,
        fixture_dir=tmp_path / "fixtures",
        output_path=tmp_path / "result.json",
        runner="runner",
        loom_link=None,
        iree_run_loom=None,
        repo_root=tmp_path,
    )
    command = prepared.command
    assert f"0:input:{src_dtype}:4:{tmp_path / descriptor['bindings'][0]['path']}" in command
    assert f"1:output:{dst_dtype}:4:{tmp_path / descriptor['bindings'][1]['path']}" in command


@pytest.mark.parametrize(
    (
        "family",
        "route_id",
        "params",
        "case_values",
        "src0_dtype",
        "src0_storage_dtype",
        "expected_root",
    ),
    [
        (
            "mul_mat_f32_f32",
            "mul_mat_f32_f32_contiguous_4d",
            ["d0", "d1", "d2", "d3", "src0_d0", "src1_d0", "k", "rows", "cols"],
            [16, 16, 1, 1, 4, 4, 4, 16, 16],
            "f32",
            np.float32,
            "@hrx2_mul_mat_f32_f32_static",
        ),
        (
            "mul_mat_f16_f32_batched",
            "mul_mat_f16_f32_batched_contiguous_4d",
            [
                "d0",
                "d1",
                "d2",
                "d3",
                "src0_d0",
                "src0_d1",
                "src1_d0",
                "k",
                "rows",
                "cols",
                "src1_d2_stride",
                "src1_d3_stride",
                "dst_d2_stride",
                "dst_d3_stride",
            ],
            [16, 1, 1, 1, 4, 16, 4, 4, 16, 1, 4, 4, 16, 16],
            "f16",
            np.int16,
            "@hrx2_mul_mat_f16_f32_batched",
        ),
        (
            "mul_mat_f16_f32_tiled_batched",
            "mul_mat_f16_f32_tiled_batched_4d",
            ["d0", "d1", "d2", "d3", "src0_d0", "src0_d1", "src0_d3", "src1_d0"],
            [16, 1, 1, 2, 4, 16, 1, 4],
            "f16",
            np.int16,
            "@hrx2_mul_mat_f16_f32_tiled_batched",
        ),
    ],
)
def test_descriptor_from_generated_mul_mat_float_case_uses_kernel_abi(
    tmp_path: Path,
    family: str,
    route_id: str,
    params: list[str],
    case_values: list[int],
    src0_dtype: str,
    src0_storage_dtype: object,
    expected_root: str,
) -> None:
    assets = materialize_asset_root(tmp_path / "assets", force=True)
    result = descriptor_from_generated_case(
        config_data=_generated_mul_mat_config(family, route_id, params, case_values, src0_dtype=src0_dtype),
        case_id="mul-mat-small",
        case_values=case_values,
        kernel_dir=assets / "kernels" / "v2",
        routing_dir=assets / "catalog" / "v2",
        target="gfx1100",
        max_elements=65536,
        oracle_fixture_dir=tmp_path / "oracle-fixtures",
        descriptor_dir=tmp_path,
    )

    assert result.status == "emitted", result.reason
    assert result.descriptor is not None
    descriptor = result.descriptor
    assert descriptor["root"] == expected_root
    assert [binding["name"] for binding in descriptor["bindings"]] == ["src0", "src1", "dst"]
    assert [binding["dtype"] for binding in descriptor["bindings"]] == [src0_dtype, "f32", "f32"]
    src0 = np.load(tmp_path / descriptor["bindings"][0]["path"])
    src1 = np.load(tmp_path / descriptor["bindings"][1]["path"])
    expected = np.load(tmp_path / descriptor["bindings"][2]["expect"]["path"])
    assert src0.dtype == src0_storage_dtype
    assert src1.dtype == np.float32
    assert expected.dtype == np.float32

    descriptor_path = _write_descriptor(tmp_path, descriptor)
    prepared = prepare_execution(
        descriptor_path=descriptor_path,
        fixture_dir=tmp_path / "fixtures",
        output_path=tmp_path / "result.json",
        runner="runner",
        loom_link=None,
        iree_run_loom=None,
        repo_root=tmp_path,
    )
    command = prepared.command
    assert f"0:input:{src0_dtype}:{src0.size}:{tmp_path / descriptor['bindings'][0]['path']}" in command
    assert f"2:output:f32:{expected.size}:{tmp_path / descriptor['bindings'][2]['path']}" in command


@pytest.mark.parametrize(
    ("family", "route_id", "params", "case_values", "src0_dtype", "expected_root"),
    [
        (
            "mul_mat_q4_k_f32",
            "mul_mat_q4_k_f32_direct_contiguous_4d",
            ["d0", "d1", "d2", "d3", "src0_d0", "src1_d0", "k", "rows", "cols"],
            [16, 16, 1, 1, 256, 256, 256, 16, 16],
            "q4_k",
            "@hrx2_mul_mat_q4_k_f32_static",
        ),
        (
            "mul_mat_q5_k_f32",
            "mul_mat_q5_k_f32_dot16_contiguous_cols1_4d",
            ["d0", "d1", "d2", "d3", "src0_d0", "src0_d1", "src1_d0", "k", "rows", "cols"],
            [16, 1, 1, 1, 256, 16, 256, 256, 16, 1],
            "q5_k",
            "@hrx2_mul_mat_q5_k_f32_dot16_static",
        ),
        (
            "mul_mat_q6_k_f32",
            "mul_mat_q6_k_f32_direct_contiguous_4d",
            ["d0", "d1", "d2", "d3", "src0_d0", "src0_d1", "src1_d0", "k", "rows", "cols"],
            [16, 1, 1, 1, 256, 16, 256, 256, 16, 1],
            "q6_k",
            "@hrx2_mul_mat_q6_k_f32_static",
        ),
        (
            "mul_mat_q8_0_f32",
            "mul_mat_q8_0_f32_contiguous_4d",
            ["d0", "d1", "d2", "d3", "src0_d0", "src1_d0", "k", "rows", "cols"],
            [16, 16, 1, 1, 256, 256, 256, 16, 16],
            "q8_0",
            "@hrx2_mul_mat_q8_0_f32_static",
        ),
    ],
)
def test_descriptor_from_generated_mul_mat_quantized_case_uses_int8_storage(
    tmp_path: Path,
    family: str,
    route_id: str,
    params: list[str],
    case_values: list[int],
    src0_dtype: str,
    expected_root: str,
) -> None:
    assets = materialize_asset_root(tmp_path / "assets", force=True)
    result = descriptor_from_generated_case(
        config_data=_generated_mul_mat_config(family, route_id, params, case_values, src0_dtype=src0_dtype),
        case_id="mul-mat-packed-small",
        case_values=case_values,
        kernel_dir=assets / "kernels" / "v2",
        routing_dir=assets / "catalog" / "v2",
        target="gfx1100",
        max_elements=65536,
        oracle_fixture_dir=tmp_path / "oracle-fixtures",
        descriptor_dir=tmp_path,
    )

    assert result.status == "emitted", result.reason
    assert result.descriptor is not None
    descriptor = result.descriptor
    assert descriptor["root"] == expected_root
    assert [binding["name"] for binding in descriptor["bindings"]] == ["src0", "src1", "dst"]
    assert [binding["dtype"] for binding in descriptor["bindings"]] == [src0_dtype, "f32", "f32"]
    src0 = np.load(tmp_path / descriptor["bindings"][0]["path"])
    src1 = np.load(tmp_path / descriptor["bindings"][1]["path"])
    expected = np.load(tmp_path / descriptor["bindings"][2]["expect"]["path"])
    assert src0.dtype == np.int8
    assert src1.dtype == np.float32
    assert expected.dtype == np.float32
    assert descriptor["metadata"]["oracle_array_element_counts"]["src0"] == src0.size

    descriptor_path = _write_descriptor(tmp_path, descriptor)
    prepared = prepare_execution(
        descriptor_path=descriptor_path,
        fixture_dir=tmp_path / "fixtures",
        output_path=tmp_path / "result.json",
        runner="runner",
        loom_link=None,
        iree_run_loom=None,
        repo_root=tmp_path,
    )
    command = prepared.command
    assert f"0:input:{src0_dtype}:{src0.size}:{tmp_path / descriptor['bindings'][0]['path']}" in command
    assert f"1:input:f32:{src1.size}:{tmp_path / descriptor['bindings'][1]['path']}" in command
    assert f"2:output:f32:{expected.size}:{tmp_path / descriptor['bindings'][2]['path']}" in command


def test_descriptor_from_generated_swiglu_f32_case_uses_packed_input(tmp_path: Path) -> None:
    assets = materialize_asset_root(tmp_path / "assets", force=True)
    result = descriptor_from_generated_case(
        config_data=_generated_swiglu_config(),
        case_id="d0-4-d1-2-d2-1-d3-1-src0-d0-8",
        case_values=[4, 2, 1, 1, 8],
        kernel_dir=assets / "kernels" / "v2",
        routing_dir=assets / "catalog" / "v2",
        target="gfx1100",
        max_elements=32,
        oracle_fixture_dir=tmp_path / "oracle-fixtures",
        descriptor_dir=tmp_path,
    )

    assert result.status == "emitted", result.reason
    assert result.descriptor is not None
    descriptor = result.descriptor
    assert descriptor["root"] == "@hrx2_swiglu_f32"
    assert descriptor["metadata"]["element_counts"] == {"dst": 8, "src0": 16}
    assert [binding["name"] for binding in descriptor["bindings"]] == ["src0", "dst"]
    src0 = np.load(tmp_path / descriptor["bindings"][0]["path"])
    expected = np.load(tmp_path / descriptor["bindings"][1]["expect"]["path"])
    assert src0.shape == (16,)
    assert expected.shape == (8,)


@pytest.mark.parametrize(("masked", "expected_root"), [(False, "@hrx2_soft_max_f32"), (True, "@hrx2_soft_max_f32_mask")])
def test_descriptor_from_generated_soft_max_f32_case_materializes_row_fixtures(
    tmp_path: Path,
    masked: bool,
    expected_root: str,
) -> None:
    assets = materialize_asset_root(tmp_path / "assets", force=True)
    result = descriptor_from_generated_case(
        config_data=_generated_soft_max_config(masked=masked),
        case_id="d0-4-d1-2-d2-1-d3-1",
        case_values=[4, 2, 1, 1],
        kernel_dir=assets / "kernels" / "v2",
        routing_dir=assets / "catalog" / "v2",
        target="gfx1100",
        max_elements=32,
        oracle_fixture_dir=tmp_path / "oracle-fixtures",
        descriptor_dir=tmp_path,
    )

    assert result.status == "emitted", result.reason
    assert result.descriptor is not None
    descriptor = result.descriptor
    assert descriptor["root"] == expected_root
    assert descriptor["scalars"] == [
        {
            "name": "scale",
            "position": 0,
            "dtype": "f32",
            "value": 0.75,
        }
    ]
    expected_binding_names = ["src0", "mask", "dst"] if masked else ["src0", "dst"]
    assert [binding["name"] for binding in descriptor["bindings"]] == expected_binding_names
    assert descriptor["metadata"]["element_counts"]["dst"] == 8
    assert descriptor["metadata"]["dispatch"]["workgroup_count"] == [2, 1, 1]
    expected = np.load(tmp_path / descriptor["bindings"][-1]["expect"]["path"])
    assert expected.shape == (8,)
    rows = expected.reshape(2, 4)
    np.testing.assert_allclose(rows.sum(axis=1), np.ones((2,), dtype=np.float32), rtol=1e-5, atol=1e-5)

    descriptor_path = _write_descriptor(tmp_path, descriptor)
    prepared = prepare_execution(
        descriptor_path=descriptor_path,
        fixture_dir=tmp_path / "fixtures",
        output_path=tmp_path / "result.json",
        runner="runner",
        loom_link=None,
        iree_run_loom=None,
        repo_root=tmp_path,
    )
    command = prepared.command
    assert "--scalar" in command
    assert "0:f32:0.75" in command
    if masked:
        mask = np.load(tmp_path / descriptor["bindings"][1]["path"])
        assert mask.shape == (8,)
        assert f"2:input:f32:8:{tmp_path / descriptor['bindings'][1]['path']}" in command
        assert f"3:output:f32:8:{tmp_path / descriptor['bindings'][2]['path']}" in command
    else:
        assert f"2:output:f32:8:{tmp_path / descriptor['bindings'][1]['path']}" in command


@pytest.mark.parametrize(("neox", "expected_root"), [(False, "@hrx2_rope_normal_f32"), (True, "@hrx2_rope_neox_f32")])
def test_descriptor_from_generated_rope_f32_case_uses_scalar_and_position_abi(
    tmp_path: Path,
    neox: bool,
    expected_root: str,
) -> None:
    assets = materialize_asset_root(tmp_path / "assets", force=True)
    case_values = _generated_rope_config(neox=neox)["cases"][0]
    result = descriptor_from_generated_case(
        config_data=_generated_rope_config(neox=neox),
        case_id="rope-small",
        case_values=case_values,
        kernel_dir=assets / "kernels" / "v2",
        routing_dir=assets / "catalog" / "v2",
        target="gfx1100",
        max_elements=32768,
        oracle_fixture_dir=tmp_path / "oracle-fixtures",
        descriptor_dir=tmp_path,
    )

    assert result.status == "emitted", result.reason
    assert result.descriptor is not None
    descriptor = result.descriptor
    assert descriptor["root"] == expected_root
    assert descriptor["scalars"] == [
        {"name": "theta_scale", "position": 0, "dtype": "f32", "value": 0.75},
        {"name": "freq_scale", "position": 1, "dtype": "f32", "value": 1.1},
        {"name": "attn_factor", "position": 2, "dtype": "f32", "value": 0.9},
    ]
    assert [binding["name"] for binding in descriptor["bindings"]] == ["src0", "src1", "dst"]
    assert [binding["position"] for binding in descriptor["bindings"]] == [3, 4, 5]
    assert [binding["dtype"] for binding in descriptor["bindings"]] == ["f32", "i32", "f32"]
    positions = np.load(tmp_path / descriptor["bindings"][1]["path"])
    expected = np.load(tmp_path / descriptor["bindings"][2]["expect"]["path"])
    assert positions.dtype == np.int32
    assert positions.tolist() == [1, 2]
    assert expected.dtype == np.float32

    descriptor_path = _write_descriptor(tmp_path, descriptor)
    prepared = prepare_execution(
        descriptor_path=descriptor_path,
        fixture_dir=tmp_path / "fixtures",
        output_path=tmp_path / "result.json",
        runner="runner",
        loom_link=None,
        iree_run_loom=None,
        repo_root=tmp_path,
    )
    command = prepared.command
    assert "0:f32:0.75" in command
    assert "1:f32:1.1" in command
    assert "2:f32:0.9" in command
    assert f"4:input:i32:{positions.size}:{tmp_path / descriptor['bindings'][1]['path']}" in command
    assert f"5:output:f32:{expected.size}:{tmp_path / descriptor['bindings'][2]['path']}" in command


def test_descriptor_from_generated_add_f16_case_uses_int16_storage(tmp_path: Path) -> None:
    assets = materialize_asset_root(tmp_path / "assets", force=True)
    result = descriptor_from_generated_case(
        config_data=_generated_add_f16_config(),
        case_id="d0-4-d1-1-d2-1-d3-1",
        case_values=[4, 1, 1, 1],
        kernel_dir=assets / "kernels" / "v2",
        routing_dir=assets / "catalog" / "v2",
        target="gfx1100",
        max_elements=32,
        oracle_fixture_dir=tmp_path / "oracle-fixtures",
        descriptor_dir=tmp_path,
    )

    assert result.status == "emitted", result.reason
    assert result.descriptor is not None
    descriptor = result.descriptor
    assert descriptor["root"] == "@add_f16_generic_4d"
    assert [binding["dtype"] for binding in descriptor["bindings"]] == ["f16", "f16", "f16"]
    src0 = np.load(tmp_path / descriptor["bindings"][0]["path"])
    expected = np.load(tmp_path / descriptor["bindings"][2]["expect"]["path"])
    assert src0.dtype == np.int16
    assert expected.dtype == np.int16
    assert descriptor["metadata"]["oracle_array_element_counts"] == {
        "dst_init": 4,
        "expected": 4,
        "src0": 4,
        "src1": 4,
    }

    descriptor_path = _write_descriptor(tmp_path, descriptor)
    prepared = prepare_execution(
        descriptor_path=descriptor_path,
        fixture_dir=tmp_path / "fixtures",
        output_path=tmp_path / "result.json",
        runner="runner",
        loom_link=None,
        iree_run_loom=None,
        repo_root=tmp_path,
    )
    command = prepared.command
    assert f"0:input:f16:4:{tmp_path / descriptor['bindings'][0]['path']}" in command
    assert f"2:output:f16:4:{tmp_path / descriptor['bindings'][2]['path']}" in command


def test_descriptor_from_generated_abs_f16_case_uses_int16_storage(tmp_path: Path) -> None:
    assets = materialize_asset_root(tmp_path / "assets", force=True)
    result = descriptor_from_generated_case(
        config_data=_generated_unary_f16_config("abs_f16"),
        case_id="d0-4-d1-1-d2-1-d3-1",
        case_values=[4, 1, 1, 1],
        kernel_dir=assets / "kernels" / "v2",
        routing_dir=assets / "catalog" / "v2",
        target="gfx1100",
        max_elements=32,
        oracle_fixture_dir=tmp_path / "oracle-fixtures",
        descriptor_dir=tmp_path,
    )

    assert result.status == "emitted", result.reason
    assert result.descriptor is not None
    descriptor = result.descriptor
    assert descriptor["root"] == "@abs_f16"
    assert [binding["position"] for binding in descriptor["bindings"]] == [0, 1]
    assert [binding["dtype"] for binding in descriptor["bindings"]] == ["f16", "f16"]
    src0 = np.load(tmp_path / descriptor["bindings"][0]["path"])
    expected = np.load(tmp_path / descriptor["bindings"][1]["expect"]["path"])
    assert src0.dtype == np.int16
    assert expected.dtype == np.int16

    descriptor_path = _write_descriptor(tmp_path, descriptor)
    prepared = prepare_execution(
        descriptor_path=descriptor_path,
        fixture_dir=tmp_path / "fixtures",
        output_path=tmp_path / "result.json",
        runner="runner",
        loom_link=None,
        iree_run_loom=None,
        repo_root=tmp_path,
    )
    command = prepared.command
    assert f"0:input:f16:4:{tmp_path / descriptor['bindings'][0]['path']}" in command
    assert f"1:output:f16:4:{tmp_path / descriptor['bindings'][1]['path']}" in command


def test_descriptor_from_generated_get_rows_f32_case_uses_i32_indices(tmp_path: Path) -> None:
    assets = materialize_asset_root(tmp_path / "assets", force=True)
    result = descriptor_from_generated_case(
        config_data=_generated_get_rows_f32_config(),
        case_id="d0-1-d1-2-d2-1-d3-1-src0-d1-8-src1-d0-2-src1-d1-1",
        case_values=[1, 2, 1, 1, 8, 2, 1],
        kernel_dir=assets / "kernels" / "v2",
        routing_dir=assets / "catalog" / "v2",
        target="gfx1100",
        max_elements=32,
        oracle_fixture_dir=tmp_path / "oracle-fixtures",
        descriptor_dir=tmp_path,
    )

    assert result.status == "emitted", result.reason
    assert result.descriptor is not None
    descriptor = result.descriptor
    assert descriptor["root"] == "@hrx2_get_rows_f32"
    assert [binding["name"] for binding in descriptor["bindings"]] == ["src0", "src1", "dst"]
    assert [binding["dtype"] for binding in descriptor["bindings"]] == ["f32", "i32", "f32"]
    assert descriptor["bindings"][1]["path"].endswith("/indices.npy")
    indices = np.load(tmp_path / descriptor["bindings"][1]["path"])
    expected = np.load(tmp_path / descriptor["bindings"][2]["expect"]["path"])
    assert indices.dtype == np.int32
    assert expected.dtype == np.float32

    descriptor_path = _write_descriptor(tmp_path, descriptor)
    prepared = prepare_execution(
        descriptor_path=descriptor_path,
        fixture_dir=tmp_path / "fixtures",
        output_path=tmp_path / "result.json",
        runner="runner",
        loom_link=None,
        iree_run_loom=None,
        repo_root=tmp_path,
    )
    command = prepared.command
    assert f"1:input:i32:2:{tmp_path / descriptor['bindings'][1]['path']}" in command
    assert f"2:output:f32:2:{tmp_path / descriptor['bindings'][2]['path']}" in command


def test_descriptor_from_generated_set_rows_f32_case_uses_i32_indices(tmp_path: Path) -> None:
    assets = materialize_asset_root(tmp_path / "assets", force=True)
    result = descriptor_from_generated_case(
        config_data=_generated_set_rows_f32_config(),
        case_id="d0-3-d1-3-d2-14-d3-3-src1-d1-2-src2-d0-2-src2-d1-7-src2-d2-1-src2-d3-1",
        case_values=[3, 3, 14, 3, 2, 2, 7, 1, 1],
        kernel_dir=assets / "kernels" / "v2",
        routing_dir=assets / "catalog" / "v2",
        target="gfx1100",
        max_elements=1024,
        oracle_fixture_dir=tmp_path / "oracle-fixtures",
        descriptor_dir=tmp_path,
    )

    assert result.status == "emitted", result.reason
    assert result.descriptor is not None
    descriptor = result.descriptor
    assert descriptor["root"] == "@set_rows_f32_f32"
    assert [binding["name"] for binding in descriptor["bindings"]] == ["src1", "src2", "dst"]
    assert [binding["dtype"] for binding in descriptor["bindings"]] == ["f32", "i32", "f32"]
    assert descriptor["bindings"][0]["path"].endswith("/src0.npy")
    assert descriptor["bindings"][1]["path"].endswith("/indices.npy")
    indices = np.load(tmp_path / descriptor["bindings"][1]["path"])
    expected = np.load(tmp_path / descriptor["bindings"][2]["expect"]["path"])
    assert indices.dtype == np.int32
    assert expected.dtype == np.float32

    descriptor_path = _write_descriptor(tmp_path, descriptor)
    prepared = prepare_execution(
        descriptor_path=descriptor_path,
        fixture_dir=tmp_path / "fixtures",
        output_path=tmp_path / "result.json",
        runner="runner",
        loom_link=None,
        iree_run_loom=None,
        repo_root=tmp_path,
    )
    command = prepared.command
    assert f"1:input:i32:{indices.size}:{tmp_path / descriptor['bindings'][1]['path']}" in command
    assert f"2:output:f32:{expected.size}:{tmp_path / descriptor['bindings'][2]['path']}" in command


def test_descriptor_from_generated_cont_set_rows_f32_f16_case_uses_i32_indices(
    tmp_path: Path,
) -> None:
    assets = materialize_asset_root(tmp_path / "assets", force=True)
    result = descriptor_from_generated_case(
        config_data=_generated_cont_set_rows_f32_f16_config(),
        case_id="d0-1024-d1-8192-d2-1-d3-1-src0-d1-512-src1-d0-512-src1-d1-1-src1-d2-1-src1-d3-1",
        case_values=[1024, 8192, 1, 1, 512, 512, 1, 1, 1],
        kernel_dir=assets / "kernels" / "v2",
        routing_dir=assets / "catalog" / "v2",
        target="gfx1100",
        max_elements=1073741824,
        oracle_fixture_dir=tmp_path / "oracle-fixtures",
        descriptor_dir=tmp_path,
    )

    assert result.status == "emitted", result.reason
    assert result.descriptor is not None
    descriptor = result.descriptor
    assert descriptor["root"] == "@cont_set_rows_f32_f16"
    assert [binding["name"] for binding in descriptor["bindings"]] == ["src0", "src1", "dst"]
    assert [binding["dtype"] for binding in descriptor["bindings"]] == ["f32", "i32", "f16"]
    indices = np.load(tmp_path / descriptor["bindings"][1]["path"])
    expected = np.load(tmp_path / descriptor["bindings"][2]["expect"]["path"])
    assert indices.dtype == np.int32
    assert expected.dtype == np.int16

    descriptor_path = _write_descriptor(tmp_path, descriptor)
    prepared = prepare_execution(
        descriptor_path=descriptor_path,
        fixture_dir=tmp_path / "fixtures",
        output_path=tmp_path / "result.json",
        runner="runner",
        loom_link=None,
        iree_run_loom=None,
        repo_root=tmp_path,
    )
    command = prepared.command
    assert f"1:input:i32:{indices.size}:{tmp_path / descriptor['bindings'][1]['path']}" in command
    assert f"2:output:f16:{expected.size}:{tmp_path / descriptor['bindings'][2]['path']}" in command


def test_descriptor_from_generated_add_case_requires_execution_abi(tmp_path: Path) -> None:
    config = _generated_add_config()
    config.pop("execution_abi")

    result = descriptor_from_generated_case(
        config_data=config,
        case_id="d0-4-d1-1-d2-1-d3-1",
        case_values=[4, 1, 1, 1],
        kernel_dir=Path("kernels/v2"),
        routing_dir=Path("catalog/v2"),
        target="gfx1100",
        max_elements=32,
        oracle_fixture_dir=tmp_path / "oracle-fixtures",
        descriptor_dir=tmp_path,
    )

    assert result.status == "unsupported"
    assert result.reason == "generated config is missing execution_abi"


def test_descriptor_from_generated_add_case_skips_large_fixtures(tmp_path: Path) -> None:
    result = descriptor_from_generated_case(
        config_data=_generated_add_config(),
        case_id="too-large",
        case_values=[128, 1, 1, 1],
        kernel_dir=Path("kernels/v2"),
        routing_dir=Path("catalog/v2"),
        target="gfx1100",
        max_elements=32,
        oracle_fixture_dir=tmp_path / "oracle-fixtures",
        descriptor_dir=tmp_path,
    )

    assert result.status == "skipped"
    assert "above max 32" in str(result.reason)


def test_write_generated_execution_descriptors_writes_manifest_and_descriptor(tmp_path: Path) -> None:
    config_path = tmp_path / "add-config.json"
    config_path.write_text(json.dumps(_generated_add_config()) + "\n", encoding="utf-8")
    manifest_path = tmp_path / "generated-kernel-tests.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "ggml_hrx_kernel_bench.generated_kernel_tests.v1",
                "entry_count": 1,
                "entries": [
                    {
                        "config_path": str(config_path),
                        "config_name": config_path.name,
                        "kernel": "add_f32",
                        "case_count": 1,
                        "route_id": "add_f32_generic_4d",
                        "op": "ADD",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    descriptor_manifest = write_generated_execution_descriptors(
        manifest_path=manifest_path,
        output_dir=tmp_path / "descriptors",
        kernel_dir=Path("kernels/v2"),
        routing_dir=Path("catalog/v2"),
        target="gfx1100",
        max_elements=32,
    )

    assert descriptor_manifest["schema"] == DESCRIPTOR_MANIFEST_SCHEMA
    assert descriptor_manifest["emitted_count"] == 1
    descriptor_path = Path(descriptor_manifest["entries"][0]["descriptor_path"])
    assert descriptor_path.is_file()
    emitted = json.loads(descriptor_path.read_text(encoding="utf-8"))
    assert emitted["schema"] == SCHEMA
    prepared = prepare_execution(
        descriptor_path=descriptor_path,
        fixture_dir=tmp_path / "fixtures",
        output_path=tmp_path / "result.json",
        runner="runner",
        loom_link=None,
        iree_run_loom=None,
        repo_root=Path.cwd(),
    )
    assert "--workgroup-count" in prepared.command
    assert "1,1,1" in prepared.command


def test_write_generated_execution_descriptors_filters_manifest_entries(tmp_path: Path) -> None:
    add_config_path = tmp_path / "add-config.json"
    add_config_path.write_text(json.dumps(_generated_add_config()) + "\n", encoding="utf-8")
    f16_config_path = tmp_path / "add-f16-config.json"
    f16_config_path.write_text(
        json.dumps(
            {
                "kernel": "add_f16",
                "params": ["d0", "d1", "d2", "d3"],
                "cases": [[4, 1, 1, 1]],
                "route_id": "add_f16_generic_4d",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "generated-kernel-tests.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "ggml_hrx_kernel_bench.generated_kernel_tests.v1",
                "entry_count": 2,
                "entries": [
                    {
                        "config_path": str(f16_config_path),
                        "config_name": f16_config_path.name,
                        "kernel": "add_f16",
                        "case_count": 1,
                        "route_id": "add_f16_generic_4d",
                    },
                    {
                        "config_path": str(add_config_path),
                        "config_name": add_config_path.name,
                        "kernel": "add_f32",
                        "case_count": 1,
                        "route_id": "add_f32_generic_4d",
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    descriptor_manifest = write_generated_execution_descriptors(
        manifest_path=manifest_path,
        output_dir=tmp_path / "descriptors",
        kernel_dir=Path("kernels/v2"),
        routing_dir=Path("catalog/v2"),
        target="gfx1100",
        max_elements=32,
        kernels={"add_f32"},
    )

    assert descriptor_manifest["filtered_count"] == 1
    assert descriptor_manifest["unsupported_count"] == 0
    assert descriptor_manifest["emitted_count"] == 1


def _write_descriptor_manifest(tmp_path: Path) -> tuple[Path, Path]:
    descriptor_path = _write_descriptor(tmp_path, _descriptor())
    manifest_path = tmp_path / "loom-execution-descriptors.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": DESCRIPTOR_MANIFEST_SCHEMA,
                "source_manifest_path": "generated-kernel-tests.json",
                "target": "gfx1100",
                "max_elements": 32,
                "entry_count": 1,
                "emitted_count": 1,
                "skipped_count": 0,
                "unsupported_count": 0,
                "filtered_count": 0,
                "entries": [
                    {
                        "status": "emitted",
                        "descriptor_path": str(descriptor_path),
                        "config_path": "add-config.json",
                        "config_name": "add-config.json",
                        "kernel": "add_f32",
                        "route_id": "add_f32_contiguous_1d",
                        "case_id": "case0",
                        "case_values": [4],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path, descriptor_path


def test_run_execution_descriptor_manifest_prepares_commands(tmp_path: Path) -> None:
    manifest_path, descriptor_path = _write_descriptor_manifest(tmp_path)

    run_manifest = run_execution_descriptor_manifest(
        manifest_path=manifest_path,
        output_dir=tmp_path / "runs",
        runner="runner",
        loom_link="loom-link",
        iree_run_loom="iree-run-loom",
        repo_root=Path.cwd(),
    )

    assert run_manifest["prepared_count"] == 1
    assert run_manifest["executed_count"] == 0
    assert run_manifest["failed_count"] == 0
    assert run_manifest["entries"][0]["descriptor_path"] == str(descriptor_path.resolve())
    assert "--execute-iree-run-loom-command" not in run_manifest["entries"][0]["command"]
    assert (tmp_path / "runs" / "loom-execution-runs.json").is_file()


def test_run_execution_descriptor_manifest_executes_fake_runner(tmp_path: Path) -> None:
    manifest_path, _ = _write_descriptor_manifest(tmp_path)
    runner = tmp_path / "fake-runner.py"
    runner.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import json",
                "import sys",
                "output = sys.argv[sys.argv.index('--output') + 1]",
                "open(output, 'w').write(json.dumps({'status': 'run_passed'}) + '\\n')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    os.chmod(runner, 0o755)
    iree_run_loom = tmp_path / "iree-run-loom"

    run_manifest = run_execution_descriptor_manifest(
        manifest_path=manifest_path,
        output_dir=tmp_path / "runs",
        runner=runner,
        loom_link=None,
        iree_run_loom=iree_run_loom,
        repo_root=Path.cwd(),
        execute=True,
    )

    assert run_manifest["executed_count"] == 1
    assert run_manifest["passed_count"] == 1
    assert run_manifest["failed_count"] == 0
    assert run_manifest["entries"][0]["status"] == "run_passed"


def test_run_loom_execution_descriptors_script_prepares(tmp_path: Path) -> None:
    manifest_path, _ = _write_descriptor_manifest(tmp_path)
    script = Path("tests/infra/run_loom_execution_descriptors.py")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            str(manifest_path),
            "--output-dir",
            str(tmp_path / "runs"),
            "--runner",
            "runner",
            "--repo-root",
            str(Path.cwd()),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["prepared_count"] == 1
    assert payload["executed_count"] == 0


def test_generate_loom_execution_descriptors_script(tmp_path: Path) -> None:
    config_path = tmp_path / "add-config.json"
    config_path.write_text(json.dumps(_generated_add_config()) + "\n", encoding="utf-8")
    manifest_path = tmp_path / "generated-kernel-tests.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "ggml_hrx_kernel_bench.generated_kernel_tests.v1",
                "entry_count": 1,
                "entries": [
                    {
                        "config_path": str(config_path),
                        "config_name": config_path.name,
                        "kernel": "add_f32",
                        "case_count": 1,
                        "route_id": "add_f32_generic_4d",
                        "op": "ADD",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    script = Path("tests/infra/generate_loom_execution_descriptors.py")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            str(manifest_path),
            "--output-dir",
            str(tmp_path / "descriptors"),
            "--kernel-dir",
            "kernels/v2",
            "--routing-dir",
            "catalog/v2",
            "--max-elements",
            "32",
            "--limit",
            "1",
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["emitted_count"] == 1
    assert Path(payload["entries"][0]["descriptor_path"]).is_file()
