from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ggml_hrx_kernel_bench.materialized_assets import materialize_asset_root
from ggml_hrx_kernel_bench.oracles import generate_oracle, write_workbench
from ggml_hrx_kernel_bench.routing.api import Candidate


def _candidate(
    *,
    candidate_id: str,
    shape: dict[str, int],
    family: str = "add_f32",
    source_id: str = "add_f32",
    root_symbol: str = "@hrx2_add_f32",
    export_name: str = "hrx2_add_f32",
    op: str = "ADD",
    source_path: str = "kernels/v2/add/contiguous_1d.loom",
) -> Candidate:
    return Candidate(
        id=candidate_id,
        family=family,
        op=op,
        source_id=source_id,
        source_path=Path(source_path),
        root_symbol=root_symbol,
        export_name=export_name,
        route_id=f"{family}_test",
        route=None,
        shape=shape,
        values={},
        config={},
        dispatch={},
        supports={},
        coverage="route_backed",
    )


def _materialized_copy_source_path(tmp_path: Path, route_id: str) -> str:
    asset_root = materialize_asset_root(tmp_path / "assets", force=True)
    return str(asset_root / "kernels" / "v2" / "copy" / f"{route_id}.loom")


def _materialized_copy_non_contiguous_source_path(tmp_path: Path, route_id: str) -> str:
    asset_root = materialize_asset_root(tmp_path / "assets", force=True)
    return str(asset_root / "kernels" / "v2" / "copy" / f"{route_id}.loom")


def test_add_oracle_uses_ranked_shape_for_fixture_size(tmp_path: Path) -> None:
    candidate = _candidate(candidate_id="add_ranked_4d", shape={"d0": 10, "d1": 5, "d2": 4, "d3": 3})

    result = generate_oracle(candidate, tmp_path / "fixtures", force=True)

    assert result.status == "fixtures_ready"
    assert np.load(tmp_path / "fixtures" / "src0.npy").shape == (600,)
    assert np.load(tmp_path / "fixtures" / "src1.npy").shape == (600,)
    assert np.load(tmp_path / "fixtures" / "dst_init.npy").shape == (600,)
    assert np.load(tmp_path / "fixtures" / "expected.npy").shape == (600,)


def test_add_workbench_uses_ranked_shape_for_tensor_length(tmp_path: Path) -> None:
    candidate = _candidate(candidate_id="add_ranked_2d", shape={"d0": 16, "d1": 64})
    linked_source = tmp_path / "linked.loom"
    linked_source.write_text("kernel.def export(\"hrx2_add_f32\") @hrx2_add_f32() {}\n", encoding="utf-8")

    _, metadata = write_workbench(candidate, linked_source, tmp_path / "workbench.loom", tmp_path / "fixtures")

    assert metadata["status"] == "ok"
    workbench = (tmp_path / "workbench.loom").read_text(encoding="utf-8")
    assert "tensor<1024xf32>" in workbench


def test_add_oracle_and_workbench_support_ranked_src1_overrides(tmp_path: Path) -> None:
    candidate = _candidate(
        candidate_id="add_ranked_4d_broadcast",
        shape={"d0": 10, "d1": 5, "d2": 4, "d3": 6, "src1_d3": 3},
    )

    result = generate_oracle(candidate, tmp_path / "fixtures", force=True)

    assert result.status == "fixtures_ready"
    assert np.load(tmp_path / "fixtures" / "src0.npy").shape == (1200,)
    assert np.load(tmp_path / "fixtures" / "src1.npy").shape == (600,)
    assert np.load(tmp_path / "fixtures" / "dst_init.npy").shape == (1200,)
    assert np.load(tmp_path / "fixtures" / "expected.npy").shape == (1200,)

    linked_source = tmp_path / "linked.loom"
    linked_source.write_text("kernel.def export(\"hrx2_add_f32\") @hrx2_add_f32() {}\n", encoding="utf-8")
    _, metadata = write_workbench(candidate, linked_source, tmp_path / "workbench.loom", tmp_path / "fixtures")

    assert metadata["status"] == "ok"
    workbench = (tmp_path / "workbench.loom").read_text(encoding="utf-8")
    assert "tensor<1200xf32>, tensor<600xf32>, tensor<1200xf32>" in workbench


@pytest.mark.parametrize(
    ("family", "op", "src_dtype", "dst_dtype"),
    (
        ("copy_bf16_bf16", "CPY", np.int16, np.int16),
        ("copy_bf16_f16", "CPY", np.int16, np.int16),
        ("copy_bf16_f32", "CPY", np.int16, np.float32),
        ("copy_f16_bf16", "CPY", np.int16, np.int16),
        ("copy_f16_f16", "CPY", np.int16, np.int16),
        ("copy_f16_f32", "CPY", np.int16, np.float32),
        ("copy_f32_bf16", "CPY", np.float32, np.int16),
        ("copy_f32_f16", "CPY", np.float32, np.int16),
        ("copy_f32_f32", "CPY", np.float32, np.float32),
    ),
)
def test_copy_oracle_and_workbench_use_expected_buffer_types(
    tmp_path: Path,
    family: str,
    op: str,
    src_dtype: type[np.generic],
    dst_dtype: type[np.generic],
) -> None:
    route_id = f"{family}_contiguous_1d"
    candidate = _candidate(
        candidate_id=f"{family}_ranked_4d",
        shape={"d0": 16, "d1": 4, "d2": 2, "d3": 2},
        family=family,
        source_id=family,
        root_symbol=f"@{route_id}",
        export_name=route_id,
        op=op,
        source_path=_materialized_copy_source_path(tmp_path, route_id),
    )

    result = generate_oracle(candidate, tmp_path / "fixtures", force=True)

    assert result.status == "fixtures_ready"
    assert np.load(tmp_path / "fixtures" / "src0.npy").dtype == src_dtype
    assert np.load(tmp_path / "fixtures" / "dst_init.npy").dtype == dst_dtype
    assert np.load(tmp_path / "fixtures" / "expected.npy").dtype == dst_dtype

    linked_source = tmp_path / "linked.loom"
    linked_source.write_text(f"kernel.def export(\"{route_id}\") @{route_id}() {{}}\n", encoding="utf-8")
    _, metadata = write_workbench(candidate, linked_source, tmp_path / "workbench.loom", tmp_path / "fixtures")

    assert metadata["status"] == "ok"
    workbench = (tmp_path / "workbench.loom").read_text(encoding="utf-8")
    src_tensor = "tensor<256xi16>" if src_dtype is np.int16 else "tensor<256xf32>"
    dst_tensor = "tensor<256xi16>" if dst_dtype is np.int16 else "tensor<256xf32>"
    assert f"{src_tensor}, {dst_tensor}" in workbench
    if dst_dtype is np.int16:
        assert "check.expect.equal" in workbench
    else:
        assert "check.expect.close" in workbench


def test_copy_oracle_and_workbench_support_transposed_f32_source(tmp_path: Path) -> None:
    candidate = _candidate(
        candidate_id="copy_f32_f32_non_contiguous_4d",
        shape={
            "d0": 4,
            "d1": 256,
            "d2": 3,
            "d3": 3,
            "src0_d0_stride": 256,
            "src0_d1_stride": 1,
        },
        family="copy_f32_f32",
        source_id="copy_f32_f32",
        root_symbol="@copy_f32_f32_non_contiguous_4d",
        export_name="copy_f32_f32_non_contiguous_4d",
        op="CPY",
        source_path=_materialized_copy_non_contiguous_source_path(tmp_path, "copy_f32_f32_non_contiguous_4d"),
    )

    result = generate_oracle(candidate, tmp_path / "fixtures", force=True)

    assert result.status == "fixtures_ready"
    src0 = np.load(tmp_path / "fixtures" / "src0.npy")
    expected = np.load(tmp_path / "fixtures" / "expected.npy")
    src0_view = src0.reshape(3, 3, 4, 256).transpose(0, 1, 3, 2).reshape(-1)
    assert np.array_equal(expected, src0_view)
    assert not np.array_equal(expected, src0)

    linked_source = tmp_path / "linked.loom"
    linked_source.write_text(
        'kernel.def export("copy_f32_f32_non_contiguous_4d") @copy_f32_f32_non_contiguous_4d() {}\n',
        encoding="utf-8",
    )
    _, metadata = write_workbench(candidate, linked_source, tmp_path / "workbench.loom", tmp_path / "fixtures")

    assert metadata["status"] == "ok"
    workbench = (tmp_path / "workbench.loom").read_text(encoding="utf-8")
    assert "tensor<9216xf32>, tensor<9216xf32>" in workbench
    assert "check.expect.close" in workbench


def test_set_rows_oracle_and_workbench_support_ranked_v2_shape(tmp_path: Path) -> None:
    candidate = _candidate(
        candidate_id="set_rows_f32_ranked_4d",
        shape={
            "d0": 4,
            "d1": 8,
            "d2": 1,
            "d3": 3,
            "src0_d1": 2,
            "src1_d0": 2,
            "src1_d1": 1,
            "src1_d2": 1,
            "src1_d3": 1,
        },
        family="set_rows_f32",
        source_id="set_rows_f32",
        root_symbol="@set_rows_f32_f32",
        export_name="set_rows_f32_f32",
        op="SET_ROWS",
        source_path="kernels/hrx2/set_rows_f32.loom",
    )

    result = generate_oracle(candidate, tmp_path / "fixtures", force=True)

    assert result.status == "fixtures_ready"
    assert np.load(tmp_path / "fixtures" / "src0.npy").shape == (24,)
    assert np.load(tmp_path / "fixtures" / "indices.npy").shape == (4,)
    assert np.load(tmp_path / "fixtures" / "dst_init.npy").shape == (96,)
    assert np.load(tmp_path / "fixtures" / "expected.npy").shape == (96,)

    linked_source = tmp_path / "linked.loom"
    linked_source.write_text('kernel.def export("set_rows_f32_f32") @set_rows_f32_f32() {}\n', encoding="utf-8")
    _, metadata = write_workbench(candidate, linked_source, tmp_path / "workbench.loom", tmp_path / "fixtures")

    assert metadata["status"] == "ok"
    workbench = (tmp_path / "workbench.loom").read_text(encoding="utf-8")
    assert "tensor<24xf32>, tensor<4xi32>, tensor<96xf32>" in workbench
    assert "check.expect.close" in workbench


@pytest.mark.parametrize(
    ("family", "op", "source_path", "root_symbol", "export_name"),
    (
        ("add_f16", "ADD", "kernels/v2/add/contiguous_1d.loom", "@hrx2_add_f16", "hrx2_add_f16"),
        ("mul_f16", "MUL", "kernels/v2/mul/contiguous_1d.loom", "@hrx2_mul_f16", "hrx2_mul_f16"),
        ("div_f16", "DIV", "kernels/v2/div/contiguous_1d.loom", "@hrx2_div_f16", "hrx2_div_f16"),
        ("sub_f16", "SUB", "kernels/v2/sub/contiguous_1d.loom", "@hrx2_sub_f16", "hrx2_sub_f16"),
    ),
)
def test_pointwise_f16_oracle_and_workbench_use_i16_buffers(
    tmp_path: Path,
    family: str,
    op: str,
    source_path: str,
    root_symbol: str,
    export_name: str,
) -> None:
    candidate = _candidate(
        candidate_id=f"{family}_ranked_2d",
        shape={"d0": 16, "d1": 64},
        family=family,
        source_id=family,
        root_symbol=root_symbol,
        export_name=export_name,
        op=op,
        source_path=source_path,
    )

    result = generate_oracle(candidate, tmp_path / "fixtures", force=True)

    assert result.status == "fixtures_ready"
    assert np.load(tmp_path / "fixtures" / "src0.npy").dtype == np.int16
    assert np.load(tmp_path / "fixtures" / "src1.npy").dtype == np.int16
    assert np.load(tmp_path / "fixtures" / "dst_init.npy").dtype == np.int16
    assert np.load(tmp_path / "fixtures" / "expected.npy").dtype == np.int16

    linked_source = tmp_path / "linked.loom"
    linked_source.write_text(f"kernel.def export(\"{export_name}\") {root_symbol}() {{}}\n", encoding="utf-8")
    _, metadata = write_workbench(candidate, linked_source, tmp_path / "workbench.loom", tmp_path / "fixtures")

    assert metadata["status"] == "ok"
    workbench = (tmp_path / "workbench.loom").read_text(encoding="utf-8")
    assert "tensor<1024xi16>, tensor<1024xi16>, tensor<1024xi16>" in workbench
    assert "check.expect.equal" in workbench
