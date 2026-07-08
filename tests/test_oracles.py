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


def test_sum_rows_oracle_and_workbench_use_reduced_dst_shape(tmp_path: Path) -> None:
    candidate = _candidate(
        candidate_id="sum_rows_f32_ranked_4d",
        shape={"d0": 33, "d1": 8, "d2": 2, "d3": 1},
        family="sum_rows_f32",
        source_id="sum_rows_f32",
        root_symbol="@hrx2_sum_rows_f32",
        export_name="hrx2_sum_rows_f32",
        op="SUM_ROWS",
        source_path="kernels/v2/sum_rows/contiguous_4d.loom",
    )

    result = generate_oracle(candidate, tmp_path / "fixtures", force=True)

    assert result.status == "fixtures_ready"
    assert np.load(tmp_path / "fixtures" / "src0.npy").shape == (528,)
    assert np.load(tmp_path / "fixtures" / "expected.npy").shape == (16,)

    linked_source = tmp_path / "linked.loom"
    linked_source.write_text('kernel.def export("hrx2_sum_rows_f32") @hrx2_sum_rows_f32() {}\n', encoding="utf-8")
    _, metadata = write_workbench(candidate, linked_source, tmp_path / "workbench.loom", tmp_path / "fixtures")

    assert metadata["status"] == "ok"
    workbench = (tmp_path / "workbench.loom").read_text(encoding="utf-8")
    assert "tensor<528xf32>, tensor<16xf32>" in workbench
    assert "check.expect.close" in workbench


def test_rms_norm_oracle_and_workbench_use_ranked_shape(tmp_path: Path) -> None:
    candidate = _candidate(
        candidate_id="rms_norm_f32_ranked_4d",
        shape={"d0": 64, "d1": 5, "d2": 4, "d3": 3},
        family="rms_norm_f32",
        source_id="rms_norm_f32",
        root_symbol="@hrx2_rms_norm_f32",
        export_name="hrx2_rms_norm_f32",
        op="RMS_NORM",
        source_path="kernels/v2/rms_norm/contiguous_4d.loom",
    )

    result = generate_oracle(candidate, tmp_path / "fixtures", force=True)

    assert result.status == "fixtures_ready"
    assert np.load(tmp_path / "fixtures" / "src.npy").shape == (3840,)
    assert np.load(tmp_path / "fixtures" / "expected.npy").shape == (3840,)

    linked_source = tmp_path / "linked.loom"
    linked_source.write_text('kernel.def export("hrx2_rms_norm_f32") @hrx2_rms_norm_f32() {}\n', encoding="utf-8")
    _, metadata = write_workbench(candidate, linked_source, tmp_path / "workbench.loom", tmp_path / "fixtures")

    assert metadata["status"] == "ok"
    workbench = (tmp_path / "workbench.loom").read_text(encoding="utf-8")
    assert "%eps = check.literal value(0.0) : f32" in workbench
    assert "tensor<3840xf32>, tensor<3840xf32>" in workbench
    assert "check.expect.close" in workbench


def test_swiglu_oracle_and_workbench_use_packed_src_shape(tmp_path: Path) -> None:
    candidate = _candidate(
        candidate_id="swiglu_f32_ranked_4d",
        shape={"d0": 64, "d1": 5, "d2": 4, "d3": 3, "src0_d0": 128},
        family="swiglu_f32",
        source_id="swiglu_f32",
        root_symbol="@hrx2_swiglu_f32",
        export_name="hrx2_swiglu_f32",
        op="SWIGLU",
        source_path="kernels/v2/swiglu/contiguous_4d.loom",
    )

    result = generate_oracle(candidate, tmp_path / "fixtures", force=True)

    assert result.status == "fixtures_ready"
    assert np.load(tmp_path / "fixtures" / "src0.npy").shape == (7680,)
    assert np.load(tmp_path / "fixtures" / "expected.npy").shape == (3840,)

    linked_source = tmp_path / "linked.loom"
    linked_source.write_text('kernel.def export("hrx2_swiglu_f32") @hrx2_swiglu_f32() {}\n', encoding="utf-8")
    _, metadata = write_workbench(candidate, linked_source, tmp_path / "workbench.loom", tmp_path / "fixtures")

    assert metadata["status"] == "ok"
    workbench = (tmp_path / "workbench.loom").read_text(encoding="utf-8")
    assert "tensor<7680xf32>, tensor<3840xf32>" in workbench
    assert "check.expect.close" in workbench


def test_get_rows_oracle_and_workbench_use_src0_row_override(tmp_path: Path) -> None:
    candidate = Candidate(
        id="get_rows_f32_ranked_2d",
        family="get_rows_f32",
        op="GET_ROWS",
        source_id="get_rows_f32",
        source_path=Path("kernels/v2/get_rows/embedding_rows_2d.loom"),
        root_symbol="@hrx2_get_rows_f32",
        export_name="hrx2_get_rows_f32",
        route_id="get_rows_f32_test",
        route=None,
        shape={"d0": 256, "d1": 4, "src0_d1": 5, "src1_d0": 1},
        values={
            "shape.get_rows.src0_nrows": 5,
            "shape.get_rows.idx_row_stride": 1,
        },
        config={},
        dispatch={},
        supports={},
        coverage="route_backed",
    )

    result = generate_oracle(candidate, tmp_path / "fixtures", force=True)

    assert result.status == "fixtures_ready"
    assert np.load(tmp_path / "fixtures" / "src0.npy").shape == (1280,)
    assert np.load(tmp_path / "fixtures" / "indices.npy").shape == (4,)
    assert np.load(tmp_path / "fixtures" / "expected.npy").shape == (1024,)

    linked_source = tmp_path / "linked.loom"
    linked_source.write_text('kernel.def export("hrx2_get_rows_f32") @hrx2_get_rows_f32() {}\n', encoding="utf-8")
    _, metadata = write_workbench(candidate, linked_source, tmp_path / "workbench.loom", tmp_path / "fixtures")

    assert metadata["status"] == "ok"
    workbench = (tmp_path / "workbench.loom").read_text(encoding="utf-8")
    assert "tensor<1280xf32>, tensor<4xi32>, tensor<1024xf32>" in workbench


def test_soft_max_oracle_and_workbench_use_flattened_rows(tmp_path: Path) -> None:
    candidate = _candidate(
        candidate_id="soft_max_f32_ranked_2d",
        shape={"d0": 16, "d1": 16},
        family="soft_max_f32",
        source_id="soft_max_f32",
        root_symbol="@hrx2_soft_max_f32",
        export_name="hrx2_soft_max_f32",
        op="SOFT_MAX",
        source_path="kernels/v2/soft_max/contiguous_f32.loom",
    )

    result = generate_oracle(candidate, tmp_path / "fixtures", force=True)

    assert result.status == "fixtures_ready"
    assert np.load(tmp_path / "fixtures" / "src0.npy").shape == (256,)
    assert np.load(tmp_path / "fixtures" / "expected.npy").shape == (256,)

    linked_source = tmp_path / "linked.loom"
    linked_source.write_text('kernel.def export("hrx2_soft_max_f32") @hrx2_soft_max_f32() {}\n', encoding="utf-8")
    _, metadata = write_workbench(candidate, linked_source, tmp_path / "workbench.loom", tmp_path / "fixtures")

    assert metadata["status"] == "ok"
    workbench = (tmp_path / "workbench.loom").read_text(encoding="utf-8")
    assert "tensor<256xf32>, tensor<256xf32>" in workbench
    assert "check.expect.close" in workbench


@pytest.mark.parametrize(
    ("family", "op", "root_symbol", "export_name", "source_path"),
    (
        ("exp_f16", "EXP", "@exp_f16", "exp_f16", "kernels/v2/exp/contiguous_4d.loom"),
        ("neg_f16", "NEG", "@neg_f16", "neg_f16", "kernels/v2/neg/contiguous_4d.loom"),
        ("relu_f16", "RELU", "@relu_f16", "relu_f16", "kernels/v2/relu/contiguous_4d.loom"),
        ("sqr_f16", "SQR", "@sqr_f16", "sqr_f16", "kernels/v2/sqr/contiguous_4d.loom"),
        ("sqrt_f16", "SQRT", "@sqrt_f16", "sqrt_f16", "kernels/v2/sqrt/contiguous_4d.loom"),
    ),
)
def test_unary_f16_oracle_and_workbench_use_i16_buffers(
    tmp_path: Path,
    family: str,
    op: str,
    root_symbol: str,
    export_name: str,
    source_path: str,
) -> None:
    candidate = _candidate(
        candidate_id=f"{family}_ranked_3d",
        shape={"d0": 4, "d1": 8, "d2": 2},
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
    assert np.load(tmp_path / "fixtures" / "dst_init.npy").dtype == np.int16
    assert np.load(tmp_path / "fixtures" / "expected.npy").dtype == np.int16

    linked_source = tmp_path / "linked.loom"
    linked_source.write_text(f"kernel.def export(\"{export_name}\") {root_symbol}() {{}}\n", encoding="utf-8")
    _, metadata = write_workbench(candidate, linked_source, tmp_path / "workbench.loom", tmp_path / "fixtures")

    assert metadata["status"] == "ok"
    workbench = (tmp_path / "workbench.loom").read_text(encoding="utf-8")
    assert "tensor<64xi16>, tensor<64xi16>" in workbench
    assert "check.expect.equal" in workbench


@pytest.mark.parametrize(
    ("family", "op", "root_symbol", "export_name", "source_path"),
    (
        ("exp_f32", "EXP", "@exp_f32", "exp_f32", "kernels/v2/exp/contiguous_4d.loom"),
        ("neg_f32", "NEG", "@neg_f32", "neg_f32", "kernels/v2/neg/contiguous_4d.loom"),
        ("relu_f32", "RELU", "@relu_f32", "relu_f32", "kernels/v2/relu/contiguous_4d.loom"),
        ("sqr_f32", "SQR", "@sqr_f32", "sqr_f32", "kernels/v2/sqr/contiguous_4d.loom"),
        ("sqrt_f32", "SQRT", "@sqrt_f32", "sqrt_f32", "kernels/v2/sqrt/contiguous_4d.loom"),
    ),
)
def test_unary_f32_oracle_and_workbench_use_f32_buffers(
    tmp_path: Path,
    family: str,
    op: str,
    root_symbol: str,
    export_name: str,
    source_path: str,
) -> None:
    candidate = _candidate(
        candidate_id=f"{family}_ranked_3d",
        shape={"d0": 4, "d1": 8, "d2": 2},
        family=family,
        source_id=family,
        root_symbol=root_symbol,
        export_name=export_name,
        op=op,
        source_path=source_path,
    )

    result = generate_oracle(candidate, tmp_path / "fixtures", force=True)

    assert result.status == "fixtures_ready"
    assert np.load(tmp_path / "fixtures" / "src0.npy").dtype == np.float32
    assert np.load(tmp_path / "fixtures" / "dst_init.npy").dtype == np.float32
    assert np.load(tmp_path / "fixtures" / "expected.npy").dtype == np.float32

    linked_source = tmp_path / "linked.loom"
    linked_source.write_text(f"kernel.def export(\"{export_name}\") {root_symbol}() {{}}\n", encoding="utf-8")
    _, metadata = write_workbench(candidate, linked_source, tmp_path / "workbench.loom", tmp_path / "fixtures")

    assert metadata["status"] == "ok"
    workbench = (tmp_path / "workbench.loom").read_text(encoding="utf-8")
    assert "tensor<64xf32>, tensor<64xf32>" in workbench
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
