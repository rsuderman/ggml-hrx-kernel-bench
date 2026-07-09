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


def _softmax_kqv_shape(k: int, *, rows: int = 128, cols: int = 24, nheads_kv: int = 8) -> dict[str, int]:
    return {
        "d0": cols,
        "d1": rows,
        "src0_d0": k,
        "src0_d1": cols,
        "mask_d0": k,
        "mask_d1": 1,
        "src1_d0": k,
        "src1_d1": rows * nheads_kv,
        "dst_d0": rows,
        "dst_d1": cols,
        "k": k,
        "rows": rows,
        "cols": cols,
        "nheads_kv": nheads_kv,
    }


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


# ggml unary v=1 view of ne_a=[2,3,2,2]: parent inflated per dim by [3,2,5,4] -> src0 element
# strides [1, 3*ne0, 6*ne0*ne1, 30*ne0*ne1*ne2] = [1, 6, 36, 360]; padded buffer len 410, dst 24.
_ABS_STRIDED_SHAPE = {"d0": 2, "d1": 3, "d2": 2, "d3": 2, "src0_d1_stride": 6, "src0_d2_stride": 36, "src0_d3_stride": 360}
_ABS_STRIDED_DIMS = (2, 3, 2, 2)
_ABS_STRIDED_STRIDES = (1, 6, 36, 360)
_ABS_STRIDED_BUFFER_LEN = 410
_ABS_STRIDED_DST_ELEMS = 24


def _abs_gathered_view(src0: np.ndarray) -> np.ndarray:
    out = np.empty(_ABS_STRIDED_DST_ELEMS, dtype=src0.dtype)
    d0, d1, d2, d3 = _ABS_STRIDED_DIMS
    s0, s1, s2, s3 = _ABS_STRIDED_STRIDES
    for i3 in range(d3):
        for i2 in range(d2):
            for i1 in range(d1):
                for i0 in range(d0):
                    off = i0 * s0 + i1 * s1 + i2 * s2 + i3 * s3
                    lin = i0 + i1 * d0 + i2 * d0 * d1 + i3 * d0 * d1 * d2
                    out[lin] = src0[off]
    return out


def test_abs_oracle_and_workbench_model_strided_src0_view_f32(tmp_path: Path) -> None:
    candidate = _candidate(
        candidate_id="abs_f32_non_contiguous_4d",
        shape=dict(_ABS_STRIDED_SHAPE),
        family="abs_f32",
        source_id="abs_f32",
        root_symbol="@hrx2_abs_f32_non_contiguous_4d",
        export_name="hrx2_abs_f32_non_contiguous_4d",
        op="ABS",
        source_path="kernels/v2/abs/non_contiguous_4d.loom",
    )

    result = generate_oracle(candidate, tmp_path / "fixtures", force=True)

    assert result.status == "fixtures_ready"
    src0 = np.load(tmp_path / "fixtures" / "src0.npy")
    expected = np.load(tmp_path / "fixtures" / "expected.npy")
    assert src0.shape == (_ABS_STRIDED_BUFFER_LEN,)
    assert expected.shape == (_ABS_STRIDED_DST_ELEMS,)
    assert np.array_equal(expected, np.abs(_abs_gathered_view(src0)))
    # A naive contiguous read of the first dst_elems values is NOT the view -> strides matter.
    assert not np.array_equal(expected, np.abs(src0[:_ABS_STRIDED_DST_ELEMS]))

    linked_source = tmp_path / "linked.loom"
    linked_source.write_text(
        'kernel.def export("hrx2_abs_f32_non_contiguous_4d") @hrx2_abs_f32_non_contiguous_4d() {}\n',
        encoding="utf-8",
    )
    _, metadata = write_workbench(candidate, linked_source, tmp_path / "workbench.loom", tmp_path / "fixtures")

    assert metadata["status"] == "ok"
    workbench = (tmp_path / "workbench.loom").read_text(encoding="utf-8")
    assert "tensor<410xf32>, tensor<24xf32>" in workbench
    assert "check.expect.close" in workbench


def test_abs_oracle_and_workbench_model_strided_src0_view_f16(tmp_path: Path) -> None:
    candidate = _candidate(
        candidate_id="abs_f16_non_contiguous_4d",
        shape=dict(_ABS_STRIDED_SHAPE),
        family="abs_f16",
        source_id="abs_f16",
        root_symbol="@hrx2_abs_f16_non_contiguous_4d",
        export_name="hrx2_abs_f16_non_contiguous_4d",
        op="ABS",
        source_path="kernels/v2/abs/non_contiguous_4d.loom",
    )

    result = generate_oracle(candidate, tmp_path / "fixtures", force=True)

    assert result.status == "fixtures_ready"
    src0_bits = np.load(tmp_path / "fixtures" / "src0.npy")
    expected = np.load(tmp_path / "fixtures" / "expected.npy")
    assert src0_bits.dtype == np.int16
    assert src0_bits.shape == (_ABS_STRIDED_BUFFER_LEN,)
    assert expected.shape == (_ABS_STRIDED_DST_ELEMS,)
    view = _abs_gathered_view(src0_bits.view(np.float16))
    want = np.abs(view.astype(np.float32)).astype(np.float16).view(np.int16)
    assert np.array_equal(expected, want)

    linked_source = tmp_path / "linked.loom"
    linked_source.write_text(
        'kernel.def export("hrx2_abs_f16_non_contiguous_4d") @hrx2_abs_f16_non_contiguous_4d() {}\n',
        encoding="utf-8",
    )
    _, metadata = write_workbench(candidate, linked_source, tmp_path / "workbench.loom", tmp_path / "fixtures")

    assert metadata["status"] == "ok"
    workbench = (tmp_path / "workbench.loom").read_text(encoding="utf-8")
    assert "tensor<410xi16>, tensor<24xi16>" in workbench
    assert "check.expect.equal" in workbench


def test_abs_oracle_contiguous_view_is_unchanged(tmp_path: Path) -> None:
    # v=0 (contiguous) shape: the strided gate is inert and the oracle stays the flat total-size path.
    candidate = _candidate(
        candidate_id="abs_f32_contiguous_1d",
        shape={"d0": 2, "d1": 3, "d2": 2, "d3": 2, "pointwise.d1": 12},
        family="abs_f32",
        source_id="abs_f32",
        root_symbol="@hrx2_abs_f32_contiguous_1d",
        export_name="hrx2_abs_f32_contiguous_1d",
        op="ABS",
        source_path="kernels/v2/abs/contiguous_1d.loom",
    )

    result = generate_oracle(candidate, tmp_path / "fixtures", force=True)

    assert result.status == "fixtures_ready"
    src0 = np.load(tmp_path / "fixtures" / "src0.npy")
    expected = np.load(tmp_path / "fixtures" / "expected.npy")
    assert src0.shape == (24,)
    assert expected.shape == (24,)
    assert np.array_equal(expected, np.abs(src0))


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


def test_rms_norm_mul_oracle_and_workbench_use_fused_kernel_abi(tmp_path: Path) -> None:
    candidate = _candidate(
        candidate_id="rms_norm_mul_f32_n16_r60_vector_tail",
        shape={"d0": 16, "d1": 60, "ncols": 16, "nrows": 60},
        family="rms_norm_mul_f32",
        source_id="rms_norm_mul_f32",
        root_symbol="@rms_norm_mul_f32_static_vector_tail",
        export_name="rms_norm_mul_f32_static_vector_tail",
        op="MUL",
        source_path="kernels/v2/mul/rms_norm_mul_f32_vector_tail.loom",
    )

    result = generate_oracle(candidate, tmp_path / "fixtures", force=True)

    assert result.status == "fixtures_ready"
    assert np.load(tmp_path / "fixtures" / "src.npy").shape == (960,)
    assert np.load(tmp_path / "fixtures" / "weight.npy").shape == (16,)
    assert np.load(tmp_path / "fixtures" / "dst_init.npy").shape == (960,)
    assert np.load(tmp_path / "fixtures" / "expected.npy").shape == (960,)

    linked_source = tmp_path / "linked.loom"
    linked_source.write_text(
        'kernel.def export("rms_norm_mul_f32_static_vector_tail") @rms_norm_mul_f32_static_vector_tail() {}\n',
        encoding="utf-8",
    )
    _, metadata = write_workbench(candidate, linked_source, tmp_path / "workbench.loom", tmp_path / "fixtures")

    assert metadata["status"] == "ok"
    workbench = (tmp_path / "workbench.loom").read_text(encoding="utf-8")
    assert "%eps = check.literal value(0.0) : f32" in workbench
    assert "tensor<960xf32>, tensor<16xf32>, tensor<960xf32>" in workbench
    assert "check.expect.close" in workbench


def test_add_rms_norm_mul_oracle_and_workbench_use_fused_kernel_abi(tmp_path: Path) -> None:
    candidate = _candidate(
        candidate_id="add_rms_norm_mul_f32_n64_r60_vector_tail",
        shape={"d0": 64, "d1": 60, "ncols": 64, "nrows": 60},
        family="add_rms_norm_mul_f32",
        source_id="add_rms_norm_mul_f32",
        root_symbol="@add_rms_norm_mul_f32_static_vector_tail",
        export_name="add_rms_norm_mul_f32_static_vector_tail",
        op="ADD_RMS_NORM",
        source_path="kernels/v2/mul/add_rms_norm_mul_f32_vector_tail.loom",
    )

    result = generate_oracle(candidate, tmp_path / "fixtures", force=True)

    assert result.status == "fixtures_ready"
    assert np.load(tmp_path / "fixtures" / "src0.npy").shape == (3840,)
    assert np.load(tmp_path / "fixtures" / "src1.npy").shape == (3840,)
    assert np.load(tmp_path / "fixtures" / "add_dst_init.npy").shape == (3840,)
    assert np.load(tmp_path / "fixtures" / "weight.npy").shape == (64,)
    assert np.load(tmp_path / "fixtures" / "dst_init.npy").shape == (3840,)
    assert np.load(tmp_path / "fixtures" / "added.npy").shape == (3840,)
    assert np.load(tmp_path / "fixtures" / "expected.npy").shape == (3840,)

    linked_source = tmp_path / "linked.loom"
    linked_source.write_text(
        'kernel.def export("add_rms_norm_mul_f32_static_vector_tail") @add_rms_norm_mul_f32_static_vector_tail() {}\n',
        encoding="utf-8",
    )
    _, metadata = write_workbench(candidate, linked_source, tmp_path / "workbench.loom", tmp_path / "fixtures")

    assert metadata["status"] == "ok"
    workbench = (tmp_path / "workbench.loom").read_text(encoding="utf-8")
    assert "%eps = check.literal value(0.0) : f32" in workbench
    assert "tensor<3840xf32>, tensor<3840xf32>, tensor<3840xf32>, tensor<64xf32>, tensor<3840xf32>" in workbench
    assert "actual(%add_dst) expected(%added)" in workbench
    assert "actual(%dst) expected(%expected)" in workbench


def test_rms_norm_mul_quantize_oracle_and_workbench_use_fused_kernel_abi(tmp_path: Path) -> None:
    candidate = _candidate(
        candidate_id="rms_norm_mul_quantize_q8_1_f32_n3072_r1_x4_recompute",
        shape={"d0": 3072, "d1": 1, "weight_d1": 1, "weight_d1_stride": 0, "ncols": 3072, "nrows": 1},
        family="rms_norm_mul_quantize_q8_1_f32",
        source_id="rms_norm_mul_quantize_q8_1_f32",
        root_symbol="@rms_norm_mul_quantize_q8_1_x4_f32_recompute",
        export_name="rms_norm_mul_quantize_q8_1_x4_f32_recompute",
        op="QUANTIZE",
        source_path="kernels/v2/quantize/rms_norm_mul_quantize_q8_1_f32_x4_recompute.loom",
    )

    result = generate_oracle(candidate, tmp_path / "fixtures", force=True)

    assert result.status == "fixtures_ready"
    assert np.load(tmp_path / "fixtures" / "src.npy").shape == (3072,)
    assert np.load(tmp_path / "fixtures" / "weight.npy").shape == (3072,)
    assert np.load(tmp_path / "fixtures" / "dst_init.npy").shape == (3456,)
    assert np.load(tmp_path / "fixtures" / "expected.npy").shape == (3456,)

    linked_source = tmp_path / "linked.loom"
    linked_source.write_text(
        (
            'kernel.def export("rms_norm_mul_quantize_q8_1_x4_f32_recompute") '
            "@rms_norm_mul_quantize_q8_1_x4_f32_recompute() {}\n"
        ),
        encoding="utf-8",
    )
    _, metadata = write_workbench(candidate, linked_source, tmp_path / "workbench.loom", tmp_path / "fixtures")

    assert metadata["status"] == "ok"
    workbench = (tmp_path / "workbench.loom").read_text(encoding="utf-8")
    assert "%eps = check.literal value(0.0) : f32" in workbench
    assert "tensor<3072xf32>, tensor<3072xf32>, tensor<3456xi8>" in workbench
    assert "func.call @rms_norm_mul_quantize_q8_1_x4_f32_recompute(%eps, %src, %weight, %dst)" in workbench
    assert "check.expect.equal actual(%dst) expected(%expected)" in workbench


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


def test_get_rows_q8_0_oracle_and_workbench_use_packed_src0(tmp_path: Path) -> None:
    candidate = Candidate(
        id="get_rows_q8_0_f32_ranked_2d",
        family="get_rows_q8_0_f32",
        op="GET_ROWS",
        source_id="get_rows_q8_0_f32",
        source_path=Path("kernels/v2/get_rows/q8_0_f32_embedding_rows_2d.loom"),
        root_symbol="@get_rows_q8_0_f32",
        export_name="get_rows_q8_0_f32",
        route_id="get_rows_q8_0_f32_test",
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
    assert np.load(tmp_path / "fixtures" / "src0.npy").shape == (1360,)
    assert np.load(tmp_path / "fixtures" / "src0.npy").dtype == np.int8
    assert np.load(tmp_path / "fixtures" / "indices.npy").shape == (4,)
    assert np.load(tmp_path / "fixtures" / "expected.npy").shape == (1024,)

    linked_source = tmp_path / "linked.loom"
    linked_source.write_text('kernel.def export("get_rows_q8_0_f32") @get_rows_q8_0_f32() {}\n', encoding="utf-8")
    _, metadata = write_workbench(candidate, linked_source, tmp_path / "workbench.loom", tmp_path / "fixtures")

    assert metadata["status"] == "ok"
    workbench = (tmp_path / "workbench.loom").read_text(encoding="utf-8")
    assert "tensor<1360xi8>, tensor<4xi32>, tensor<1024xf32>" in workbench


def test_get_rows_q4_k_oracle_and_workbench_use_packed_src0(tmp_path: Path) -> None:
    candidate = Candidate(
        id="get_rows_q4_k_f32_ranked_2d",
        family="get_rows_q4_k_f32",
        op="GET_ROWS",
        source_id="get_rows_q4_k_f32",
        source_path=Path("kernels/v2/get_rows/q4_k_f32_embedding_rows_2d.loom"),
        root_symbol="@get_rows_q4_k_f32",
        export_name="get_rows_q4_k_f32",
        route_id="get_rows_q4_k_f32_test",
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
    assert np.load(tmp_path / "fixtures" / "src0.npy").shape == (720,)
    assert np.load(tmp_path / "fixtures" / "src0.npy").dtype == np.int8
    assert np.load(tmp_path / "fixtures" / "indices.npy").shape == (4,)
    assert np.load(tmp_path / "fixtures" / "expected.npy").shape == (1024,)

    linked_source = tmp_path / "linked.loom"
    linked_source.write_text('kernel.def export("get_rows_q4_k_f32") @get_rows_q4_k_f32() {}\n', encoding="utf-8")
    _, metadata = write_workbench(candidate, linked_source, tmp_path / "workbench.loom", tmp_path / "fixtures")

    assert metadata["status"] == "ok"
    workbench = (tmp_path / "workbench.loom").read_text(encoding="utf-8")
    assert "tensor<720xi8>, tensor<4xi32>, tensor<1024xf32>" in workbench


def test_get_rows_q5_k_oracle_and_workbench_use_packed_src0(tmp_path: Path) -> None:
    candidate = Candidate(
        id="get_rows_q5_k_f32_ranked_2d",
        family="get_rows_q5_k_f32",
        op="GET_ROWS",
        source_id="get_rows_q5_k_f32",
        source_path=Path("kernels/v2/get_rows/q5_k_f32_embedding_rows_2d.loom"),
        root_symbol="@get_rows_q5_k_f32",
        export_name="get_rows_q5_k_f32",
        route_id="get_rows_q5_k_f32_test",
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
    assert np.load(tmp_path / "fixtures" / "src0.npy").shape == (880,)
    assert np.load(tmp_path / "fixtures" / "src0.npy").dtype == np.int8
    assert np.load(tmp_path / "fixtures" / "indices.npy").shape == (4,)
    assert np.load(tmp_path / "fixtures" / "expected.npy").shape == (1024,)

    linked_source = tmp_path / "linked.loom"
    linked_source.write_text('kernel.def export("get_rows_q5_k_f32") @get_rows_q5_k_f32() {}\n', encoding="utf-8")
    _, metadata = write_workbench(candidate, linked_source, tmp_path / "workbench.loom", tmp_path / "fixtures")

    assert metadata["status"] == "ok"
    workbench = (tmp_path / "workbench.loom").read_text(encoding="utf-8")
    assert "tensor<880xi8>, tensor<4xi32>, tensor<1024xf32>" in workbench


def test_get_rows_q6_k_oracle_and_workbench_use_packed_src0(tmp_path: Path) -> None:
    candidate = Candidate(
        id="get_rows_q6_k_f32_ranked_2d",
        family="get_rows_q6_k_f32",
        op="GET_ROWS",
        source_id="get_rows_q6_k_f32",
        source_path=Path("kernels/v2/get_rows/q6_k_f32_embedding_rows_2d.loom"),
        root_symbol="@get_rows_q6_k_f32",
        export_name="get_rows_q6_k_f32",
        route_id="get_rows_q6_k_f32_test",
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
    assert np.load(tmp_path / "fixtures" / "src0.npy").shape == (1050,)
    assert np.load(tmp_path / "fixtures" / "src0.npy").dtype == np.int8
    assert np.load(tmp_path / "fixtures" / "indices.npy").shape == (4,)
    assert np.load(tmp_path / "fixtures" / "expected.npy").shape == (1024,)

    linked_source = tmp_path / "linked.loom"
    linked_source.write_text('kernel.def export("get_rows_q6_k_f32") @get_rows_q6_k_f32() {}\n', encoding="utf-8")
    _, metadata = write_workbench(candidate, linked_source, tmp_path / "workbench.loom", tmp_path / "fixtures")

    assert metadata["status"] == "ok"
    workbench = (tmp_path / "workbench.loom").read_text(encoding="utf-8")
    assert "tensor<1050xi8>, tensor<4xi32>, tensor<1024xf32>" in workbench


def test_get_rows_moe_weights_oracle_and_workbench_use_token_strides(tmp_path: Path) -> None:
    candidate = Candidate(
        id="get_rows_moe_weights_f32_ranked_2d",
        family="get_rows_moe_weights_f32",
        op="GET_ROWS",
        source_id="get_rows_moe_weights_f32",
        source_path=Path("kernels/v2/get_rows/moe_weights_f32_topk_view_2d.loom"),
        root_symbol="@get_rows_moe_weights_f32",
        export_name="get_rows_moe_weights_f32",
        route_id="get_rows_moe_weights_f32_test",
        route=None,
        shape={"d0": 8, "d1": 16, "src0_d0": 128},
        values={
            "shape.get_rows_moe.nexperts": 128,
            "shape.get_rows_moe.nselected": 8,
            "shape.get_rows_moe.ntokens": 16,
            "shape.get_rows_moe.src0_token_stride": 128,
            "shape.get_rows_moe.idx_token_stride": 8,
            "shape.get_rows_moe.dst_token_stride": 8,
        },
        config={},
        dispatch={},
        supports={},
        coverage="route_backed",
    )

    result = generate_oracle(candidate, tmp_path / "fixtures", force=True)

    assert result.status == "fixtures_ready"
    assert np.load(tmp_path / "fixtures" / "src0.npy").shape == (2048,)
    assert np.load(tmp_path / "fixtures" / "indices.npy").shape == (128,)
    assert np.load(tmp_path / "fixtures" / "dst_init.npy").shape == (128,)
    assert np.load(tmp_path / "fixtures" / "expected.npy").shape == (128,)

    linked_source = tmp_path / "linked.loom"
    linked_source.write_text('kernel.def export("get_rows_moe_weights_f32") @get_rows_moe_weights_f32() {}\n', encoding="utf-8")
    _, metadata = write_workbench(candidate, linked_source, tmp_path / "workbench.loom", tmp_path / "fixtures")

    assert metadata["status"] == "ok"
    workbench = (tmp_path / "workbench.loom").read_text(encoding="utf-8")
    assert "tensor<2048xf32>, tensor<128xi32>, tensor<128xf32>" in workbench


def test_argsort_oracle_and_workbench_use_flattened_rows(tmp_path: Path) -> None:
    candidate = _candidate(
        candidate_id="argsort_f32_i32_n128_r1_desc_wg128",
        shape={"d0": 128, "d1": 1},
        family="argsort_f32_i32",
        source_id="argsort_f32_i32",
        root_symbol="@hrx2_argsort_f32_i32_desc",
        export_name="hrx2_argsort_f32_i32_desc",
        op="ARGSORT",
        source_path="kernels/v2/argsort/argsort_f32_i32.loom",
    )

    result = generate_oracle(candidate, tmp_path / "fixtures", force=True)

    assert result.status == "fixtures_ready"
    assert np.load(tmp_path / "fixtures" / "src0.npy").shape == (128,)
    assert np.load(tmp_path / "fixtures" / "dst_init.npy").shape == (128,)
    assert np.load(tmp_path / "fixtures" / "expected.npy").shape == (128,)

    linked_source = tmp_path / "linked.loom"
    linked_source.write_text('kernel.def export("hrx2_argsort_f32_i32_desc") @hrx2_argsort_f32_i32_desc() {}\n', encoding="utf-8")
    _, metadata = write_workbench(candidate, linked_source, tmp_path / "workbench.loom", tmp_path / "fixtures")

    assert metadata["status"] == "ok"
    workbench = (tmp_path / "workbench.loom").read_text(encoding="utf-8")
    assert "tensor<128xf32>, tensor<128xi32>" in workbench
    assert "check.expect.equal" in workbench


def test_mul_mat_f32_oracle_and_workbench_use_kernel_abi_buffers(tmp_path: Path) -> None:
    candidate = _candidate(
        candidate_id="mul_mat_f32_f32_contiguous_small_2d",
        shape={"k": 256, "rows": 16, "cols": 8},
        family="mul_mat_f32_f32",
        source_id="mul_mat_f32_f32",
        root_symbol="@hrx2_mul_mat_f32_f32_static",
        export_name="hrx2_mul_mat_f32_f32_static",
        op="MUL_MAT",
        source_path="kernels/v2/mul_mat/contiguous_f32_f32.loom",
    )

    result = generate_oracle(candidate, tmp_path / "fixtures", force=True)

    assert result.status == "fixtures_ready"
    assert np.load(tmp_path / "fixtures" / "src0_logical_f32.npy").shape == (4096,)
    assert np.load(tmp_path / "fixtures" / "src1.npy").shape == (2048,)
    assert np.load(tmp_path / "fixtures" / "dst_init.npy").shape == (128,)
    assert np.load(tmp_path / "fixtures" / "expected.npy").shape == (128,)

    linked_source = tmp_path / "linked.loom"
    linked_source.write_text(
        'kernel.def export("hrx2_mul_mat_f32_f32_static") @hrx2_mul_mat_f32_f32_static() {}\n',
        encoding="utf-8",
    )
    _, metadata = write_workbench(candidate, linked_source, tmp_path / "workbench.loom", tmp_path / "fixtures")

    assert metadata["status"] == "ok"
    workbench = (tmp_path / "workbench.loom").read_text(encoding="utf-8")
    assert "tensor<4096xf32>, tensor<2048xf32>, tensor<128xf32>" in workbench
    assert "check.expect.close" in workbench


def test_mul_mat_f16_f32_batched_oracle_and_workbench_use_kernel_abi_buffers(tmp_path: Path) -> None:
    candidate = _candidate(
        candidate_id="mul_mat_f16_f32_batched_logits_cols1_2d",
        shape={"k": 1056, "rows": 128, "cols": 1},
        family="mul_mat_f16_f32_batched",
        source_id="mul_mat_f16_f32_batched",
        root_symbol="@hrx2_mul_mat_f16_f32_batched",
        export_name="hrx2_mul_mat_f16_f32_batched",
        op="MUL_MAT",
        source_path="kernels/v2/mul_mat/f16_f32_batched.loom",
    )

    result = generate_oracle(candidate, tmp_path / "fixtures", force=True)

    assert result.status == "fixtures_ready"
    assert np.load(tmp_path / "fixtures" / "src0.npy").shape == (135168,)
    assert np.load(tmp_path / "fixtures" / "src1.npy").shape == (1056,)
    assert np.load(tmp_path / "fixtures" / "dst_init.npy").shape == (128,)
    assert np.load(tmp_path / "fixtures" / "expected.npy").shape == (128,)

    linked_source = tmp_path / "linked.loom"
    linked_source.write_text(
        'kernel.def export("hrx2_mul_mat_f16_f32_batched") @hrx2_mul_mat_f16_f32_batched() {}\n',
        encoding="utf-8",
    )
    _, metadata = write_workbench(candidate, linked_source, tmp_path / "workbench.loom", tmp_path / "fixtures")

    assert metadata["status"] == "ok"
    workbench = (tmp_path / "workbench.loom").read_text(encoding="utf-8")
    assert "tensor<135168xf16>, tensor<1056xf32>, tensor<128xf32>" in workbench
    assert "check.expect.close" in workbench


def test_mul_mat_f16_f32_batched_4d_oracle_and_workbench_use_kernel_abi_buffers(tmp_path: Path) -> None:
    candidate = _candidate(
        candidate_id="mul_mat_f16_f32_batched_logits_cols1_4d",
        shape={
            "d0": 128,
            "d1": 1,
            "d2": 2,
            "d3": 3,
            "src0_d0": 1056,
            "src0_d1": 128,
            "src1_d0": 1056,
            "k": 1056,
            "rows": 128,
            "cols": 1,
        },
        family="mul_mat_f16_f32_batched",
        source_id="mul_mat_f16_f32_batched",
        root_symbol="@hrx2_mul_mat_f16_f32_batched",
        export_name="hrx2_mul_mat_f16_f32_batched",
        op="MUL_MAT",
        source_path="kernels/v2/mul_mat/f16_f32_batched.loom",
    )

    result = generate_oracle(candidate, tmp_path / "fixtures", force=True)

    assert result.status == "fixtures_ready"
    assert np.load(tmp_path / "fixtures" / "src0.npy").shape == (135168,)
    assert np.load(tmp_path / "fixtures" / "src1.npy").shape == (6336,)
    assert np.load(tmp_path / "fixtures" / "dst_init.npy").shape == (768,)
    assert np.load(tmp_path / "fixtures" / "expected.npy").shape == (768,)

    linked_source = tmp_path / "linked.loom"
    linked_source.write_text(
        'kernel.def export("hrx2_mul_mat_f16_f32_batched") @hrx2_mul_mat_f16_f32_batched() {}\n',
        encoding="utf-8",
    )
    _, metadata = write_workbench(candidate, linked_source, tmp_path / "workbench.loom", tmp_path / "fixtures")

    assert metadata["status"] == "ok"
    workbench = (tmp_path / "workbench.loom").read_text(encoding="utf-8")
    assert "tensor<135168xf16>, tensor<6336xf32>, tensor<768xf32>" in workbench
    assert "check.expect.close" in workbench


def test_mul_mat_q4_k_oracle_and_workbench_use_kernel_abi_buffers(tmp_path: Path) -> None:
    candidate = _candidate(
        candidate_id="mul_mat_q4_k_f32_direct_contiguous_2d",
        shape={"k": 256, "rows": 16, "cols": 8},
        family="mul_mat_q4_k_f32",
        source_id="mul_mat_q4_k_f32",
        root_symbol="@hrx2_mul_mat_q4_k_f32_static",
        export_name="hrx2_mul_mat_q4_k_f32_static",
        op="MUL_MAT",
        source_path="kernels/v2/mul_mat/q4_k_f32_direct.loom",
    )

    result = generate_oracle(candidate, tmp_path / "fixtures", force=True)

    assert result.status == "fixtures_ready"
    assert np.load(tmp_path / "fixtures" / "src0.npy").shape == (2304,)
    assert np.load(tmp_path / "fixtures" / "src1.npy").shape == (2048,)
    assert np.load(tmp_path / "fixtures" / "dst_init.npy").shape == (128,)
    assert np.load(tmp_path / "fixtures" / "expected.npy").shape == (128,)

    linked_source = tmp_path / "linked.loom"
    linked_source.write_text(
        'kernel.def export("hrx2_mul_mat_q4_k_f32_static") @hrx2_mul_mat_q4_k_f32_static() {}\n',
        encoding="utf-8",
    )
    _, metadata = write_workbench(candidate, linked_source, tmp_path / "workbench.loom", tmp_path / "fixtures")

    assert metadata["status"] == "ok"
    workbench = (tmp_path / "workbench.loom").read_text(encoding="utf-8")
    assert "tensor<2304xi8>, tensor<2048xf32>, tensor<128xf32>" in workbench
    assert "check.expect.close" in workbench


def test_mul_mat_id_q4_k_oracle_and_workbench_use_kernel_abi_buffers(tmp_path: Path) -> None:
    candidate = _candidate(
        candidate_id="mul_mat_id_q4_k_f32_expert_planes_3d",
        shape={
            "k": 256,
            "rows": 16,
            "nexperts": 4,
            "nselected": 2,
            "ntokens": 3,
            "src1_selected_stride": 256,
            "src1_token_stride": 512,
            "idx_token_stride": 2,
            "dst_token_stride": 32,
        },
        family="mul_mat_id_q4_k_f32",
        source_id="mul_mat_id_q4_k_f32",
        root_symbol="@mul_mat_id_q4_k_f32_static",
        export_name="mul_mat_id_q4_k_f32_static",
        op="MUL_MAT_ID",
        source_path="kernels/v2/mul_mat_id/q4_k_f32_expert_planes.loom",
    )

    result = generate_oracle(candidate, tmp_path / "fixtures", force=True)

    assert result.status == "fixtures_ready"
    assert np.load(tmp_path / "fixtures" / "src0.npy").shape == (9216,)
    assert np.load(tmp_path / "fixtures" / "src1.npy").shape == (1536,)
    assert np.load(tmp_path / "fixtures" / "idx.npy").shape == (6,)
    assert np.load(tmp_path / "fixtures" / "dst_init.npy").shape == (96,)
    assert np.load(tmp_path / "fixtures" / "expected.npy").shape == (96,)

    linked_source = tmp_path / "linked.loom"
    linked_source.write_text(
        'kernel.def export("mul_mat_id_q4_k_f32_static") @mul_mat_id_q4_k_f32_static() {}\n',
        encoding="utf-8",
    )
    _, metadata = write_workbench(candidate, linked_source, tmp_path / "workbench.loom", tmp_path / "fixtures")

    assert metadata["status"] == "ok"
    workbench = (tmp_path / "workbench.loom").read_text(encoding="utf-8")
    assert "tensor<9216xi8>, tensor<1536xf32>, tensor<6xi32>, tensor<96xf32>" in workbench
    assert "check.expect.close" in workbench


def test_mul_mat_id_q5_k_oracle_and_workbench_use_kernel_abi_buffers(tmp_path: Path) -> None:
    candidate = _candidate(
        candidate_id="mul_mat_id_q5_k_f32_expert_planes_3d",
        shape={
            "k": 256,
            "rows": 16,
            "nexperts": 4,
            "nselected": 2,
            "ntokens": 3,
            "src1_selected_stride": 256,
            "src1_token_stride": 512,
            "idx_token_stride": 2,
            "dst_token_stride": 32,
        },
        family="mul_mat_id_q5_k_f32",
        source_id="mul_mat_id_q5_k_f32",
        root_symbol="@mul_mat_id_q5_k_f32_static",
        export_name="mul_mat_id_q5_k_f32_static",
        op="MUL_MAT_ID",
        source_path="kernels/v2/mul_mat_id/q5_k_f32_expert_planes.loom",
    )

    result = generate_oracle(candidate, tmp_path / "fixtures", force=True)

    assert result.status == "fixtures_ready"
    assert np.load(tmp_path / "fixtures" / "src0.npy").shape == (11264,)
    assert np.load(tmp_path / "fixtures" / "src1.npy").shape == (1536,)
    assert np.load(tmp_path / "fixtures" / "idx.npy").shape == (6,)
    assert np.load(tmp_path / "fixtures" / "dst_init.npy").shape == (96,)
    assert np.load(tmp_path / "fixtures" / "expected.npy").shape == (96,)

    linked_source = tmp_path / "linked.loom"
    linked_source.write_text(
        'kernel.def export("mul_mat_id_q5_k_f32_static") @mul_mat_id_q5_k_f32_static() {}\n',
        encoding="utf-8",
    )
    _, metadata = write_workbench(candidate, linked_source, tmp_path / "workbench.loom", tmp_path / "fixtures")

    assert metadata["status"] == "ok"
    workbench = (tmp_path / "workbench.loom").read_text(encoding="utf-8")
    assert "tensor<11264xi8>, tensor<1536xf32>, tensor<6xi32>, tensor<96xf32>" in workbench
    assert "check.expect.close" in workbench


def test_mul_mat_id_q6_k_oracle_and_workbench_use_kernel_abi_buffers(tmp_path: Path) -> None:
    candidate = _candidate(
        candidate_id="mul_mat_id_q6_k_f32_expert_planes_3d",
        shape={
            "k": 256,
            "rows": 16,
            "nexperts": 4,
            "nselected": 2,
            "ntokens": 3,
            "src1_selected_stride": 256,
            "src1_token_stride": 512,
            "idx_token_stride": 2,
            "dst_token_stride": 32,
        },
        family="mul_mat_id_q6_k_f32",
        source_id="mul_mat_id_q6_k_f32",
        root_symbol="@mul_mat_id_q6_k_f32_static",
        export_name="mul_mat_id_q6_k_f32_static",
        op="MUL_MAT_ID",
        source_path="kernels/v2/mul_mat_id/q6_k_f32_expert_planes.loom",
    )

    result = generate_oracle(candidate, tmp_path / "fixtures", force=True)

    assert result.status == "fixtures_ready"
    assert np.load(tmp_path / "fixtures" / "src0.npy").shape == (13440,)
    assert np.load(tmp_path / "fixtures" / "src1.npy").shape == (1536,)
    assert np.load(tmp_path / "fixtures" / "idx.npy").shape == (6,)
    assert np.load(tmp_path / "fixtures" / "dst_init.npy").shape == (96,)
    assert np.load(tmp_path / "fixtures" / "expected.npy").shape == (96,)

    linked_source = tmp_path / "linked.loom"
    linked_source.write_text(
        'kernel.def export("mul_mat_id_q6_k_f32_static") @mul_mat_id_q6_k_f32_static() {}\n',
        encoding="utf-8",
    )
    _, metadata = write_workbench(candidate, linked_source, tmp_path / "workbench.loom", tmp_path / "fixtures")

    assert metadata["status"] == "ok"
    workbench = (tmp_path / "workbench.loom").read_text(encoding="utf-8")
    assert "tensor<13440xi8>, tensor<1536xf32>, tensor<6xi32>, tensor<96xf32>" in workbench
    assert "check.expect.close" in workbench


def test_mul_mat_q6_k_oracle_and_workbench_use_packed_kernel_abi_buffers(tmp_path: Path) -> None:
    candidate = _candidate(
        candidate_id="mul_mat_q6_k_f32_direct_contiguous_2d",
        shape={"k": 256, "rows": 16, "cols": 8},
        family="mul_mat_q6_k_f32",
        source_id="mul_mat_q6_k_f32",
        root_symbol="@hrx2_mul_mat_q6_k_f32_static",
        export_name="hrx2_mul_mat_q6_k_f32_static",
        op="MUL_MAT",
        source_path="kernels/v2/mul_mat/q6_k_f32_direct.loom",
    )

    result = generate_oracle(candidate, tmp_path / "fixtures", force=True)

    assert result.status == "fixtures_ready"
    assert np.load(tmp_path / "fixtures" / "src0.npy").shape == (3360,)
    assert np.load(tmp_path / "fixtures" / "src1.npy").shape == (2048,)
    assert np.load(tmp_path / "fixtures" / "dst_init.npy").shape == (128,)
    assert np.load(tmp_path / "fixtures" / "expected.npy").shape == (128,)

    linked_source = tmp_path / "linked.loom"
    linked_source.write_text(
        'kernel.def export("hrx2_mul_mat_q6_k_f32_static") @hrx2_mul_mat_q6_k_f32_static() {}\n',
        encoding="utf-8",
    )
    _, metadata = write_workbench(candidate, linked_source, tmp_path / "workbench.loom", tmp_path / "fixtures")

    assert metadata["status"] == "ok"
    workbench = (tmp_path / "workbench.loom").read_text(encoding="utf-8")
    assert "tensor<3360xi8>, tensor<2048xf32>, tensor<128xf32>" in workbench
    assert "check.expect.close" in workbench


def _emulate_mul_mat_q5_dot16_rows(src0: np.ndarray, src1: np.ndarray) -> np.ndarray:
    rows = src0.size // 176
    blocks = src0.view(np.uint8).reshape(rows, 176)
    rhs = src1.reshape(256).astype(np.float32)
    out = np.empty((rows,), dtype=np.float32)
    for row in range(rows):
        block = blocks[row]
        d = block[0:2].copy().view(np.float16).astype(np.float32)[0]
        dmin = block[2:4].copy().view(np.float16).astype(np.float32)[0]
        total = np.float32(0.0)
        for lane in range(16):
            itid = lane
            il = itid // 4
            ir = itid % 4
            v_im = il // 2
            v_in = il % 2
            lane_q = ir * 4 + v_in * 2
            v_im32 = v_im * 32
            v_im64 = v_im * 64
            qs0_rel = v_im32 + lane_q
            qs4_rel = v_im32 + 64 + lane_q

            scale0 = int(block[4 + v_im * 2]) | (int(block[5 + v_im * 2]) << 8)
            scale4 = int(block[4 + (v_im + 2) * 2]) | (int(block[5 + (v_im + 2) * 2]) << 8)
            scale8_raw = int(block[4 + (v_im + 4) * 2]) | (int(block[5 + (v_im + 4) * 2]) << 8)
            scale_0_4_l = scale0 | (scale4 << 16)
            scale_0_4_h = (scale_0_4_l & 0xC0C0C0C0) >> 2
            scale_0_4_l6 = scale_0_4_l & 0x3F3F3F3F
            scale8_dup = (scale8_raw << 12) | scale8_raw
            scale8 = (scale8_dup & 0x0F0F0F0F) | scale_0_4_h

            def i32_from_4(base: int) -> int:
                return (
                    int(block[base])
                    | (int(block[base + 1]) << 8)
                    | (int(block[base + 16]) << 16)
                    | (int(block[base + 17]) << 24)
                )

            qs0 = i32_from_4(48 + qs0_rel)
            qs4 = i32_from_4(48 + qs4_rel)
            qh = i32_from_4(16 + lane_q)
            qh_shifted = qh >> (v_im * 2)
            qs0_lo = (qs0 & 0x0F0F0F0F) + ((qh_shifted & 0x01010101) << 4)
            qs0_hi = ((qs0 >> 4) & 0x0F0F0F0F) + ((qh_shifted & 0x02020202) << 3)
            qs4_lo = (qs4 & 0x0F0F0F0F) + (qh_shifted & 0x10101010)
            qs4_hi = ((qs4 >> 4) & 0x0F0F0F0F) + ((qh_shifted & 0x20202020) >> 1)

            def bytes4(word: int) -> np.ndarray:
                return np.array([(word >> shift) & 0xFF for shift in (0, 8, 16, 24)], dtype=np.float32)

            q0, q1, q2, q3 = map(bytes4, (qs0_lo, qs0_hi, qs4_lo, qs4_hi))
            base = v_im64 + lane_q
            idxs = (
                base,
                base + 1,
                base + 16,
                base + 17,
                base + 32,
                base + 33,
                base + 48,
                base + 49,
                base + 128,
                base + 129,
                base + 144,
                base + 145,
                base + 160,
                base + 161,
                base + 176,
                base + 177,
            )
            by0 = np.array([rhs[idxs[0]], rhs[idxs[1]], rhs[idxs[2]], rhs[idxs[3]]], dtype=np.float32)
            by1 = np.array([rhs[idxs[4]], rhs[idxs[5]], rhs[idxs[6]], rhs[idxs[7]]], dtype=np.float32)
            by2 = np.array([rhs[idxs[8]], rhs[idxs[9]], rhs[idxs[10]], rhs[idxs[11]]], dtype=np.float32)
            by3 = np.array([rhs[idxs[12]], rhs[idxs[13]], rhs[idxs[14]], rhs[idxs[15]]], dtype=np.float32)
            sx, sy, sz, sw = [np.sum(lhs * rhs_block, dtype=np.float32) for lhs, rhs_block in ((q0, by0), (q1, by1), (q2, by2), (q3, by3))]
            by0_sum, by1_sum, by2_sum, by3_sum = [np.sum(values, dtype=np.float32) for values in (by0, by1, by2, by3)]
            scale_a, scale_b, min_a, min_b = [np.float32((scale_0_4_l6 >> shift) & 0xFF) for shift in (0, 8, 16, 24)]
            scale_c, scale_d, min_c, min_d = [np.float32((scale8 >> shift) & 0xFF) for shift in (0, 8, 16, 24)]
            total += d * (sx * scale_a + sy * scale_b + sz * scale_c + sw * scale_d)
            total -= dmin * (by0_sum * min_a + by1_sum * min_b + by2_sum * min_c + by3_sum * min_d)
        out[row] = total
    return out


def test_mul_mat_q5_k_oracle_and_workbench_use_packed_kernel_abi_buffers(tmp_path: Path) -> None:
    candidate = _candidate(
        candidate_id="mul_mat_q5_k_f32_dot16_contiguous_cols1_2d",
        shape={"k": 256, "rows": 16, "cols": 1},
        family="mul_mat_q5_k_f32",
        source_id="mul_mat_q5_k_f32",
        root_symbol="@hrx2_mul_mat_q5_k_f32_dot16_static",
        export_name="hrx2_mul_mat_q5_k_f32_dot16_static",
        op="MUL_MAT",
        source_path="kernels/v2/mul_mat/q5_k_f32_dot16.loom",
    )

    result = generate_oracle(candidate, tmp_path / "fixtures", force=True)

    assert result.status == "fixtures_ready"
    src0 = np.load(tmp_path / "fixtures" / "src0.npy")
    src1 = np.load(tmp_path / "fixtures" / "src1.npy")
    expected = np.load(tmp_path / "fixtures" / "expected.npy")
    assert src0.shape == (2816,)
    assert src1.shape == (256,)
    assert np.load(tmp_path / "fixtures" / "dst_init.npy").shape == (16,)
    assert expected.shape == (16,)
    assert np.allclose(_emulate_mul_mat_q5_dot16_rows(src0, src1), expected, atol=1e-5, rtol=1e-5)

    linked_source = tmp_path / "linked.loom"
    linked_source.write_text(
        'kernel.def export("hrx2_mul_mat_q5_k_f32_dot16_static") @hrx2_mul_mat_q5_k_f32_dot16_static() {}\n',
        encoding="utf-8",
    )
    _, metadata = write_workbench(candidate, linked_source, tmp_path / "workbench.loom", tmp_path / "fixtures")

    assert metadata["status"] == "ok"
    workbench = (tmp_path / "workbench.loom").read_text(encoding="utf-8")
    assert "tensor<2816xi8>, tensor<256xf32>, tensor<16xf32>" in workbench
    assert "check.expect.close" in workbench


def test_mul_mat_q8_0_oracle_and_workbench_use_packed_kernel_abi_buffers(tmp_path: Path) -> None:
    candidate = _candidate(
        candidate_id="mul_mat_q8_0_f32_contiguous_2d",
        shape={"k": 256, "rows": 16, "cols": 8},
        family="mul_mat_q8_0_f32",
        source_id="mul_mat_q8_0_f32",
        root_symbol="@hrx2_mul_mat_q8_0_f32_static",
        export_name="hrx2_mul_mat_q8_0_f32_static",
        op="MUL_MAT",
        source_path="kernels/v2/mul_mat/q8_0_f32_static.loom",
    )

    result = generate_oracle(candidate, tmp_path / "fixtures", force=True)

    assert result.status == "fixtures_ready"
    assert np.load(tmp_path / "fixtures" / "src0.npy").shape == (4352,)
    assert np.load(tmp_path / "fixtures" / "src1.npy").shape == (2048,)
    assert np.load(tmp_path / "fixtures" / "dst_init.npy").shape == (128,)
    assert np.load(tmp_path / "fixtures" / "expected.npy").shape == (128,)

    linked_source = tmp_path / "linked.loom"
    linked_source.write_text(
        'kernel.def export("hrx2_mul_mat_q8_0_f32_static") @hrx2_mul_mat_q8_0_f32_static() {}\n',
        encoding="utf-8",
    )
    _, metadata = write_workbench(candidate, linked_source, tmp_path / "workbench.loom", tmp_path / "fixtures")

    assert metadata["status"] == "ok"
    workbench = (tmp_path / "workbench.loom").read_text(encoding="utf-8")
    assert "tensor<4352xi8>, tensor<2048xf32>, tensor<128xf32>" in workbench
    assert "check.expect.close" in workbench


def test_rope_oracle_and_workbench_use_shape_rope_values(tmp_path: Path) -> None:
    candidate = Candidate(
        id="rope_f32_normal_n128_h32_t2_contiguous_4d",
        family="rope_f32",
        op="ROPE",
        source_id="rope_f32",
        source_path=Path("kernels/v2/rope/normal_f32.loom"),
        root_symbol="@hrx2_rope_normal_f32",
        export_name="hrx2_rope_normal_f32",
        route_id="rope_f32_normal_n128_h32_t2_contiguous_4d",
        route=None,
        shape={"d0": 128, "d1": 32, "d2": 2, "d3": 1},
        values={
            "shape.rope.ncols": 128,
            "shape.rope.n_dims": 128,
            "shape.rope.nheads": 32,
            "shape.rope.ntokens": 2,
            "shape.rope.src0_head_stride": 128,
            "shape.rope.src0_token_stride": 4096,
            "shape.rope.dst_head_stride": 128,
            "shape.rope.dst_token_stride": 4096,
            "shape.rope.pos_token_stride": 1,
        },
        config={},
        dispatch={},
        supports={},
        coverage="route_backed",
    )

    result = generate_oracle(candidate, tmp_path / "fixtures", force=True)

    assert result.status == "fixtures_ready"
    assert np.load(tmp_path / "fixtures" / "src0.npy").shape == (8192,)
    assert np.load(tmp_path / "fixtures" / "positions.npy").shape == (2,)
    assert np.load(tmp_path / "fixtures" / "expected.npy").shape == (8192,)

    linked_source = tmp_path / "linked.loom"
    linked_source.write_text('kernel.def export("hrx2_rope_normal_f32") @hrx2_rope_normal_f32() {}\n', encoding="utf-8")
    _, metadata = write_workbench(candidate, linked_source, tmp_path / "workbench.loom", tmp_path / "fixtures")

    assert metadata["status"] == "ok"
    workbench = (tmp_path / "workbench.loom").read_text(encoding="utf-8")
    assert "tensor<8192xf32>, tensor<2xi32>, tensor<8192xf32>" in workbench
    assert "check.expect.close" in workbench


def test_rope_neox_oracle_and_workbench_use_shape_rope_values(tmp_path: Path) -> None:
    candidate = Candidate(
        id="rope_neox_f32_n64_h128_t2_contiguous_4d",
        family="rope_neox_f32",
        op="ROPE",
        source_id="rope_neox_f32",
        source_path=Path("kernels/v2/rope/neox_f32.loom"),
        root_symbol="@hrx2_rope_neox_f32",
        export_name="hrx2_rope_neox_f32",
        route_id="rope_neox_f32_n64_h128_t2_contiguous_4d",
        route=None,
        shape={"d0": 64, "d1": 128, "d2": 2, "d3": 1},
        values={
            "shape.rope.ncols": 64,
            "shape.rope.n_dims": 64,
            "shape.rope.nheads": 128,
            "shape.rope.ntokens": 2,
            "shape.rope.src0_head_stride": 64,
            "shape.rope.src0_token_stride": 8192,
            "shape.rope.dst_head_stride": 64,
            "shape.rope.dst_token_stride": 8192,
            "shape.rope.pos_token_stride": 1,
        },
        config={},
        dispatch={},
        supports={},
        coverage="route_backed",
    )

    result = generate_oracle(candidate, tmp_path / "fixtures", force=True)

    assert result.status == "fixtures_ready"
    assert np.load(tmp_path / "fixtures" / "src0.npy").shape == (16384,)
    assert np.load(tmp_path / "fixtures" / "positions.npy").shape == (2,)
    assert np.load(tmp_path / "fixtures" / "expected.npy").shape == (16384,)

    linked_source = tmp_path / "linked.loom"
    linked_source.write_text('kernel.def export("hrx2_rope_neox_f32") @hrx2_rope_neox_f32() {}\n', encoding="utf-8")
    _, metadata = write_workbench(candidate, linked_source, tmp_path / "workbench.loom", tmp_path / "fixtures")

    assert metadata["status"] == "ok"
    workbench = (tmp_path / "workbench.loom").read_text(encoding="utf-8")
    assert "tensor<16384xf32>, tensor<2xi32>, tensor<16384xf32>" in workbench
    assert "check.expect.close" in workbench


def test_rope_set_rows_oracle_and_workbench_use_f16_dst_abi(tmp_path: Path) -> None:
    candidate = Candidate(
        id="rope_set_rows_f16_normal_n128_h32_t1_contiguous_4d",
        family="rope_set_rows_f32",
        op="ROPE_SET_ROWS",
        source_id="rope_set_rows_f32",
        source_path=Path("kernels/v2/rope_set_rows/f32.loom"),
        root_symbol="@hrx2_rope_normal_f32_freq_set_rows_f16",
        export_name="hrx2_rope_normal_f32_freq_set_rows_f16",
        route_id="rope_set_rows_f16_normal_n128_h32_t1_contiguous_4d",
        route=None,
        shape={
            "d0": 4096,
            "d1": 4,
            "d2": 1,
            "d3": 1,
            "src0_d0": 128,
            "src0_d1": 32,
            "pos_d0": 1,
            "pos_d1": 1,
            "freq_d0": 64,
            "freq_d1": 1,
            "src1_d0": 1,
            "src1_d1": 1,
        },
        values={
            "shape.rope.ncols": 128,
            "shape.rope.n_dims": 128,
            "shape.rope.nheads": 32,
            "shape.rope.ntokens": 1,
            "shape.rope.src0_head_stride": 128,
            "shape.rope.src0_token_stride": 4096,
            "shape.rope.pos_token_stride": 1,
            "shape.set_rows.ne1": 4,
            "shape.set_rows.ne11": 1,
            "shape.set_rows.ne12": 1,
        },
        config={},
        dispatch={},
        supports={},
        coverage="route_backed",
    )

    result = generate_oracle(candidate, tmp_path / "fixtures", force=True)

    assert result.status == "fixtures_ready"
    assert np.load(tmp_path / "fixtures" / "src0.npy").shape == (4096,)
    assert np.load(tmp_path / "fixtures" / "positions.npy").shape == (1,)
    assert np.load(tmp_path / "fixtures" / "freq.npy").shape == (64,)
    assert np.load(tmp_path / "fixtures" / "indices.npy").shape == (1,)
    assert np.load(tmp_path / "fixtures" / "dst_init.npy").shape == (16384,)
    assert np.load(tmp_path / "fixtures" / "expected.npy").shape == (16384,)
    assert np.load(tmp_path / "fixtures" / "dst_f32_init.npy").shape == (16384,)
    assert np.load(tmp_path / "fixtures" / "expected_f32.npy").shape == (16384,)

    linked_source = tmp_path / "linked.loom"
    linked_source.write_text(
        'kernel.def export("hrx2_rope_normal_f32_freq_set_rows_f16") @hrx2_rope_normal_f32_freq_set_rows_f16() {}\n',
        encoding="utf-8",
    )
    _, metadata = write_workbench(candidate, linked_source, tmp_path / "workbench.loom", tmp_path / "fixtures")

    assert metadata["status"] == "ok"
    workbench = (tmp_path / "workbench.loom").read_text(encoding="utf-8")
    assert "tensor<4096xf32>" in workbench
    assert "tensor<1xi64>" in workbench
    assert "tensor<16384xi16>" in workbench
    assert "tensor<16384xf32>" in workbench
    assert "check.expect.close" in workbench


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


def test_masked_soft_max_oracle_and_workbench_include_mask_buffer(tmp_path: Path) -> None:
    candidate = _candidate(
        candidate_id="soft_max_f32_mask_ranked_2d",
        shape={"d0": 16, "d1": 16},
        family="soft_max_f32",
        source_id="soft_max_f32",
        root_symbol="@hrx2_soft_max_f32_mask",
        export_name="hrx2_soft_max_f32_mask",
        op="SOFT_MAX",
        source_path="kernels/v2/soft_max/masked_contiguous_f32.loom",
    )

    result = generate_oracle(candidate, tmp_path / "fixtures", force=True)

    assert result.status == "fixtures_ready"
    assert np.load(tmp_path / "fixtures" / "src0.npy").shape == (256,)
    assert np.load(tmp_path / "fixtures" / "mask.npy").shape == (256,)
    assert np.load(tmp_path / "fixtures" / "expected.npy").shape == (256,)

    linked_source = tmp_path / "linked.loom"
    linked_source.write_text('kernel.def export("hrx2_soft_max_f32_mask") @hrx2_soft_max_f32_mask() {}\n', encoding="utf-8")
    _, metadata = write_workbench(candidate, linked_source, tmp_path / "workbench.loom", tmp_path / "fixtures")

    assert metadata["status"] == "ok"
    workbench = (tmp_path / "workbench.loom").read_text(encoding="utf-8")
    assert "tensor<256xf32>, tensor<256xf32>, tensor<256xf32>" in workbench
    assert "check.expect.close" in workbench


def test_softmax_kqv_oracle_and_workbench_use_exact_attention_abi(tmp_path: Path) -> None:
    candidate = _candidate(
        candidate_id="softmax_kqv_f32_f16_decode_kv512",
        shape=_softmax_kqv_shape(512),
        family="softmax_kqv_f32_f16",
        source_id="softmax_kqv_f32_f16",
        root_symbol="@hrx2_softmax_kqv_f32_f16_decode_kv512_rows128_wg256",
        export_name="hrx2_softmax_kqv_f32_f16_decode_kv512_rows128_wg256",
        op="FLASH_ATTN_EXT",
        source_path="kernels/v2/softmax_kqv/decode_kv512_rows128_wg256.loom",
    )

    result = generate_oracle(candidate, tmp_path / "fixtures", force=True)

    assert result.status == "fixtures_ready"
    assert np.load(tmp_path / "fixtures" / "kq.npy").shape == (12288,)
    assert np.load(tmp_path / "fixtures" / "mask.npy").shape == (512,)
    assert np.load(tmp_path / "fixtures" / "v.npy").shape == (524288,)
    assert np.load(tmp_path / "fixtures" / "v.npy").dtype == np.float16
    assert np.load(tmp_path / "fixtures" / "dst_init.npy").shape == (3072,)
    assert np.load(tmp_path / "fixtures" / "expected.npy").shape == (3072,)

    linked_source = tmp_path / "linked.loom"
    linked_source.write_text(
        'kernel.def export("hrx2_softmax_kqv_f32_f16_decode_kv512_rows128_wg256") @hrx2_softmax_kqv_f32_f16_decode_kv512_rows128_wg256() {}\n',
        encoding="utf-8",
    )
    _, metadata = write_workbench(candidate, linked_source, tmp_path / "workbench.loom", tmp_path / "fixtures")

    assert metadata["status"] == "ok"
    workbench = (tmp_path / "workbench.loom").read_text(encoding="utf-8")
    assert "%scale = check.literal value(0.75) : f32" in workbench
    assert "tensor<12288xf32>" in workbench
    assert "tensor<512xf32>" in workbench
    assert "tensor<524288xf16>" in workbench
    assert "tensor<3072xf32>" in workbench
    assert "check.expect.close" in workbench


def test_softmax_kqv_oracle_supports_masked_identity_h8_attention_abi(tmp_path: Path) -> None:
    candidate = _candidate(
        candidate_id="softmax_kqv_f32_f16_masked_identity_kv1024_h8",
        shape=_softmax_kqv_shape(1024, cols=8),
        family="softmax_kqv_f32_f16",
        source_id="softmax_kqv_f32_f16",
        root_symbol="@hrx2_softmax_kqv_f32_f16_masked_identity_kv512_4096_d128_h8_wg256_row1",
        export_name="hrx2_softmax_kqv_f32_f16_masked_identity_kv512_4096_d128_h8_wg256_row1",
        op="FLASH_ATTN_EXT",
        source_path="kernels/v2/softmax_kqv/masked_identity_kv512_4096_d128_h8_wg256_row1.loom",
    )

    result = generate_oracle(candidate, tmp_path / "fixtures", force=True)

    assert result.status == "fixtures_ready"
    assert np.load(tmp_path / "fixtures" / "kq.npy").shape == (8192,)
    assert np.load(tmp_path / "fixtures" / "mask.npy").shape == (1024,)
    assert np.load(tmp_path / "fixtures" / "v.npy").shape == (1048576,)
    assert np.load(tmp_path / "fixtures" / "v.npy").dtype == np.float16
    assert np.load(tmp_path / "fixtures" / "dst_init.npy").shape == (1024,)
    assert np.load(tmp_path / "fixtures" / "expected.npy").shape == (1024,)


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
