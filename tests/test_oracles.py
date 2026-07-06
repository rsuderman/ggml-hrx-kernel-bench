from __future__ import annotations

from pathlib import Path

import numpy as np

from ggml_hrx_kernel_bench.oracles import generate_oracle, write_workbench
from ggml_hrx_kernel_bench.routing.api import Candidate


def _candidate(*, candidate_id: str, shape: dict[str, int]) -> Candidate:
    return Candidate(
        id=candidate_id,
        family="add_f32",
        op="ADD",
        source_id="add_f32",
        source_path=Path("kernels/v2/add/contiguous_1d.loom"),
        root_symbol="@hrx2_add_f32",
        export_name="hrx2_add_f32",
        route_id="add_f32_test",
        route=None,
        shape=shape,
        values={},
        config={},
        dispatch={},
        supports={},
        coverage="route_backed",
    )


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
