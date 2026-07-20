from __future__ import annotations

import json
from pathlib import Path

from ggml_hrx_kernel_bench.route_optimization import route_inventory_payload


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_route_inventory_reports_flash_attention_selection(tmp_path: Path) -> None:
    route_id = "flash_attn_ext_f32_f16_tiled_decode_d128_h32_hkv8_kv512_8192_t1"
    import_dir = tmp_path / "import"
    op_dir = import_dir / "ops" / "FLASH_ATTN_EXT"
    routing_dir = tmp_path / "catalog" / "v2"
    kernel_dir = tmp_path / "kernels" / "v2"

    _write_json(
        op_dir / "route-matches.json",
        {
            "schema": "ggml_hrx_kernel_bench.yaml_route_import.v1",
            "rows": [
                {
                    "case_index": 0,
                    "source_id": "model.yaml:FLASH_ATTN_EXT[0]",
                    "source_path": "model.yaml",
                    "status": "matched",
                    "matched_route_ids": [route_id],
                    "candidate_matched_route_ids": [route_id],
                    "case": {
                        "op": "FLASH_ATTN_EXT",
                        "attributes": {"precision": 10},
                    },
                }
            ],
        },
    )
    _write_json(
        op_dir / "route-unmatched.json",
        {"schema": "ggml_hrx_kernel_bench.yaml_route_import.v1", "rows": []},
    )
    _write_json(
        op_dir / "route-import-summary.json",
        {
            "schema": "ggml_hrx_kernel_bench.yaml_route_import.v1",
            "op": "FLASH_ATTN_EXT",
            "case_count": 1,
            "matched_case_count": 1,
        },
    )
    _write_json(
        routing_dir / "router.json",
        {
            "schema": "ggml_hrx_kernel_bench.routing_descriptors.v2",
            "routes": {"FLASH_ATTN_EXT": ["flash_attn_ext/decode.json"]},
        },
    )
    _write_json(
        routing_dir / "flash_attn_ext" / "decode.json",
        {
            "id": route_id,
            "family": "flash_attn_ext_f32_f16",
            "op": "FLASH_ATTN_EXT",
            "attributes": {"precision": {"type": "i32"}},
            "kernel": {
                "source_id": "flash_attn_ext_f32_f16",
                "path": "flash_attn_ext/decode.loom",
                "root_symbol": "@decode",
                "export_name": "decode",
            },
            "tensors": {
                "src0": {
                    "dtype": "F32",
                    "dimensions": "src0_dimensions",
                    "strides": "src0_strides",
                },
                "dst": {
                    "dtype": "F32",
                    "dimensions": "dst_dimensions",
                    "strides": "dst_strides",
                },
            },
            "constraints": [{"name": "dst_dimensions", "length": 4}],
            "launch": {"workgroup_size": [256, 1, 1]},
            "config": {
                "bindings": [
                    {"key": "@shape.flash.head_dim", "source": "tensor.dst.dimensions.d0.size"}
                ]
            },
        },
    )
    kernel_path = kernel_dir / "flash_attn_ext" / "decode.loom"
    kernel_path.parent.mkdir(parents=True, exist_ok=True)
    kernel_path.write_text("kernel.def @decode\n", encoding="utf-8")

    payload = route_inventory_payload(
        op="FLASH_ATTN_EXT",
        generated_import_dir=import_dir,
        routing_dir=routing_dir,
        kernel_dir=kernel_dir,
        target="gfx1100",
        repo_root=tmp_path,
    )

    assert payload["schema"] == "ggml_hrx_kernel_bench.route_inventory.v1"
    assert payload["case_count"] == 1
    assert payload["selected_route_ids"] == [route_id]
    assert payload["cases"][0]["selected_route_id"] == route_id
    assert payload["cases"][0]["routes"][0]["source_exists"] is True


def test_route_inventory_rejects_non_flash_attention(tmp_path: Path) -> None:
    try:
        route_inventory_payload(
            op="MUL_MAT",
            generated_import_dir=tmp_path,
            routing_dir=tmp_path,
            kernel_dir=tmp_path,
            target="gfx1100",
            repo_root=tmp_path,
        )
    except ValueError as exc:
        assert "FLASH_ATTN_EXT" in str(exc)
    else:
        raise AssertionError("expected route inventory to reject non-flash ops")
