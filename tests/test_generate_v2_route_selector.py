from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from ggml_hrx_kernel_bench.materialized_assets import materialize_asset_root


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATOR = PROJECT_ROOT / "scripts" / "generate_v2_route_selector.py"


@pytest.fixture(scope="module")
def materialized_routing_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    asset_root = materialize_asset_root(
        tmp_path_factory.mktemp("v2-route-selector") / "assets",
        force=True,
    )
    return asset_root / "catalog" / "v2"


def _run_generator(
    *,
    routing_dir: Path,
    output: Path,
    operations: tuple[str, ...] = (),
    all_operations: bool = False,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(GENERATOR),
        "--routing-dir",
        str(routing_dir),
        "--output",
        str(output),
    ]
    if all_operations:
        command.append("--all")
    for operation in operations:
        command.extend(("--op", operation))
    return subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def _route_ids(routing_dir: Path, operation: str) -> list[str]:
    router = json.loads((routing_dir / "router.json").read_text(encoding="utf-8"))
    return [
        str(json.loads((routing_dir / relative_path).read_text(encoding="utf-8"))["id"])
        for relative_path in router["routes"][operation]
    ]


def _generated_route_ids(contents: str) -> dict[str, list[str]]:
    generated: dict[str, list[str]] = {}
    operation: str | None = None
    operation_prefix = "    // Routes for operation "
    route_prefix = "/* .id = */ "
    for line in contents.splitlines():
        if line.startswith(operation_prefix):
            operation = str(json.loads(line.removeprefix(operation_prefix).removesuffix(".")))
            generated[operation] = []
            continue
        if route_prefix not in line:
            continue
        assert operation is not None
        encoded_route_id = line.split(route_prefix, maxsplit=1)[1].removesuffix(",")
        generated[operation].append(str(json.loads(encoded_route_id)))
    return generated


def _copy_routing_dir(source: Path, destination: Path) -> Path:
    return Path(shutil.copytree(source, destination))


def test_generator_preserves_every_materialized_operation_and_route_order(
    materialized_routing_dir: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "generated" / "ggml_hrx_v2_routes.inc.cpp"

    result = _run_generator(
        routing_dir=materialized_routing_dir,
        output=output,
        all_operations=True,
    )

    assert result.returncode == 0, result.stderr
    router = json.loads(
        (materialized_routing_dir / "router.json").read_text(encoding="utf-8")
    )
    expected = {
        operation: _route_ids(materialized_routing_dir, operation)
        for operation in sorted(router["routes"])
    }
    generated = _generated_route_ids(output.read_text(encoding="utf-8"))
    assert len(expected) == 52
    assert sum(len(route_ids) for route_ids in expected.values()) == 214
    assert list(generated) == sorted(expected)
    assert generated == expected


def test_generator_preserves_exact_materialized_abs_route_order(
    materialized_routing_dir: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "generated" / "ggml_hrx_v2_routes.inc.cpp"

    result = _run_generator(
        routing_dir=materialized_routing_dir,
        output=output,
        operations=("ABS",),
    )

    assert result.returncode == 0, result.stderr
    expected_route_ids = _route_ids(materialized_routing_dir, "ABS")
    assert expected_route_ids == [
        "abs_f16_contiguous_4d",
        "abs_f32_contiguous_4d",
        "abs_f16_non_contiguous_4d",
        "abs_f32_non_contiguous_4d",
    ]
    contents = output.read_text(encoding="utf-8")
    positions = [
        contents.index(f'            /* .id = */ "{route_id}",') for route_id in expected_route_ids
    ]
    assert positions == sorted(positions)
    assert contents.count('    {"ABS",') == 1
    assert contents.count("/* .tensors = */") == 4
    assert contents.count("/* .values = */") == 4
    assert contents.count("/* .constraints = */") == 4


def test_generator_output_is_deterministic_across_requested_op_order(
    materialized_routing_dir: Path,
    tmp_path: Path,
) -> None:
    first_output = tmp_path / "first.inc.cpp"
    second_output = tmp_path / "second.inc.cpp"

    first = _run_generator(
        routing_dir=materialized_routing_dir,
        output=first_output,
        operations=("EXP", "ABS"),
    )
    second = _run_generator(
        routing_dir=materialized_routing_dir,
        output=second_output,
        operations=(" abs ", "exp"),
    )

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert first_output.read_bytes() == second_output.read_bytes()
    contents = first_output.read_text(encoding="utf-8")
    assert contents.index('    {"ABS",') < contents.index('    {"EXP",')


def test_generator_rejects_missing_operation_without_replacing_output(
    materialized_routing_dir: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "routes.inc.cpp"
    output.write_text("keep-existing-output\n", encoding="utf-8")

    result = _run_generator(
        routing_dir=materialized_routing_dir,
        output=output,
        operations=("DOES_NOT_EXIST",),
    )

    assert result.returncode != 0
    assert "missing" in result.stderr
    assert output.read_text(encoding="utf-8") == "keep-existing-output\n"


def test_generator_rejects_all_combined_with_op(
    materialized_routing_dir: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "routes.inc.cpp"

    result = _run_generator(
        routing_dir=materialized_routing_dir,
        output=output,
        operations=("ABS",),
        all_operations=True,
    )

    assert result.returncode == 2
    assert "not allowed with argument" in result.stderr
    assert not output.exists()


def test_generator_rejects_duplicate_requested_operations_without_replacing_output(
    materialized_routing_dir: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "routes.inc.cpp"
    output.write_text("keep-existing-output\n", encoding="utf-8")

    result = _run_generator(
        routing_dir=materialized_routing_dir,
        output=output,
        operations=("ABS", " abs "),
    )

    assert result.returncode != 0
    assert "duplicate requested operations" in result.stderr
    assert output.read_text(encoding="utf-8") == "keep-existing-output\n"


def test_generator_renders_permutations_and_multi_source_values(
    materialized_routing_dir: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "cpy.inc.cpp"

    result = _run_generator(
        routing_dir=materialized_routing_dir,
        output=output,
        operations=("CPY",),
    )

    assert result.returncode == 0, result.stderr
    contents = output.read_text(encoding="utf-8")
    assert _generated_route_ids(contents) == {
        "CPY": _route_ids(materialized_routing_dir, "CPY")
    }
    assert '"src0_permutation"' in contents
    assert '"dst_permutation"' in contents
    assert "ValueKind::inverse_permutation" in contents
    assert "ValueKind::chain_permutations" in contents
    assert 'ValueKind::chain_permutations, {"src0_permutation", "dst_permutation_inverse"}' in contents
    assert "ValueKind::permuted_contiguous_strides" in contents
    assert 'ValueKind::permuted_contiguous_strides, {"dst_dimensions", "effective_src0_permutation"}' in contents


def test_generator_renders_wildcard_dtype(
    materialized_routing_dir: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "argsort.inc.cpp"

    result = _run_generator(
        routing_dir=materialized_routing_dir,
        output=output,
        operations=("ARGSORT",),
    )

    assert result.returncode == 0, result.stderr
    contents = output.read_text(encoding="utf-8")
    assert _generated_route_ids(contents) == {
        "ARGSORT": ["argsort_f32_i32_n128_r1_desc_wg128"]
    }
    assert (
        '{"dst", std::nullopt, "dst_dimensions", "dst_strides", std::nullopt}'
        in contents
    )


def test_generator_renders_every_materialized_value_and_constraint_form(
    materialized_routing_dir: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "all.inc.cpp"

    result = _run_generator(
        routing_dir=materialized_routing_dir,
        output=output,
        all_operations=True,
    )

    assert result.returncode == 0, result.stderr
    contents = output.read_text(encoding="utf-8")
    for value_kind in (
        "chain_permutations",
        "contiguous_strides",
        "element",
        "head",
        "inverse_permutation",
        "permuted_contiguous_strides",
        "product",
        "tail",
    ):
        assert f"ValueKind::{value_kind}" in contents
    for helper in (
        "divides(",
        "equals(",
        "exact_length(",
        "indexed_bounds(",
        "rank_range(",
        "scalar_bounds(",
    ):
        assert helper in contents
    assert "indexed_bounds(\"dst_dimensions\", 0" in contents
    assert ", 512)" in contents


def test_generator_omits_attribute_constraints_from_native_tensor_selector(
    materialized_routing_dir: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "flash_attn_ext.inc.cpp"

    result = _run_generator(
        routing_dir=materialized_routing_dir,
        output=output,
        operations=("FLASH_ATTN_EXT",),
    )

    assert result.returncode == 0, result.stderr
    contents = output.read_text(encoding="utf-8")
    assert '"attribute.precision"' not in contents
    assert '"attribute.sinks"' not in contents
    assert '"attribute.logit_softcap_enabled"' not in contents
    assert '"attribute.max_bias_enabled"' not in contents


def test_generator_renders_schema_supported_iota_constraint(
    materialized_routing_dir: Path,
    tmp_path: Path,
) -> None:
    routing_dir = _copy_routing_dir(
        materialized_routing_dir,
        tmp_path / "routing",
    )
    router = json.loads((routing_dir / "router.json").read_text(encoding="utf-8"))
    route_path = routing_dir / router["routes"]["ABS"][0]
    route = json.loads(route_path.read_text(encoding="utf-8"))
    route["constraints"].append({"name": "dst_dimensions", "iota": True})
    route_path.write_text(json.dumps(route, indent=2) + "\n", encoding="utf-8")
    output = tmp_path / "abs.inc.cpp"

    result = _run_generator(
        routing_dir=routing_dir,
        output=output,
        operations=("ABS",),
    )

    assert result.returncode == 0, result.stderr
    assert 'iota("dst_dimensions")' in output.read_text(encoding="utf-8")


def test_generator_rejects_duplicate_route_ids_across_full_table(
    materialized_routing_dir: Path,
    tmp_path: Path,
) -> None:
    routing_dir = _copy_routing_dir(
        materialized_routing_dir,
        tmp_path / "routing",
    )
    router_path = routing_dir / "router.json"
    router = json.loads(router_path.read_text(encoding="utf-8"))
    duplicate_route = json.loads((routing_dir / router["routes"]["ABS"][0]).read_text(encoding="utf-8"))
    duplicate_route["op"] = "ARGSORT"
    duplicate_route_path = routing_dir / "argsort" / "duplicate_abs_f16.json"
    duplicate_route_path.write_text(json.dumps(duplicate_route, indent=2) + "\n", encoding="utf-8")
    router["routes"]["ARGSORT"] = ["argsort/duplicate_abs_f16.json"]
    router_path.write_text(json.dumps(router, indent=2) + "\n", encoding="utf-8")
    output = tmp_path / "routes.inc.cpp"
    output.write_text("keep-existing-output\n", encoding="utf-8")

    result = _run_generator(
        routing_dir=routing_dir,
        output=output,
        all_operations=True,
    )

    assert result.returncode != 0
    assert "duplicate route id 'abs_f16_contiguous_4d'" in result.stderr
    assert output.read_text(encoding="utf-8") == "keep-existing-output\n"


def test_generator_malformed_late_descriptor_does_not_replace_output(
    materialized_routing_dir: Path,
    tmp_path: Path,
) -> None:
    routing_dir = _copy_routing_dir(
        materialized_routing_dir,
        tmp_path / "routing",
    )
    router = json.loads((routing_dir / "router.json").read_text(encoding="utf-8"))
    route_path = routing_dir / router["routes"]["XIELU"][-1]
    route = json.loads(route_path.read_text(encoding="utf-8"))
    route["tensors"]["dst"]["dimensions"] = None
    route_path.write_text(json.dumps(route, indent=2) + "\n", encoding="utf-8")
    output = tmp_path / "routes.inc.cpp"
    output.write_text("keep-existing-output\n", encoding="utf-8")

    result = _run_generator(
        routing_dir=routing_dir,
        output=output,
        all_operations=True,
    )

    assert result.returncode != 0
    assert "dimensions" in result.stderr
    assert "XIELU" in result.stderr
    assert output.read_text(encoding="utf-8") == "keep-existing-output\n"
