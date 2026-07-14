from __future__ import annotations

import json
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
    operations: tuple[str, ...],
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(GENERATOR),
        "--routing-dir",
        str(routing_dir),
        "--output",
        str(output),
    ]
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


def test_generator_rejects_partially_supported_operation_without_replacing_output(
    materialized_routing_dir: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "routes.inc.cpp"
    output.write_text("keep-existing-output\n", encoding="utf-8")

    result = _run_generator(
        routing_dir=materialized_routing_dir,
        output=output,
        operations=("CPY",),
    )

    assert result.returncode != 0
    assert "copy_bf16_bf16_non_contiguous_4d" in result.stderr
    assert "unsupported permutation capture" in result.stderr
    assert output.read_text(encoding="utf-8") == "keep-existing-output\n"


def test_generator_rejects_fully_unsupported_operation_before_creating_output(
    materialized_routing_dir: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "uncreated" / "routes.inc.cpp"

    result = _run_generator(
        routing_dir=materialized_routing_dir,
        output=output,
        operations=("ARGSORT",),
    )

    assert result.returncode != 0
    assert "argsort_f32_i32_n128_r1_desc_wg128" in result.stderr
    assert "unsupported" in result.stderr or "no fixed dtype" in result.stderr
    assert not output.parent.exists()
