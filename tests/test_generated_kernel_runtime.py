from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INFRA_DIR = ROOT / "tests" / "infra"


def _load_script_module(relative_path: str, module_name: str):
    if str(INFRA_DIR) not in sys.path:
        sys.path.insert(0, str(INFRA_DIR))
    spec = importlib.util.spec_from_file_location(module_name, ROOT / relative_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generate_kernel_runtime_tests_cmake_omits_case_selector(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module(
        "tests/infra/generate_kernel_runtime_tests_cmake.py",
        "test_generate_kernel_runtime_tests_cmake",
    )
    grouped_yaml_path = tmp_path / "suite.yaml"
    grouped_yaml_path.write_text("ops:\n  ADD: {}\n", encoding="utf-8")
    output_path = tmp_path / "generated-tests.cmake"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_kernel_runtime_tests_cmake.py",
            "--output",
            str(output_path),
            "--name",
            "llama-cpp-tests-v2",
            "--grouped-yaml",
            str(grouped_yaml_path),
            "--generated-import-dir",
            str(tmp_path / "generated-import-dir"),
            "--python-executable",
            sys.executable,
            "--python-module-dir",
            str(tmp_path / "python"),
            "--runner-script",
            str(ROOT / "tests" / "infra" / "run_generated_kernel_tests.py"),
            "--runtime-output-dir",
            str(tmp_path / "runtime-output"),
        ],
    )

    assert module.main() == 0
    generated = output_path.read_text(encoding="utf-8")
    assert "--case-selector" not in generated
    assert "kernel-run-llama-cpp-tests-v2-ADD-generated" in generated
    assert (
        f'ENVIRONMENT_MODIFICATION "PYTHONPATH=path_list_prepend:{tmp_path / "python"}"'
        in generated
    )
    assert "if(CMAKE_VERSION" not in generated
    assert f'"PYTHONPATH={tmp_path / "python"}"' not in generated


def test_run_generated_kernel_tests_executes_all_cases(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module(
        "tests/infra/run_generated_kernel_tests.py",
        "test_run_generated_kernel_tests",
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "kernel": "copy_f32_f32",
                "params": ["n"],
                "cases": [[1], [2]],
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
                "source_path": "test.yaml",
                "entry_count": 1,
                "entries": [
                    {
                        "config_path": str(config_path),
                        "config_name": config_path.name,
                        "kernel": "copy_f32_f32",
                        "case_count": 2,
                        "route_id": "copy_f32_f32_contiguous_1d",
                        "op": "CPY",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    calls: list[tuple[str, list[int], Path]] = []

    class _FakeContext:
        kernel_dir = None
        routing_dir = None

    class _FakeRouter:
        context = _FakeContext()

        def select_cases(self, config: dict[str, object], selectors):
            assert selectors is None
            assert config["cases"] == [[1], [2]]
            return [("n1", [1]), ("n2", [2])]

        def execute_case(self, request) -> dict[str, object]:
            calls.append(
                (
                    request.current_case_id,
                    list(request.current_case_values),
                    Path(request.output_dir),
                )
            )
            return {
                "case_id": request.current_case_id,
                "values": list(request.current_case_values),
                "status": "ran",
                "correctness_ok": True,
                "results_path": str(Path(request.output_dir) / "results.json"),
            }

        def case_result(self, execution: dict[str, object]) -> dict[str, object]:
            return dict(execution)

    monkeypatch.setattr(module, "create_router", lambda **_: _FakeRouter())
    monkeypatch.setattr(module, "load_config", lambda _: json.loads(config_path.read_text()))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_generated_kernel_tests.py",
            str(manifest_path),
            "--output-dir",
            str(tmp_path / "runtime-output"),
            "--routing-version",
            "v2",
        ],
    )

    assert module.main() == 0
    assert [(case_id, values) for case_id, values, _ in calls] == [("n1", [1]), ("n2", [2])]
    assert calls[0][2] != calls[1][2]
    assert calls[0][2].name.endswith("n1")
    assert calls[1][2].name.endswith("n2")


def test_run_generated_kernel_tests_truncates_long_case_directory_names(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module(
        "tests/infra/run_generated_kernel_tests.py",
        "test_run_generated_kernel_tests_long_names",
    )
    config_path = tmp_path / ("config-" + ("x" * 180) + ".json")
    config_path.write_text(
        json.dumps(
            {
                "kernel": "rope_set_rows_f32",
                "params": ["n"],
                "cases": [[1]],
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
                "source_path": "test.yaml",
                "entry_count": 1,
                "entries": [
                    {
                        "config_path": str(config_path),
                        "config_name": config_path.name,
                        "kernel": "rope_set_rows_f32",
                        "case_count": 1,
                        "route_id": "rope_set_rows_f16_normal_n128_h32_t1_512_contiguous_4d",
                        "op": "ROPE_SET_ROWS",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    calls: list[Path] = []

    class _FakeContext:
        kernel_dir = None
        routing_dir = None

    class _FakeRouter:
        context = _FakeContext()

        def select_cases(self, config: dict[str, object], selectors):
            assert selectors is None
            assert config["cases"] == [[1]]
            return [("case-" + ("y" * 240), [1])]

        def execute_case(self, request) -> dict[str, object]:
            output_dir = Path(request.output_dir)
            calls.append(output_dir)
            return {
                "case_id": request.current_case_id,
                "values": list(request.current_case_values),
                "status": "ran",
                "correctness_ok": True,
                "results_path": str(output_dir / "results.json"),
            }

        def case_result(self, execution: dict[str, object]) -> dict[str, object]:
            return dict(execution)

    monkeypatch.setattr(module, "create_router", lambda **_: _FakeRouter())
    monkeypatch.setattr(module, "load_config", lambda _: json.loads(config_path.read_text()))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_generated_kernel_tests.py",
            str(manifest_path),
            "--output-dir",
            str(tmp_path / "runtime-output"),
            "--routing-version",
            "v2",
        ],
    )

    assert module.main() == 0
    assert len(calls) == 1
    assert len(calls[0].parent.name) <= 100
    assert len(calls[0].name) <= 100
