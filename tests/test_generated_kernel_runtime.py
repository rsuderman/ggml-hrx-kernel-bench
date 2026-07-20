from __future__ import annotations

import importlib.util
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
    assert "SKIP_RETURN_CODE 125" in generated
    assert "ENVIRONMENT_MODIFICATION" not in generated
    assert "if(CMAKE_VERSION" not in generated

