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


def test_generate_loom_descriptor_tests_cmake_registers_build_prepare_without_prepare_test(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module(
        "tests/infra/generate_loom_descriptor_tests_cmake.py",
        "test_generate_loom_descriptor_tests_cmake",
    )
    grouped_yaml_path = tmp_path / "suite.yaml"
    grouped_yaml_path.write_text("ops:\n  ADD: []\n", encoding="utf-8")
    output_path = tmp_path / "generated-descriptor-tests.cmake"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_loom_descriptor_tests_cmake.py",
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
            "--descriptor-generator-script",
            str(ROOT / "tests" / "infra" / "generate_loom_execution_descriptors.py"),
            "--descriptor-runner-script",
            str(ROOT / "tests" / "infra" / "run_loom_execution_descriptors.py"),
            "--descriptor-output-dir",
            str(tmp_path / "descriptors"),
            "--prepare-output-dir",
            str(tmp_path / "prepare"),
            "--kernel-dir",
            str(ROOT / "kernels" / "v2"),
            "--routing-dir",
            str(ROOT / "catalog" / "v2"),
            "--runner",
            "$<TARGET_FILE:ggml-hrx-run-loom-simple>",
            "--tool-dir",
            str(tmp_path / "tools"),
            "--repo-root",
            str(ROOT),
            "--build-prepare-target",
            "kernel-descriptor-prepare-llama-cpp-tests-v2-generated",
            "--import-target",
            "kernel-llama-cpp-tests-v2",
            "--all-ops",
        ],
    )

    assert module.main() == 0
    generated = output_path.read_text(encoding="utf-8")
    assert "kernel-descriptor-generate-llama-cpp-tests-v2-ADD-generated" in generated
    assert "kernel-descriptor-execute-llama-cpp-tests-v2-ADD-generated" not in generated
    assert "--execute" not in generated
    assert "FIXTURES_SETUP descriptor-llama-cpp-tests-v2-ADD" in generated
    assert "  NAME kernel-descriptor-prepare-llama-cpp-tests-v2-ADD-generated" not in generated
    assert "FIXTURES_REQUIRED descriptor-llama-cpp-tests-v2-ADD" not in generated
    assert "$<TARGET_FILE:ggml-hrx-run-loom-simple>" in generated
    assert "add_custom_target(kernel-descriptor-prepare-llama-cpp-tests-v2-generated ALL" in generated
    assert "add_dependencies(kernel-descriptor-prepare-llama-cpp-tests-v2-generated kernel-llama-cpp-tests-v2)" in generated
    assert "add_dependencies(kernel-descriptor-prepare-llama-cpp-tests-v2-generated ggml-hrx-run-loom-simple)" in generated
    assert "--quiet" in generated
    assert "ENVIRONMENT_MODIFICATION" not in generated

    manifest_path = (
        tmp_path
        / "generated-import-dir"
        / "ops"
        / "ADD"
        / "generated-kernel-tests.json"
    ).resolve()
    import_stamp_path = (tmp_path / "generated-import-dir" / "route-import.stamp").resolve()
    custom_command = generated.split("add_custom_command(\n", maxsplit=1)[1].split(
        "\n)\n", maxsplit=1
    )[0]
    commands, dependencies = custom_command.split("  DEPENDS\n", maxsplit=1)
    generate_command = commands.split("  COMMAND\n", maxsplit=1)[1].split(
        "  COMMAND\n", maxsplit=1
    )[0]
    prepare_command = commands.split("  COMMAND\n", maxsplit=2)[2]
    dependencies = dependencies.split("  VERBATIM", maxsplit=1)[0]
    descriptor_generator_script = (
        ROOT / "tests" / "infra" / "generate_loom_execution_descriptors.py"
    ).resolve()
    descriptor_runner_script = (
        ROOT / "tests" / "infra" / "run_loom_execution_descriptors.py"
    ).resolve()
    assert f'"{sys.executable}" "{descriptor_generator_script}"' in generate_command
    assert f'"{sys.executable}" "{descriptor_runner_script}"' in prepare_command
    assert f'"{manifest_path}"' in generate_command
    assert f'"{import_stamp_path}"' in dependencies
    assert f'"{descriptor_generator_script}"' in dependencies
    assert f'"{descriptor_runner_script}"' in dependencies
    assert f'"{manifest_path}"' not in dependencies


def test_generate_loom_descriptor_tests_cmake_registers_hsa_execution_when_enabled(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module(
        "tests/infra/generate_loom_descriptor_tests_cmake.py",
        "test_generate_loom_descriptor_tests_cmake_hsa",
    )
    grouped_yaml_path = tmp_path / "suite.yaml"
    grouped_yaml_path.write_text("ops:\n  ADD: []\n", encoding="utf-8")
    output_path = tmp_path / "generated-descriptor-tests.cmake"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_loom_descriptor_tests_cmake.py",
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
            "--descriptor-generator-script",
            str(ROOT / "tests" / "infra" / "generate_loom_execution_descriptors.py"),
            "--descriptor-runner-script",
            str(ROOT / "tests" / "infra" / "run_loom_execution_descriptors.py"),
            "--descriptor-output-dir",
            str(tmp_path / "descriptors"),
            "--prepare-output-dir",
            str(tmp_path / "prepare"),
            "--execute-output-dir",
            str(tmp_path / "execute"),
            "--runner",
            "$<TARGET_FILE:ggml-hrx-run-loom-simple>",
            "--repo-root",
            str(ROOT),
            "--all-ops",
            "--execute-hsa",
        ],
    )

    assert module.main() == 0
    generated = output_path.read_text(encoding="utf-8")
    assert "kernel-descriptor-execute-llama-cpp-tests-v2-ADD-generated" in generated
    assert "--execute" in generated
    assert "--progress" in generated
    assert "--quiet" in generated
    assert 'LABELS "hsa;runtime;loom-descriptor"' in generated
    assert "ENVIRONMENT_MODIFICATION" not in generated
