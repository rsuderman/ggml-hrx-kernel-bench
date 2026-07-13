from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

import bootstrap  # noqa: F401

from ggml_hrx_kernel_bench.required_tools import require_tool


class RequiredToolAvailabilityTest(unittest.TestCase):
    def test_loom_link_available(self) -> None:
        self.assertTrue(require_tool("loom-link"))

    def test_loom_compile_available(self) -> None:
        self.assertTrue(require_tool("loom-compile"))

    def test_iree_test_loom_available(self) -> None:
        self.assertTrue(require_tool("iree-test-loom"))

    def test_iree_run_loom_available(self) -> None:
        self.assertTrue(require_tool("iree-run-loom"))

    def test_iree_benchmark_loom_available(self) -> None:
        self.assertTrue(require_tool("iree-benchmark-loom"))

    def test_env_tool_dir_is_used(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tool_path = os.path.join(tmpdir, "loom-link")
            with open(tool_path, "w", encoding="utf-8") as handle:
                handle.write("#!/bin/sh\n")
            with mock.patch.dict(os.environ, {"GGML_HRX_TOOL_DIR": tmpdir}, clear=False):
                self.assertEqual(require_tool("loom-link"), tool_path)

    def test_env_tool_dir_path_list_is_used_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as first_tmpdir, tempfile.TemporaryDirectory() as second_tmpdir:
            tool_path = os.path.join(second_tmpdir, "iree-test-loom")
            with open(tool_path, "w", encoding="utf-8") as handle:
                handle.write("#!/bin/sh\n")
            with mock.patch.dict(
                os.environ,
                {"GGML_HRX_TOOL_DIR": os.pathsep.join((first_tmpdir, second_tmpdir))},
                clear=False,
            ):
                self.assertEqual(require_tool("iree-test-loom"), tool_path)

    def test_configured_tool_dir_does_not_fall_back_to_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                mock.patch.dict(os.environ, {"GGML_HRX_TOOL_DIR": tmpdir}, clear=False),
                mock.patch("shutil.which", return_value="/usr/bin/loom-link"),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    f"required tool is not available in {tmpdir}: loom-link",
                ):
                    require_tool("loom-link")


if __name__ == "__main__":
    unittest.main()
