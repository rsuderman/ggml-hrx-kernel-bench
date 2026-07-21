from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from unittest import mock

import bootstrap  # noqa: F401

from ggml_hrx_kernel_bench.required_tools import (
    require_ggml_hrx_run_loom_expected_buffer_tolerance,
    require_tool,
)


class RequiredToolAvailabilityTest(unittest.TestCase):
    def test_loom_link_available(self) -> None:
        self.assertTrue(require_tool("loom-link"))

    def test_loom_compile_available(self) -> None:
        self.assertTrue(require_tool("loom-compile"))

    def test_iree_test_loom_available(self) -> None:
        self.assertTrue(require_tool("iree-test-loom"))

    def test_ggml_hrx_run_loom_available(self) -> None:
        self.assertTrue(require_tool("ggml-hrx-run-loom"))

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

    def test_ggml_hrx_run_loom_tolerance_capability_accepts_supported_tool(self) -> None:
        completed = subprocess.CompletedProcess(
            args=["/tmp/ggml-hrx-run-loom", "--help"],
            returncode=0,
            stdout="usage\n  --expected-kernel-buffer-tolerance=atol,rtol\n",
        )
        with mock.patch("subprocess.run", return_value=completed) as run:
            self.assertEqual(
                require_ggml_hrx_run_loom_expected_buffer_tolerance(tool_path="/tmp/ggml-hrx-run-loom"),
                "/tmp/ggml-hrx-run-loom",
            )
        run.assert_called_once_with(
            ["/tmp/ggml-hrx-run-loom", "--help"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=10.0,
        )

    def test_ggml_hrx_run_loom_tolerance_capability_rejects_stale_tool(self) -> None:
        completed = subprocess.CompletedProcess(
            args=["/tmp/ggml-hrx-run-loom", "--help"],
            returncode=0,
            stdout="usage\n",
        )
        with mock.patch("subprocess.run", return_value=completed):
            with self.assertRaisesRegex(
                RuntimeError,
                "does not support --expected-kernel-buffer-tolerance",
            ):
                require_ggml_hrx_run_loom_expected_buffer_tolerance(tool_path="/tmp/ggml-hrx-run-loom")

    def test_ggml_hrx_run_loom_tolerance_capability_reports_probe_failure(self) -> None:
        with mock.patch("subprocess.run", side_effect=OSError("permission denied")):
            with self.assertRaisesRegex(
                RuntimeError,
                "failed to query ggml-hrx-run-loom capabilities",
            ):
                require_ggml_hrx_run_loom_expected_buffer_tolerance(tool_path="/tmp/ggml-hrx-run-loom")


if __name__ == "__main__":
    unittest.main()
