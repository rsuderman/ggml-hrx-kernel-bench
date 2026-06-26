from __future__ import annotations

import unittest

import bootstrap  # noqa: F401

from ggml_hrx_kernel_bench.required_tools import require_tool


class RequiredToolAvailabilityTest(unittest.TestCase):
    def test_loom_link_available(self) -> None:
        self.assertTrue(require_tool("loom-link"))

    def test_loom_compile_available(self) -> None:
        self.assertTrue(require_tool("loom-compile"))

    def test_iree_benchmark_loom_available(self) -> None:
        self.assertTrue(require_tool("iree-benchmark-loom"))


if __name__ == "__main__":
    unittest.main()
