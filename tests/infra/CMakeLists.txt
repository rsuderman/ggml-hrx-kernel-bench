add_required_tool_test(loom-link)
add_required_tool_test(loom-compile)
add_required_tool_test(iree-benchmark-loom)

add_grouped_yaml_import_test(
  grouped-yaml-import-test-suite
  ${GGML_HRX_TESTS_ROOT}/infra/data/test_suite.yaml
  ${CMAKE_CURRENT_BINARY_DIR}/artifacts/grouped-yaml-import-test-suite
  --expect-mapped-cases 1
  --expect-unmapped-cases 4
  --expect-config-count 1
  --expect-unmapped-reason no_kernel_family_mapping=3
  --expect-unmapped-reason shape_lowering_not_implemented=1
  --expect-kernel-case copy_f32_f16:1,16384
)
