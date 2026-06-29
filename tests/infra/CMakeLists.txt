add_required_tool_test(loom-link)
add_required_tool_test(loom-compile)
add_required_tool_test(iree-benchmark-loom)

add_grouped_yaml_import_validation_target(
  grouped-yaml-import-test-suite
  ${GGML_HRX_TESTS_ROOT}/infra/data/test_suite.yaml
  ${CMAKE_CURRENT_BINARY_DIR}/artifacts/grouped-yaml-import-test-suite
  ${GGML_HRX_TESTS_ROOT}/infra/data/test_suite.import-coverage.json
)
