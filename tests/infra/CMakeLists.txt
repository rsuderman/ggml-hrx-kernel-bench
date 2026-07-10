add_required_tool_test(loom-link)
add_required_tool_test(loom-compile)
add_required_tool_test(iree-test-loom)
add_required_tool_test(iree-benchmark-loom)

if(GGML_HRX_ENABLE_LEGACY_GROUPED_YAML_IMPORTS)
  add_grouped_yaml_import_validation_target(
    NAME grouped-yaml-import-test-suite
    YAML_PATH ${GGML_HRX_TESTS_ROOT}/infra/data/test_suite.yaml
    OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/artifacts/grouped-yaml-import-test-suite
    EXPECTED_COVERAGE ${GGML_HRX_TESTS_ROOT}/infra/data/test_suite.import-coverage.json
  )
endif()
