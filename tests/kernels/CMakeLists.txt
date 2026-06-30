set(GGML_HRX_GROUPED_YAML ${CMAKE_CURRENT_SOURCE_DIR}/data/llamacpp_test.yaml)
set(GGML_HRX_EXPECTED_IMPORT_COVERAGE ${CMAKE_CURRENT_SOURCE_DIR}/data/llamacpp_test.import-coverage.json)
set(GGML_HRX_GENERATED_IMPORT_DIR ${CMAKE_CURRENT_BINARY_DIR}/artifacts/grouped-yaml-import)

add_grouped_yaml_import_validation_target(
  kernel-grouped-yaml-import-coverage
  ${GGML_HRX_GROUPED_YAML}
  ${GGML_HRX_GENERATED_IMPORT_DIR}
  ${GGML_HRX_EXPECTED_IMPORT_COVERAGE}
  --split-by-op
)

add_generated_kernel_runtime_tests(
  NAME grouped-yaml
  GENERATED_IMPORT_DIR ${GGML_HRX_GENERATED_IMPORT_DIR}
  GROUPED_YAML ${GGML_HRX_GROUPED_YAML}
)
