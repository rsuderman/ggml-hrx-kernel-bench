set(GGML_HRX_GROUPED_YAML ${CMAKE_CURRENT_SOURCE_DIR}/data/llamacpp_test.yaml)
set(GGML_HRX_EXPECTED_IMPORT_COVERAGE ${CMAKE_CURRENT_SOURCE_DIR}/data/llamacpp_test.import-coverage.json)
set(GGML_HRX_EXPECTED_IMPORT_COVERAGE_V2 ${CMAKE_CURRENT_SOURCE_DIR}/data/llamacpp_test.v2.import-coverage.json)
set(GGML_HRX_ROUTING_DIR_V1 ${CMAKE_SOURCE_DIR}/catalog/hrx2)
set(GGML_HRX_ROUTING_DIR_V2 ${CMAKE_SOURCE_DIR}/catalog/v2)
set(GGML_HRX_GENERATED_IMPORT_DIR ${CMAKE_CURRENT_BINARY_DIR}/artifacts/llama-cpp-tests-import-v1)
set(GGML_HRX_GENERATED_IMPORT_DIR_V2 ${CMAKE_CURRENT_BINARY_DIR}/artifacts/llama-cpp-tests-import-v2)

add_grouped_yaml_import_validation_target(
  kernel-llama-cpp-tests-import-coverage
  ${GGML_HRX_GROUPED_YAML}
  ${GGML_HRX_GENERATED_IMPORT_DIR}
  ${GGML_HRX_EXPECTED_IMPORT_COVERAGE}
  --routing-version v1
  --routing-dir ${GGML_HRX_ROUTING_DIR_V1}
  --split-by-op
)

add_generated_kernel_runtime_tests(
  NAME llama-cpp-tests
  GENERATED_IMPORT_DIR ${GGML_HRX_GENERATED_IMPORT_DIR}
  GROUPED_YAML ${GGML_HRX_GROUPED_YAML}
  ROUTING_VERSION v1
  ROUTING_DIR ${GGML_HRX_ROUTING_DIR_V1}
)

add_grouped_yaml_import_validation_target(
  kernel-llama-cpp-tests-import-coverage-v2
  ${GGML_HRX_GROUPED_YAML}
  ${GGML_HRX_GENERATED_IMPORT_DIR_V2}
  ${GGML_HRX_EXPECTED_IMPORT_COVERAGE_V2}
  --routing-version v2
  --routing-dir ${GGML_HRX_ROUTING_DIR_V2}
  --split-by-op
)

add_generated_kernel_runtime_tests(
  NAME llama-cpp-tests-v2
  GENERATED_IMPORT_DIR ${GGML_HRX_GENERATED_IMPORT_DIR_V2}
  GROUPED_YAML ${GGML_HRX_GROUPED_YAML}
  ROUTING_VERSION v2
  ROUTING_DIR ${GGML_HRX_ROUTING_DIR_V2}
)
