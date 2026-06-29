set(GGML_HRX_GROUPED_YAML ${CMAKE_CURRENT_SOURCE_DIR}/data/llamacpp_test.yaml)
set(GGML_HRX_EXPECTED_IMPORT_COVERAGE ${CMAKE_CURRENT_SOURCE_DIR}/data/llamacpp_test.import-coverage.json)
set(GGML_HRX_GENERATED_IMPORT_DIR ${CMAKE_CURRENT_BINARY_DIR}/artifacts/grouped-yaml-import)
set(GGML_HRX_GENERATED_COPY_CONFIG ${GGML_HRX_GENERATED_IMPORT_DIR}/ops/CPY/generated-import-configs)
set(GGML_HRX_GENERATED_ADD_CONFIG_DIR ${GGML_HRX_GENERATED_IMPORT_DIR}/ops/ADD/generated-import-configs)
set(GGML_HRX_GENERATED_ADD_CONTIGUOUS_CONFIG ${GGML_HRX_GENERATED_ADD_CONFIG_DIR}/add_f32.add_f32_contiguous_2d_r64_1048576_wg256.json)
set(GGML_HRX_GENERATED_ADD_GENERIC_CONFIG ${GGML_HRX_GENERATED_ADD_CONFIG_DIR}/add_f32.add_f32_generic_wg256.json)

add_grouped_yaml_import_validation_target(
  kernel-grouped-yaml-import-coverage
  ${GGML_HRX_GROUPED_YAML}
  ${GGML_HRX_GENERATED_IMPORT_DIR}
  ${GGML_HRX_EXPECTED_IMPORT_COVERAGE}
  --split-by-op
)

add_generated_kernel_test_suite(
  NAME copy-f32-f16
  CONFIG_PATH ${GGML_HRX_GENERATED_COPY_CONFIG}
  EXPECT_CASE 1,16384
)

add_generated_kernel_test_suite(
  NAME add-f32-contiguous
  CONFIG_PATH ${GGML_HRX_GENERATED_ADD_CONTIGUOUS_CONFIG}
  EXPECT_ROUTE_ID add_f32_contiguous_2d_r64_1048576_wg256
  EXPECT_CASE 64,262144,64,64,64
)

add_generated_kernel_test_suite(
  NAME add-f32-generic
  CONFIG_PATH ${GGML_HRX_GENERATED_ADD_GENERIC_CONFIG}
  EXPECT_ROUTE_ID add_f32_generic_wg256
  EXPECT_CASE 10,5,10,10,10
)
