set(GGML_HRX_GROUPED_YAML ${CMAKE_CURRENT_SOURCE_DIR}/data/llamacpp_test.yaml)
set(GGML_HRX_GENERATED_IMPORT_DIR ${CMAKE_CURRENT_BINARY_DIR}/artifacts/grouped-yaml-import)
set(GGML_HRX_GENERATED_COPY_CONFIG ${GGML_HRX_GENERATED_IMPORT_DIR}/ops/CPY/generated-import-configs/copy_f32_f16.json)

set(import_grouped_yaml_command
  ${Python3_EXECUTABLE}
  ${GGML_HRX_TESTS_ROOT}/infra/import_grouped_yaml.py
  ${GGML_HRX_GROUPED_YAML}
  --split-by-op
  --output-dir ${GGML_HRX_GENERATED_IMPORT_DIR}
)
add_test(
  NAME kernel-generate-grouped-yaml-import
  COMMAND ${import_grouped_yaml_command}
)
set_tests_properties(kernel-generate-grouped-yaml-import PROPERTIES FIXTURES_SETUP grouped_yaml_import)

add_test(
  NAME kernel-test-config-copy-f32-f16-generated
  COMMAND
    ${Python3_EXECUTABLE}
    ${GGML_HRX_TESTS_ROOT}/infra/check_kernel_test_config.py
    ${GGML_HRX_GENERATED_COPY_CONFIG}
    --expect-case 1,16384
)
set_tests_properties(kernel-test-config-copy-f32-f16-generated PROPERTIES FIXTURES_REQUIRED grouped_yaml_import)

set(kernel_run_generated_command
  ${Python3_EXECUTABLE}
  ${GGML_HRX_TESTS_ROOT}/infra/run_kernel_test_config.py
  ${GGML_HRX_GENERATED_COPY_CONFIG}
  0
  --output-dir ${CMAKE_CURRENT_BINARY_DIR}/artifacts/kernel-run-copy-f32-f16-generated
)
if(GGML_HRX_TOOL_DIR)
  list(APPEND kernel_run_generated_command --tool-dir ${GGML_HRX_TOOL_DIR})
endif()
if(GGML_HRX_ROCM_PATH)
  list(APPEND kernel_run_generated_command --rocm-path ${GGML_HRX_ROCM_PATH})
endif()
add_test(
  NAME kernel-run-copy-f32-f16-generated
  COMMAND ${kernel_run_generated_command}
)
set_tests_properties(kernel-run-copy-f32-f16-generated PROPERTIES FIXTURES_REQUIRED grouped_yaml_import)
