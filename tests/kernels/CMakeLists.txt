add_kernel_test_config_test(
  kernel-test-config-copy-f32-f16
  ${CMAKE_CURRENT_SOURCE_DIR}/data/copy_f32_f16.test.json
)

add_kernel_execution_test(
  kernel-run-copy-f32-f16
  ${CMAKE_CURRENT_SOURCE_DIR}/data/copy_f32_f16.test.json
  1
)
