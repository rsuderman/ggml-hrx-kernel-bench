find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(GGML_HRX_TESTS_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(GGML_HRX_TOOL_DIR "" CACHE PATH "Optional directory containing loom-link, loom-compile, and iree-benchmark-loom")
set(GGML_HRX_ROCM_PATH "" CACHE PATH "Optional ROCm root for runtime tests")

function(add_required_tool_test tool_name)
  set(test_command
    ${Python3_EXECUTABLE}
    ${GGML_HRX_TESTS_ROOT}/infra/check_required_tool.py
    ${tool_name}
  )
  if(GGML_HRX_TOOL_DIR)
    list(APPEND test_command --tool-dir ${GGML_HRX_TOOL_DIR})
  endif()
  add_test(
    NAME required-tool-${tool_name}
    COMMAND ${test_command}
  )
endfunction()

function(add_grouped_yaml_import_validation_target target_name yaml_path output_dir expected_coverage)
  set(stamp_path ${output_dir}/import-coverage.stamp)
  set(check_command
    ${Python3_EXECUTABLE}
    ${GGML_HRX_TESTS_ROOT}/infra/check_grouped_yaml_import.py
    ${yaml_path}
    ${output_dir}
    --expected-coverage ${expected_coverage}
    ${ARGN}
  )
  if(GGML_HRX_TOOL_DIR)
    list(APPEND check_command --tool-dir ${GGML_HRX_TOOL_DIR})
    set(grouped_yaml_env_command
      ${CMAKE_COMMAND} -E env
      PATH=${GGML_HRX_TOOL_DIR}:$ENV{PATH}
      ${check_command}
    )
  else()
    set(grouped_yaml_env_command ${check_command})
  endif()
  add_custom_command(
    OUTPUT ${stamp_path}
    COMMAND ${grouped_yaml_env_command}
    COMMAND ${CMAKE_COMMAND} -E touch ${stamp_path}
    DEPENDS
      ${yaml_path}
      ${expected_coverage}
      ${GGML_HRX_TESTS_ROOT}/infra/bootstrap.py
      ${GGML_HRX_TESTS_ROOT}/infra/check_grouped_yaml_import.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/grouped_yaml_import.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/import_mapping_registry.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/import_models.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/family_specs.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/hrx2.py
    COMMENT "Validating grouped YAML import coverage for ${target_name}"
    VERBATIM
  )
  add_custom_target(${target_name} ALL DEPENDS ${stamp_path})
endfunction()

function(add_generated_kernel_test_suite)
  set(options)
  set(one_value_args NAME CONFIG_PATH EXPECT_CASE EXPECT_ROUTE_ID CASE_SELECTOR RUNTIME_OUTPUT_DIR)
  cmake_parse_arguments(GGML_HRX_GKTS "${options}" "${one_value_args}" "" ${ARGN})

  if(NOT GGML_HRX_GKTS_NAME)
    message(FATAL_ERROR "add_generated_kernel_test_suite requires NAME")
  endif()
  if(NOT GGML_HRX_GKTS_CONFIG_PATH)
    message(FATAL_ERROR "add_generated_kernel_test_suite requires CONFIG_PATH")
  endif()

  set(config_test_command
    ${Python3_EXECUTABLE}
    ${GGML_HRX_TESTS_ROOT}/infra/check_kernel_test_config.py
    ${GGML_HRX_GKTS_CONFIG_PATH}
  )
  if(GGML_HRX_GKTS_EXPECT_ROUTE_ID)
    list(APPEND config_test_command --expect-route-id ${GGML_HRX_GKTS_EXPECT_ROUTE_ID})
  endif()
  if(GGML_HRX_GKTS_EXPECT_CASE)
    list(APPEND config_test_command --expect-case ${GGML_HRX_GKTS_EXPECT_CASE})
  endif()
  add_test(
    NAME kernel-test-config-${GGML_HRX_GKTS_NAME}-generated
    COMMAND ${config_test_command}
  )

  set(kernel_case_selector 0)
  if(GGML_HRX_GKTS_CASE_SELECTOR)
    set(kernel_case_selector ${GGML_HRX_GKTS_CASE_SELECTOR})
  endif()

  set(runtime_output_dir ${GGML_HRX_GKTS_RUNTIME_OUTPUT_DIR})
  if(NOT runtime_output_dir)
    set(runtime_output_dir ${CMAKE_CURRENT_BINARY_DIR}/artifacts/kernel-run-${GGML_HRX_GKTS_NAME}-generated)
  endif()

  set(runtime_test_command
    ${Python3_EXECUTABLE}
    ${GGML_HRX_TESTS_ROOT}/infra/run_kernel_test_config.py
    ${GGML_HRX_GKTS_CONFIG_PATH}
    ${kernel_case_selector}
    --output-dir ${runtime_output_dir}
  )
  if(GGML_HRX_TOOL_DIR)
    list(APPEND runtime_test_command --tool-dir ${GGML_HRX_TOOL_DIR})
  endif()
  if(GGML_HRX_ROCM_PATH)
    list(APPEND runtime_test_command --rocm-path ${GGML_HRX_ROCM_PATH})
  endif()
  add_test(
    NAME kernel-run-${GGML_HRX_GKTS_NAME}-generated
    COMMAND ${runtime_test_command}
  )
endfunction()

add_subdirectory(infra)
add_subdirectory(kernels)
