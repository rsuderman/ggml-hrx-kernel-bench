include_guard(GLOBAL)

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
      ${GGML_HRX_TESTS_ROOT}/infra/generate_kernel_runtime_tests_cmake.py
      ${GGML_HRX_TESTS_ROOT}/infra/run_generated_kernel_tests.py
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

function(add_generated_kernel_runtime_tests)
  set(options)
  set(one_value_args NAME GENERATED_IMPORT_DIR EXPECTED_COVERAGE CASE_SELECTOR RUNTIME_OUTPUT_DIR)
  cmake_parse_arguments(GGML_HRX_GKRT "${options}" "${one_value_args}" "" ${ARGN})

  if(NOT GGML_HRX_GKRT_NAME)
    message(FATAL_ERROR "add_generated_kernel_runtime_tests requires NAME")
  endif()
  if(NOT GGML_HRX_GKRT_GENERATED_IMPORT_DIR)
    message(FATAL_ERROR "add_generated_kernel_runtime_tests requires GENERATED_IMPORT_DIR")
  endif()
  if(NOT GGML_HRX_GKRT_EXPECTED_COVERAGE)
    message(FATAL_ERROR "add_generated_kernel_runtime_tests requires EXPECTED_COVERAGE")
  endif()

  set(kernel_case_selector 0)
  if(GGML_HRX_GKRT_CASE_SELECTOR)
    set(kernel_case_selector ${GGML_HRX_GKRT_CASE_SELECTOR})
  endif()

  set(runtime_output_dir ${GGML_HRX_GKRT_RUNTIME_OUTPUT_DIR})
  if(NOT runtime_output_dir)
    set(runtime_output_dir ${CMAKE_CURRENT_BINARY_DIR}/artifacts/kernel-run-${GGML_HRX_GKRT_NAME}-generated)
  endif()

  set(generated_tests_include ${CMAKE_CURRENT_BINARY_DIR}/${GGML_HRX_GKRT_NAME}-generated-kernel-runtime-tests.cmake)
  set(generate_tests_command
    ${Python3_EXECUTABLE}
    ${GGML_HRX_TESTS_ROOT}/infra/generate_kernel_runtime_tests_cmake.py
    --output ${generated_tests_include}
    --name ${GGML_HRX_GKRT_NAME}
    --expected-coverage ${GGML_HRX_GKRT_EXPECTED_COVERAGE}
    --generated-import-dir ${GGML_HRX_GKRT_GENERATED_IMPORT_DIR}
    --python-executable ${Python3_EXECUTABLE}
    --runner-script ${GGML_HRX_TESTS_ROOT}/infra/run_generated_kernel_tests.py
    --runtime-output-dir ${runtime_output_dir}
    --case-selector ${kernel_case_selector}
  )
  if(GGML_HRX_TOOL_DIR)
    list(APPEND generate_tests_command --tool-dir ${GGML_HRX_TOOL_DIR})
  endif()
  if(GGML_HRX_ROCM_PATH)
    list(APPEND generate_tests_command --rocm-path ${GGML_HRX_ROCM_PATH})
  endif()

  execute_process(
    COMMAND ${generate_tests_command}
    COMMAND_ERROR_IS_FATAL ANY
  )
  include(${generated_tests_include})
endfunction()
