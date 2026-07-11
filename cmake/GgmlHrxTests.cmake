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

function(add_yaml_route_import_target)
  set(options)
  set(one_value_args NAME OUTPUT_DIR ROUTING_DIR EXPECTED_COVERAGE)
  set(multi_value_args YAML_PATHS DEPENDS)
  cmake_parse_arguments(GGML_HRX_YRI "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  if(NOT GGML_HRX_YRI_NAME)
    message(FATAL_ERROR "add_yaml_route_import_target requires NAME")
  endif()
  if(NOT GGML_HRX_YRI_OUTPUT_DIR)
    message(FATAL_ERROR "add_yaml_route_import_target requires OUTPUT_DIR")
  endif()
  if(NOT GGML_HRX_YRI_ROUTING_DIR)
    message(FATAL_ERROR "add_yaml_route_import_target requires ROUTING_DIR")
  endif()
  if(NOT GGML_HRX_YRI_EXPECTED_COVERAGE)
    message(FATAL_ERROR "add_yaml_route_import_target requires EXPECTED_COVERAGE")
  endif()
  if(NOT GGML_HRX_YRI_YAML_PATHS)
    message(FATAL_ERROR "add_yaml_route_import_target requires YAML_PATHS")
  endif()
  if(GGML_HRX_TOOL_BUILD_TARGET)
    list(APPEND GGML_HRX_YRI_DEPENDS ${GGML_HRX_TOOL_BUILD_TARGET})
  endif()

  set(stamp_path ${GGML_HRX_YRI_OUTPUT_DIR}/route-import.stamp)
  set(check_command
    ${Python3_EXECUTABLE}
    ${GGML_HRX_TESTS_ROOT}/infra/check_yaml_route_import.py
    ${GGML_HRX_YRI_OUTPUT_DIR}
    --routing-dir ${GGML_HRX_YRI_ROUTING_DIR}
    --expected-coverage ${GGML_HRX_YRI_EXPECTED_COVERAGE}
  )
  foreach(yaml_path IN LISTS GGML_HRX_YRI_YAML_PATHS)
    list(APPEND check_command --yaml ${yaml_path})
  endforeach()
  if(GGML_HRX_TOOL_DIR)
    list(APPEND check_command --tool-dir ${GGML_HRX_TOOL_DIR})
    set(yaml_route_import_env_command
      ${CMAKE_COMMAND} -E env
      PATH=${GGML_HRX_TOOL_DIR}:$ENV{PATH}
      ${check_command}
    )
  else()
    set(yaml_route_import_env_command ${check_command})
  endif()
  add_custom_command(
    OUTPUT ${stamp_path}
    COMMAND ${yaml_route_import_env_command}
    COMMAND ${CMAKE_COMMAND} -E touch ${stamp_path}
    DEPENDS
      ${GGML_HRX_YRI_YAML_PATHS}
      ${GGML_HRX_YRI_EXPECTED_COVERAGE}
      ${GGML_HRX_YRI_DEPENDS}
      ${GGML_HRX_TESTS_ROOT}/infra/bootstrap.py
      ${GGML_HRX_TESTS_ROOT}/infra/check_yaml_route_import.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/yaml_route_import.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/routing/v2/catalog.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/routing/v2/layout.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/routing/v2/matching.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/routing/v2/models.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/routing/v2/query.py
    COMMENT "Materializing descriptor YAML route import reports for ${GGML_HRX_YRI_NAME}"
    VERBATIM
  )
  add_custom_target(${GGML_HRX_YRI_NAME} ALL DEPENDS ${stamp_path})
endfunction()

function(add_yaml_route_import_runtime_tests)
  set(options)
  set(one_value_args NAME GENERATED_IMPORT_DIR GROUPED_YAML RUNTIME_OUTPUT_DIR ROUTING_DIR KERNEL_DIR)
  set(multi_value_args EXCLUDE_OPS)
  cmake_parse_arguments(GGML_HRX_YRIRT "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  if(NOT GGML_HRX_YRIRT_NAME)
    message(FATAL_ERROR "add_yaml_route_import_runtime_tests requires NAME")
  endif()
  if(NOT GGML_HRX_YRIRT_GENERATED_IMPORT_DIR)
    message(FATAL_ERROR "add_yaml_route_import_runtime_tests requires GENERATED_IMPORT_DIR")
  endif()
  if(NOT GGML_HRX_YRIRT_GROUPED_YAML)
    message(FATAL_ERROR "add_yaml_route_import_runtime_tests requires GROUPED_YAML")
  endif()

  set(runtime_output_dir ${GGML_HRX_YRIRT_RUNTIME_OUTPUT_DIR})
  if(NOT runtime_output_dir)
    set(runtime_output_dir ${CMAKE_CURRENT_BINARY_DIR}/artifacts/kernel-run-${GGML_HRX_YRIRT_NAME}-generated)
  endif()

  set(generated_tests_include ${CMAKE_CURRENT_BINARY_DIR}/${GGML_HRX_YRIRT_NAME}-generated-kernel-runtime-tests.cmake)
  set(generate_tests_command
    ${Python3_EXECUTABLE}
    ${GGML_HRX_TESTS_ROOT}/infra/generate_kernel_runtime_tests_cmake.py
    --output ${generated_tests_include}
    --name ${GGML_HRX_YRIRT_NAME}
    --grouped-yaml ${GGML_HRX_YRIRT_GROUPED_YAML}
    --generated-import-dir ${GGML_HRX_YRIRT_GENERATED_IMPORT_DIR}
    --python-executable ${Python3_EXECUTABLE}
    --runner-script ${GGML_HRX_TESTS_ROOT}/infra/run_generated_kernel_tests.py
    --runtime-output-dir ${runtime_output_dir}
    --routing-version v2
    --all-ops
  )
  if(GGML_HRX_YRIRT_ROUTING_DIR)
    list(APPEND generate_tests_command --routing-dir ${GGML_HRX_YRIRT_ROUTING_DIR})
  endif()
  if(GGML_HRX_YRIRT_KERNEL_DIR)
    list(APPEND generate_tests_command --kernel-dir ${GGML_HRX_YRIRT_KERNEL_DIR})
  endif()
  if(GGML_HRX_TOOL_DIR)
    list(APPEND generate_tests_command --tool-dir ${GGML_HRX_TOOL_DIR})
  endif()
  if(GGML_HRX_ROCM_PATH)
    list(APPEND generate_tests_command --rocm-path ${GGML_HRX_ROCM_PATH})
  endif()
  foreach(excluded_op IN LISTS GGML_HRX_YRIRT_EXCLUDE_OPS)
    list(APPEND generate_tests_command --exclude-op ${excluded_op})
  endforeach()

  execute_process(
    COMMAND ${generate_tests_command}
    COMMAND_ERROR_IS_FATAL ANY
  )
  include(${generated_tests_include})
endfunction()
