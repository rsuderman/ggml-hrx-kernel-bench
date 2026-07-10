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

function(add_grouped_yaml_import_validation_target)
  set(options)
  set(one_value_args NAME YAML_PATH OUTPUT_DIR EXPECTED_COVERAGE)
  set(multi_value_args COMMAND_ARGS DEPENDS)
  cmake_parse_arguments(GGML_HRX_GYI "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  if(NOT GGML_HRX_GYI_NAME)
    message(FATAL_ERROR "add_grouped_yaml_import_validation_target requires NAME")
  endif()
  if(NOT GGML_HRX_GYI_YAML_PATH)
    message(FATAL_ERROR "add_grouped_yaml_import_validation_target requires YAML_PATH")
  endif()
  if(NOT GGML_HRX_GYI_OUTPUT_DIR)
    message(FATAL_ERROR "add_grouped_yaml_import_validation_target requires OUTPUT_DIR")
  endif()
  if(NOT GGML_HRX_GYI_EXPECTED_COVERAGE)
    message(FATAL_ERROR "add_grouped_yaml_import_validation_target requires EXPECTED_COVERAGE")
  endif()
  if(GGML_HRX_TOOL_BUILD_TARGET)
    list(APPEND GGML_HRX_GYI_DEPENDS ${GGML_HRX_TOOL_BUILD_TARGET})
  endif()

  set(stamp_path ${GGML_HRX_GYI_OUTPUT_DIR}/import-coverage.stamp)
  set(check_command
    ${Python3_EXECUTABLE}
    ${GGML_HRX_TESTS_ROOT}/infra/check_grouped_yaml_import.py
    ${GGML_HRX_GYI_YAML_PATH}
    ${GGML_HRX_GYI_OUTPUT_DIR}
    --expected-coverage ${GGML_HRX_GYI_EXPECTED_COVERAGE}
    ${GGML_HRX_GYI_COMMAND_ARGS}
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
      ${GGML_HRX_GYI_YAML_PATH}
      ${GGML_HRX_GYI_EXPECTED_COVERAGE}
      ${GGML_HRX_GYI_DEPENDS}
      ${GGML_HRX_TESTS_ROOT}/infra/bootstrap.py
      ${GGML_HRX_TESTS_ROOT}/infra/check_grouped_yaml_import.py
      ${GGML_HRX_TESTS_ROOT}/infra/generate_kernel_runtime_tests_cmake.py
      ${GGML_HRX_TESTS_ROOT}/infra/run_generated_kernel_tests.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/grouped_yaml_import.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/import_mapping_registry.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/import_models.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/routing/api.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/routing/case_selection.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/routing/models.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/routing/__init__.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/routing/v1/__init__.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/routing/v1/backend.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/routing/v1/export.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/routing/v1/family_specs.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/routing/v1/importer.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/routing/v1/routes.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/routing/v1/runtime.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/routing/v1/schedules.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/routing/v2/__init__.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/routing/v2/backend.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/routing/v2/candidates.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/routing/v2/catalog.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/routing/v2/import_resolution.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/routing/v2/manifest.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/routing/v2/matching.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/routing/v2/models.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/routing/v2/query.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/routing/v2/runtime.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/routing/v2/layout.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/routing/v2/serialization.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/routing/v2/shape.py
    COMMENT "Validating grouped YAML import coverage for ${GGML_HRX_GYI_NAME}"
    VERBATIM
  )
  add_custom_target(${GGML_HRX_GYI_NAME} ALL DEPENDS ${stamp_path})
endfunction()

function(add_generated_kernel_runtime_tests)
  set(options)
  set(one_value_args NAME GENERATED_IMPORT_DIR GROUPED_YAML RUNTIME_OUTPUT_DIR ROUTING_VERSION ROUTING_DIR KERNEL_DIR)
  set(multi_value_args EXCLUDE_OPS)
  cmake_parse_arguments(GGML_HRX_GKRT "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  if(NOT GGML_HRX_GKRT_NAME)
    message(FATAL_ERROR "add_generated_kernel_runtime_tests requires NAME")
  endif()
  if(NOT GGML_HRX_GKRT_GENERATED_IMPORT_DIR)
    message(FATAL_ERROR "add_generated_kernel_runtime_tests requires GENERATED_IMPORT_DIR")
  endif()
  if(NOT GGML_HRX_GKRT_GROUPED_YAML)
    message(FATAL_ERROR "add_generated_kernel_runtime_tests requires GROUPED_YAML")
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
    --grouped-yaml ${GGML_HRX_GKRT_GROUPED_YAML}
    --generated-import-dir ${GGML_HRX_GKRT_GENERATED_IMPORT_DIR}
    --python-executable ${Python3_EXECUTABLE}
    --runner-script ${GGML_HRX_TESTS_ROOT}/infra/run_generated_kernel_tests.py
    --runtime-output-dir ${runtime_output_dir}
  )
  if(GGML_HRX_GKRT_ROUTING_VERSION)
    list(APPEND generate_tests_command --routing-version ${GGML_HRX_GKRT_ROUTING_VERSION})
  endif()
  if(GGML_HRX_GKRT_ROUTING_DIR)
    list(APPEND generate_tests_command --routing-dir ${GGML_HRX_GKRT_ROUTING_DIR})
  endif()
  if(GGML_HRX_GKRT_KERNEL_DIR)
    list(APPEND generate_tests_command --kernel-dir ${GGML_HRX_GKRT_KERNEL_DIR})
  endif()
  if(GGML_HRX_TOOL_DIR)
    list(APPEND generate_tests_command --tool-dir ${GGML_HRX_TOOL_DIR})
  endif()
  if(GGML_HRX_ROCM_PATH)
    list(APPEND generate_tests_command --rocm-path ${GGML_HRX_ROCM_PATH})
  endif()
  foreach(excluded_op IN LISTS GGML_HRX_GKRT_EXCLUDE_OPS)
    list(APPEND generate_tests_command --exclude-op ${excluded_op})
  endforeach()

  execute_process(
    COMMAND ${generate_tests_command}
    COMMAND_ERROR_IS_FATAL ANY
  )
  include(${generated_tests_include})
endfunction()
