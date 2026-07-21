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
  list(APPEND GGML_HRX_YRI_DEPENDS ggml-hrx-v2-route-selector)

  set(route_queries_path ${GGML_HRX_YRI_OUTPUT_DIR}/route-queries.jsonl)
  set(route_query_import_path ${GGML_HRX_YRI_OUTPUT_DIR}/route-query-import.json)
  set(query_import_stamp_path ${GGML_HRX_YRI_OUTPUT_DIR}/route-query-import.stamp)
  set(route_import_stamp_path ${GGML_HRX_YRI_OUTPUT_DIR}/route-import.stamp)
  set(query_import_command
    ${Python3_EXECUTABLE}
    ${GGML_HRX_TESTS_ROOT}/infra/import_yaml_route_queries.py
    ${GGML_HRX_YRI_OUTPUT_DIR}
    --routing-dir ${GGML_HRX_YRI_ROUTING_DIR}
  )
  set(route_config_command
    ${Python3_EXECUTABLE}
    ${GGML_HRX_TESTS_ROOT}/infra/generate_route_configs.py
    ${GGML_HRX_YRI_OUTPUT_DIR}
    --route-queries ${route_queries_path}
    --import-metadata ${route_query_import_path}
    --routing-dir ${GGML_HRX_YRI_ROUTING_DIR}
    --expected-coverage ${GGML_HRX_YRI_EXPECTED_COVERAGE}
  )
  foreach(yaml_path IN LISTS GGML_HRX_YRI_YAML_PATHS)
    list(APPEND query_import_command --yaml ${yaml_path})
  endforeach()
  set(yaml_route_import_environment
    "GGML_HRX_V2_ROUTE_SELECTOR=$<TARGET_FILE:ggml-hrx-v2-route-selector>"
  )
  if(GGML_HRX_TOOL_DIR)
    list(APPEND query_import_command --tool-dir ${GGML_HRX_TOOL_DIR})
    list(APPEND route_config_command --tool-dir ${GGML_HRX_TOOL_DIR})
    list(APPEND yaml_route_import_environment
      "PATH=${GGML_HRX_TOOL_DIR}:$ENV{PATH}"
    )
  endif()
  set(query_import_env_command
    ${CMAKE_COMMAND} -E env
    ${yaml_route_import_environment}
    ${query_import_command}
  )
  set(route_config_env_command
    ${CMAKE_COMMAND} -E env
    ${yaml_route_import_environment}
    ${route_config_command}
  )
  add_custom_command(
    OUTPUT ${query_import_stamp_path}
    BYPRODUCTS
      ${route_queries_path}
      ${route_query_import_path}
    COMMAND ${query_import_env_command}
    COMMAND ${CMAKE_COMMAND} -E touch ${query_import_stamp_path}
    DEPENDS
      ${GGML_HRX_YRI_YAML_PATHS}
      ${GGML_HRX_YRI_DEPENDS}
      ${GGML_HRX_TESTS_ROOT}/infra/bootstrap.py
      ${GGML_HRX_TESTS_ROOT}/infra/import_yaml_route_queries.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/route_query_config.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/yaml_route_import.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/routing/v2/catalog.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/routing/v2/layout.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/routing/v2/matching.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/routing/v2/models.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/routing/v2/query.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/routing/v2/selection.py
    COMMENT "Importing descriptor YAML route queries for ${GGML_HRX_YRI_NAME}"
    VERBATIM
  )
  add_custom_command(
    OUTPUT ${route_import_stamp_path}
    BYPRODUCTS
      ${GGML_HRX_YRI_OUTPUT_DIR}/generated-kernel-tests.json
      ${GGML_HRX_YRI_OUTPUT_DIR}/route-import-summary.json
    COMMAND ${route_config_env_command}
    COMMAND ${CMAKE_COMMAND} -E touch ${route_import_stamp_path}
    DEPENDS
      ${query_import_stamp_path}
      ${route_queries_path}
      ${route_query_import_path}
      ${GGML_HRX_YRI_EXPECTED_COVERAGE}
      ${GGML_HRX_YRI_DEPENDS}
      ${GGML_HRX_TESTS_ROOT}/infra/bootstrap.py
      ${GGML_HRX_TESTS_ROOT}/infra/check_yaml_route_import.py
      ${GGML_HRX_TESTS_ROOT}/infra/generate_route_configs.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/kernel_test_config.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/route_query_config.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/yaml_route_import.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/routing/v2/catalog.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/routing/v2/layout.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/routing/v2/matching.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/routing/v2/models.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/routing/v2/query.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/routing/v2/selection.py
    COMMENT "Generating descriptor route configs for ${GGML_HRX_YRI_NAME}"
    VERBATIM
  )
  add_custom_target(${GGML_HRX_YRI_NAME} ALL DEPENDS ${route_import_stamp_path})
endfunction()

function(add_yaml_route_import_descriptor_tests)
  set(options ENABLE_HSA_EXECUTION)
  set(one_value_args
    NAME
    GENERATED_IMPORT_DIR
    GROUPED_YAML
    DESCRIPTOR_OUTPUT_DIR
    PREPARE_OUTPUT_DIR
    EXECUTE_OUTPUT_DIR
    ROUTING_DIR
    KERNEL_DIR
    TARGET
    MAX_ELEMENTS
    LIMIT
    RUNNER
    REPO_ROOT
    IMPORT_TARGET
    PREPARE_TARGET
  )
  set(multi_value_args EXCLUDE_OPS)
  cmake_parse_arguments(GGML_HRX_YRIDT "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  if(NOT GGML_HRX_YRIDT_NAME)
    message(FATAL_ERROR "add_yaml_route_import_descriptor_tests requires NAME")
  endif()
  if(NOT GGML_HRX_YRIDT_GENERATED_IMPORT_DIR)
    message(FATAL_ERROR "add_yaml_route_import_descriptor_tests requires GENERATED_IMPORT_DIR")
  endif()
  if(NOT GGML_HRX_YRIDT_GROUPED_YAML)
    message(FATAL_ERROR "add_yaml_route_import_descriptor_tests requires GROUPED_YAML")
  endif()
  set_property(
    DIRECTORY
    APPEND
    PROPERTY CMAKE_CONFIGURE_DEPENDS
      ${GGML_HRX_TESTS_ROOT}/infra/generate_loom_descriptor_tests_cmake.py
  )

  set(descriptor_output_dir ${GGML_HRX_YRIDT_DESCRIPTOR_OUTPUT_DIR})
  if(NOT descriptor_output_dir)
    set(descriptor_output_dir ${CMAKE_CURRENT_BINARY_DIR}/artifacts/kernel-descriptors-${GGML_HRX_YRIDT_NAME})
  endif()
  set(prepare_output_dir ${GGML_HRX_YRIDT_PREPARE_OUTPUT_DIR})
  if(NOT prepare_output_dir)
    set(prepare_output_dir ${CMAKE_CURRENT_BINARY_DIR}/artifacts/kernel-prepare-${GGML_HRX_YRIDT_NAME})
  endif()
  set(execute_output_dir ${GGML_HRX_YRIDT_EXECUTE_OUTPUT_DIR})
  if(NOT execute_output_dir)
    set(execute_output_dir ${CMAKE_CURRENT_BINARY_DIR}/artifacts/kernel-execute-${GGML_HRX_YRIDT_NAME})
  endif()
  set(target ${GGML_HRX_YRIDT_TARGET})
  if(NOT target)
    set(target gfx1100)
  endif()
  set(max_elements ${GGML_HRX_YRIDT_MAX_ELEMENTS})
  if(NOT max_elements)
    set(max_elements 65536)
  endif()
  set(runner ${GGML_HRX_YRIDT_RUNNER})
  if(NOT runner)
    set(runner "$<TARGET_FILE:ggml-hrx-run-loom-simple>")
  endif()
  set(repo_root ${GGML_HRX_YRIDT_REPO_ROOT})
  if(NOT repo_root)
    set(repo_root ${CMAKE_SOURCE_DIR})
  endif()
  set(import_target ${GGML_HRX_YRIDT_IMPORT_TARGET})
  if(NOT import_target)
    set(import_target kernel-${GGML_HRX_YRIDT_NAME})
  endif()
  set(prepare_target ${GGML_HRX_YRIDT_PREPARE_TARGET})
  if(NOT prepare_target)
    set(prepare_target kernel-prepare-${GGML_HRX_YRIDT_NAME}-generated)
  endif()

  set(generated_tests_include ${CMAKE_CURRENT_BINARY_DIR}/${GGML_HRX_YRIDT_NAME}-loom-descriptor-tests.cmake)
  set(generate_tests_command
    ${Python3_EXECUTABLE}
    ${GGML_HRX_TESTS_ROOT}/infra/generate_loom_descriptor_tests_cmake.py
    --output ${generated_tests_include}
    --name ${GGML_HRX_YRIDT_NAME}
    --grouped-yaml ${GGML_HRX_YRIDT_GROUPED_YAML}
    --generated-import-dir ${GGML_HRX_YRIDT_GENERATED_IMPORT_DIR}
    --python-executable ${Python3_EXECUTABLE}
    --descriptor-generator-script ${GGML_HRX_TESTS_ROOT}/infra/generate_loom_execution_descriptors.py
    --descriptor-runner-script ${GGML_HRX_TESTS_ROOT}/infra/run_loom_execution_descriptors.py
    --descriptor-output-dir ${descriptor_output_dir}
    --prepare-output-dir ${prepare_output_dir}
    --execute-output-dir ${execute_output_dir}
    --target ${target}
    --max-elements ${max_elements}
    --runner ${runner}
    --repo-root ${repo_root}
    --build-prepare-target ${prepare_target}
    --import-target ${import_target}
    --all-ops
  )
  if(GGML_HRX_YRIDT_ROUTING_DIR)
    list(APPEND generate_tests_command --routing-dir ${GGML_HRX_YRIDT_ROUTING_DIR})
  endif()
  if(GGML_HRX_YRIDT_KERNEL_DIR)
    list(APPEND generate_tests_command --kernel-dir ${GGML_HRX_YRIDT_KERNEL_DIR})
  endif()
  if(GGML_HRX_TOOL_DIR)
    list(APPEND generate_tests_command --tool-dir ${GGML_HRX_TOOL_DIR})
  endif()
  if(GGML_HRX_YRIDT_LIMIT)
    list(APPEND generate_tests_command --limit ${GGML_HRX_YRIDT_LIMIT})
  endif()
  if(GGML_HRX_YRIDT_ENABLE_HSA_EXECUTION)
    list(APPEND generate_tests_command --execute-hsa)
  endif()
  foreach(excluded_op IN LISTS GGML_HRX_YRIDT_EXCLUDE_OPS)
    list(APPEND generate_tests_command --exclude-op ${excluded_op})
  endforeach()

  execute_process(
    COMMAND ${generate_tests_command}
    COMMAND_ERROR_IS_FATAL ANY
  )
  include(${generated_tests_include})
endfunction()
