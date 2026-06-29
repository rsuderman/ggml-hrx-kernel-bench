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
  add_custom_command(
    OUTPUT ${stamp_path}
    COMMAND ${check_command}
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

add_subdirectory(infra)
add_subdirectory(kernels)
