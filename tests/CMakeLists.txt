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

function(add_grouped_yaml_import_test test_name yaml_path output_dir)
  add_test(
    NAME ${test_name}
    COMMAND
      ${Python3_EXECUTABLE}
      ${GGML_HRX_TESTS_ROOT}/infra/check_grouped_yaml_import.py
      ${yaml_path}
      ${output_dir}
      ${ARGN}
  )
endfunction()

add_subdirectory(infra)
add_subdirectory(kernels)
