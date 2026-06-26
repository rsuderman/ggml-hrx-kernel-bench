find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(GGML_HRX_TOOL_DIR "" CACHE PATH "Optional directory containing loom-link, loom-compile, and iree-benchmark-loom")

function(add_required_tool_test tool_name)
  set(test_command
    ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/check_required_tool.py
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

add_required_tool_test(loom-link)
add_required_tool_test(loom-compile)
add_required_tool_test(iree-benchmark-loom)
