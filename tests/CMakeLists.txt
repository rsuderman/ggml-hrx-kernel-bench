find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(GGML_HRX_TESTS_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(_ggml_hrx_tool_dir_default "")
if(DEFINED ENV{GGML_HRX_TOOL_DIR} AND NOT "$ENV{GGML_HRX_TOOL_DIR}" STREQUAL "")
  set(_ggml_hrx_tool_dir_default $ENV{GGML_HRX_TOOL_DIR})
endif()
set(GGML_HRX_TOOL_DIR "${_ggml_hrx_tool_dir_default}" CACHE PATH "Optional directory containing loom-link, loom-compile, and iree-benchmark-loom")
set(GGML_HRX_ROCM_PATH "" CACHE PATH "Optional ROCm root for runtime tests")

include(GgmlHrxTests)

add_subdirectory(infra)
add_subdirectory(kernels)
