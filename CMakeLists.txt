cmake_minimum_required(VERSION 3.22)

project(ggml_hrx_kernel_bench LANGUAGES C CXX)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(Python3 REQUIRED COMPONENTS Interpreter)

option(
  GGML_HRX_BUILD_V2_NATIVE_SELECTOR
  "Build the experimental native v2 route selector Python module"
  OFF
)

if(GGML_HRX_BUILD_V2_NATIVE_SELECTOR)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m nanobind --cmake_dir
    RESULT_VARIABLE GGML_HRX_NANOBIND_CMAKE_DIR_RESULT
    OUTPUT_VARIABLE GGML_HRX_NANOBIND_CMAKE_DIR
    ERROR_VARIABLE GGML_HRX_NANOBIND_CMAKE_DIR_ERROR
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(NOT GGML_HRX_NANOBIND_CMAKE_DIR_RESULT EQUAL 0)
    message(
      FATAL_ERROR
      "GGML_HRX_BUILD_V2_NATIVE_SELECTOR requires nanobind in "
      "${Python3_EXECUTABLE}: ${GGML_HRX_NANOBIND_CMAKE_DIR_ERROR}"
    )
  endif()

  # Nanobind uses FindPython. Pin it to the interpreter already selected by the
  # project so discovery cannot silently switch Python installations.
  set(Python_EXECUTABLE "${Python3_EXECUTABLE}")
  find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)
  find_package(
    nanobind CONFIG REQUIRED
    PATHS "${GGML_HRX_NANOBIND_CMAKE_DIR}"
    NO_DEFAULT_PATH
  )
endif()

include(GgmlHrxLoomTools)
ggml_hrx_configure_loom_tools()

set(
  GGML_HRX_ASSET_ROOT
  "${CMAKE_BINARY_DIR}/generated/assets"
  CACHE PATH
  "Materialized runtime asset output root"
)
set(
  GGML_HRX_ACTIVE_ASSET_ROOT_METADATA_PATH
  "${CMAKE_SOURCE_DIR}/build/generated/active-asset-root.json"
  CACHE FILEPATH
  "Published metadata for the currently active materialized runtime asset root"
)
set(
  GGML_HRX_RUNTIME_ASSET_TARGET
  "runtime-assets"
  CACHE STRING
  "Standalone target that materializes runtime kernels and routing descriptors"
)

include(GgmlHrxAssets)
add_materialized_asset_target(
  NAME ${GGML_HRX_RUNTIME_ASSET_TARGET}
  OUTPUT_DIR ${GGML_HRX_ASSET_ROOT}
  METADATA_PATH ${GGML_HRX_ACTIVE_ASSET_ROOT_METADATA_PATH}
)

if(GGML_HRX_BUILD_V2_NATIVE_SELECTOR)
  add_subdirectory(native/v2_route_selector)
endif()

include(CTest)

add_subdirectory(tools/run-loom-simple)

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
