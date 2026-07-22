cmake_minimum_required(VERSION 3.22)

project(ggml_hrx_kernel_bench LANGUAGES C CXX)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(Python3 REQUIRED COMPONENTS Interpreter)

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

include(CTest)

add_subdirectory(native/v2_route_selector)

add_subdirectory(tools/v2-route-selector)
add_subdirectory(tools/run-loom-simple)

option(
  GGML_HRX_ENABLE_BENCHMARKS
  "Configure Loom benchmark materialization targets"
  OFF
)
if(GGML_HRX_ENABLE_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
