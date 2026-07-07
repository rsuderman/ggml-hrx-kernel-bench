cmake_minimum_required(VERSION 3.20)

project(ggml_hrx_kernel_bench LANGUAGES NONE)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(Python3 REQUIRED COMPONENTS Interpreter)

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

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
