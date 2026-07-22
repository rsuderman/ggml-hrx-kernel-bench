set(GGML_HRX_GROUPED_YAML ${CMAKE_CURRENT_SOURCE_DIR}/data/llamacpp_test.v2.yaml)
set(GGML_HRX_EXPECTED_YAML_ROUTE_IMPORT_COVERAGE_V2 ${CMAKE_CURRENT_SOURCE_DIR}/data/llamacpp.import-coverage.json)
set(GGML_HRX_EXPECTED_NATIVE_ROUTE_COUNT_V2 ${CMAKE_CURRENT_SOURCE_DIR}/data/llamacpp.native-route-count.json)
set(GGML_HRX_KERNEL_DIR_V2 ${GGML_HRX_ASSET_ROOT}/kernels/v2)
set(GGML_HRX_ROUTING_DIR_V2 ${GGML_HRX_ASSET_ROOT}/catalog/v2)
set(GGML_HRX_LLAMA_CPP_YAML_ROUTE_IMPORT_DIR_V2 ${CMAKE_CURRENT_BINARY_DIR}/artifacts/llama-cpp-yaml-route-import-v2)

add_yaml_route_import_target(
  NAME kernel-llama-cpp-yaml-route-import-v2
  YAML_PATHS
    ${GGML_HRX_GROUPED_YAML}
  OUTPUT_DIR ${GGML_HRX_LLAMA_CPP_YAML_ROUTE_IMPORT_DIR_V2}
  ROUTING_DIR ${GGML_HRX_ROUTING_DIR_V2}
  EXPECTED_COVERAGE ${GGML_HRX_EXPECTED_YAML_ROUTE_IMPORT_COVERAGE_V2}
  EXPECTED_NATIVE_ROUTE_COUNT ${GGML_HRX_EXPECTED_NATIVE_ROUTE_COUNT_V2}
  DEPENDS
    ${GGML_HRX_RUNTIME_ASSET_TARGET}
    ${GGML_HRX_ACTIVE_ASSET_ROOT_METADATA_PATH}
)

if(GGML_HRX_ENABLE_HSA_DESCRIPTOR_TESTS)
  set(_ggml_hrx_descriptor_hsa_arg ENABLE_HSA_EXECUTION)
else()
  set(_ggml_hrx_descriptor_hsa_arg)
endif()

add_yaml_route_import_descriptor_tests(
  NAME llama-cpp-v2
  GENERATED_IMPORT_DIR ${GGML_HRX_LLAMA_CPP_YAML_ROUTE_IMPORT_DIR_V2}
  GROUPED_YAML ${GGML_HRX_GROUPED_YAML}
  ROUTING_DIR ${GGML_HRX_ROUTING_DIR_V2}
  KERNEL_DIR ${GGML_HRX_KERNEL_DIR_V2}
  ${_ggml_hrx_descriptor_hsa_arg}
)

add_custom_target(
  kernel-benchmark-llama-cpp-v2-materialize
  DEPENDS
    kernel-prepare-llama-cpp-v2-generated
)

set(GGML_HRX_LLAMA_CPP_KERNEL_BENCHMARK_DIR ${CMAKE_BINARY_DIR}/benchmarks/loom-kernels)
set(GGML_HRX_LLAMA_CPP_KERNEL_BENCHMARK_INDEX
  ${GGML_HRX_LLAMA_CPP_KERNEL_BENCHMARK_DIR}/catalog/v2/index.json
)
set(GGML_HRX_LLAMA_CPP_KERNEL_BENCHMARK_COMMAND
  ${Python3_EXECUTABLE}
  ${CMAKE_SOURCE_DIR}/tests/infra/materialize_loom_benchmarks.py
  --prepare-root ${CMAKE_CURRENT_BINARY_DIR}/artifacts/kernel-prepare-llama-cpp-v2
  --repo-root ${CMAKE_SOURCE_DIR}
  --asset-root ${GGML_HRX_ASSET_ROOT}
  --output-root ${GGML_HRX_LLAMA_CPP_KERNEL_BENCHMARK_DIR}
)
if(GGML_HRX_TOOL_DIR)
  list(APPEND GGML_HRX_LLAMA_CPP_KERNEL_BENCHMARK_COMMAND
    --tool-dir ${GGML_HRX_TOOL_DIR}
  )
endif()

set(GGML_HRX_LLAMA_CPP_KERNEL_BENCHMARK_DEPENDS
  kernel-benchmark-llama-cpp-v2-materialize
  ${GGML_HRX_LLAMA_CPP_YAML_ROUTE_IMPORT_DIR_V2}/route-import.stamp
  ${CMAKE_SOURCE_DIR}/tests/infra/materialize_loom_benchmarks.py
  ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/benchmarking/common.py
  ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/benchmarking/discovery.py
  ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/benchmarking/materialize.py
  ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/benchmarking/workbench.py
)
if(GGML_HRX_TOOL_BUILD_TARGET)
  list(APPEND GGML_HRX_LLAMA_CPP_KERNEL_BENCHMARK_DEPENDS
    ${GGML_HRX_TOOL_BUILD_TARGET}
  )
endif()

add_custom_command(
  OUTPUT ${GGML_HRX_LLAMA_CPP_KERNEL_BENCHMARK_INDEX}
  COMMAND ${GGML_HRX_LLAMA_CPP_KERNEL_BENCHMARK_COMMAND}
  DEPENDS
    ${GGML_HRX_LLAMA_CPP_KERNEL_BENCHMARK_DEPENDS}
  COMMENT "Generating Loom kernel benchmark scripts"
  VERBATIM
)

add_custom_target(
  kernel-benchmark-llama-cpp-v2-scripts
  DEPENDS
    ${GGML_HRX_LLAMA_CPP_KERNEL_BENCHMARK_INDEX}
)

add_custom_target(
  kernel-benchmark-llama-cpp-v2
  DEPENDS
    kernel-benchmark-llama-cpp-v2-scripts
)
