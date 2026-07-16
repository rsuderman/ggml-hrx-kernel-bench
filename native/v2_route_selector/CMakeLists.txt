set(_ggml_hrx_v2_route_selector_generated_dir
  ${CMAKE_CURRENT_BINARY_DIR}/generated
)
set(_ggml_hrx_v2_route_selector_generated_include
  ${_ggml_hrx_v2_route_selector_generated_dir}/ggml_hrx_v2_routes.inc.cpp
)
set(_ggml_hrx_v2_route_selector_route_template
  ${CMAKE_CURRENT_SOURCE_DIR}/templates/route_descriptor.inc.cpp.tmpl
)
set(_ggml_hrx_v2_route_selector_catalog_module
  ${PROJECT_SOURCE_DIR}/src/ggml_hrx_kernel_bench/routing/v2/catalog.py
)
set(_ggml_hrx_v2_route_selector_models_module
  ${PROJECT_SOURCE_DIR}/src/ggml_hrx_kernel_bench/routing/v2/models.py
)

add_custom_command(
  OUTPUT ${_ggml_hrx_v2_route_selector_generated_include}
  COMMAND
    ${Python3_EXECUTABLE}
    ${PROJECT_SOURCE_DIR}/scripts/generate_v2_route_selector.py
    --routing-dir ${GGML_HRX_ASSET_ROOT}/catalog/v2
    --output ${_ggml_hrx_v2_route_selector_generated_include}
    --all
  DEPENDS
    ${PROJECT_SOURCE_DIR}/scripts/generate_v2_route_selector.py
    ${_ggml_hrx_v2_route_selector_route_template}
    ${_ggml_hrx_v2_route_selector_catalog_module}
    ${_ggml_hrx_v2_route_selector_models_module}
    ${GGML_HRX_RUNTIME_ASSET_TARGET}
    ${GGML_HRX_ASSET_ROOT}/.materialized.stamp
  COMMENT "Generating native v2 route descriptors"
  VERBATIM
)

add_custom_target(
  ggml_hrx_v2_route_selector_generated
  DEPENDS ${_ggml_hrx_v2_route_selector_generated_include}
)

set_source_files_properties(
  ${_ggml_hrx_v2_route_selector_generated_include}
  PROPERTIES
    GENERATED TRUE
    HEADER_FILE_ONLY TRUE
)

add_library(
  ggml_hrx_v2_route_selector_core STATIC
  src/selector.cpp
  ${_ggml_hrx_v2_route_selector_generated_include}
)
add_dependencies(
  ggml_hrx_v2_route_selector_core
  ggml_hrx_v2_route_selector_generated
)
target_include_directories(
  ggml_hrx_v2_route_selector_core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE
    ${_ggml_hrx_v2_route_selector_generated_dir}
)
target_compile_features(ggml_hrx_v2_route_selector_core PUBLIC cxx_std_17)
