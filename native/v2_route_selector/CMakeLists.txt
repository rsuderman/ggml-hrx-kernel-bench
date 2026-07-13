add_library(
  ggml_hrx_v2_route_selector_core STATIC
  src/selector.cpp
)
target_include_directories(
  ggml_hrx_v2_route_selector_core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_features(ggml_hrx_v2_route_selector_core PUBLIC cxx_std_17)
set_target_properties(
  ggml_hrx_v2_route_selector_core
  PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

nanobind_add_module(
  _ggml_hrx_v2_selector_native
  NB_STATIC
  src/python_module.cpp
)
target_compile_features(_ggml_hrx_v2_selector_native PRIVATE cxx_std_17)
target_link_libraries(
  _ggml_hrx_v2_selector_native
  PRIVATE
    ggml_hrx_v2_route_selector_core
)

# A generator expression prevents multi-config generators from adding a
# configuration-specific directory below build/python.
set(_ggml_hrx_native_selector_output "$<1:${CMAKE_BINARY_DIR}/python>")
set_target_properties(
  _ggml_hrx_v2_selector_native
  PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${_ggml_hrx_native_selector_output}"
    RUNTIME_OUTPUT_DIRECTORY "${_ggml_hrx_native_selector_output}"
)

# Keep the extension's type information next to the built module so editors and
# static analyzers resolve it through the same build/python path.
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/python")
configure_file(
  "${PROJECT_SOURCE_DIR}/src/_ggml_hrx_v2_selector_native.pyi"
  "${CMAKE_BINARY_DIR}/python/_ggml_hrx_v2_selector_native.pyi"
  COPYONLY
)
