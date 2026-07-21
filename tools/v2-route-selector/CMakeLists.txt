include(GgmlHrxNlohmannJson)
ggml_hrx_ensure_nlohmann_json()

add_library(ggml_hrx_v2_route_selector_cli_lib STATIC
  v2_route_selector_cli.cpp
)
target_include_directories(ggml_hrx_v2_route_selector_cli_lib
  PUBLIC
    ${PROJECT_SOURCE_DIR}
)
target_compile_features(ggml_hrx_v2_route_selector_cli_lib PUBLIC cxx_std_17)
target_link_libraries(ggml_hrx_v2_route_selector_cli_lib
  PRIVATE
    ggml_hrx_v2_route_query_parser
    ggml_hrx_v2_route_selector_core
)

add_executable(ggml-hrx-v2-route-selector
  main.cpp
)
target_compile_features(ggml-hrx-v2-route-selector PRIVATE cxx_std_17)
target_link_libraries(ggml-hrx-v2-route-selector
  PRIVATE
    ggml_hrx_v2_route_selector_cli_lib
)

add_executable(ggml_hrx_v2_route_selector_cli_unit_tests
  v2_route_selector_cli_test.cpp
)
target_compile_features(
  ggml_hrx_v2_route_selector_cli_unit_tests
  PRIVATE
    cxx_std_17
)
target_link_libraries(ggml_hrx_v2_route_selector_cli_unit_tests
  PRIVATE
    ggml_hrx_v2_route_selector_cli_lib
)
add_test(
  NAME v2-route-selector-cli-unit
  COMMAND ggml_hrx_v2_route_selector_cli_unit_tests
)
add_test(
  NAME v2-route-selector-cli-smoke
  COMMAND
    ggml-hrx-v2-route-selector
    --input ${CMAKE_CURRENT_SOURCE_DIR}/testdata/abs_f32_contiguous_4d.json
    --expect-route abs_f32_contiguous_4d
)
