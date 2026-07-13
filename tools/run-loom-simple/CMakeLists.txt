add_library(ggml_hrx_run_loom_simple_lib
  run_loom_simple.cpp
)
target_include_directories(ggml_hrx_run_loom_simple_lib
  PUBLIC
    ${CMAKE_SOURCE_DIR}
)
target_compile_features(ggml_hrx_run_loom_simple_lib PUBLIC cxx_std_17)

add_executable(ggml-hrx-run-loom-simple
  main.cpp
)
target_link_libraries(ggml-hrx-run-loom-simple
  PRIVATE
    ggml_hrx_run_loom_simple_lib
)

if(BUILD_TESTING)
  add_executable(ggml_hrx_run_loom_simple_unit_tests
    run_loom_simple_test.cpp
  )
  target_link_libraries(ggml_hrx_run_loom_simple_unit_tests
    PRIVATE
      ggml_hrx_run_loom_simple_lib
  )
  add_test(
    NAME run-loom-simple-unit
    COMMAND ggml_hrx_run_loom_simple_unit_tests
  )
endif()
