cmake_minimum_required(VERSION 3.20)

project(ggml_hrx_kernel_bench LANGUAGES NONE)

include(CTest)

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
