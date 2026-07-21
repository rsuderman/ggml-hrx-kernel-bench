include_guard(GLOBAL)

function(_ggml_hrx_add_local_run_loom_target out_target hrx_source_dir)
  set(_ggml_hrx_local_target ggml_hrx_run_loom)
  if(TARGET ${_ggml_hrx_local_target})
    set(${out_target} ${_ggml_hrx_local_target} PARENT_SCOPE)
    return()
  endif()

  if(NOT CMAKE_C_COMPILER_LOADED)
    enable_language(C)
  endif()

  set(_ggml_hrx_platform_deps loom::tooling::execution::execution_provider)
  set(_ggml_hrx_platform_defs)
  if(TARGET loom::tooling::target::amdgpu::execution::provider)
    list(APPEND _ggml_hrx_platform_defs GGML_HRX_RUN_LOOM_HAVE_AMDGPU=1)
    list(APPEND _ggml_hrx_platform_deps
      loom::tooling::target::amdgpu::execution::provider
    )
  endif()
  if(TARGET loom::tooling::execution::ireevm::provider)
    list(APPEND _ggml_hrx_platform_defs GGML_HRX_RUN_LOOM_HAVE_IREE_VM=1)
    list(APPEND _ggml_hrx_platform_deps loom::tooling::execution::ireevm::provider)
  endif()
  if(TARGET loom::tooling::target::spirv::execution::provider)
    list(APPEND _ggml_hrx_platform_defs GGML_HRX_RUN_LOOM_HAVE_SPIRV=1)
    list(APPEND _ggml_hrx_platform_deps
      loom::tooling::target::spirv::execution::provider
    )
  endif()

  add_executable(${_ggml_hrx_local_target}
    "${CMAKE_SOURCE_DIR}/tools/ggml-hrx-run-loom/ggml_hrx_run_loom.c"
    "${CMAKE_SOURCE_DIR}/tools/ggml-hrx-run-loom/ggml_hrx_run_loom_main.c"
    "${CMAKE_SOURCE_DIR}/tools/ggml-hrx-run-loom/ggml_hrx_hal_invocation.c"
    "${CMAKE_SOURCE_DIR}/tools/ggml-hrx-run-loom/ggml_hrx_run_loom_tolerance.c"
  )
  target_compile_definitions(${_ggml_hrx_local_target} PRIVATE
    ${_ggml_hrx_platform_defs}
  )
  target_include_directories(${_ggml_hrx_local_target} PRIVATE
    "${CMAKE_SOURCE_DIR}"
  )
  target_link_libraries(${_ggml_hrx_local_target} PRIVATE
    iree::base
    iree::base::tooling::flags
    iree::hal
    iree::io::file_handle
    iree::tooling::buffer_view_matchers
    iree::tooling::value_io
    loom::error::diagnostic
    loom::ir
    loom::sanitizer::options
    loom::target::launch
    loom::target::provider
    loom::target::types
    loom::tooling::cli::help
    loom::tooling::compile::pipeline
    loom::tooling::context
    loom::tooling::execution::compile_options
    loom::tooling::execution::compile_report_capture
    loom::tooling::execution::execution_backend
    loom::tooling::execution::hal::artifact
    loom::tooling::execution::hal::runtime
    loom::tooling::execution::one_shot
    loom::tooling::execution::session
    loom::tooling::io::file
    ${_ggml_hrx_platform_deps}
  )
  set_target_properties(${_ggml_hrx_local_target} PROPERTIES
    OUTPUT_NAME ggml-hrx-run-loom
  )
  set(${out_target} ${_ggml_hrx_local_target} PARENT_SCOPE)
endfunction()

function(_ggml_hrx_add_local_iree_test_loom_target out_target hrx_source_dir)
  set(_ggml_hrx_local_target ggml_hrx_iree_test_loom)
  if(TARGET ${_ggml_hrx_local_target})
    set(${out_target} ${_ggml_hrx_local_target} PARENT_SCOPE)
    return()
  endif()

  if(NOT CMAKE_C_COMPILER_LOADED)
    enable_language(C)
  endif()

  set(_ggml_hrx_platform_deps loom::tooling::execution::execution_provider)
  set(_ggml_hrx_platform_defs)
  if(TARGET loom::tooling::target::amdgpu::artifact_provider)
    list(APPEND _ggml_hrx_platform_defs IREE_TEST_LOOM_HAVE_AMDGPU=1)
    list(APPEND _ggml_hrx_platform_deps
      loom::target::arch::amdgpu::provider
      loom::tooling::target::amdgpu::artifact_provider
      loom::tooling::target::amdgpu::testbench_requirements
    )
  endif()
  if(TARGET loom::tooling::execution::ireevm::provider)
    list(APPEND _ggml_hrx_platform_defs IREE_TEST_LOOM_HAVE_IREE_VM=1)
    list(APPEND _ggml_hrx_platform_deps loom::tooling::execution::ireevm::provider)
  endif()
  if(TARGET loom::tooling::target::spirv::artifact_provider)
    list(APPEND _ggml_hrx_platform_defs IREE_TEST_LOOM_HAVE_SPIRV=1)
    list(APPEND _ggml_hrx_platform_deps
      loom::target::arch::spirv::provider
      loom::tooling::target::spirv::artifact_provider
      loom::tooling::target::spirv::testbench_requirements
    )
  endif()

  add_executable(${_ggml_hrx_local_target}
    "${CMAKE_SOURCE_DIR}/tools/iree-test-loom/ggml_hrx_iree_test_loom_main_shim.c"
    "${hrx_source_dir}/loom/src/loom/tools/iree-test-loom/iree-test-loom.c"
  )
  target_compile_definitions(${_ggml_hrx_local_target} PRIVATE
    ${_ggml_hrx_platform_defs}
    GGML_HRX_UPSTREAM_IREE_TEST_LOOM_MAIN_C="${hrx_source_dir}/loom/src/loom/tools/iree-test-loom/main.c"
  )
  target_link_libraries(${_ggml_hrx_local_target} PRIVATE
    iree::base
    iree::base::internal::path
    iree::base::tooling::flags
    iree::io::file_handle
    iree::io::stream
    loom::sanitizer::options
    loom::target::provider
    loom::tooling::cli::help
    loom::tooling::config
    loom::tooling::context
    loom::tooling::execution::hal::artifact
    loom::tooling::execution::hal::testbench_actual
    loom::tooling::execution::session
    loom::tooling::io::file
    loom::tooling::testbench::device_event
    loom::tooling::testbench::executor
    loom::tooling::testbench::issue_report
    loom::tooling::testbench::reference
    loom::tooling::testbench::requirements
    loom::util::json
    loom::util::stream
    ${_ggml_hrx_platform_deps}
  )
  set_target_properties(${_ggml_hrx_local_target} PROPERTIES
    OUTPUT_NAME ggml-hrx-iree-test-loom
  )
  set(${out_target} ${_ggml_hrx_local_target} PARENT_SCOPE)
endfunction()

function(_ggml_hrx_normalize_hrx_systems_source_dir out_var candidate_dir)
  if(NOT candidate_dir)
    set(${out_var} "" PARENT_SCOPE)
    return()
  endif()

  get_filename_component(_ggml_hrx_candidate_dir "${candidate_dir}" ABSOLUTE)
  set(_ggml_hrx_direct_root "${_ggml_hrx_candidate_dir}")
  set(_ggml_hrx_nested_hrx_root "${_ggml_hrx_candidate_dir}/hrx")

  if(EXISTS "${_ggml_hrx_direct_root}/CMakeLists.txt" AND EXISTS "${_ggml_hrx_direct_root}/loom")
    set(${out_var} "${_ggml_hrx_direct_root}" PARENT_SCOPE)
    return()
  endif()

  if(EXISTS "${_ggml_hrx_nested_hrx_root}/CMakeLists.txt" AND EXISTS "${_ggml_hrx_nested_hrx_root}/loom")
    set(${out_var} "${_ggml_hrx_nested_hrx_root}" PARENT_SCOPE)
    return()
  endif()

  message(FATAL_ERROR
    "GGML_HRX_HRX_SYSTEMS_SOURCE_DIR must point to the hrx-systems source tree "
    "that contains the HRX CMake project and its loom/ subtree. Supported layouts "
    "are either <dir>/CMakeLists.txt with <dir>/loom/, or <dir>/hrx/CMakeLists.txt "
    "with <dir>/hrx/loom/. Received: ${candidate_dir}"
  )
endfunction()

function(ggml_hrx_configure_loom_tools)
  option(
    GGML_HRX_BUILD_LOOM_TOOLS
    "Build loom-link, loom-compile, ggml-hrx-run-loom, and iree-test-loom from an hrx-systems source tree."
    ON
  )

  set(_ggml_hrx_hrx_systems_source_dir_default "")
  if(DEFINED ENV{GGML_HRX_HRX_SYSTEMS_SOURCE_DIR} AND NOT "$ENV{GGML_HRX_HRX_SYSTEMS_SOURCE_DIR}" STREQUAL "")
    set(_ggml_hrx_hrx_systems_source_dir_default "$ENV{GGML_HRX_HRX_SYSTEMS_SOURCE_DIR}")
  endif()
  set(
    GGML_HRX_HRX_SYSTEMS_SOURCE_DIR
    "${_ggml_hrx_hrx_systems_source_dir_default}"
    CACHE PATH
    "Path to the hrx-systems source tree that contains the HRX CMake project and loom sources."
  )
  set(
    GGML_HRX_HRX_SYSTEMS_BINARY_DIR
    "${CMAKE_BINARY_DIR}/_deps/hrx-systems-build"
    CACHE PATH
    "In-tree binary directory used for the hrx-systems subdirectory build."
  )
  set(
    GGML_HRX_TOOLS_DIR
    "${CMAKE_BINARY_DIR}/tools"
    CACHE PATH
    "Bench-owned staging directory for loom tool executables."
  )

  set(_ggml_hrx_rocm_path_default "")
  if(DEFINED ENV{GGML_HRX_ROCM_PATH} AND NOT "$ENV{GGML_HRX_ROCM_PATH}" STREQUAL "")
    set(_ggml_hrx_rocm_path_default "$ENV{GGML_HRX_ROCM_PATH}")
  endif()
  set(
    GGML_HRX_ROCM_PATH
    "${_ggml_hrx_rocm_path_default}"
    CACHE PATH
    "Optional ROCm root for runtime tests and the integrated loom utility build."
  )

  set(GGML_HRX_BUILT_TOOL_DIR "${GGML_HRX_TOOLS_DIR}")
  set(GGML_HRX_TOOL_BUILD_TARGET "")

  if(GGML_HRX_BUILD_LOOM_TOOLS)
    if(NOT GGML_HRX_HRX_SYSTEMS_SOURCE_DIR)
      message(FATAL_ERROR
        "GGML_HRX_BUILD_LOOM_TOOLS=ON requires GGML_HRX_HRX_SYSTEMS_SOURCE_DIR. "
        "Point it at the hrx-systems checkout that contains the HRX CMake project "
        "and loom sources, for example:\n"
        "  cmake -S ${CMAKE_SOURCE_DIR} -B ${CMAKE_BINARY_DIR} "
        "-DGGML_HRX_HRX_SYSTEMS_SOURCE_DIR=/path/to/hrx-systems"
      )
    endif()

    _ggml_hrx_normalize_hrx_systems_source_dir(
      _ggml_hrx_hrx_systems_source_dir
      "${GGML_HRX_HRX_SYSTEMS_SOURCE_DIR}"
    )

    block()
      set(IREE_BUILD_TESTS OFF)
      set(IREE_BUILD_BENCHMARKS OFF)
      set(LIBHRX_BUILD OFF)
      set(LOOM_IMPORT_MLIR OFF)
      set(LOOM_IMPORT_TILELANG OFF)
      set(LOOM_TARGET_DEFAULTS OFF)
      set(LOOM_EXECUTE_DEFAULTS OFF)
      set(LOOM_TARGET_AMDGPU ON)
      set(LOOM_TARGET_IREE_VM OFF)
      set(LOOM_TARGET_LLVMIR OFF)
      set(LOOM_TARGET_SPIRV OFF)
      set(LOOM_TARGET_WASM OFF)
      set(LOOM_TARGET_X86 OFF)
      set(LOOM_EXECUTE_IREE_HAL ON)
      set(LOOM_EXECUTE_IREE_VM OFF)
      set(IREE_HAL_DRIVER_AMDGPU ON)
      set(IREE_HAL_DRIVER_LOCAL_SYNC ON)
      set(IREE_HAL_DRIVER_LOCAL_TASK ON)
      set(IREE_HAL_DRIVER_NULL ON)
      set(IREE_HAL_DRIVER_VULKAN OFF)
      if(GGML_HRX_ROCM_PATH)
        set(IREE_ROCM_PATH "${GGML_HRX_ROCM_PATH}")
      endif()

      add_subdirectory(
        "${_ggml_hrx_hrx_systems_source_dir}"
        "${GGML_HRX_HRX_SYSTEMS_BINARY_DIR}"
        EXCLUDE_FROM_ALL
      )
    endblock()

    _ggml_hrx_add_local_iree_test_loom_target(
      _ggml_hrx_local_iree_test_loom_target
      "${_ggml_hrx_hrx_systems_source_dir}"
    )
    _ggml_hrx_add_local_run_loom_target(
      _ggml_hrx_local_run_loom_target
      "${_ggml_hrx_hrx_systems_source_dir}"
    )

    set(_ggml_hrx_stage_tool_targets
      loom_tools_loom-link_loom-link
      loom_tools_loom-compile_loom-compile
      ${_ggml_hrx_local_run_loom_target}
      ${_ggml_hrx_local_iree_test_loom_target}
    )
    set(_ggml_hrx_staged_tool_paths
      ${GGML_HRX_BUILT_TOOL_DIR}/loom-link${CMAKE_EXECUTABLE_SUFFIX}
      ${GGML_HRX_BUILT_TOOL_DIR}/loom-compile${CMAKE_EXECUTABLE_SUFFIX}
      ${GGML_HRX_BUILT_TOOL_DIR}/ggml-hrx-run-loom${CMAKE_EXECUTABLE_SUFFIX}
      ${GGML_HRX_BUILT_TOOL_DIR}/iree-test-loom${CMAKE_EXECUTABLE_SUFFIX}
    )

    add_custom_command(
      OUTPUT ${_ggml_hrx_staged_tool_paths}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${GGML_HRX_BUILT_TOOL_DIR}
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:loom_tools_loom-link_loom-link>
        ${GGML_HRX_BUILT_TOOL_DIR}/loom-link${CMAKE_EXECUTABLE_SUFFIX}
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:loom_tools_loom-compile_loom-compile>
        ${GGML_HRX_BUILT_TOOL_DIR}/loom-compile${CMAKE_EXECUTABLE_SUFFIX}
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:${_ggml_hrx_local_run_loom_target}>
        ${GGML_HRX_BUILT_TOOL_DIR}/ggml-hrx-run-loom${CMAKE_EXECUTABLE_SUFFIX}
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:${_ggml_hrx_local_iree_test_loom_target}>
        ${GGML_HRX_BUILT_TOOL_DIR}/iree-test-loom${CMAKE_EXECUTABLE_SUFFIX}
      DEPENDS
        ${_ggml_hrx_stage_tool_targets}
        $<TARGET_FILE:loom_tools_loom-link_loom-link>
        $<TARGET_FILE:loom_tools_loom-compile_loom-compile>
        $<TARGET_FILE:${_ggml_hrx_local_run_loom_target}>
        $<TARGET_FILE:${_ggml_hrx_local_iree_test_loom_target}>
      COMMENT "Staging loom tools in ${GGML_HRX_BUILT_TOOL_DIR}"
      VERBATIM
    )

    add_custom_target(ggml-hrx-loom-tools ALL
      DEPENDS ${_ggml_hrx_staged_tool_paths}
    )

    set(GGML_HRX_TOOL_BUILD_TARGET ggml-hrx-loom-tools)
  endif()

  set(_ggml_hrx_tool_dir_default "")
  if(GGML_HRX_BUILD_LOOM_TOOLS)
    set(_ggml_hrx_tool_dir_default "${GGML_HRX_BUILT_TOOL_DIR}")
  elseif(DEFINED ENV{GGML_HRX_TOOL_DIR} AND NOT "$ENV{GGML_HRX_TOOL_DIR}" STREQUAL "")
    set(_ggml_hrx_tool_dir_default "$ENV{GGML_HRX_TOOL_DIR}")
  endif()
  if(GGML_HRX_BUILD_LOOM_TOOLS)
    set(
      GGML_HRX_TOOL_DIR
      "${_ggml_hrx_tool_dir_default}"
      CACHE STRING
      "PATH-style search list containing loom-link, loom-compile, ggml-hrx-run-loom, and iree-test-loom"
      FORCE
    )
  else()
    set(
      GGML_HRX_TOOL_DIR
      "${_ggml_hrx_tool_dir_default}"
      CACHE STRING
      "PATH-style search list containing loom-link, loom-compile, ggml-hrx-run-loom, and iree-test-loom"
    )
  endif()

  set(GGML_HRX_BUILT_TOOL_DIR "${GGML_HRX_BUILT_TOOL_DIR}" PARENT_SCOPE)
  set(GGML_HRX_TOOL_BUILD_TARGET "${GGML_HRX_TOOL_BUILD_TARGET}" PARENT_SCOPE)
endfunction()
