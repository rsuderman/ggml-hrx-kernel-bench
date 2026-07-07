include_guard(GLOBAL)

function(add_materialized_asset_target)
  set(options)
  set(one_value_args NAME OUTPUT_DIR METADATA_PATH)
  cmake_parse_arguments(GGML_HRX_MVA "${options}" "${one_value_args}" "" ${ARGN})

  if(NOT GGML_HRX_MVA_NAME)
    message(FATAL_ERROR "add_materialized_asset_target requires NAME")
  endif()
  if(NOT GGML_HRX_MVA_OUTPUT_DIR)
    message(FATAL_ERROR "add_materialized_asset_target requires OUTPUT_DIR")
  endif()
  if(NOT GGML_HRX_MVA_METADATA_PATH)
    message(FATAL_ERROR "add_materialized_asset_target requires METADATA_PATH")
  endif()

  file(GLOB_RECURSE ggml_hrx_exact_kernel_sources
    CONFIGURE_DEPENDS
    RELATIVE ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/kernels/hrx2/*.loom
    ${CMAKE_SOURCE_DIR}/kernels/v2/*.loom
  )
  list(TRANSFORM ggml_hrx_exact_kernel_sources PREPEND ${CMAKE_SOURCE_DIR}/)
  file(GLOB_RECURSE ggml_hrx_route_sources
    CONFIGURE_DEPENDS
    RELATIVE ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/catalog/hrx2/*.json
    ${CMAKE_SOURCE_DIR}/catalog/v2/*.json
  )
  list(TRANSFORM ggml_hrx_route_sources PREPEND ${CMAKE_SOURCE_DIR}/)
  file(GLOB ggml_hrx_copy_generator_sources
    CONFIGURE_DEPENDS
    RELATIVE ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/generators/*.py
  )
  list(TRANSFORM ggml_hrx_copy_generator_sources PREPEND ${CMAKE_SOURCE_DIR}/)
  file(GLOB ggml_hrx_copy_kernel_templates
    CONFIGURE_DEPENDS
    RELATIVE ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/kernels/v2/copy/*.tmpl
  )
  list(TRANSFORM ggml_hrx_copy_kernel_templates PREPEND ${CMAKE_SOURCE_DIR}/)
  file(GLOB ggml_hrx_copy_route_templates
    CONFIGURE_DEPENDS
    RELATIVE ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/catalog/v2/copy/*.tmpl
  )
  list(TRANSFORM ggml_hrx_copy_route_templates PREPEND ${CMAKE_SOURCE_DIR}/)

  set(ggml_hrx_materialize_command
    ${Python3_EXECUTABLE}
    ${CMAKE_SOURCE_DIR}/scripts/materialize_assets.py
    --output ${GGML_HRX_MVA_OUTPUT_DIR}
    --metadata-output ${GGML_HRX_MVA_METADATA_PATH}
  )

  set(ggml_hrx_asset_stamp ${GGML_HRX_MVA_OUTPUT_DIR}/.materialized.stamp)

  add_custom_command(
    OUTPUT
      ${ggml_hrx_asset_stamp}
      ${GGML_HRX_MVA_METADATA_PATH}
    COMMAND ${ggml_hrx_materialize_command}
    DEPENDS
      ${ggml_hrx_exact_kernel_sources}
      ${ggml_hrx_route_sources}
      ${ggml_hrx_copy_kernel_templates}
      ${ggml_hrx_copy_route_templates}
      ${CMAKE_SOURCE_DIR}/scripts/materialize_assets.py
      ${CMAKE_SOURCE_DIR}/src/ggml_hrx_kernel_bench/materialized_assets.py
      ${ggml_hrx_copy_generator_sources}
    COMMENT "Materializing runtime assets in ${GGML_HRX_MVA_OUTPUT_DIR}"
    VERBATIM
  )

  add_custom_target(${GGML_HRX_MVA_NAME}
    DEPENDS
      ${ggml_hrx_asset_stamp}
      ${GGML_HRX_MVA_METADATA_PATH}
  )
endfunction()
