include_guard(GLOBAL)

function(ggml_hrx_ensure_nlohmann_json)
  if(TARGET nlohmann_json::nlohmann_json)
    return()
  endif()

  find_package(nlohmann_json CONFIG QUIET)
  if(TARGET nlohmann_json::nlohmann_json)
    return()
  endif()

  include(FetchContent)
  FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  FetchContent_MakeAvailable(nlohmann_json)
endfunction()
