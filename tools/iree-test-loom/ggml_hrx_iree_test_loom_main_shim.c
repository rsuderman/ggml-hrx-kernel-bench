// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string.h>

#include "iree/base/api.h"
#include "iree/base/internal/path.h"
#include "iree/io/stdio_stream.h"
#include "loom/tooling/io/file.h"
#include "loom/tooling/testbench/executor.h"
#include "loom/tools/iree-test-loom/main.h"

typedef struct ggml_hrx_iree_test_loom_file_provider_t {
  iree_allocator_t host_allocator;
  iree_string_view_t input_dir;
} ggml_hrx_iree_test_loom_file_provider_t;

static ggml_hrx_iree_test_loom_file_provider_t
    ggml_hrx_iree_test_loom_file_provider = {0};

static bool ggml_hrx_iree_test_loom_path_is_absolute(iree_string_view_t path) {
  if (iree_string_view_is_empty(path)) {
    return false;
  }
  if (path.data[0] == '/' || path.data[0] == '\\') {
    return true;
  }
  if (path.size >= 3 && path.data[1] == ':' &&
      (path.data[2] == '/' || path.data[2] == '\\')) {
    const char drive = path.data[0];
    return (drive >= 'a' && drive <= 'z') || (drive >= 'A' && drive <= 'Z');
  }
  return false;
}

static iree_status_t ggml_hrx_iree_test_loom_dup_string_view(
    iree_string_view_t value, iree_allocator_t allocator, char** out_storage) {
  *out_storage = NULL;
  char* storage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, value.size + 1, (void**)&storage));
  iree_string_view_to_cstring(value, storage, value.size + 1);
  *out_storage = storage;
  return iree_ok_status();
}

static iree_status_t ggml_hrx_iree_test_loom_resolve_path(
    const ggml_hrx_iree_test_loom_file_provider_t* provider,
    iree_string_view_t path, char** out_path) {
  if (ggml_hrx_iree_test_loom_path_is_absolute(path) ||
      iree_string_view_is_empty(provider->input_dir)) {
    return ggml_hrx_iree_test_loom_dup_string_view(
        path, provider->host_allocator, out_path);
  }
  return iree_file_path_join(provider->input_dir, path, provider->host_allocator,
                             out_path);
}

static iree_status_t ggml_hrx_iree_test_loom_open_file_for_read(
    void* user_data, iree_string_view_t path, iree_io_stream_t** out_stream) {
  *out_stream = NULL;
  if (loom_tooling_file_path_is_stdio(path)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "check.file.read paths must name a file");
  }
  const ggml_hrx_iree_test_loom_file_provider_t* provider =
      (const ggml_hrx_iree_test_loom_file_provider_t*)user_data;
  char* resolved_path = NULL;
  IREE_RETURN_IF_ERROR(
      ggml_hrx_iree_test_loom_resolve_path(provider, path, &resolved_path));
  iree_status_t status = iree_io_stdio_stream_open(
      IREE_IO_STDIO_STREAM_MODE_READ, iree_make_cstring_view(resolved_path),
      provider->host_allocator, out_stream);
  iree_allocator_free(provider->host_allocator, resolved_path);
  return status;
}

static iree_status_t ggml_hrx_iree_test_loom_open_file_for_write(
    void* user_data, iree_string_view_t path, iree_io_stream_t** out_stream) {
  *out_stream = NULL;
  if (loom_tooling_file_path_is_stdio(path)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "check.file.write paths must name a file");
  }
  ggml_hrx_iree_test_loom_file_provider_t* provider =
      (ggml_hrx_iree_test_loom_file_provider_t*)user_data;
  char* resolved_path = NULL;
  IREE_RETURN_IF_ERROR(
      ggml_hrx_iree_test_loom_resolve_path(provider, path, &resolved_path));
  const iree_string_view_t resolved_path_view =
      iree_make_cstring_view(resolved_path);
  IREE_RETURN_IF_ERROR(loom_tooling_create_directory_if_needed(
      iree_file_path_dirname(resolved_path_view), provider->host_allocator));
  iree_status_t status = iree_io_stdio_stream_open(
      IREE_IO_STDIO_STREAM_MODE_WRITE | IREE_IO_STDIO_STREAM_MODE_DISCARD,
      resolved_path_view, provider->host_allocator, out_stream);
  iree_allocator_free(provider->host_allocator, resolved_path);
  return status;
}

static void ggml_hrx_iree_test_loom_case_execution_options_initialize(
    loom_testbench_case_execution_options_t* out_options) {
  loom_testbench_case_execution_options_initialize(out_options);
  ggml_hrx_iree_test_loom_file_provider.host_allocator =
      out_options->materializer.host_allocator;
  out_options->materializer.open_read_file =
      (loom_testbench_file_open_callback_t){
          .fn = ggml_hrx_iree_test_loom_open_file_for_read,
          .user_data = &ggml_hrx_iree_test_loom_file_provider,
      };
  out_options->materializer.open_write_file =
      (loom_testbench_file_open_callback_t){
          .fn = ggml_hrx_iree_test_loom_open_file_for_write,
          .user_data = &ggml_hrx_iree_test_loom_file_provider,
      };
}

#define iree_test_loom_main ggml_hrx_iree_test_loom_main_impl
#define loom_testbench_case_execution_options_initialize \
  ggml_hrx_iree_test_loom_case_execution_options_initialize
#include GGML_HRX_UPSTREAM_IREE_TEST_LOOM_MAIN_C
#undef loom_testbench_case_execution_options_initialize
#undef iree_test_loom_main

int iree_test_loom_main(int argc, char** argv,
                        const iree_test_loom_configuration_t* configuration) {
  iree_string_view_t filename = IREE_SV("<stdin>");
  if (argc >= 2 && strcmp(argv[1], "-") != 0) {
    filename = iree_make_cstring_view(argv[1]);
  }
  ggml_hrx_iree_test_loom_file_provider.host_allocator =
      iree_allocator_system();
  ggml_hrx_iree_test_loom_file_provider.input_dir =
      iree_string_view_equal(filename, IREE_SV("<stdin>"))
          ? iree_string_view_empty()
          : iree_file_path_dirname(filename);
  return ggml_hrx_iree_test_loom_main_impl(argc, argv, configuration);
}
