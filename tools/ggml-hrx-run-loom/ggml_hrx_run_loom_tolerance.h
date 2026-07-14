// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef GGML_HRX_TOOLS_GGML_HRX_RUN_LOOM_TOLERANCE_H_
#define GGML_HRX_TOOLS_GGML_HRX_RUN_LOOM_TOLERANCE_H_

#include <stdbool.h>

#include "iree/base/api.h"
#include "iree/tooling/buffer_view_matchers.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  GGML_HRX_RUN_LOOM_MAX_EXPECTED_BUFFER_TOLERANCE_COUNT = 64,
};

typedef struct ggml_hrx_run_loom_expected_buffer_tolerance_t {
  double absolute_tolerance;
  double relative_tolerance;
} ggml_hrx_run_loom_expected_buffer_tolerance_t;

void ggml_hrx_run_loom_expected_buffer_tolerances_reset(void);

iree_status_t
ggml_hrx_run_loom_expected_buffer_tolerance_append(iree_string_view_t value);

iree_host_size_t ggml_hrx_run_loom_expected_buffer_tolerance_count(void);

bool ggml_hrx_run_loom_expected_buffer_tolerance_at(
    iree_host_size_t index,
    ggml_hrx_run_loom_expected_buffer_tolerance_t *out_tolerance);

iree_hal_buffer_equality_t ggml_hrx_run_loom_expected_buffer_tolerance_equality(
    const ggml_hrx_run_loom_expected_buffer_tolerance_t *tolerance);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // GGML_HRX_TOOLS_GGML_HRX_RUN_LOOM_TOLERANCE_H_
