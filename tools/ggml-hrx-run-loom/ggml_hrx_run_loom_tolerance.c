// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tools/ggml-hrx-run-loom/ggml_hrx_run_loom_tolerance.h"

typedef struct ggml_hrx_run_loom_expected_buffer_tolerances_t {
  ggml_hrx_run_loom_expected_buffer_tolerance_t
      values[GGML_HRX_RUN_LOOM_MAX_EXPECTED_BUFFER_TOLERANCE_COUNT];
  iree_host_size_t count;
} ggml_hrx_run_loom_expected_buffer_tolerances_t;

static ggml_hrx_run_loom_expected_buffer_tolerances_t
    ggml_hrx_run_loom_expected_buffer_tolerances = {0};

void ggml_hrx_run_loom_expected_buffer_tolerances_reset(void) {
  ggml_hrx_run_loom_expected_buffer_tolerances =
      (ggml_hrx_run_loom_expected_buffer_tolerances_t){0};
}

iree_status_t
ggml_hrx_run_loom_expected_buffer_tolerance_append(iree_string_view_t value) {
  if (ggml_hrx_run_loom_expected_buffer_tolerances.count >=
      GGML_HRX_RUN_LOOM_MAX_EXPECTED_BUFFER_TOLERANCE_COUNT) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "too many expected HAL binding tolerances; maximum is %d",
        GGML_HRX_RUN_LOOM_MAX_EXPECTED_BUFFER_TOLERANCE_COUNT);
  }
  iree_string_view_t atol_text;
  iree_string_view_t rtol_text;
  iree_string_view_split(value, ',', &atol_text, &rtol_text);
  float atol = 0.0f;
  float rtol = 0.0f;
  if (iree_string_view_is_empty(atol_text) ||
      iree_string_view_is_empty(rtol_text) ||
      !iree_string_view_atof(atol_text, &atol) ||
      !iree_string_view_atof(rtol_text, &rtol) || atol < 0.0 || rtol < 0.0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "invalid expected HAL binding tolerance '%.*s'; expected "
        "`atol,rtol` with non-negative floating point values",
        (int)value.size, value.data);
  }
  const iree_host_size_t index =
      ggml_hrx_run_loom_expected_buffer_tolerances.count++;
  ggml_hrx_run_loom_expected_buffer_tolerances.values[index] =
      (ggml_hrx_run_loom_expected_buffer_tolerance_t){
          .absolute_tolerance = (double)atol,
          .relative_tolerance = (double)rtol,
      };
  return iree_ok_status();
}

iree_host_size_t ggml_hrx_run_loom_expected_buffer_tolerance_count(void) {
  return ggml_hrx_run_loom_expected_buffer_tolerances.count;
}

bool ggml_hrx_run_loom_expected_buffer_tolerance_at(
    iree_host_size_t index,
    ggml_hrx_run_loom_expected_buffer_tolerance_t *out_tolerance) {
  if (index >= ggml_hrx_run_loom_expected_buffer_tolerances.count) {
    return false;
  }
  *out_tolerance = ggml_hrx_run_loom_expected_buffer_tolerances.values[index];
  return true;
}

iree_hal_buffer_equality_t ggml_hrx_run_loom_expected_buffer_tolerance_equality(
    const ggml_hrx_run_loom_expected_buffer_tolerance_t *tolerance) {
  if (tolerance == NULL || (tolerance->absolute_tolerance == 0.0 &&
                            tolerance->relative_tolerance == 0.0)) {
    return (iree_hal_buffer_equality_t){
        .mode = IREE_HAL_BUFFER_EQUALITY_EXACT,
    };
  }
  return (iree_hal_buffer_equality_t){
      .mode = IREE_HAL_BUFFER_EQUALITY_APPROXIMATE,
      .f16_atol = (float)tolerance->absolute_tolerance,
      .f32_atol = (float)tolerance->absolute_tolerance,
      .f64_atol = tolerance->absolute_tolerance,
      .bf16_atol = (float)tolerance->absolute_tolerance,
      .rtol = tolerance->relative_tolerance,
  };
}
