#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace ggml_hrx::run_loom_simple {

enum class BindingKind {
  kInput,
  kOutput,
};

enum class DType {
  kF32,
  kF16,
};

enum class CheckMode {
  kClose,
};

struct Binding {
  int position = -1;
  BindingKind kind = BindingKind::kInput;
  DType dtype = DType::kF32;
  std::size_t elements = 0;
  std::string path;
};

struct ScalarArg {
  int position = -1;
  DType dtype = DType::kF32;
  std::string value;
};

struct Expectation {
  int position = -1;
  CheckMode mode = CheckMode::kClose;
  std::string path;
  double atol = 0.0;
  double rtol = 0.0;
};

struct Invocation {
  std::string kernel_path;
  std::string root_symbol;
  std::string target;
  std::string output_path;
  std::string loom_link_path = "loom-link";
  std::string linked_kernel_output;
  std::string iree_run_loom_path = "iree-run-loom";
  std::string workgroup_count;
  bool emit_iree_run_loom_command = false;
  bool execute_iree_run_loom_command = false;
  std::vector<std::pair<std::string, std::string>> configs;
  std::vector<ScalarArg> scalars;
  std::vector<Binding> bindings;
  std::vector<Expectation> expectations;
};

struct ParseResult {
  std::optional<Invocation> invocation;
  std::vector<std::string> errors;
};

struct F32Tensor {
  std::vector<float> values;
};

struct F32NpyLoadResult {
  std::optional<F32Tensor> tensor;
  std::string error;
};

struct NpyLoadResult {
  bool loaded = false;
  std::string error;
};

struct CloseCompareResult {
  bool passed = false;
  std::size_t compared_elements = 0;
  double max_abs_error = 0.0;
  double max_rel_error = 0.0;
  std::optional<std::size_t> first_failing_index;
  float first_failing_actual = 0.0f;
  float first_failing_expected = 0.0f;
  double first_failing_abs_error = 0.0;
  double first_failing_rel_error = 0.0;
  std::string error;
};

struct IreeRunLoomCommandResult {
  std::optional<std::vector<std::string>> loom_link_args;
  std::optional<std::vector<std::string>> args;
  std::string error;
};

ParseResult ParseArgs(const std::vector<std::string> &args);
F32NpyLoadResult LoadF32Npy1D(const std::string &path,
                              std::size_t expected_elements);
NpyLoadResult ValidateNpyStorage1D(const std::string &path, DType dtype,
                                   std::size_t expected_elements);
CloseCompareResult CompareClose(const std::vector<float> &actual,
                                const std::vector<float> &expected, double atol,
                                double rtol);
IreeRunLoomCommandResult BuildIreeRunLoomCommand(const Invocation &invocation);

std::string ToString(BindingKind kind);
std::string ToString(DType dtype);
std::string ToString(CheckMode mode);
std::string BackendForTarget(const std::string &target);
std::string RenderResultJson(const Invocation &invocation);
std::string RenderNotRunResultJson(const Invocation &invocation);

} // namespace ggml_hrx::run_loom_simple
