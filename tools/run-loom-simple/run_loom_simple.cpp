#include "tools/run-loom-simple/run_loom_simple.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string_view>
#include <sys/wait.h>
#include <unistd.h>
#include <unordered_map>

namespace ggml_hrx::run_loom_simple {
namespace {

struct ProcessResult {
  int exit_code = -1;
  std::string output;
  std::string error;
};

struct BridgeRenderResult {
  IreeRunLoomCommandResult command;
  std::optional<ProcessResult> loom_link_execution;
  std::optional<ProcessResult> target_source_materialization;
  std::optional<ProcessResult> run_execution;
  std::string status;
};

std::vector<std::string> Split(std::string_view value, char delimiter) {
  std::vector<std::string> parts;
  std::size_t start = 0;
  while (start <= value.size()) {
    const std::size_t end = value.find(delimiter, start);
    if (end == std::string_view::npos) {
      parts.emplace_back(value.substr(start));
      break;
    }
    parts.emplace_back(value.substr(start, end - start));
    start = end + 1;
  }
  return parts;
}

bool ParseInt(std::string_view text, int *out) {
  if (text.empty()) {
    return false;
  }
  int value = 0;
  const char *begin = text.data();
  const char *end = begin + text.size();
  const auto result = std::from_chars(begin, end, value);
  if (result.ec != std::errc() || result.ptr != end) {
    return false;
  }
  *out = value;
  return true;
}

bool ParseSize(std::string_view text, std::size_t *out) {
  if (text.empty()) {
    return false;
  }
  std::size_t value = 0;
  const char *begin = text.data();
  const char *end = begin + text.size();
  const auto result = std::from_chars(begin, end, value);
  if (result.ec != std::errc() || result.ptr != end) {
    return false;
  }
  *out = value;
  return true;
}

bool ParseDouble(std::string_view text, double *out) {
  if (text.empty()) {
    return false;
  }
  std::string copy(text);
  char *parse_end = nullptr;
  const double value = std::strtod(copy.c_str(), &parse_end);
  if (parse_end != copy.c_str() + copy.size() || !std::isfinite(value)) {
    return false;
  }
  *out = value;
  return true;
}

std::optional<BindingKind> ParseBindingKind(std::string_view value) {
  if (value == "input") {
    return BindingKind::kInput;
  }
  if (value == "output") {
    return BindingKind::kOutput;
  }
  return std::nullopt;
}

std::optional<DType> ParseDType(std::string_view value) {
  if (value == "bf16") {
    return DType::kBF16;
  }
  if (value == "f32") {
    return DType::kF32;
  }
  if (value == "f16") {
    return DType::kF16;
  }
  if (value == "i32") {
    return DType::kI32;
  }
  return std::nullopt;
}

std::optional<CheckMode> ParseCheckMode(std::string_view value) {
  if (value == "close") {
    return CheckMode::kClose;
  }
  return std::nullopt;
}

std::optional<std::pair<std::string, std::string>>
ParseConfig(std::string_view raw, std::vector<std::string> *errors) {
  const std::size_t split = raw.find('=');
  if (split == std::string_view::npos || split == 0 ||
      split + 1 >= raw.size()) {
    errors->push_back("--config must have form KEY=VALUE: " + std::string(raw));
    return std::nullopt;
  }
  return std::pair<std::string, std::string>(
      std::string(raw.substr(0, split)), std::string(raw.substr(split + 1)));
}

std::optional<std::string>
ParseWorkgroupCount(std::string_view raw, std::vector<std::string> *errors) {
  const std::vector<std::string> parts = Split(raw, ',');
  if (parts.size() != 3) {
    errors->push_back("--workgroup-count must have form X,Y,Z: " +
                      std::string(raw));
    return std::nullopt;
  }
  for (const std::string &part : parts) {
    std::size_t value = 0;
    if (!ParseSize(part, &value) || value == 0) {
      errors->push_back("workgroup-count values must be positive integers: " +
                        part);
      return std::nullopt;
    }
  }
  return std::string(raw);
}

std::optional<Binding> ParseBindingFlag(std::string_view raw,
                                        std::vector<std::string> *errors) {
  const std::vector<std::string> parts = Split(raw, ':');
  if (parts.size() != 5) {
    errors->push_back(
        "--binding must have form POSITION:KIND:DTYPE:ELEMENTS:PATH: " +
        std::string(raw));
    return std::nullopt;
  }

  Binding binding;
  if (!ParseInt(parts[0], &binding.position) || binding.position < 0) {
    errors->push_back("binding position must be a non-negative integer: " +
                      parts[0]);
    return std::nullopt;
  }

  std::optional<BindingKind> kind = ParseBindingKind(parts[1]);
  if (!kind.has_value()) {
    errors->push_back("unsupported binding kind: " + parts[1]);
    return std::nullopt;
  }
  binding.kind = *kind;

  std::optional<DType> dtype = ParseDType(parts[2]);
  if (!dtype.has_value()) {
    errors->push_back("unsupported dtype: " + parts[2]);
    return std::nullopt;
  }
  binding.dtype = *dtype;

  if (!ParseSize(parts[3], &binding.elements) || binding.elements == 0) {
    errors->push_back("binding elements must be a positive integer: " +
                      parts[3]);
    return std::nullopt;
  }

  if (parts[4].empty()) {
    errors->push_back("binding path must not be empty");
    return std::nullopt;
  }
  binding.path = parts[4];
  return binding;
}

std::optional<ScalarArg> ParseScalarFlag(std::string_view raw,
                                         std::vector<std::string> *errors) {
  const std::vector<std::string> parts = Split(raw, ':');
  if (parts.size() != 3) {
    errors->push_back("--scalar must have form POSITION:DTYPE:VALUE: " +
                      std::string(raw));
    return std::nullopt;
  }

  ScalarArg scalar;
  if (!ParseInt(parts[0], &scalar.position) || scalar.position < 0) {
    errors->push_back("scalar position must be a non-negative integer: " +
                      parts[0]);
    return std::nullopt;
  }

  std::optional<DType> dtype = ParseDType(parts[1]);
  if (!dtype.has_value()) {
    errors->push_back("unsupported scalar dtype: " + parts[1]);
    return std::nullopt;
  }
  scalar.dtype = *dtype;

  double parsed_value = 0.0;
  if (!ParseDouble(parts[2], &parsed_value)) {
    errors->push_back("scalar value must be a finite number: " + parts[2]);
    return std::nullopt;
  }
  scalar.value = parts[2];
  return scalar;
}

std::optional<Expectation>
ParseExpectationFlag(std::string_view raw, std::vector<std::string> *errors) {
  const std::vector<std::string> parts = Split(raw, ':');
  if (parts.size() != 5) {
    errors->push_back("--expect must have form POSITION:MODE:PATH:ATOL:RTOL: " +
                      std::string(raw));
    return std::nullopt;
  }

  Expectation expectation;
  if (!ParseInt(parts[0], &expectation.position) || expectation.position < 0) {
    errors->push_back("expect position must be a non-negative integer: " +
                      parts[0]);
    return std::nullopt;
  }

  std::optional<CheckMode> mode = ParseCheckMode(parts[1]);
  if (!mode.has_value()) {
    errors->push_back("unsupported check mode: " + parts[1]);
    return std::nullopt;
  }
  expectation.mode = *mode;

  if (parts[2].empty()) {
    errors->push_back("expect path must not be empty");
    return std::nullopt;
  }
  expectation.path = parts[2];

  if (!ParseDouble(parts[3], &expectation.atol) || expectation.atol < 0.0) {
    errors->push_back("expect atol must be a non-negative finite number: " +
                      parts[3]);
    return std::nullopt;
  }
  if (!ParseDouble(parts[4], &expectation.rtol) || expectation.rtol < 0.0) {
    errors->push_back("expect rtol must be a non-negative finite number: " +
                      parts[4]);
    return std::nullopt;
  }
  return expectation;
}

void AddMissingRequired(const Invocation &invocation,
                        std::vector<std::string> *errors) {
  if (invocation.kernel_path.empty()) {
    errors->push_back("missing required --kernel");
  }
  if (invocation.root_symbol.empty()) {
    errors->push_back("missing required --root");
  }
  if (invocation.target.empty()) {
    errors->push_back("missing required --target");
  }
  if (invocation.output_path.empty()) {
    errors->push_back("missing required --output");
  }
  if (invocation.bindings.empty()) {
    errors->push_back("at least one --binding is required");
  }
}

void ValidateBindingsAndExpectation(Invocation *invocation,
                                    std::vector<std::string> *errors) {
  std::sort(invocation->scalars.begin(), invocation->scalars.end(),
            [](const ScalarArg &lhs, const ScalarArg &rhs) {
              return lhs.position < rhs.position;
            });
  std::sort(invocation->bindings.begin(), invocation->bindings.end(),
            [](const Binding &lhs, const Binding &rhs) {
              return lhs.position < rhs.position;
            });

  std::unordered_map<int, BindingKind> by_position;
  for (const ScalarArg &scalar : invocation->scalars) {
    const auto [_, inserted] =
        by_position.emplace(scalar.position, BindingKind::kInput);
    if (!inserted) {
      errors->push_back("duplicate ABI position: " +
                        std::to_string(scalar.position));
    }
  }
  for (const Binding &binding : invocation->bindings) {
    const auto [_, inserted] =
        by_position.emplace(binding.position, binding.kind);
    if (!inserted) {
      errors->push_back("duplicate ABI position: " +
                        std::to_string(binding.position));
    }
  }

  std::unordered_map<int, bool> expected_positions;
  for (const Expectation &expectation : invocation->expectations) {
    const auto [_, inserted] =
        expected_positions.emplace(expectation.position, true);
    if (!inserted) {
      errors->push_back("duplicate expectation position: " +
                        std::to_string(expectation.position));
    }
    const auto found = by_position.find(expectation.position);
    if (found == by_position.end()) {
      errors->push_back("--expect references missing binding position: " +
                        std::to_string(expectation.position));
      continue;
    }
    if (found->second != BindingKind::kOutput) {
      errors->push_back("--expect references a non-output binding position: " +
                        std::to_string(expectation.position));
    }
  }
}

std::string JsonEscape(std::string_view value) {
  std::ostringstream os;
  for (const char ch : value) {
    switch (ch) {
    case '\\':
      os << "\\\\";
      break;
    case '"':
      os << "\\\"";
      break;
    case '\n':
      os << "\\n";
      break;
    case '\r':
      os << "\\r";
      break;
    case '\t':
      os << "\\t";
      break;
    default:
      os << ch;
      break;
    }
  }
  return os.str();
}

void WriteJsonStringArray(std::ostringstream *os,
                          const std::vector<std::string> &values) {
  *os << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      *os << ", ";
    }
    *os << "\"" << JsonEscape(values[i]) << "\"";
  }
  *os << "]";
}

ProcessResult ExecuteProcess(const std::vector<std::string> &args) {
  if (args.empty()) {
    return ProcessResult{-1, "", "cannot execute an empty command"};
  }

  int output_pipe[2] = {-1, -1};
  if (pipe(output_pipe) != 0) {
    return ProcessResult{-1, "",
                         "pipe failed: " + std::string(std::strerror(errno))};
  }

  const pid_t pid = fork();
  if (pid < 0) {
    const std::string error =
        "fork failed: " + std::string(std::strerror(errno));
    close(output_pipe[0]);
    close(output_pipe[1]);
    return ProcessResult{-1, "", error};
  }

  if (pid == 0) {
    close(output_pipe[0]);
    dup2(output_pipe[1], STDOUT_FILENO);
    dup2(output_pipe[1], STDERR_FILENO);
    close(output_pipe[1]);

    std::vector<char *> argv;
    argv.reserve(args.size() + 1);
    for (const std::string &arg : args) {
      argv.push_back(const_cast<char *>(arg.c_str()));
    }
    argv.push_back(nullptr);
    execvp(argv[0], argv.data());
    const std::string message =
        "exec failed: " + std::string(std::strerror(errno)) + "\n";
    write(STDERR_FILENO, message.data(), message.size());
    _exit(127);
  }

  close(output_pipe[1]);
  std::string output;
  char buffer[4096];
  while (true) {
    const ssize_t count = read(output_pipe[0], buffer, sizeof(buffer));
    if (count > 0) {
      output.append(buffer, static_cast<std::size_t>(count));
      continue;
    }
    if (count == 0) {
      break;
    }
    if (errno == EINTR) {
      continue;
    }
    const std::string error =
        "read failed: " + std::string(std::strerror(errno));
    close(output_pipe[0]);
    return ProcessResult{-1, output, error};
  }
  close(output_pipe[0]);

  int status = 0;
  if (waitpid(pid, &status, 0) < 0) {
    return ProcessResult{
        -1, output, "waitpid failed: " + std::string(std::strerror(errno))};
  }
  if (WIFEXITED(status)) {
    return ProcessResult{WEXITSTATUS(status), output, ""};
  }
  if (WIFSIGNALED(status)) {
    return ProcessResult{128 + WTERMSIG(status), output, ""};
  }
  return ProcessResult{-1, output, "process ended with unknown status"};
}

std::optional<std::string> ExtractHeaderString(std::string_view header,
                                               std::string_view key) {
  const std::string single_key = "'" + std::string(key) + "'";
  const std::string double_key = "\"" + std::string(key) + "\"";
  std::size_t key_pos = header.find(single_key);
  if (key_pos == std::string_view::npos) {
    key_pos = header.find(double_key);
  }
  if (key_pos == std::string_view::npos) {
    return std::nullopt;
  }
  const std::size_t colon = header.find(':', key_pos);
  if (colon == std::string_view::npos) {
    return std::nullopt;
  }
  const std::size_t quote = header.find_first_of("'\"", colon + 1);
  if (quote == std::string_view::npos) {
    return std::nullopt;
  }
  const char quote_ch = header[quote];
  const std::size_t end_quote = header.find(quote_ch, quote + 1);
  if (end_quote == std::string_view::npos) {
    return std::nullopt;
  }
  return std::string(header.substr(quote + 1, end_quote - quote - 1));
}

bool HeaderHasFalse(std::string_view header, std::string_view key) {
  const std::string single_key = "'" + std::string(key) + "'";
  const std::string double_key = "\"" + std::string(key) + "\"";
  std::size_t key_pos = header.find(single_key);
  if (key_pos == std::string_view::npos) {
    key_pos = header.find(double_key);
  }
  if (key_pos == std::string_view::npos) {
    return false;
  }
  const std::size_t colon = header.find(':', key_pos);
  if (colon == std::string_view::npos) {
    return false;
  }
  const std::size_t false_pos = header.find("False", colon + 1);
  return false_pos != std::string_view::npos;
}

std::optional<std::size_t> ExtractOneDimShape(std::string_view header,
                                              std::string *error) {
  const std::size_t shape_pos = header.find("'shape'");
  const std::size_t shape_pos_double = header.find("\"shape\"");
  const std::size_t key_pos =
      shape_pos == std::string_view::npos ? shape_pos_double : shape_pos;
  if (key_pos == std::string_view::npos) {
    *error = "missing shape in npy header";
    return std::nullopt;
  }
  const std::size_t open = header.find('(', key_pos);
  const std::size_t close =
      header.find(')', open == std::string_view::npos ? key_pos : open);
  if (open == std::string_view::npos || close == std::string_view::npos ||
      close <= open + 1) {
    *error = "invalid shape in npy header";
    return std::nullopt;
  }
  const std::string inside(header.substr(open + 1, close - open - 1));
  const std::vector<std::string> dims = Split(inside, ',');
  std::vector<std::string> non_empty_dims;
  for (std::string dim : dims) {
    dim.erase(
        std::remove_if(dim.begin(), dim.end(),
                       [](unsigned char ch) { return std::isspace(ch) != 0; }),
        dim.end());
    if (!dim.empty()) {
      non_empty_dims.push_back(dim);
    }
  }
  if (non_empty_dims.size() != 1) {
    *error = "only one-dimensional npy arrays are supported";
    return std::nullopt;
  }
  std::size_t elements = 0;
  if (!ParseSize(non_empty_dims[0], &elements)) {
    *error = "invalid one-dimensional npy shape";
    return std::nullopt;
  }
  return elements;
}

std::uint16_t ReadLe16(const unsigned char *bytes) {
  return static_cast<std::uint16_t>(bytes[0]) |
         (static_cast<std::uint16_t>(bytes[1]) << 8);
}

std::uint32_t ReadLe32(const unsigned char *bytes) {
  return static_cast<std::uint32_t>(bytes[0]) |
         (static_cast<std::uint32_t>(bytes[1]) << 8) |
         (static_cast<std::uint32_t>(bytes[2]) << 16) |
         (static_cast<std::uint32_t>(bytes[3]) << 24);
}

std::optional<std::string> BuildNpyStorageBindingSpec(const std::string &path,
                                                      DType dtype,
                                                      std::size_t elements,
                                                      std::string *error) {
  const NpyLoadResult loaded = ValidateNpyStorage1D(path, dtype, elements);
  if (!loaded.loaded) {
    *error = loaded.error;
    return std::nullopt;
  }
  return "&@" + path;
}

std::optional<std::string> BuildNpyExpectedBindingSpec(const std::string &path,
                                                       DType dtype,
                                                       std::size_t elements,
                                                       std::string *error) {
  const NpyLoadResult loaded = ValidateNpyStorage1D(path, dtype, elements);
  if (!loaded.loaded) {
    *error = loaded.error;
    return std::nullopt;
  }
  return "@" + path;
}

bool IsAmdGpuTargetKey(const std::string &target) {
  return target.rfind("gfx", 0) == 0;
}

std::string TargetSymbolForInvocation() {
  return "@ggml_hrx_run_loom_simple_target";
}

std::string TargetedKernelPath(const Invocation &invocation,
                               const std::string &kernel_path) {
  if (!IsAmdGpuTargetKey(invocation.target)) {
    return kernel_path;
  }
  if (!invocation.linked_kernel_output.empty()) {
    return invocation.linked_kernel_output + ".target.loom";
  }
  return invocation.output_path + ".target.loom";
}

std::string FormatDouble(double value) {
  std::ostringstream os;
  os << value;
  return os.str();
}

std::optional<std::size_t>
FindKernelDefForRoot(std::string_view source, const std::string &root_symbol) {
  const std::size_t root_pos = source.find(root_symbol);
  if (root_pos == std::string_view::npos) {
    return std::nullopt;
  }
  const std::size_t def_pos = source.rfind("kernel.def", root_pos);
  if (def_pos == std::string_view::npos) {
    return std::nullopt;
  }
  return def_pos;
}

ProcessResult MaterializeTargetedKernelSource(const Invocation &invocation,
                                              const std::string &input_path,
                                              const std::string &output_path) {
  if (!IsAmdGpuTargetKey(invocation.target)) {
    return ProcessResult{0, "", ""};
  }

  std::ifstream input(input_path);
  if (!input) {
    return ProcessResult{-1, "", "failed to open kernel source: " + input_path};
  }
  std::ostringstream buffer;
  buffer << input.rdbuf();
  if (!input.good() && !input.eof()) {
    return ProcessResult{-1, "", "failed to read kernel source: " + input_path};
  }
  std::string source = buffer.str();
  const std::string target_symbol = TargetSymbolForInvocation();

  const std::optional<std::size_t> def_pos =
      FindKernelDefForRoot(source, invocation.root_symbol);
  if (!def_pos.has_value()) {
    return ProcessResult{-1, "",
                         "failed to find kernel.def for root " +
                             invocation.root_symbol + " in " + input_path};
  }

  const std::size_t line_end = source.find('\n', *def_pos);
  const std::string_view def_line(
      source.data() + *def_pos,
      (line_end == std::string::npos ? source.size() : line_end) - *def_pos);
  if (def_line.find("target(") == std::string_view::npos) {
    const std::size_t insert_pos =
        *def_pos + std::string_view("kernel.def").size();
    source.insert(insert_pos, " target(" + target_symbol + ")");
  }
  if (source.find(std::string("amdgpu.target<") + invocation.target + "> " +
                  target_symbol) == std::string::npos) {
    source.insert(0, "amdgpu.target<" + invocation.target + "> " +
                         target_symbol + " {subgroup_size = 64}\n");
  }

  std::ofstream output(output_path);
  if (!output) {
    return ProcessResult{
        -1, "", "failed to open targeted kernel source: " + output_path};
  }
  output << source;
  if (!output) {
    return ProcessResult{
        -1, "", "failed to write targeted kernel source: " + output_path};
  }
  return ProcessResult{0, "wrote targeted kernel source: " + output_path + "\n",
                       ""};
}

BridgeRenderResult BuildBridgeRenderResult(const Invocation &invocation) {
  BridgeRenderResult result;
  result.command = BuildIreeRunLoomCommand(invocation);
  if (!result.command.args.has_value()) {
    result.status = "blocked";
    return result;
  }
  if (!invocation.execute_iree_run_loom_command) {
    result.status = "ready";
    return result;
  }

  if (result.command.loom_link_args.has_value()) {
    result.loom_link_execution = ExecuteProcess(*result.command.loom_link_args);
    if (result.loom_link_execution->exit_code != 0 ||
        !result.loom_link_execution->error.empty()) {
      result.status = "link_failed";
      return result;
    }
  }

  if (IsAmdGpuTargetKey(invocation.target)) {
    const std::string input_path = invocation.configs.empty()
                                       ? invocation.kernel_path
                                       : invocation.linked_kernel_output;
    const std::string output_path = TargetedKernelPath(invocation, input_path);
    result.target_source_materialization =
        MaterializeTargetedKernelSource(invocation, input_path, output_path);
    if (result.target_source_materialization->exit_code != 0 ||
        !result.target_source_materialization->error.empty()) {
      result.status = "target_source_failed";
      return result;
    }
  }

  result.run_execution = ExecuteProcess(*result.command.args);
  if (result.run_execution->exit_code != 0 ||
      !result.run_execution->error.empty()) {
    result.status = "run_failed";
    return result;
  }
  result.status = "run_passed";
  return result;
}

} // namespace

ParseResult ParseArgs(const std::vector<std::string> &args) {
  Invocation invocation;
  std::vector<std::string> errors;

  for (std::size_t i = 0; i < args.size(); ++i) {
    const std::string &flag = args[i];
    auto require_value = [&](const char *name) -> std::optional<std::string> {
      if (i + 1 >= args.size()) {
        errors.push_back(std::string("missing value for ") + name);
        return std::nullopt;
      }
      ++i;
      return args[i];
    };

    if (flag == "--kernel") {
      if (auto value = require_value("--kernel")) {
        invocation.kernel_path = *value;
      }
    } else if (flag == "--root") {
      if (auto value = require_value("--root")) {
        invocation.root_symbol = *value;
      }
    } else if (flag == "--target") {
      if (auto value = require_value("--target")) {
        invocation.target = *value;
      }
    } else if (flag == "--output") {
      if (auto value = require_value("--output")) {
        invocation.output_path = *value;
      }
    } else if (flag == "--loom-link") {
      if (auto value = require_value("--loom-link")) {
        invocation.loom_link_path = *value;
      }
    } else if (flag == "--linked-kernel-output") {
      if (auto value = require_value("--linked-kernel-output")) {
        invocation.linked_kernel_output = *value;
      }
    } else if (flag == "--iree-run-loom") {
      if (auto value = require_value("--iree-run-loom")) {
        invocation.iree_run_loom_path = *value;
      }
    } else if (flag == "--workgroup-count") {
      if (auto value = require_value("--workgroup-count")) {
        if (auto workgroup_count = ParseWorkgroupCount(*value, &errors)) {
          invocation.workgroup_count = *workgroup_count;
        }
      }
    } else if (flag == "--emit-iree-run-loom-command") {
      invocation.emit_iree_run_loom_command = true;
    } else if (flag == "--execute-iree-run-loom-command") {
      invocation.emit_iree_run_loom_command = true;
      invocation.execute_iree_run_loom_command = true;
    } else if (flag == "--config") {
      if (auto value = require_value("--config")) {
        if (auto config = ParseConfig(*value, &errors)) {
          invocation.configs.push_back(*config);
        }
      }
    } else if (flag == "--binding") {
      if (auto value = require_value("--binding")) {
        if (auto binding = ParseBindingFlag(*value, &errors)) {
          invocation.bindings.push_back(*binding);
        }
      }
    } else if (flag == "--scalar") {
      if (auto value = require_value("--scalar")) {
        if (auto scalar = ParseScalarFlag(*value, &errors)) {
          invocation.scalars.push_back(*scalar);
        }
      }
    } else if (flag == "--expect") {
      if (auto value = require_value("--expect")) {
        if (auto expectation = ParseExpectationFlag(*value, &errors)) {
          invocation.expectations.push_back(*expectation);
        }
      }
    } else {
      errors.push_back("unknown flag: " + flag);
    }
  }

  AddMissingRequired(invocation, &errors);
  ValidateBindingsAndExpectation(&invocation, &errors);

  if (!errors.empty()) {
    return ParseResult{std::nullopt, errors};
  }
  return ParseResult{invocation, {}};
}

F32NpyLoadResult LoadF32Npy1D(const std::string &path,
                              std::size_t expected_elements) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return F32NpyLoadResult{std::nullopt, "failed to open npy file: " + path};
  }

  unsigned char prefix[10] = {};
  input.read(reinterpret_cast<char *>(prefix), 8);
  if (input.gcount() != 8 || std::memcmp(prefix, "\x93NUMPY", 6) != 0) {
    return F32NpyLoadResult{std::nullopt, "invalid npy magic: " + path};
  }

  const unsigned char major = prefix[6];
  std::size_t header_len = 0;
  if (major == 1) {
    input.read(reinterpret_cast<char *>(prefix + 8), 2);
    if (input.gcount() != 2) {
      return F32NpyLoadResult{std::nullopt,
                              "truncated npy v1 header length: " + path};
    }
    header_len = ReadLe16(prefix + 8);
  } else if (major == 2) {
    unsigned char len_bytes[4] = {};
    input.read(reinterpret_cast<char *>(len_bytes), 4);
    if (input.gcount() != 4) {
      return F32NpyLoadResult{std::nullopt,
                              "truncated npy v2 header length: " + path};
    }
    header_len = ReadLe32(len_bytes);
  } else {
    return F32NpyLoadResult{
        std::nullopt, "unsupported npy version: " + std::to_string(major) +
                          "." + std::to_string(prefix[7])};
  }

  std::string header(header_len, '\0');
  input.read(header.data(), static_cast<std::streamsize>(header.size()));
  if (static_cast<std::size_t>(input.gcount()) != header.size()) {
    return F32NpyLoadResult{std::nullopt, "truncated npy header: " + path};
  }

  const std::optional<std::string> descr = ExtractHeaderString(header, "descr");
  if (!descr.has_value()) {
    return F32NpyLoadResult{std::nullopt,
                            "missing descr in npy header: " + path};
  }
  if (*descr != "<f4" && *descr != "|f4") {
    return F32NpyLoadResult{
        std::nullopt, "expected f32 npy dtype '<f4', saw '" + *descr + "'"};
  }
  if (!HeaderHasFalse(header, "fortran_order")) {
    return F32NpyLoadResult{
        std::nullopt, "only C-contiguous npy arrays are supported: " + path};
  }

  std::string shape_error;
  const std::optional<std::size_t> elements =
      ExtractOneDimShape(header, &shape_error);
  if (!elements.has_value()) {
    return F32NpyLoadResult{std::nullopt, shape_error + ": " + path};
  }
  if (*elements != expected_elements) {
    return F32NpyLoadResult{
        std::nullopt, "npy element count mismatch for " + path + ": expected " +
                          std::to_string(expected_elements) + ", saw " +
                          std::to_string(*elements)};
  }

  std::vector<unsigned char> bytes(expected_elements * sizeof(float));
  input.read(reinterpret_cast<char *>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  if (static_cast<std::size_t>(input.gcount()) != bytes.size()) {
    return F32NpyLoadResult{std::nullopt, "truncated npy data: " + path};
  }

  F32Tensor tensor;
  tensor.values.resize(expected_elements);
  if (!bytes.empty()) {
    std::memcpy(tensor.values.data(), bytes.data(), bytes.size());
  }
  return F32NpyLoadResult{tensor, ""};
}

NpyLoadResult ValidateNpyStorage1D(const std::string &path, DType dtype,
                                   std::size_t expected_elements) {
  if (dtype == DType::kF32) {
    const F32NpyLoadResult loaded = LoadF32Npy1D(path, expected_elements);
    return NpyLoadResult{loaded.tensor.has_value(), loaded.error};
  }

  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return NpyLoadResult{false, "failed to open npy file: " + path};
  }

  unsigned char prefix[10] = {};
  input.read(reinterpret_cast<char *>(prefix), 8);
  if (input.gcount() != 8 || std::memcmp(prefix, "\x93NUMPY", 6) != 0) {
    return NpyLoadResult{false, "invalid npy magic: " + path};
  }

  const unsigned char major = prefix[6];
  std::size_t header_len = 0;
  if (major == 1) {
    input.read(reinterpret_cast<char *>(prefix + 8), 2);
    if (input.gcount() != 2) {
      return NpyLoadResult{false, "truncated npy v1 header length: " + path};
    }
    header_len = ReadLe16(prefix + 8);
  } else if (major == 2) {
    unsigned char len_bytes[4] = {};
    input.read(reinterpret_cast<char *>(len_bytes), 4);
    if (input.gcount() != 4) {
      return NpyLoadResult{false, "truncated npy v2 header length: " + path};
    }
    header_len = ReadLe32(len_bytes);
  } else {
    return NpyLoadResult{false,
                         "unsupported npy version: " + std::to_string(major) +
                             "." + std::to_string(prefix[7])};
  }

  std::string header(header_len, '\0');
  input.read(header.data(), static_cast<std::streamsize>(header.size()));
  if (static_cast<std::size_t>(input.gcount()) != header.size()) {
    return NpyLoadResult{false, "truncated npy header: " + path};
  }

  const std::optional<std::string> descr = ExtractHeaderString(header, "descr");
  if (!descr.has_value()) {
    return NpyLoadResult{false, "missing descr in npy header: " + path};
  }
  if ((dtype == DType::kBF16 || dtype == DType::kF16) &&
      *descr != "<i2" && *descr != "|i2") {
    return NpyLoadResult{
        false, "expected " + ToString(dtype) +
                   " storage npy dtype '<i2', saw '" + *descr + "'"};
  }
  if (dtype == DType::kI32 && *descr != "<i4" && *descr != "|i4") {
    return NpyLoadResult{false,
                         "expected i32 npy dtype '<i4', saw '" + *descr + "'"};
  }
  if (!HeaderHasFalse(header, "fortran_order")) {
    return NpyLoadResult{false,
                         "only C-contiguous npy arrays are supported: " + path};
  }

  std::string shape_error;
  const std::optional<std::size_t> elements =
      ExtractOneDimShape(header, &shape_error);
  if (!elements.has_value()) {
    return NpyLoadResult{false, shape_error + ": " + path};
  }
  if (*elements != expected_elements) {
    return NpyLoadResult{false, "npy element count mismatch for " + path +
                                    ": expected " +
                                    std::to_string(expected_elements) +
                                    ", saw " + std::to_string(*elements)};
  }

  const std::size_t element_bytes =
      dtype == DType::kBF16 || dtype == DType::kF16 ? 2 : 4;
  std::vector<unsigned char> bytes(expected_elements * element_bytes);
  input.read(reinterpret_cast<char *>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  if (static_cast<std::size_t>(input.gcount()) != bytes.size()) {
    return NpyLoadResult{false, "truncated npy data: " + path};
  }
  return NpyLoadResult{true, ""};
}

CloseCompareResult CompareClose(const std::vector<float> &actual,
                                const std::vector<float> &expected, double atol,
                                double rtol) {
  CloseCompareResult result;
  if (actual.size() != expected.size()) {
    result.error = "actual and expected element counts differ";
    return result;
  }
  result.compared_elements = actual.size();
  result.passed = true;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const double actual_value = static_cast<double>(actual[i]);
    const double expected_value = static_cast<double>(expected[i]);
    const double abs_error = std::abs(actual_value - expected_value);
    const double rel_error = expected_value == 0.0
                                 ? abs_error
                                 : abs_error / std::abs(expected_value);
    result.max_abs_error = std::max(result.max_abs_error, abs_error);
    result.max_rel_error = std::max(result.max_rel_error, rel_error);

    const bool finite =
        std::isfinite(actual_value) && std::isfinite(expected_value);
    const bool close =
        finite && abs_error <= atol + rtol * std::abs(expected_value);
    if (!close && !result.first_failing_index.has_value()) {
      result.first_failing_index = i;
      result.first_failing_actual = actual[i];
      result.first_failing_expected = expected[i];
      result.first_failing_abs_error = abs_error;
      result.first_failing_rel_error = rel_error;
    }
    if (!close) {
      result.passed = false;
    }
  }
  return result;
}

IreeRunLoomCommandResult BuildIreeRunLoomCommand(const Invocation &invocation) {
  std::optional<std::vector<std::string>> loom_link_args;
  std::string kernel_path = invocation.kernel_path;
  if (!invocation.configs.empty() && invocation.linked_kernel_output.empty()) {
    return IreeRunLoomCommandResult{
        std::nullopt, std::nullopt,
        "config-bound kernels require --linked-kernel-output so the bridge can "
        "stage a loom-link command before iree-run-loom"};
  }
  if (!invocation.configs.empty()) {
    std::vector<std::string> link_args;
    link_args.push_back(invocation.loom_link_path);
    link_args.push_back(invocation.kernel_path);
    link_args.push_back("--mode=link");
    link_args.push_back("--to=text");
    link_args.push_back("--require-resolved-config");
    link_args.push_back("--root=" + invocation.root_symbol);
    link_args.push_back("--output=" + invocation.linked_kernel_output);
    for (const auto &config : invocation.configs) {
      link_args.push_back("--config=" + config.first + "=" + config.second);
    }
    loom_link_args = link_args;
    kernel_path = invocation.linked_kernel_output;
  }
  kernel_path = TargetedKernelPath(invocation, kernel_path);
  if (invocation.expectations.empty()) {
    return IreeRunLoomCommandResult{
        std::nullopt, std::nullopt,
        "iree-run-loom command emission requires --expect"};
  }
  std::unordered_map<int, const Expectation *> expectations_by_position;
  for (const Expectation &expectation : invocation.expectations) {
    expectations_by_position.emplace(expectation.position, &expectation);
  }

  std::vector<std::string> args;
  args.push_back(invocation.iree_run_loom_path);
  args.push_back(kernel_path);
  const std::string backend = BackendForTarget(invocation.target);
  args.push_back("--backend=" + backend);
  args.push_back("--function=" + invocation.root_symbol);
  if (!invocation.workgroup_count.empty()) {
    args.push_back("--workgroup-count=" + invocation.workgroup_count);
  }
  for (const ScalarArg &scalar : invocation.scalars) {
    args.push_back("--kernel-input-value=" + ToString(scalar.dtype) + "=" +
                   scalar.value);
  }

  std::vector<std::string> expected_specs;
  std::vector<std::string> expected_tolerances;
  expected_specs.reserve(invocation.bindings.size());
  expected_tolerances.reserve(invocation.bindings.size());
  for (const Binding &binding : invocation.bindings) {
    std::string error;
    std::optional<std::string> binding_spec = BuildNpyStorageBindingSpec(
        binding.path, binding.dtype, binding.elements, &error);
    if (!binding_spec.has_value()) {
      return IreeRunLoomCommandResult{std::nullopt, std::nullopt, error};
    }
    args.push_back("--kernel-input-buffer=" + *binding_spec);

    std::string expected_path = binding.path;
    double expected_atol = 0.0;
    double expected_rtol = 0.0;
    const auto expectation = expectations_by_position.find(binding.position);
    if (expectation != expectations_by_position.end()) {
      expected_path = expectation->second->path;
      expected_atol = expectation->second->atol;
      expected_rtol = expectation->second->rtol;
    }
    std::optional<std::string> expected_spec = BuildNpyExpectedBindingSpec(
        expected_path, binding.dtype, binding.elements, &error);
    if (!expected_spec.has_value()) {
      return IreeRunLoomCommandResult{std::nullopt, std::nullopt, error};
    }
    expected_specs.push_back(*expected_spec);
    expected_tolerances.push_back(FormatDouble(expected_atol) + "," +
                                  FormatDouble(expected_rtol));
  }
  for (const std::string &expected_spec : expected_specs) {
    args.push_back("--expected-kernel-buffer=" + expected_spec);
  }
  for (const std::string &expected_tolerance : expected_tolerances) {
    args.push_back("--expected-kernel-buffer-tolerance=" + expected_tolerance);
  }

  return IreeRunLoomCommandResult{loom_link_args, args, ""};
}

std::string ToString(BindingKind kind) {
  switch (kind) {
  case BindingKind::kInput:
    return "input";
  case BindingKind::kOutput:
    return "output";
  }
  return "unknown";
}

std::string ToString(DType dtype) {
  switch (dtype) {
  case DType::kBF16:
    return "bf16";
  case DType::kF32:
    return "f32";
  case DType::kF16:
    return "f16";
  case DType::kI32:
    return "i32";
  }
  return "unknown";
}

std::string ToString(CheckMode mode) {
  switch (mode) {
  case CheckMode::kClose:
    return "close";
  }
  return "unknown";
}

std::string BackendForTarget(const std::string &target) {
  if (target.rfind("gfx", 0) == 0 || target == "amdgpu" ||
      target == "amdgpu-hal") {
    return "amdgpu-hal";
  }
  return target;
}

std::string RenderResultJson(const Invocation &invocation) {
  std::optional<BridgeRenderResult> bridge_result;
  if (invocation.emit_iree_run_loom_command) {
    bridge_result = BuildBridgeRenderResult(invocation);
  }

  std::ostringstream os;
  os << "{\n";
  os << "  \"status\": \""
     << (bridge_result.has_value() && invocation.execute_iree_run_loom_command
             ? bridge_result->status
             : "not_run")
     << "\",\n";
  os << "  \"kernel\": \"" << JsonEscape(invocation.kernel_path) << "\",\n";
  os << "  \"root\": \"" << JsonEscape(invocation.root_symbol) << "\",\n";
  os << "  \"target\": \"" << JsonEscape(invocation.target) << "\",\n";
  os << "  \"backend\": \"" << JsonEscape(BackendForTarget(invocation.target))
     << "\",\n";
  os << "  \"scalar_count\": " << invocation.scalars.size() << ",\n";
  os << "  \"scalars\": [";
  for (std::size_t i = 0; i < invocation.scalars.size(); ++i) {
    const ScalarArg &scalar = invocation.scalars[i];
    if (i != 0) {
      os << ",";
    }
    os << "\n";
    os << "    {\"position\": " << scalar.position << ", \"dtype\": \""
       << ToString(scalar.dtype) << "\", \"value\": \""
       << JsonEscape(scalar.value) << "\"}";
  }
  if (!invocation.scalars.empty()) {
    os << "\n  ";
  }
  os << "],\n";
  os << "  \"binding_count\": " << invocation.bindings.size() << ",\n";
  os << "  \"bindings\": [";
  for (std::size_t i = 0; i < invocation.bindings.size(); ++i) {
    const Binding &binding = invocation.bindings[i];
    if (i != 0) {
      os << ",";
    }
    os << "\n";
    os << "    {\"position\": " << binding.position << ", \"kind\": \""
       << ToString(binding.kind) << "\", \"dtype\": \""
       << ToString(binding.dtype) << "\", \"elements\": " << binding.elements
       << ", \"path\": \"" << JsonEscape(binding.path) << "\"}";
  }
  if (!invocation.bindings.empty()) {
    os << "\n  ";
  }
  os << "],\n";
  os << "  \"checks\": [";
  for (std::size_t i = 0; i < invocation.expectations.size(); ++i) {
    const Expectation &expectation = invocation.expectations[i];
    std::string check_status = "not_run";
    if (bridge_result.has_value() && invocation.execute_iree_run_loom_command) {
      if (bridge_result->status == "run_passed") {
        check_status = "passed";
      } else if (bridge_result->status == "run_failed") {
        check_status = "failed";
      } else {
        check_status = bridge_result->status;
      }
    }
    if (i != 0) {
      os << ",";
    }
    os << "\n";
    os << "    {\"binding\": " << expectation.position << ", \"mode\": \""
       << ToString(expectation.mode) << "\", \"expected_path\": \""
       << JsonEscape(expectation.path) << "\", \"atol\": " << expectation.atol
       << ", \"rtol\": " << expectation.rtol << ", \"status\": \""
       << check_status << "\"}\n";
  }
  if (!invocation.expectations.empty()) {
    os << "  ";
  }
  if (invocation.emit_iree_run_loom_command) {
    os << "],\n";
    os << "  \"iree_run_loom_bridge\": {\n";
    os << "    \"status\": \"" << bridge_result->status << "\"";
    if (bridge_result->command.args.has_value()) {
      os << ",\n";
      if (bridge_result->command.loom_link_args.has_value()) {
        os << "    \"loom_link_args\": ";
        WriteJsonStringArray(&os, *bridge_result->command.loom_link_args);
        os << ",\n";
      }
      os << "    \"args\": ";
      WriteJsonStringArray(&os, *bridge_result->command.args);
      if (bridge_result->loom_link_execution.has_value()) {
        os << ",\n";
        os << "    \"loom_link_exit_code\": "
           << bridge_result->loom_link_execution->exit_code << ",\n";
        os << "    \"loom_link_output\": \""
           << JsonEscape(bridge_result->loom_link_execution->output) << "\"";
        if (!bridge_result->loom_link_execution->error.empty()) {
          os << ",\n";
          os << "    \"loom_link_error\": \""
             << JsonEscape(bridge_result->loom_link_execution->error) << "\"";
        }
      }
      if (bridge_result->run_execution.has_value()) {
        os << ",\n";
        os << "    \"run_exit_code\": "
           << bridge_result->run_execution->exit_code << ",\n";
        os << "    \"run_output\": \""
           << JsonEscape(bridge_result->run_execution->output) << "\"";
        if (!bridge_result->run_execution->error.empty()) {
          os << ",\n";
          os << "    \"run_error\": \""
             << JsonEscape(bridge_result->run_execution->error) << "\"";
        }
      }
      os << "\n";
    } else {
      os << ",\n";
      os << "    \"error\": \"" << JsonEscape(bridge_result->command.error)
         << "\"\n";
    }
    os << "  }\n";
  } else {
    os << "]\n";
  }
  os << "}\n";
  return os.str();
}

std::string RenderNotRunResultJson(const Invocation &invocation) {
  return RenderResultJson(invocation);
}

} // namespace ggml_hrx::run_loom_simple
