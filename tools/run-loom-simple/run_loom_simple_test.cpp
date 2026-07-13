#include "tools/run-loom-simple/run_loom_simple.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

using ggml_hrx::run_loom_simple::BackendForTarget;
using ggml_hrx::run_loom_simple::BindingKind;
using ggml_hrx::run_loom_simple::BuildIreeRunLoomCommand;
using ggml_hrx::run_loom_simple::CompareClose;
using ggml_hrx::run_loom_simple::DType;
using ggml_hrx::run_loom_simple::LoadF32Npy1D;
using ggml_hrx::run_loom_simple::ParseArgs;
using ggml_hrx::run_loom_simple::RenderResultJson;
using ggml_hrx::run_loom_simple::ValidateNpyStorage1D;

int g_failures = 0;

void Expect(bool condition, const std::string &message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    ++g_failures;
  }
}

std::vector<std::string> ValidArgs() {
  return {
      "--kernel",
      "kernels/v2/add_f32.loom",
      "--root",
      "@add_f32",
      "--target",
      "gfx1100",
      "--workgroup-count",
      "16,1,1",
      "--config",
      "BLOCK_SIZE=256",
      "--binding",
      "1:input:f32:4096:fixtures/src1.npy",
      "--binding",
      "0:input:f32:4096:fixtures/src0.npy",
      "--binding",
      "2:output:f32:4096:fixtures/dst_init.npy",
      "--expect",
      "2:close:fixtures/expected.npy:1e-5:1e-5",
      "--output",
      "result.json",
  };
}

std::vector<std::string> ValidArgsWithoutConfig() {
  return {
      "--kernel",  "linked.loom",
      "--root",    "@add_f32",
      "--target",  "gfx1100",
      "--binding", "0:input:f32:4:fixtures/src0.npy",
      "--binding", "1:output:f32:4:fixtures/dst_init.npy",
      "--expect",  "1:close:fixtures/expected.npy:1e-5:1e-5",
      "--output",  "result.json",
  };
}

bool HasErrorContaining(const std::vector<std::string> &errors,
                        const std::string &needle) {
  for (const std::string &error : errors) {
    if (error.find(needle) != std::string::npos) {
      return true;
    }
  }
  return false;
}

std::filesystem::path TempDir() {
  const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() /
      ("ggml-hrx-run-loom-simple-tests-" + std::to_string(now));
  std::filesystem::create_directories(path);
  return path;
}

void WriteTextFile(const std::filesystem::path &path,
                   const std::string &contents) {
  std::ofstream output(path);
  output << contents;
}

std::string ReadTextFile(const std::filesystem::path &path) {
  std::ifstream input(path);
  std::string contents;
  std::string line;
  while (std::getline(input, line)) {
    contents += line;
    contents += "\n";
  }
  return contents;
}

void WriteF32Npy(const std::filesystem::path &path,
                 const std::vector<float> &values, int dims = 1,
                 const std::string &descr = "<f4", bool fortran_order = false,
                 unsigned char major = 1) {
  std::ofstream output(path, std::ios::binary);
  const std::string fortran = fortran_order ? "True" : "False";
  std::string shape;
  if (dims == 1) {
    shape = "(" + std::to_string(values.size()) + ",)";
  } else {
    shape = "(1, " + std::to_string(values.size()) + ")";
  }
  std::string header = "{'descr': '" + descr +
                       "', 'fortran_order': " + fortran +
                       ", 'shape': " + shape + ", }";
  const std::size_t prefix_size = major == 1 ? 10 : 12;
  const std::size_t newline_size = 1;
  const std::size_t remainder =
      (prefix_size + header.size() + newline_size) % 16;
  const std::size_t padding = remainder == 0 ? 0 : 16 - remainder;
  header.append(padding, ' ');
  header.push_back('\n');

  output.write("\x93NUMPY", 6);
  const unsigned char version[2] = {major, 0};
  output.write(reinterpret_cast<const char *>(version), 2);
  if (major == 1) {
    const std::uint16_t len = static_cast<std::uint16_t>(header.size());
    const unsigned char bytes[2] = {
        static_cast<unsigned char>(len & 0xFF),
        static_cast<unsigned char>((len >> 8) & 0xFF),
    };
    output.write(reinterpret_cast<const char *>(bytes), 2);
  } else {
    const std::uint32_t len = static_cast<std::uint32_t>(header.size());
    const unsigned char bytes[4] = {
        static_cast<unsigned char>(len & 0xFF),
        static_cast<unsigned char>((len >> 8) & 0xFF),
        static_cast<unsigned char>((len >> 16) & 0xFF),
        static_cast<unsigned char>((len >> 24) & 0xFF),
    };
    output.write(reinterpret_cast<const char *>(bytes), 4);
  }
  output.write(header.data(), static_cast<std::streamsize>(header.size()));
  for (const float value : values) {
    output.write(reinterpret_cast<const char *>(&value), sizeof(value));
  }
}

void WriteI16Npy(const std::filesystem::path &path,
                 const std::vector<std::int16_t> &values,
                 const std::string &descr = "<i2") {
  std::ofstream output(path, std::ios::binary);
  std::string header = "{'descr': '" + descr +
                       "', 'fortran_order': False, 'shape': (" +
                       std::to_string(values.size()) + ",), }";
  const std::size_t prefix_size = 10;
  const std::size_t newline_size = 1;
  const std::size_t remainder =
      (prefix_size + header.size() + newline_size) % 16;
  const std::size_t padding = remainder == 0 ? 0 : 16 - remainder;
  header.append(padding, ' ');
  header.push_back('\n');

  output.write("\x93NUMPY", 6);
  const unsigned char version[2] = {1, 0};
  output.write(reinterpret_cast<const char *>(version), 2);
  const std::uint16_t len = static_cast<std::uint16_t>(header.size());
  const unsigned char bytes[2] = {
      static_cast<unsigned char>(len & 0xFF),
      static_cast<unsigned char>((len >> 8) & 0xFF),
  };
  output.write(reinterpret_cast<const char *>(bytes), 2);
  output.write(header.data(), static_cast<std::streamsize>(header.size()));
  for (const std::int16_t value : values) {
    output.write(reinterpret_cast<const char *>(&value), sizeof(value));
  }
}

void WriteI32Npy(const std::filesystem::path &path,
                 const std::vector<std::int32_t> &values,
                 const std::string &descr = "<i4") {
  std::ofstream output(path, std::ios::binary);
  std::string header = "{'descr': '" + descr +
                       "', 'fortran_order': False, 'shape': (" +
                       std::to_string(values.size()) + ",), }";
  const std::size_t prefix_size = 10;
  const std::size_t newline_size = 1;
  const std::size_t remainder =
      (prefix_size + header.size() + newline_size) % 16;
  const std::size_t padding = remainder == 0 ? 0 : 16 - remainder;
  header.append(padding, ' ');
  header.push_back('\n');

  output.write("\x93NUMPY", 6);
  const unsigned char version[2] = {1, 0};
  output.write(reinterpret_cast<const char *>(version), 2);
  const std::uint16_t len = static_cast<std::uint16_t>(header.size());
  const unsigned char bytes[2] = {
      static_cast<unsigned char>(len & 0xFF),
      static_cast<unsigned char>((len >> 8) & 0xFF),
  };
  output.write(reinterpret_cast<const char *>(bytes), 2);
  output.write(header.data(), static_cast<std::streamsize>(header.size()));
  for (const std::int32_t value : values) {
    output.write(reinterpret_cast<const char *>(&value), sizeof(value));
  }
}

void WriteExecutableScript(const std::filesystem::path &path,
                           const std::string &body) {
  std::ofstream output(path);
  output << "#!/bin/sh\n";
  output << body;
  output.close();
  std::filesystem::permissions(path,
                               std::filesystem::perms::owner_read |
                                   std::filesystem::perms::owner_write |
                                   std::filesystem::perms::owner_exec,
                               std::filesystem::perm_options::replace);
}

void TestParsesValidCommand() {
  const auto parsed = ParseArgs(ValidArgs());
  Expect(parsed.invocation.has_value(), "valid command parses");
  if (!parsed.invocation.has_value()) {
    return;
  }
  const auto &invocation = *parsed.invocation;
  Expect(invocation.kernel_path == "kernels/v2/add_f32.loom",
         "kernel path parsed");
  Expect(invocation.root_symbol == "@add_f32", "root symbol parsed");
  Expect(invocation.target == "gfx1100", "target parsed");
  Expect(invocation.output_path == "result.json", "output path parsed");
  Expect(invocation.workgroup_count == "16,1,1", "workgroup count parsed");
  Expect(BackendForTarget(invocation.target) == "amdgpu-hal",
         "gfx target maps to amdgpu-hal");
  Expect(invocation.configs.size() == 1, "one config parsed");
  Expect(invocation.configs[0].first == "BLOCK_SIZE", "config key parsed");
  Expect(invocation.configs[0].second == "256", "config value parsed");
  Expect(invocation.bindings.size() == 3, "three bindings parsed");
  Expect(invocation.bindings[0].position == 0, "bindings sorted by position");
  Expect(invocation.bindings[1].position == 1,
         "second binding sorted by position");
  Expect(invocation.bindings[2].position == 2,
         "third binding sorted by position");
  Expect(invocation.bindings[2].kind == BindingKind::kOutput,
         "output binding kind parsed");
  Expect(invocation.expectations.size() == 1, "expectation parsed");
  Expect(invocation.expectations[0].position == 2,
         "expectation position parsed");
}

void TestRejectsMissingKernel() {
  auto args = ValidArgs();
  args.erase(args.begin(), args.begin() + 2);
  const auto parsed = ParseArgs(args);
  Expect(!parsed.invocation.has_value(), "missing kernel rejected");
  Expect(HasErrorContaining(parsed.errors, "missing required --kernel"),
         "missing kernel error");
}

void TestRejectsMissingRoot() {
  auto args = ValidArgs();
  args.erase(args.begin() + 2, args.begin() + 4);
  const auto parsed = ParseArgs(args);
  Expect(!parsed.invocation.has_value(), "missing root rejected");
  Expect(HasErrorContaining(parsed.errors, "missing required --root"),
         "missing root error");
}

void TestRejectsMalformedConfig() {
  auto args = ValidArgs();
  for (std::size_t i = 0; i + 1 < args.size(); ++i) {
    if (args[i] == "--config") {
      args[i + 1] = "BLOCK_SIZE";
      break;
    }
  }
  const auto parsed = ParseArgs(args);
  Expect(!parsed.invocation.has_value(), "malformed config rejected");
  Expect(HasErrorContaining(parsed.errors, "--config must have form"),
         "malformed config error");
}

void TestRejectsMalformedWorkgroupCount() {
  auto args = ValidArgs();
  for (std::size_t i = 0; i + 1 < args.size(); ++i) {
    if (args[i] == "--workgroup-count") {
      args[i + 1] = "16,1";
      break;
    }
  }
  const auto parsed = ParseArgs(args);
  Expect(!parsed.invocation.has_value(), "malformed workgroup count rejected");
  Expect(HasErrorContaining(parsed.errors, "--workgroup-count must have form"),
         "malformed workgroup count error");
}

void TestRejectsMalformedBinding() {
  auto args = ValidArgs();
  for (std::size_t i = 0; i + 1 < args.size(); ++i) {
    if (args[i] == "--binding") {
      args[i + 1] = "0:input:f32";
      break;
    }
  }
  const auto parsed = ParseArgs(args);
  Expect(!parsed.invocation.has_value(), "malformed binding rejected");
  Expect(HasErrorContaining(parsed.errors, "--binding must have form"),
         "malformed binding error");
}

void TestRejectsDuplicateBindingPosition() {
  auto args = ValidArgs();
  args.push_back("--binding");
  args.push_back("2:output:f32:4096:fixtures/other.npy");
  const auto parsed = ParseArgs(args);
  Expect(!parsed.invocation.has_value(), "duplicate binding rejected");
  Expect(HasErrorContaining(parsed.errors, "duplicate ABI position: 2"),
         "duplicate binding error");
}

void TestParsesScalarCommand() {
  auto args = ValidArgsWithoutConfig();
  args.push_back("--scalar");
  args.push_back("0:f32:0.00001");
  const auto parsed = ParseArgs(args);
  Expect(!parsed.invocation.has_value(), "scalar duplicate position rejected");
  Expect(HasErrorContaining(parsed.errors, "duplicate ABI position: 0"),
         "scalar duplicate position error");

  args = {
      "--kernel",  "linked.loom",
      "--root",    "@scale_f32",
      "--target",  "gfx1100",
      "--scalar",  "0:f32:0.5",
      "--binding", "1:input:f32:4:fixtures/src0.npy",
      "--binding", "2:output:f32:4:fixtures/dst_init.npy",
      "--expect",  "2:close:fixtures/expected.npy:1e-5:1e-5",
      "--output",  "result.json",
  };
  const auto parsed_with_scalar = ParseArgs(args);
  Expect(parsed_with_scalar.invocation.has_value(), "scalar command parses");
  if (!parsed_with_scalar.invocation.has_value()) {
    return;
  }
  Expect(parsed_with_scalar.invocation->scalars.size() == 1,
         "one scalar parsed");
  Expect(parsed_with_scalar.invocation->scalars[0].position == 0,
         "scalar position parsed");
  Expect(parsed_with_scalar.invocation->scalars[0].value == "0.5",
         "scalar value parsed");
  Expect(parsed_with_scalar.invocation->bindings[0].position == 1,
         "buffer after scalar sorted");
}

void TestRejectsUnsupportedDType() {
  auto args = ValidArgs();
  for (std::size_t i = 0; i + 1 < args.size(); ++i) {
    if (args[i] == "--binding") {
      args[i + 1] = "0:input:q8_0:4096:fixtures/src0.npy";
      break;
    }
  }
  const auto parsed = ParseArgs(args);
  Expect(!parsed.invocation.has_value(), "unsupported dtype rejected");
  Expect(HasErrorContaining(parsed.errors, "unsupported dtype: q8_0"),
         "unsupported dtype error");
}

void TestParsesF16BindingDType() {
  auto args = ValidArgsWithoutConfig();
  for (std::size_t i = 0; i + 1 < args.size(); ++i) {
    if (args[i] == "--binding") {
      args[i + 1] = "0:input:f16:4:fixtures/src0.npy";
      break;
    }
  }
  const auto parsed = ParseArgs(args);
  Expect(parsed.invocation.has_value(), "f16 dtype parses");
  if (!parsed.invocation.has_value()) {
    return;
  }
  Expect(parsed.invocation->bindings[0].dtype == DType::kF16,
         "f16 dtype recorded");
}

void TestParsesI32BindingDType() {
  auto args = ValidArgsWithoutConfig();
  for (std::size_t i = 0; i + 1 < args.size(); ++i) {
    if (args[i] == "--binding") {
      args[i + 1] = "0:input:i32:4:fixtures/indices.npy";
      break;
    }
  }
  const auto parsed = ParseArgs(args);
  Expect(parsed.invocation.has_value(), "i32 dtype parses");
  if (!parsed.invocation.has_value()) {
    return;
  }
  Expect(parsed.invocation->bindings[0].dtype == DType::kI32,
         "i32 dtype recorded");
}

void TestRejectsUnsupportedKind() {
  auto args = ValidArgs();
  for (std::size_t i = 0; i + 1 < args.size(); ++i) {
    if (args[i] == "--binding") {
      args[i + 1] = "0:temporary:f32:4096:fixtures/src0.npy";
      break;
    }
  }
  const auto parsed = ParseArgs(args);
  Expect(!parsed.invocation.has_value(), "unsupported kind rejected");
  Expect(
      HasErrorContaining(parsed.errors, "unsupported binding kind: temporary"),
      "unsupported kind error");
}

void TestRejectsExpectForInputBinding() {
  auto args = ValidArgs();
  for (std::size_t i = 0; i + 1 < args.size(); ++i) {
    if (args[i] == "--expect") {
      args[i + 1] = "0:close:fixtures/expected.npy:1e-5:1e-5";
      break;
    }
  }
  const auto parsed = ParseArgs(args);
  Expect(!parsed.invocation.has_value(), "expect on input rejected");
  Expect(HasErrorContaining(
             parsed.errors,
             "--expect references a non-output binding position: 0"),
         "expect on input error");
}

void TestParsesMultipleExpectations() {
  auto args = std::vector<std::string>{
      "--kernel",  "linked.loom",
      "--root",    "@two_output",
      "--target",  "gfx1100",
      "--binding", "0:input:f32:4:fixtures/src0.npy",
      "--binding", "1:output:f32:4:fixtures/tmp.npy",
      "--binding", "2:output:f32:4:fixtures/dst.npy",
      "--expect",  "1:close:fixtures/tmp_expected.npy:1e-5:1e-5",
      "--expect",  "2:close:fixtures/dst_expected.npy:1e-5:1e-5",
      "--output",  "result.json",
  };
  const auto parsed = ParseArgs(args);
  Expect(parsed.invocation.has_value(), "multiple expectations parse");
  if (!parsed.invocation.has_value()) {
    return;
  }
  Expect(parsed.invocation->expectations.size() == 2,
         "two expectations parsed");

  args.push_back("--expect");
  args.push_back("2:close:fixtures/duplicate.npy:1e-5:1e-5");
  const auto duplicate = ParseArgs(args);
  Expect(!duplicate.invocation.has_value(), "duplicate expectation rejected");
  Expect(HasErrorContaining(duplicate.errors, "duplicate expectation position"),
         "duplicate expectation error");
}

void TestRenderNotRunJson() {
  const auto parsed = ParseArgs(ValidArgs());
  Expect(parsed.invocation.has_value(), "valid command parses for JSON");
  if (!parsed.invocation.has_value()) {
    return;
  }
  const std::string json =
      ggml_hrx::run_loom_simple::RenderNotRunResultJson(*parsed.invocation);
  Expect(json.find("\"status\": \"not_run\"") != std::string::npos,
         "JSON has status");
  Expect(json.find("\"binding_count\": 3") != std::string::npos,
         "JSON has binding count");
  Expect(json.find("\"binding\": 2") != std::string::npos,
         "JSON has expectation binding");
}

void TestParsesIreeRunLoomBridgeFlags() {
  auto args = ValidArgsWithoutConfig();
  args.push_back("--loom-link");
  args.push_back("/tmp/loom-link");
  args.push_back("--linked-kernel-output");
  args.push_back("/tmp/linked.loom");
  args.push_back("--iree-run-loom");
  args.push_back("/tmp/iree-run-loom");
  args.push_back("--emit-iree-run-loom-command");
  args.push_back("--execute-iree-run-loom-command");
  const auto parsed = ParseArgs(args);
  Expect(parsed.invocation.has_value(), "bridge flags parse");
  if (!parsed.invocation.has_value()) {
    return;
  }
  Expect(parsed.invocation->loom_link_path == "/tmp/loom-link",
         "loom-link tool path parsed");
  Expect(parsed.invocation->linked_kernel_output == "/tmp/linked.loom",
         "linked kernel output parsed");
  Expect(parsed.invocation->iree_run_loom_path == "/tmp/iree-run-loom",
         "bridge tool path parsed");
  Expect(parsed.invocation->emit_iree_run_loom_command,
         "bridge emission flag parsed");
  Expect(parsed.invocation->execute_iree_run_loom_command,
         "bridge execution flag parsed");
}

void TestLoadsF32NpyV1() {
  const std::filesystem::path dir = TempDir();
  const std::filesystem::path path = dir / "values.npy";
  WriteF32Npy(path, {1.0f, 2.5f, -3.0f});
  const auto loaded = LoadF32Npy1D(path.string(), 3);
  Expect(loaded.tensor.has_value(), "loads f32 npy v1");
  if (loaded.tensor.has_value()) {
    Expect(loaded.tensor->values.size() == 3, "loaded npy element count");
    Expect(loaded.tensor->values[1] == 2.5f, "loaded npy value");
  }
}

void TestLoadsF32NpyV2() {
  const std::filesystem::path dir = TempDir();
  const std::filesystem::path path = dir / "values-v2.npy";
  WriteF32Npy(path, {4.0f, 5.0f}, 1, "<f4", false, 2);
  const auto loaded = LoadF32Npy1D(path.string(), 2);
  Expect(loaded.tensor.has_value(), "loads f32 npy v2");
  if (loaded.tensor.has_value()) {
    Expect(loaded.tensor->values[0] == 4.0f, "loaded v2 npy value");
  }
}

void TestValidatesF16StorageNpy() {
  const std::filesystem::path dir = TempDir();
  const std::filesystem::path path = dir / "values-f16-storage.npy";
  WriteI16Npy(path, {0x3C00, 0x4000, -0x4200});
  const auto loaded = ValidateNpyStorage1D(path.string(), DType::kF16, 3);
  Expect(loaded.loaded, "validates f16 int16-backed npy");
}

void TestRejectsF16StorageDTypeMismatch() {
  const std::filesystem::path dir = TempDir();
  const std::filesystem::path path = dir / "values-f32.npy";
  WriteF32Npy(path, {1.0f, 2.0f});
  const auto loaded = ValidateNpyStorage1D(path.string(), DType::kF16, 2);
  Expect(!loaded.loaded, "f16 storage dtype mismatch rejected");
  Expect(loaded.error.find("expected f16 storage npy dtype") !=
             std::string::npos,
         "f16 storage dtype mismatch error");
}

void TestValidatesI32StorageNpy() {
  const std::filesystem::path dir = TempDir();
  const std::filesystem::path path = dir / "indices.npy";
  WriteI32Npy(path, {0, 3, 1, 4});
  const auto loaded = ValidateNpyStorage1D(path.string(), DType::kI32, 4);
  Expect(loaded.loaded, "validates i32 npy");
}

void TestRejectsI32StorageDTypeMismatch() {
  const std::filesystem::path dir = TempDir();
  const std::filesystem::path path = dir / "indices-f32.npy";
  WriteF32Npy(path, {1.0f, 2.0f});
  const auto loaded = ValidateNpyStorage1D(path.string(), DType::kI32, 2);
  Expect(!loaded.loaded, "i32 storage dtype mismatch rejected");
  Expect(loaded.error.find("expected i32 npy dtype") != std::string::npos,
         "i32 storage dtype mismatch error");
}

void TestRejectsMissingNpyFile() {
  const auto loaded = LoadF32Npy1D("/tmp/ggml-hrx-missing-file.npy", 1);
  Expect(!loaded.tensor.has_value(), "missing npy file rejected");
  Expect(loaded.error.find("failed to open") != std::string::npos,
         "missing npy file error");
}

void TestRejectsInvalidNpyMagic() {
  const std::filesystem::path dir = TempDir();
  const std::filesystem::path path = dir / "bad.npy";
  std::ofstream(path, std::ios::binary) << "not-npy";
  const auto loaded = LoadF32Npy1D(path.string(), 1);
  Expect(!loaded.tensor.has_value(), "invalid npy magic rejected");
  Expect(loaded.error.find("invalid npy magic") != std::string::npos,
         "invalid npy magic error");
}

void TestRejectsNpyDTypeMismatch() {
  const std::filesystem::path dir = TempDir();
  const std::filesystem::path path = dir / "f64.npy";
  WriteF32Npy(path, {1.0f}, 1, "<f8");
  const auto loaded = LoadF32Npy1D(path.string(), 1);
  Expect(!loaded.tensor.has_value(), "dtype mismatch rejected");
  Expect(loaded.error.find("expected f32 npy dtype") != std::string::npos,
         "dtype mismatch error");
}

void TestRejectsNpyShapeMismatch() {
  const std::filesystem::path dir = TempDir();
  const std::filesystem::path path = dir / "values.npy";
  WriteF32Npy(path, {1.0f, 2.0f});
  const auto loaded = LoadF32Npy1D(path.string(), 3);
  Expect(!loaded.tensor.has_value(), "shape mismatch rejected");
  Expect(loaded.error.find("element count mismatch") != std::string::npos,
         "shape mismatch error");
}

void TestRejectsTwoDimensionalNpy() {
  const std::filesystem::path dir = TempDir();
  const std::filesystem::path path = dir / "two-d.npy";
  WriteF32Npy(path, {1.0f, 2.0f}, 2);
  const auto loaded = LoadF32Npy1D(path.string(), 2);
  Expect(!loaded.tensor.has_value(), "two-dimensional npy rejected");
  Expect(loaded.error.find("one-dimensional") != std::string::npos,
         "two-dimensional npy error");
}

void TestRejectsFortranOrderNpy() {
  const std::filesystem::path dir = TempDir();
  const std::filesystem::path path = dir / "fortran.npy";
  WriteF32Npy(path, {1.0f}, 1, "<f4", true);
  const auto loaded = LoadF32Npy1D(path.string(), 1);
  Expect(!loaded.tensor.has_value(), "fortran order npy rejected");
  Expect(loaded.error.find("C-contiguous") != std::string::npos,
         "fortran order npy error");
}

void TestCompareClosePassesExactValues() {
  const auto result = CompareClose({1.0f, 2.0f}, {1.0f, 2.0f}, 0.0, 0.0);
  Expect(result.passed, "close comparison passes exact values");
  Expect(result.compared_elements == 2, "close comparison element count");
  Expect(!result.first_failing_index.has_value(), "no failing index on pass");
}

void TestCompareClosePassesWithinTolerance() {
  const auto result = CompareClose({1.01f}, {1.0f}, 0.02, 0.0);
  Expect(result.passed, "close comparison passes within atol");
}

void TestCompareCloseFailsOutsideTolerance() {
  const auto result = CompareClose({1.2f, 2.0f}, {1.0f, 2.0f}, 0.01, 0.0);
  Expect(!result.passed, "close comparison fails outside tolerance");
  Expect(result.first_failing_index.has_value() &&
             *result.first_failing_index == 0,
         "close comparison first failing index");
  Expect(result.max_abs_error > 0.19, "close comparison max abs error");
}

void TestCompareCloseRejectsNaN() {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const auto result = CompareClose({nan}, {nan}, 0.0, 0.0);
  Expect(!result.passed, "close comparison rejects nan");
  Expect(result.first_failing_index.has_value(), "nan failure index");
}

void TestIreeRunLoomBridgeRejectsConfigWithoutLinkedOutput() {
  const auto parsed = ParseArgs(ValidArgs());
  Expect(parsed.invocation.has_value(), "valid command parses for bridge");
  if (!parsed.invocation.has_value()) {
    return;
  }
  const auto command = BuildIreeRunLoomCommand(*parsed.invocation);
  Expect(!command.args.has_value(), "bridge rejects unstaged config bindings");
  Expect(command.error.find("--linked-kernel-output") != std::string::npos,
         "bridge linked output rejection error");
}

void TestIreeRunLoomBridgeStagesConfigKernel() {
  const std::filesystem::path dir = TempDir();
  WriteF32Npy(dir / "src0.npy", {1.0f, 2.0f, 3.0f, 4.0f});
  WriteF32Npy(dir / "dst_init.npy", {0.0f, 0.0f, 0.0f, 0.0f});
  WriteF32Npy(dir / "expected.npy", {2.0f, 3.0f, 4.0f, 5.0f});
  auto args = std::vector<std::string>{
      "--kernel",
      "kernels/v2/add_f32.loom",
      "--root",
      "@add_f32",
      "--target",
      "gfx1100",
      "--config",
      "BLOCK_SIZE=256",
      "--loom-link",
      "/tmp/loom-link",
      "--linked-kernel-output",
      (dir / "linked.loom").string(),
      "--iree-run-loom",
      "/tmp/iree-run-loom",
      "--binding",
      "0:input:f32:4:" + (dir / "src0.npy").string(),
      "--binding",
      "1:output:f32:4:" + (dir / "dst_init.npy").string(),
      "--expect",
      "1:close:" + (dir / "expected.npy").string() + ":1e-5:1e-5",
      "--output",
      "result.json",
  };
  const auto parsed = ParseArgs(args);
  Expect(parsed.invocation.has_value(), "config bridge command parses");
  if (!parsed.invocation.has_value()) {
    return;
  }
  const auto command = BuildIreeRunLoomCommand(*parsed.invocation);
  Expect(command.loom_link_args.has_value(), "link command is staged");
  Expect(command.args.has_value(), "staged bridge command builds");
  if (!command.loom_link_args.has_value() || !command.args.has_value()) {
    std::cerr << "bridge error: " << command.error << "\n";
    return;
  }
  Expect((*command.loom_link_args)[0] == "/tmp/loom-link", "link command tool");
  Expect(std::find(command.loom_link_args->begin(),
                   command.loom_link_args->end(),
                   "--config=BLOCK_SIZE=256") != command.loom_link_args->end(),
         "link command config");
  Expect(std::find(command.loom_link_args->begin(),
                   command.loom_link_args->end(),
                   "--output=" + (dir / "linked.loom").string()) !=
             command.loom_link_args->end(),
         "link command output");
  Expect((*command.args)[1] == (dir / "linked.loom.target.loom").string(),
         "run command uses targeted linked kernel");
}

void TestIreeRunLoomBridgeBuildsNpyBackedCommand() {
  const std::filesystem::path dir = TempDir();
  WriteF32Npy(dir / "src0.npy", {1.0f, 2.0f, 3.0f, 4.0f});
  WriteF32Npy(dir / "dst_init.npy", {0.0f, 0.0f, 0.0f, 0.0f});
  WriteF32Npy(dir / "expected.npy", {2.0f, 3.0f, 4.0f, 5.0f});
  auto args = std::vector<std::string>{
      "--kernel",
      "linked.loom",
      "--root",
      "@add_f32",
      "--target",
      "gfx1100",
      "--workgroup-count",
      "4,1,1",
      "--iree-run-loom",
      "/tmp/iree-run-loom",
      "--binding",
      "0:input:f32:4:" + (dir / "src0.npy").string(),
      "--binding",
      "1:output:f32:4:" + (dir / "dst_init.npy").string(),
      "--expect",
      "1:close:" + (dir / "expected.npy").string() + ":1e-5:1e-5",
      "--output",
      "result.json",
  };
  const auto parsed = ParseArgs(args);
  Expect(parsed.invocation.has_value(), "npy bridge command parses");
  if (!parsed.invocation.has_value()) {
    return;
  }
  const auto command = BuildIreeRunLoomCommand(*parsed.invocation);
  Expect(command.args.has_value(), "npy bridge command builds");
  if (!command.args.has_value()) {
    std::cerr << "bridge error: " << command.error << "\n";
    return;
  }
  Expect(!command.loom_link_args.has_value(),
         "no link command for pre-linked kernels");
  Expect((*command.args)[0] == "/tmp/iree-run-loom", "bridge command tool");
  Expect(std::find(command.args->begin(), command.args->end(),
                   "--backend=amdgpu-hal") != command.args->end(),
         "bridge command backend");
  Expect(std::find(command.args->begin(), command.args->end(),
                   "--target=gfx1100") == command.args->end(),
         "bridge command does not require iree-run-loom target support");
  Expect(std::find(command.args->begin(), command.args->end(),
                   "--workgroup-count=4,1,1") != command.args->end(),
         "bridge command workgroup count");
  Expect(std::find(command.args->begin(), command.args->end(),
                   "--kernel-input-buffer=&@" + (dir / "src0.npy").string()) !=
             command.args->end(),
         "bridge command input spec");
  Expect(std::find(command.args->begin(), command.args->end(),
                   "--expected-kernel-buffer=@" +
                       (dir / "expected.npy").string()) != command.args->end(),
         "bridge command expected spec");
  Expect(std::find(command.args->begin(), command.args->end(),
                   "--expected-kernel-buffer-tolerance=0,0") !=
             command.args->end(),
         "bridge command exact input tolerance");
  Expect(std::find(command.args->begin(), command.args->end(),
                   "--expected-kernel-buffer-tolerance=1e-05,1e-05") !=
             command.args->end(),
         "bridge command output tolerance");
}

void TestIreeRunLoomBridgeBuildsF16NpyBackedCommand() {
  const std::filesystem::path dir = TempDir();
  WriteI16Npy(dir / "src0.npy", {0x3C00, 0x4000, 0x4200, 0x4400});
  WriteI16Npy(dir / "dst_init.npy", {0, 0, 0, 0});
  WriteI16Npy(dir / "expected.npy", {0x3C00, 0x4000, 0x4200, 0x4400});
  auto args = std::vector<std::string>{
      "--kernel",        "linked.loom",
      "--root",          "@add_f16",
      "--target",        "gfx1100",
      "--iree-run-loom", "/tmp/iree-run-loom",
      "--binding",       "0:input:f16:4:" + (dir / "src0.npy").string(),
      "--binding",       "1:output:f16:4:" + (dir / "dst_init.npy").string(),
      "--expect",        "1:close:" + (dir / "expected.npy").string() + ":0:0",
      "--output",        "result.json",
  };
  const auto parsed = ParseArgs(args);
  Expect(parsed.invocation.has_value(), "f16 npy bridge command parses");
  if (!parsed.invocation.has_value()) {
    return;
  }
  const auto command = BuildIreeRunLoomCommand(*parsed.invocation);
  Expect(command.args.has_value(), "f16 npy bridge command builds");
  if (!command.args.has_value()) {
    std::cerr << "bridge error: " << command.error << "\n";
    return;
  }
  Expect(std::find(command.args->begin(), command.args->end(),
                   "--kernel-input-buffer=&@" + (dir / "src0.npy").string()) !=
             command.args->end(),
         "f16 bridge command input spec");
  Expect(std::find(command.args->begin(), command.args->end(),
                   "--expected-kernel-buffer=@" +
                       (dir / "expected.npy").string()) != command.args->end(),
         "f16 bridge command expected spec");
  Expect(std::find(command.args->begin(), command.args->end(),
                   "--expected-kernel-buffer-tolerance=0,0") !=
             command.args->end(),
         "f16 bridge command exact tolerance");
}

void TestIreeRunLoomBridgeBuildsI32NpyBackedCommand() {
  const std::filesystem::path dir = TempDir();
  WriteF32Npy(dir / "src0.npy", {1.0f, 2.0f, 3.0f, 4.0f});
  WriteI32Npy(dir / "indices.npy", {0, 1});
  WriteF32Npy(dir / "dst_init.npy", {0.0f, 0.0f});
  WriteF32Npy(dir / "expected.npy", {1.0f, 2.0f});
  auto args = std::vector<std::string>{
      "--kernel",
      "linked.loom",
      "--root",
      "@get_rows_f32",
      "--target",
      "gfx1100",
      "--iree-run-loom",
      "/tmp/iree-run-loom",
      "--binding",
      "0:input:f32:4:" + (dir / "src0.npy").string(),
      "--binding",
      "1:input:i32:2:" + (dir / "indices.npy").string(),
      "--binding",
      "2:output:f32:2:" + (dir / "dst_init.npy").string(),
      "--expect",
      "2:close:" + (dir / "expected.npy").string() + ":1e-5:1e-5",
      "--output",
      "result.json",
  };
  const auto parsed = ParseArgs(args);
  Expect(parsed.invocation.has_value(), "i32 npy bridge command parses");
  if (!parsed.invocation.has_value()) {
    return;
  }
  const auto command = BuildIreeRunLoomCommand(*parsed.invocation);
  Expect(command.args.has_value(), "i32 npy bridge command builds");
  if (!command.args.has_value()) {
    std::cerr << "bridge error: " << command.error << "\n";
    return;
  }
  Expect(std::find(command.args->begin(), command.args->end(),
                   "--kernel-input-buffer=&@" +
                       (dir / "indices.npy").string()) != command.args->end(),
         "i32 bridge command input spec");
}

void TestIreeRunLoomBridgeAcceptsNonSplatTensor() {
  const std::filesystem::path dir = TempDir();
  WriteF32Npy(dir / "src0.npy", {1.0f, 2.0f, 1.0f, 1.0f});
  WriteF32Npy(dir / "dst_init.npy", {0.0f, 0.0f, 0.0f, 0.0f});
  WriteF32Npy(dir / "expected.npy", {2.0f, 2.0f, 2.0f, 2.0f});
  auto args = std::vector<std::string>{
      "--kernel",  "linked.loom",
      "--root",    "@add_f32",
      "--target",  "gfx1100",
      "--binding", "0:input:f32:4:" + (dir / "src0.npy").string(),
      "--binding", "1:output:f32:4:" + (dir / "dst_init.npy").string(),
      "--expect",  "1:close:" + (dir / "expected.npy").string() + ":1e-5:1e-5",
      "--output",  "result.json",
  };
  const auto parsed = ParseArgs(args);
  Expect(parsed.invocation.has_value(), "non-splat bridge command parses");
  if (!parsed.invocation.has_value()) {
    return;
  }
  const auto command = BuildIreeRunLoomCommand(*parsed.invocation);
  Expect(command.args.has_value(), "non-splat npy bridge command accepted");
  if (!command.args.has_value()) {
    std::cerr << "bridge error: " << command.error << "\n";
  }
}

void TestIreeRunLoomBridgeBuildsScalarCommand() {
  const std::filesystem::path dir = TempDir();
  WriteF32Npy(dir / "src0.npy", {1.0f, 2.0f, 3.0f, 4.0f});
  WriteF32Npy(dir / "dst_init.npy", {0.0f, 0.0f, 0.0f, 0.0f});
  WriteF32Npy(dir / "expected.npy", {2.0f, 3.0f, 4.0f, 5.0f});
  auto args = std::vector<std::string>{
      "--kernel",  "linked.loom",
      "--root",    "@scale_f32",
      "--target",  "gfx1100",
      "--scalar",  "0:f32:0.5",
      "--binding", "1:input:f32:4:" + (dir / "src0.npy").string(),
      "--binding", "2:output:f32:4:" + (dir / "dst_init.npy").string(),
      "--expect",  "2:close:" + (dir / "expected.npy").string() + ":1e-5:1e-5",
      "--output",  "result.json",
  };
  const auto parsed = ParseArgs(args);
  Expect(parsed.invocation.has_value(), "scalar bridge command parses");
  if (!parsed.invocation.has_value()) {
    return;
  }
  const auto command = BuildIreeRunLoomCommand(*parsed.invocation);
  Expect(command.args.has_value(), "scalar bridge command builds");
  if (!command.args.has_value()) {
    std::cerr << "bridge error: " << command.error << "\n";
    return;
  }
  Expect(std::find(command.args->begin(), command.args->end(),
                   "--kernel-input-value=f32=0.5") != command.args->end(),
         "bridge command scalar spec");
}

void TestExecuteBridgeRunsNoConfigCommand() {
  const std::filesystem::path dir = TempDir();
  WriteTextFile(dir / "linked.loom",
                "kernel.def export(\"add_f32\") @add_f32() {\n"
                "  %one = index.constant 1 : index\n"
                "  kernel.launch.config workgroups(%one, %one, %one) "
                "workgroup_size(%one, %one, %one) : index\n"
                "} launch(%src: buffer, %dst: buffer) {\n"
                "  kernel.return\n"
                "}\n");
  WriteF32Npy(dir / "src0.npy", {1.0f, 2.0f, 3.0f, 4.0f});
  WriteF32Npy(dir / "dst_init.npy", {0.0f, 0.0f, 0.0f, 0.0f});
  WriteF32Npy(dir / "expected.npy", {2.0f, 3.0f, 4.0f, 5.0f});
  WriteExecutableScript(dir / "iree-run-loom", "echo run-tool \"$@\"\n"
                                               "exit 0\n");

  auto args = std::vector<std::string>{
      "--kernel",
      (dir / "linked.loom").string(),
      "--root",
      "@add_f32",
      "--target",
      "gfx1100",
      "--iree-run-loom",
      (dir / "iree-run-loom").string(),
      "--binding",
      "0:input:f32:4:" + (dir / "src0.npy").string(),
      "--binding",
      "1:output:f32:4:" + (dir / "dst_init.npy").string(),
      "--expect",
      "1:close:" + (dir / "expected.npy").string() + ":1e-5:1e-5",
      "--output",
      (dir / "result.json").string(),
      "--execute-iree-run-loom-command",
  };
  const auto parsed = ParseArgs(args);
  Expect(parsed.invocation.has_value(), "executable bridge command parses");
  if (!parsed.invocation.has_value()) {
    return;
  }

  const std::string json = RenderResultJson(*parsed.invocation);
  Expect(json.find("\"status\": \"run_passed\"") != std::string::npos,
         "executed bridge reports pass");
  Expect(json.find("\"status\": \"passed\"") != std::string::npos,
         "executed bridge updates check status");
  Expect(json.find("\"run_exit_code\": 0") != std::string::npos,
         "executed bridge records run exit code");
  Expect(json.find("run-tool " + (dir / "result.json.target.loom").string()) !=
             std::string::npos,
         "executed bridge captures run output");
  const std::string targeted_source =
      ReadTextFile(dir / "result.json.target.loom");
  Expect(targeted_source.find("amdgpu.target<gfx1100> "
                              "@ggml_hrx_run_loom_simple_target") !=
             std::string::npos,
         "targeted source declares amdgpu target");
  Expect(targeted_source.find(
             "kernel.def target(@ggml_hrx_run_loom_simple_target) "
             "export(\"add_f32\") @add_f32") != std::string::npos,
         "targeted source annotates selected kernel");
}

void TestExecuteBridgeStopsOnLinkFailure() {
  const std::filesystem::path dir = TempDir();
  WriteF32Npy(dir / "src0.npy", {1.0f, 2.0f, 3.0f, 4.0f});
  WriteF32Npy(dir / "dst_init.npy", {0.0f, 0.0f, 0.0f, 0.0f});
  WriteF32Npy(dir / "expected.npy", {2.0f, 3.0f, 4.0f, 5.0f});
  WriteExecutableScript(dir / "loom-link", "echo link-failed \"$@\"\n"
                                           "exit 7\n");
  WriteExecutableScript(dir / "iree-run-loom", "echo run-should-not-execute\n"
                                               "exit 0\n");

  auto args = std::vector<std::string>{
      "--kernel",
      "kernels/v2/add_f32.loom",
      "--root",
      "@add_f32",
      "--target",
      "gfx1100",
      "--config",
      "BLOCK_SIZE=256",
      "--loom-link",
      (dir / "loom-link").string(),
      "--linked-kernel-output",
      (dir / "linked.loom").string(),
      "--iree-run-loom",
      (dir / "iree-run-loom").string(),
      "--binding",
      "0:input:f32:4:" + (dir / "src0.npy").string(),
      "--binding",
      "1:output:f32:4:" + (dir / "dst_init.npy").string(),
      "--expect",
      "1:close:" + (dir / "expected.npy").string() + ":1e-5:1e-5",
      "--output",
      "result.json",
      "--execute-iree-run-loom-command",
  };
  const auto parsed = ParseArgs(args);
  Expect(parsed.invocation.has_value(), "link-failure bridge command parses");
  if (!parsed.invocation.has_value()) {
    return;
  }

  const std::string json = RenderResultJson(*parsed.invocation);
  Expect(json.find("\"status\": \"link_failed\"") != std::string::npos,
         "executed bridge reports link failure");
  Expect(json.find("\"loom_link_exit_code\": 7") != std::string::npos,
         "executed bridge records link exit code");
  Expect(json.find("link-failed kernels/v2/add_f32.loom") != std::string::npos,
         "executed bridge captures link output");
  Expect(json.find("run-should-not-execute") == std::string::npos,
         "run command is skipped after link failure");
}

} // namespace

int main() {
  TestParsesValidCommand();
  TestRejectsMissingKernel();
  TestRejectsMissingRoot();
  TestRejectsMalformedConfig();
  TestRejectsMalformedWorkgroupCount();
  TestRejectsMalformedBinding();
  TestRejectsDuplicateBindingPosition();
  TestParsesScalarCommand();
  TestRejectsUnsupportedDType();
  TestParsesF16BindingDType();
  TestParsesI32BindingDType();
  TestRejectsUnsupportedKind();
  TestRejectsExpectForInputBinding();
  TestParsesMultipleExpectations();
  TestRenderNotRunJson();
  TestParsesIreeRunLoomBridgeFlags();
  TestLoadsF32NpyV1();
  TestLoadsF32NpyV2();
  TestValidatesF16StorageNpy();
  TestRejectsF16StorageDTypeMismatch();
  TestValidatesI32StorageNpy();
  TestRejectsI32StorageDTypeMismatch();
  TestRejectsMissingNpyFile();
  TestRejectsInvalidNpyMagic();
  TestRejectsNpyDTypeMismatch();
  TestRejectsNpyShapeMismatch();
  TestRejectsTwoDimensionalNpy();
  TestRejectsFortranOrderNpy();
  TestCompareClosePassesExactValues();
  TestCompareClosePassesWithinTolerance();
  TestCompareCloseFailsOutsideTolerance();
  TestCompareCloseRejectsNaN();
  TestIreeRunLoomBridgeRejectsConfigWithoutLinkedOutput();
  TestIreeRunLoomBridgeStagesConfigKernel();
  TestIreeRunLoomBridgeBuildsNpyBackedCommand();
  TestIreeRunLoomBridgeBuildsF16NpyBackedCommand();
  TestIreeRunLoomBridgeBuildsI32NpyBackedCommand();
  TestIreeRunLoomBridgeAcceptsNonSplatTensor();
  TestIreeRunLoomBridgeBuildsScalarCommand();
  TestExecuteBridgeRunsNoConfigCommand();
  TestExecuteBridgeStopsOnLinkFailure();

  if (g_failures != 0) {
    std::cerr << g_failures << " test failure(s)\n";
    return 1;
  }
  return 0;
}
