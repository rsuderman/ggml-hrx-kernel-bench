#include "tools/run-loom-simple/run_loom_simple.h"

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char **argv) {
  std::vector<std::string> args;
  args.reserve(static_cast<std::size_t>(argc > 0 ? argc - 1 : 0));
  for (int i = 1; i < argc; ++i) {
    args.emplace_back(argv[i]);
  }

  ggml_hrx::run_loom_simple::ParseResult parse_result =
      ggml_hrx::run_loom_simple::ParseArgs(args);
  if (!parse_result.invocation.has_value()) {
    for (const std::string &error : parse_result.errors) {
      std::cerr << "error: " << error << "\n";
    }
    return 2;
  }

  const std::string json =
      ggml_hrx::run_loom_simple::RenderResultJson(*parse_result.invocation);
  std::ofstream output(parse_result.invocation->output_path);
  if (!output) {
    std::cerr << "error: failed to open output path: "
              << parse_result.invocation->output_path << "\n";
    return 1;
  }
  output << json;
  if (!output) {
    std::cerr << "error: failed to write output path: "
              << parse_result.invocation->output_path << "\n";
    return 1;
  }
  return 0;
}
