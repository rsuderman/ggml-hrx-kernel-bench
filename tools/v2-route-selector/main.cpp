#include "tools/v2-route-selector/v2_route_selector_cli.h"

#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char **argv) {
  std::vector<std::string> args;
  args.reserve(static_cast<std::size_t>(argc > 0 ? argc - 1 : 0));
  for (int i = 1; i < argc; ++i) {
    args.emplace_back(argv[i]);
  }

  return ggml_hrx::v2_route_selector_cli::Run(args, std::cin, std::cout,
                                               std::cerr);
}
