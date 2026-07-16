#pragma once

#include <iosfwd>
#include <string>
#include <vector>

namespace ggml_hrx::v2_route_selector_cli {

int Run(const std::vector<std::string> &args, std::istream &standard_input,
        std::ostream &standard_output, std::ostream &standard_error);

} // namespace ggml_hrx::v2_route_selector_cli
