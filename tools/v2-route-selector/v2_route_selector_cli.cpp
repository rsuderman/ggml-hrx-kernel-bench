#include "tools/v2-route-selector/v2_route_selector_cli.h"

#include "ggml_hrx/v2_route_selector/query_parser.h"
#include "ggml_hrx/v2_route_selector/selector.h"

#include <algorithm>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace ggml_hrx::v2_route_selector_cli {
namespace {

namespace selector = ggml_hrx::routing::v2;
namespace query_parser = ggml_hrx::routing::v2::query_parser;

constexpr std::string_view kUsage =
    "Usage: ggml-hrx-v2-route-selector --input <file|-> "
    "[--expect-route <route-id>]\n";

struct Arguments {
  std::string input_path;
  std::optional<std::string> expected_route;
};

struct ArgumentParseResult {
  std::optional<Arguments> arguments;
  bool show_help = false;
  std::string error;
};

int ReportError(std::ostream &error_stream, int exit_code,
                const std::string &message) {
  error_stream << "error: " << message << '\n';
  return exit_code;
}

ArgumentParseResult ParseArguments(const std::vector<std::string> &args) {
  if (std::find(args.begin(), args.end(), "--help") != args.end()) {
    ArgumentParseResult result;
    if (args.size() == 1) {
      result.show_help = true;
    } else {
      result.error = "--help cannot be combined with other options";
    }
    return result;
  }

  Arguments parsed;
  bool has_input = false;
  bool has_expected_route = false;
  for (std::size_t index = 0; index < args.size(); ++index) {
    const std::string &argument = args[index];
    if (argument == "--input") {
      if (has_input) {
        ArgumentParseResult result;
        result.error = "duplicate option --input";
        return result;
      }
      if (index + 1 >= args.size()) {
        ArgumentParseResult result;
        result.error = "missing value for --input";
        return result;
      }
      parsed.input_path = args[++index];
      has_input = true;
      continue;
    }
    if (argument == "--expect-route") {
      if (has_expected_route) {
        ArgumentParseResult result;
        result.error = "duplicate option --expect-route";
        return result;
      }
      if (index + 1 >= args.size()) {
        ArgumentParseResult result;
        result.error = "missing value for --expect-route";
        return result;
      }
      parsed.expected_route = args[++index];
      has_expected_route = true;
      continue;
    }
    ArgumentParseResult result;
    result.error = "unknown option '" + argument + "'";
    return result;
  }

  if (!has_input) {
    ArgumentParseResult result;
    result.error = "missing required --input";
    return result;
  }
  ArgumentParseResult result;
  result.arguments = std::move(parsed);
  return result;
}

int SelectRoute(const std::string &operation, const selector::Query &query,
                const std::optional<std::string> &expected_route,
                std::ostream &output_stream, std::ostream &error_stream) {
  if (selector::supported_route_ids(operation).empty()) {
    return ReportError(error_stream, 1,
                       "UNSUPPORTED: operation '" + operation +
                           "' is not supported");
  }

  const selector::Selection selection = selector::select(operation, query);
  if (selection.status == selector::SelectionStatus::no_match) {
    return ReportError(error_stream, 1,
                       "NO_MATCH: no route matched operation '" + operation +
                           "'");
  }
  if (selection.status == selector::SelectionStatus::unsupported) {
    return ReportError(error_stream, 1,
                       "UNSUPPORTED: selector cannot evaluate operation '" +
                           operation + "'");
  }
  if (expected_route.has_value() && selection.route_id != *expected_route) {
    return ReportError(error_stream, 1,
                       "expected route '" + *expected_route +
                           "' but selected '" + selection.route_id + "'");
  }

  output_stream << selection.route_id << '\n';
  return 0;
}

int RunWithInput(const Arguments &arguments, std::istream &input_stream,
                 std::ostream &output_stream, std::ostream &error_stream) {
  query_parser::ParseResult parsed = query_parser::parse(input_stream);
  if (const auto *error = std::get_if<query_parser::ParseError>(&parsed)) {
    return ReportError(error_stream, 2, error->diagnostic);
  }

  auto &query = std::get<query_parser::ParsedQuery>(parsed);
  return SelectRoute(query.op, query.query, arguments.expected_route,
                     output_stream, error_stream);
}

} // namespace

int Run(const std::vector<std::string> &args, std::istream &standard_input,
        std::ostream &standard_output, std::ostream &standard_error) {
  const ArgumentParseResult parsed = ParseArguments(args);
  if (parsed.show_help) {
    standard_output << kUsage;
    return 0;
  }
  if (!parsed.arguments.has_value()) {
    return ReportError(standard_error, 2, parsed.error);
  }

  if (parsed.arguments->input_path == "-") {
    return RunWithInput(*parsed.arguments, standard_input, standard_output,
                        standard_error);
  }

  std::ifstream input(parsed.arguments->input_path);
  if (!input) {
    return ReportError(standard_error, 2,
                       "cannot read input file '" +
                           parsed.arguments->input_path + "'");
  }
  return RunWithInput(*parsed.arguments, input, standard_output,
                      standard_error);
}

} // namespace ggml_hrx::v2_route_selector_cli
