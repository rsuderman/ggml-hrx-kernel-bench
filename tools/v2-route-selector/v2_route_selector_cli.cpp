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

#include <nlohmann/json.hpp>

namespace ggml_hrx::v2_route_selector_cli {
namespace {

namespace selector = ggml_hrx::routing::v2;
namespace query_parser = ggml_hrx::routing::v2::query_parser;

constexpr std::string_view kUsage =
    "Usage: ggml-hrx-v2-route-selector --input <file|-> "
    "[--batch] [--expect-route <route-id>]\n";

struct Arguments {
  std::string input_path;
  std::optional<std::string> expected_route;
  bool batch = false;
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
  bool has_batch = false;
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
    if (argument == "--batch") {
      if (has_batch) {
        ArgumentParseResult result;
        result.error = "duplicate option --batch";
        return result;
      }
      parsed.batch = true;
      has_batch = true;
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
  if (parsed.batch && parsed.expected_route.has_value()) {
    ArgumentParseResult result;
    result.error = "--batch cannot be combined with --expect-route";
    return result;
  }
  ArgumentParseResult result;
  result.arguments = std::move(parsed);
  return result;
}

struct SelectionOutcome {
  selector::SelectionStatus status = selector::SelectionStatus::no_match;
  std::string route_id;
  std::string diagnostic;
};

SelectionOutcome EvaluateRoute(const std::string &operation,
                               const selector::Query &query) {
  if (selector::supported_route_ids(operation).empty()) {
    return {selector::SelectionStatus::unsupported, {},
            "operation '" + operation + "' is not supported"};
  }

  const selector::Selection selection = selector::select(operation, query);
  if (selection.status == selector::SelectionStatus::no_match) {
    return {selection.status, {},
            "no route matched operation '" + operation + "'"};
  }
  if (selection.status == selector::SelectionStatus::unsupported) {
    return {selection.status, {},
            "selector cannot evaluate operation '" + operation + "'"};
  }
  return {selection.status, selection.route_id, {}};
}

int SelectRoute(const std::string &operation, const selector::Query &query,
                const std::optional<std::string> &expected_route,
                std::ostream &output_stream, std::ostream &error_stream) {
  const SelectionOutcome outcome = EvaluateRoute(operation, query);
  if (outcome.status != selector::SelectionStatus::match) {
    return ReportError(error_stream, 1,
                       std::string(selector::status_name(outcome.status)) +
                           ": " + outcome.diagnostic);
  }
  if (expected_route.has_value() && outcome.route_id != *expected_route) {
    return ReportError(error_stream, 1,
                       "expected route '" + *expected_route +
                           "' but selected '" + outcome.route_id + "'");
  }

  output_stream << outcome.route_id << '\n';
  return 0;
}

int RunSingleWithInput(const Arguments &arguments, std::istream &input_stream,
                       std::ostream &output_stream,
                       std::ostream &error_stream) {
  query_parser::ParseResult parsed = query_parser::parse(input_stream);
  if (const auto *error = std::get_if<query_parser::ParseError>(&parsed)) {
    return ReportError(error_stream, 2, error->diagnostic);
  }

  auto &query = std::get<query_parser::ParsedQuery>(parsed);
  return SelectRoute(query.op, query.query, arguments.expected_route,
                     output_stream, error_stream);
}

bool IsBlankLine(std::string_view line) {
  return std::all_of(line.begin(), line.end(), [](const char ch) {
    return ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n' ||
           ch == '\v' || ch == '\f';
  });
}

std::string JsonString(std::string_view value) {
  return nlohmann::json(std::string(value)).dump();
}

bool WriteBatchMatch(std::size_t line_number, std::string_view route_id,
                     std::ostream &output_stream) {
  output_stream << "{\"line\":" << line_number
                << ",\"status\":\"MATCH\",\"route_id\":"
                << JsonString(route_id) << "}\n";
  return static_cast<bool>(output_stream);
}

bool WriteBatchFailure(std::size_t line_number, std::string_view status,
                       std::string_view diagnostic,
                       std::ostream &output_stream) {
  output_stream << "{\"line\":" << line_number
                << ",\"status\":" << JsonString(status)
                << ",\"diagnostic\":" << JsonString(diagnostic) << "}\n";
  return static_cast<bool>(output_stream);
}

int RunBatchWithInput(std::istream &input_stream, std::ostream &output_stream,
                      std::ostream &error_stream) {
  std::size_t line_number = 0;
  std::string line;
  while (std::getline(input_stream, line)) {
    ++line_number;
    if (IsBlankLine(line)) {
      continue;
    }

    query_parser::ParseResult parsed = query_parser::parse(line);
    if (const auto *error = std::get_if<query_parser::ParseError>(&parsed)) {
      if (!WriteBatchFailure(line_number, "ERROR", error->diagnostic,
                             output_stream)) {
        return ReportError(error_stream, 2, "failed while writing output");
      }
      continue;
    }

    const auto &query = std::get<query_parser::ParsedQuery>(parsed);
    const SelectionOutcome outcome = EvaluateRoute(query.op, query.query);
    const bool write_succeeded =
        outcome.status == selector::SelectionStatus::match
            ? WriteBatchMatch(line_number, outcome.route_id, output_stream)
            : WriteBatchFailure(line_number,
                                selector::status_name(outcome.status),
                                outcome.diagnostic, output_stream);
    if (!write_succeeded) {
      return ReportError(error_stream, 2, "failed while writing output");
    }
  }

  if (input_stream.bad() || (input_stream.fail() && !input_stream.eof())) {
    return ReportError(error_stream, 2, "failed while reading input");
  }
  output_stream.flush();
  if (!output_stream) {
    return ReportError(error_stream, 2, "failed while writing output");
  }
  return 0;
}

int RunWithInput(const Arguments &arguments, std::istream &input_stream,
                 std::ostream &output_stream, std::ostream &error_stream) {
  if (arguments.batch) {
    return RunBatchWithInput(input_stream, output_stream, error_stream);
  }
  return RunSingleWithInput(arguments, input_stream, output_stream,
                            error_stream);
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
