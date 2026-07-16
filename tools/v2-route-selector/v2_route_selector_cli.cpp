#include "tools/v2-route-selector/v2_route_selector_cli.h"

#include "ggml_hrx/v2_route_selector/selector.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace ggml_hrx::v2_route_selector_cli {
namespace {

using Json = nlohmann::json;
namespace selector = ggml_hrx::routing::v2;

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

class InputError : public std::runtime_error {
public:
  explicit InputError(std::string message)
      : std::runtime_error(std::move(message)) {}
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

bool HasField(std::initializer_list<std::string_view> fields,
              std::string_view candidate) {
  return std::find(fields.begin(), fields.end(), candidate) != fields.end();
}

void ValidateObjectFields(const Json &object, std::string_view path,
                          std::initializer_list<std::string_view> required,
                          std::initializer_list<std::string_view> optional = {}) {
  if (!object.is_object()) {
    throw InputError(std::string(path) + " must be an object");
  }

  for (auto field = object.begin(); field != object.end(); ++field) {
    if (!HasField(required, field.key()) &&
        !HasField(optional, field.key())) {
      throw InputError(std::string(path) + " contains unknown field '" +
                       field.key() + "'");
    }
  }
  for (const std::string_view field : required) {
    if (object.find(std::string(field)) == object.end()) {
      throw InputError(std::string(path) + " is missing required field '" +
                       std::string(field) + "'");
    }
  }
}

const Json &RequireFieldType(const Json &object, std::string_view path,
                             std::string_view field, Json::value_t type,
                             std::string_view type_name) {
  const Json &value = object.at(std::string(field));
  if (value.type() != type) {
    throw InputError(std::string(path) + " field '" + std::string(field) +
                     "' must be " + std::string(type_name));
  }
  return value;
}

std::vector<std::int64_t>
ParseIntegerArray(const Json &value, std::string_view path,
                  std::string_view field) {
  if (!value.is_array()) {
    throw InputError(std::string(path) + " field '" + std::string(field) +
                     "' must be an array");
  }

  std::vector<std::int64_t> result;
  result.reserve(value.size());
  for (std::size_t index = 0; index < value.size(); ++index) {
    const Json &element = value[index];
    if (element.is_number_unsigned()) {
      const auto unsigned_value = element.get<std::uint64_t>();
      if (unsigned_value >
          static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
        throw InputError(std::string(path) + " field '" +
                         std::string(field) + "' element " +
                         std::to_string(index) +
                         " is outside the signed 64-bit integer range");
      }
      result.push_back(static_cast<std::int64_t>(unsigned_value));
      continue;
    }
    if (element.is_number_integer()) {
      result.push_back(element.get<std::int64_t>());
      continue;
    }
    throw InputError(std::string(path) + " field '" + std::string(field) +
                     "' element " + std::to_string(index) +
                     " must be a signed 64-bit integer");
  }
  return result;
}

selector::Tensor ParseTensor(const Json &value, const std::string &role) {
  const std::string path = "input tensor '" + role + "'";
  ValidateObjectFields(value, path, {"dtype", "dimensions", "strides"},
                       {"permutation"});

  const Json &dtype = RequireFieldType(value, path, "dtype",
                                       Json::value_t::string, "a string");
  selector::Tensor tensor;
  tensor.dtype = dtype.get<std::string>();
  tensor.dimensions = ParseIntegerArray(value.at("dimensions"), path,
                                        "dimensions");
  tensor.strides = ParseIntegerArray(value.at("strides"), path, "strides");
  if (tensor.dimensions.size() != tensor.strides.size()) {
    throw InputError(path +
                     " dimensions and strides must have equal length");
  }
  const auto permutation = value.find("permutation");
  if (permutation != value.end() && !permutation->is_null()) {
    tensor.permutation = ParseIntegerArray(*permutation, path, "permutation");
  }
  return tensor;
}

std::vector<std::string> ParseAllowedRouteIds(const Json &value) {
  if (!value.is_array()) {
    throw InputError(
        "input field 'allowed_route_ids' must be null or an array of strings");
  }

  std::vector<std::string> result;
  result.reserve(value.size());
  for (std::size_t index = 0; index < value.size(); ++index) {
    if (!value[index].is_string()) {
      throw InputError("input field 'allowed_route_ids' element " +
                       std::to_string(index) + " must be a string");
    }
    result.push_back(value[index].get<std::string>());
  }
  return result;
}

std::pair<std::string, selector::Query> ParseQuery(const Json &input) {
  ValidateObjectFields(input, "input", {"op", "tensors"},
                       {"allowed_route_ids"});
  const Json &op = RequireFieldType(input, "input", "op",
                                    Json::value_t::string, "a string");
  const Json &tensors = RequireFieldType(input, "input", "tensors",
                                         Json::value_t::object, "an object");

  selector::Query query;
  for (auto tensor = tensors.begin(); tensor != tensors.end(); ++tensor) {
    query.tensors.emplace(tensor.key(), ParseTensor(tensor.value(), tensor.key()));
  }

  const auto allowed = input.find("allowed_route_ids");
  if (allowed != input.end() && !allowed->is_null()) {
    query.allowed_route_ids = ParseAllowedRouteIds(*allowed);
  }
  return {op.get<std::string>(), std::move(query)};
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
  Json input;
  try {
    input = Json::parse(input_stream);
  } catch (const Json::out_of_range &) {
    return ReportError(error_stream, 2,
                       "input contains a number outside the supported range");
  } catch (const Json::exception &) {
    return ReportError(error_stream, 2, "malformed JSON");
  }

  try {
    auto [operation, query] = ParseQuery(input);
    return SelectRoute(operation, query, arguments.expected_route,
                       output_stream, error_stream);
  } catch (const InputError &error) {
    return ReportError(error_stream, 2, error.what());
  }
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
