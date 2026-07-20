#include "ggml_hrx/v2_route_selector/query_parser.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <istream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace ggml_hrx::routing::v2::query_parser {
namespace {

using Json = nlohmann::json;

class InputError : public std::runtime_error {
 public:
  explicit InputError(std::string message)
      : std::runtime_error(std::move(message)) {}
};

class LexicalIntegerRangeSax final : public Json::json_sax_t {
 public:
  bool null() override { return true; }
  bool boolean(bool) override { return true; }
  bool number_integer(number_integer_t) override { return true; }
  bool number_unsigned(number_unsigned_t) override { return true; }

  bool number_float(number_float_t, const string_t& token) override {
    // nlohmann represents an integer token outside its integer storage range
    // as number_float. Preserve JSON's lexical distinction so such values are
    // not silently accepted as floating-point attributes.
    if (token.find_first_of(".eE") == string_t::npos) {
      integer_out_of_range = true;
      return false;
    }
    return true;
  }

  bool string(string_t&) override { return true; }
  bool binary(binary_t&) override { return true; }
  bool start_object(std::size_t) override { return true; }
  bool key(string_t&) override { return true; }
  bool end_object() override { return true; }
  bool start_array(std::size_t) override { return true; }
  bool end_array() override { return true; }
  bool parse_error(std::size_t, const std::string&,
                   const Json::exception&) override {
    return false;
  }

  bool integer_out_of_range = false;
};

bool HasField(std::initializer_list<std::string_view> fields,
              std::string_view candidate) {
  return std::find(fields.begin(), fields.end(), candidate) != fields.end();
}

void ValidateObjectFields(
    const Json& object, std::string_view path,
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

const Json& RequireFieldType(const Json& object, std::string_view path,
                             std::string_view field, Json::value_t type,
                             std::string_view type_name) {
  const Json& value = object.at(std::string(field));
  if (value.type() != type) {
    throw InputError(std::string(path) + " field '" + std::string(field) +
                     "' must be " + std::string(type_name));
  }
  return value;
}

std::vector<std::int64_t> ParseIntegerArray(const Json& value,
                                            std::string_view path,
                                            std::string_view field) {
  if (!value.is_array()) {
    throw InputError(std::string(path) + " field '" + std::string(field) +
                     "' must be an array");
  }

  std::vector<std::int64_t> result;
  result.reserve(value.size());
  for (std::size_t index = 0; index < value.size(); ++index) {
    const Json& element = value[index];
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

Tensor ParseTensor(const Json& value, const std::string& role) {
  const std::string path = "input tensor '" + role + "'";
  ValidateObjectFields(value, path, {"dtype", "dimensions", "strides"},
                       {"permutation"});

  const Json& dtype = RequireFieldType(value, path, "dtype",
                                       Json::value_t::string, "a string");
  Tensor tensor;
  tensor.dtype = dtype.get<std::string>();
  tensor.dimensions =
      ParseIntegerArray(value.at("dimensions"), path, "dimensions");
  tensor.strides = ParseIntegerArray(value.at("strides"), path, "strides");
  if (tensor.dimensions.size() != tensor.strides.size()) {
    throw InputError(path +
                     " dimensions and strides must have equal length");
  }
  const auto permutation = value.find("permutation");
  if (permutation != value.end() && !permutation->is_null()) {
    tensor.permutation =
        ParseIntegerArray(*permutation, path, "permutation");
  }
  return tensor;
}

std::vector<std::string> ParseAllowedRouteIds(const Json& value) {
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

AttributeValue ParseAttributeValue(const Json& value,
                                   const std::string& path) {
  AttributeValue result;
  if (value.is_null()) {
    result.value = nullptr;
    return result;
  }
  if (value.is_boolean()) {
    result.value = value.get<bool>();
    return result;
  }
  if (value.is_number_unsigned()) {
    const auto unsigned_value = value.get<std::uint64_t>();
    if (unsigned_value >
        static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
      throw InputError(path +
                       " is outside the signed 64-bit integer range");
    }
    result.value = static_cast<std::int64_t>(unsigned_value);
    return result;
  }
  if (value.is_number_integer()) {
    result.value = value.get<std::int64_t>();
    return result;
  }
  if (value.is_number_float()) {
    const auto number = value.get<double>();
    if (!std::isfinite(number)) {
      throw InputError(path + " must be a finite floating-point number");
    }
    result.value = number;
    return result;
  }
  if (value.is_string()) {
    result.value = value.get<std::string>();
    return result;
  }
  if (value.is_array()) {
    AttributeArray array;
    array.reserve(value.size());
    for (std::size_t index = 0; index < value.size(); ++index) {
      array.push_back(ParseAttributeValue(
          value[index], path + " element " + std::to_string(index)));
    }
    result.value = std::move(array);
    return result;
  }
  if (value.is_object()) {
    AttributeObject object;
    for (auto field = value.begin(); field != value.end(); ++field) {
      object.emplace(field.key(),
                     ParseAttributeValue(field.value(), path + " field '" +
                                                            field.key() + "'"));
    }
    result.value = std::move(object);
    return result;
  }
  throw InputError(path + " has an unsupported JSON type");
}

ParsedQuery ParseQuery(const Json& input) {
  ValidateObjectFields(input, "input", {"op", "tensors"},
                       {"attributes", "allowed_route_ids"});
  const Json& op = RequireFieldType(input, "input", "op",
                                    Json::value_t::string, "a string");
  const Json& tensors = RequireFieldType(input, "input", "tensors",
                                         Json::value_t::object, "an object");

  ParsedQuery parsed;
  parsed.op = op.get<std::string>();
  for (auto tensor = tensors.begin(); tensor != tensors.end(); ++tensor) {
    parsed.query.tensors.emplace(
        tensor.key(), ParseTensor(tensor.value(), tensor.key()));
  }

  const auto attributes = input.find("attributes");
  if (attributes != input.end()) {
    if (!attributes->is_object()) {
      throw InputError("input field 'attributes' must be an object");
    }
    for (auto attribute = attributes->begin(); attribute != attributes->end();
         ++attribute) {
      parsed.query.attributes.emplace(
          attribute.key(),
          ParseAttributeValue(attribute.value(),
                              "input attribute '" + attribute.key() + "'"));
    }
  }

  const auto allowed = input.find("allowed_route_ids");
  if (allowed != input.end() && !allowed->is_null()) {
    parsed.query.allowed_route_ids = ParseAllowedRouteIds(*allowed);
  }
  return parsed;
}

}  // namespace

ParseResult parse(std::istream& input) {
  const std::string input_text(std::istreambuf_iterator<char>(input), {});
  Json document;
  try {
    document = Json::parse(input_text);
  } catch (const Json::out_of_range&) {
    return ParseError{ParseErrorKind::number_out_of_range,
                      "input contains a number outside the supported range"};
  } catch (const Json::exception&) {
    return ParseError{ParseErrorKind::malformed_json, "malformed JSON"};
  }

  LexicalIntegerRangeSax lexical_number_validator;
  Json::sax_parse(input_text, &lexical_number_validator);
  if (lexical_number_validator.integer_out_of_range) {
    return ParseError{
        ParseErrorKind::schema,
        "input contains an integer outside the signed 64-bit range"};
  }

  try {
    return ParseQuery(document);
  } catch (const InputError& error) {
    return ParseError{ParseErrorKind::schema, error.what()};
  } catch (const Json::out_of_range&) {
    return ParseError{ParseErrorKind::number_out_of_range,
                      "input contains a number outside the supported range"};
  }
}

}  // namespace ggml_hrx::routing::v2::query_parser
