#include "ggml_hrx/v2_route_selector/query_parser.h"

#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

namespace {

namespace parser = ggml_hrx::routing::v2::query_parser;
using ggml_hrx::routing::v2::AttributeArray;
using ggml_hrx::routing::v2::AttributeObject;

int g_failures = 0;

void Expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++g_failures;
  }
}

parser::ParseResult Parse(const std::string& input) {
  return parser::parse(std::string_view(input));
}

parser::ParseResult ParseStream(const std::string& input) {
  std::istringstream stream(input);
  return parser::parse(stream);
}

const parser::ParsedQuery* ExpectParsed(const parser::ParseResult& result,
                                        const std::string& case_name) {
  const auto* parsed = std::get_if<parser::ParsedQuery>(&result);
  if (parsed == nullptr) {
    const auto& error = std::get<parser::ParseError>(result);
    Expect(false, case_name + ": parse failed: " + error.diagnostic);
  }
  return parsed;
}

void ExpectError(const parser::ParseResult& result,
                 parser::ParseErrorKind expected_kind,
                 const std::string& expected_diagnostic,
                 const std::string& case_name) {
  const auto* error = std::get_if<parser::ParseError>(&result);
  if (error == nullptr) {
    Expect(false, case_name + ": unexpectedly parsed successfully");
    return;
  }
  Expect(error->kind == expected_kind, case_name + ": wrong error kind");
  Expect(error->diagnostic == expected_diagnostic,
         case_name + ": diagnostic was '" + error->diagnostic + "'");
}

void TestParsesExistingQueryShape() {
  const auto result = Parse(R"json({
    "op": "ABS",
    "tensors": {
      "src0": {
        "dtype": "F32",
        "dimensions": [5, 7],
        "strides": [1, 5],
        "permutation": [0, 1]
      },
      "dst": {"dtype": "F32", "dimensions": [5, 7], "strides": [1, 5]}
    },
    "allowed_route_ids": ["abs_f32_contiguous_4d"]
  })json");

  const auto* parsed = ExpectParsed(result, "existing query shape");
  if (parsed == nullptr) {
    return;
  }
  Expect(parsed->op == "ABS", "existing query operation");
  Expect(parsed->query.tensors.size() == 2, "existing query tensor count");
  const auto& src0 = parsed->query.tensors.at("src0");
  Expect(src0.dtype == "F32", "existing query dtype");
  Expect(src0.dimensions == std::vector<std::int64_t>({5, 7}),
         "existing query dimensions");
  Expect(src0.permutation.has_value() &&
             *src0.permutation == std::vector<std::int64_t>({0, 1}),
         "existing query permutation");
  Expect(parsed->query.attributes.empty(), "missing attributes default empty");
  Expect(parsed->query.allowed_route_ids.has_value() &&
             *parsed->query.allowed_route_ids ==
                 std::vector<std::string>({"abs_f32_contiguous_4d"}),
         "existing query allowlist");
}

void TestParsesRecursiveAttributes() {
  const auto result = Parse(R"json({
    "op": "ABS",
    "tensors": {},
    "attributes": {
      "nothing": null,
      "enabled": true,
      "count": -7,
      "positive": 8,
      "ratio": 1.25,
      "name": "example",
      "items": [false, 3, {"nested": "value"}],
      "settings": {"mode": "fast"}
    }
  })json");

  const auto* parsed = ExpectParsed(result, "recursive attributes");
  if (parsed == nullptr) {
    return;
  }
  const auto& attributes = parsed->query.attributes;
  Expect(attributes.size() == 8, "recursive attribute count");
  Expect(std::holds_alternative<std::nullptr_t>(
             attributes.at("nothing").value),
         "null attribute");
  Expect(std::get<bool>(attributes.at("enabled").value),
         "boolean attribute");
  Expect(std::get<std::int64_t>(attributes.at("count").value) == -7,
         "negative integer attribute");
  Expect(std::get<std::int64_t>(attributes.at("positive").value) == 8,
         "unsigned JSON integer normalized to signed");
  Expect(std::get<double>(attributes.at("ratio").value) == 1.25,
         "floating-point attribute");
  Expect(std::get<std::string>(attributes.at("name").value) == "example",
         "string attribute");

  const auto& items =
      std::get<AttributeArray>(attributes.at("items").value);
  Expect(items.size() == 3, "attribute array size");
  Expect(!std::get<bool>(items[0].value), "nested array boolean");
  Expect(std::get<std::int64_t>(items[1].value) == 3,
         "nested array integer");
  const auto& nested = std::get<AttributeObject>(items[2].value);
  Expect(std::get<std::string>(nested.at("nested").value) == "value",
         "object nested in array");

  const auto& settings =
      std::get<AttributeObject>(attributes.at("settings").value);
  Expect(std::get<std::string>(settings.at("mode").value) == "fast",
         "nested attribute object");
}

void TestEmptyAttributesRemainEmpty() {
  const auto result =
      Parse(R"json({"op":"ABS","tensors":{},"attributes":{}})json");
  const auto* parsed = ExpectParsed(result, "empty attributes");
  if (parsed != nullptr) {
    Expect(parsed->query.attributes.empty(), "explicit empty attributes");
  }
}

void TestLargeLexicalFloatsRemainFloatingPoint() {
  const auto result = Parse(R"json({
    "op": "ABS",
    "tensors": {},
    "attributes": {
      "scientific": 1e20,
      "decimal": 18446744073709551616.0
    }
  })json");
  const auto* parsed = ExpectParsed(result, "large lexical floats");
  if (parsed == nullptr) {
    return;
  }
  Expect(std::holds_alternative<double>(
             parsed->query.attributes.at("scientific").value),
         "scientific syntax remains floating point");
  Expect(std::holds_alternative<double>(
             parsed->query.attributes.at("decimal").value),
         "decimal syntax remains floating point");
}

void TestStructuredErrors() {
  ExpectError(Parse(R"json({"op":"ABS")json"),
              parser::ParseErrorKind::malformed_json, "malformed JSON",
              "malformed JSON");
  ExpectError(
      Parse(R"json({"op":"ABS","tensors":{},"attributes":{"value":1e10000}})json"),
      parser::ParseErrorKind::number_out_of_range,
      "input contains a number outside the supported range",
      "JSON number range");
  ExpectError(Parse(R"json([])json"), parser::ParseErrorKind::schema,
              "input must be an object", "non-object root");
  ExpectError(
      Parse(R"json({"op":"ABS","tensors":{},"attributes":[]})json"),
      parser::ParseErrorKind::schema,
      "input field 'attributes' must be an object", "attribute object type");
  ExpectError(
      Parse(R"json({"op":"ABS","tensors":{},"attributes":{"value":9223372036854775808}})json"),
      parser::ParseErrorKind::schema,
      "input attribute 'value' is outside the signed 64-bit integer range",
      "attribute integer range");
  ExpectError(
      Parse(R"json({"op":"ABS","tensors":{},"attributes":{"values":[0,9223372036854775808]}})json"),
      parser::ParseErrorKind::schema,
      "input attribute 'values' element 1 is outside the signed 64-bit integer range",
      "nested attribute integer range");
  ExpectError(
      Parse(R"json({"op":"ABS","tensors":{},"attributes":{"value":-9223372036854775809}})json"),
      parser::ParseErrorKind::schema,
      "input contains an integer outside the signed 64-bit range",
      "attribute below signed integer range");
  ExpectError(
      Parse(R"json({"op":"ABS","tensors":{},"attributes":{"value":18446744073709551616}})json"),
      parser::ParseErrorKind::schema,
      "input contains an integer outside the signed 64-bit range",
      "attribute above unsigned parser range");
  ExpectError(
      Parse(R"json({"op":"ABS","tensors":{},"unexpected":true})json"),
      parser::ParseErrorKind::schema,
      "input contains unknown field 'unexpected'", "unknown root field");
}

void TestStreamParsingRemainsCompatible() {
  const std::string valid =
      R"json({"op":"ABS","tensors":{},"allowed_route_ids":[]})json";
  const auto text_result = Parse(valid);
  const auto stream_result = ParseStream(valid);
  const auto* text_query = ExpectParsed(text_result, "text overload");
  const auto* stream_query = ExpectParsed(stream_result, "stream overload");
  if (text_query != nullptr && stream_query != nullptr) {
    Expect(stream_query->op == text_query->op,
           "stream overload operation matches text overload");
    Expect(stream_query->query.allowed_route_ids ==
               text_query->query.allowed_route_ids,
           "stream overload allowlist matches text overload");
  }

  const auto malformed_stream = ParseStream(R"json({"op":"ABS")json");
  ExpectError(malformed_stream, parser::ParseErrorKind::malformed_json,
              "malformed JSON", "stream malformed JSON compatibility");

  const auto schema_stream = ParseStream(R"json([])json");
  ExpectError(schema_stream, parser::ParseErrorKind::schema,
              "input must be an object", "stream schema compatibility");
}

}  // namespace

int main() {
  TestParsesExistingQueryShape();
  TestParsesRecursiveAttributes();
  TestEmptyAttributesRemainEmpty();
  TestLargeLexicalFloatsRemainFloatingPoint();
  TestStructuredErrors();
  TestStreamParsingRemainsCompatible();

  if (g_failures != 0) {
    std::cerr << g_failures << " test failure(s)\n";
    return 1;
  }
  return 0;
}
