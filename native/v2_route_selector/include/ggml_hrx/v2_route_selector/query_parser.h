#pragma once

#include "ggml_hrx/v2_route_selector/selector.h"

#include <iosfwd>
#include <string>
#include <string_view>
#include <variant>

namespace ggml_hrx::routing::v2::query_parser {

enum class ParseErrorKind {
  malformed_json,
  number_out_of_range,
  schema,
};

struct ParseError {
  ParseErrorKind kind = ParseErrorKind::schema;
  std::string diagnostic;
};

struct ParsedQuery {
  std::string op;
  Query query;
};

using ParseResult = std::variant<ParsedQuery, ParseError>;

// Parses one selector query from JSON.
//
// NOTE: Returned ParseError::diagnostic values don't include an "error:"
// prefix, allowing callers to format them consistently with their interface.
ParseResult parse(std::string_view input);

// Reads the complete stream and delegates to the text parser.
ParseResult parse(std::istream& input);

}  // namespace ggml_hrx::routing::v2::query_parser
