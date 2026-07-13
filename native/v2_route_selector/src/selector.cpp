#include "ggml_hrx/v2_route_selector/selector.h"

#include <algorithm>
#include <cctype>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <variant>

namespace ggml_hrx::routing::v2 {
namespace {

using VectorValue = std::vector<std::int64_t>;
using CapturedValue = std::variant<std::int64_t, VectorValue>;
using Captures = std::map<std::string, CapturedValue, std::less<>>;

struct TensorDescriptor {
  std::string role;
  std::string dtype;
  std::string dimensions_capture;
  std::string strides_capture;
};

enum class ValueKind {
  contiguous_strides,
  head,
  tail,
  product,
};

struct ValueDefinition {
  std::string name;
  ValueKind kind;
  std::string source;
  std::size_t parameter = 0;
};

enum class ConstraintKind {
  scalar_bounds,
  rank_range,
  exact_length,
  equals,
};

struct Constraint {
  ConstraintKind kind;
  std::string name;
  std::optional<std::int64_t> minimum;
  std::optional<std::int64_t> maximum;
  std::optional<std::size_t> rank_minimum;
  std::optional<std::size_t> rank_maximum;
  std::optional<std::size_t> exact_length;
  std::vector<std::string> equals;
};

struct RouteDescriptor {
  std::string id;
  std::vector<TensorDescriptor> tensors;
  std::vector<ValueDefinition> values;
  std::vector<Constraint> constraints;
};

enum class Evaluation {
  match,
  no_match,
  unsupported,
};

std::string normalize_token(std::string_view value) {
  const auto first = std::find_if_not(value.begin(), value.end(), [](unsigned char ch) {
    return std::isspace(ch) != 0;
  });
  const auto last = std::find_if_not(value.rbegin(), value.rend(), [](unsigned char ch) {
    return std::isspace(ch) != 0;
  }).base();
  if (first >= last) {
    return {};
  }

  std::string normalized(first, last);
  std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char ch) {
    return static_cast<char>(std::toupper(ch));
  });
  return normalized;
}

bool checked_multiply(std::int64_t lhs, std::int64_t rhs, std::int64_t& result) {
  constexpr auto minimum = std::numeric_limits<std::int64_t>::min();
  constexpr auto maximum = std::numeric_limits<std::int64_t>::max();

  if (lhs > 0) {
    if ((rhs > 0 && lhs > maximum / rhs) ||
        (rhs < 0 && rhs < minimum / lhs)) {
      return false;
    }
  } else if (lhs < 0) {
    if ((rhs > 0 && lhs < minimum / rhs) ||
        (rhs < 0 && lhs < maximum / rhs)) {
      return false;
    }
  }

  result = lhs * rhs;
  return true;
}

bool store_capture(Captures& captures, const std::string& name, CapturedValue value) {
  const auto found = captures.find(name);
  if (found != captures.end()) {
    return found->second == value;
  }
  captures.emplace(name, std::move(value));
  return true;
}

Constraint scalar_bounds(
    std::string name,
    std::optional<std::int64_t> minimum,
    std::optional<std::int64_t> maximum) {
  Constraint result{};
  result.kind = ConstraintKind::scalar_bounds;
  result.name = std::move(name);
  result.minimum = minimum;
  result.maximum = maximum;
  return result;
}

Constraint rank_range(
    std::string name,
    std::optional<std::size_t> minimum,
    std::optional<std::size_t> maximum) {
  Constraint result{};
  result.kind = ConstraintKind::rank_range;
  result.name = std::move(name);
  result.rank_minimum = minimum;
  result.rank_maximum = maximum;
  return result;
}

Constraint exact_length(std::string name, std::size_t length) {
  Constraint result{};
  result.kind = ConstraintKind::exact_length;
  result.name = std::move(name);
  result.exact_length = length;
  return result;
}

Constraint equals(std::vector<std::string> names) {
  Constraint result{};
  result.kind = ConstraintKind::equals;
  result.equals = std::move(names);
  return result;
}

const std::vector<RouteDescriptor>& abs_routes() {
  static const std::vector<RouteDescriptor> routes = {
      {
          "abs_f16_contiguous_4d",
          {
              {"src0", "F16", "src0_dimensions", "src0_strides"},
              {"dst", "F16", "dst_dimensions", "dst_strides"},
          },
          {
              {"contiguous_strides", ValueKind::contiguous_strides, "dst_dimensions"},
              {"leading_dimensions", ValueKind::head, "dst_dimensions", 1},
              {"trailing_dimensions", ValueKind::tail, "dst_dimensions", 1},
              {"flattened_trailing_dimensions", ValueKind::product, "trailing_dimensions"},
              {"total_size", ValueKind::product, "dst_dimensions"},
          },
          {
              scalar_bounds("total_size", 1, 1073741824),
              rank_range("dst_dimensions", 2, 4),
              equals({"src0_dimensions", "dst_dimensions"}),
              equals({"contiguous_strides", "src0_strides", "dst_strides"}),
          },
      },
      {
          "abs_f32_contiguous_4d",
          {
              {"src0", "F32", "src0_dimensions", "src0_strides"},
              {"dst", "F32", "dst_dimensions", "dst_strides"},
          },
          {
              {"contiguous_strides", ValueKind::contiguous_strides, "dst_dimensions"},
              {"leading_dimensions", ValueKind::head, "dst_dimensions", 1},
              {"trailing_dimensions", ValueKind::tail, "dst_dimensions", 1},
              {"flattened_trailing_dimensions", ValueKind::product, "trailing_dimensions"},
              {"total_size", ValueKind::product, "dst_dimensions"},
          },
          {
              scalar_bounds("total_size", 1, 1073741824),
              rank_range("dst_dimensions", 2, 4),
              equals({"src0_dimensions", "dst_dimensions"}),
              equals({"contiguous_strides", "src0_strides", "dst_strides"}),
          },
      },
      {
          "abs_f16_non_contiguous_4d",
          {
              {"src0", "F16", "src0_dimensions", "src0_strides"},
              {"dst", "F16", "dst_dimensions", "dst_strides"},
          },
          {
              {"contiguous_strides", ValueKind::contiguous_strides, "dst_dimensions"},
              {"total_size", ValueKind::product, "dst_dimensions"},
          },
          {
              exact_length("dst_dimensions", 4),
              equals({"src0_dimensions", "dst_dimensions"}),
              equals({"contiguous_strides", "dst_strides"}),
          },
      },
      {
          "abs_f32_non_contiguous_4d",
          {
              {"src0", "F32", "src0_dimensions", "src0_strides"},
              {"dst", "F32", "dst_dimensions", "dst_strides"},
          },
          {
              {"contiguous_strides", ValueKind::contiguous_strides, "dst_dimensions"},
              {"total_size", ValueKind::product, "dst_dimensions"},
          },
          {
              exact_length("dst_dimensions", 4),
              equals({"src0_dimensions", "dst_dimensions"}),
              equals({"contiguous_strides", "dst_strides"}),
          },
      },
  };
  return routes;
}

bool route_is_allowed(const Query& query, std::string_view route_id) {
  if (!query.allowed_route_ids.has_value()) {
    return true;
  }
  return std::find(
             query.allowed_route_ids->begin(),
             query.allowed_route_ids->end(),
             route_id) != query.allowed_route_ids->end();
}

bool all_allowed_routes_are_supported(const Query& query) {
  const auto& routes = abs_routes();
  if (!query.allowed_route_ids.has_value()) {
    return true;
  }
  for (const auto& allowed : *query.allowed_route_ids) {
    const auto found = std::find_if(routes.begin(), routes.end(), [&](const auto& route) {
      return route.id == allowed;
    });
    if (found == routes.end()) {
      return false;
    }
  }
  return true;
}

Evaluation capture_tensors(
    const RouteDescriptor& route,
    const Query& query,
    Captures& captures) {
  if (query.tensors.size() != route.tensors.size()) {
    return Evaluation::no_match;
  }

  for (const auto& descriptor : route.tensors) {
    const auto tensor_it = query.tensors.find(descriptor.role);
    if (tensor_it == query.tensors.end()) {
      return Evaluation::no_match;
    }
    const auto& tensor = tensor_it->second;
    if (normalize_token(tensor.dtype) != descriptor.dtype ||
        tensor.dimensions.size() != tensor.strides.size()) {
      return Evaluation::no_match;
    }
    if (!store_capture(captures, descriptor.dimensions_capture, tensor.dimensions) ||
        !store_capture(captures, descriptor.strides_capture, tensor.strides)) {
      return Evaluation::no_match;
    }
  }
  return Evaluation::match;
}

Evaluation resolve_values(const RouteDescriptor& route, Captures& captures) {
  for (const auto& definition : route.values) {
    const auto source_it = captures.find(definition.source);
    if (source_it == captures.end()) {
      return Evaluation::unsupported;
    }
    const auto* source = std::get_if<VectorValue>(&source_it->second);
    if (source == nullptr) {
      return Evaluation::unsupported;
    }

    CapturedValue result;
    switch (definition.kind) {
      case ValueKind::contiguous_strides: {
        VectorValue strides;
        strides.reserve(source->size());
        std::int64_t stride = 1;
        for (const auto extent : *source) {
          strides.push_back(stride);
          if (!checked_multiply(stride, extent, stride)) {
            return Evaluation::unsupported;
          }
        }
        result = std::move(strides);
        break;
      }
      case ValueKind::head: {
        const auto length = std::min(definition.parameter, source->size());
        result = VectorValue(source->begin(), source->begin() + length);
        break;
      }
      case ValueKind::tail: {
        const auto drop = std::min(definition.parameter, source->size());
        result = VectorValue(source->begin() + drop, source->end());
        break;
      }
      case ValueKind::product: {
        std::int64_t product = 1;
        for (const auto extent : *source) {
          if (!checked_multiply(product, extent, product)) {
            return Evaluation::unsupported;
          }
        }
        result = product;
        break;
      }
    }

    if (!store_capture(captures, definition.name, std::move(result))) {
      return Evaluation::no_match;
    }
  }
  return Evaluation::match;
}

bool constraint_accepts(const Constraint& constraint, const Captures& captures) {
  if (constraint.kind == ConstraintKind::equals) {
    if (constraint.equals.empty()) {
      return false;
    }
    const auto first = captures.find(constraint.equals.front());
    if (first == captures.end()) {
      return false;
    }
    return std::all_of(
        constraint.equals.begin() + 1,
        constraint.equals.end(),
        [&](const auto& name) {
          const auto current = captures.find(name);
          return current != captures.end() && current->second == first->second;
        });
  }

  const auto found = captures.find(constraint.name);
  if (found == captures.end()) {
    return false;
  }
  if (constraint.kind == ConstraintKind::scalar_bounds) {
    const auto* value = std::get_if<std::int64_t>(&found->second);
    if (value == nullptr) {
      return false;
    }
    return (!constraint.minimum.has_value() || *value >= *constraint.minimum) &&
           (!constraint.maximum.has_value() || *value <= *constraint.maximum);
  }

  const auto* value = std::get_if<VectorValue>(&found->second);
  if (value == nullptr) {
    return false;
  }
  if (constraint.kind == ConstraintKind::exact_length) {
    return value->size() == constraint.exact_length;
  }
  if (constraint.kind == ConstraintKind::rank_range) {
    return (!constraint.rank_minimum.has_value() || value->size() >= *constraint.rank_minimum) &&
           (!constraint.rank_maximum.has_value() || value->size() <= *constraint.rank_maximum);
  }
  return false;
}

Evaluation evaluate(const RouteDescriptor& route, const Query& query) {
  Captures captures;
  const auto capture_result = capture_tensors(route, query, captures);
  if (capture_result != Evaluation::match) {
    return capture_result;
  }
  const auto resolve_result = resolve_values(route, captures);
  if (resolve_result != Evaluation::match) {
    return resolve_result;
  }
  const bool accepted = std::all_of(
      route.constraints.begin(), route.constraints.end(), [&](const auto& constraint) {
        return constraint_accepts(constraint, captures);
      });
  return accepted ? Evaluation::match : Evaluation::no_match;
}

}  // namespace

Selection select(std::string_view op, const Query& query) {
  if (normalize_token(op) != "ABS") {
    return {SelectionStatus::unsupported, {}};
  }
  if (!all_allowed_routes_are_supported(query)) {
    return {SelectionStatus::unsupported, {}};
  }

  const auto& routes = abs_routes();
  for (const auto& route : routes) {
    if (!route_is_allowed(query, route.id)) {
      continue;
    }
    const auto result = evaluate(route, query);
    if (result == Evaluation::unsupported) {
      return {SelectionStatus::unsupported, {}};
    }
    if (result == Evaluation::match) {
      return {SelectionStatus::match, route.id};
    }
  }
  return {SelectionStatus::no_match, {}};
}

std::vector<std::string_view> supported_route_ids(std::string_view op) {
  if (normalize_token(op) != "ABS") {
    return {};
  }
  std::vector<std::string_view> ids;
  ids.reserve(abs_routes().size());
  for (const auto& route : abs_routes()) {
    ids.emplace_back(route.id);
  }
  return ids;
}

std::string_view status_name(SelectionStatus status) {
  switch (status) {
    case SelectionStatus::match:
      return "MATCH";
    case SelectionStatus::no_match:
      return "NO_MATCH";
    case SelectionStatus::unsupported:
      return "UNSUPPORTED";
  }
  return "UNSUPPORTED";
}

}  // namespace ggml_hrx::routing::v2
