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
  std::optional<std::string> dtype;
  std::string dimensions_capture;
  std::string strides_capture;
  std::optional<std::string> permutation_capture;
};

enum class ValueKind {
  contiguous_strides,
  product,
  inverse_permutation,
  element,
  head,
  tail,
  chain_permutations,
  permuted_contiguous_strides,
};

struct ValueDefinition {
  std::string name;
  ValueKind kind;
  std::vector<std::string> sources;
  std::size_t parameter = 0;
};

enum class ConstraintKind {
  scalar_bounds,
  indexed_bounds,
  rank_range,
  exact_length,
  iota,
  equals,
  divides,
};

struct Constraint {
  ConstraintKind kind;
  std::string name;
  std::optional<std::int64_t> minimum;
  std::optional<std::int64_t> maximum;
  std::optional<std::int64_t> multiple_of;
  std::optional<std::size_t> index;
  std::optional<std::size_t> rank_minimum;
  std::optional<std::size_t> rank_maximum;
  std::optional<std::size_t> exact_length;
  std::vector<std::string> names;
};

struct RouteDescriptor {
  std::string id;
  std::vector<TensorDescriptor> tensors;
  std::vector<ValueDefinition> values;
  std::vector<Constraint> constraints;
};

using RouteTable = std::map<std::string, std::vector<RouteDescriptor>>;

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
    std::optional<std::int64_t> maximum,
    std::optional<std::int64_t> multiple_of) {
  Constraint result{};
  result.kind = ConstraintKind::scalar_bounds;
  result.name = std::move(name);
  result.minimum = minimum;
  result.maximum = maximum;
  result.multiple_of = multiple_of;
  return result;
}

Constraint indexed_bounds(
    std::string name,
    std::size_t index,
    std::optional<std::int64_t> minimum,
    std::optional<std::int64_t> maximum,
    std::optional<std::int64_t> multiple_of) {
  Constraint result{};
  result.kind = ConstraintKind::indexed_bounds;
  result.name = std::move(name);
  result.minimum = minimum;
  result.maximum = maximum;
  result.multiple_of = multiple_of;
  result.index = index;
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

Constraint iota(std::string name) {
  Constraint result{};
  result.kind = ConstraintKind::iota;
  result.name = std::move(name);
  return result;
}

Constraint equals(std::vector<std::string> names) {
  Constraint result{};
  result.kind = ConstraintKind::equals;
  result.names = std::move(names);
  return result;
}

Constraint divides(std::vector<std::string> names) {
  Constraint result{};
  result.kind = ConstraintKind::divides;
  result.names = std::move(names);
  return result;
}

const RouteTable& route_table() {
  static const RouteTable routes = {
#include "ggml_hrx_v2_routes.inc.cpp"
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
    if ((descriptor.dtype.has_value() &&
         normalize_token(tensor.dtype) != *descriptor.dtype) ||
        tensor.dimensions.size() != tensor.strides.size() ||
        (tensor.permutation.has_value() &&
         tensor.permutation->size() != tensor.dimensions.size())) {
      return Evaluation::no_match;
    }
    if (!store_capture(captures, descriptor.dimensions_capture, tensor.dimensions) ||
        !store_capture(captures, descriptor.strides_capture, tensor.strides)) {
      return Evaluation::no_match;
    }
    if (descriptor.permutation_capture.has_value()) {
      VectorValue permutation;
      if (tensor.permutation.has_value()) {
        permutation = *tensor.permutation;
      } else {
        permutation.reserve(tensor.dimensions.size());
        for (std::size_t index = 0; index < tensor.dimensions.size(); ++index) {
          if (index > static_cast<std::size_t>(
                          std::numeric_limits<std::int64_t>::max())) {
            return Evaluation::unsupported;
          }
          permutation.push_back(static_cast<std::int64_t>(index));
        }
      }
      if (!store_capture(
              captures, *descriptor.permutation_capture, std::move(permutation))) {
        return Evaluation::no_match;
      }
    }
  }
  return Evaluation::match;
}

const VectorValue* find_vector(const Captures& captures, const std::string& name) {
  const auto found = captures.find(name);
  if (found == captures.end()) {
    return nullptr;
  }
  return std::get_if<VectorValue>(&found->second);
}

bool is_permutation(const VectorValue& permutation) {
  std::vector<bool> seen(permutation.size(), false);
  for (const auto axis : permutation) {
    if (axis < 0) {
      return false;
    }
    const auto index = static_cast<std::uint64_t>(axis);
    if (index >= permutation.size() || seen[static_cast<std::size_t>(index)]) {
      return false;
    }
    seen[static_cast<std::size_t>(index)] = true;
  }
  return true;
}

Evaluation contiguous_strides(
    const VectorValue& dimensions,
    VectorValue& result) {
  result.clear();
  result.reserve(dimensions.size());
  std::int64_t stride = 1;
  for (std::size_t index = 0; index < dimensions.size(); ++index) {
    result.push_back(stride);
    if (index + 1 < dimensions.size() &&
        !checked_multiply(stride, dimensions[index], stride)) {
      return Evaluation::unsupported;
    }
  }
  return Evaluation::match;
}

Evaluation resolve_values(const RouteDescriptor& route, Captures& captures) {
  for (const auto& definition : route.values) {
    CapturedValue result;
    switch (definition.kind) {
      case ValueKind::contiguous_strides: {
        if (definition.sources.size() != 1) {
          return Evaluation::no_match;
        }
        const auto* source = find_vector(captures, definition.sources[0]);
        if (source == nullptr) {
          return Evaluation::no_match;
        }
        VectorValue strides;
        const auto evaluation = contiguous_strides(*source, strides);
        if (evaluation != Evaluation::match) {
          return evaluation;
        }
        result = std::move(strides);
        break;
      }
      case ValueKind::product: {
        if (definition.sources.size() != 1) {
          return Evaluation::no_match;
        }
        const auto* source = find_vector(captures, definition.sources[0]);
        if (source == nullptr) {
          return Evaluation::no_match;
        }
        std::int64_t product = 1;
        for (const auto extent : *source) {
          if (!checked_multiply(product, extent, product)) {
            return Evaluation::unsupported;
          }
        }
        result = product;
        break;
      }
      case ValueKind::inverse_permutation: {
        if (definition.sources.size() != 1) {
          return Evaluation::no_match;
        }
        const auto* source = find_vector(captures, definition.sources[0]);
        if (source == nullptr || !is_permutation(*source)) {
          return Evaluation::no_match;
        }
        VectorValue inverse(source->size(), 0);
        for (std::size_t index = 0; index < source->size(); ++index) {
          if (index > static_cast<std::size_t>(
                          std::numeric_limits<std::int64_t>::max())) {
            return Evaluation::unsupported;
          }
          inverse[static_cast<std::size_t>((*source)[index])] =
              static_cast<std::int64_t>(index);
        }
        result = std::move(inverse);
        break;
      }
      case ValueKind::element: {
        if (definition.sources.size() != 1) {
          return Evaluation::no_match;
        }
        const auto* source = find_vector(captures, definition.sources[0]);
        if (source == nullptr || definition.parameter >= source->size()) {
          return Evaluation::no_match;
        }
        result = (*source)[definition.parameter];
        break;
      }
      case ValueKind::head: {
        if (definition.sources.size() != 1) {
          return Evaluation::no_match;
        }
        const auto* source = find_vector(captures, definition.sources[0]);
        if (source == nullptr) {
          return Evaluation::no_match;
        }
        const auto length = std::min(definition.parameter, source->size());
        result = VectorValue(source->begin(), source->begin() + length);
        break;
      }
      case ValueKind::tail: {
        if (definition.sources.size() != 1) {
          return Evaluation::no_match;
        }
        const auto* source = find_vector(captures, definition.sources[0]);
        if (source == nullptr) {
          return Evaluation::no_match;
        }
        const auto drop = std::min(definition.parameter, source->size());
        result = VectorValue(source->begin() + drop, source->end());
        break;
      }
      case ValueKind::chain_permutations: {
        if (definition.sources.size() != 2) {
          return Evaluation::no_match;
        }
        const auto* first = find_vector(captures, definition.sources[0]);
        const auto* second = find_vector(captures, definition.sources[1]);
        if (first == nullptr || second == nullptr ||
            first->size() != second->size() || !is_permutation(*first) ||
            !is_permutation(*second)) {
          return Evaluation::no_match;
        }
        VectorValue chained;
        chained.reserve(first->size());
        for (const auto axis : *first) {
          chained.push_back((*second)[static_cast<std::size_t>(axis)]);
        }
        result = std::move(chained);
        break;
      }
      case ValueKind::permuted_contiguous_strides: {
        if (definition.sources.size() != 2) {
          return Evaluation::no_match;
        }
        const auto* dimensions = find_vector(captures, definition.sources[0]);
        const auto* permutation = find_vector(captures, definition.sources[1]);
        if (dimensions == nullptr || permutation == nullptr ||
            dimensions->size() != permutation->size() ||
            !is_permutation(*permutation)) {
          return Evaluation::no_match;
        }
        VectorValue base_dimensions;
        base_dimensions.reserve(dimensions->size());
        for (const auto axis : *permutation) {
          base_dimensions.push_back(
              (*dimensions)[static_cast<std::size_t>(axis)]);
        }
        VectorValue base_strides;
        const auto evaluation =
            contiguous_strides(base_dimensions, base_strides);
        if (evaluation != Evaluation::match) {
          return evaluation;
        }
        VectorValue logical_strides(permutation->size(), 0);
        for (std::size_t base_axis = 0; base_axis < permutation->size();
             ++base_axis) {
          logical_strides[static_cast<std::size_t>((*permutation)[base_axis])] =
              base_strides[base_axis];
        }
        result = std::move(logical_strides);
        break;
      }
    }

    captures.insert_or_assign(definition.name, std::move(result));
  }
  return Evaluation::match;
}

bool scalar_accepts(const Constraint& constraint, std::int64_t value) {
  if ((constraint.minimum.has_value() && value < *constraint.minimum) ||
      (constraint.maximum.has_value() && value > *constraint.maximum)) {
    return false;
  }
  if (!constraint.multiple_of.has_value()) {
    return true;
  }
  const auto divisor = *constraint.multiple_of;
  if (divisor == 0) {
    return false;
  }
  if (divisor == -1) {
    return true;
  }
  return value % divisor == 0;
}

bool constraint_accepts(const Constraint& constraint, const Captures& captures) {
  if (constraint.kind == ConstraintKind::equals) {
    if (constraint.names.empty()) {
      return false;
    }
    const auto first = captures.find(constraint.names.front());
    if (first == captures.end()) {
      return false;
    }
    return std::all_of(
        constraint.names.begin() + 1,
        constraint.names.end(),
        [&](const auto& name) {
          const auto current = captures.find(name);
          return current != captures.end() && current->second == first->second;
        });
  }
  if (constraint.kind == ConstraintKind::divides) {
    if (constraint.names.empty()) {
      return false;
    }
    const auto* divisors = find_vector(captures, constraint.names.front());
    if (divisors == nullptr) {
      return false;
    }
    for (auto name = constraint.names.begin() + 1;
         name != constraint.names.end(); ++name) {
      const auto* values = find_vector(captures, *name);
      if (values == nullptr || values->size() != divisors->size()) {
        return false;
      }
      for (std::size_t index = 0; index < divisors->size(); ++index) {
        const auto divisor = (*divisors)[index];
        if (divisor <= 0 || (*values)[index] % divisor != 0) {
          return false;
        }
      }
    }
    return true;
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
    return scalar_accepts(constraint, *value);
  }

  const auto* value = std::get_if<VectorValue>(&found->second);
  if (value == nullptr) {
    return false;
  }
  if (constraint.kind == ConstraintKind::indexed_bounds) {
    if (!constraint.index.has_value() || *constraint.index >= value->size()) {
      return false;
    }
    return scalar_accepts(constraint, (*value)[*constraint.index]);
  }
  if (constraint.kind == ConstraintKind::exact_length) {
    return constraint.exact_length.has_value() &&
           value->size() == *constraint.exact_length;
  }
  if (constraint.kind == ConstraintKind::rank_range) {
    return (!constraint.rank_minimum.has_value() || value->size() >= *constraint.rank_minimum) &&
           (!constraint.rank_maximum.has_value() || value->size() <= *constraint.rank_maximum);
  }
  if (constraint.kind == ConstraintKind::iota) {
    for (std::size_t index = 0; index < value->size(); ++index) {
      if (index > static_cast<std::size_t>(
                      std::numeric_limits<std::int64_t>::max()) ||
          (*value)[index] != static_cast<std::int64_t>(index)) {
        return false;
      }
    }
    return true;
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
  const auto normalized_op = normalize_token(op);
  const auto& table = route_table();
  const auto operation = table.find(normalized_op);
  if (operation == table.end()) {
    return {SelectionStatus::unsupported, {}};
  }

  const auto& routes = operation->second;
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
  const auto normalized_op = normalize_token(op);
  const auto& table = route_table();
  const auto operation = table.find(normalized_op);
  if (operation == table.end()) {
    return {};
  }

  const auto& routes = operation->second;
  std::vector<std::string_view> ids;
  ids.reserve(routes.size());
  for (const auto& route : routes) {
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
