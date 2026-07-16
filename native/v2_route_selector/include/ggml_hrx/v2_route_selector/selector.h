#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ggml_hrx::routing::v2 {

enum class SelectionStatus {
  match,
  no_match,
  unsupported,
};

struct Tensor {
  std::string dtype;
  std::vector<std::int64_t> dimensions;
  std::vector<std::int64_t> strides;
  std::optional<std::vector<std::int64_t>> permutation;
};

struct Query {
  std::map<std::string, Tensor, std::less<>> tensors;
  std::optional<std::vector<std::string>> allowed_route_ids;
};

struct Selection {
  SelectionStatus status = SelectionStatus::no_match;
  std::string route_id;
};

// Selects the first matching route in descriptor order.
Selection select(std::string_view op, const Query& query);

std::vector<std::string_view> supported_route_ids(std::string_view op);

std::string_view status_name(SelectionStatus status);

}  // namespace ggml_hrx::routing::v2
