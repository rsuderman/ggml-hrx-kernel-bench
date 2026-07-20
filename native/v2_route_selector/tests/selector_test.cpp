#include "ggml_hrx/v2_route_selector/selector.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace selector = ggml_hrx::routing::v2;

constexpr const char* kContiguousRoute = "abs_f32_contiguous_4d";
constexpr const char* kNonContiguousRoute = "abs_f32_non_contiguous_4d";

int g_failures = 0;

void Expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++g_failures;
  }
}

selector::Query ContiguousAbsQuery() {
  selector::Query query;
  const selector::Tensor tensor{
      "F32", {5, 7, 11, 13}, {1, 5, 35, 385}, std::nullopt};
  query.tensors.emplace("src0", tensor);
  query.tensors.emplace("dst", tensor);
  return query;
}

void TestMissingAndEmptyAttributesPreserveSelection() {
  const selector::Query query = ContiguousAbsQuery();
  Expect(query.attributes.empty(), "attributes default to empty");
  const auto selection = selector::select("ABS", query);
  Expect(selection.status == selector::SelectionStatus::match,
         "default attributes select a route");
  Expect(selection.route_id == kContiguousRoute,
         "default attributes preserve selected route");

  selector::Query explicitly_empty = query;
  explicitly_empty.attributes = {};
  const auto empty_selection = selector::select("ABS", explicitly_empty);
  Expect(empty_selection.status == selector::SelectionStatus::match,
         "explicit empty attributes select a route");
  Expect(empty_selection.route_id == kContiguousRoute,
         "empty attributes preserve selected route");
}

void TestNonemptyAttributesAreUnsupported() {
  selector::Query query = ContiguousAbsQuery();
  selector::AttributeValue value;
  value.value = std::int64_t{1};
  query.attributes.emplace("axis", std::move(value));

  const auto selection = selector::select("ABS", query);
  Expect(selection.status == selector::SelectionStatus::unsupported,
         "nonempty attributes are unsupported");
  Expect(selection.route_id.empty(),
         "unsupported attribute selection has no route ID");
}

void TestCatalogOrderRemainsFirstMatch() {
  selector::Query query = ContiguousAbsQuery();

  // This rank-four query satisfies both the contiguous route and the later
  // non-contiguous fallback. The unrestricted selector must retain catalog
  // ordering, while an explicit CLI allowlist can still reach the fallback.
  const auto preferred = selector::select("ABS", query);
  Expect(preferred.status == selector::SelectionStatus::match,
         "overlapping routes produce a match");
  Expect(preferred.route_id == kContiguousRoute,
         "the first matching catalog route wins");

  query.allowed_route_ids = std::vector<std::string>{kNonContiguousRoute};
  const auto filtered = selector::select("ABS", query);
  Expect(filtered.status == selector::SelectionStatus::match,
         "allowlist can select a later matching route");
  Expect(filtered.route_id == kNonContiguousRoute,
         "allowlist behavior remains available to the CLI");
}

}  // namespace

int main() {
  TestMissingAndEmptyAttributesPreserveSelection();
  TestNonemptyAttributesAreUnsupported();
  TestCatalogOrderRemainsFirstMatch();

  if (g_failures != 0) {
    std::cerr << g_failures << " test failure(s)\n";
    return 1;
  }
  return 0;
}
