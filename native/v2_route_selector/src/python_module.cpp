#include "ggml_hrx/v2_route_selector/selector.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <cstdint>
#include <string>
#include <vector>

namespace nb = nanobind;
namespace selector = ggml_hrx::routing::v2;

namespace {

enum class ParseResult {
  ok,
  unrepresentable,
};

bool is_sequence(nb::handle value) {
  return nb::isinstance<nb::list>(value) || nb::isinstance<nb::tuple>(value);
}

std::size_t sequence_size(nb::handle value) {
  return static_cast<std::size_t>(nb::len(value));
}

nb::handle sequence_item(nb::handle value, std::size_t index) {
  return value[nb::int_(index)];
}

ParseResult parse_int64(nb::handle value, std::int64_t& output) {
  if (!nb::isinstance<nb::int_>(value)) {
    throw nb::type_error("tensor dimensions and strides must contain integers");
  }
  int overflow = 0;
  const auto parsed = PyLong_AsLongLongAndOverflow(value.ptr(), &overflow);
  if (overflow != 0) {
    PyErr_Clear();
    return ParseResult::unrepresentable;
  }
  if (PyErr_Occurred() != nullptr) {
    throw nb::python_error();
  }
  output = static_cast<std::int64_t>(parsed);
  return ParseResult::ok;
}

ParseResult parse_int64_sequence(nb::handle value, std::vector<std::int64_t>& output) {
  if (!is_sequence(value)) {
    throw nb::type_error("tensor dimensions and strides must be lists or tuples");
  }
  output.reserve(sequence_size(value));
  for (std::size_t index = 0; index < sequence_size(value); ++index) {
    std::int64_t parsed = 0;
    if (parse_int64(sequence_item(value, index), parsed) == ParseResult::unrepresentable) {
      return ParseResult::unrepresentable;
    }
    output.push_back(parsed);
  }
  return ParseResult::ok;
}

nb::handle required_item(nb::dict mapping, const char* key) {
  nb::str name(key);
  if (!mapping.contains(name)) {
    throw nb::key_error(key);
  }
  return mapping[name];
}

ParseResult parse_tensor(nb::handle payload, selector::Tensor& tensor) {
  if (!nb::isinstance<nb::dict>(payload)) {
    throw nb::type_error("each tensor must be a dict");
  }
  nb::dict mapping = nb::borrow<nb::dict>(payload);
  tensor.dtype = nb::cast<std::string>(required_item(mapping, "dtype"));
  if (parse_int64_sequence(required_item(mapping, "dimensions"), tensor.dimensions) ==
      ParseResult::unrepresentable) {
    return ParseResult::unrepresentable;
  }
  return parse_int64_sequence(required_item(mapping, "strides"), tensor.strides);
}

ParseResult parse_query(nb::handle payload, selector::Query& query) {
  if (!nb::isinstance<nb::dict>(payload)) {
    throw nb::type_error("route query must be a dict");
  }
  nb::dict mapping = nb::borrow<nb::dict>(payload);
  nb::handle tensor_payload = required_item(mapping, "tensors");
  if (!nb::isinstance<nb::dict>(tensor_payload)) {
    throw nb::type_error("query 'tensors' must be a dict keyed by tensor role");
  }
  nb::dict tensors = nb::borrow<nb::dict>(tensor_payload);
  for (auto [name, value] : tensors) {
    selector::Tensor tensor;
    if (parse_tensor(value, tensor) == ParseResult::unrepresentable) {
      return ParseResult::unrepresentable;
    }
    query.tensors.emplace(nb::cast<std::string>(name), std::move(tensor));
  }

  nb::str allowlist_name("allowed_route_ids");
  if (!mapping.contains(allowlist_name) || mapping[allowlist_name].is_none()) {
    return ParseResult::ok;
  }
  nb::handle allowlist = mapping[allowlist_name];
  if (!is_sequence(allowlist)) {
    throw nb::type_error("query 'allowed_route_ids' must be None, a list, or a tuple");
  }
  query.allowed_route_ids.emplace();
  query.allowed_route_ids->reserve(sequence_size(allowlist));
  for (std::size_t index = 0; index < sequence_size(allowlist); ++index) {
    query.allowed_route_ids->push_back(nb::cast<std::string>(sequence_item(allowlist, index)));
  }
  return ParseResult::ok;
}

nb::tuple selection_tuple(const selector::Selection& selection) {
  const auto status = selector::status_name(selection.status);
  if (selection.status == selector::SelectionStatus::match) {
    return nb::make_tuple(
        nb::str(status.data(), status.size()),
        nb::str(selection.route_id.data(), selection.route_id.size()));
  }
  return nb::make_tuple(
      nb::str(status.data(), status.size()),
      nb::none());
}

nb::tuple select_from_python(const std::string& op, nb::handle query_payload) {
  selector::Query query;
  if (parse_query(query_payload, query) == ParseResult::unrepresentable) {
    return selection_tuple({selector::SelectionStatus::unsupported, {}});
  }
  return selection_tuple(selector::select(op, query));
}

nb::list supported_route_ids_from_python(const std::string& op) {
  nb::list result;
  for (const auto route_id : selector::supported_route_ids(op)) {
    result.append(nb::str(route_id.data(), route_id.size()));
  }
  return result;
}

}  // namespace

NB_MODULE(_ggml_hrx_v2_selector_native, module) {
  module.doc() = "Experimental native route selector for the v2 catalog";
  module.def(
      "select",
      &select_from_python,
      nb::arg("op"),
      nb::arg("query"),
      "Select a route from one normalized query.");
  module.def(
      "supported_route_ids",
      &supported_route_ids_from_python,
      nb::arg("op"),
      "Return native route IDs for an operation in selection order.");
}
