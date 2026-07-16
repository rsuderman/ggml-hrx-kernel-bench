#include "tools/v2-route-selector/v2_route_selector_cli.h"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

using ggml_hrx::v2_route_selector_cli::Run;

constexpr const char *kUsage =
    "Usage: ggml-hrx-v2-route-selector --input <file|-> "
    "[--expect-route <route-id>]\n";
constexpr const char *kF32Route = "abs_f32_contiguous_4d";
constexpr const char *kPermutedCopyRoute =
    "copy_f32_f32_non_contiguous_4d";

int g_failures = 0;

void Expect(bool condition, const std::string &message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    ++g_failures;
  }
}

struct RunResult {
  int exit_code;
  std::string stdout_text;
  std::string stderr_text;
};

RunResult Invoke(const std::vector<std::string> &args,
                 const std::string &stdin_text = {}) {
  std::istringstream input(stdin_text);
  std::ostringstream output;
  std::ostringstream error;
  const int exit_code = Run(args, input, output, error);
  return {exit_code, output.str(), error.str()};
}

std::string RenderForFailure(const std::string &value) {
  std::string rendered;
  rendered.reserve(value.size() + 2);
  rendered.push_back('"');
  for (const char ch : value) {
    switch (ch) {
      case '\n':
        rendered += "\\n";
        break;
      case '\r':
        rendered += "\\r";
        break;
      case '\t':
        rendered += "\\t";
        break;
      case '"':
        rendered += "\\\"";
        break;
      case '\\':
        rendered += "\\\\";
        break;
      default:
        rendered.push_back(ch);
        break;
    }
  }
  rendered.push_back('"');
  return rendered;
}

void ExpectResult(const RunResult &actual, int expected_exit_code,
                  const std::string &expected_stdout,
                  const std::string &expected_stderr,
                  const std::string &case_name) {
  Expect(actual.exit_code == expected_exit_code,
         case_name + ": exit code was " + std::to_string(actual.exit_code) +
             ", expected " + std::to_string(expected_exit_code));
  Expect(actual.stdout_text == expected_stdout,
         case_name + ": stdout was " + RenderForFailure(actual.stdout_text) +
             ", expected " + RenderForFailure(expected_stdout));
  Expect(actual.stderr_text == expected_stderr,
         case_name + ": stderr was " + RenderForFailure(actual.stderr_text) +
             ", expected " + RenderForFailure(expected_stderr));
}

const std::string &ValidQuery() {
  static const std::string query = R"json({
  "op": "ABS",
  "tensors": {
    "src0": {
      "dtype": "F32",
      "dimensions": [5, 7, 11, 13],
      "strides": [1, 5, 35, 385]
    },
    "dst": {
      "dtype": "F32",
      "dimensions": [5, 7, 11, 13],
      "strides": [1, 5, 35, 385]
    }
  }
})json";
  return query;
}

std::string PermutedCopyQuery(const std::string &src0_permutation) {
  return R"json({
  "op": "CPY",
  "tensors": {
    "src0": {
      "dtype": "F32",
      "dimensions": [1, 2, 3, 4],
      "strides": [1, 12, 4, 1],
      "permutation": )json" +
         src0_permutation + R"json(
    },
    "dst": {
      "dtype": "F32",
      "dimensions": [1, 2, 3, 4],
      "strides": [1, 1, 2, 6],
      "permutation": [0, 2, 1, 3]
    }
  },
  "allowed_route_ids": ["copy_f32_f32_non_contiguous_4d"]
})json";
}

void TestSelectsContiguousAbs() {
  const auto result = Invoke({"--input", "-"}, ValidQuery());
  ExpectResult(result, 0, std::string(kF32Route) + "\n", "",
               "contiguous ABS selection");
}

void TestNormalizesOperationAndDTypes() {
  const std::string query = R"json({
  "op": "  aBs  ",
  "tensors": {
    "src0": {
      "dtype": "\tf32\n",
      "dimensions": [5, 7, 11, 13],
      "strides": [1, 5, 35, 385]
    },
    "dst": {
      "dtype": " F32 ",
      "dimensions": [5, 7, 11, 13],
      "strides": [1, 5, 35, 385]
    }
  }
})json";

  const auto result = Invoke({"--input", "-"}, query);
  ExpectResult(result, 0, std::string(kF32Route) + "\n", "",
               "operation and dtype normalization");
}

void TestNullAndExplicitIdentityPermutations() {
  const std::string query = R"json({
  "op": "ABS",
  "tensors": {
    "src0": {
      "dtype": "F32",
      "dimensions": [5, 7],
      "strides": [1, 5],
      "permutation": null
    },
    "dst": {
      "dtype": "F32",
      "dimensions": [5, 7],
      "strides": [1, 5],
      "permutation": [0, 1]
    }
  }
})json";

  const auto result = Invoke({"--input", "-"}, query);
  ExpectResult(result, 0, std::string(kF32Route) + "\n", "",
               "null and explicit identity permutations");
}

void TestSelectsPermutedCopy() {
  const auto result =
      Invoke({"--input", "-"}, PermutedCopyQuery("[0, 3, 1, 2]"));
  ExpectResult(result, 0, std::string(kPermutedCopyRoute) + "\n", "",
               "permuted CPY selection");
}

void TestInvalidPermutationsDoNotMatch() {
  struct InvalidPermutationCase {
    const char *name;
    const char *permutation;
  };
  const std::vector<InvalidPermutationCase> cases = {
      {"rank mismatch", "[0, 3, 1]"},
      {"duplicate axis", "[0, 3, 1, 1]"},
      {"axis above rank", "[0, 3, 1, 4]"},
      {"negative axis", "[0, 3, 1, -1]"},
  };

  for (const auto &test_case : cases) {
    const auto result = Invoke(
        {"--input", "-"}, PermutedCopyQuery(test_case.permutation));
    ExpectResult(result, 1, "",
                 "error: NO_MATCH: no route matched operation 'CPY'\n",
                 std::string("invalid permutation: ") + test_case.name);
  }
}

void TestAllowlistAcceptsSelectedRoute() {
  const std::string query = R"json({
  "op": "ABS",
  "tensors": {
    "src0": {"dtype": "F32", "dimensions": [5, 7], "strides": [1, 5]},
    "dst": {"dtype": "F32", "dimensions": [5, 7], "strides": [1, 5]}
  },
  "allowed_route_ids": ["abs_f32_contiguous_4d"]
})json";

  const auto result = Invoke({"--input", "-"}, query);
  ExpectResult(result, 0, std::string(kF32Route) + "\n", "",
               "matching allowlist");
}

void TestNullAllowlistAllowsAllRoutes() {
  const std::string query = R"json({
  "op": "ABS",
  "tensors": {
    "src0": {"dtype": "F32", "dimensions": [5, 7], "strides": [1, 5]},
    "dst": {"dtype": "F32", "dimensions": [5, 7], "strides": [1, 5]}
  },
  "allowed_route_ids": null
})json";

  const auto result = Invoke({"--input", "-"}, query);
  ExpectResult(result, 0, std::string(kF32Route) + "\n", "",
               "null allowlist");
}

void TestAllowlistFiltersSelectedRoute() {
  const std::string query = R"json({
  "op": "ABS",
  "tensors": {
    "src0": {"dtype": "F32", "dimensions": [5, 7], "strides": [1, 5]},
    "dst": {"dtype": "F32", "dimensions": [5, 7], "strides": [1, 5]}
  },
  "allowed_route_ids": ["abs_f16_contiguous_4d"]
})json";

  const auto result = Invoke({"--input", "-"}, query);
  ExpectResult(result, 1, "",
               "error: NO_MATCH: no route matched operation 'ABS'\n",
               "allowlist filtering");
}

void TestEmptyAllowlistMatchesNoRoutes() {
  const std::string query = R"json({
  "op": "ABS",
  "tensors": {
    "src0": {"dtype": "F32", "dimensions": [5, 7], "strides": [1, 5]},
    "dst": {"dtype": "F32", "dimensions": [5, 7], "strides": [1, 5]}
  },
  "allowed_route_ids": []
})json";

  const auto result = Invoke({"--input", "-"}, query);
  ExpectResult(result, 1, "",
               "error: NO_MATCH: no route matched operation 'ABS'\n",
               "empty allowlist");
}

void TestRouteIdsRemainCaseSensitive() {
  const std::string query = R"json({
  "op": "ABS",
  "tensors": {
    "src0": {"dtype": "F32", "dimensions": [5, 7], "strides": [1, 5]},
    "dst": {"dtype": "F32", "dimensions": [5, 7], "strides": [1, 5]}
  },
  "allowed_route_ids": ["ABS_F32_CONTIGUOUS_4D"]
})json";

  const auto result = Invoke({"--input", "-"}, query);
  ExpectResult(result, 1, "",
               "error: NO_MATCH: no route matched operation 'ABS'\n",
               "case-sensitive allowlist route ID");
}

void TestMatchingExpectedRoute() {
  const auto result =
      Invoke({"--input", "-", "--expect-route", kF32Route}, ValidQuery());
  ExpectResult(result, 0, std::string(kF32Route) + "\n", "",
               "matching expected route");
}

void TestMismatchingExpectedRoute() {
  const auto result = Invoke(
      {"--input", "-", "--expect-route", "abs_f16_contiguous_4d"},
      ValidQuery());
  ExpectResult(
      result, 1, "",
      "error: expected route 'abs_f16_contiguous_4d' but selected "
      "'abs_f32_contiguous_4d'\n",
      "mismatching expected route");
}

void TestValidQueryWithNoMatch() {
  const std::string query = R"json({
  "op": "ABS",
  "tensors": {
    "src0": {"dtype": "I32", "dimensions": [5, 7], "strides": [1, 5]},
    "dst": {"dtype": "I32", "dimensions": [5, 7], "strides": [1, 5]}
  }
})json";

  const auto result = Invoke({"--input", "-"}, query);
  ExpectResult(result, 1, "",
               "error: NO_MATCH: no route matched operation 'ABS'\n",
               "valid query without a matching route");
}

void TestTensorRolesRemainCaseSensitive() {
  const std::string query = R"json({
  "op": "ABS",
  "tensors": {
    "Src0": {"dtype": "F32", "dimensions": [5, 7], "strides": [1, 5]},
    "dst": {"dtype": "F32", "dimensions": [5, 7], "strides": [1, 5]}
  }
})json";

  const auto result = Invoke({"--input", "-"}, query);
  ExpectResult(result, 1, "",
               "error: NO_MATCH: no route matched operation 'ABS'\n",
               "case-sensitive tensor role");
}

void TestUnsupportedOperation() {
  const std::string query = R"json({
  "op": "DOES_NOT_EXIST",
  "tensors": {
    "src0": {"dtype": "F32", "dimensions": [5, 7], "strides": [1, 5]},
    "dst": {"dtype": "F32", "dimensions": [5, 7], "strides": [1, 5]}
  }
})json";

  const auto result = Invoke({"--input", "-"}, query);
  ExpectResult(result, 1, "",
               "error: UNSUPPORTED: operation 'DOES_NOT_EXIST' is not "
               "supported\n",
               "unsupported operation");
}

void TestSupportedOperationArithmeticOverflow() {
  const std::string query = R"json({
  "op": "ABS",
  "tensors": {
    "src0": {
      "dtype": "F32",
      "dimensions": [3037000500, 3037000500],
      "strides": [1, 3037000500]
    },
    "dst": {
      "dtype": "F32",
      "dimensions": [3037000500, 3037000500],
      "strides": [1, 3037000500]
    }
  }
})json";

  const auto result = Invoke({"--input", "-"}, query);
  ExpectResult(
      result, 1, "",
      "error: UNSUPPORTED: selector cannot evaluate operation 'ABS'\n",
      "supported operation arithmetic overflow");
}

void TestMalformedJson() {
  const auto result = Invoke({"--input", "-"}, R"json({"op": "ABS")json");
  ExpectResult(result, 2, "", "error: malformed JSON\n", "malformed JSON");
}

struct SchemaErrorCase {
  const char *name;
  const char *json;
  const char *error;
};

void TestSchemaErrors() {
  const std::vector<SchemaErrorCase> cases = {
      {"non-object root", R"json([])json",
       "error: input must be an object\n"},
      {"missing op", R"json({"tensors": {}})json",
       "error: input is missing required field 'op'\n"},
      {"wrong op type", R"json({"op": 7, "tensors": {}})json",
       "error: input field 'op' must be a string\n"},
      {"missing tensors", R"json({"op": "ABS"})json",
       "error: input is missing required field 'tensors'\n"},
      {"wrong tensors type", R"json({"op": "ABS", "tensors": []})json",
       "error: input field 'tensors' must be an object\n"},
      {"unknown root field",
       R"json({"op": "ABS", "tensors": {}, "unexpected": true})json",
       "error: input contains unknown field 'unexpected'\n"},
      {"wrong tensor type",
       R"json({"op": "ABS", "tensors": {"src0": []}})json",
       "error: input tensor 'src0' must be an object\n"},
      {"missing dtype",
       R"json({"op":"ABS","tensors":{"src0":{"dimensions":[],"strides":[]}}})json",
       "error: input tensor 'src0' is missing required field 'dtype'\n"},
      {"wrong dtype type",
       R"json({"op":"ABS","tensors":{"src0":{"dtype":4,"dimensions":[],"strides":[]}}})json",
       "error: input tensor 'src0' field 'dtype' must be a string\n"},
      {"missing dimensions",
       R"json({"op":"ABS","tensors":{"src0":{"dtype":"F32","strides":[]}}})json",
       "error: input tensor 'src0' is missing required field 'dimensions'\n"},
      {"wrong dimensions type",
       R"json({"op":"ABS","tensors":{"src0":{"dtype":"F32","dimensions":4,"strides":[]}}})json",
       "error: input tensor 'src0' field 'dimensions' must be an array\n"},
      {"missing strides",
       R"json({"op":"ABS","tensors":{"src0":{"dtype":"F32","dimensions":[]}}})json",
       "error: input tensor 'src0' is missing required field 'strides'\n"},
      {"wrong strides type",
       R"json({"op":"ABS","tensors":{"src0":{"dtype":"F32","dimensions":[],"strides":4}}})json",
       "error: input tensor 'src0' field 'strides' must be an array\n"},
      {"wrong permutation type",
       R"json({"op":"ABS","tensors":{"src0":{"dtype":"F32","dimensions":[],"strides":[],"permutation":"identity"}}})json",
       "error: input tensor 'src0' field 'permutation' must be an array\n"},
      {"non-integer permutation entry",
       R"json({"op":"ABS","tensors":{"src0":{"dtype":"F32","dimensions":[1],"strides":[1],"permutation":[1.5]}}})json",
       "error: input tensor 'src0' field 'permutation' element 0 must be a "
       "signed 64-bit integer\n"},
      {"out-of-int64 permutation entry",
       R"json({"op":"ABS","tensors":{"src0":{"dtype":"F32","dimensions":[1],"strides":[1],"permutation":[9223372036854775808]}}})json",
       "error: input tensor 'src0' field 'permutation' element 0 is outside "
       "the signed 64-bit integer range\n"},
      {"unknown tensor field",
       R"json({"op":"ABS","tensors":{"src0":{"dtype":"F32","dimensions":[],"strides":[],"extra":0}}})json",
       "error: input tensor 'src0' contains unknown field 'extra'\n"},
      {"wrong allowlist type",
       R"json({"op":"ABS","tensors":{},"allowed_route_ids":"abs_f32_contiguous_4d"})json",
       "error: input field 'allowed_route_ids' must be null or an array of "
       "strings\n"},
      {"wrong allowlist entry type",
       R"json({"op":"ABS","tensors":{},"allowed_route_ids":[7]})json",
       "error: input field 'allowed_route_ids' element 0 must be a string\n"},
  };

  for (const auto &test_case : cases) {
    const auto result = Invoke({"--input", "-"}, test_case.json);
    ExpectResult(result, 2, "", test_case.error,
                 std::string("schema error: ") + test_case.name);
  }
}

void TestUnequalDimensionsAndStrides() {
  const std::string query = R"json({
  "op": "ABS",
  "tensors": {
    "src0": {"dtype": "F32", "dimensions": [5, 7], "strides": [1]},
    "dst": {"dtype": "F32", "dimensions": [5, 7], "strides": [1, 5]}
  }
})json";

  const auto result = Invoke({"--input", "-"}, query);
  ExpectResult(
      result, 2, "",
      "error: input tensor 'src0' dimensions and strides must have equal "
      "length\n",
      "unequal dimensions and strides");
}

void TestNonIntegerDimension() {
  const std::string query = R"json({
  "op": "ABS",
  "tensors": {
    "src0": {"dtype": "F32", "dimensions": [5, 1.5], "strides": [1, 5]},
    "dst": {"dtype": "F32", "dimensions": [5, 7], "strides": [1, 5]}
  }
})json";

  const auto result = Invoke({"--input", "-"}, query);
  ExpectResult(
      result, 2, "",
      "error: input tensor 'src0' field 'dimensions' element 1 must be a "
      "signed 64-bit integer\n",
      "non-integer dimension");
}

void TestNonIntegerStride() {
  const std::string query = R"json({
  "op": "ABS",
  "tensors": {
    "src0": {"dtype": "F32", "dimensions": [5, 7], "strides": [1, "5"]},
    "dst": {"dtype": "F32", "dimensions": [5, 7], "strides": [1, 5]}
  }
})json";

  const auto result = Invoke({"--input", "-"}, query);
  ExpectResult(
      result, 2, "",
      "error: input tensor 'src0' field 'strides' element 1 must be a signed "
      "64-bit integer\n",
      "non-integer stride");
}

void TestOutOfInt64Value() {
  const std::string query = R"json({
  "op": "ABS",
  "tensors": {
    "src0": {
      "dtype": "F32",
      "dimensions": [9223372036854775808],
      "strides": [1]
    },
    "dst": {"dtype": "F32", "dimensions": [1], "strides": [1]}
  }
})json";

  const auto result = Invoke({"--input", "-"}, query);
  ExpectResult(
      result, 2, "",
      "error: input tensor 'src0' field 'dimensions' element 0 is outside "
      "the signed 64-bit integer range\n",
      "out-of-int64 dimension");
}

void TestHelp() {
  const auto result = Invoke({"--help"});
  ExpectResult(result, 0, kUsage, "", "help");
}

struct ArgumentErrorCase {
  const char *name;
  std::vector<std::string> args;
  const char *error;
};

void TestArgumentErrors() {
  const std::vector<ArgumentErrorCase> cases = {
      {"missing input", {}, "error: missing required --input\n"},
      {"missing input value", {"--input"},
       "error: missing value for --input\n"},
      {"duplicate input", {"--input", "-", "--input", "-"},
       "error: duplicate option --input\n"},
      {"missing expected route value", {"--input", "-", "--expect-route"},
       "error: missing value for --expect-route\n"},
      {"duplicate expected route",
       {"--input", "-", "--expect-route", kF32Route, "--expect-route",
        kF32Route},
       "error: duplicate option --expect-route\n"},
      {"unknown option", {"--bogus"},
       "error: unknown option '--bogus'\n"},
      {"help combined with another option", {"--help", "--bogus"},
       "error: --help cannot be combined with other options\n"},
  };

  for (const auto &test_case : cases) {
    const auto result = Invoke(test_case.args, ValidQuery());
    ExpectResult(result, 2, "", test_case.error,
                 std::string("argument error: ") + test_case.name);
  }
}

void TestUnreadableInputFile() {
  const auto suffix =
      std::chrono::steady_clock::now().time_since_epoch().count();
  const std::filesystem::path missing_path =
      std::filesystem::temp_directory_path() /
      ("ggml-hrx-v2-route-selector-missing-" + std::to_string(suffix) +
       ".json");
  const auto result = Invoke({"--input", missing_path.string()});
  ExpectResult(result, 2, "",
               "error: cannot read input file '" + missing_path.string() +
                   "'\n",
               "unreadable input file");
}

}  // namespace

int main() {
  TestSelectsContiguousAbs();
  TestNormalizesOperationAndDTypes();
  TestNullAndExplicitIdentityPermutations();
  TestSelectsPermutedCopy();
  TestInvalidPermutationsDoNotMatch();
  TestAllowlistAcceptsSelectedRoute();
  TestNullAllowlistAllowsAllRoutes();
  TestAllowlistFiltersSelectedRoute();
  TestEmptyAllowlistMatchesNoRoutes();
  TestRouteIdsRemainCaseSensitive();
  TestMatchingExpectedRoute();
  TestMismatchingExpectedRoute();
  TestValidQueryWithNoMatch();
  TestTensorRolesRemainCaseSensitive();
  TestUnsupportedOperation();
  TestSupportedOperationArithmeticOverflow();
  TestMalformedJson();
  TestSchemaErrors();
  TestUnequalDimensionsAndStrides();
  TestNonIntegerDimension();
  TestNonIntegerStride();
  TestOutOfInt64Value();
  TestHelp();
  TestArgumentErrors();
  TestUnreadableInputFile();

  if (g_failures != 0) {
    std::cerr << g_failures << " test failure(s)\n";
    return 1;
  }
  return 0;
}
