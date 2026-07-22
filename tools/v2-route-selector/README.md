# V2 route selector CLIs

`ggml-hrx-v2-route-selector` evaluates one JSON selector query against the
native v2 route table.

The build requires a globally installed nlohmann-json development package that
provides the `nlohmann_json` CMake config package. For example, Debian and
Ubuntu package it as `nlohmann-json3-dev`.

Run the tool with a file, or pass `-` to read the query from standard input:

```sh
build/tools/v2-route-selector/ggml-hrx-v2-route-selector \
  --input tools/v2-route-selector/testdata/abs_f32_contiguous_4d.json \
  --expect-route abs_f32_contiguous_4d
```

The input contains an operation and its tensor descriptors. Optional
`attributes` contains JSON-neutral operation attributes, and an optional
`allowed_route_ids` array restricts which routes may be selected:

```json
{
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
  },
  "allowed_route_ids": ["abs_f32_contiguous_4d"]
}
```

Each tensor may also provide a `permutation` array. Omitting it or setting it
to `null` means the identity permutation. An explicit permutation must contain
one signed 64-bit integer per tensor dimension. Rank mismatches, duplicate
axes, and axes outside the tensor rank produce no match; invalid JSON types or
integers outside the signed 64-bit range are input errors.

An absent or empty `attributes` object preserves tensor-only selection.
Attribute values may recursively contain null, booleans, signed integers,
floating-point values, strings, arrays, and string-keyed objects. Until native
attribute predicates are implemented, a nonempty object returns `UNSUPPORTED`.

On success the command prints the selected route ID. `--expect-route` checks
the result without changing the input allowlist.

## Python selector CLI

`python_v2_route_selector.py` provides the same single-query interface through
the Python v2 routing implementation. Pass a JSON file to `--input`, or use
`-` to read one query from standard input:

```sh
head -n 1 build/tests/kernels/artifacts/llama-cpp-yaml-route-import-v2/route-queries.jsonl \
  | python tools/v2-route-selector/python_v2_route_selector.py \
      --input - \
      --routing-dir build/generated/assets/catalog/v2
```

The command prints only the selected route ID. It exits with status `0` when a
route matches, `1` when no route matches, and `2` for invalid input, catalog,
or command-line usage.

## Batch mode

Both selector CLIs accept `--batch` together with `--input <file|->`. Batch
input is JSONL, and blank physical lines are ignored while retaining their
line numbers. For example:

```sh
python tools/v2-route-selector/python_v2_route_selector.py \
  --input build/tests/kernels/artifacts/llama-cpp-yaml-route-import-v2/route-queries.jsonl \
  --routing-dir build/generated/assets/catalog/v2 \
  --batch
```

Batch output is compact JSONL with one record for each nonblank input line:

```jsonl
{"line":2,"status":"MATCH","route_id":"abs_f32_contiguous_4d"}
{"line":4,"status":"NO_MATCH","diagnostic":"no route matched operation 'ABS'"}
{"line":7,"status":"ERROR","diagnostic":"malformed JSON"}
```

The possible statuses are `MATCH`, `NO_MATCH`, `UNSUPPORTED`, and `ERROR`.
Malformed rows and selection failures do not stop later rows from being
processed. A completely processed stream exits `0`, even when it contains
non-matching or invalid rows; command-line, input-file, catalog, and stream I/O
failures exit `2`. The native CLI rejects combining `--batch` with
`--expect-route`.

## Native/Python parity checker

`check_route_selector_parity.py` compares both selector CLIs for every
nonblank record in a route-query JSONL file:

```sh
python tests/infra/check_route_selector_parity.py \
  --route-queries build/tests/kernels/artifacts/llama-cpp-yaml-route-import-v2/route-queries.jsonl \
  --routing-dir build/generated/assets/catalog/v2 \
  --python-selector tools/v2-route-selector/python_v2_route_selector.py \
  --native-selector build/tools/v2-route-selector/ggml-hrx-v2-route-selector
```

Parity is strict: both selectors must return the same nonempty route ID for
every query. No-match, unsupported queries, selector errors, timeouts,
malformed output, and differing route IDs are all failures. The checker sends
the complete input to each CLI once in batch mode, validates both result
streams, accumulates all line-level failures, and reports each one with its
physical JSONL line number and original JSON.

`add_yaml_route_import_target()` registers the same check as
`<import-target>-route-selector-parity`, labeled `routing;parity`. Build the
project before running CTest so the generated JSONL inputs and native selector
exist. Route mismatches are soft by default: the checker exits with code `77`,
which CTest records as `Skipped` through its `SKIP_RETURN_CODE` property.
Process, catalog, I/O, and result-protocol failures still fail the test.

Set `ENFORCE_ROUTER_PARITY` to any value when mismatches should fail normally:

```sh
ENFORCE_ROUTER_PARITY=1 ctest --test-dir build -R 'route-selector-parity$'
```
