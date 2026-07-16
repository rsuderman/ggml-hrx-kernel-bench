# Native v2 route selector CLI

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

The input contains an operation and its tensor descriptors. An optional
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

On success the command prints the selected route ID. `--expect-route` checks
the result without changing the input allowlist.
