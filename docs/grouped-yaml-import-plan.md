# Grouped YAML Route Import Workflow

Grouped YAML import now means descriptor route import. The importer should read
grouped YAML cases, materialize tensor descriptors and scalar values, and ask
the v2 route catalog whether a route matches. It must not lower cases through
custom Python mapping registries or op-specific route-resolution code.

## Current Path

- Entry point: `src/ggml_hrx_kernel_bench/yaml_route_import.py`
- Build-time checker: `tests/infra/check_yaml_route_import.py`
- Descriptor test materializer:
  `tests/infra/generate_loom_descriptor_tests_cmake.py`
- Descriptor execution runner:
  `tests/infra/run_loom_execution_descriptors.py`
- Llama.cpp input: `tests/kernels/data/llamacpp_test.v2.yaml`
- Model input: `tests/models/data/Llama-3.3-8B-Instruct.Q8_0.v2.yaml`
- Llama.cpp expected coverage:
  `tests/kernels/data/llamacpp.import-coverage.json`
- Model expected coverage:
  `tests/models/data/llama-8b-q8.import-coverage.json`

The build-time import targets are:

- `kernel-llama-cpp-yaml-route-import-v2`
- `kernel-model-llama-3-3-8b-q8-0-yaml-route-import-v2`
- `kernel-yaml-route-import-v2`

The old generated runtime CTest suites were named with
`kernel-run-*-yaml-route-import-v2-<OP>-generated`. They have been retired in
favor of descriptor generate/prepare/execute tests backed by
`ggml-hrx-run-loom-simple`.

## Retirement Objective

The objective is to remove the old generated-runtime testing tool and its
`kernel-run-*` CTest registrations. Do not add unrelated operation coverage
unless it directly eliminates a dependency on the old tool.

Retirement is complete when:

- no generated `kernel-run-*-yaml-route-import-v2-<OP>-generated` tests are
  registered for kernel or model YAML route-import targets
- descriptor generate/prepare/execute tests are the only generated runtime
  validation path for route-import artifacts
- the old generated-runtime helper code is removed or no longer reachable from
  CMake
- the harness inventory reports zero legacy runtime registrations

## Descriptor Harness Migration TODO

- [x] Add a simple descriptor runner, `ggml-hrx-run-loom-simple`, that can load
  compact execution descriptors and invoke Loom kernels.
- [x] Generate compact descriptor manifests from route-import artifacts.
- [x] Register descriptor generate, prepare, and execute tests from CMake.
- [x] Enable descriptor execution through the default generated harness path.
- [x] Gate descriptor execute tests that require HSA resources behind the HSA
  descriptor-test option.
- [x] Preserve descriptor `close` tolerances when bridging descriptor execution
  to `iree-run-loom` expected-buffer checks.
- [x] Validate the new path with targeted `EXP` and `SQRT` descriptor execution
  smoke tests.
- [ ] 1. Add HRX-side unit tests for tolerant HAL expected-buffer comparison.
  Cover a floating-point expected buffer that passes within tolerance, a
  floating-point expected buffer that fails outside tolerance, and the zero
  tolerance path that remains exact. This is dependency-owned HRX work and
  should not be implemented in this repository's checkout.
- [x] 2. Verify the paired HRX `iree-run-loom`
  `--expected-kernel-buffer-tolerance` support is present in every environment
  that runs the new harness. Add a lightweight capability check if stale tools
  can be selected accidentally.
- [x] 3. Build a descriptor-vs-legacy harness inventory. For each op, report
  route-import matched counts, descriptor emitted/skipped/unsupported counts,
  descriptor CTest generate/prepare/execute registration, legacy
  `kernel-run-*` registration, and whether HSA execution is gated or enabled.
  Current reports:
  `/home/rsuderman/codex/ggml-hrx-kernel-bench-harness-inventory-kernels-20260713.{json,md}`
  and
  `/home/rsuderman/codex/ggml-hrx-kernel-bench-harness-inventory-model-20260713.{json,md}`.
  The kernel inventory covers 115 ops, with 115 descriptor execute tests and
  115 legacy runtime tests registered. The model inventory covers 10 ops, with
  9 descriptor execute tests and 8 legacy runtime tests registered.
- [x] 4. Migrate supported ops from legacy generated-runtime execution to
  descriptor execution. Start with ops already validated through descriptor
  execution, such as `EXP`, `SQRT`, and model `SET_ROWS`, then move through
  low-risk pointwise and indexed families.
  - [x] Migrate kernel `EXP` and `SQRT` off legacy `kernel-run-*`
    registration. Descriptor generate, prepare, and execute registrations
    remain active for both ops.
  - [x] Migrate the validated small unary pointwise kernel slice `ABS`, `NEG`,
    `RELU`, and `SQR` off legacy `kernel-run-*` registration after targeted
    descriptor HSA execution passed.
  - [x] Migrate the validated scalar pointwise kernel slice `CLAMP` and
    `SCALE` off legacy `kernel-run-*` registration after targeted descriptor
    HSA execution passed.
  - [x] Migrate the validated data movement/indexed kernel slice `CONT` and
    `GET_ROWS` off legacy `kernel-run-*` registration after targeted descriptor
    HSA execution passed.
  - [x] Migrate the validated binary pointwise kernel slice `ADD`, `DIV`,
    `MUL`, and `SUB` off legacy `kernel-run-*` registration after targeted
    descriptor HSA execution passed.
  - [x] Confirm model `SET_ROWS` is descriptor-only; it remains excluded from
    legacy model runtime registration. Migrate the separately generated kernel
    `SET_ROWS` suite off legacy `kernel-run-*` registration after targeted
    descriptor HSA execution passed.
  - [x] Migrate the validated normalization kernel slice `RMS_NORM` off legacy
    `kernel-run-*` registration after preserving per-case `eps` scalar ABI,
    correcting row-based descriptor dispatch, and passing targeted descriptor
    HSA execution.
  - [x] Migrate the validated gated activation kernel slice `SWIGLU` off legacy
    `kernel-run-*` registration after enabling packed-input descriptor emission
    and passing targeted descriptor HSA execution.
  - [x] Migrate the validated softmax kernel slice `SOFT_MAX` off legacy
    `kernel-run-*` registration after adding scalar `scale` ABI, treating
    `mask` as an input binding, correcting row-based descriptor dispatch, and
    passing targeted descriptor HSA execution.
  - [x] Migrate the validated copy/cast kernel slice `CPY` off legacy
    `kernel-run-*` registration after enabling generated copy descriptor
    families, adding `bf16` buffer binding support, exposing CPY oracle fixture
    arrays, and passing targeted descriptor HSA execution.
  - [x] Migrate the validated rotary embedding kernel slice `ROPE` off legacy
    `kernel-run-*` registration after adding scalar `theta_scale`,
    `freq_scale`, and `attn_factor` ABI, mapping `src1` to generated
    `positions` fixtures, and passing targeted descriptor HSA execution.
  - [x] Migrate the validated matrix multiplication kernel slice `MUL_MAT` off
    legacy `kernel-run-*` registration after enabling f32/f16 and packed
    q4_k/q5_k/q6_k/q8_0 buffer descriptor families, correcting descriptor
    dispatch order for matmul workgroups, and passing targeted descriptor HSA
    execution.
  Latest step-4 inventories:
  `/home/rsuderman/codex/ggml-hrx-kernel-bench-harness-inventory-kernels-step4l-20260713.{json,md}`
  and
  `/home/rsuderman/codex/ggml-hrx-kernel-bench-harness-inventory-model-step4-20260713.{json,md}`.
  The kernel suite now has 94 legacy runtime registrations remaining, and no
  op with emitted descriptor cases still has legacy runtime registration.
- [x] 5. Stop registering empty legacy generated-runtime tests. For ops with
  zero generated manifest entries, remove the old `kernel-run-*` CTest
  registration instead of keeping a no-op legacy test as a placeholder.
- [x] 6. For any remaining legacy test with generated manifest entries, either
  validate and move it to descriptor execution or record the blocker that keeps
  it on the old tool. The retirement inventory has no remaining legacy tests.
- [x] 7. Remove the old generated-runtime CMake path once the legacy
  registration count reaches zero. This includes the CMake helper that creates
  `kernel-run-*` tests for route-import artifacts and any stale build target
  dependency that exists only for the old path.
- [x] 8. Remove or archive the old generated-runtime Python runner only after
  CMake no longer references it and descriptor execution covers the intended
  route-import validation path.
- [x] 9. Refresh the harness inventory and workflow docs after each retirement
  step. Current retirement inventories:
  `/home/rsuderman/codex/ggml-hrx-kernel-bench-harness-inventory-kernels-old-tool-retirement-20260713.{json,md}`
  and
  `/home/rsuderman/codex/ggml-hrx-kernel-bench-harness-inventory-model-old-tool-retirement-20260713.{json,md}`.
  The kernel inventory covers 115 ops, with 115 descriptor execute tests, 401
  emitted descriptor cases, and zero legacy runtime registrations. The model
  inventory covers 10 ops, with 9 descriptor execute tests, 14 emitted
  descriptor cases, and zero legacy runtime registrations.
  Validation run on 2026-07-13:
  `ctest --test-dir build -N -R 'kernel-run-.*yaml-route-import-v2'`
  reported zero tests, and targeted model descriptor
  generate/prepare/execute for `ADD`, `CPY`, `GET_ROWS`, `MUL`, and
  `RMS_NORM` passed outside the sandbox.

## Expected Outputs

The importer should keep every YAML case visible in the generated status
artifacts. Supported cases become route matches; unsupported cases remain in
unmatched reports with enough detail to explain why no descriptor route matched.

Per-op artifacts are written under:

`build/tests/kernels/artifacts/grouped-yaml-import/ops/<OP>/`

or the corresponding build output directory for the selected CMake target.

Do not reintroduce the old generated `imported-workload.json` /
`unmapped.json` model. Current status is tracked by the route-import coverage
JSON and route match/unmatch artifacts.

## Iteration Process

1. Pick one op family and one narrow dtype/layout slice.
2. Inspect the YAML cases and generated per-op route-import artifacts.
3. Report the operation surface area before changing code:
   - dtype combinations
   - ranks and layouts
   - scalar/config fields used by the YAML
   - which cases already match routes
   - representative missing or unsupported cases
4. Implement the smallest descriptor, route predicate, or kernel-surface change
   needed for that slice.
5. Validate generated configs structurally before updating expected coverage.
6. Run the targeted build-time route-import target.
7. If the change widens executing kernel coverage, run the generated runtime
   tests for that op unless the environment blocks runtime execution.
8. Report what changed, what issues were found, and the next step.

Do not update expected coverage until the intended slice is validated and the
build-time coverage target passes.

## Kernel Source Layout

When route support needs a new or ported kernel variant, keep one routed variant
per `.loom` file. If two catalog routes target different exported kernels, split
them into separate `.loom` files and update each route's `kernel.path`.

Before runtime validation, rerun generated asset materialization so
`build/generated/assets/kernels/v2/` contains the route's referenced file.
