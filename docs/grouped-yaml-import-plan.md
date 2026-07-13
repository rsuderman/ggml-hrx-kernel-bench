# Grouped YAML Route Import Workflow

Grouped YAML import now means descriptor route import. The importer should read
grouped YAML cases, materialize tensor descriptors and scalar values, and ask
the v2 route catalog whether a route matches. It must not lower cases through
custom Python mapping registries or op-specific route-resolution code.

## Current Path

- Entry point: `src/ggml_hrx_kernel_bench/yaml_route_import.py`
- Build-time checker: `tests/infra/check_yaml_route_import.py`
- Runtime test materializer: `tests/infra/generate_kernel_runtime_tests_cmake.py`
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

The generated runtime CTest suites are named with
`kernel-run-*-yaml-route-import-v2-<OP>-generated`.

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
- [ ] 4. Migrate supported ops from legacy generated-runtime execution to
  descriptor execution. Start with ops already validated through descriptor
  execution, such as `EXP`, `SQRT`, and model `SET_ROWS`, then move through
  low-risk pointwise and indexed families.
- [ ] 5. Simplify or narrow legacy generated-runtime registration once the
  inventory shows descriptor coverage is sufficient for an op. Keep legacy
  runtime only for ops the descriptor harness cannot yet represent.
- [ ] 6. Expand descriptor execution coverage beyond the current validated set.
  Prefer small f32 approximate pointwise slices first, then compact model-level
  cases. Every widened executing surface requires targeted HSA runtime
  validation outside the sandboxed harness path.
- [ ] 7. Update this TODO as migration items land. Replace broad coverage and
  legacy-runtime tasks with concrete remaining op lists once the inventory
  exists.

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
