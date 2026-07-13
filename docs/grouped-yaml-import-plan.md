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
- [ ] Land or otherwise carry the paired HRX `iree-run-loom` support for
  `--expected-kernel-buffer-tolerance` in every environment that runs the new
  harness.
- [ ] Add HRX-side unit tests for tolerant HAL expected-buffer comparison.
- [ ] Expand descriptor execution coverage beyond the current smoke-tested
  cases, starting with narrow f32 approximate and model-level slices.
- [ ] Re-triage current unmatched route-import artifacts before picking the
  next operation slice.
- [ ] Validate every widened executing kernel surface with targeted HSA runtime
  tests outside the sandboxed harness path.
- [ ] Simplify or retire redundant legacy generated-runtime execution paths once
  descriptor execution has enough operation coverage to be the primary runtime
  validation utility.

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
