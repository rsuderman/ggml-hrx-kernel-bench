# Scope

- Date: 2026-07-06
- Area or backend: grouped-YAML import, ADD route lowering, v1/v2 routing
- User request: fix the build after grouped YAML changed `perm1` to an explicit permutation order
- Primary correctness command: `cmake --build build --target kernel-llama-cpp-tests-import-coverage-v2`
- Escalation or device access used: unsandboxed `ctest` for runtime kernel validation

# Run Status

- Build status: passing
- Main run status: passing after importer and routing updates
- First failing point: v2 import coverage expected `ADD` pass count `32`, actual `45`
- Hard blockers, if any: none

# Commands

- `PYTHONPATH=src pytest tests/test_routing_api.py`
- `cmake --build build --target kernel-llama-cpp-tests-import-coverage`
- `cmake --build build --target kernel-llama-cpp-tests-import-coverage-v2`
- `ctest --test-dir build -R '^kernel-run-llama-cpp-tests-v2-ADD-generated$' --output-on-failure`
- `cmake --build build`

# Real Worklist

## No Remaining Build Blocker

- Operation or feature: ADD grouped-YAML import for explicit `perm1` permutations
- Expected behavior: base-order `perm1` should preserve existing mappings; v2 generic 4D routing should lower the known `src1` permutation used by llama.cpp
- Actual behavior: fixed
- Evidence: import coverage now passes and the v2 ADD generated runtime test passes
- Shapes, dtypes, layouts, or config: `f32`, `nf=1`, contiguous and generic 4D ADD cases, including `perm1=[1, 2, 0, 3]`
- Source references: `src/ggml_hrx_kernel_bench/import_mapping_registry.py`, `src/ggml_hrx_kernel_bench/routing/v2/import_resolution.py`
- Validation status: validated
- Next concrete action: none required for this build fix

# Intentional Negative Coverage

- `ADD` `f16` remains unmapped in v1/v2 because there is no matching dtype route.
- `ADD` cases with `nf != 1` remain unmapped in v2 because pointwise lowering still requires `nf=1`.
- v1 ADD lowering still requires `perm1=[0, 1, 2, 3]`; only the v2 generic 4D route was widened for permuted `src1`.

# Supplemental Probes

- Probe: inspected generated per-op ADD import summaries under `build/tests/kernels/artifacts/llama-cpp-tests-import-v1/ops/ADD/` and `.../v2/ops/ADD/`
- Why it was needed: confirm whether the higher v2 pass count was legitimate support growth or an over-broad match
- Result: v2 now maps all `f32` `nf=1` ADD cases into either `add_f32_contiguous_1d` or `add_f32_generic_4d`; unmapped cases are limited to dtype gaps and `nf != 1`

# Repository Changes

- Files changed in repository: `src/ggml_hrx_kernel_bench/import_mapping_registry.py`, `src/ggml_hrx_kernel_bench/routing/v2/import_resolution.py`, `tests/test_routing_api.py`, `tests/kernels/data/llamacpp_test.v2.import-coverage.json`
- Files changed outside repository: none

# External Artifacts

- `build/tests/kernels/artifacts/llama-cpp-tests-import-v1/`
- `build/tests/kernels/artifacts/llama-cpp-tests-import-v2/`

# Next Iteration

- Highest-priority next fix: widen dtype coverage if ADD `f16` support is desired
- Fastest validation command after that fix: `cmake --build build --target kernel-llama-cpp-tests-import-coverage-v2`
- Remaining unknowns: no additional unknowns for the `perm1` schema migration

# Current Unsupported F32 ADD Cases

- V1 remaining unsupported `f32` ADD cases: `26`
- V1 `perm1`-gated cases: source case indices `9, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34`
- V1 broadcast/repeat cases rejected by same-shape lowering: source case indices `21, 23, 25, 27, 29, 33`
- V1 `nf != 1` cases: source case indices `35, 36, 37, 38, 46, 47, 48, 49`
- V2 remaining unsupported `f32` ADD cases: `8`
- V2 `nf != 1` cases: source case indices `35, 36, 37, 38, 46, 47, 48, 49`

# Current ADD Correctness Triage

- Date: 2026-07-06
- User request: triage the ADD tests that are currently failing
- Primary correctness command: `ctest --test-dir build -R 'ADD' --output-on-failure`
- Escalation or device access used: unsandboxed `ctest` per repo policy

## Run Status

- Runtime ADD CTest status: passing
- Import materialization status: no rebuild needed, existing v2 ADD import artifacts still show `45` mapped and `53` unmapped cases
- Current failing point: none in the runtime ADD CTest path; remaining ADD failures are import-side unmapped cases, not kernel execution failures
- Evidence:
  - `ctest --test-dir build -R 'ADD' --output-on-failure` passed for both `kernel-run-llama-cpp-tests-ADD-generated` and `kernel-run-llama-cpp-tests-v2-ADD-generated`
  - `build/tests/kernels/artifacts/llama-cpp-tests-import-v2/ops/ADD/import-summary.md` reports `45` mapped and `53` unmapped cases

## Real Worklist

### ADD v2 `f32` `nf != 1` grouped-YAML cases remain unsupported

- Operation or feature: ADD grouped-YAML import lowering for v2 generic pointwise cases with `nf != 1`
- Expected behavior: these `f32` ADD cases should materialize to a v2 route if the importer is meant to support grouped pointwise factorization beyond `nf=1`
- Actual behavior: all eight such cases remain unmapped as `shape_lowering_not_implemented`
- Evidence: the per-op summary and unmapped artifact show source case indices `35, 36, 37, 38, 46, 47, 48, 49` with detail `contiguous pointwise routing requires nf=1`
- Shapes, dtypes, layouts, or config:
  - `ne=[10,5,4,3]`, `nr=[2,1,1,1]`, `nf=2`
  - `ne=[10,5,4,3]`, `nr=[1,1,2,1]`, `nf=4`
  - `ne=[10,5,4,3]`, `nr=[1,1,2,2]`, `nf=6`
  - `ne=[10,5,4,3]`, `nr=[1,2,2,2]`, `nf=7`
  - `ne=[16,5,4,3]`, `nr=[1,1,1,1]`, `nf=16`
  - `ne=[16,5,4,3]`, `nr=[1,2,1,1]`, `nf=3`
  - `ne=[16,5,4,3]`, `nr=[1,1,1,2]`, `nf=5`
  - `ne=[16,5,4,3]`, `nr=[2,2,2,2]`, `nf=8`
- Source references:
  - `src/ggml_hrx_kernel_bench/routing/v2/import_resolution.py:118`
  - `src/ggml_hrx_kernel_bench/routing/v2/import_resolution.py:147`
  - `build/tests/kernels/artifacts/llama-cpp-tests-import-v2/ops/ADD/import-summary.md`
- Validation status: confirmed by artifact inspection and fresh unsandboxed ADD-focused CTest
- Next concrete action: decide whether `nf != 1` should lower into the generic 4D route; if yes, widen `lower_generic_pointwise_tensors()` and rerun v2 import coverage plus the ADD runtime CTest

## Intentional Negative Coverage

- `ADD` `f16` remains unmapped in v2 as `no_dtype_mapping` for `45` cases because only `add_f32` routes exist today.

## Supplemental Probes

- Probe: inspected `build/tests/kernels/artifacts/llama-cpp-tests-import-v2/ops/ADD/unmapped.json`
- Why it was needed: distinguish real runtime test failures from import support gaps after the stale `LastTestsFailed.log` entry
- Result: the only remaining v2 ADD `f32` gaps are the eight `nf != 1` cases above
