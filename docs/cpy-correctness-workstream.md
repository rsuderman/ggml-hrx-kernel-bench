# CPY Correctness Workstream

## Scope

- Date: 2026-07-06
- Area or backend: grouped-YAML v2 CPY import and generated runtime validation
- User request: Triage the missing support for CPY
- Primary correctness command: `ctest --test-dir build --output-on-failure -R kernel-run-llama-cpp-tests-v2-CPY-generated`
- Escalation or device access used: yes, unsandboxed `ctest` per repository validation policy

## Run Status

- Build status: existing build artifacts were used; no rebuild was required for triage
- Main run status: pass for all currently mapped v2 CPY generated runtime cases
- First failing point: there is no runtime failure in the mapped slice; the support gap is in grouped-YAML import coverage
- Hard blockers, if any: none for triage

## Commands

- `sed -n '1,220p' build/tests/kernels/artifacts/llama-cpp-tests-import-v2/ops/CPY/import-summary.md`
- `python` probes over `build/tests/kernels/artifacts/llama-cpp-tests-import-v2/ops/CPY/unmapped.json`
- `python` probe over `build/tests/kernels/artifacts/llama-cpp-tests-import-v2/ops/CPY/imported-workload.json`
- `ctest --test-dir build --output-on-failure -R kernel-run-llama-cpp-tests-v2-CPY-generated`

## Real Worklist

### Strided Or Permuted Copy For Existing f16/f32 Families

- Operation or feature: CPY with non-identity source or destination permutation
- Expected behavior: grouped-YAML CPY cases for the already-routed `f16 -> f16`, `f32 -> f32`, `f16 -> f32`, and `f32 -> f16` pairs should either map to a strided/permuted COPY route or remain explicitly unsupported by design
- Actual behavior: `21` cases stop in import lowering with `copy lowering requires permute_src=[0, 0, 0, 0]`
- Evidence:
  - `build/tests/kernels/artifacts/llama-cpp-tests-import-v2/ops/CPY/import-summary.md`
  - `build/tests/kernels/artifacts/llama-cpp-tests-import-v2/ops/CPY/unmapped.json`
- Shapes, dtypes, layouts, or config:
  - `f16 -> f16`: `10` shape gaps total, `8` permutation-driven
  - `f32 -> f32`: `12` shape gaps total, `9` permutation-driven
  - `f16 -> f32`: `2` permutation-driven gaps
  - `f32 -> f16`: `2` permutation-driven gaps
  - Representative grouped-YAML permutations:
    - `permute_src=[0,2,1,3]`
    - `permute_src=[1,0,2,3]`
    - `permute_src=[0,3,1,2]` with `permute_dst=[0,2,1,3]`
- Source references:
  - `src/ggml_hrx_kernel_bench/routing/v2/import_resolution.py`
  - `catalog/v2/copy/copy_f32_f32_contiguous_1d.json`
  - `kernels/v2/copy/copy_f32_f32_contiguous_1d.loom`
- Validation status: confirmed import-side gap; runtime path for mapped cases passes
- Next concrete action: decide whether CPY needs a generic strided COPY route/kernel surface. The current contiguous v2 route requires equal contiguous strides, so this is not just an importer issue.

### Transposed Copy For Existing f16/f32 Families

- Operation or feature: CPY with `_src_transpose=1`
- Expected behavior: transposed grouped-YAML CPY cases should map only if the route/kernel contract can represent transposed source strides
- Actual behavior: `5` cases stop in import lowering with `copy lowering requires _src_transpose=0`
- Evidence:
  - `build/tests/kernels/artifacts/llama-cpp-tests-import-v2/ops/CPY/import-summary.md`
  - `build/tests/kernels/artifacts/llama-cpp-tests-import-v2/ops/CPY/unmapped.json`
- Shapes, dtypes, layouts, or config:
  - `f16 -> f16`: `2` transposed gaps
  - `f32 -> f32`: `3` transposed gaps
  - Representative shapes:
    - `ne=[256,4,1,1]`
    - `ne=[256,4,3,1]`
    - `ne=[256,4,3,3]`
- Source references:
  - `src/ggml_hrx_kernel_bench/routing/v2/import_resolution.py`
  - `catalog/v2/copy/copy_f32_f32_contiguous_1d.json`
  - `kernels/v2/copy/copy_f32_f32_contiguous_1d.loom`
- Validation status: confirmed import-side gap; current kernel ABI has no stride inputs, so the contiguous route cannot implement transpose as-is
- Next concrete action: if transposed CPY is in scope, add a route/kernel form that captures source and destination strides rather than only `total_size`.

### Missing Plain Scalar Dtype Families

- Operation or feature: CPY dtype coverage for non-quantized scalar families beyond the current four routed pairs
- Expected behavior: common scalar copies and casts should have explicit v2 families or be declared out of scope
- Actual behavior: `27` cases are `no_dtype_mapping`
- Evidence:
  - `build/tests/kernels/artifacts/llama-cpp-tests-import-v2/ops/CPY/import-summary.md`
  - `build/tests/kernels/artifacts/llama-cpp-tests-import-v2/ops/CPY/unmapped.json`
- Shapes, dtypes, layouts, or config:
  - `bf16 -> bf16`: `13`
  - `f16 -> bf16`: `2`
  - `f32 -> bf16`: `2`
  - `bf16 -> f16`: `2`
  - `bf16 -> f32`: `2`
  - `i32 -> i32`: `2`
  - `i32 -> f32`: `2`
  - `f32 -> i32`: `2`
- Source references:
  - `catalog/v2/router.json`
  - `catalog/v2/copy/`
- Validation status: confirmed route-family gap
- Next concrete action: choose whether the next narrow slice is `bf16` COPY or integer/scalar cast COPY, because both are kernel-simple compared to quantized CPY.

### Missing Quantized Self-Copy Families

- Operation or feature: CPY for same-type packed or quantized formats
- Expected behavior: same-format copies should have explicit raw-layout-preserving COPY families if those tensors are intended to run through v2
- Actual behavior: `189` cases are `no_dtype_mapping`
- Evidence:
  - `build/tests/kernels/artifacts/llama-cpp-tests-import-v2/ops/CPY/import-summary.md`
  - `build/tests/kernels/artifacts/llama-cpp-tests-import-v2/ops/CPY/unmapped.json`
- Shapes, dtypes, layouts, or config:
  - `9` cases each for:
    - `iq1_m`, `iq1_s`, `iq2_s`, `iq2_xs`, `iq2_xxs`
    - `iq3_s`, `iq3_xxs`, `iq4_nl`, `iq4_xs`
    - `mxfp4`, `nvfp4`
    - `q2_K`, `q3_K`, `q4_0`, `q4_1`, `q4_K`, `q5_0`, `q5_1`, `q5_K`, `q6_K`, `q8_0`
- Source references:
  - `build/tests/kernels/artifacts/llama-cpp-tests-import-v2/ops/CPY/unmapped.json`
- Validation status: confirmed dtype-family gap; shape requirements behind these rows are still partially hidden by the dtype failure
- Next concrete action: decide whether packed self-copy should be modeled as raw byte copies first. After a dtype route exists, rerun import triage because some of these rows will likely still split into permutation/transpose sub-gaps.

### Missing Quantized-To-f32 Dequantizing Copy Families

- Operation or feature: CPY from packed or integer source formats into `f32`
- Expected behavior: formats that support dequantizing copy should have explicit v2 families rather than falling through as unmapped
- Actual behavior: `42` cases are `no_dtype_mapping`
- Evidence:
  - `build/tests/kernels/artifacts/llama-cpp-tests-import-v2/ops/CPY/import-summary.md`
  - `build/tests/kernels/artifacts/llama-cpp-tests-import-v2/ops/CPY/unmapped.json`
- Shapes, dtypes, layouts, or config:
  - `2` cases each for:
    - `i32 -> f32`
    - `iq1_m`, `iq1_s`, `iq2_s`, `iq2_xs`, `iq2_xxs`
    - `iq3_s`, `iq3_xxs`, `iq4_nl`, `iq4_xs`
    - `mxfp4`, `nvfp4`
    - `q2_K`, `q3_K`, `q4_0`, `q4_1`, `q4_K`, `q5_0`, `q5_1`, `q5_K`, `q6_K`, `q8_0` into `f32`
- Source references:
  - `build/tests/kernels/artifacts/llama-cpp-tests-import-v2/ops/CPY/unmapped.json`
- Validation status: confirmed dtype-family gap
- Next concrete action: split integer cast COPY from real dequantizing COPY. `i32 -> f32` is a scalar cast; the quantized families need dequantization semantics and should not be grouped with raw self-copy work.

### Missing Scalar-To-Quantized Quantizing Copy Families

- Operation or feature: CPY from `bf16`, `f16`, or `f32` into packed quantized destinations
- Expected behavior: if quantizing copy is intended to be supported in v2, it needs explicit route/kernel families and validation
- Actual behavior: `126` cases are `no_dtype_mapping`
- Evidence:
  - `build/tests/kernels/artifacts/llama-cpp-tests-import-v2/ops/CPY/import-summary.md`
  - `build/tests/kernels/artifacts/llama-cpp-tests-import-v2/ops/CPY/unmapped.json`
- Shapes, dtypes, layouts, or config:
  - `2` cases each for `bf16`, `f16`, and `f32` into each of:
    - `iq1_m`, `iq1_s`, `iq2_s`, `iq2_xs`, `iq2_xxs`
    - `iq3_s`, `iq3_xxs`, `iq4_nl`, `iq4_xs`
    - `mxfp4`, `nvfp4`
    - `q2_K`, `q3_K`, `q4_0`, `q4_1`, `q4_K`, `q5_0`, `q5_1`, `q5_K`, `q6_K`, `q8_0`
- Source references:
  - `build/tests/kernels/artifacts/llama-cpp-tests-import-v2/ops/CPY/unmapped.json`
- Validation status: confirmed dtype-family gap
- Next concrete action: treat this as a separate workstream from self-copy and dequantizing copy. Quantizing COPY likely needs format-specific math rather than a widened contiguous byte-copy kernel.

## Intentional Negative Coverage

- None identified in the current CPY grouped-YAML slice. All `410` unmapped rows appear to be real absent v2 support rather than explicit negative-test expectations.

## Supplemental Probes

- Probe: `python` classification of `unmapped.json` reason counts and dtype buckets
- Why it was needed: the summary markdown showed totals, but the actionable split between scalar, quantized, permuted, and transposed CPY gaps required raw-row classification
- Result: the missing support separates cleanly into `384` dtype-family gaps and `26` layout/stride gaps

- Probe: `python` classification of `imported-workload.json`
- Why it was needed: determine exactly which grouped-YAML cases the current four COPY families already cover
- Result:
  - `copy_f16_f16`: `4` cases
  - `copy_f32_f32`: `4` cases
  - `copy_f16_f32`: `1` case
  - `copy_f32_f16`: `1` case

- Probe: `ctest --test-dir build --output-on-failure -R kernel-run-llama-cpp-tests-v2-CPY-generated`
- Why it was needed: verify whether the current CPY problem was runtime correctness or only missing import coverage
- Result: all mapped v2 CPY runtime cases pass

## Repository Changes

- Files changed in repository:
  - `docs/cpy-correctness-workstream.md`
- Files changed outside repository:
  - none

## External Artifacts

- `build/tests/kernels/artifacts/llama-cpp-tests-import-v2/ops/CPY/import-summary.md`
- `build/tests/kernels/artifacts/llama-cpp-tests-import-v2/ops/CPY/unmapped.json`
- `build/tests/kernels/artifacts/llama-cpp-tests-import-v2/ops/CPY/imported-workload.json`

## Next Iteration

- Highest-priority next fix: decide whether the next CPY slice should widen layout support for the existing four f16/f32 families or add one new dtype family such as `bf16`
- Fastest validation command after that fix: `ctest --test-dir build --output-on-failure -R kernel-run-llama-cpp-tests-v2-CPY-generated`
- Remaining unknowns:
  - whether permuted and transposed CPY should be handled by a generic strided copy kernel or intentionally left unsupported
  - whether quantized CPY should be split into raw self-copy, dequantizing copy, and quantizing copy workstreams with separate kernels and validation
