# HRX2 Operation Support Tracking

This document tracks operation support gaps as they are reached during grouped-YAML
import and runtime validation work.

## How To Use This Document

- Add new entries incrementally when an operation support gap is confirmed.
- Prefer linking each entry to the generated import artifacts or runtime test
  evidence that exposed it.
- Keep unsupported cases visible until the importer, routing layer, or kernel
  surface is widened and validated.

## Current Tracked Gaps

### Unary Pointwise Ops

#### View-Backed Cases (`v != 0`)

- Status: partially supported
- Scope: grouped-YAML unary import lowering for contiguous v2 pointwise routes
- Correctness status: contiguous unary cases for `ABS`, `EXP`, `NEG`, and `RELU`
  now map to v2 routes; the remaining negatives are import-only view-backed
  cases and do not currently reach kernel execution
- Current behavior: contiguous unary lowering only accepts `v=0`, so grouped
  YAML view-backed cases remain unmapped as `shape_lowering_not_implemented`
- Evidence:
  - `build/tests/kernels/artifacts/llama-cpp-tests-import-v2/ops/ABS/import-summary.md`
  - `build/tests/kernels/artifacts/llama-cpp-tests-import-v2/ops/EXP/import-summary.md`
  - `build/tests/kernels/artifacts/llama-cpp-tests-import-v2/ops/NEG/import-summary.md`
  - `build/tests/kernels/artifacts/llama-cpp-tests-import-v2/ops/RELU/import-summary.md`
- Current counts:
  - `ABS`: `4` mapped, `4` `shape_lowering_not_implemented`
  - `EXP`: `4` mapped, `4` `shape_lowering_not_implemented`
  - `NEG`: `4` mapped, `4` `shape_lowering_not_implemented`
  - `RELU`: `4` mapped, `4` `shape_lowering_not_implemented`
- Why they fail:
  - `lower_contiguous_unary_tensors()` rejects view-backed cases before route
    matching with `contiguous unary routing requires contiguous input (v=0)`
- Source references:
  - `src/ggml_hrx_kernel_bench/routing/v2/import_resolution.py`
- Next action: decide whether unary `v=1` should lower into the existing
  contiguous route contract through explicit tensor stride materialization or
  whether it requires a distinct non-contiguous unary route family

### ADD

#### Fused Add Cases (`nf != 1`)

- Status: unsupported
- Scope: grouped-YAML ADD import lowering
- Correctness status: mapped v2 ADD runtime cases pass; the remaining failing ADD
  cases are import-only negatives and do not currently reach kernel execution
- Current behavior: v2 ADD import lowering only supports `nf=1`, so `nf != 1`
  cases remain unmapped as `shape_lowering_not_implemented`
- Interpretation: these cases are currently treated as fused add operations
- Evidence:
  - `build/tests/kernels/artifacts/llama-cpp-tests-import-v2/ops/ADD/import-summary.md`
  - `build/tests/kernels/artifacts/llama-cpp-tests-import-v2/ops/ADD/unmapped.json`
  - `ctest --test-dir build --output-on-failure -R kernel-run-llama-cpp-tests-v2-ADD-generated`
- Known affected source case indices: `35, 36, 37, 38, 46, 47, 48, 49`
- Affected grouped-YAML shapes:
  - `ne=[10,5,4,3], nf=2, nr=[2,1,1,1]`
  - `ne=[10,5,4,3], nf=4, nr=[1,1,2,1]`
  - `ne=[10,5,4,3], nf=6, nr=[1,1,2,2]`
  - `ne=[10,5,4,3], nf=7, nr=[1,2,2,2]`
  - `ne=[16,5,4,3], nf=16, nr=[1,1,1,1]`
  - `ne=[16,5,4,3], nf=3, nr=[1,2,1,1]`
  - `ne=[16,5,4,3], nf=5, nr=[1,1,1,2]`
  - `ne=[16,5,4,3], nf=8, nr=[2,2,2,2]`
- Why they fail:
  - `lower_contiguous_pointwise_shape()` rejects every `nf != 1` case before
    route matching with `contiguous pointwise routing requires nf=1`
  - `lower_generic_pointwise_tensors()` also rejects every `nf != 1` case
    before tensor descriptors are materialized, so the generic 4D route never
    gets a chance to match them either
  - The current v2 ADD route and kernel contracts only model two inputs
    (`src0`, `src1`) plus broadcast and striding metadata; they do not encode
    any additional operand grouping or fusion information associated with `nf`
- Source references:
  - `src/ggml_hrx_kernel_bench/routing/v2/import_resolution.py`
  - `src/ggml_hrx_kernel_bench/import_mapping_registry.py`
- Next action: decide whether `nf` should be lowered into the existing generic
  4D ADD route semantics or whether these grouped-YAML cases require a distinct
  fused-ADD route contract with more than the current two-input tensor surface

### DIV

#### Fused Div Case (`nf != 1`)

- Status: unsupported
- Scope: grouped-YAML DIV import lowering
- Correctness status: mapped v2 DIV runtime cases pass; the remaining failing DIV
  case is an import-only negative and does not currently reach kernel execution
- Current behavior: v2 DIV import lowering only supports `nf=1`, so the grouped
  YAML `nf=16` case remains unmapped as `shape_lowering_not_implemented`
- Evidence:
  - `build/tests/kernels/artifacts/llama-cpp-tests-import-v2/ops/DIV/import-summary.md`
  - `build/tests/kernels/artifacts/llama-cpp-tests-import-v2/ops/DIV/unmapped.json`
  - `ctest --test-dir build --output-on-failure -R kernel-run-llama-cpp-tests-v2-DIV-generated`
- Known affected source case index: `42`
- Affected grouped-YAML shape:
  - `ne=[16,5,4,3], nf=16, nr=[1,1,1,1]`
- Why it fails:
  - `lower_contiguous_pointwise_shape()` rejects every `nf != 1` case before
    route matching with `contiguous pointwise routing requires nf=1`
  - `lower_generic_pointwise_tensors()` also rejects every `nf != 1` case
    before tensor descriptors are materialized
- Source references:
  - `src/ggml_hrx_kernel_bench/routing/v2/import_resolution.py`

### CPY

#### Current Support Gaps

- Status: partially supported
- Scope: grouped-YAML CPY import lowering for the v2 catalogue
- Correctness status: the validated v2 CPY runtime slice covers contiguous
  copies for `bf16`, `f16`, and `f32` across all current float/bfloat source
  and destination pairs, plus `_src_transpose=1` for `f32 -> f32` via the
  non-contiguous 4D route; the remaining CPY negatives are import-only and do
  not currently reach kernel execution
- Current behavior: v2 CPY lowering accepts identity-permutation contiguous
  copies for the 9 `bf16`/`f16`/`f32` source-destination pairs, and accepts
  `_src_transpose=1` with identity source and destination permutations for
  `f32 -> f32`
- Evidence:
  - `build/tests/kernels/artifacts/llama-cpp-tests-import-v2/ops/CPY/import-summary.md`
  - `build/tests/kernels/artifacts/llama-cpp-tests-import-v2/ops/CPY/unmapped.json`
  - `build/tests/kernels/artifacts/llama-cpp-tests-import-v2/ops/CPY/generated-kernel-tests.json`
  - `ctest --test-dir build --output-on-failure -R kernel-run-llama-cpp-tests-v2-CPY-generated`
- Current counts:
  - `21` mapped cases
  - `363` `no_dtype_mapping` cases
  - `32` `shape_lowering_not_implemented` cases with
    `copy lowering requires permute_src=[0, 0, 0, 0]`
  - `4` `shape_lowering_not_implemented` cases with
    `copy lowering requires _src_transpose=0`
- Why the remaining supported-dtype cases fail:
  - grouped-YAML cases with non-identity source permutations such as
    `permute_src=[0,2,1,3]` and `permute_src=[1,0,2,3]` are rejected before
    route matching
  - `_src_transpose=1` remains unsupported for `bf16 -> bf16` and
    `f16 -> f16`; only `f32 -> f32` is currently lowered to the non-contiguous
    4D route
  - non-float and quantized dtype combinations remain entirely unmapped as
    `no_dtype_mapping`
- Source references:
  - `src/ggml_hrx_kernel_bench/routing/v2/import_resolution.py`
  - `src/ggml_hrx_kernel_bench/generators/copy.py`
  - `src/ggml_hrx_kernel_bench/generators/copy_contiguous.py`
  - `src/ggml_hrx_kernel_bench/generators/copy_non_contiguous.py`
- Loom support notes and prioritization impact:
  - Loom already supports the required `view.load`, `view.store`,
    `scalar.extf`, and `scalar.fptrunc` operations used by the existing COPY
    kernels, and the generated non-contiguous 4D kernel already exists for the
    current `bf16`/`f16`/`f32` dtype matrix
  - That means the highest-yield remaining gaps are importer and route-surface
    gaps, not new Loom-kernel capability gaps
- First validation target:
  - add one narrow source-strided COPY slice for `f32 -> f32` with
    `_src_transpose=0`, `permute_src=[0,2,1,3]`, and
    `permute_dst=[0,0,0,0]`
  - expected route: a v2 non-contiguous COPY route backed by
    `copy/copy_f32_f32_non_contiguous_4d.loom`
  - expected coverage gain: `4` grouped-YAML cases in the existing `f32 -> f32`
    family
  - validation:
    - `PYTHONPATH=src pytest tests/test_copy_contiguous_codegen.py tests/test_oracles.py tests/test_routing_api.py -k copy`
    - `cmake --build build --target kernel-llama-cpp-tests-import-coverage-v2`
    - `ctest --test-dir build --output-on-failure -R kernel-run-llama-cpp-tests-v2-CPY-generated`
- Prioritized workstreams:
  - `1. Source-strided current float/bfloat copies`
    - widen the non-contiguous 4D route and lowering path to accept
      source-strided cases where `permute_dst` stays identity
    - prioritize `permute_src=[0,2,1,3]` first because it is the largest
      remaining supported-dtype family at `18` cases and is directly adjacent
      to the existing non-contiguous route contract
    - once the first `f32 -> f32` slice is validated, widen the same route
      family to the remaining current dtype matrix (`bf16`, `f16`, `f32`)
  - `2. Same-dtype transpose expansion`
    - widen `_src_transpose=1` beyond `f32 -> f32` to the same-type `bf16` and
      `f16` slices
    - expected gain: `4` additional grouped-YAML cases
  - `3. Additional source permutation families`
    - add the smaller source-only permutation families
      `permute_src=[1,0,2,3]` (`4` cases) and
      `permute_src=[1,2,0,3]` (`1` case)
    - treat these as follow-ons after the `permute_src=[0,2,1,3]` path proves
      the generalized stride-based lowering
  - `4. Source-and-destination permutation pairs`
    - address cases like `permute_src=[0,3,1,2]` with
      `permute_dst=[0,2,1,3]` (`9` cases)
    - defer until the route contract can represent whether destination
      permutation should be lowered as a shape transform or requires a distinct
      non-contiguous destination path
  - `5. Non-float and quantized dtype expansion`
    - the remaining `363` failures are `no_dtype_mapping` cases, not layout
      cases
    - defer these until there is a clear plan for encode/decode behavior and
      for avoiding quadratic hand-written kernels across quantized layouts

### MUL

#### Fused Mul Case (`nf != 1`)

- Status: unsupported
- Scope: grouped-YAML MUL import lowering
- Correctness status: mapped v2 MUL runtime cases pass; the remaining failing MUL
  case is an import-only negative and does not currently reach kernel execution
- Current behavior: v2 MUL import lowering only supports `nf=1`, so the grouped
  YAML `nf=16` case remains unmapped as `shape_lowering_not_implemented`
- Evidence:
  - `build/tests/kernels/artifacts/llama-cpp-tests-import-v2/ops/MUL/import-summary.md`
  - `build/tests/kernels/artifacts/llama-cpp-tests-import-v2/ops/MUL/unmapped.json`
  - `ctest --test-dir build --output-on-failure -R kernel-run-llama-cpp-tests-v2-MUL-generated`
- Known affected source case index: `42`
- Affected grouped-YAML shape:
  - `ne=[16,5,4,3], nf=16, nr=[1,1,1,1]`
- Why it fails:
  - `lower_contiguous_pointwise_shape()` rejects every `nf != 1` case before
    route matching with `contiguous pointwise routing requires nf=1`
  - `lower_generic_pointwise_tensors()` also rejects every `nf != 1` case
    before tensor descriptors are materialized
- Source references:
  - `src/ggml_hrx_kernel_bench/routing/v2/import_resolution.py`

### SUB

#### Fused Sub Case (`nf != 1`)

- Status: unsupported
- Scope: grouped-YAML SUB import lowering
- Correctness status: mapped v2 SUB runtime cases pass; the remaining failing SUB
  case is an import-only negative and does not currently reach kernel execution
- Current behavior: v2 SUB import lowering only supports `nf=1`, so the grouped
  YAML `nf=16` case remains unmapped as `shape_lowering_not_implemented`
- Evidence:
  - `build/tests/kernels/artifacts/llama-cpp-tests-import-v2/ops/SUB/import-summary.md`
  - `build/tests/kernels/artifacts/llama-cpp-tests-import-v2/ops/SUB/unmapped.json`
  - `ctest --test-dir build --output-on-failure -R kernel-run-llama-cpp-tests-v2-SUB-generated`
- Known affected source case index: `42`
- Affected grouped-YAML shape:
  - `ne=[16,5,4,3], nf=16, nr=[1,1,1,1]`
- Why it fails:
  - `lower_contiguous_pointwise_shape()` rejects every `nf != 1` case before
    route matching with `contiguous pointwise routing requires nf=1`
  - `lower_generic_pointwise_tensors()` also rejects every `nf != 1` case
    before tensor descriptors are materialized
- Source references:
  - `src/ggml_hrx_kernel_bench/routing/v2/import_resolution.py`
