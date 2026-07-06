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

#### Non-Identity Permutation And Transposed Copy Cases

- Status: partially supported
- Scope: grouped-YAML CPY import lowering
- Correctness status: the validated v2 CPY runtime slice covers contiguous
  copies for `f16 -> f16`, `f32 -> f32`, `f16 -> f32`, and `f32 -> f16`, plus
  the `_src_transpose=1` `f32 -> f32` cases via a separate non-contiguous 4D
  kernel; the remaining CPY negatives are import-only and do not currently
  reach kernel execution
- Current behavior: v2 CPY lowering accepts identity-permutation contiguous
  copies for those four dtype pairs, and also accepts `_src_transpose=1` with
  identity source and destination permutations for `f32 -> f32`
- Evidence:
  - `build/tests/kernels/artifacts/llama-cpp-tests-import-v2/ops/CPY/import-summary.md`
  - `build/tests/kernels/artifacts/llama-cpp-tests-import-v2/ops/CPY/unmapped.json`
  - `ctest --test-dir build --output-on-failure -R kernel-run-llama-cpp-tests-v2-CPY-generated`
- Current counts:
  - `13` mapped cases
  - `23` `shape_lowering_not_implemented` cases for the supported f16/f32 dtype pairs
  - `384` `no_dtype_mapping` cases for other source/destination dtype combinations
- Why the remaining supported-dtype cases fail:
  - grouped-YAML permutations such as `permute_src=[0,2,1,3]`,
    `permute_src=[1,0,2,3]`, and destination permutations such as
    `permute_dst=[0,2,1,3]` are rejected before route matching
  - `_src_transpose=1` remains unsupported for the other dtype pairs because
    only the `f32 -> f32` transpose slice is lowered to the non-contiguous 4D
    route today
- Source references:
  - `src/ggml_hrx_kernel_bench/routing/v2/import_resolution.py`

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
