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
- Current behavior: v2 ADD import lowering only supports `nf=1`, so `nf != 1`
  cases remain unmapped as `shape_lowering_not_implemented`
- Interpretation: these cases are currently treated as fused add operations
- Evidence:
  - `build/tests/kernels/artifacts/llama-cpp-tests-import-v2/ops/ADD/import-summary.md`
  - `build/tests/kernels/artifacts/llama-cpp-tests-import-v2/ops/ADD/unmapped.json`
- Known affected source case indices: `35, 36, 37, 38, 46, 47, 48, 49`
- Source references:
  - `src/ggml_hrx_kernel_bench/routing/v2/import_resolution.py`
  - `src/ggml_hrx_kernel_bench/import_mapping_registry.py`
- Next action: decide whether fused ADD cases should lower through the existing
  generic 4D route or require a separate route contract
