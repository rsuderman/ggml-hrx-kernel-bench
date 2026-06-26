# Grouped YAML Import Plan

This document turns the grouped YAML importer sketch into concrete repo
components for future implementation.

## Goal

Import grouped llama.cpp workload YAML such as:

`results/test-backend-ops-materialization/.../build-hrx.grouped.short.yaml`

into the current benchmark system while keeping unsupported or unmapped cases
visible as explicit backlog items instead of silently dropping them.

## Required Components

### 1. YAML Loader

Purpose:
- read grouped YAML under top-level `ops`
- preserve op group order and case order
- produce raw imported records before any kernel assumptions

Suggested file:
- `tests/infra/import_grouped_yaml.py`

Expected output:
- `ImportedSuite`
- `ImportedOpGroup`
- `ImportedCase`

### 2. Case Normalizer

Purpose:
- convert op-specific keys like `ne_a`, `ne_b`, `nr`, `perm1`, `stride_dim`
  into a stable normalized parameter object
- preserve original raw case data for debugging and import reports

Suggested file:
- `tests/infra/import_shape_lowering.py`

Expected behavior:
- op-specific normalizers
- no route selection yet
- no kernel family assumption yet

### 3. Mapping Registry

Purpose:
- map `(op, dtype, normalized constraints)` to a current kernel family
- remain declarative and searchable
- allow ambiguous matches to be reported instead of guessed

Suggested file:
- `src/ggml_hrx_kernel_bench/import_mapping_registry.py`

Bootstrap policy:
- only a small number of rules should exist at first
- everything else should become `UnmappedCase`

### 4. Shape Lowering

Purpose:
- convert normalized imported params into benchmark config params and values
- example:
  - llama tensor extents -> `nrows`, `ncols`
  - or matmul extents -> `k`, `rows`, `cols`

Suggested file:
- `tests/infra/import_shape_lowering.py`

This layer is kernel-family specific and will be the main growth area.

### 5. Route Resolver

Purpose:
- resolve a mapped kernel family to a concrete route
- return:
  - mapped route
  - no route
  - ambiguous route
  - mapped but not runnable

Suggested file:
- `tests/infra/import_route_resolution.py`

Important:
- ambiguous route resolution must produce `UnmappedCase` rows with reason
  `ambiguous_route_match`

### 6. Config Emitter

Purpose:
- emit compact benchmark configs using current format:
  - `kernel`
  - `params`
  - `cases`

Suggested file:
- `tests/infra/import_emit_configs.py`

Output location:
- `generated-import-configs/<kernel>.json`

### 7. Benchmark Driver

Purpose:
- run emitted configs through the existing benchmark service
- write:
  - heavy outputs to `benchmark-artifacts/`
  - performance summaries to `benchmark-results/`

Existing files reused:
- `tests/infra/benchmark_kernel_test_config.py`

### 8. Import Report Generator

Purpose:
- summarize the importer outcome even when coverage is incomplete
- make unmapped tests highly visible

Suggested files:
- `tests/infra/import_report.py`
- `schemas/imported-workload.schema.json`
- `schemas/import-unmapped-cases.schema.json`

Suggested output files:
- `imported-workload.json`
- `unmapped.json`
- `import-summary.md`

## Unmapped Cases Are Required Output

This is non-negotiable: if an imported test has no current kernel mapping, it
must still appear in output.

An imported case must end in exactly one bucket:

1. `mapped`
2. `unmapped`
3. `ambiguous`

### Unmapped Reasons

The initial enums are:

- `no_kernel_family_mapping`
- `no_dtype_mapping`
- `shape_lowering_not_implemented`
- `no_route_match`
- `ambiguous_route_match`

## Concrete Model Files

Implemented skeletons:

- `src/ggml_hrx_kernel_bench/import_models.py`
- `src/ggml_hrx_kernel_bench/import_mapping_registry.py`
- `schemas/imported-workload.schema.json`
- `schemas/import-unmapped-cases.schema.json`

These files define:
- normalized imported cases
- resolved benchmark cases
- unmapped case records
- mapping status and reason enums

## First Implementation Cut

The first working importer should intentionally be narrow:

- parse YAML
- normalize all cases
- map only a tiny bootstrap subset
- emit configs for mapped cases
- emit unmapped backlog for everything else

This is better than pretending broad coverage exists.

## Recommended First Milestone

1. Implement loader for grouped YAML.
2. Normalize all imported cases into `ImportedSuite`.
3. Emit `unmapped.json` for every case by default.
4. Add one or two real mapping rules.
5. Emit benchmark configs only for those mapped cases.
6. Run existing benchmark service on emitted configs.

That gives:
- immediate visibility into unsupported workloads
- a stable schema for backlog tracking
- a safe path to expand coverage incrementally
