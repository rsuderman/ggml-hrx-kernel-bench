# V2 Operation Coverage Plan

This document tracks the remaining grouped-YAML descriptor route-import work.
Use it as the task list for returning to coverage parity without reintroducing
the deleted custom importer.

## Current Baseline

Coverage status is versioned in:

- `tests/kernels/data/llamacpp.import-coverage.json`
- `tests/models/data/llama-8b-q8.import-coverage.json`

Generated drilldown artifacts come from the build-time route-import targets:

- `cmake --build build --target kernel-llama-cpp-yaml-route-import-v2`
- `cmake --build build --target kernel-model-llama-3-3-8b-q8-0-yaml-route-import-v2`
- `cmake --build build --target kernel-yaml-route-import-v2`

Start each investigation from the generated `import-coverage.json`, then inspect
the per-op `route-matches.json`, `route-unmatched.json`, and generated runtime
test manifests under the target's artifact directory.

## Active Task List

- [x] Retire the legacy grouped YAML importer and its v1/v2 coverage fixtures.
- [x] Retire the legacy v1/hrx2 routing catalog and kernel tree.
- [x] Keep llama.cpp and model YAML route-import coverage validated at build
  time.
- [x] Keep generated runtime CTest suites materialized from descriptor
  route-import artifacts.
- [ ] Refresh coverage from the current YAML artifacts after the importer
  cleanup lands.
- [ ] Re-triage unmatched llama.cpp cases from current route-import artifacts.
- [ ] Re-triage unmatched model cases from current route-import artifacts.
- [ ] Pick the next narrow op/dtype/layout slice from the refreshed unmatched
  set.
- [ ] Validate any widened executing kernel surface with targeted generated
  runtime tests.

## Triage Requirements

For each op slice, record:

- total YAML cases
- matched and unmatched counts
- dtype and layout families present
- representative unmatched samples
- whether the gap is importer descriptor work, route/catalog work, kernel
  surface work, or underspecified op behavior
- validation commands run
- issues encountered
- next step

Do not hide unsupported cases by filtering them out of importer outputs.

## Preferred Fix Order

Prefer fixes in this order:

1. Descriptor materialization in `yaml_route_import.py`.
2. Existing route predicate or value binding adjustments.
3. Catalog descriptor additions that target existing kernels.
4. Narrow kernel-surface additions when no existing route/kernel can express the
   case.

When a new `.loom` kernel is unavoidable, document why the existing route/kernel
could not be widened safely.
