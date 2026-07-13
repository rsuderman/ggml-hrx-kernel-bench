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

The active generated artifact roots are currently:

- `build/tests/kernels/artifacts/llama-cpp-yaml-route-import-v2/`
- `build/tests/models/artifacts/Llama-3.3-8B-Instruct.Q8_0-route-import-v2/`

As of 2026-07-13, the generated llama.cpp route import reported 805 matched
cases and 10,392 unmatched cases across 115 operations. The generated model
route import reported 24 matched cases and 8 unmatched cases across 10
operations, with the remaining unmatched model surface in `FLASH_ATTN_EXT`,
`ROPE`, and `SWIGLU`.

## Active Task List

- [x] Retire the legacy grouped YAML importer and its v1/v2 coverage fixtures.
- [x] Retire the legacy v1/hrx2 routing catalog and kernel tree.
- [x] Keep llama.cpp and model YAML route-import coverage validated at build
  time.
- [x] Keep generated runtime CTest suites materialized from descriptor
  route-import artifacts.
- [x] Enable descriptor execution through `ggml-hrx-run-loom-simple` as part of
  the default generated harness path.
- [x] Preserve descriptor `close` tolerances when bridging descriptor execution
  to `iree-run-loom` expected-buffer checks.
- [ ] Land or otherwise carry the paired HRX `iree-run-loom` support for
  `--expected-kernel-buffer-tolerance` with the bench harness changes.
- [ ] Add HRX-side unit coverage for tolerant HAL expected-buffer comparison,
  so approximate descriptor execution is protected below the bench runner.
- [ ] Refresh coverage from the current YAML artifacts after the tolerant
  descriptor-execution changes land.
- [ ] Re-triage unmatched llama.cpp cases from current route-import artifacts.
- [ ] Re-triage unmatched model cases from current route-import artifacts.
- [ ] Pick the next narrow op/dtype/layout slice from the refreshed unmatched
  set.
- [ ] Validate any widened executing kernel surface with targeted generated
  runtime tests.
- [x] Add and validate the model `SET_ROWS` f32-to-f16 slice through descriptor
  route import and generated descriptor execution.
- [ ] Expand descriptor execution coverage to additional f32 approximate
  families now that `close` tolerances are preserved through the runner.
- [ ] Simplify or retire redundant legacy generated-runtime execution paths once
  descriptor execution has enough operation coverage to be the primary runtime
  validation utility.

## Next Candidate Slices

Prefer small model-level gaps before broad llama.cpp gaps when the work is
otherwise equivalent. `SET_ROWS` was validated on 2026-07-13 and is no longer a
model gap.

- `SWIGLU`: two unmatched model cases; compact surface, but may require checking
  activation-specific ABI and route assumptions.
- `ROPE`: four unmatched model cases; only revisit after confirming whether the
  current route predicates intentionally exclude this model mode.
- `FLASH_ATTN_EXT`: two unmatched model cases; keep later unless the goal is to
  tackle attention-specific route and kernel surface.

For llama.cpp coverage, start from the operations with existing partial support
before tackling large unsupported families. Good first candidates are partial
elementwise or movement ops such as `ADD`, `MUL`, `DIV`, `SUB`, `CPY`, `CONT`,
`GET_ROWS`, `RMS_NORM`, `ROPE`, `SET_ROWS`, `SOFT_MAX`, and `SWIGLU`.

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
Any test that requires HSA resources must be run outside of the normal sandboxed
harness path. If runtime validation cannot run, record the exact blocker instead
of treating a sandboxed or prepare-only result as sufficient.

## Preferred Fix Order

Prefer fixes in this order:

1. Descriptor materialization in `yaml_route_import.py`.
2. Existing route predicate or value binding adjustments.
3. Catalog descriptor additions that target existing kernels.
4. Narrow kernel-surface additions when no existing route/kernel can express the
   case.

When a new `.loom` kernel is unavoidable, document why the existing route/kernel
could not be widened safely.
