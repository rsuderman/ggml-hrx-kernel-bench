# V2 Operation Coverage Plan

This document tracks the remaining grouped-YAML descriptor route-import work.
The current execution objective is to remove the old generated-runtime testing
tool. Coverage expansion is only in scope when it is required to eliminate a
remaining `kernel-run-*` dependency. As of the current retirement pass, the old
generated-runtime tool is no longer registered by CMake; keep future work on
descriptor route-import coverage unless the old path is deliberately
reintroduced for investigation.

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
- [x] Keep descriptor generate/prepare/execute CTest suites materialized from
  descriptor route-import artifacts.
- [x] Enable descriptor execution through `ggml-hrx-run-loom-simple` as part of
  the default generated harness path.
- [x] Preserve descriptor `close` tolerances when bridging descriptor execution
  to `iree-run-loom` expected-buffer checks.
- [ ] Land or otherwise carry the paired HRX `iree-run-loom` support for
  `--expected-kernel-buffer-tolerance` with the bench harness changes.
- [ ] Add HRX-side unit coverage for tolerant HAL expected-buffer comparison,
  so approximate descriptor execution is protected below the bench runner.
- [x] Refresh the harness inventory after each old-tool retirement step and
  drive the legacy runtime registration count to zero.
- [x] Remove no-op old generated-runtime registrations for ops whose
  `generated-kernel-tests.json` manifests contain zero entries.
- [x] For any old generated-runtime registration with non-empty generated
  entries, move the case to descriptor execution or record the exact blocker.
- [x] Validate any widened descriptor execution surface with targeted HSA
  descriptor tests outside the sandboxed harness path.
- [x] Add and validate the model `SET_ROWS` f32-to-f16 slice through descriptor
  route import and generated descriptor execution.
- [x] Remove the old generated-runtime CMake registration path once the
  inventory reports zero legacy runtime registrations.
- [x] Remove or archive the old generated-runtime Python runner after CMake no
  longer references it.

Current retirement inventories:

- `/home/rsuderman/codex/ggml-hrx-kernel-bench-harness-inventory-kernels-old-tool-retirement-20260713.{json,md}`:
  115 ops, 115 descriptor execute registrations, 401 emitted descriptor cases,
  and zero legacy runtime registrations.
- `/home/rsuderman/codex/ggml-hrx-kernel-bench-harness-inventory-model-old-tool-retirement-20260713.{json,md}`:
  10 ops, 9 descriptor execute registrations, 14 emitted descriptor cases, and
  zero legacy runtime registrations.

Validation run on 2026-07-13: `ctest --test-dir build -N -R
'kernel-run-.*yaml-route-import-v2'` reported zero tests, and targeted model
descriptor generate/prepare/execute for `ADD`, `CPY`, `GET_ROWS`, `MUL`, and
`RMS_NORM` passed outside the sandbox.

## Next Candidate Slices

Do not pick a new operation merely to improve coverage while retiring the old
testing tool. First remove stale legacy registrations that have empty generated
manifests. Only pick an op slice when an old generated-runtime registration has
non-empty generated entries and cannot be removed until descriptor execution is
validated for that slice.

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
