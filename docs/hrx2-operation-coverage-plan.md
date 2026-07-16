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

Expected coverage fixtures must keep their canonical JSON key order. Do not
copy sorted generated JSON into `tests/kernels/data/llamacpp.import-coverage.json`
or `tests/models/data/llama-8b-q8.import-coverage.json`; update only the
validated counts while preserving the existing object order.

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
- [x] Keep descriptor generate/execute CTest suites materialized from
  descriptor route-import artifacts, with descriptor prepare handled by the
  build graph.
- [x] Enable descriptor execution through `ggml-hrx-run-loom-simple` as part of
  the default generated harness path.
- [x] Preserve descriptor `close` tolerances when bridging descriptor execution
  to `ggml-hrx-run-loom` expected-buffer checks.
- [ ] Land or otherwise carry the paired HRX `ggml-hrx-run-loom` support for
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
descriptor generate/build-prepare/execute coverage for `ADD`, `CPY`,
`GET_ROWS`, `MUL`, and `RMS_NORM` passed outside the sandbox.

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

Triage is not complete until it identifies the full set of reasonable
implementation slices for the selected operation. The plan should cover all
visible unsupported cases that can be handled with known semantics and current
tool support, even if the first validation checkpoint is deliberately narrow.

The implementation plan must also describe how the missing cases overlap. Note
which cases can share route attributes, constraint checks, tensor layouts,
kernel configuration, and generated code. The purpose is to minimize the number
of functional kernels while still keeping unsupported or ambiguous cases visible
in the route-import artifacts.

Do not hide unsupported cases by filtering them out of importer outputs.
Any test that requires HSA resources must be run outside of the normal sandboxed
harness path. If runtime validation cannot run, record the exact blocker instead
of treating a sandboxed or prepare-only result as sufficient.

## Coverage Expansion Goals

Coverage expansion is a functional-correctness activity. The goal is to make
the broadest reasonable set of YAML cases route, materialize descriptors,
execute, and compare correctly. It is not a performance optimization pass.

Use the minimum number of kernels that can express the validated functional
surface. Later optimization work is expected to add more specialized kernels,
so coverage work should avoid creating a large variant set prematurely. Prefer
routes, constraints, attributes, and kernel configuration when they can express
the variation safely.

When new kernels are required, prefer simple tiled implementations first. Tiled
kernels should make bounds, layout, and compile-time configuration explicit and
easy to validate. Do not spend initial coverage work on hand-tuned schedules,
shape-specific micro-optimizations, or performance-only special cases.

Generalized functional kernels are fallback coverage paths. They should be used
to guarantee correctness coverage for cases that do not match preferred routes,
not to run ahead of more specific or optimized kernels. In `router.json`, place
generalized routes at the lowest preference for the operation, after
specialized tiled, shape-specific, or optimized routes.

Loom kernel sources for coverage should primarily encode:

- semantic ABI differences, such as packed versus split inputs
- special values that affect correctness
- compile-time values that Loom needs as configuration
- constraints that define the safe executing surface

Small dtype replication is acceptable when the variants are few and directly
mirror an established pattern. If the dtype/layout/mode matrix is large or
mechanically repetitive, prefer a generator or template-backed route/kernel
source over hand-written copies.

## Implementation Execution Policy

When the task is to implement coverage, execute the reasonable plan rather than
stopping at the first successfully validated slice.

1. Start with the smallest high-value checkpoint that proves the route,
   descriptor, oracle, or kernel path.
2. Validate that checkpoint with targeted route import and runtime execution
   when required.
3. Continue to adjacent planned slices that reuse the same semantics, route
   shape, or kernel family.
4. Stop only when the reasonable plan is complete or when the remaining cases
   have a concrete blocker, such as unknown op semantics, unsupported Loom
   primitives, unavailable device validation, or excessive test/runtime cost.
5. Report the final coverage delta, validation results, and all intentionally
   deferred cases with reasons.

Subagents launched for implementation should receive the full reasonable plan
for the operation and should complete all non-blocked slices before reporting
back. Do not scope implementation agents to only a fallback route, proof case,
or first kernel unless the user explicitly asks for that limited task. When the
agent completes, it must notify the spawning agent with the worktree path, final
coverage delta, validation results, and any blocked or intentionally deferred
cases.

## Preferred Fix Order

Prefer fixes in this order:

1. Descriptor materialization in `yaml_route_import.py`.
2. Existing route predicate or value binding adjustments.
3. Attribute declarations and constraint-system checks that bind variation by
   name instead of hardcoding shape-specific routes.
4. Catalog descriptor additions that target existing kernels.
5. Kernel configuration changes that generalize an existing functional kernel.
6. Narrow tiled kernel-surface additions when no existing route/kernel can
   express the case.
7. Generators or shared templates when the number of dtype/layout/mode variants
   would otherwise create excessive hand-maintained repetition.

When a new `.loom` kernel is unavoidable, document why the existing route/kernel
could not be widened safely.
