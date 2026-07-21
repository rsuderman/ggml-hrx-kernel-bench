# AGENTS

## Default Support Workflow

For grouped-YAML operation support work in this repository, the default process
is:

- [docs/grouped-yaml-import-plan.md](docs/grouped-yaml-import-plan.md)
- [docs/hrx2-operation-coverage-plan.md](docs/hrx2-operation-coverage-plan.md)

Use that workflow unless the user explicitly asks for a different process.

## Materialized Markdown Outputs

- Do not store newly materialized markdown outputs under this repository's
  `docs/` tree unless the user explicitly asks for an in-repo document.
- Repository `docs/` files remain the source for standing workflow and design
  references. Materialized run outputs belong in the Codex workspace folder
  above.

## Required Starting Point

- Start from `build/tests/kernels/artifacts/grouped-yaml-import/import-coverage.json`.
- Use per-op drilldown artifacts under
  `build/tests/kernels/artifacts/grouped-yaml-import/ops/<OP>/`.
- Keep the first validation slice narrow enough to debug quickly: one op
  family, one dtype family, and one layout or mode family when practical.
- Do not stop at that first slice when the requested work is implementation.
  Use the first slice as the proof point, then continue through the remaining
  route, config, importer, and kernel changes that are reasonable for the same
  operation before reporting back.

## Triage And Implementation Policy

- Triage must produce an implementation plan that covers as much of the
  operation's visible unsupported surface as is reasonable in the current
  branch. The plan should distinguish:
  - cases to implement now
  - cases intentionally deferred because semantics, runtime support, or test
    cost are unclear
  - cases blocked by a concrete external dependency
- Implementation work should execute that plan end to end. After each validated
  slice, continue to the next reasonable slice instead of reporting back for
  permission to keep going.
- Report back only when the planned reasonable coverage has been implemented
  and validated, or when a real blocker prevents further progress. A blocker
  must name the exact missing semantic decision, tool support, device access, or
  failing validation.
- When launching a subagent for implementation, give it ownership of the full
  reasonable implementation plan for the operation, not only the first proof
  slice. The subagent should complete all non-blocked slices before returning.
- Implementation subagents must notify the spawning agent when they complete,
  including the worktree path, final coverage delta, validation results, and
  any blocked or intentionally deferred cases.

## Implementation Architecture Policy

Implementation plans must include cleanup and organization as first-class work.
Do not treat architecture as a follow-up after functionality appears to work.

Hard requirements:

- Before implementing a non-trivial feature or tool, identify the subsystem
  boundary it belongs to and name the files or modules that should own each
  responsibility.
- Do not dump new behavior into a large existing file simply because useful
  helpers are already there. Reuse helpers through explicit module boundaries,
  or move those helpers into a focused shared module first.
- Keep CLI entry points thin. Argument parsing, user-facing command dispatch,
  and error presentation may live in CLI files; discovery, materialization,
  generation, execution, aggregation, reporting, and persistence should live in
  focused implementation modules.
- If a change adds multiple responsibilities, split them into coherent modules
  in the same implementation pass unless there is a concrete blocker. Document
  the blocker if the split cannot be completed.
- When extending an existing subsystem, remove or simplify obsolete paths made
  redundant by the new design. Do not leave compatibility shims, duplicate
  runners, dead helper functions, or parallel command paths unless the user has
  explicitly asked for a migration window.
- Tests should reinforce the intended boundaries. Prefer unit tests for focused
  modules and a smaller number of CLI integration tests over tests that require
  one monolithic command module to own all behavior.
- A final implementation report must call out the resulting module ownership
  and any remaining intentionally deferred cleanup. Missing cleanup is a defect,
  not a polish item.

## Kernel Coverage Expansion Goals

- Coverage expansion is for basic functional correctness first, not
  performance. Do not hand-optimize kernels while the operation surface is still
  mostly unsupported.
- Minimize the number of new `.loom` kernels needed for correctness. Later
  performance work is expected to add specialized variants, so coverage work
  should avoid creating that variant explosion early.
- Before adding kernels, analyze which missing cases can overlap through shared
  route attributes, constraints, tensor layouts, and kernel configuration.
- Prefer tiled implementations for new functional kernels. Tiling should make
  shape and layout coverage explicit and easier to validate, not serve as a
  performance-tuning exercise.
- Loom kernels should primarily encode semantic special cases, required
  compile-time values, ABI differences, and configuration constraints. Avoid
  hand-tuned schedules or micro-optimizations in initial coverage kernels.
- Generalized functional kernels are fallback coverage paths. In `router.json`,
  list them after more specific or optimized routes for the same operation so
  they are selected only when preferred routes do not match.
- For a small number of dtype variants, direct replication of an established
  pattern is acceptable. If the dtype/layout/mode matrix becomes large, prefer
  a generator or shared template mechanism over hand-maintaining many similar
  kernels and routes.

## Validation Policy

- Run targeted import materialization before broader coverage gates.
- Run `ctest` outside the sandbox. If sandbox restrictions block validation,
  record that `ctest` requires unsandboxed execution instead of treating the
  sandboxed run as sufficient.
- Validate emitted generated configs structurally before updating expected
  coverage.
- If a change adds or widens the executing kernel surface, runtime kernel
  validation is required unless the environment is blocked. Record the exact
  blocker when runtime execution cannot be run.
- Do not update
  `tests/kernels/data/llamacpp_test.import-coverage.json` until the intended
  slice is validated and the build-time coverage target passes.
- When expected coverage fixtures are updated, preserve the existing canonical
  JSON key order. Do not copy `sort_keys=True` output or otherwise reorder
  fixture objects; only the coverage counts for validated cases should change.

## Coverage Policy

- Keep unsupported or ambiguous cases visible in importer outputs instead of
  dropping them.
- Prefer extending importer lowering, family specs, route predicates, or kernel
  configuration before creating a new `.loom` kernel.
- When route support can be widened through attributes and constraints, bind
  those values by name and validate them through the constraint system instead
  of hardcoding a shape-specific route.
- When a new `.loom` kernel is unavoidable, document why an existing kernel or
  route could not be widened to cover the case.
- When a routed kernel variant is added or ported, keep it in its own `.loom`
  file. Do not leave multiple independently routed kernel variants combined in a
  shared source file.

## Commit Policy

- Commit messages must be thorough.
- Use a concise subject line that names the main change.
- Add a commit body that explains:
  - what changed
  - why the change was made
  - the most important validation that was run
  - any important limitations, blockers, or intentionally excluded follow-up work
