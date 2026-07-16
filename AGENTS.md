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
