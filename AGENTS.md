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
- Keep the first implementation slice narrow: one op family, one dtype family,
  and one layout or mode family when practical.

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
