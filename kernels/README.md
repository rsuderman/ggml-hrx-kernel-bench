# Kernels

Place Loom kernel sources owned by this workbench here.

Active routed kernels live under `kernels/v2`. The legacy `kernels/hrx2`
corpus has been removed with the v1 routing system.

## Naming

Use succinct kernel and route names. Names should identify the operation,
required element types, and the routing-visible specialization.

- Use `generic` for fallback kernels.
- Use `contiguous` for contiguous tensor specializations.
- Do not include default operation behavior such as `scalar` or `batched` in
  names when the operation already requires it for normal coverage.

For example, prefer `mul_mat_f16_f16_generic` and
`mul_mat_f16_f16_contiguous` over names that include default operation
behavior.
