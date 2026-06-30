# Catalog-Driven Kernel Import Design

## Objective

The grouped-YAML importer exists to answer one question for each input case:

- does the HRX2 catalog contain a kernel route that can legally execute this case

If the answer is yes, the importer emits a generated kernel test config for that
route. If the answer is no, the importer records the case as unsupported with
the real reason it could not be covered.

The importer should not:

- invent a second routing system
- mirror expected coverage files to decide what to import
- hardcode per-op routing when the catalog already contains the routing data

## Default Support Scope

The default implementation focus should remain narrow:

- pointwise `ADD`
- pointwise `MUL`
- copy `CPY`

This is the preferred first slice because it preserves the currently validated
surface while the importer is redesigned around catalog-driven resolution.

## Coverage Definition

A grouped-YAML case is covered if there exists at least one catalog route for
`case.op` such that all of the following are true:

1. the route supports the case dtype combination
2. the importer can lower the raw grouped-YAML case into the route's normalized
   ABI or shape form
3. the lowered shape satisfies the route's shape domain
4. the lowered shape satisfies the route's guards and layout conditions
5. any required importer-side feature support for that route contract exists

Coverage is existential:

- a case is covered iff some catalog route can execute it
- a case is unsupported iff no catalog route can execute it

## Required Decision Procedure

For each grouped-YAML case:

1. Load all catalog routes whose `op` matches `case.op`.
2. If no routes exist, mark the case `no_kernel_family_mapping`.
3. Filter those routes by dtype support.
4. If no routes survive dtype filtering, mark the case `no_dtype_mapping`.
5. For each remaining route:
   - identify the generic lowering contract for the route
   - try to lower the raw case into the route's normalized shape
   - if lowering fails, keep the failure as evidence for unsupported reporting
   - if lowering succeeds, apply the route's shape domain and guards
6. If one or more lowered routes satisfy the route predicates:
   - pick the highest-priority route
   - emit the generated kernel test config
7. If no lowered route satisfies the predicates:
   - report the real failure class:
     - `shape_lowering_not_implemented` if no route could be lowered
     - `no_route_match` if lowering succeeded but the lowered shape matched no
       route
     - `ambiguous_route_match` if multiple highest-priority routes remain

## Architecture

The design should be split into three concerns.

### 1. Catalog Route Selection

This layer is responsible for:

- loading catalog routes
- grouping by `op`
- filtering by dtype support
- applying shape domains and guards
- resolving priority and ambiguity

This layer should not contain op-specific import behavior.

### 2. Generic Case Lowering

This layer is responsible for:

- inspecting a route's specialization and supported layout
- determining whether the importer understands that route contract
- lowering the raw grouped-YAML case into the normalized route ABI

This layer should be keyed by route contract, not by operation name.

Examples of route contracts:

- pointwise contiguous
- pointwise rhs row broadcast
- pointwise rhs column broadcast
- copy contiguous src to contiguous dst

The lowerers may reject a case because its raw configuration does not satisfy
the contract. That is not custom routing; it is contract validation.

### 3. Runtime Test Materialization

This layer is responsible for:

- emitting generated config JSON
- emitting per-op manifests
- emitting aggregate manifests
- registering runtime tests from emitted manifests

Runtime registration must be based on generated importer outputs, not on the
checked-in expected coverage suite.

## Non-Goals

The importer is not responsible for:

- deciding which operations should exist in the catalog
- asserting that a route is correct at runtime
- hiding unsupported cases to make coverage look better

Unsupported and ambiguous cases must remain visible in importer artifacts.

## What Should Be Generic

The following logic should be shared across operations whenever the underlying
route contract is the same:

- contiguous pointwise checks
- rhs row broadcast checks
- rhs column broadcast checks
- contiguous copy checks
- common dtype filtering
- route priority selection

The only op-specific routing fact should be the catalog route's `op` field.

## Acceptable Explicit Scope Limits

It is acceptable to have a small explicit list of route contracts that the
importer currently understands.

It is also acceptable to have a small whitelist of operations currently
supported by import materialization and runtime registration if that is
strictly a temporary surface-management tool.

That whitelist should preserve the current validated coverage surface:

- `ADD`
- `MUL`
- `CPY`

What is not acceptable is reproducing catalog routing by writing separate
operation-to-kernel routing code in the importer.

## Failure Taxonomy

The importer should preserve real failure classes:

- `no_kernel_family_mapping`
- `no_dtype_mapping`
- `shape_lowering_not_implemented`
- `no_route_match`
- `ambiguous_route_match`

These are useful because they separate:

- no catalog support
- catalog support for the op but not the dtype
- importer does not yet understand the route contract
- importer understands the route contract but this case does not satisfy any
  route predicate
- catalog ambiguity

## Design Consequences

If this design is followed:

- importer coverage is determined from ground-truth catalog data plus raw case
  inputs
- unsupported cases remain visible instead of being silently dropped
- adding support for a new family usually means adding a new generic route
  contract lowerer, not writing per-op routing code
- expected coverage remains a validation artifact, not an import decision input

## Remaining Design Work

A temporary supported-op whitelist of `ADD`, `MUL`, and `CPY` is useful for
preserving the currently validated surface, but it does not by itself complete
the redesign.

The redesign is only complete when the following structural issues are
addressed.

### 1. Separate Supported Surface From Routing Logic

The operation whitelist must be only a temporary surface-management control.

It must not:

- decide which kernel family an op uses
- bypass catalog route evaluation
- encode route-specific behavior

Its only acceptable role is:

- limiting which ops are currently materialized for the validated slice
- limiting which runtime tests are currently registered for that slice

### 2. Make `import_mapping_registry.py` A Pure Lowerer Registry

`import_mapping_registry.py` should not be an operation routing file.

Its responsibilities should be reduced to:

- identifying route contracts from route metadata
- exposing generic lowerers for those contracts
- rejecting unsupported raw-case layouts with contract-specific errors

It should not:

- choose kernel families
- map operations to kernels
- contain duplicated routing policy already present in the catalog

### 3. Keep Route Resolution Fully Catalog-Driven

`import_route_resolution.py` should remain the only place that decides whether a
case is covered by a route.

That decision must come from:

- route `op`
- route dtype support
- generic lowerer availability
- lowered shape
- route shape domain
- route guards
- route priority

This is the core redesign requirement. No second routing layer should exist in
parallel.

### 4. Drive Runtime Registration From Generated Importer Outputs

The long-term design should not require configure-time operation discovery from
an expected coverage artifact or from a permanent manual whitelist.

The intended end state is:

- importer emits per-op and aggregate manifests
- runtime tests are registered from those emitted manifests

The temporary whitelist is acceptable during the transition, but runtime test
existence should ultimately be derived from generated importer outputs.

### 5. Preserve Failure Transparency

The redesign is not complete if unsupported cases become hidden or silently
dropped.

The importer must continue to surface:

- ops with no catalog routes
- dtype mismatches
- missing lowering implementations
- lowered cases that satisfy no route
- route ambiguity

This is required so coverage remains diagnosable while support expands.

## End State

The intended end state is:

- grouped YAML supplies raw operation inputs
- the catalog supplies legal kernel routes
- generic lowerers translate raw inputs into route ABI shapes
- route predicates decide coverage
- emitted manifests drive runtime tests
- any temporary op whitelist is removed or reduced to a non-routing validation
  surface control

No second routing layer should exist outside the catalog.
