# Loom Kernel Benchmark Script Design

## Objective

Generate a benchmark tree at build time that mirrors the route catalog and
contains directly runnable scripts for each Loom kernel implementation. A user
should be able to discover a kernel, run its generated benchmark script,
materialize a candidate benchmark tree, rerun the same route, and inspect
collected results without
reconstructing Python command lines or reprocessing grouped YAML.

The benchmark system is not a test system. Build-time generation prepares
benchmark assets, manifests, and scripts only. Runtime benchmarking remains an
explicit user action because it needs device access, takes variable time, and
produces machine-specific performance data.

## Public Tools

The public CLI surface is intentionally small:

- `loom-bench-materialize`
- `loom-bench-collect`
- `loom-bench-compare`

Generated shell scripts are the benchmark execution surface. Python must not
grow a direct benchmark runner.

## Hard Requirements

- Keep each installed tool single-purpose.
- Do not add a central benchmark dispatcher with subcommands.
- Do not expose every helper as a CLI tool.
- Keep implementation responsibilities in focused modules under
  `ggml_hrx_kernel_bench.benchmarking`.
- Reuse existing materialized descriptor artifacts. Do not reprocess grouped
  YAML during benchmark materialization.
- Preserve generated route-local `runs/` results when regenerating scripts.
- Remove obsolete runner paths and duplicate helper families in the same pass
  that introduces replacement behavior.
- Tests must import and validate subsystem modules directly.

## Tool Responsibilities

### `loom-bench-materialize`

Build-time/materialization tool.

Owns:

- discovery from existing prepared descriptor artifacts;
- bucketing descriptor cases by op and routed kernel implementation;
- writing catalog, op, route, and case manifests;
- preparing baseline `benchmark.loom` files;
- optionally baking a candidate kernel source into generated `benchmark.loom`
  files;
- writing route and case `run.sh` scripts;
- writing route `collect.sh` and catalog/op `run-all.sh` scripts;
- cleaning stale generated scripts/manifests/cases while preserving `runs/`.

Does not run benchmarks.

### `loom-bench-collect`

Post-processing tool.

Owns:

- reading a generated route manifest and route run directory;
- parsing `iree-benchmark-loom` JSONL output;
- distinguishing missing output, zero benchmark rows, bad rows, and nonzero
  process return codes;
- computing timing summaries and estimated FLOP/s;
- writing `results.jsonl`, `summary.json`, and `summary.md`.

### `loom-bench-compare`

Optimization-loop analysis tool.

Owns:

- comparing two collected result sets;
- reporting improvements, regressions, missing cases, untimed cases, and failed
  cases;
- grouping comparisons by selected shape metadata;
- applying simple regression policies.

Compile summaries remain part of collected result rows and comparison data.
They do not justify a separate public compile-diff tool yet.

## Internal Modules

```text
src/ggml_hrx_kernel_bench/benchmarking/
  common.py
  discovery.py
  workbench.py
  materialize.py
  result_parsing.py
  collect.py
  compare.py
```

Ownership:

- `common.py`: schema constants, JSON helpers, hashing, timestamps, executable
  script writes.
- `discovery.py`: `DescriptorCase`, `BenchmarkBucket`, descriptor digesting,
  FLOP estimates, shape buckets, and descriptor bucketing.
- `workbench.py`: descriptor-to-`benchmark.loom` conversion and fixture
  preparation.
- `materialize.py`: public materialization CLI and generated script/manifests.
- `result_parsing.py`: parsing raw benchmark JSONL and extracting timing,
  compile, and throughput summaries.
- `collect.py`: public result collection CLI over generated run directories.
- `compare.py`: public comparison CLI over collected result files.

## Generated Layout

Generate the benchmark tree under:

```text
build/benchmarks/loom-kernels/catalog/v2/
  index.json
  run-all.sh
  MUL_MAT/
    index.json
    run-all.sh
    mul_mat_f16_f32_tiled_batched_4d/
      manifest.json
      run.sh
      collect.sh
      cases/
        <case-id>-<digest>/
          manifest.json
          benchmark.loom
          run.sh
```

The generated tree is derived from descriptor prepare artifacts and should not
be checked in.

Regeneration removes stale generated files from routes listed in the previous
generated index:

- old route `manifest.json`, `run.sh`, and `collect.sh`;
- old route `cases/`;
- old op/catalog indexes and `run-all.sh`.

It preserves every route-local `runs/` directory.

## Execution Model

Build-time command:

```bash
cmake --build build --target kernel-benchmark-llama-cpp-v2-scripts
```

That target invokes `tests/infra/materialize_loom_benchmarks.py`, which calls
`loom-bench-materialize` logic without inline CMake Python snippets.

Runtime examples:

Materialize a candidate benchmark tree:

```bash
loom-bench-materialize \
  --prepare-root build/tests/kernels/artifacts/kernel-prepare-llama-cpp-v2 \
  --repo-root . \
  --asset-root build/generated/assets \
  --op MUL_MAT \
  --route-id mul_mat_f16_f32_tiled_batched_4d \
  --kernel-source /tmp/candidates/f16_f32_tiled_batched.loom \
  --output-root /tmp/mul-mat-candidate-tree
```

Run a route. The first positional argument is the run directory; every later
argument is passed directly to `iree-benchmark-loom`.

```bash
build/benchmarks/loom-kernels/catalog/v2/MUL_MAT/mul_mat_f16_f32_tiled_batched_4d/run.sh \
  /tmp/mul-mat-baseline \
  --device=hip://0 \
  --iterations=100
```

```bash
/tmp/mul-mat-candidate-tree/catalog/v2/MUL_MAT/mul_mat_f16_f32_tiled_batched_4d/run.sh \
  /tmp/mul-mat-candidate \
  --device=hip://0 \
  --iterations=100
```

```bash
loom-bench-compare \
  --baseline /tmp/mul-mat-baseline/results.jsonl \
  --candidate local-memory-tile=/tmp/mul-mat-candidate/results.jsonl \
  --group-by k \
  --group-by flop-bucket
```

## Non-Goals

- no benchmark execution during build;
- no CTest integration for benchmark execution;
- no grouped YAML reprocessing;
- no Python benchmark runner;
- no public candidate, DB, doctor, report, compile-diff, or prepare-case tools
  until those responsibilities prove large enough to justify dedicated tools.
