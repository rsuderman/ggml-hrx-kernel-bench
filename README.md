# GGML HRX Kernel Bench

Standalone Python project for developing and verifying GGML HRX Loom kernels
outside llama.cpp.

This project is intentionally path-neutral. It does not assume a workspace
layout, build directory, cache directory, ROCm installation, or llama.cpp
checkout. Tool paths, kernel paths, and output directories are supplied by CLI
flags or config files.

## Status

This project contains the imported HRX2 Loom corpus and route/correctness
infrastructure:

- HRX2 kernel and catalog import validation,
- route-backed candidate planning,
- explicit per-family config binding specs,
- route schedule sweeps for representative coverage points,
- NumPy fixture and golden generation for pilot families,
- focused Loom link/compile/correctness-run commands,
- JSONL ledgers plus preserved evidence directories,
- catalog candidate summaries.

Unsupported or broken kernels are represented as ledger rows instead of being
excluded. Imported HRX2 kernels should remain source-faithful; target-specific
experiments belong in run metadata and temporary evidence, not kernel rewrites.

## Layout

```text
ggml-hrx-kernel-bench/
  pyproject.toml
  README.md
  src/ggml_hrx_kernel_bench/
  kernels/
    v2/
  catalog/
    v2/
  schemas/
```

`kernels/v2/` contains the bench-owned HRX2 Loom corpus. `catalog/v2/`
contains source metadata and `catalog/v2/routing/` contains the route metadata
used by the v2 router.

Route schedule and shape policy live under
`src/ggml_hrx_kernel_bench/routing/v2/`. Add or correct tensor descriptors,
route predicates, config extraction, and fallback schedule behavior there
instead of adding global string-matching fallbacks.

## Install

From this directory:

```bash
python3 -m pip install -e .
```

This installs the Python dependencies used by the CMake build and tooling.

For fixture and golden generation:

```bash
python3 -m pip install -e ".[numpy]"
```

## CMake Build

The default CMake build now materializes this repo's runtime assets and pulls
the Loom utilities into the same build graph from an `hrx-systems` checkout.
Configure CMake with a reference to the source tree that contains the HRX CMake
project and its `loom/` subtree:

```bash
cmake -S . -B build \
  -DGGML_HRX_HRX_SYSTEMS_SOURCE_DIR=/path/to/hrx-systems
cmake --build build
```

That build produces `loom-link`, `loom-compile`, `ggml-hrx-run-loom`, and
`iree-test-loom` under `build/tools`, and the tests/import validation targets
use that in-tree directory automatically. It also builds the standalone native
v2 route selector.

If you only need the Python/materialized-asset targets and do not want the
nested Loom build, disable it explicitly:

```bash
cmake -S . -B build -DGGML_HRX_BUILD_LOOM_TOOLS=OFF
```

## Basic Usage

Plan the full corpus without compiling:

```bash
python3 -m ggml_hrx_kernel_bench \
  --output-dir runs/hrx2-plan \
  plan
```

Plan a bounded atlas sweep:

```bash
python3 -m ggml_hrx_kernel_bench \
  --output-dir runs/hrx2-edge-plan \
  --sweep edge \
  plan
```

`--sweep edge` emits up to roughly six schedule points per route, depending on
family. The points are intentionally opinionated rather than exhaustive; each
candidate ledger row includes a `candidate.schedule` object with the selected
scenario, source, and weight.

Plan or run a pilot family slice:

```bash
python3 -m ggml_hrx_kernel_bench \
  --output-dir runs/pilot-fixtures \
  --family mul_mat_q4_k_f32,rms_norm_f32,copy_f32_f16,cont_f32 \
  --limit 8 \
  fixtures
```

Compile a focused candidate set:

```bash
python3 -m ggml_hrx_kernel_bench \
  --output-dir runs/copy-compile \
  --loom-link /path/to/loom-link \
  --loom-compile /path/to/loom-compile \
  --rocm-path /path/to/rocm \
  --family copy_f32_f16 \
  --limit 1 \
  compile
```


## Outputs

Commands write to the caller-provided `--output-dir`. Generated files should be
treated as run artifacts. Promote only intentional specs, kernels, concise
summaries, or catalog rows into source control.

The primary evidence file is:

```text
<output-dir>/ledger.jsonl
```

Rows are append-only JSON objects carrying candidate identity, config, tool paths,
shape parameters, and command results.

Current smoke evidence from this workspace is under:

```text
cache/ggml-hrx-kernel-bench/import-smoke/
cache/ggml-hrx-kernel-bench/plan-family-spec-smoke/
cache/ggml-hrx-kernel-bench/fixtures-pilot/
cache/ggml-hrx-kernel-bench/compile-copy-smoke/
cache/ggml-hrx-kernel-bench/compile-rms-smoke2/
cache/ggml-hrx-kernel-bench/compile-q4-smoke/
cache/ggml-hrx-kernel-bench/run-copy-smoke2/
```
