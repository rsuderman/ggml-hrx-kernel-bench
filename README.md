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
- atlas-first route schedules with optional observed-shape refinement,
- NumPy fixture and golden generation for pilot families,
- focused Loom link/compile/correctness-run commands,
- flash-attention-first route inventory setup,
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
  specs/
  kernels/
    hrx2/
  catalog/
    hrx2/
  schemas/
```

`kernels/hrx2/` contains the bench-owned HRX2 Loom corpus. `catalog/hrx2/`
contains the route/source metadata imported from the HRX2 handoff.

Family-specific route binding policy lives in
`src/ggml_hrx_kernel_bench/family_specs.py`. Add or correct kernel shape/config
requirements there instead of adding global string-matching fallbacks.

Search schedule policy lives in `src/ggml_hrx_kernel_bench/route_schedules.py`.
The default path is atlas-first: schedules encode a small set of representative
decode, short prompt, tail, and prefill shapes for each family class. Observed
live shapes are metadata that refine the atlas later; they are not required to
make a route visible.

Observed live shapes live in `catalog/hrx2/observed_shapes.json`. Synthetic
trace ingestion is available so llama.cpp profiling can extend the atlas once
there are enough anchored wins to justify broadening coverage.

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

Generate an HRX2 import manifest:

```bash
python3 -m ggml_hrx_kernel_bench \
  --output-dir runs/hrx2-import \
  --original-hrx2-root /path/to/llama.cpp-ref/ggml/src/ggml-hrx2 \
  import-hrx2
```

Plan the full corpus without compiling:

```bash
python3 -m ggml_hrx_kernel_bench \
  --output-dir runs/hrx2-plan \
  plan
```

Plan using accumulated live-profiled shapes:

```bash
python3 -m ggml_hrx_kernel_bench \
  --output-dir runs/hrx2-observed-plan \
  --sweep observed \
  plan
```

If no observed shape matches a route, `--sweep observed` falls back to that
route's minimal smoke shape so the route remains visible in ledgers.

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

Merge extracted llama.cpp shape traces into the bench metadata:

```bash
python3 -m ggml_hrx_kernel_bench \
  --output-dir runs/shape-accumulate \
  --shape-trace runs/llama-shapes.jsonl \
  accumulate-shapes
```

The intended trace row shape is JSONL, one observed dispatch/route shape per
line:

```json
{
  "family": "mul_mat_q4_k_f32",
  "source_id": "mul_mat_q4_k_f32",
  "route_id": "mul_mat_q4_k_f32_wmma64x64_f16acc_k256_8192_r64_32768_c64_wg128",
  "root_symbol": "@hrx2_mul_mat_q4_k_f32_wmma64x64_f16acc",
  "shape": {"k": 4096, "rows": 4096, "cols": 64},
  "count": 37,
  "tags": ["llama-bench", "prefill"],
  "source": {
    "program": "llama-bench",
    "model": "qwen2.5-7b-q4_k_m",
    "args": "-p 512 -b 512 -ub 512"
  }
}
```

Rows may also be copied from bench ledgers: if a row contains a `candidate`
object, `accumulate-shapes` reads `candidate.family`, `candidate.route_id`,
`candidate.root_symbol`, `candidate.source_id`, and `candidate.shape`.

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

Inventory the initial flash attention route surface:

```bash
python3 -m ggml_hrx_kernel_bench \
  --output-dir runs/flash-attn-route-inventory \
  route-inventory --op FLASH_ATTN_EXT
```

## Flash Attention Route Inventory

The old standalone benchmark materialization, comparison, timing report, and
profitability reducer tooling has been removed. The maintained route inventory
tool starts with `FLASH_ATTN_EXT`, which has the smallest model-visible surface
in the current imported artifacts.

The first maintained command is:

```text
route-inventory
```

It consumes route import artifacts and writes:

```text
<output-dir>/route-inventory.json
<output-dir>/ledger.jsonl
```

Use `--generated-import-dir` to point at a specific route import artifact root.
By default the command reads the generated Llama 3.3 8B Q8_0 model route import
under `build/tests/models/artifacts/`.

Legacy single-spec mode is still available:

```bash
python3 -m ggml_hrx_kernel_bench \
  --spec specs/mul_mat_q4_k_f32.json \
  --output-dir runs/q4k-plan \
  plan
```

Compile commands require explicit Loom tool paths:

```bash
python3 -m ggml_hrx_kernel_bench \
  --spec specs/mul_mat_q4_k_f32.json \
  --kernel-source /path/to/mul_mat_q4_k_f32.loom \
  --loom-compile /path/to/loom-compile \
  --target gfx1100 \
  --output-dir runs/q4k-compile \
  compile
```

When this project is promoted to its own repository, CI and local examples
should provide a small synthetic Loom source rather than depending on the HRX
workspace checkout.

## Outputs

Commands write to the caller-provided `--output-dir`. Generated files should be
treated as run artifacts. Promote only intentional specs, kernels, concise
summaries, or catalog rows into source control.

The primary evidence file is:

```text
<output-dir>/ledger.jsonl
```

Rows are append-only JSON objects carrying spec identity, config, tool paths,
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
