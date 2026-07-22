from __future__ import annotations

import argparse
import json
import shlex
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

from ggml_hrx_kernel_bench.required_tools import require_tool, resolve_tool

from .common import (
    SCRIPT_CASE_MANIFEST_SCHEMA,
    SCRIPT_INDEX_SCHEMA,
    SCRIPT_ROUTE_MANIFEST_SCHEMA,
    load_json as _load_json,
    safe_name as _safe_name,
    timestamp as _timestamp,
    write_executable_script as _write_executable_script,
    write_json_file as _write_json_file,
)
from .discovery import (
    BenchmarkBucket,
    DescriptorCase,
    _selected_buckets,
    estimate_case_flops,
    shape_bucket_for_case,
)
from .workbench import _write_descriptor_workbench


def _script_literal(value: str | Path) -> str:
    return shlex.quote(str(value))


def _script_array_literal(values: Sequence[str | Path]) -> str:
    return "\n".join(f"  {_script_literal(value)}" for value in values)


def _case_script_text(
    *,
    benchmark_runner: str,
    benchmark_device: str,
    benchmark_measure: str,
    benchmark_symbol: str,
) -> str:
    return f"""#!/usr/bin/env bash
set -uo pipefail

CASE_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
BENCHMARK_RUNNER={_script_literal(benchmark_runner)}
BENCHMARK_DEVICE={_script_literal(benchmark_device)}
BENCHMARK_MEASURE={_script_literal(benchmark_measure)}
BENCHMARK_SYMBOL={_script_literal(benchmark_symbol)}

RUN_CASE_DIR="${{1:-$CASE_DIR/runs/$(date -u +%Y%m%dT%H%M%SZ)}}"
if [[ $# -gt 0 ]]; then shift; fi
mkdir -p "$RUN_CASE_DIR/artifacts"

OUTPUT="$RUN_CASE_DIR/benchmark-results.jsonl"
COMMAND=(
  "$BENCHMARK_RUNNER"
  "$CASE_DIR/benchmark.loom"
  "--device=$BENCHMARK_DEVICE"
  "--measure=$BENCHMARK_MEASURE"
  "--benchmark=$BENCHMARK_SYMBOL"
  "--output-format=jsonl"
  "--output=$OUTPUT"
  "--artifact-bundle-dir=$RUN_CASE_DIR/artifacts"
  "$@"
)

printf "%s\\n" "${{COMMAND[@]}}" > "$RUN_CASE_DIR/command.txt"
"${{COMMAND[@]}}" > "$RUN_CASE_DIR/stdout.txt" 2> "$RUN_CASE_DIR/stderr.txt"
STATUS=$?
printf "%s\\n" "$STATUS" > "$RUN_CASE_DIR/returncode.txt"
echo "output: $OUTPUT"
echo "status: $STATUS"
exit "$STATUS"
"""


def _route_script_text() -> str:
    return """#!/usr/bin/env bash
set -uo pipefail

ROUTE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_DIR="${1:-$ROUTE_DIR/runs/$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ $# -gt 0 ]]; then shift; fi
mkdir -p "$RUN_DIR/cases"

STATUS=0
for CASE_SCRIPT in "$ROUTE_DIR"/cases/*/run.sh; do
  [[ -f "$CASE_SCRIPT" ]] || continue
  CASE_NAME="$(basename "$(dirname "$CASE_SCRIPT")")"
  bash "$CASE_SCRIPT" "$RUN_DIR/cases/$CASE_NAME" "$@"
  CASE_STATUS=$?
  if [[ "$CASE_STATUS" -ne 0 ]]; then STATUS=1; fi
done

bash "$ROUTE_DIR/collect.sh" "$RUN_DIR"
COLLECT_STATUS=$?
if [[ "$COLLECT_STATUS" -ne 0 ]]; then STATUS="$COLLECT_STATUS"; fi

echo "results: $RUN_DIR/results.jsonl"
echo "summary: $RUN_DIR/summary.json"
echo "markdown: $RUN_DIR/summary.md"
exit "$STATUS"
"""


def _route_collect_script_text(*, collect_command: Sequence[str | Path], repo_root: Path) -> str:
    return f"""#!/usr/bin/env bash
set -uo pipefail

ROUTE_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
REPO_ROOT={_script_literal(repo_root)}
COLLECT_COMMAND=(
{_script_array_literal(collect_command)}
)

RUN_DIR="${{1:-}}"

if [[ -z "$RUN_DIR" ]]; then
  echo "usage: $0 RUN_DIR" >&2
  exit 2
fi

if [[ -n "${{PYTHONPATH:-}}" ]]; then
  export PYTHONPATH="$REPO_ROOT/src:$PYTHONPATH"
else
  export PYTHONPATH="$REPO_ROOT/src"
fi

"${{COLLECT_COMMAND[@]}}" \\
  --manifest "$ROUTE_DIR/manifest.json" \\
  --run-dir "$RUN_DIR" \\
  --output "$RUN_DIR/summary.json" \\
  --results "$RUN_DIR/results.jsonl" \\
  --markdown "$RUN_DIR/summary.md"
"""


def _run_all_script_text(*, level: str) -> str:
    glob = "*/run.sh" if level == "op" else "*/*/run.sh"
    return f"""#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
RUN_ROOT="${{1:-$ROOT_DIR/runs/$(date -u +%Y%m%dT%H%M%SZ)}}"
if [[ $# -gt 0 ]]; then shift; fi
STATUS=0
for RUN_SCRIPT in "$ROOT_DIR"/{glob}; do
  [[ -f "$RUN_SCRIPT" ]] || continue
  RUN_NAME="$(basename "$(dirname "$RUN_SCRIPT")")"
  bash "$RUN_SCRIPT" "$RUN_ROOT/$RUN_NAME" "$@"
  RUN_STATUS=$?
  if [[ "$RUN_STATUS" -ne 0 ]]; then STATUS=1; fi
done
echo "run root: $RUN_ROOT"
exit "$STATUS"
"""


def _run_case_dir_name(case: DescriptorCase) -> str:
    return _safe_name(f"{case.case_id}-{case.execution_digest[:12]}")


def _remove_generated_route_files(route_dir: Path) -> None:
    for name in ("manifest.json", "run.sh", "aggregate.sh", "collect.sh"):
        path = route_dir / name
        if path.is_file():
            path.unlink()
    cases_dir = route_dir / "cases"
    if cases_dir.is_dir():
        shutil.rmtree(cases_dir)


def _clean_generated_catalog(catalog_root: Path) -> None:
    index_path = catalog_root / "index.json"
    if not index_path.is_file():
        return
    index = _load_json(index_path)
    routes = index.get("routes")
    if isinstance(routes, list):
        for route in routes:
            if not isinstance(route, dict) or not isinstance(route.get("manifest_path"), str):
                continue
            route_dir = Path(route["manifest_path"]).parent.resolve()
            try:
                route_dir.relative_to(catalog_root.resolve())
            except ValueError:
                continue
            _remove_generated_route_files(route_dir)
    for name in ("index.json", "run-all.sh"):
        path = catalog_root / name
        if path.is_file():
            path.unlink()
    for op_dir in catalog_root.iterdir():
        if not op_dir.is_dir():
            continue
        for name in ("index.json", "run-all.sh"):
            path = op_dir / name
            if path.is_file():
                path.unlink()


def _generated_script_case_manifest(
    *,
    case: DescriptorCase,
    route_dir: Path,
    case_dir: Path,
    workbench_path: Path,
    bench_symbol: str,
    preparation: dict[str, Any],
    repo_root: Path,
    benchmark_runner: str,
    benchmark_device: str,
    benchmark_measure: str,
) -> dict[str, Any]:
    flop_estimate = estimate_case_flops(case)
    run_case_dir_name = _run_case_dir_name(case)
    return {
        "schema": SCRIPT_CASE_MANIFEST_SCHEMA,
        "timestamp": _timestamp(),
        "op": case.op,
        "route_id": case.route_id,
        "implementation_id": case.implementation_id,
        "case_id": case.case_id,
        "run_case_dir_name": run_case_dir_name,
        "descriptor_path": str(case.descriptor_path),
        "run_manifest_path": str(case.run_manifest_path),
        "prepared_entry": case.prepared_entry,
        "root": case.root,
        "kernel_source": case.kernel_source,
        "normalized_kernel_source": case.normalized_kernel_source,
        "source_content_hash": case.source_content_hash,
        "descriptor_execution_digest": case.execution_digest,
        "benchmark_symbol": bench_symbol,
        "workbench_path": str(workbench_path),
        "case_dir": str(case_dir),
        "route_dir": str(route_dir),
        "repo_root": str(repo_root),
        "shape_bucket": shape_bucket_for_case(case, flop_estimate=flop_estimate),
        "flop_estimate": flop_estimate,
        "estimated_flops": flop_estimate.get("estimated_flops"),
        "preparation": preparation,
        "defaults": {
            "benchmark_runner": benchmark_runner,
            "benchmark_device": benchmark_device,
            "benchmark_measure": benchmark_measure,
        },
    }


def _generated_script_route_manifest(
    *,
    bucket: BenchmarkBucket,
    route_dir: Path,
    case_manifests: Sequence[dict[str, Any]],
    prepare_root: Path,
    repo_root: Path,
    asset_root: Path | None,
    output_root: Path,
    benchmark_runner: str,
    benchmark_device: str,
    benchmark_measure: str,
) -> dict[str, Any]:
    return {
        "schema": SCRIPT_ROUTE_MANIFEST_SCHEMA,
        "timestamp": _timestamp(),
        "op": bucket.op,
        "route_id": bucket.route_id,
        "implementation_id": bucket.implementation_id,
        "root": bucket.root,
        "kernel_source": bucket.kernel_source,
        "normalized_kernel_source": bucket.normalized_kernel_source,
        "source_content_hash": bucket.source_content_hash,
        "prepare_root": str(prepare_root),
        "repo_root": str(repo_root),
        "asset_root": str(asset_root) if asset_root else None,
        "output_root": str(output_root),
        "route_dir": str(route_dir),
        "case_count": len(case_manifests),
        "case_ids": [str(item["case_id"]) for item in case_manifests],
        "cases": [
            {
                "case_id": item["case_id"],
                "run_case_dir_name": item["run_case_dir_name"],
                "manifest_path": str(Path(item["case_dir"]) / "manifest.json"),
                "descriptor_path": item["descriptor_path"],
                "descriptor_execution_digest": item["descriptor_execution_digest"],
                "benchmark_symbol": item["benchmark_symbol"],
                "workbench_path": item["workbench_path"],
                "estimated_flops": item["estimated_flops"],
                "flop_estimate": item["flop_estimate"],
                "shape_bucket": item["shape_bucket"],
            }
            for item in case_manifests
        ],
        "defaults": {
            "benchmark_runner": benchmark_runner,
            "benchmark_device": benchmark_device,
            "benchmark_measure": benchmark_measure,
        },
    }


def command_generate_scripts(args: argparse.Namespace) -> int:
    selection_args = argparse.Namespace(**vars(args))
    selection_args.kernel_source = None
    buckets = _selected_buckets(selection_args)
    if not buckets:
        raise RuntimeError("no benchmark buckets matched the requested selection")

    prepare_root = args.prepare_root.resolve()
    repo_root = args.repo_root.resolve()
    asset_root = args.asset_root.resolve() if args.asset_root else None
    output_root = args.output_root.resolve()
    catalog_root = output_root / "catalog" / "v2"
    benchmark_runner = args.benchmark_runner or resolve_tool("iree-benchmark-loom", tool_dir=args.tool_dir)
    benchmark_runner = benchmark_runner or "iree-benchmark-loom"
    benchmark_device = getattr(args, "benchmark_device", None) or "amdgpu"
    benchmark_measure = getattr(args, "benchmark_measure", None) or "dispatch_complete"
    loom_link = args.loom_link or require_tool("loom-link", tool_dir=args.tool_dir)
    collect_tool = getattr(args, "collect_tool", None)
    collect_command: Sequence[str | Path]
    if collect_tool:
        collect_command = shlex.split(collect_tool)
    else:
        collect_command = (sys.executable, "-m", "ggml_hrx_kernel_bench.benchmarking.collect")
    kernel_source_override = args.kernel_source.resolve() if args.kernel_source else None

    _clean_generated_catalog(catalog_root)

    generated_routes: list[dict[str, Any]] = []
    for bucket in buckets:
        route_dir = catalog_root / bucket.op / bucket.route_id
        case_manifests: list[dict[str, Any]] = []
        for case in bucket.cases:
            case_dir = route_dir / "cases" / _run_case_dir_name(case)
            workbench_path, bench_symbol, preparation = _write_descriptor_workbench(
                case=case,
                run_dir=case_dir,
                repo_root=repo_root,
                loom_link=loom_link,
                kernel_source_override=kernel_source_override,
            )
            case_manifest = _generated_script_case_manifest(
                case=case,
                route_dir=route_dir,
                case_dir=case_dir,
                workbench_path=workbench_path,
                bench_symbol=bench_symbol,
                preparation=preparation,
                repo_root=repo_root,
                benchmark_runner=benchmark_runner,
                benchmark_device=benchmark_device,
                benchmark_measure=benchmark_measure,
            )
            _write_json_file(case_dir / "manifest.json", case_manifest)
            _write_executable_script(
                case_dir / "run.sh",
                _case_script_text(
                    benchmark_runner=benchmark_runner,
                    benchmark_device=benchmark_device,
                    benchmark_measure=benchmark_measure,
                    benchmark_symbol=bench_symbol,
                ),
            )
            case_manifests.append(case_manifest)

        route_manifest = _generated_script_route_manifest(
            bucket=bucket,
            route_dir=route_dir,
            case_manifests=case_manifests,
            prepare_root=prepare_root,
            repo_root=repo_root,
            asset_root=asset_root,
            output_root=output_root,
            benchmark_runner=benchmark_runner,
            benchmark_device=benchmark_device,
            benchmark_measure=benchmark_measure,
        )
        _write_json_file(route_dir / "manifest.json", route_manifest)
        _write_executable_script(
            route_dir / "run.sh",
            _route_script_text(),
        )
        _write_executable_script(
            route_dir / "collect.sh",
            _route_collect_script_text(collect_command=collect_command, repo_root=repo_root),
        )
        generated_routes.append(
            {
                "op": bucket.op,
                "route_id": bucket.route_id,
                "implementation_id": bucket.implementation_id,
                "case_count": len(case_manifests),
                "manifest_path": str(route_dir / "manifest.json"),
                "run_script": str(route_dir / "run.sh"),
            }
        )

    ops = sorted({route["op"] for route in generated_routes})
    for op in ops:
        op_dir = catalog_root / op
        op_routes = [route for route in generated_routes if route["op"] == op]
        _write_json_file(
            op_dir / "index.json",
            {
                "schema": SCRIPT_INDEX_SCHEMA,
                "timestamp": _timestamp(),
                "level": "op",
                "op": op,
                "route_count": len(op_routes),
                "case_count": sum(int(route["case_count"]) for route in op_routes),
                "routes": op_routes,
            },
        )
        _write_executable_script(op_dir / "run-all.sh", _run_all_script_text(level="op"))

    index = {
        "schema": SCRIPT_INDEX_SCHEMA,
        "timestamp": _timestamp(),
        "level": "catalog",
        "catalog": "v2",
        "output_root": str(output_root),
        "catalog_root": str(catalog_root),
        "route_count": len(generated_routes),
        "case_count": sum(int(route["case_count"]) for route in generated_routes),
        "ops": ops,
        "routes": generated_routes,
    }
    _write_json_file(catalog_root / "index.json", index)
    _write_executable_script(catalog_root / "run-all.sh", _run_all_script_text(level="catalog"))
    print(json.dumps(index, indent=2, sort_keys=True))
    return 0


def _add_common_selection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prepare-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--op")
    parser.add_argument("--route-id")
    parser.add_argument("--implementation-id")
    parser.add_argument("--root")
    parser.add_argument("--case-id")
    parser.add_argument("--no-dedupe", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize descriptor-backed Loom kernel benchmark scripts."
    )
    _add_common_selection_args(parser)
    parser.add_argument("--output-root", type=Path, default=Path("build/benchmarks/loom-kernels"))
    parser.add_argument("--tool-dir")
    parser.add_argument("--benchmark-runner")
    parser.add_argument("--benchmark-device", default="amdgpu")
    parser.add_argument("--benchmark-measure", default="dispatch_complete")
    parser.add_argument("--loom-link")
    parser.add_argument("--kernel-source", type=Path, help="candidate kernel source to bake into generated benchmarks")
    parser.add_argument("--collect-tool", help="command used by generated collect.sh scripts")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return command_generate_scripts(args)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
