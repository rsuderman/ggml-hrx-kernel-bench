from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from .common import (
    RESULT_SCHEMA,
    SCRIPT_ROUTE_MANIFEST_SCHEMA,
    SUMMARY_SCHEMA,
    load_json as _load_json,
    timestamp as _timestamp,
    write_json_file as _write_json_file,
)
from .result_parsing import _benchmark_result_summary


def _read_returncode(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except ValueError:
        return None


def _read_text_if_present(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _collect_case_result(
    *,
    route_manifest: dict[str, Any],
    case_manifest: dict[str, Any],
    run_dir: Path,
    candidate_name: str,
) -> dict[str, Any]:
    run_case_dir = run_dir / "cases" / str(case_manifest["run_case_dir_name"])
    output_path = run_case_dir / "benchmark-results.jsonl"
    returncode = _read_returncode(run_case_dir / "returncode.txt")
    flop_estimate = case_manifest.get("flop_estimate")
    flop_estimate = flop_estimate if isinstance(flop_estimate, dict) else None
    benchmark_summary = _benchmark_result_summary(output_path, flop_estimate=flop_estimate)
    generated_preparation = (
        case_manifest.get("preparation") if isinstance(case_manifest.get("preparation"), dict) else {}
    )
    preparation = generated_preparation
    status = "pass" if returncode == 0 and benchmark_summary.get("status") == "ok" else "failed"
    return {
        "schema": RESULT_SCHEMA,
        "timestamp": _timestamp(),
        "candidate_name": candidate_name,
        "implementation_id": route_manifest.get("implementation_id"),
        "op": route_manifest.get("op"),
        "case_id": case_manifest.get("case_id"),
        "route_id": route_manifest.get("route_id"),
        "kernel_source": route_manifest.get("kernel_source"),
        "descriptor_kernel_source": preparation.get("descriptor_kernel_source"),
        "descriptor_kernel_source_hash": preparation.get("descriptor_kernel_source_hash"),
        "effective_kernel_source": preparation.get("effective_kernel_source"),
        "effective_kernel_source_hash": preparation.get("effective_kernel_source_hash"),
        "source_override_used": preparation.get("source_override_used", False),
        "root": route_manifest.get("root"),
        "descriptor_path": case_manifest.get("descriptor_path"),
        "descriptor_execution_digest": case_manifest.get("descriptor_execution_digest"),
        "normalized_kernel_source": route_manifest.get("normalized_kernel_source"),
        "source_content_hash": route_manifest.get("source_content_hash"),
        "benchmark_symbol": case_manifest.get("benchmark_symbol"),
        "workbench_path": case_manifest.get("workbench_path"),
        "benchmark_output_path": str(output_path),
        "command": _read_text_if_present(run_case_dir / "command.txt").splitlines(),
        "process_returncode": returncode,
        "stdout": _read_text_if_present(run_case_dir / "stdout.txt"),
        "stderr": _read_text_if_present(run_case_dir / "stderr.txt"),
        "preparation": preparation,
        "benchmark_summary": benchmark_summary,
        "compile_summary": benchmark_summary.get("compile_summary"),
        "flop_estimate": flop_estimate,
        "estimated_flops": case_manifest.get("estimated_flops"),
        "shape_bucket": case_manifest.get("shape_bucket") if isinstance(case_manifest.get("shape_bucket"), dict) else {},
        "status": status,
    }


def _timed_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    timed: list[dict[str, Any]] = []
    for row in rows:
        summary = row.get("benchmark_summary")
        timing = summary.get("operation_timing_ns") if isinstance(summary, dict) else None
        if isinstance(timing, dict) and isinstance(timing.get("p50"), int | float):
            timed.append(row)
    return timed


def _row_p50(row: dict[str, Any]) -> float:
    summary = row.get("benchmark_summary")
    timing = summary.get("operation_timing_ns") if isinstance(summary, dict) else {}
    return float(timing["p50"])


def _row_flops_per_second(row: dict[str, Any]) -> float | None:
    summary = row.get("benchmark_summary")
    throughput = summary.get("throughput") if isinstance(summary, dict) else None
    rows = throughput.get("rows") if isinstance(throughput, dict) else None
    if not isinstance(rows, list) or not rows:
        return None
    value = rows[0].get("flops_per_second_from_p50_ns") if isinstance(rows[0], dict) else None
    return float(value) if isinstance(value, int | float) else None


def _case_brief(row: dict[str, Any]) -> dict[str, Any]:
    summary = row.get("benchmark_summary")
    timing = summary.get("operation_timing_ns") if isinstance(summary, dict) else None
    p50 = timing.get("p50") if isinstance(timing, dict) else None
    return {
        "case_id": row.get("case_id"),
        "status": row.get("status"),
        "p50_ns": float(p50) if isinstance(p50, int | float) else None,
        "estimated_flops": row.get("estimated_flops"),
        "flops_per_second": _row_flops_per_second(row),
        "shape_bucket": row.get("shape_bucket") if isinstance(row.get("shape_bucket"), dict) else {},
    }


def _collect_summary(
    *,
    route_manifest: dict[str, Any],
    rows: Sequence[dict[str, Any]],
    run_dir: Path,
    result_path: Path,
) -> dict[str, Any]:
    timed = _timed_rows(rows)
    flops_rows = [row for row in timed if _row_flops_per_second(row) is not None]
    return {
        "schema": SUMMARY_SCHEMA,
        "timestamp": _timestamp(),
        "run_root": str(run_dir),
        "result_path": str(result_path),
        "op": route_manifest.get("op"),
        "route_id": route_manifest.get("route_id"),
        "implementation_id": route_manifest.get("implementation_id"),
        "candidate_name": next((row.get("candidate_name") for row in rows if row.get("candidate_name")), "default"),
        "case_count": len(rows),
        "passed_count": sum(1 for row in rows if row.get("status") == "pass"),
        "failed_count": sum(1 for row in rows if row.get("status") != "pass"),
        "timed_count": len(timed),
        "total_estimated_flops": sum(
            int(row["estimated_flops"]) for row in rows if isinstance(row.get("estimated_flops"), int)
        ),
        "fastest_by_p50_ns": [_case_brief(row) for row in sorted(timed, key=_row_p50)[:10]],
        "slowest_by_p50_ns": [_case_brief(row) for row in sorted(timed, key=_row_p50, reverse=True)[:10]],
        "fastest_by_flops_per_second": [
            _case_brief(row)
            for row in sorted(
                flops_rows,
                key=lambda item: _row_flops_per_second(item) or 0,
                reverse=True,
            )[:10]
        ],
        "slowest_by_flops_per_second": [
            _case_brief(row)
            for row in sorted(flops_rows, key=lambda item: _row_flops_per_second(item) or 0)[:10]
        ],
        "failed_cases": [
            {
                "case_id": row.get("case_id"),
                "returncode": row.get("process_returncode"),
                "stderr": str(row.get("stderr") or "").strip().splitlines()[:5],
            }
            for row in rows
            if row.get("status") != "pass"
        ],
    }


def _collect_markdown(summary: dict[str, Any], rows: Sequence[dict[str, Any]]) -> str:
    lines = [
        "# Loom Kernel Benchmark Summary",
        "",
        f"- Route: `{summary.get('route_id')}`",
        f"- Candidate: `{summary.get('candidate_name')}`",
        f"- Cases: `{summary.get('case_count')}`",
        f"- Passed: `{summary.get('passed_count')}`",
        f"- Failed: `{summary.get('failed_count')}`",
        "",
        "| Case | Status | p50 ns | Estimated FLOPs | FLOP/s | Shape |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        timing = None
        benchmark_summary = row.get("benchmark_summary")
        if isinstance(benchmark_summary, dict):
            timing = benchmark_summary.get("operation_timing_ns")
        p50 = timing.get("p50") if isinstance(timing, dict) else None
        flop_s = _row_flops_per_second(row)
        shape = row.get("shape_bucket") if isinstance(row.get("shape_bucket"), dict) else {}
        shape_text = ",".join(
            f"{key}={shape.get(key)}"
            for key in ("k", "rows", "cols", "batch_product", "layout_kind", "flop_bucket")
        )
        flop_s_text = f"{flop_s:.3f}" if flop_s is not None else "n/a"
        lines.append(
            f"| `{row.get('case_id')}` | `{row.get('status')}` | "
            f"{p50 if isinstance(p50, int | float) else 'n/a'} | "
            f"{row.get('estimated_flops') if row.get('estimated_flops') is not None else 'n/a'} | "
            f"{flop_s_text} | `{shape_text}` |"
        )
    return "\n".join(lines) + "\n"


def command_collect(args: argparse.Namespace) -> int:
    route_manifest = _load_json(args.manifest.resolve())
    if route_manifest.get("schema") != SCRIPT_ROUTE_MANIFEST_SCHEMA:
        raise RuntimeError(f"unsupported route manifest schema in {args.manifest}")
    run_dir = args.run_dir.resolve()
    case_rows: list[dict[str, Any]] = []
    cases = route_manifest.get("cases")
    if not isinstance(cases, list):
        raise RuntimeError("route manifest cases must be a list")
    for case_entry in cases:
        if not isinstance(case_entry, dict):
            raise RuntimeError("route manifest case entries must be objects")
        case_manifest_path = Path(str(case_entry["manifest_path"]))
        case_manifest = _load_json(case_manifest_path)
        case_rows.append(
            _collect_case_result(
                route_manifest=route_manifest,
                case_manifest=case_manifest,
                run_dir=run_dir,
                candidate_name=args.candidate_name,
            )
        )

    result_path = args.results.resolve() if args.results else run_dir / "results.jsonl"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in case_rows) + "\n", encoding="utf-8")
    summary = _collect_summary(
        route_manifest=route_manifest,
        rows=case_rows,
        run_dir=run_dir,
        result_path=result_path,
    )
    output_path = args.output.resolve() if args.output else run_dir / "summary.json"
    _write_json_file(output_path, summary)
    if args.markdown:
        args.markdown.resolve().write_text(_collect_markdown(summary, case_rows), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["failed_count"] == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect results from a generated Loom route benchmark run."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--candidate-name", default="default")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return command_collect(args)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
