from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

from .common import COMPARE_SCHEMA, load_json as _load_json, timestamp as _timestamp
from .discovery import _flop_bucket
from .result_parsing import _compile_summary, _load_jsonl_objects, _operation_timing_summary

def _candidate_spec(value: str) -> tuple[str, Path]:
    if "=" in value:
        name, path = value.split("=", 1)
        if not name or not path:
            raise RuntimeError(f"invalid candidate spec {value!r}; expected NAME=PATH or PATH")
        return name, Path(path)
    path = Path(value)
    return path.parent.name or path.stem, path


def _resolve_benchmark_output_path(raw_path: object, *, result_path: Path) -> Path | None:
    if not isinstance(raw_path, str) or not raw_path:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (result_path.parent / path).resolve()


def _timing_for_result_row(row: dict[str, Any], *, result_path: Path) -> dict[str, Any] | None:
    benchmark_summary = row.get("benchmark_summary")
    if isinstance(benchmark_summary, dict) and isinstance(benchmark_summary.get("operation_timing_ns"), dict):
        return benchmark_summary["operation_timing_ns"]
    output_path = _resolve_benchmark_output_path(row.get("benchmark_output_path"), result_path=result_path)
    if output_path is None:
        return None
    benchmark_rows = [item for item in _load_jsonl_objects(output_path) if item.get("row") == "benchmark"]
    return _operation_timing_summary(benchmark_rows)


def _compile_summary_for_result_row(row: dict[str, Any], *, result_path: Path) -> dict[str, Any] | None:
    compile_summary = row.get("compile_summary")
    if isinstance(compile_summary, dict):
        return compile_summary
    benchmark_summary = row.get("benchmark_summary")
    if isinstance(benchmark_summary, dict) and isinstance(benchmark_summary.get("compile_summary"), dict):
        return benchmark_summary["compile_summary"]
    output_path = _resolve_benchmark_output_path(row.get("benchmark_output_path"), result_path=result_path)
    if output_path is None:
        return None
    return _compile_summary(_load_jsonl_objects(output_path))


def _measurement_rows(result_path: Path, *, candidate_name: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in _load_jsonl_objects(result_path):
        case_id = row.get("case_id")
        if not isinstance(case_id, str):
            continue
        timing = _timing_for_result_row(row, result_path=result_path)
        rows[case_id] = {
            "candidate_name": row.get("candidate_name") or candidate_name,
            "case_id": case_id,
            "status": row.get("status"),
            "op": row.get("op"),
            "route_id": row.get("route_id"),
            "estimated_flops": row.get("estimated_flops"),
            "shape_bucket": row.get("shape_bucket") if isinstance(row.get("shape_bucket"), dict) else {},
            "operation_timing_ns": timing,
            "compile_summary": _compile_summary_for_result_row(row, result_path=result_path),
            "source": {
                "effective_kernel_source": row.get("effective_kernel_source"),
                "effective_kernel_source_hash": row.get("effective_kernel_source_hash"),
                "descriptor_kernel_source": row.get("descriptor_kernel_source"),
                "descriptor_kernel_source_hash": row.get("descriptor_kernel_source_hash"),
                "source_override_used": row.get("source_override_used"),
            },
        }
    return rows


def _geomean(values: Sequence[float]) -> float | None:
    positives = [value for value in values if value > 0]
    if not positives:
        return None
    return math.exp(sum(math.log(value) for value in positives) / len(positives))


def _median(values: Sequence[float]) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def _group_value(row: dict[str, Any], group_by: str) -> str:
    bucket = row.get("shape_bucket")
    bucket = bucket if isinstance(bucket, dict) else {}
    if group_by == "layout":
        return str(bucket.get("layout_kind", "unknown"))
    if group_by == "flop-bucket":
        return str(bucket.get("flop_bucket") or _flop_bucket(row.get("estimated_flops")))
    if group_by in {"k", "rows", "cols", "batch_product", "dtype_family"}:
        return str(bucket.get(group_by, "unknown"))
    return "all"


def _group_key(row: dict[str, Any], group_bys: Sequence[str]) -> str:
    if not group_bys:
        return "all"
    return "|".join(f"{group}={_group_value(row, group)}" for group in group_bys)


def _compare_measurement_maps(
    *,
    baseline: dict[str, dict[str, Any]],
    candidates: Sequence[tuple[str, str, dict[str, dict[str, Any]]]],
    threshold: float,
    group_by: Sequence[str] = (),
) -> list[dict[str, Any]]:
    candidate_results: list[dict[str, Any]] = []
    for candidate_name, candidate_path, candidate in candidates:
        case_rows: list[dict[str, Any]] = []
        missing_cases: list[str] = []
        untimed_cases: list[str] = []
        for case_id, base_row in sorted(baseline.items()):
            candidate_row = candidate.get(case_id)
            if candidate_row is None:
                missing_cases.append(case_id)
                continue
            base_timing = base_row.get("operation_timing_ns")
            candidate_timing = candidate_row.get("operation_timing_ns")
            if not isinstance(base_timing, dict) or not isinstance(candidate_timing, dict):
                untimed_cases.append(case_id)
                continue
            base_mean = base_timing.get("mean")
            candidate_mean = candidate_timing.get("mean")
            if not isinstance(base_mean, int | float) or not isinstance(candidate_mean, int | float) or base_mean <= 0:
                untimed_cases.append(case_id)
                continue
            ratio = float(candidate_mean) / float(base_mean)
            case_rows.append(
                {
                    "case_id": case_id,
                    "op": candidate_row.get("op") or base_row.get("op"),
                    "route_id": candidate_row.get("route_id") or base_row.get("route_id"),
                    "baseline_mean_ns": base_mean,
                    "candidate_mean_ns": candidate_mean,
                    "time_ratio": ratio,
                    "estimated_flops": candidate_row.get("estimated_flops") or base_row.get("estimated_flops"),
                    "shape_bucket": candidate_row.get("shape_bucket") or base_row.get("shape_bucket"),
                    "group": _group_key(candidate_row if candidate_row.get("shape_bucket") else base_row, group_by),
                    "baseline_status": base_row.get("status"),
                    "candidate_status": candidate_row.get("status"),
                    "compile_summary": candidate_row.get("compile_summary"),
                    "source": candidate_row.get("source"),
                }
            )
        ratios = [row["time_ratio"] for row in case_rows]
        groups: dict[str, list[float]] = {}
        for row in case_rows:
            groups.setdefault(str(row["group"]), []).append(float(row["time_ratio"]))
        group_summaries = {
            name: {
                "case_count": len(values),
                "geomean_time_ratio": _geomean(values),
                "median_time_ratio": _median(values),
                "wins": sum(1 for value in values if value < 1.0 - threshold),
                "neutral": sum(1 for value in values if 1.0 - threshold <= value <= 1.0 + threshold),
                "losses": sum(1 for value in values if value > 1.0 + threshold),
            }
            for name, values in sorted(groups.items())
        }
        failed_candidate_cases = [
            case_id for case_id, row in sorted(candidate.items()) if row.get("status") not in {"pass", None}
        ]
        candidate_results.append(
            {
                "candidate_name": candidate_name,
                "candidate_path": str(candidate_path),
                "case_count": len(case_rows),
                "missing_case_count": len(missing_cases),
                "missing_cases": missing_cases,
                "untimed_case_count": len(untimed_cases),
                "untimed_cases": untimed_cases,
                "failed_candidate_cases": failed_candidate_cases,
                "geomean_time_ratio": _geomean(ratios),
                "median_time_ratio": _median(ratios),
                "wins": sum(1 for value in ratios if value < 1.0 - threshold),
                "neutral": sum(1 for value in ratios if 1.0 - threshold <= value <= 1.0 + threshold),
                "losses": sum(1 for value in ratios if value > 1.0 + threshold),
                "best_improvements": sorted(case_rows, key=lambda row: row["time_ratio"])[:10],
                "worst_regressions": sorted(case_rows, key=lambda row: row["time_ratio"], reverse=True)[:10],
                "groups": group_summaries,
            }
        )
    return candidate_results


def compare_result_sets(
    *,
    baseline_path: Path,
    candidates: Sequence[tuple[str, Path]],
    threshold: float,
    group_by: Sequence[str] = (),
) -> dict[str, Any]:
    baseline = _measurement_rows(baseline_path, candidate_name="baseline")
    candidate_results = _compare_measurement_maps(
        baseline=baseline,
        candidates=[
            (candidate_name, str(candidate_path), _measurement_rows(candidate_path, candidate_name=candidate_name))
            for candidate_name, candidate_path in candidates
        ],
        threshold=threshold,
        group_by=group_by,
    )
    return {
        "schema": COMPARE_SCHEMA,
        "timestamp": _timestamp(),
        "baseline_path": str(baseline_path),
        "baseline_case_count": len(baseline),
        "threshold": threshold,
        "group_by": list(group_by),
        "candidates": candidate_results,
    }


def _print_compare_summary(compare: dict[str, Any]) -> None:
    print("candidate               cases  geomean  median   wins  neutral  losses  missing  untimed")
    for candidate in compare["candidates"]:
        geomean = candidate.get("geomean_time_ratio")
        median = candidate.get("median_time_ratio")
        print(
            f"{str(candidate['candidate_name'])[:22]:22} "
            f"{candidate['case_count']:5d} "
            f"{geomean if isinstance(geomean, float) else float('nan'):8.4f} "
            f"{median if isinstance(median, float) else float('nan'):8.4f} "
            f"{candidate['wins']:6d} "
            f"{candidate['neutral']:8d} "
            f"{candidate['losses']:7d} "
            f"{candidate['missing_case_count']:8d} "
            f"{candidate['untimed_case_count']:7d}"
        )


def _load_acceptance_policy(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    policy = _load_json(path)
    allowed = policy.get("allowed_regression_cases")
    if allowed is not None and not isinstance(allowed, list):
        raise RuntimeError("policy allowed_regression_cases must be a list")
    return policy


def _acceptance_limits(args: argparse.Namespace, policy: dict[str, Any]) -> dict[str, Any]:
    return {
        "max_geomean_regression": policy.get(
            "max_geomean_regression",
            getattr(args, "fail_on_geomean_regression", None),
        ),
        "max_case_regression": policy.get(
            "max_case_regression",
            getattr(args, "fail_on_case_regression", None),
        ),
        "fail_on_correctness_failure": bool(
            policy.get(
                "fail_on_correctness_failure",
                getattr(args, "fail_on_correctness_failure", False),
            )
        ),
        "allowed_regression_cases": [
            str(case_id) for case_id in policy.get("allowed_regression_cases", []) if isinstance(case_id, str)
        ],
    }


def _apply_acceptance_policy(compare: dict[str, Any], limits: dict[str, Any]) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    allowed_cases = set(limits["allowed_regression_cases"])
    max_geomean = limits.get("max_geomean_regression")
    max_case = limits.get("max_case_regression")
    for candidate in compare["candidates"]:
        name = candidate["candidate_name"]
        if limits["fail_on_correctness_failure"] and candidate["failed_candidate_cases"]:
            failures.append(
                {
                    "candidate_name": name,
                    "kind": "correctness",
                    "failed_cases": candidate["failed_candidate_cases"],
                }
            )
        geomean = candidate.get("geomean_time_ratio")
        if isinstance(max_geomean, int | float) and isinstance(geomean, float) and geomean > 1.0 + max_geomean:
            failures.append(
                {
                    "candidate_name": name,
                    "kind": "geomean_regression",
                    "limit": max_geomean,
                    "time_ratio": geomean,
                }
            )
        if isinstance(max_case, int | float):
            regressions = [
                row
                for row in candidate["worst_regressions"]
                if row["case_id"] not in allowed_cases and row["time_ratio"] > 1.0 + max_case
            ]
            if regressions:
                failures.append(
                    {
                        "candidate_name": name,
                        "kind": "case_regression",
                        "limit": max_case,
                        "case_count": len(regressions),
                        "worst": regressions[0],
                    }
                )
    return {
        "passed": not failures,
        "failures": failures,
        "limits": limits,
    }


def command_compare(args: argparse.Namespace) -> int:
    policy_path = getattr(args, "policy", None)
    policy = _load_acceptance_policy(policy_path.resolve() if policy_path else None)
    threshold = float(policy.get("threshold", args.threshold))
    candidates = [_candidate_spec(value) for value in args.candidate]
    compare = compare_result_sets(
        baseline_path=args.baseline.resolve(),
        candidates=[(name, path.resolve()) for name, path in candidates],
        threshold=threshold,
        group_by=args.group_by,
    )
    compare["policy_path"] = str(policy_path) if policy_path else None
    compare["acceptance"] = _apply_acceptance_policy(compare, _acceptance_limits(args, policy))
    output_path = getattr(args, "output", None)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(compare, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(compare, indent=2, sort_keys=True))
    else:
        _print_compare_summary(compare)
    return 0 if compare["acceptance"]["passed"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare collected Loom benchmark result files.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="candidate results as PATH or NAME=PATH",
    )
    parser.add_argument("--threshold", type=float, default=0.02)
    parser.add_argument(
        "--group-by",
        action="append",
        default=[],
        choices=["k", "rows", "cols", "batch_product", "layout", "flop-bucket", "dtype_family"],
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", type=Path, help="write compare JSON to this path")
    parser.add_argument("--policy", type=Path, help="acceptance policy JSON")
    parser.add_argument("--fail-on-geomean-regression", type=float)
    parser.add_argument("--fail-on-case-regression", type=float)
    parser.add_argument("--fail-on-correctness-failure", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return command_compare(args)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
