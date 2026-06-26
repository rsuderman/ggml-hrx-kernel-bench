from __future__ import annotations

from collections import OrderedDict, defaultdict
from typing import Any

from .reporting import timing_ns


SCHEMA = "ggml_hrx_kernel_bench.reduced_routes.v1"


def reduce_routes(catalog_rows: list[dict[str, Any]]) -> OrderedDict[str, Any]:
    accepted: list[OrderedDict[str, Any]] = []
    rejected: list[OrderedDict[str, Any]] = []
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in catalog_rows:
        candidate = row.get("candidate") if isinstance(row.get("candidate"), dict) else {}
        by_family[str(candidate.get("family") or "unknown")].append(row)

    for family, rows in sorted(by_family.items()):
        ready = [row for row in rows if row.get("catalog_ready")]
        if not ready:
            rejected.append(
                OrderedDict(
                    [
                        ("family", family),
                        ("reason", "no_catalog_ready_candidates"),
                        ("candidate_count", len(rows)),
                    ]
                )
            )
            continue
        for row in sorted(ready, key=_row_sort_key):
            accepted.append(_accepted_route(row))

    return OrderedDict(
        [
            ("schema", SCHEMA),
            ("summary", _summary(accepted, rejected, catalog_rows)),
            ("accepted", accepted),
            ("rejected", rejected),
        ]
    )


def _accepted_route(row: dict[str, Any]) -> OrderedDict[str, Any]:
    candidate = row["candidate"]
    route = candidate.get("route") if isinstance(candidate.get("route"), dict) else {}
    schedule = candidate.get("schedule") if isinstance(candidate.get("schedule"), dict) else {}
    compile_row = row.get("compile") if isinstance(row.get("compile"), dict) else {}
    benchmark = row.get("benchmark") if isinstance(row.get("benchmark"), dict) else {}
    return OrderedDict(
        [
            ("family", candidate.get("family")),
            ("route_id", candidate.get("route_id")),
            ("candidate_id", candidate.get("candidate_id")),
            ("source_id", candidate.get("source_id")),
            ("root_symbol", candidate.get("root_symbol")),
            ("shape", candidate.get("shape") or {}),
            ("shape_domain", route.get("shape_domain") or {}),
            ("shape_guards", route.get("shape_guards") or {}),
            ("schedule", schedule),
            ("config_bindings", candidate.get("config_bindings") or {}),
            ("supports", candidate.get("supports") or {}),
            ("timing_ns", timing_ns(_catalog_row_as_ledger_row(row))),
            ("compile", _compile_summary(compile_row)),
            ("benchmark", _benchmark_summary(benchmark)),
        ]
    )


def _compile_summary(row: dict[str, Any]) -> OrderedDict[str, Any]:
    return OrderedDict(
        [
            ("report", row.get("report")),
            ("manifest", row.get("manifest")),
            ("target_artifact", row.get("target_artifact")),
            ("target_artifact_bytes", row.get("target_artifact_bytes")),
            ("report_summary", row.get("report_summary") or {}),
        ]
    )


def _benchmark_summary(row: dict[str, Any]) -> OrderedDict[str, Any]:
    return OrderedDict(
        [
            ("results_path", row.get("results_path")),
            ("artifact_bundle_dir", row.get("artifact_bundle_dir")),
            ("summary", row.get("summary") or {}),
        ]
    )


def _catalog_row_as_ledger_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate": row.get("candidate"),
        "benchmark": row.get("benchmark"),
        "compile": row.get("compile"),
        "status": "ran" if row.get("benchmark") else "compiled",
    }


def _row_sort_key(row: dict[str, Any]) -> tuple[str, float, str]:
    candidate = row.get("candidate") if isinstance(row.get("candidate"), dict) else {}
    family = str(candidate.get("family") or "")
    timing = timing_ns(_catalog_row_as_ledger_row(row))
    return (family, timing if timing is not None else float("inf"), str(candidate.get("candidate_id") or ""))


def _summary(
    accepted: list[OrderedDict[str, Any]],
    rejected: list[OrderedDict[str, Any]],
    catalog_rows: list[dict[str, Any]],
) -> OrderedDict[str, Any]:
    families = sorted(
        {
            str((row.get("candidate") if isinstance(row.get("candidate"), dict) else {}).get("family") or "unknown")
            for row in catalog_rows
        }
    )
    return OrderedDict(
        [
            ("family_count", len(families)),
            ("candidate_count", len(catalog_rows)),
            ("accepted_count", len(accepted)),
            ("rejected_family_count", len(rejected)),
            ("families", families),
        ]
    )
