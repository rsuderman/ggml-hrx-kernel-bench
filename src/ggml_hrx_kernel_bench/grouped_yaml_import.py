from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import yaml

from .family_specs import ShapeDomain, normalize_shape
from .hrx2 import iter_routes
from .import_mapping_registry import (
    IMPORT_MAPPING_RULES,
    compatible_rules_for_op,
    compatible_rules_for_op_dtype,
    match_rules,
)
from .import_models import (
    ImportedCase,
    ImportedOpGroup,
    ImportedSuite,
    MappingStatus,
    ResolvedBenchmarkCase,
    UnmappedCase,
    UnmappedReason,
)


IMPORTED_WORKLOAD_SCHEMA = "ggml_hrx_kernel_bench.imported_workload.v1"
UNMAPPED_CASES_SCHEMA = "ggml_hrx_kernel_bench.import_unmapped_cases.v1"


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _normalize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _normalize_value(current)
            for key, current in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, list):
        return [_normalize_value(current) for current in value]
    return value


def _dtype_label(dtype: dict[str, Any]) -> str:
    parts = [f"{key}={value}" for key, value in sorted(dtype.items())]
    return ", ".join(parts) if parts else "unknown"


def _safe_name(name: str) -> str:
    return "".join(
        char if char.isalnum() or char in ("-", "_") else "_" for char in name
    )


def load_grouped_yaml_suite(path: Path) -> ImportedSuite:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    _expect(isinstance(data, dict), f"grouped YAML must contain a top-level mapping: {path}")
    ops = data.get("ops")
    _expect(isinstance(ops, dict), f"grouped YAML must contain an ops mapping: {path}")

    op_groups: list[ImportedOpGroup] = []
    for op_name, raw_groups in ops.items():
        _expect(isinstance(op_name, str) and op_name, "op names must be non-empty strings")
        _expect(isinstance(raw_groups, list), f"ops.{op_name} must be a list")
        for group_index, raw_group in enumerate(raw_groups):
            _expect(
                isinstance(raw_group, dict),
                f"ops.{op_name}[{group_index}] must be a mapping",
            )
            dtype = _normalize_value(raw_group.get("dtype") or {})
            _expect(
                isinstance(dtype, dict),
                f"ops.{op_name}[{group_index}].dtype must be a mapping",
            )
            raw_cases = raw_group.get("cases")
            _expect(
                isinstance(raw_cases, list),
                f"ops.{op_name}[{group_index}].cases must be a list",
            )
            cases: list[ImportedCase] = []
            for case_index, raw_case in enumerate(raw_cases):
                _expect(
                    isinstance(raw_case, dict),
                    f"ops.{op_name}[{group_index}].cases[{case_index}] must be a mapping",
                )
                normalized_case = _normalize_value(raw_case)
                cases.append(
                    ImportedCase(
                        op=op_name,
                        dtype=dtype,
                        raw_case=normalized_case,
                        normalized_params=normalized_case,
                        source_path=str(path),
                        source_group_index=group_index,
                        source_case_index=case_index,
                    )
                )
            op_groups.append(
                ImportedOpGroup(
                    op=op_name,
                    dtype=dtype,
                    source_path=str(path),
                    cases=tuple(cases),
                )
            )

    return ImportedSuite(
        schema=IMPORTED_WORKLOAD_SCHEMA,
        source_path=str(path),
        op_groups=op_groups,
    )


def split_suite_by_op(suite: ImportedSuite) -> dict[str, ImportedSuite]:
    grouped: dict[str, list[ImportedOpGroup]] = defaultdict(list)
    for op_group in suite.op_groups:
        grouped[op_group.op].append(op_group)
    return {
        op_name: ImportedSuite(
            schema=suite.schema,
            source_path=suite.source_path,
            op_groups=list(op_groups),
        )
        for op_name, op_groups in sorted(grouped.items())
    }


def _routes_by_family(catalog_dir: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for route in iter_routes(catalog_dir):
        family = str(route.get("family") or route.get("source_id") or "")
        if family:
            grouped[family].append(route)
    return grouped


def _shape_from_resolved(params: list[str], values: list[int]) -> dict[str, int]:
    return normalize_shape(dict(zip(params, values, strict=True)))


def _route_accepts_shape(route: dict[str, Any], shape: dict[str, int]) -> bool:
    domain = route.get("shape_domain") or {}
    if not isinstance(domain, dict) or not domain:
        return True
    guards = route.get("shape_guards") or {}
    ctx = ShapeDomain(
        family=str(route.get("family") or route.get("source_id") or ""),
        route_id=str(route.get("id") or ""),
        root_symbol=str(route.get("root_symbol") or ""),
        domain=domain,
        guards=guards if isinstance(guards, dict) else {},
    )
    return ctx.accepts(shape)


def _resolve_route(
    kernel_family: str,
    params: list[str],
    values: list[int],
    routes_by_family: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], UnmappedReason | None]:
    family_routes = list(routes_by_family.get(kernel_family, ()))
    if not family_routes:
        return None, [], UnmappedReason.NO_ROUTE_MATCH
    shape = _shape_from_resolved(params, values)
    matching = [route for route in family_routes if _route_accepts_shape(route, shape)]
    if not matching:
        return None, family_routes, UnmappedReason.NO_ROUTE_MATCH
    if len(matching) > 1:
        return None, matching, UnmappedReason.AMBIGUOUS_ROUTE_MATCH
    return matching[0], matching, None


def _candidate_kernel_families(rules: Iterable[Any]) -> tuple[str, ...]:
    families = sorted({str(rule.kernel_family) for rule in rules})
    return tuple(families)


def _unmapped_case(
    case: ImportedCase,
    *,
    status: MappingStatus,
    reason: UnmappedReason,
    detail: str | None = None,
    candidate_kernel_families: tuple[str, ...] = (),
    candidate_route_ids: tuple[str, ...] = (),
) -> UnmappedCase:
    return UnmappedCase(
        imported=case,
        mapping_status=status,
        reason=reason,
        detail=detail,
        candidate_kernel_families=candidate_kernel_families,
        candidate_route_ids=candidate_route_ids,
    )


def resolve_imported_suite(suite: ImportedSuite, *, catalog_dir: Path) -> ImportedSuite:
    routes_by_family = _routes_by_family(catalog_dir)
    resolved: list[ResolvedBenchmarkCase] = []
    unmapped: list[UnmappedCase] = []

    for group in suite.op_groups:
        for case in group.cases:
            strict_matches = match_rules(case, IMPORT_MAPPING_RULES)
            if not strict_matches:
                dtype_matches = compatible_rules_for_op_dtype(case, IMPORT_MAPPING_RULES)
                if dtype_matches:
                    unmapped.append(
                        _unmapped_case(
                            case,
                            status=MappingStatus.UNMAPPED,
                            reason=UnmappedReason.SHAPE_LOWERING_NOT_IMPLEMENTED,
                            detail=(
                                "matching op/dtype rule exists, but this case does not satisfy "
                                "its supported layout constraints"
                            ),
                            candidate_kernel_families=_candidate_kernel_families(
                                dtype_matches
                            ),
                        )
                    )
                    continue
                op_matches = compatible_rules_for_op(case, IMPORT_MAPPING_RULES)
                if op_matches:
                    unmapped.append(
                        _unmapped_case(
                            case,
                            status=MappingStatus.UNMAPPED,
                            reason=UnmappedReason.NO_DTYPE_MAPPING,
                            detail=(
                                "matching op mapping exists, but not for this dtype "
                                "combination"
                            ),
                            candidate_kernel_families=_candidate_kernel_families(
                                op_matches
                            ),
                        )
                    )
                    continue
                unmapped.append(
                    _unmapped_case(
                        case,
                        status=MappingStatus.UNMAPPED,
                        reason=UnmappedReason.NO_KERNEL_FAMILY_MAPPING,
                        detail="no import mapping rule exists for this op",
                    )
                )
                continue

            if len(strict_matches) > 1:
                unmapped.append(
                    _unmapped_case(
                        case,
                        status=MappingStatus.AMBIGUOUS,
                        reason=UnmappedReason.AMBIGUOUS_ROUTE_MATCH,
                        detail="multiple import mapping rules matched this case",
                        candidate_kernel_families=_candidate_kernel_families(
                            match.rule for match in strict_matches
                        ),
                    )
                )
                continue

            match = strict_matches[0]
            try:
                params, values = match.rule.lowering(case)
            except (NotImplementedError, ValueError) as exc:
                unmapped.append(
                    _unmapped_case(
                        case,
                        status=MappingStatus.UNMAPPED,
                        reason=UnmappedReason.SHAPE_LOWERING_NOT_IMPLEMENTED,
                        detail=str(exc),
                        candidate_kernel_families=(match.rule.kernel_family,),
                    )
                )
                continue

            route, route_candidates, route_reason = _resolve_route(
                match.rule.kernel_family,
                params,
                values,
                routes_by_family,
            )
            if route_reason is not None:
                status = (
                    MappingStatus.AMBIGUOUS
                    if route_reason == UnmappedReason.AMBIGUOUS_ROUTE_MATCH
                    else MappingStatus.UNMAPPED
                )
                unmapped.append(
                    _unmapped_case(
                        case,
                        status=status,
                        reason=route_reason,
                        detail=(
                            f"could not resolve a unique route for kernel family "
                            f"{match.rule.kernel_family}"
                        ),
                        candidate_kernel_families=(match.rule.kernel_family,),
                        candidate_route_ids=tuple(
                            str(candidate.get("id") or "")
                            for candidate in route_candidates
                            if candidate.get("id")
                        ),
                    )
                )
                continue

            resolved.append(
                ResolvedBenchmarkCase(
                    imported=case,
                    kernel_family=match.rule.kernel_family,
                    route_id=str(route.get("id") or ""),
                    params=list(params),
                    values=list(values),
                )
            )

    suite.resolved = resolved
    suite.unmapped = unmapped
    return suite


def emit_compact_configs(suite: ImportedSuite, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[str, str | None, tuple[str, ...]], list[ResolvedBenchmarkCase]] = defaultdict(list)
    for row in suite.resolved:
        grouped[(row.kernel_family, row.route_id, tuple(row.params))].append(row)

    emitted: list[Path] = []
    for (kernel_family, route_id, params_key), rows in sorted(
        grouped.items(),
        key=lambda item: (item[0][0], item[0][1] or "", item[0][2]),
    ):
        cases: list[list[int]] = []
        seen_cases: set[tuple[int, ...]] = set()
        for row in rows:
            key = tuple(row.values)
            if key in seen_cases:
                continue
            seen_cases.add(key)
            cases.append(list(row.values))
        payload: dict[str, Any] = {
            "kernel": kernel_family,
            "params": list(params_key),
            "cases": cases,
        }
        if route_id:
            payload["route_id"] = route_id
        filename = (
            f"{kernel_family}.{route_id}.json" if route_id else f"{kernel_family}.json"
        )
        path = output_dir / filename
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        emitted.append(path)
    return emitted


def _status_counts(suite: ImportedSuite) -> dict[str, int]:
    counts = Counter(row.mapping_status.value for row in suite.unmapped)
    counts[MappingStatus.MAPPED.value] = len(suite.resolved)
    return {
        MappingStatus.MAPPED.value: counts.get(MappingStatus.MAPPED.value, 0),
        MappingStatus.UNMAPPED.value: counts.get(MappingStatus.UNMAPPED.value, 0),
        MappingStatus.AMBIGUOUS.value: counts.get(MappingStatus.AMBIGUOUS.value, 0),
    }


def summary_payload(suite: ImportedSuite, config_paths: list[Path]) -> dict[str, Any]:
    counts = _status_counts(suite)
    total_cases = len(suite.resolved) + len(suite.unmapped)
    return {
        "source_path": suite.source_path,
        "total_cases": total_cases,
        "mapped_case_count": counts[MappingStatus.MAPPED.value],
        "unmapped_case_count": counts[MappingStatus.UNMAPPED.value],
        "ambiguous_case_count": counts[MappingStatus.AMBIGUOUS.value],
        "generated_config_count": len(config_paths),
    }


def write_imported_workload_json(suite: ImportedSuite, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(suite.to_json(), indent=2) + "\n", encoding="utf-8")


def write_unmapped_json(suite: ImportedSuite, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": UNMAPPED_CASES_SCHEMA,
        "source_path": suite.source_path,
        "case_count": len(suite.unmapped),
        "rows": [row.to_json() for row in suite.unmapped],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _markdown_summary(suite: ImportedSuite, config_paths: list[Path]) -> str:
    counts = summary_payload(suite, config_paths)
    reason_counts = Counter(row.reason.value for row in suite.unmapped)
    lines = [
        f"# Grouped YAML Import: {Path(suite.source_path).name}",
        "",
        f"- Source: `{suite.source_path}`",
        f"- Total cases: `{counts['total_cases']}`",
        f"- Mapped: `{counts['mapped_case_count']}`",
        f"- Unmapped: `{counts['unmapped_case_count']}`",
        f"- Ambiguous: `{counts['ambiguous_case_count']}`",
        f"- Generated benchmark configs: `{counts['generated_config_count']}`",
        "",
        "## Generated Configs",
        "",
    ]
    if config_paths:
        lines.extend([
            "| Config | Cases |",
            "| --- | ---: |",
        ])
        for path in sorted(config_paths):
            payload = json.loads(path.read_text(encoding="utf-8"))
            lines.append(f"| `{path.name}` | {len(payload.get('cases', []))} |")
    else:
        lines.append("No benchmark configs were generated.")
    lines.extend(["", "## Unmapped Reasons", ""])
    if reason_counts:
        lines.extend([
            "| Reason | Count |",
            "| --- | ---: |",
        ])
        for reason, count in sorted(reason_counts.items()):
            lines.append(f"| `{reason}` | {count} |")
    else:
        lines.append("No unmapped cases.")
    lines.extend(["", "## Case Details", ""])
    if suite.unmapped:
        lines.extend([
            "| Op | DType | Status | Reason | Group | Case | Detail |",
            "| --- | --- | --- | --- | ---: | ---: | --- |",
        ])
        for row in suite.unmapped:
            imported = row.imported
            detail = row.detail or ""
            lines.append(
                "| "
                + " | ".join(
                    [
                        imported.op,
                        _dtype_label(imported.dtype),
                        row.mapping_status.value,
                        row.reason.value,
                        str(imported.source_group_index),
                        str(imported.source_case_index),
                        detail.replace("|", "/"),
                    ]
                )
                + " |"
            )
    else:
        lines.append("All imported cases mapped cleanly.")
    lines.append("")
    return "\n".join(lines)


def write_import_summary_markdown(
    suite: ImportedSuite,
    config_paths: list[Path],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_markdown_summary(suite, config_paths), encoding="utf-8")


def materialize_suite_bundle(
    suite: ImportedSuite,
    *,
    output_dir: Path,
    catalog_dir: Path,
) -> dict[str, Any]:
    config_dir = output_dir / "generated-import-configs"
    resolve_imported_suite(suite, catalog_dir=catalog_dir)
    config_paths = emit_compact_configs(suite, config_dir)
    imported_workload_path = output_dir / "imported-workload.json"
    unmapped_path = output_dir / "unmapped.json"
    summary_markdown_path = output_dir / "import-summary.md"
    write_imported_workload_json(suite, imported_workload_path)
    write_unmapped_json(suite, unmapped_path)
    write_import_summary_markdown(suite, config_paths, summary_markdown_path)
    return {
        **summary_payload(suite, config_paths),
        "output_dir": str(output_dir),
        "imported_workload_path": str(imported_workload_path),
        "unmapped_path": str(unmapped_path),
        "summary_markdown_path": str(summary_markdown_path),
        "generated_config_paths": [str(path) for path in config_paths],
    }


def materialize_grouped_yaml(
    yaml_path: Path,
    *,
    output_dir: Path,
    catalog_dir: Path,
    split_by_op: bool,
) -> dict[str, Any]:
    suite = load_grouped_yaml_suite(yaml_path)
    if not split_by_op:
        return materialize_suite_bundle(suite, output_dir=output_dir, catalog_dir=catalog_dir)

    operations_dir = output_dir / "ops"
    operations_dir.mkdir(parents=True, exist_ok=True)
    operation_payloads: dict[str, dict[str, Any]] = {}
    for op_name, op_suite in split_suite_by_op(suite).items():
        op_output_dir = operations_dir / _safe_name(op_name)
        operation_payloads[op_name] = materialize_suite_bundle(
            op_suite,
            output_dir=op_output_dir,
            catalog_dir=catalog_dir,
        )

    payload: dict[str, Any] = {
        "source_path": str(yaml_path),
        "output_dir": str(output_dir),
        "operation_count": len(operation_payloads),
        "operations": operation_payloads,
    }
    index_path = output_dir / "operation-index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    payload["operation_index_path"] = str(index_path)
    return payload
