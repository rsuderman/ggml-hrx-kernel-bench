from __future__ import annotations

import json
import shutil
import subprocess
from collections import OrderedDict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .hrx2 import iter_routes, load_json, read_sources
from .route_schedules import select_test_route, test_scenario_for_family, test_shape_for_route
from .specs import file_sha256


SCHEMA = "ggml-hrx-catalog-v1"
LAYOUT = "ggml-hrx-split-catalog-v1"


_SCHEDULE_FAMILIES = {
    "add_f32",
    "add_rms_norm_mul_f32",
    "argsort_f32_i32",
    "clamp_f32",
    "cont_f32",
    "cont_set_rows_f32",
    "copy_f32_f16",
    "div_f32",
    "get_rows_f32",
    "get_rows_moe_weights_f32",
    "get_rows_q4_k_f32",
    "get_rows_q5_k_f32",
    "get_rows_q6_k_f32",
    "get_rows_q8_0_f32",
    "mul_f32",
    "mul_mat_f32_f32",
    "mul_mat_f16_f32_batched",
    "mul_mat_f16_f32_batched_cont",
    "mul_mat_id_q4_k_f32",
    "mul_mat_id_q5_k_f32",
    "mul_mat_id_q6_k_f32",
    "mul_mat_q4_k_f32",
    "mul_mat_q5_k_f32",
    "mul_mat_q6_k_f32",
    "mul_mat_q8_0_f32",
    "mul_mat_q4_k_swiglu_f32",
    "quantize_q8_1_f32",
    "rms_norm_f32",
    "rms_norm_mul_f32",
    "rope_f32",
    "rope_neox_f32",
    "rope_scale_f32",
    "rope_set_rows_f32",
    "scale_f32",
    "set_rows_f32",
    "soft_max_f32",
    "softmax_kqv_f32_f16",
    "sum_rows_f32",
    "swiglu_f32",
}

_SCHEDULE_TOLERANCE_BY_FAMILY = {
    "get_rows_q4_k_f32": 0.0,
    "get_rows_q5_k_f32": 0.0,
    "get_rows_q6_k_f32": 0.0,
    "get_rows_q8_0_f32": 0.0,
    "mul_mat_q4_k_f32": 8.0e-2,
    "mul_mat_q5_k_f32": 8.0e-2,
    "mul_mat_q6_k_f32": 8.0e-2,
    "mul_mat_q8_0_f32": 8.0e-2,
    "mul_mat_id_q4_k_f32": 8.0e-2,
    "mul_mat_id_q5_k_f32": 8.0e-2,
    "mul_mat_id_q6_k_f32": 8.0e-2,
    "mul_mat_q4_k_swiglu_f32": 1.0e-1,
    "mul_mat_f32_f32": 1.0e-4,
    "mul_mat_f16_f32_batched": 3.0e-2,
    "mul_mat_f16_f32_batched_cont": 3.0e-2,
    "quantize_q8_1_f32": 0.0,
    "rms_norm_f32": 1.0e-4,
    "rms_norm_mul_f32": 3.0e-5,
    "add_rms_norm_mul_f32": 3.0e-5,
    "rope_f32": 1.0e-4,
    "rope_neox_f32": 1.0e-4,
    "rope_scale_f32": 1.0e-4,
    "rope_set_rows_f32": 1.0e-3,
    "soft_max_f32": 1.0e-4,
    "softmax_kqv_f32_f16": 5.0e-2,
    "swiglu_f32": 1.0e-4,
}


_RUNTIME_EXCLUDED_ROUTE_REASONS = {
    "mul_mat_q4_k_q8_1_x4_mmq64x32_k256_8192_r1_32768_c16_wg256": (
        "current loom verifier rejects this root: workgroup barriers are "
        "control-dependent on lane-varying row/column bounds"
    ),
    "mul_mat_id_q4_k_f32_k768_2048_r768_2048_s8_t1_512_wg256": (
        "current loom verifier rejects this root with subrange proof failures "
        "on quantized block vector loads during HRX3 JIT specialization"
    ),
    "mul_mat_id_q5_k_f32_k256_32768_r1_262144_s1_128_t1_512_wg256": (
        "current loom verifier rejects this root with subrange proof failures "
        "on quantized block vector loads during HRX3 JIT specialization"
    ),
    "mul_mat_id_q6_k_f32_k256_32768_r1_262144_s1_128_t1_512_wg256": (
        "current loom verifier rejects this root with subrange proof failures "
        "on quantized block loads during HRX3 JIT specialization"
    ),
}


@dataclass(frozen=True)
class LlamaCatalogExportResult:
    output_dir: Path
    target_key: str
    family_count: int
    source_count: int
    route_count: int
    test_case_count: int
    written_paths: tuple[Path, ...]

    def to_ledger(self) -> dict[str, Any]:
        return {
            "schema": "ggml_hrx_kernel_bench.llama_catalog_export.v1",
            "output_dir": str(self.output_dir),
            "target_key": self.target_key,
            "family_count": self.family_count,
            "source_count": self.source_count,
            "route_count": self.route_count,
            "test_case_count": self.test_case_count,
            "written_paths": [str(path) for path in self.written_paths],
        }


def export_llama_catalog(
    *,
    output_dir: Path,
    kernel_dir: Path,
    catalog_dir: Path,
    target_key: str,
    families: set[str] | None,
    catalog_id: str | None = None,
) -> LlamaCatalogExportResult:
    sources = read_sources(catalog_dir)
    family_rows = _load_families(catalog_dir)
    selected_routes = [
        route
        for route in iter_routes(catalog_dir)
        if _route_selected(route, target_key=target_key, families=families)
    ]
    selected_routes.sort(key=lambda route: (-int(route.get("priority", 0) or 0), str(route.get("id", ""))))

    source_ids = sorted({str(route["source_id"]) for route in selected_routes if route.get("source_id")})
    family_ids = sorted({str(route["family"]) for route in selected_routes if route.get("family")})
    route_status = _route_export_status(catalog_dir, target_key=target_key, families=families)

    missing_sources = [source_id for source_id in source_ids if source_id not in sources]
    if missing_sources:
        raise ValueError("selected routes reference missing source ids: " + ", ".join(missing_sources))

    written: list[Path] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    _clear_generated_catalog_dir(output_dir, target_key)
    for rel in ("kernels", "defs", f"targets/{target_key}", f"tests/{target_key}"):
        (output_dir / rel).mkdir(parents=True, exist_ok=True)

    metadata = OrderedDict(
        [
            ("schema", SCHEMA),
            ("layout", LAYOUT),
            ("catalog_id", catalog_id or _default_catalog_id(target_key, family_ids)),
            ("generated_at", datetime.now(UTC).isoformat()),
            ("generator", "ggml-hrx-kernel-bench export-llama"),
            ("targets", [{"target_key": target_key, "target_variant": None}]),
            ("coverage", route_status),
        ]
    )
    written.append(_write_json(output_dir / "metadata.json", metadata))

    copied_sources: dict[str, dict[str, Any]] = OrderedDict()
    for source_id in source_ids:
        source = sources[source_id]
        source_path = kernel_dir / Path(str(source["path"])).name
        if not source_path.exists():
            raise ValueError(f"source {source_id} path does not exist: {source_path}")
        dest = output_dir / "kernels" / f"{source_id}.loom"
        shutil.copyfile(source_path, dest)
        written.append(dest)
        copied_sources[source_id] = OrderedDict(
            [
                ("path", str(dest.relative_to(output_dir))),
                ("sha256", file_sha256(dest)),
            ]
        )

    routes_by_family: dict[str, list[dict[str, Any]]] = {family_id: [] for family_id in family_ids}
    for route in selected_routes:
        normalized = _normalize_route(route, target_key=target_key)
        routes_by_family.setdefault(str(route["family"]), []).append(normalized)

    for family_id in family_ids:
        family = family_rows.get(family_id, {"family": family_id, "op": routes_by_family[family_id][0].get("op")})
        source_ids_for_family = sorted({str(route["source_id"]) for route in routes_by_family[family_id]})
        definition = OrderedDict(
            [
                ("schema", "ggml-hrx-kernel-def-v1"),
                ("family", family_id),
                ("op", family.get("op")),
                ("sources", OrderedDict((source_id, copied_sources[source_id]) for source_id in source_ids_for_family)),
                ("artifacts", _artifacts_for_routes(routes_by_family[family_id])),
            ]
        )
        written.append(_write_json(output_dir / "defs" / f"{family_id}.json", definition))

        route_file = OrderedDict(
            [
                ("schema", "ggml-hrx-target-routes-v1"),
                ("target_key", target_key),
                ("key", family_id),
                ("routes", routes_by_family[family_id]),
            ]
        )
        written.append(_write_json(output_dir / "targets" / target_key / f"{family_id}.json", route_file))

    test_case_count = 0
    for family_id in family_ids:
        schedule = _test_schedule_for_family(
            target_key=target_key,
            family_id=family_id,
            routes=routes_by_family[family_id],
        )
        if schedule is None:
            continue
        test_case_count += len(schedule["cases"])
        written.append(_write_json(output_dir / "tests" / target_key / f"{family_id}.json", schedule))

    return LlamaCatalogExportResult(
        output_dir=output_dir,
        target_key=target_key,
        family_count=len(family_ids),
        source_count=len(source_ids),
        route_count=len(selected_routes),
        test_case_count=test_case_count,
        written_paths=tuple(written),
    )


def current_git_commit(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _load_families(catalog_dir: Path) -> dict[str, dict[str, Any]]:
    path = catalog_dir / "families.json"
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in load_json(path):
        if isinstance(row, dict) and isinstance(row.get("family"), str):
            out[str(row["family"])] = dict(row)
    return out


def _route_selected(route: dict[str, Any], *, target_key: str, families: set[str] | None) -> bool:
    route_target = str(route.get("target_key") or "")
    if route_target and route_target != target_key:
        return False
    if route.get("op") == "EXPERIMENT":
        return False
    if str(route.get("id") or "") in _RUNTIME_EXCLUDED_ROUTE_REASONS:
        return False
    if families is None:
        return True
    return bool(
        str(route.get("family") or "") in families
        or str(route.get("source_id") or "") in families
        or str(route.get("id") or "") in families
    )


def _normalize_route(route: dict[str, Any], *, target_key: str) -> OrderedDict[str, Any]:
    out = OrderedDict()
    for key in (
        "id",
        "family",
        "op",
        "source_id",
        "artifact_id",
        "root_symbol",
        "export_name",
        "abi",
        "dispatch",
        "shape_domain",
        "shape_guards",
        "constraints",
        "specialization",
    ):
        if key in route:
            out[key] = route[key]
    if isinstance(route.get("supports"), dict):
        out["supports"] = _normalize_supports(route["supports"])
    out["artifact_id"] = _route_artifact_id(route)
    out["target_key"] = target_key
    if "loader_format" in route:
        out["loader_format"] = route["loader_format"]
    if "constraints" not in out:
        family = str(route.get("family") or "")
        if (
            family.startswith("mul_mat_")
            and "batched" not in family
            and not family.startswith("mul_mat_id_")
        ):
            out["constraints"] = [
                {"source": "src0.ne2", "eq": 1},
                {"source": "src0.ne3", "eq": 1},
                {"source": "src1.ne2", "eq": 1},
                {"source": "src1.ne3", "eq": 1},
                {"source": "dst.ne2", "eq": 1},
                {"source": "dst.ne3", "eq": 1},
                {"source": "src0.ne0", "eq_source": "src1.ne0"},
                {"source": "dst.ne0", "eq_source": "src0.ne1"},
                {"source": "dst.ne1", "eq_source": "src1.ne1"},
            ]
    # Workload arguments are launch-config region arguments, not general shape
    # facts. Static HRX3 routes pass problem size through config bindings; only
    # preserve explicit workload_arguments emitted by the source catalog.
    if "evidence_summary" in route:
        out["evidence"] = {"hrx2": route["evidence_summary"]}
    return out


def _route_export_status(
    catalog_dir: Path,
    *,
    target_key: str,
    families: set[str] | None,
) -> OrderedDict[str, Any]:
    rows_by_family: dict[str, list[dict[str, Any]]] = {}
    for route in iter_routes(catalog_dir):
        family = str(route.get("family") or "")
        if not family:
            continue
        if families is not None and family not in families and str(route.get("source_id") or "") not in families:
            continue
        rows_by_family.setdefault(family, []).append(route)
    routes_dir = catalog_dir / "routes"
    if routes_dir.exists():
        for path in sorted(routes_dir.glob("*.json")):
            if path.stem == "index":
                continue
            if families is not None and path.stem not in families:
                continue
            rows_by_family.setdefault(path.stem, [])

    families_with_routes = OrderedDict()
    skipped_routes: list[OrderedDict[str, Any]] = []
    for family, rows in sorted(rows_by_family.items()):
        selected = [
            route for route in rows
            if _route_selected(route, target_key=target_key, families=families)
        ]
        reason = "exported"
        if not rows:
            reason = "no_routes"
        elif not selected:
            if all(route.get("op") == "EXPERIMENT" for route in rows):
                reason = "experiment_only"
            elif not any(str(route.get("target_key") or "") in ("", target_key) for route in rows):
                reason = "target_mismatch"
            elif all(str(route.get("id") or "") in _RUNTIME_EXCLUDED_ROUTE_REASONS for route in rows):
                reason = "runtime_excluded"
            else:
                reason = "not_runtime_exported"
        elif family not in _SCHEDULE_FAMILIES:
            reason = "exported_no_smoke_schedule"
        families_with_routes[family] = OrderedDict(
            [
                ("route_count", len(rows)),
                ("exported_route_count", len(selected)),
                ("test_schedule", "generated" if family in _SCHEDULE_FAMILIES and selected else "none"),
                ("status", reason),
            ]
        )
        for route in rows:
            if _route_selected(route, target_key=target_key, families=families):
                continue
            skipped_routes.append(
                OrderedDict(
                    [
                        ("family", family),
                        ("id", route.get("id")),
                        ("op", route.get("op")),
                        ("target_key", route.get("target_key", "")),
                        (
                            "reason",
                            _skipped_route_reason(route),
                        ),
                        ("details", _RUNTIME_EXCLUDED_ROUTE_REASONS.get(str(route.get("id") or ""))),
                    ]
                )
            )

    return OrderedDict(
        [
            ("target_key", target_key),
            ("families", families_with_routes),
            ("skipped_routes", skipped_routes),
        ]
    )


def _clear_generated_catalog_dir(output_dir: Path, target_key: str) -> None:
    for rel in ("kernels", "defs", f"targets/{target_key}", f"tests/{target_key}"):
        path = output_dir / rel
        if path.exists():
            shutil.rmtree(path)
    metadata = output_dir / "metadata.json"
    if metadata.exists():
        metadata.unlink()


def _skipped_route_reason(route: dict[str, Any]) -> str:
    if route.get("op") == "EXPERIMENT":
        return "experiment"
    if str(route.get("id") or "") in _RUNTIME_EXCLUDED_ROUTE_REASONS:
        return "runtime_excluded"
    return "target_mismatch"


def _normalize_supports(value: Any) -> OrderedDict[str, Any]:
    normalized: OrderedDict[str, Any] = OrderedDict()
    if not isinstance(value, dict):
        return normalized
    for key, item in value.items():
        if isinstance(item, bool):
            normalized[str(key)] = item
        elif isinstance(item, str):
            normalized[str(key)] = item
        else:
            normalized[str(key)] = str(item)
    return normalized


def _artifacts_for_routes(routes: list[dict[str, Any]]) -> OrderedDict[str, Any]:
    artifacts: OrderedDict[str, Any] = OrderedDict()
    for route in routes:
        artifact_id = str(route["artifact_id"])
        if artifact_id in artifacts:
            continue
        artifacts[artifact_id] = OrderedDict(
            [
                ("format", "loom-bytecode"),
                ("source_id", route["source_id"]),
                ("root_symbol", route["root_symbol"]),
                ("path", f"artifacts/{artifact_id}.loombc"),
                ("storage", "embedded"),
            ]
        )
    return artifacts


def _test_schedule_for_family(
    *,
    target_key: str,
    family_id: str,
    routes: list[dict[str, Any]],
) -> OrderedDict[str, Any] | None:
    if family_id not in _SCHEDULE_FAMILIES:
        return None
    route = select_test_route(family_id, routes)
    if route is None:
        return None
    shape = test_shape_for_route(family_id, route)
    if shape is None:
        return None
    scenario = test_scenario_for_family(family_id, route)
    supports = _normalize_supports(route.get("supports")) if isinstance(route.get("supports"), dict) else OrderedDict()
    return OrderedDict(
        [
            ("schema", "ggml-hrx-test-schedule-v1"),
            ("target_key", target_key),
            ("family", family_id),
            ("cases", [
                OrderedDict(
                    [
                        ("id", f"{family_id}_{route['id']}_smoke"),
                        ("op", route["op"]),
                        ("scenario", scenario),
                        ("expected_route_id", route["id"]),
                        ("schedule", OrderedDict([("source", "atlas-test"), ("scenario", scenario)])),
                        ("supports", supports),
                        ("shape", shape),
                        ("tolerance", _SCHEDULE_TOLERANCE_BY_FAMILY.get(family_id, 1.0e-6)),
                        ("repeat", 2),
                    ]
                )
            ]),
        ]
    )


def _default_catalog_id(target_key: str, family_ids: list[str]) -> str:
    family_part = "empty" if not family_ids else "-".join(family_ids)
    return f"hrx3-{target_key}-{family_part}"


def _route_artifact_id(route: dict[str, Any]) -> str:
    return str(route["id"]) + "_loombc"


def _write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=False) + "\n", encoding="utf-8", newline="\n")
    return path
