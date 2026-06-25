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
from .specs import file_sha256


SCHEMA = "ggml-hrx-catalog-v1"
LAYOUT = "ggml-hrx-split-catalog-v1"


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

    missing_sources = [source_id for source_id in source_ids if source_id not in sources]
    if missing_sources:
        raise ValueError("selected routes reference missing source ids: " + ", ".join(missing_sources))

    written: list[Path] = []
    output_dir.mkdir(parents=True, exist_ok=True)
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
    if route.get("op") != "MUL_MAT":
        return False
    supports = route.get("supports") or {}
    if supports.get("src0_type") == "Q4_K" and supports.get("src1_type") == "F32" and supports.get("dst_type") == "F32":
        if "rhs_type" in supports:
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
        "supports",
    ):
        if key in route:
            out[key] = route[key]
    out["artifact_id"] = _route_artifact_id(route)
    out["target_key"] = target_key
    if "loader_format" in route:
        out["loader_format"] = route["loader_format"]
    if "constraints" not in out:
        family = str(route.get("family") or "")
        if family.startswith("mul_mat_"):
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
    if family_id != "mul_mat_q4_k_f32":
        return None
    direct_route = next((route for route in routes if "_direct_" in str(route.get("id", ""))), None)
    if direct_route is None:
        return None
    supports = direct_route.get("supports") if isinstance(direct_route.get("supports"), dict) else {}
    return OrderedDict(
        [
            ("schema", "ggml-hrx-test-schedule-v1"),
            ("target_key", target_key),
            ("family", family_id),
            ("cases", [
                OrderedDict(
                    [
                        ("id", "mul_mat_q4_k_f32_direct_decode_k256_r3_c1"),
                        ("op", "MUL_MAT"),
                        ("expected_route_id", direct_route["id"]),
                        ("supports", OrderedDict(
                            [
                                ("src0_type", supports.get("src0_type", "Q4_K")),
                                ("src1_type", supports.get("src1_type", "F32")),
                                ("dst_type", supports.get("dst_type", "F32")),
                            ]
                        )),
                        ("shape", OrderedDict([("k", 256), ("rows", 3), ("cols", 1)])),
                        ("tolerance", 8.0e-2),
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
