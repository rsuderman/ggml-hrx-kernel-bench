from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import BenchConfig, ToolPaths
from .routing.api import (
    Candidate,
    CandidateQuery,
    DEFAULT_ROUTING_VERSION,
    ExportRequest,
    create_router,
    supported_routing_versions,
)
from .ledger import LedgerWriter, utc_run_id
from .oracles import generate_oracle, write_workbench
from .tools import CommandResult, run_command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ggml-hrx-kernel-bench")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target", default="gfx1100")
    parser.add_argument("--rocm-path", type=Path)
    parser.add_argument("--loom-link", type=Path)
    parser.add_argument("--loom-compile", type=Path)
    parser.add_argument("--iree-test-loom", type=Path)
    parser.add_argument(
        "--routing-version",
        choices=supported_routing_versions(),
        default=DEFAULT_ROUTING_VERSION,
    )
    parser.add_argument("--kernel-dir", type=Path)
    parser.add_argument("--routing-dir", type=Path)
    parser.add_argument("--family", action="append", default=[], help="family/source/route filter; may be repeated or comma separated")
    parser.add_argument("--limit", type=int, help="limit corpus candidates")
    parser.add_argument("--sweep", choices=["minimal", "edge"], default="minimal")
    parser.add_argument("--include-source-only", action="store_true", help="include source-only/probe kernel rows in addition to route-backed catalog rows")
    parser.add_argument("--sanitizers", default="none", help="comma list, for example none,asan,tsan")
    parser.add_argument("--llama-catalog-dir", type=Path, help="sparse llama.cpp generated/catalog directory to update")
    parser.add_argument("--llama-catalog-id", help="catalog id to write when exporting llama.cpp catalog metadata")

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan")
    subparsers.add_parser("fixtures")
    subparsers.add_parser("link")
    subparsers.add_parser("compile")
    subparsers.add_parser("run")
    subparsers.add_parser("verify")
    subparsers.add_parser("catalog")
    subparsers.add_parser("export-llama")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ledger = LedgerWriter(args.output_dir / "ledger.jsonl")
    config = BenchConfig(
        output_dir=args.output_dir,
        target=args.target,
        rocm_path=args.rocm_path,
        tools=ToolPaths(
            loom_link=args.loom_link,
            loom_compile=args.loom_compile,
            iree_test_loom=args.iree_test_loom,
        ),
    )

    try:
        if args.command == "export-llama":
            return command_export_llama(args, ledger)
        return command_corpus(args, config, ledger)
    except Exception as exc:
        ledger.append(
            {
                "schema": "ggml_hrx_kernel_bench.ledger.v1",
                "run_id": utc_run_id(),
                "action": args.command,
                "status": "tool_error",
                "error": type(exc).__name__,
                "message": str(exc),
            }
        )
        raise
def command_corpus(args: argparse.Namespace, config: BenchConfig, ledger: LedgerWriter) -> int:
    candidates = selected_candidates(args)
    if args.command == "plan":
        ledger.write_all(corpus_row(args, candidate, action="plan") for candidate in candidates)
        write_summary(args.output_dir, candidates)
        return 0
    if args.command == "fixtures":
        rows = [fixtures_row(args, candidate) for candidate in candidates]
        ledger.write_all(rows)
        return status_code_from_rows(rows)
    if args.command == "link":
        rows = [link_candidate_row(args, config, candidate) for candidate in candidates]
        ledger.write_all(rows)
        return status_code_from_rows(rows)
    if args.command == "compile":
        rows: list[dict[str, Any]] = []
        for sanitizer in sanitizer_list(args):
            for candidate in candidates:
                rows.append(compile_candidate_row(args, config, candidate, sanitizer=sanitizer))
        ledger.write_all(rows)
        return status_code_from_rows(rows)
    if args.command == "run":
        rows: list[dict[str, Any]] = []
        for sanitizer in sanitizer_list(args):
            for candidate in candidates:
                rows.append(run_candidate_test_row(args, config, candidate, sanitizer=sanitizer))
        ledger.write_all(rows)
        return status_code_from_rows(rows)
    if args.command == "verify":
        rows = [fixtures_row(args, candidate, action="verify") for candidate in candidates]
        ledger.write_all(rows)
        return status_code_from_rows(rows)
    if args.command == "catalog":
        return command_catalog(args, candidates, ledger)
    raise ValueError(f"unsupported command {args.command}")

def command_export_llama(args: argparse.Namespace, ledger: LedgerWriter) -> int:
    if args.llama_catalog_dir is None:
        raise ValueError("export-llama requires --llama-catalog-dir")
    router = create_router(
        version=args.routing_version,
        kernel_dir=args.kernel_dir,
        routing_dir=args.routing_dir,
    )
    result = router.export(
        ExportRequest(
            output_dir=args.llama_catalog_dir,
            target_key=args.target,
            routing_id=args.llama_catalog_id,
        )
    )
    ledger.append(
        {
            "schema": "ggml_hrx_kernel_bench.ledger.v1",
            "run_id": utc_run_id(),
            "action": "export-llama",
            "status": "ok",
            "export": result.to_ledger(),
        }
    )
    return 0

def selected_candidates(args: argparse.Namespace) -> list[Candidate]:
    families = filter_set(args.family)
    router = create_router(
        version=args.routing_version,
        kernel_dir=args.kernel_dir,
        routing_dir=args.routing_dir,
    )
    return router.candidates(
        CandidateQuery(
            families=families,
            limit=args.limit,
            sweep=args.sweep,
            include_source_only=args.include_source_only,
            target=args.target,
        )
    )


def filter_set(values: list[str]) -> set[str] | None:
    out: set[str] = set()
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part:
                out.add(part)
    return out or None


def sanitizer_list(args: argparse.Namespace) -> list[str]:
    values = [part.strip() for part in args.sanitizers.split(",") if part.strip()]
    return values or ["none"]


def config_args(bindings: dict[str, str]) -> list[str]:
    return [f"--config={key}={bindings[key]}" for key in sorted(bindings)]


def corpus_row(args: argparse.Namespace, candidate: Candidate, *, action: str) -> dict[str, Any]:
    row = {
        "schema": "ggml_hrx_kernel_bench.ledger.v1",
        "run_id": utc_run_id(),
        "action": action,
        "machine": {"target": args.target, "rocm_path": str(args.rocm_path) if args.rocm_path else None},
        "candidate": candidate.to_ledger(),
        "status": candidate.status,
    }
    if candidate.message:
        row["message"] = candidate.message
    return row


def candidate_dir(args: argparse.Namespace, candidate: Candidate, *parts: str) -> Path:
    return args.output_dir / "candidates" / candidate.id / Path(*parts)


def fixtures_row(args: argparse.Namespace, candidate: Candidate, *, action: str = "fixtures") -> dict[str, Any]:
    row = corpus_row(args, candidate, action=action)
    if candidate.status != "planned":
        row["status"] = candidate.status
        row["message"] = candidate.message
        return row
    result = generate_oracle(candidate, candidate_dir(args, candidate, "fixtures"))
    row["oracle"] = result.to_ledger()
    row["status"] = result.status
    return row


def link_candidate_row(args: argparse.Namespace, config: BenchConfig, candidate: Candidate) -> dict[str, Any]:
    row = corpus_row(args, candidate, action="link")
    if candidate.status != "planned":
        return row
    out_dir = candidate_dir(args, candidate, "link")
    out_dir.mkdir(parents=True, exist_ok=True)
    linked = out_dir / "linked.loom"
    result = run_command(
        [
            config.tools.require_loom_link(),
            candidate.source_path,
            "--mode=link",
            "--to=text",
            "--require-resolved-config",
            f"--root={candidate.root_symbol}",
            f"--output={linked}",
            *config_args(candidate.config),
        ],
        env=config.command_env(),
    )
    row["link"] = command_evidence(result, out_dir)
    row["link"]["output"] = str(linked) if linked.exists() else None
    row["status"] = "linked" if result.returncode == 0 else "link_failed"
    return row


def compile_candidate_row(args: argparse.Namespace, config: BenchConfig, candidate: Candidate, *, sanitizer: str) -> dict[str, Any]:
    row = corpus_row(args, candidate, action="compile")
    row["sanitizer"] = sanitizer
    if candidate.status != "planned":
        return row
    out_dir = candidate_dir(args, candidate, "compile", sanitizer)
    out_dir.mkdir(parents=True, exist_ok=True)
    linked = out_dir / "linked.loom"
    link_result = run_command(
        [
            config.tools.require_loom_link(),
            candidate.source_path,
            "--mode=link",
            "--to=text",
            "--require-resolved-config",
            f"--root={candidate.root_symbol}",
            f"--output={linked}",
            *config_args(candidate.config),
        ],
        env=config.command_env(),
    )
    row["link"] = command_evidence(link_result, out_dir, prefix="link")
    row["link"]["output"] = str(linked) if linked.exists() else None
    if link_result.returncode != 0:
        row["status"] = "link_failed"
        return row

    report = out_dir / "compile_report.json"
    manifest = out_dir / "artifact_manifest.json"
    artifact = out_dir / "artifact.bin"
    target_artifact = out_dir / "target.hsaco"
    compile_cmd: list[str | Path] = [
        config.tools.require_loom_compile(),
        linked,
        "--backend=amdgpu-hal",
        f"--target={config.target}",
        f"--root={candidate.root_symbol}",
        f"--output={artifact}",
        f"--emit-target-artifact={target_artifact}",
        "--compile-report=details",
        f"--compile-report-output={report}",
        "--artifact-manifest=analysis",
        f"--emit-artifact-manifest={manifest}",
    ]
    if sanitizer != "none":
        compile_cmd.append(f"--sanitizer={sanitizer}")
    compile_result = run_command(compile_cmd, env=config.command_env())
    row["compile"] = command_evidence(compile_result, out_dir, prefix="compile")
    row["compile"].update(
        {
            "report": str(report) if report.exists() else None,
            "manifest": str(manifest) if manifest.exists() else None,
            "artifact": str(artifact) if artifact.exists() else None,
            "target_artifact": str(target_artifact) if target_artifact.exists() else None,
            "target_artifact_bytes": target_artifact.stat().st_size if target_artifact.exists() else None,
            "report_summary": compile_report_summary(report),
        }
    )
    row["status"] = "compiled" if compile_result.returncode == 0 else "compile_failed"
    return row


def _case_symbol(candidate: Candidate) -> str:
    return f"@case_{candidate.id}"


def test_report_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "state": "parse_error",
            "correctness": {"parse_error": f"missing test report: {path}"},
            "failure": {"parse_error": f"missing test report: {path}"},
        }
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        message = f"failed to parse test report: {exc}"
        return {
            "state": "parse_error",
            "correctness": {"parse_error": message},
            "failure": {"parse_error": message},
        }
    if not isinstance(report, dict):
        message = "test report must be a JSON object"
        return {
            "state": "parse_error",
            "correctness": {"parse_error": message},
            "failure": {"parse_error": message},
        }
    try:
        failed_sample_count = int(report.get("failed_sample_count") or 0)
        planning_issue_count = int(report.get("planning_issue_count") or 0)
        skipped_case_count = int(report.get("skipped_case_count") or 0)
    except (TypeError, ValueError) as exc:
        message = f"invalid test report counters: {exc}"
        return {
            "state": "parse_error",
            "correctness": {"parse_error": message},
            "failure": {"parse_error": message},
        }
    failure = None
    if failed_sample_count != 0 or planning_issue_count != 0:
        failure = {
            "failed_sample_count": failed_sample_count,
            "planning_issue_count": planning_issue_count,
            "skipped_case_count": skipped_case_count,
            "samples": report.get("samples"),
            "planning_issues": report.get("planning_issues"),
        }
    return {
        "state": "ok" if failure is None else "failed",
        "correctness": report,
        "failure": failure,
        "operation_timing_ns": report.get("operation_timing_ns"),
        "mean_physical_dispatch_duration_ns": report.get(
            "mean_physical_dispatch_duration_ns"
        ),
        "physical_dispatches_per_logical_operation": report.get(
            "physical_dispatches_per_logical_operation"
        ),
    }


def run_candidate_test_row(args: argparse.Namespace, config: BenchConfig, candidate: Candidate, *, sanitizer: str) -> dict[str, Any]:
    row = corpus_row(args, candidate, action="run")
    row["sanitizer"] = sanitizer
    if candidate.status != "planned":
        return row
    out_dir = candidate_dir(args, candidate, "run", sanitizer)
    out_dir.mkdir(parents=True, exist_ok=True)
    fixture = generate_oracle(candidate, out_dir / "fixtures")
    row["oracle"] = fixture.to_ledger()
    if fixture.status != "fixtures_ready" or fixture.fixture_dir is None:
        row["status"] = fixture.status
        return row

    linked = out_dir / "linked.loom"
    link_result = run_command(
        [
            config.tools.require_loom_link(),
            candidate.source_path,
            "--mode=link",
            "--to=text",
            "--require-resolved-config",
            f"--root={candidate.root_symbol}",
            f"--output={linked}",
            *config_args(candidate.config),
        ],
        env=config.command_env(),
    )
    row["link"] = command_evidence(link_result, out_dir, prefix="link")
    if link_result.returncode != 0:
        row["status"] = "link_failed"
        return row

    workbench = out_dir / "workbench.loom"
    _, workbench_meta = write_workbench(candidate, linked, workbench, fixture.fixture_dir)
    row["workbench"] = workbench_meta
    if workbench_meta.get("status") != "ok":
        row["status"] = workbench_meta.get("status", "unsupported_golden")
        return row

    report_path = out_dir / "test-report.json"
    cmd: list[str | Path] = [
        config.tools.require_iree_test_loom().resolve(),
        workbench.resolve(),
        "--device=amdgpu",
        f"--case={_case_symbol(candidate)}",
    ]
    if sanitizer != "none":
        cmd.append(f"--sanitizer={sanitizer}")
    result = run_command(cmd, env=config.command_env(), cwd=out_dir)
    report_path.write_text(result.stdout, encoding="utf-8")
    row["test"] = command_evidence(result, out_dir, prefix="test")
    summary = test_report_summary(report_path)
    row["test"].update(
        {
            "results_path": str(report_path),
            "summary": summary,
        }
    )
    row["status"] = (
        "ran" if result.returncode == 0 and summary.get("state") == "ok" else "run_failed"
    )
    return row


def command_catalog(args: argparse.Namespace, candidates: list[Candidate], ledger: LedgerWriter) -> int:
    ledger_rows = read_jsonl(args.output_dir / "ledger.jsonl")
    by_candidate: dict[str, dict[str, Any]] = {}
    for row in ledger_rows:
        candidate_id = (((row.get("candidate") or {}).get("candidate_id")) or "")
        if not candidate_id:
            continue
        by_candidate.setdefault(candidate_id, {})[row.get("action", "unknown")] = row
    catalog_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        evidence = by_candidate.get(candidate.id, {})
        compile_row = evidence.get("compile")
        run_row = evidence.get("run")
        catalog_ready = bool(compile_row and compile_row.get("status") == "compiled")
        if run_row and run_row.get("status") != "ran":
            catalog_ready = False
        catalog_rows.append(
            {
                "candidate_id": candidate.id,
                "catalog_ready": catalog_ready,
                "candidate": candidate.to_ledger(),
                "compile": (compile_row or {}).get("compile"),
                "oracle": (run_row or {}).get("oracle"),
                "test": (run_row or {}).get("test"),
                "rejection_reasons": rejection_reasons(candidate, compile_row, run_row),
            }
        )
    catalog_dir = args.output_dir / "catalog"
    catalog_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = catalog_dir / "candidates.json"
    catalog_path.write_text(json.dumps(catalog_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    ledger.append(
        {
            "schema": "ggml_hrx_kernel_bench.ledger.v1",
            "run_id": utc_run_id(),
            "action": "catalog",
            "status": "ok",
            "catalog_path": str(catalog_path),
            "candidate_count": len(catalog_rows),
            "catalog_ready_count": sum(1 for row in catalog_rows if row["catalog_ready"]),
        }
    )
    return 0


def rejection_reasons(candidate: Candidate, compile_row: dict[str, Any] | None, run_row: dict[str, Any] | None) -> list[str]:
    reasons: list[str] = []
    if candidate.status != "planned":
        reasons.append(candidate.status)
    if not compile_row:
        reasons.append("not_compiled")
    elif compile_row.get("status") != "compiled":
        reasons.append(str(compile_row.get("status")))
    if run_row and run_row.get("status") != "ran":
        reasons.append(str(run_row.get("status")))
    return reasons


def command_evidence(result: CommandResult, out_dir: Path, *, prefix: str = "command") -> dict[str, Any]:
    stdout_path = out_dir / f"{prefix}.stdout.txt"
    stderr_path = out_dir / f"{prefix}.stderr.txt"
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    row = result.to_ledger()
    row["stdout_path"] = str(stdout_path)
    row["stderr_path"] = str(stderr_path)
    return row


def compile_report_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"parse_error": str(exc)}
    return {
        "emission_code_byte_count": dig(report, "emission", "code_byte_count"),
        "allocation_spill_count": dig(report, "allocation", "spill_count"),
        "memory_local_bytes": dig(report, "memory", "local_bytes"),
        "static_instruction_mix": report.get("static_instruction_mix"),
        "entries_row_count": len(dig(report, "entries", "rows") or []),
    }


def dig(value: dict[str, Any] | None, *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_summary(output_dir: Path, candidates: list[Candidate]) -> None:
    summary = {
        "schema": "ggml_hrx_kernel_bench.plan_summary.v1",
        "candidate_count": len(candidates),
        "planned_count": sum(1 for candidate in candidates if candidate.status == "planned"),
        "by_status": {},
        "by_family": {},
    }
    for candidate in candidates:
        summary["by_status"][candidate.status] = summary["by_status"].get(candidate.status, 0) + 1
        summary["by_family"][candidate.family] = summary["by_family"].get(candidate.family, 0) + 1
    (output_dir / "plan_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def status_code_from_rows(rows: list[dict[str, Any]]) -> int:
    hard_failures = {"tool_error"}
    if any(row.get("status") in hard_failures for row in rows):
        return 2
    return 0
