from __future__ import annotations

import datetime as _datetime
import hashlib
import json
from pathlib import Path
from typing import Any


RUN_MANIFEST_SCHEMA = "ggml_hrx_kernel_bench.loom_execution_runs.v1"
RESULT_SCHEMA = "ggml_hrx_kernel_bench.loom_benchmark_result.v1"
SUMMARY_SCHEMA = "ggml_hrx_kernel_bench.loom_benchmark_summary.v1"
COMPARE_SCHEMA = "ggml_hrx_kernel_bench.loom_benchmark_compare.v1"
SCRIPT_INDEX_SCHEMA = "ggml_hrx_kernel_bench.loom_benchmark_script_index.v1"
SCRIPT_ROUTE_MANIFEST_SCHEMA = "ggml_hrx_kernel_bench.loom_benchmark_script_route.v1"
SCRIPT_CASE_MANIFEST_SCHEMA = "ggml_hrx_kernel_bench.loom_benchmark_script_case.v1"
FLOP_ESTIMATE_SCHEMA = "ggml_hrx_kernel_bench.flop_estimate.v1"


def json_scalar(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def sha1_text(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def sha1_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_name(value: str, *, max_length: int = 96) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-")
    safe = safe or "benchmark"
    if len(safe) <= max_length:
        return safe
    digest = sha1_text(value)[:12]
    return f"{safe[: max_length - 13].rstrip('-')}-{digest}"


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def write_json_file(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_executable_script(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | 0o111)


def timestamp() -> str:
    return _datetime.datetime.now(_datetime.UTC).strftime("%Y%m%dT%H%M%SZ")


def load_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows
