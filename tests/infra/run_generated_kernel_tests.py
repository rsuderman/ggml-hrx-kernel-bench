from __future__ import annotations

import argparse
import json
from pathlib import Path

import bootstrap  # noqa: F401

from ggml_hrx_kernel_bench.kernel_test_config import load_config
from ggml_hrx_kernel_bench.kernel_test_config_runtime import case_result, execute_case, select_case
from ggml_hrx_kernel_bench.required_tools import require_tool


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _safe_name(value: str) -> str:
    return ''.join(ch.lower() if ch.isalnum() else '-' for ch in value).strip('-') or 'generated'


def _load_manifest(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _expect(isinstance(payload, dict), "manifest must be a JSON object")
    entries = payload.get("entries")
    _expect(isinstance(entries, list), "manifest entries must be a list")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run generated kernel tests listed in a manifest.")
    parser.add_argument("manifest_path")
    parser.add_argument("--case-selector", default="0")
    parser.add_argument("--tool-dir", help="optional directory containing loom-link and iree-benchmark-loom")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target", default="gfx1100")
    parser.add_argument("--rocm-path")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--warmup-iterations", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=1)
    args = parser.parse_args()

    manifest_path = Path(args.manifest_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = _load_manifest(manifest_path)

    results = []
    for index, entry in enumerate(payload["entries"]):
        _expect(isinstance(entry, dict), f"entries[{index}] must be an object")
        config_path = Path(str(entry.get("config_path") or "")).resolve()
        _expect(config_path.is_file(), f"missing generated config {config_path}")
        config_data = load_config(config_path)
        current_case_id, current_case_values = select_case(config_data, args.case_selector)
        case_output_dir = output_dir / f"{index:03d}-{_safe_name(config_path.stem)}"
        candidate, row, summary = execute_case(
            config_data=config_data,
            current_case_id=current_case_id,
            current_case_values=current_case_values,
            tool_dir=args.tool_dir,
            target=args.target,
            rocm_path=args.rocm_path,
            iterations=args.iterations,
            warmup_iterations=args.warmup_iterations,
            max_batches=args.max_batches,
            output_dir=case_output_dir,
            require_tool=require_tool,
        )
        result = case_result(
            candidate=candidate,
            current_case_id=current_case_id,
            current_case_values=current_case_values,
            row=row,
            summary=summary,
            output_dir=case_output_dir,
        )
        result["config_path"] = str(config_path)
        results.append(result)
        _expect(result["status"] == "ran", f"kernel run failed: {json.dumps(result, sort_keys=True)}")
        _expect(result["correctness_ok"], f"kernel correctness check failed: {json.dumps(result, sort_keys=True)}")

    print(json.dumps({"manifest_path": str(manifest_path), "result_count": len(results), "results": results}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
