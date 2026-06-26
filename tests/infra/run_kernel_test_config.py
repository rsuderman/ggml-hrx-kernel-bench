from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ggml_hrx_kernel_bench.cli import run_candidate_row
from ggml_hrx_kernel_bench.config import BenchConfig, ToolPaths
from ggml_hrx_kernel_bench.family_specs import normalize_shape
from ggml_hrx_kernel_bench.hrx2 import Candidate, build_config, iter_routes, load_sources_by_id, route_launch, stable_id
from ggml_hrx_kernel_bench.reporting import correctness_ok

from required_tools import require_tool


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _load_config(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    _expect(isinstance(data, dict), "config must be a JSON object")
    return data


def _case_id(params: list[str], values: list[int]) -> str:
    return "_".join(f"{name}{value}" for name, value in zip(params, values, strict=True))


def _select_case(config: dict, selector: str) -> tuple[str, list[int]]:
    params = list(config["params"])
    cases = list(config["cases"])
    if selector.isdigit():
        index = int(selector)
        _expect(0 <= index < len(cases), f"case index out of range: {index}")
        values = list(cases[index])
        return _case_id(params, values), values
    for values in cases:
        case_values = list(values)
        case_id = _case_id(params, case_values)
        if case_id == selector:
            return case_id, case_values
    raise RuntimeError(f"case not found in config: {selector}")


def _select_route(catalog_dir: Path, *, family: str) -> dict:
    matches = [
        route
        for route in iter_routes(catalog_dir)
        if str(route.get("family") or route.get("source_id") or "") == family
    ]
    _expect(matches, f"no route found for family={family}")
    _expect(len(matches) == 1, f"minimal config requires exactly one route for {family}, found {len(matches)}")
    return matches[0]


def _shape_for_case(config: dict, values: list[int]) -> dict[str, int]:
    params = list(config["params"])
    _expect(len(params) == len(values), "params and case values must have the same length")
    return normalize_shape(dict(zip(params, values, strict=True)))


def _build_candidate(config_data: dict, case_id: str, case_values: list[int]) -> Candidate:
    family = str(config_data["kernel"])
    catalog_dir = ROOT / "catalog" / "hrx2"
    kernel_dir = ROOT / "kernels" / "hrx2"
    route = _select_route(catalog_dir, family=family)
    op = str(route.get("op") or "")
    _expect(op == "CPY", f"only CPY is supported by this execution test, got {op}")
    shape = _shape_for_case(config_data, case_values)
    config_bindings, values, missing = build_config(route, shape)
    _expect(not missing, f"missing shape/config bindings: {missing}")
    sources = load_sources_by_id(kernel_dir, catalog_dir)
    source_id = str(route.get("source_id") or family)
    source = sources.get(source_id)
    _expect(source is not None, f"kernel source is not available for source_id={source_id}")
    candidate_id = f"{family}_{case_id}_{stable_id(shape, config_bindings, length=8)}"
    return Candidate(
        id=candidate_id,
        family=family,
        op=op,
        source_id=source_id,
        source_path=source.path,
        root_symbol=str(route.get("root_symbol") or ""),
        export_name=route.get("export_name"),
        route_id=str(route.get("id") or ""),
        route=route,
        shape=shape,
        values=values,
        config=config_bindings,
        dispatch=route_launch(route, shape),
        supports=dict(route.get("supports") or {}),
        coverage="route_backed",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a kernel correctness test from a kernel-test-config file.")
    parser.add_argument("config_path")
    parser.add_argument("case_selector")
    parser.add_argument("--tool-dir", help="optional directory containing loom-link and iree-benchmark-loom")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target", default="gfx1100")
    parser.add_argument("--rocm-path")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--warmup-iterations", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=1)
    args = parser.parse_args()

    config_path = Path(args.config_path).resolve()
    config_data = _load_config(config_path)
    case_id, case_values = _select_case(config_data, args.case_selector)
    candidate = _build_candidate(config_data, case_id, case_values)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    bench_config = BenchConfig(
        output_dir=output_dir,
        target=args.target,
        tools=ToolPaths(
            loom_link=Path(require_tool("loom-link", tool_dir=args.tool_dir)),
            iree_benchmark_loom=Path(require_tool("iree-benchmark-loom", tool_dir=args.tool_dir)),
        ),
        rocm_path=Path(args.rocm_path).resolve() if args.rocm_path else None,
    )
    run_args = argparse.Namespace(
        output_dir=output_dir,
        target=args.target,
        rocm_path=Path(args.rocm_path).resolve() if args.rocm_path else None,
        iterations=args.iterations,
        warmup_iterations=args.warmup_iterations,
        max_batches=args.max_batches,
    )

    row = run_candidate_row(run_args, bench_config, candidate, sanitizer="none")
    if row.get("status") != "ran":
        raise RuntimeError(f"kernel run failed with status={row.get('status')} row={json.dumps(row, sort_keys=True)}")
    benchmark = row.get("benchmark") or {}
    summary = benchmark.get("summary") or {}
    correctness = summary.get("correctness")
    if not correctness_ok(correctness):
        raise RuntimeError(
            "kernel correctness check failed: "
            + json.dumps(
                {
                    "candidate_id": candidate.id,
                    "correctness": correctness,
                    "failure": summary.get("failure"),
                    "results_path": benchmark.get("results_path"),
                    "output_dir": str(output_dir),
                },
                sort_keys=True,
            )
        )

    print(
        json.dumps(
            {
                "candidate_id": candidate.id,
                "case_id": case_id,
                "correctness": correctness,
                "results_path": benchmark.get("results_path"),
                "output_dir": str(output_dir),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
