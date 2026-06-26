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
from ggml_hrx_kernel_bench.hrx2 import Candidate, build_config, iter_routes, route_launch, stable_id
from ggml_hrx_kernel_bench.reporting import correctness_ok

from required_tools import require_tool


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _load_config(path: Path) -> dict:
    data = json.loads(path.read_text(encoding='utf-8'))
    _expect(isinstance(data, dict), 'config must be a JSON object')
    return data


def _select_case(config: dict, case_id: str) -> dict:
    for case in config.get('cases', []):
        if case.get('id') == case_id:
            return dict(case)
    raise RuntimeError(f'case not found in config: {case_id}')


def _select_route(catalog_dir: Path, *, root_symbol: str, op: str, family: str) -> dict:
    matches = [
        route
        for route in iter_routes(catalog_dir)
        if str(route.get('root_symbol') or '') == root_symbol
        and str(route.get('op') or '') == op
        and str(route.get('family') or route.get('source_id') or '') == family
    ]
    _expect(matches, f'no route found for family={family} op={op} root_symbol={root_symbol}')
    _expect(len(matches) == 1, f'expected exactly one route, found {len(matches)}')
    return matches[0]


def _copy_shape_for_case(case: dict) -> dict[str, int]:
    inputs = case.get('inputs') or {}
    outputs = case.get('outputs') or {}
    _expect('src' in inputs, 'copy case must define inputs.src')
    _expect('dst' in outputs, 'copy case must define outputs.dst')
    src_shape = normalize_shape(dict(inputs['src']))
    dst_shape = normalize_shape(dict(outputs['dst']))
    _expect(src_shape == dst_shape, 'copy case requires src and dst shapes to match')
    return dst_shape


def _build_candidate(config_path: Path, config_data: dict, case_data: dict) -> Candidate:
    op = str(config_data['op'])
    family = str(config_data['id'])
    _expect(op == 'CPY', f'only CPY is supported by this execution test, got {op}')
    root_symbol = str(config_data['root_symbol'])
    catalog_dir = ROOT / 'catalog' / 'hrx2'
    route = _select_route(catalog_dir, root_symbol=root_symbol, op=op, family=family)
    shape = _copy_shape_for_case(case_data)
    config_bindings, values, missing = build_config(route, shape)
    _expect(not missing, f'missing shape/config bindings: {missing}')
    source_path = (config_path.parent / str(config_data['source'])).resolve()
    _expect(source_path.exists(), f'kernel source does not exist: {source_path}')
    candidate_id = f"{family}_{case_data['id']}_{stable_id(shape, config_bindings, length=8)}"
    return Candidate(
        id=candidate_id,
        family=family,
        op=op,
        source_id=str(route.get('source_id') or family),
        source_path=source_path,
        root_symbol=root_symbol,
        export_name=route.get('export_name'),
        route_id=str(route.get('id') or ''),
        route=route,
        shape=shape,
        values=values,
        config=config_bindings,
        dispatch=route_launch(route, shape),
        supports=dict(route.get('supports') or {}),
        coverage='route_backed',
    )


def main() -> int:
    parser = argparse.ArgumentParser(description='Run a kernel correctness test from a kernel-test-config file.')
    parser.add_argument('config_path')
    parser.add_argument('case_id')
    parser.add_argument('--tool-dir', help='optional directory containing loom-link and iree-benchmark-loom')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--target', default='gfx1100')
    parser.add_argument('--rocm-path')
    parser.add_argument('--iterations', type=int, default=1)
    parser.add_argument('--warmup-iterations', type=int, default=0)
    parser.add_argument('--max-batches', type=int, default=1)
    args = parser.parse_args()

    config_path = Path(args.config_path).resolve()
    config_data = _load_config(config_path)
    case_data = _select_case(config_data, args.case_id)
    candidate = _build_candidate(config_path, config_data, case_data)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    bench_config = BenchConfig(
        output_dir=output_dir,
        target=args.target,
        tools=ToolPaths(
            loom_link=Path(require_tool('loom-link', tool_dir=args.tool_dir)),
            iree_benchmark_loom=Path(require_tool('iree-benchmark-loom', tool_dir=args.tool_dir)),
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

    row = run_candidate_row(run_args, bench_config, candidate, sanitizer='none')
    if row.get('status') != 'ran':
        raise RuntimeError(f"kernel run failed with status={row.get('status')} row={json.dumps(row, sort_keys=True)}")
    benchmark = row.get('benchmark') or {}
    summary = benchmark.get('summary') or {}
    correctness = summary.get('correctness')
    if not correctness_ok(correctness):
        raise RuntimeError(
            'kernel correctness check failed: '
            + json.dumps(
                {
                    'candidate_id': candidate.id,
                    'correctness': correctness,
                    'failure': summary.get('failure'),
                    'results_path': benchmark.get('results_path'),
                    'output_dir': str(output_dir),
                },
                sort_keys=True,
            )
        )

    print(
        json.dumps(
            {
                'candidate_id': candidate.id,
                'case_id': args.case_id,
                'correctness': correctness,
                'results_path': benchmark.get('results_path'),
                'output_dir': str(output_dir),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
