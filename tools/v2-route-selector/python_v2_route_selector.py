from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import sys
from pathlib import Path
from typing import TextIO


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

_BLANK_LINE_CHARACTERS = " \t\r\n\v\f"

from ggml_hrx_kernel_bench.route_query_parser import parse_route_query_json  # noqa: E402
from ggml_hrx_kernel_bench.routing.v2.query import (  # noqa: E402
    RouteCatalog,
    load_route_catalog,
)
from ggml_hrx_kernel_bench.routing.v2.selection import (  # noqa: E402
    RouteQuery,
    select_route_query,
)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select the first matching Python v2 route for RouteQuery JSON input."
    )
    parser.add_argument("--input", required=True, help="RouteQuery JSON file, or '-' for stdin")
    parser.add_argument("--routing-dir", type=Path, required=True)
    parser.add_argument(
        "--batch",
        action="store_true",
        help="read RouteQuery JSONL and write one result record per nonblank line",
    )
    return parser.parse_args(argv)


def _load_query(input_path: str, standard_input: TextIO) -> RouteQuery:
    if input_path == "-":
        text = standard_input.read()
    else:
        text = Path(input_path).read_text(encoding="utf-8")
    return parse_route_query_json(text)


def _load_catalog(routing_dir: Path) -> RouteCatalog:
    router_path = routing_dir / "router.json"
    if not router_path.is_file():
        raise OSError(f"cannot read routing catalog {router_path!s}")
    return load_route_catalog(routing_dir)


def _error_diagnostic(exc: Exception) -> str:
    return str(exc) or type(exc).__name__


def _batch_result(catalog: RouteCatalog, line_number: int, text: str) -> dict[str, object]:
    try:
        query = parse_route_query_json(text)
        selection = select_route_query(catalog, query)
        if selection.status == "matched":
            return {
                "line": line_number,
                "status": "MATCH",
                "route_id": selection.route_ids[0],
            }
        return {
            "line": line_number,
            "status": "NO_MATCH",
            "diagnostic": f"no route matched operation {query.operation!r}",
        }
    except Exception as exc:
        return {
            "line": line_number,
            "status": "ERROR",
            "diagnostic": _error_diagnostic(exc),
        }


def _run_batch(
    *,
    input_path: str,
    catalog: RouteCatalog,
    standard_input: TextIO,
    standard_output: TextIO,
) -> None:
    input_context = (
        nullcontext(standard_input)
        if input_path == "-"
        else Path(input_path).open("r", encoding="utf-8", newline="")
    )
    with input_context as input_stream:
        for line_number, text in enumerate(input_stream, start=1):
            if not text.strip(_BLANK_LINE_CHARACTERS):
                continue
            record = _batch_result(catalog, line_number, text)
            standard_output.write(json.dumps(record, separators=(",", ":")) + "\n")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.batch:
            catalog = _load_catalog(args.routing_dir)
            _run_batch(
                input_path=args.input,
                catalog=catalog,
                standard_input=sys.stdin,
                standard_output=sys.stdout,
            )
            try:
                sys.stdout.flush()
            except (OSError, ValueError):
                # Prevent interpreter shutdown from retrying a failed buffered
                # flush and replacing the intended exit status with 120.
                sys.stdout = None
                raise
            return 0

        query = _load_query(args.input, sys.stdin)
        catalog = _load_catalog(args.routing_dir)
        selection = select_route_query(catalog, query)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"python_v2_route_selector.py: error: {exc}", file=sys.stderr)
        return 2

    if selection.status == "unmatched":
        print(
            f"python_v2_route_selector.py: error: NO_MATCH: "
            f"no route matched operation {query.operation!r}",
            file=sys.stderr,
        )
        return 1

    print(selection.route_ids[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
