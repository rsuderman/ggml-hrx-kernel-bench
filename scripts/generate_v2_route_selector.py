from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from string import Template
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ggml_hrx_kernel_bench.routing.v2.catalog import (  # noqa: E402
    load_route_file,
    load_route_index,
)
from ggml_hrx_kernel_bench.routing.v2.models import (  # noqa: E402
    ConstraintCheck,
    V2Route,
    ValueDefinition,
)


_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1
_SUPPORTED_VALUE_KINDS = {
    "chain_permutations": "chain_permutations",
    "contiguous_strides": "contiguous_strides",
    "element": "element",
    "head": "head",
    "inverse_permutation": "inverse_permutation",
    "permuted_contiguous_strides": "permuted_contiguous_strides",
    "product": "product",
    "tail": "tail",
}
_ROUTE_TEMPLATE = Template(
    (
        REPO_ROOT / "native" / "v2_route_selector" / "templates" / "route_descriptor.inc.cpp.tmpl"
    ).read_text(encoding="utf-8")
)


class UnsupportedRouteDescriptor(RuntimeError):
    pass


def _require_int64(value: int, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise UnsupportedRouteDescriptor(f"{context} must be an integer")
    if value < _INT64_MIN or value > _INT64_MAX:
        raise UnsupportedRouteDescriptor(f"{context} is outside the native int64 range")
    return value


def _require_size(value: int, *, context: str, positive: bool = False) -> int:
    normalized = _require_int64(value, context=context)
    minimum = 1 if positive else 0
    if normalized < minimum:
        qualifier = "positive" if positive else "non-negative"
        raise UnsupportedRouteDescriptor(f"{context} must be {qualifier}")
    return normalized


def _cpp_int64(value: int) -> str:
    if value == _INT64_MIN:
        return "std::numeric_limits<std::int64_t>::min()"
    return str(value)


def _cpp_optional_int64(value: int | None, *, context: str) -> str:
    if value is None:
        return "std::nullopt"
    return _cpp_int64(_require_int64(value, context=context))


def _cpp_optional_size(value: int | None, *, context: str) -> str:
    if value is None:
        return "std::nullopt"
    return str(_require_size(value, context=context, positive=True))


def _cpp_optional_string(value: str | None) -> str:
    if value is None:
        return "std::nullopt"
    return json.dumps(value)


def _render_value(route: V2Route, value: ValueDefinition) -> str:
    kind = _SUPPORTED_VALUE_KINDS.get(value.operation_kind)
    if kind is None:
        raise UnsupportedRouteDescriptor(
            f"route {route.id!r} uses unsupported value operation {value.operation_kind!r}"
        )
    expected_sources = (
        2
        if value.operation_kind
        in {"chain_permutations", "permuted_contiguous_strides"}
        else 1
    )
    if len(value.sources) != expected_sources:
        raise UnsupportedRouteDescriptor(
            f"route {route.id!r} value {value.name!r} must have exactly "
            f"{expected_sources} source(s)"
        )

    fields = [
        json.dumps(value.name),
        f"ValueKind::{kind}",
        "{" + ", ".join(json.dumps(source) for source in value.sources) + "}",
    ]
    if value.operation_kind in {"element", "head", "tail"}:
        if len(value.parameters) != 1:
            raise UnsupportedRouteDescriptor(
                f"route {route.id!r} value {value.name!r} must have exactly one parameter"
            )
        fields.append(
            str(
                _require_size(
                    value.parameters[0],
                    context=f"route {route.id!r} value {value.name!r} parameter",
                )
            )
        )
    elif value.parameters:
        raise UnsupportedRouteDescriptor(
            f"route {route.id!r} value {value.name!r} has unsupported parameters"
        )
    return "{" + ", ".join(fields) + "}"


def _constraint_has_only(
    check: ConstraintCheck,
    *,
    allowed: set[str],
) -> bool:
    populated = {
        "name": check.name is not None,
        "length": check.length is not None,
        "rank_min": check.rank_min is not None,
        "rank_max": check.rank_max is not None,
        "index": check.index is not None,
        "min": check.min is not None,
        "max": check.max is not None,
        "multiple_of": check.multiple_of is not None,
        "iota": check.iota,
        "equals": bool(check.equals),
        "divides": bool(check.divides),
    }
    return all(not present or field in allowed for field, present in populated.items())


def _render_constraint(route: V2Route, check: ConstraintCheck) -> str:
    context = f"route {route.id!r} constraint"

    if check.equals:
        if len(check.equals) < 2 or not _constraint_has_only(check, allowed={"equals"}):
            raise UnsupportedRouteDescriptor(f"{context} has an unsupported equals form")
        names = ", ".join(json.dumps(name) for name in check.equals)
        return f"equals({{{names}}})"

    if check.divides:
        if len(check.divides) < 2 or not _constraint_has_only(check, allowed={"divides"}):
            raise UnsupportedRouteDescriptor(f"{context} has an unsupported divides form")
        names = ", ".join(json.dumps(name) for name in check.divides)
        return f"divides({{{names}}})"

    if check.name is None:
        raise UnsupportedRouteDescriptor(f"{context} is missing a capture name")

    name = json.dumps(check.name)
    if check.iota:
        if not _constraint_has_only(check, allowed={"name", "iota"}):
            raise UnsupportedRouteDescriptor(f"{context} has an unsupported iota form")
        return f"iota({name})"

    if check.length is not None:
        if not _constraint_has_only(check, allowed={"name", "length"}):
            raise UnsupportedRouteDescriptor(f"{context} has an unsupported length form")
        length = _require_size(check.length, context=f"{context} length")
        return f"exact_length({name}, {length})"

    if check.rank_min is not None or check.rank_max is not None:
        if not _constraint_has_only(check, allowed={"name", "rank_min", "rank_max"}):
            raise UnsupportedRouteDescriptor(f"{context} has an unsupported rank form")
        minimum = _cpp_optional_size(check.rank_min, context=f"{context} rank_min")
        maximum = _cpp_optional_size(check.rank_max, context=f"{context} rank_max")
        return f"rank_range({name}, {minimum}, {maximum})"

    if check.min is None and check.max is None and check.multiple_of is None:
        raise UnsupportedRouteDescriptor(f"{context} has no supported predicate")

    minimum = _cpp_optional_int64(check.min, context=f"{context} min")
    maximum = _cpp_optional_int64(check.max, context=f"{context} max")
    multiple_of = _cpp_optional_int64(
        check.multiple_of,
        context=f"{context} multiple_of",
    )
    if check.index is not None:
        if not _constraint_has_only(
            check,
            allowed={"name", "index", "min", "max", "multiple_of"},
        ):
            raise UnsupportedRouteDescriptor(f"{context} has an unsupported indexed-bounds form")
        index = _require_size(check.index, context=f"{context} index")
        return (
            f"indexed_bounds({name}, {index}, {minimum}, {maximum}, "
            f"{multiple_of})"
        )

    if not _constraint_has_only(
        check,
        allowed={"name", "min", "max", "multiple_of"},
    ):
        raise UnsupportedRouteDescriptor(f"{context} has an unsupported scalar-bounds form")
    return f"scalar_bounds({name}, {minimum}, {maximum}, {multiple_of})"


def _constraint_references_attribute(check: ConstraintCheck) -> bool:
    if check.name is not None and check.name.startswith("attribute."):
        return True
    return any(name.startswith("attribute.") for name in (*check.equals, *check.divides))


def _render_entries(entries: Iterable[str]) -> str:
    return "\n".join(f"                {entry}," for entry in entries)


def _render_route(route: V2Route) -> list[str]:
    tensors: list[str] = []
    for role, descriptor in route.tensors.items():
        tensors.append(
            "{"
            + ", ".join(
                (
                    json.dumps(role),
                    _cpp_optional_string(descriptor.dtype),
                    json.dumps(descriptor.dimensions_capture),
                    json.dumps(descriptor.strides_capture),
                    _cpp_optional_string(descriptor.permutation_capture),
                )
            )
            + "}"
        )

    values = [_render_value(route, value) for value in route.values]
    constraints = [
        _render_constraint(route, check)
        for check in route.constraints.checks
        if not _constraint_references_attribute(check)
    ]
    return _ROUTE_TEMPLATE.substitute(
        route_id=json.dumps(route.id),
        tensors=_render_entries(tensors),
        values=_render_entries(values),
        constraints=_render_entries(constraints),
    ).splitlines()


def _normalize_operations(raw_operations: Iterable[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for raw_operation in raw_operations:
        operation = str(raw_operation).strip().upper()
        if not operation:
            raise ValueError("--op values must not be empty")
        normalized.append(operation)
    if not normalized:
        raise ValueError("at least one --op value is required")
    duplicates = sorted(
        operation for operation in set(normalized) if normalized.count(operation) > 1
    )
    if duplicates:
        raise ValueError(f"duplicate requested operations: {duplicates!r}")
    return tuple(sorted(normalized))


def render_route_table(
    routing_dir: Path,
    operations: Sequence[str] | None = None,
    *,
    all_operations: bool = False,
) -> str:
    route_index = load_route_index(routing_dir)
    if all_operations:
        if operations:
            raise ValueError("--all cannot be combined with --op")
        normalized_operations = tuple(sorted(route_index))
    else:
        normalized_operations = _normalize_operations(operations or ())

    operation_routes: list[tuple[str, list[list[str]]]] = []
    seen_route_ids: dict[str, str] = {}

    for operation in normalized_operations:
        route_files = route_index.get(operation)
        if route_files is None:
            raise RuntimeError(
                f"requested operation {operation!r} is missing from {routing_dir / 'router.json'}"
            )
        rendered_routes: list[list[str]] = []
        for route_file in route_files:
            route = load_route_file(
                routing_dir,
                op=operation,
                route_file_name=route_file,
            )
            previous_operation = seen_route_ids.get(route.id)
            if previous_operation is not None:
                raise RuntimeError(
                    f"duplicate route id {route.id!r} in operations "
                    f"{previous_operation!r} and {operation!r}"
                )
            seen_route_ids[route.id] = operation
            # Render every descriptor now so one malformed route rejects the
            # entire requested table before the output path is touched.
            rendered_routes.append(_render_route(route))
        operation_routes.append((operation, rendered_routes))

    lines = [
        "// Generated by scripts/generate_v2_route_selector.py. Do not edit.",
        "",
    ]
    for operation_index, (operation, rendered_routes) in enumerate(operation_routes):
        if operation_index:
            lines.append("")
        lines.extend(
            [
                f"    // Routes for operation {json.dumps(operation)}.",
                f"    {{{json.dumps(operation)},",
                "     {",
            ]
        )
        for rendered_route in rendered_routes:
            lines.extend(rendered_route)
        lines.extend(["     }},"])
    return "\n".join(lines) + "\n"


def _write_if_changed(path: Path, contents: str) -> None:
    if path.is_file() and path.read_text(encoding="utf-8") == contents:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def generate_route_table(
    *,
    routing_dir: Path,
    output: Path,
    operations: Sequence[str] | None = None,
    all_operations: bool = False,
) -> None:
    contents = render_route_table(
        routing_dir,
        operations,
        all_operations=all_operations,
    )
    _write_if_changed(output, contents)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate native C++ v2 route descriptor initializers."
    )
    parser.add_argument("--routing-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--all", action="store_true", dest="all_operations")
    mode.add_argument("--op", action="append", dest="operations")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        generate_route_table(
            routing_dir=args.routing_dir,
            output=args.output,
            operations=args.operations,
            all_operations=args.all_operations,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"generate_v2_route_selector.py: error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
