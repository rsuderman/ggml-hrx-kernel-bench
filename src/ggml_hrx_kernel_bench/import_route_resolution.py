from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .family_specs import ShapeDomain, normalize_shape, resolve_binding_value
from .import_mapping_registry import lower_case_for_route
from .import_models import ImportedCase, UnmappedReason


@dataclass(frozen=True)
class RouteResolution:
    route: dict[str, Any]
    shape: dict[str, int]


def route_family(route: Mapping[str, Any]) -> str:
    return str(route.get("family") or route.get("source_id") or "")


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _shape_guard_value(family: str, guard: str, shape: Mapping[str, int]) -> bool:
    if guard == "all_pot":
        return all(_is_power_of_two(int(value)) for value in shape.values())
    if guard == "k_pow2":
        return _is_power_of_two(
            int(shape.get("k", shape.get("ncols", shape.get("cols", 0))))
        )
    if guard == "pointwise_src0_row_stride_eq_ncols":
        return int(
            shape.get(
                "src0_row_stride",
                resolve_binding_value(family, "pointwise.src0_row_stride", shape) or 0,
            )
        ) == int(shape.get("ncols", 0))
    if guard == "pointwise_src1_row_stride_eq_ncols":
        return int(
            shape.get(
                "src1_row_stride",
                resolve_binding_value(family, "pointwise.src1_row_stride", shape) or 0,
            )
        ) == int(shape.get("ncols", 0))
    if guard == "pointwise_src1_row_stride_eq_zero":
        return int(
            shape.get(
                "src1_row_stride",
                resolve_binding_value(family, "pointwise.src1_row_stride", shape) or 0,
            )
        ) == 0
    if guard == "pointwise_src1_ncols_eq_ncols":
        return int(
            shape.get(
                "src1_ncols",
                resolve_binding_value(family, "pointwise.src1_ncols", shape) or 0,
            )
        ) == int(shape.get("ncols", 0))
    return True


def _shape_guard_satisfied(
    family: str, guard: str, expected: Any, shape: Mapping[str, int]
) -> bool:
    if not isinstance(expected, bool):
        return True
    actual = _shape_guard_value(family, guard, shape)
    return actual if expected else not actual


def route_accepts_shape(route: Mapping[str, Any], shape: dict[str, int]) -> bool:
    domain = route.get("shape_domain") or {}
    if not isinstance(domain, dict) or not domain:
        return True
    parsed_guards = route.get("shape_guards") or {}
    if not isinstance(parsed_guards, dict):
        parsed_guards = {}
    ctx = ShapeDomain(
        family=route_family(route),
        route_id=str(route.get("id") or ""),
        root_symbol=str(route.get("root_symbol") or ""),
        domain=domain,
        guards=parsed_guards,
    )
    return ctx.accepts(shape) and all(
        _shape_guard_satisfied(ctx.family, guard, expected, shape)
        for guard, expected in parsed_guards.items()
    )


def _catalog_type_label(value: Any) -> str:
    return str(value).upper()


def _supports_type(
    supports: Mapping[str, Any], support_key: str, actual_value: Any
) -> bool:
    expected_value = supports.get(support_key)
    return expected_value is None or _catalog_type_label(
        expected_value
    ) == _catalog_type_label(actual_value)


def route_supports_case_dtype(route: Mapping[str, Any], case: ImportedCase) -> bool:
    supports = route.get("supports") or {}
    if not isinstance(supports, dict) or not supports:
        return True
    dtype = case.dtype
    if "type" in dtype:
        return all(
            _supports_type(supports, support_key, dtype["type"])
            for support_key in ("src0_type", "src1_type", "src2_type", "dst_type")
        )
    return all(
        dtype_key not in dtype
        or _supports_type(supports, support_key, dtype[dtype_key])
        for dtype_key, support_key in (
            ("type_src", "src0_type"),
            ("type_dst", "dst_type"),
        )
    )


def resolve_case_routes(
    case: ImportedCase,
    routes: Iterable[dict[str, Any]],
) -> tuple[
    RouteResolution | None,
    list[dict[str, Any]],
    UnmappedReason | None,
    str | None,
]:
    op_routes = list(routes)
    if not op_routes:
        return (
            None,
            [],
            UnmappedReason.NO_KERNEL_FAMILY_MAPPING,
            "no catalog route exists for this op",
        )
    dtype_matching = [
        route for route in op_routes if route_supports_case_dtype(route, case)
    ]
    if not dtype_matching:
        return (
            None,
            op_routes,
            UnmappedReason.NO_DTYPE_MAPPING,
            "matching op mapping exists, but not for this dtype combination",
        )

    lowered: list[RouteResolution] = []
    lowering_errors: list[str] = []
    for route in dtype_matching:
        try:
            shape = normalize_shape(lower_case_for_route(case, route))
        except ValueError as exc:
            lowering_errors.append(str(exc))
            continue
        lowered.append(RouteResolution(route=route, shape=shape))

    matching = [
        current for current in lowered if route_accepts_shape(current.route, current.shape)
    ]
    if not matching:
        if lowered:
            return (
                None,
                [current.route for current in lowered],
                UnmappedReason.NO_ROUTE_MATCH,
                "lowered shape did not satisfy any catalog route",
            )
        detail = lowering_errors[0] if lowering_errors else (
            "matching catalog op exists, but no raw-case lowering is implemented for its routes"
        )
        return (
            None,
            dtype_matching,
            UnmappedReason.SHAPE_LOWERING_NOT_IMPLEMENTED,
            detail,
        )

    matching.sort(
        key=lambda current: (
            -int(current.route.get("priority", 0) or 0),
            str(current.route.get("id") or ""),
        )
    )
    best_priority = int(matching[0].route.get("priority", 0) or 0)
    best_matches = [
        current
        for current in matching
        if int(current.route.get("priority", 0) or 0) == best_priority
    ]
    if len(best_matches) > 1:
        return (
            None,
            [current.route for current in best_matches],
            UnmappedReason.AMBIGUOUS_ROUTE_MATCH,
            None,
        )
    return best_matches[0], [current.route for current in matching], None, None
