from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .import_mapping_registry import lower_case_for_route
from .import_models import ImportedCase, UnmappedReason


@dataclass(frozen=True)
class RouteResolution:
    route: dict[str, Any]
    shape: dict[str, int]


_DEFAULT_AXIS_VALUES: dict[str, tuple[int, ...]] = {
    "k": (256, 512, 1024, 2048, 3072, 4096, 5120, 6144, 8192, 11008, 14336, 16384),
    "rows": (1, 2, 8, 16, 32, 64, 65, 127, 128, 129, 256, 512, 1024, 2048, 4096),
    "cols": (1, 2, 8, 16, 32, 33, 63, 64, 65, 127, 128, 129, 256, 512, 1024),
    "ncols": (1, 8, 16, 32, 64, 65, 127, 128, 129, 256, 512, 1024, 2048, 4096),
    "nrows": (1, 2, 8, 16, 32, 33, 63, 64, 65, 127, 128, 129, 256, 512, 1024),
    "n_dims": (32, 64, 80, 96, 128),
}


@dataclass(frozen=True)
class _ShapeDomain:
    family: str
    route_id: str
    root_symbol: str
    domain: Mapping[str, Any]
    guards: Mapping[str, Any]

    @property
    def axes(self) -> tuple[str, ...]:
        return tuple(name for name in _DEFAULT_AXIS_VALUES if self.has_axis(name))

    def has_axis(self, name: str) -> bool:
        return f"{name}_min" in self.domain or f"{name}_max" in self.domain

    def bounds(self, name: str) -> tuple[int, int]:
        defaults = _DEFAULT_AXIS_VALUES[name]
        lo = self.domain.get(f"{name}_min")
        hi = self.domain.get(f"{name}_max")
        return (
            int(lo if lo is not None else min(defaults)),
            int(hi if hi is not None else max(defaults)),
        )

    def multiple(self, name: str) -> int | None:
        value = self.guards.get(f"{name}_multiple_of")
        if not value:
            return None
        value_i = int(value)
        return value_i if value_i > 1 else None

    def accepts(self, shape: Mapping[str, int]) -> bool:
        for axis in self.axes:
            if axis not in shape:
                return False
            value = int(shape[axis])
            lo, hi = self.bounds(axis)
            multiple = self.multiple(axis)
            if value < lo or value > hi:
                return False
            if multiple and value % multiple != 0:
                return False
        return True


def _normalize_shape(shape: Mapping[str, int]) -> dict[str, int]:
    normalized = {str(key): int(value) for key, value in shape.items()}
    if "ncols" in normalized and "cols" not in normalized:
        normalized["cols"] = normalized["ncols"]
    if "cols" in normalized and "ncols" not in normalized:
        normalized["ncols"] = normalized["cols"]
    if "nrows" in normalized and "rows" not in normalized:
        normalized["rows"] = normalized["nrows"]
    if "rows" in normalized and "nrows" not in normalized:
        normalized["nrows"] = normalized["rows"]
    return normalized


def route_family(route: Mapping[str, Any]) -> str:
    return str(route.get("family") or route.get("source_id") or "")


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _shape_value(shape: Mapping[str, int], *names: str, default: int = 0) -> int:
    for name in names:
        if name in shape:
            return int(shape[name])
    return default


def _pointwise_guard_value(name: str, shape: Mapping[str, int]) -> int:
    if name == "src0_row_stride":
        return _shape_value(shape, "src0_row_stride", "ncols", "cols", default=1)
    if name == "src1_row_stride":
        return _shape_value(shape, "src1_row_stride", "ncols", "cols", default=1)
    if name == "src1_ncols":
        return _shape_value(shape, "src1_ncols", "ncols", "cols", default=1)
    raise KeyError(name)


def _shape_guard_value(guard: str, shape: Mapping[str, int]) -> bool:
    if guard == "all_pot":
        return all(_is_power_of_two(int(value)) for value in shape.values())
    if guard == "k_pow2":
        return _is_power_of_two(
            int(shape.get("k", shape.get("ncols", shape.get("cols", 0))))
        )
    if guard == "pointwise_src0_row_stride_eq_ncols":
        return _pointwise_guard_value("src0_row_stride", shape) == _shape_value(
            shape, "ncols", "cols"
        )
    if guard == "pointwise_src1_row_stride_eq_ncols":
        return _pointwise_guard_value("src1_row_stride", shape) == _shape_value(
            shape, "ncols", "cols"
        )
    if guard == "pointwise_src1_row_stride_eq_zero":
        return _pointwise_guard_value("src1_row_stride", shape) == 0
    if guard == "pointwise_src1_ncols_eq_ncols":
        return _pointwise_guard_value("src1_ncols", shape) == _shape_value(
            shape, "ncols", "cols"
        )
    return True


def _shape_guard_satisfied(
    guard: str, expected: Any, shape: Mapping[str, int]
) -> bool:
    if not isinstance(expected, bool):
        return True
    actual = _shape_guard_value(guard, shape)
    return actual if expected else not actual


def route_accepts_shape(route: Mapping[str, Any], shape: dict[str, int]) -> bool:
    domain = route.get("shape_domain") or {}
    if not isinstance(domain, dict) or not domain:
        return True
    parsed_guards = route.get("shape_guards") or {}
    if not isinstance(parsed_guards, dict):
        parsed_guards = {}
    ctx = _ShapeDomain(
        family=route_family(route),
        route_id=str(route.get("id") or ""),
        root_symbol=str(route.get("root_symbol") or ""),
        domain=domain,
        guards=parsed_guards,
    )
    return ctx.accepts(shape) and all(
        _shape_guard_satisfied(guard, expected, shape)
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
            shape = _normalize_shape(lower_case_for_route(case, route))
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
