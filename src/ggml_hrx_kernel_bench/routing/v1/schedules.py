from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any


DEFAULT_AXIS_VALUES: dict[str, tuple[int, ...]] = {
    "k": (256, 512, 1024, 2048, 3072, 4096, 5120, 6144, 8192, 11008, 14336, 16384),
    "rows": (1, 2, 8, 16, 32, 64, 65, 127, 128, 129, 256, 512, 1024, 2048, 4096),
    "cols": (1, 2, 8, 16, 32, 33, 63, 64, 65, 127, 128, 129, 256, 512, 1024),
    "ncols": (1, 8, 16, 32, 64, 65, 127, 128, 129, 256, 512, 1024, 2048, 4096),
    "nrows": (1, 2, 8, 16, 32, 33, 63, 64, 65, 127, 128, 129, 256, 512, 1024),
    "n_dims": (32, 64, 80, 96, 128),
}


@dataclass(frozen=True)
class ShapePoint:
    family: str
    scenario: str
    facts: dict[str, int]
    source: str
    weight: float = 1.0
    route_id: str | None = None
    root_symbol: str | None = None
    notes: tuple[str, ...] = ()

    def to_ledger(self) -> OrderedDict[str, Any]:
        row: OrderedDict[str, Any] = OrderedDict(
            [
                ("scenario", self.scenario),
                ("source", self.source),
                ("weight", self.weight),
                ("facts", self.facts),
            ]
        )
        if self.route_id:
            row["route_id"] = self.route_id
        if self.root_symbol:
            row["root_symbol"] = self.root_symbol
        if self.notes:
            row["notes"] = list(self.notes)
        return row


@dataclass(frozen=True)
class ShapeDomain:
    family: str
    route_id: str
    root_symbol: str
    domain: Mapping[str, Any]
    guards: Mapping[str, Any]

    @property
    def axes(self) -> tuple[str, ...]:
        return tuple(name for name in DEFAULT_AXIS_VALUES if self.has_axis(name))

    def has_axis(self, name: str) -> bool:
        return f"{name}_min" in self.domain or f"{name}_max" in self.domain

    def bounds(self, name: str) -> tuple[int, int]:
        defaults = DEFAULT_AXIS_VALUES[name]
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

    def align(self, name: str, value: int, *, direction: str) -> int:
        multiple = self.multiple(name)
        if not multiple:
            return value
        if direction == "up":
            return math.ceil(value / multiple) * multiple
        return math.floor(value / multiple) * multiple

    def choose(self, name: str, preferred: tuple[int, ...], *, fallback: str) -> int | None:
        if not self.has_axis(name):
            return None
        lo, hi = self.bounds(name)
        multiple = self.multiple(name)
        for value in preferred:
            if lo <= value <= hi and (not multiple or value % multiple == 0):
                return value
        if fallback == "hi":
            value = self.align(name, hi, direction="down")
        elif fallback == "mid":
            value = self.align(name, (lo + hi) // 2, direction="down")
            if value < lo:
                value = self.align(name, (lo + hi) // 2, direction="up")
        else:
            value = self.align(name, lo, direction="up")
        return value if lo <= value <= hi else None

    def point(
        self,
        preferences: Mapping[str, tuple[int, ...]],
        *,
        fallback: str = "lo",
    ) -> dict[str, int] | None:
        shape: dict[str, int] = {}
        for axis in self.axes:
            value = self.choose(axis, preferences.get(axis, ()), fallback=fallback)
            if value is None:
                return None
            shape[axis] = value
        return normalize_shape(shape)

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


ScheduleProvider = Callable[[ShapeDomain], list[ShapePoint]]


@dataclass(frozen=True)
class FamilyScheduleSpec:
    family_ids: tuple[str, ...]
    provider: ScheduleProvider
    max_points: int = 6


def normalize_shape(shape: Mapping[str, int]) -> dict[str, int]:
    normalized = {str(key): int(value) for key, value in shape.items()}
    if "ncols" in normalized and "cols" not in normalized:
        normalized["cols"] = normalized["ncols"]
    if "nrows" in normalized and "rows" not in normalized:
        normalized["rows"] = normalized["nrows"]
    return normalized


def schedule_points_for_route(
    route: Mapping[str, Any],
    *,
    sweep: str,
    observed_shapes: Iterable[Mapping[str, int]] = (),
) -> list[ShapePoint]:
    domain = _shape_domain(route)
    if domain is None:
        return [
            ShapePoint(
                family=str(route.get("family") or route.get("source_id") or "unknown"),
                scenario="default",
                facts=default_shape_for_axisless_route(route),
                source="atlas-axisless-default",
            )
        ]
    if not domain.axes:
        return [
            ShapePoint(
                family=domain.family,
                scenario="default",
                facts=default_shape_for_axisless_route(route),
                source="atlas-axisless-default",
                route_id=domain.route_id,
                root_symbol=domain.root_symbol,
            )
        ]

    spec = SCHEDULE_SPECS_BY_FAMILY.get(domain.family, GENERIC_SCHEDULE)
    if sweep == "observed":
        observed = [
            _shape_point(domain, "observed", normalize_shape(shape), "observed")
            for shape in observed_shapes
            if domain.accepts(normalize_shape(shape))
        ]
        return _dedupe_points(observed[: spec.max_points] or _primary_points(domain))
    if sweep == "minimal":
        return _primary_points(domain)
    return _dedupe_points(spec.provider(domain))[: spec.max_points]


def concrete_shapes_for_route(
    route: Mapping[str, Any],
    *,
    sweep: str,
    observed_shapes: Iterable[Mapping[str, int]] = (),
) -> list[dict[str, int]]:
    return [point.facts for point in schedule_points_for_route(route, sweep=sweep, observed_shapes=observed_shapes)]


def schedule_for_shape(
    route: Mapping[str, Any],
    shape: Mapping[str, int],
    *,
    sweep: str,
    observed_shapes: Iterable[Mapping[str, int]] = (),
) -> ShapePoint:
    shape_key = tuple(sorted(normalize_shape(shape).items()))
    for point in schedule_points_for_route(route, sweep=sweep, observed_shapes=observed_shapes):
        if tuple(sorted(point.facts.items())) == shape_key:
            return point
    domain = _shape_domain(route)
    return ShapePoint(
        family=str(route.get("family") or route.get("source_id") or "unknown"),
        scenario="ad-hoc",
        facts=normalize_shape(shape),
        source="ad-hoc",
        route_id=domain.route_id if domain else str(route.get("id") or ""),
        root_symbol=domain.root_symbol if domain else str(route.get("root_symbol") or ""),
        notes=("shape was not emitted by the active schedule",),
    )


def default_shape_for_axisless_route(route: Mapping[str, Any]) -> dict[str, int]:
    family = str(route.get("family") or route.get("source_id") or "")
    defaults = {
        "quantize_q8_1_f32": {"ncols": 256, "nrows": 1, "cols": 256, "rows": 1},
    }
    return dict(defaults.get(family, {}))


def select_test_route(family_id: str, routes: list[dict[str, Any]]) -> dict[str, Any] | None:
    exact = [
        route
        for route in routes
        if isinstance(route.get("shape_domain"), dict)
        and any(
            str(key).endswith("_min")
            and route["shape_domain"].get(key) == route["shape_domain"].get(str(key)[:-4] + "_max")
            for key in route["shape_domain"]
        )
    ]
    candidates = exact or routes
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda route: (
            _route_priority_hint(family_id, route),
            _domain_cost(route.get("shape_domain") if isinstance(route.get("shape_domain"), dict) else {}),
            str(route.get("id", "")),
        ),
    )


def test_scenario_for_family(family_id: str, route: Mapping[str, Any]) -> str:
    scenarios = {
        "add_rms_norm_mul_f32": "add_rms_norm_mul",
        "cont_set_rows_f32": "cont_set_rows",
        "mul_mat_f16_f32_batched_cont": "mul_mat_f16_batched_cont",
        "mul_mat_q4_k_swiglu_f32": "mul_mat_q4_k_swiglu",
        "quantize_q8_1_f32": "quantize_q8_1",
        "rms_norm_mul_f32": "rms_norm_mul",
        "rope_set_rows_f32": "rope_set_rows",
        "set_rows_f32": "set_rows",
        "softmax_kqv_f32_f16": "softmax_kqv",
    }
    if family_id.startswith("mul_mat_id_"):
        return "mul_mat_id"
    if family_id in {"rope_f32", "rope_neox_f32", "rope_scale_f32"}:
        return "rope"
    return scenarios.get(family_id, str(route.get("op") or "").lower())


def test_shape_for_route(family_id: str, route: Mapping[str, Any]) -> OrderedDict[str, int] | None:
    domain = route.get("shape_domain") if isinstance(route.get("shape_domain"), dict) else {}

    def pick(name: str, default: int) -> int:
        min_value = domain.get(f"{name}_min", default)
        max_value = domain.get(f"{name}_max", min_value)
        if not isinstance(min_value, int) or not isinstance(max_value, int):
            return default
        return max(min(default, max_value), min_value)

    op = str(route.get("op") or "")
    if family_id == "cont_set_rows_f32":
        cont_ncols = 128
        cont_rows = 2
        set_rows_ncols = pick("ncols", 1)
        return OrderedDict(
            [
                ("cont_ncols", cont_ncols),
                ("cont_rows", cont_rows),
                ("ncols", set_rows_ncols),
                ("nrows", cont_ncols * cont_rows // max(set_rows_ncols, 1)),
                ("dst_rows", cont_ncols * cont_rows // max(set_rows_ncols, 1) + 2),
            ]
        )
    if family_id == "rope_set_rows_f32":
        return OrderedDict(
            [
                ("ncols", pick("ncols", 128)),
                ("n_dims", pick("n_dims", 128)),
                ("nheads", pick("rows", 8)),
                ("ntokens", pick("cols", 1)),
                ("dst_rows", max(pick("cols", 1) + 2, 4)),
            ]
        )
    if family_id == "mul_mat_q4_k_swiglu_f32":
        return OrderedDict([("k", pick("k", 256)), ("rows", pick("rows", 8)), ("cols", pick("cols", 1))])
    if family_id == "mul_mat_f16_f32_batched_cont":
        return OrderedDict(
            [
                ("k", pick("k", 768)),
                ("rows", pick("rows", 128)),
                ("cols", pick("cols", 1)),
                ("nheads", 24),
                ("nheads_kv", 8),
            ]
        )
    if family_id == "softmax_kqv_f32_f16":
        return OrderedDict(
            [
                ("kv", pick("k", 256)),
                ("n", pick("rows", 1)),
                ("nheads", pick("cols", 24)),
                ("nheads_kv", 8),
                ("d", 128),
            ]
        )
    if op == "MUL_MAT":
        return OrderedDict(
            [
                ("k", pick("k", 256)),
                ("rows", pick("rows", 64 if route.get("family") != "mul_mat_q4_k_f32" else 3)),
                ("cols", pick("cols", 1)),
            ]
        )
    if op == "MUL_MAT_ID":
        return OrderedDict(
            [
                ("k", pick("k", 1024)),
                ("rows", pick("rows", 768)),
                ("nselected", pick("cols", 8)),
                ("ntokens", pick("nrows", 2)),
                ("nexperts", 4),
            ]
        )
    if op in {"ADD", "MUL", "DIV", "SCALE", "CLAMP"}:
        return OrderedDict([("ncols", pick("ncols", 8)), ("nrows", pick("nrows", 64))])
    if op in {"RMS_NORM", "SOFT_MAX", "GLU", "ARGSORT"}:
        return OrderedDict([("ncols", pick("ncols", 128)), ("nrows", pick("nrows", 1))])
    if op == "SUM_ROWS":
        return OrderedDict([("ncols", pick("ncols", 8)), ("nrows", pick("nrows", 1))])
    if op == "GET_ROWS":
        supports = route.get("supports") if isinstance(route.get("supports"), Mapping) else {}
        if supports.get("layout") == "moe_weights_topk_view":
            nselected = pick("ncols", 8)
            ntokens = pick("nrows", 16)
            return OrderedDict(
                [
                    ("ncols", nselected),
                    ("nrows", ntokens),
                    ("nselected", nselected),
                    ("ntokens", ntokens),
                    ("nexperts", 128),
                ]
            )
        return OrderedDict([("ncols", pick("ncols", 2048)), ("nrows", pick("nrows", 3))])
    if op == "SET_ROWS":
        return OrderedDict(
            [("ncols", pick("ncols", 128)), ("nrows", pick("nrows", 2)), ("dst_rows", max(pick("nrows", 2) + 2, 4))]
        )
    if op == "QUANTIZE":
        return OrderedDict([("ncols", 256), ("nrows", 2)])
    if op in {"ROPE", "ROPE_SCALE"}:
        return OrderedDict(
            [
                ("ncols", pick("ncols", 128)),
                ("n_dims", pick("n_dims", 128)),
                ("nheads", pick("rows", 8)),
                ("ntokens", pick("cols", 1)),
                ("freq_scale", 2 if op == "ROPE_SCALE" else 1),
            ]
        )
    if op in {"CONT", "CPY"}:
        return OrderedDict([("ncols", pick("ncols", 257)), ("nrows", pick("nrows", 1))])
    return None


def _shape_domain(route: Mapping[str, Any]) -> ShapeDomain | None:
    domain = route.get("shape_domain")
    if not isinstance(domain, Mapping):
        return None
    return ShapeDomain(
        family=str(route.get("family") or route.get("source_id") or "unknown"),
        route_id=str(route.get("id") or ""),
        root_symbol=str(route.get("root_symbol") or ""),
        domain=domain,
        guards=route.get("shape_guards") if isinstance(route.get("shape_guards"), Mapping) else {},
    )


def _primary_points(domain: ShapeDomain) -> list[ShapePoint]:
    return _dedupe_points([_domain_point(domain, "smoke", "atlas-smoke", _smoke_preferences(), fallback="lo")])


def _shape_point(domain: ShapeDomain, scenario: str, shape: dict[str, int], source: str, weight: float = 1.0) -> ShapePoint:
    return ShapePoint(
        family=domain.family,
        scenario=scenario,
        facts=shape,
        source=source,
        weight=weight,
        route_id=domain.route_id,
        root_symbol=domain.root_symbol,
    )


def _domain_point(
    domain: ShapeDomain,
    scenario: str,
    source: str,
    preferences: Mapping[str, tuple[int, ...]],
    *,
    fallback: str,
    weight: float = 1.0,
) -> ShapePoint | None:
    shape = domain.point(preferences, fallback=fallback)
    if shape is None:
        return None
    return _shape_point(domain, scenario, shape, source, weight)


def _dedupe_points(points: Iterable[ShapePoint | None]) -> list[ShapePoint]:
    result: list[ShapePoint] = []
    seen: set[tuple[tuple[str, int], ...]] = set()
    for point in points:
        if point is None:
            continue
        key = tuple(sorted(point.facts.items()))
        if key in seen:
            continue
        seen.add(key)
        result.append(point)
    return result


def _generic_provider(domain: ShapeDomain) -> list[ShapePoint]:
    return _dedupe_points(
        [
            _domain_point(domain, "smoke", "atlas-smoke", _smoke_preferences(), fallback="lo", weight=1.0),
            _domain_point(domain, "mid", "atlas-midpoint", {}, fallback="mid", weight=0.5),
            _domain_point(domain, "max", "atlas-upper-bound", {}, fallback="hi", weight=0.25),
        ]
    )


def _matmul_provider(domain: ShapeDomain) -> list[ShapePoint]:
    return _dedupe_points(
        [
            _domain_point(domain, "decode-1", "llm-atlas", {"k": (3072, 4096, 8192), "rows": (1, 8), "cols": (1,)}, fallback="lo", weight=1.0),
            _domain_point(domain, "decode-small", "llm-atlas", {"k": (3072, 4096, 8192), "rows": (64, 128), "cols": (1, 8)}, fallback="mid", weight=0.9),
            _domain_point(domain, "prompt-128", "llm-atlas", {"k": (3072, 4096, 8192), "rows": (64, 128, 512), "cols": (128,)}, fallback="mid", weight=0.8),
            _domain_point(domain, "prompt-tail", "llm-atlas", {"k": (3072, 4096, 8192), "rows": (127, 129), "cols": (127, 129)}, fallback="mid", weight=0.7),
            _domain_point(domain, "wide-prefill", "llm-atlas", {"k": (4096, 8192, 11008), "rows": (512, 1024), "cols": (256, 512)}, fallback="mid", weight=0.6),
            _domain_point(domain, "max", "atlas-upper-bound", {}, fallback="hi", weight=0.25),
        ]
    )


def _row_major_provider(domain: ShapeDomain) -> list[ShapePoint]:
    return _dedupe_points(
        [
            _domain_point(domain, "decode-1", "llm-atlas", {"ncols": (128, 256, 4096), "nrows": (1,), "cols": (128, 256, 4096), "rows": (1,)}, fallback="lo", weight=1.0),
            _domain_point(domain, "short-prompt", "llm-atlas", {"ncols": (128, 256, 4096), "nrows": (8, 32), "cols": (128, 256, 4096), "rows": (8, 32)}, fallback="mid", weight=0.8),
            _domain_point(domain, "pot-128", "llm-atlas", {"ncols": (128, 256, 4096), "nrows": (128,), "cols": (128, 256, 4096), "rows": (128,)}, fallback="mid", weight=0.7),
            _domain_point(domain, "tail-129", "llm-atlas", {"ncols": (129, 257, 4097), "nrows": (129,), "cols": (129, 257, 4097), "rows": (129,)}, fallback="mid", weight=0.6),
            _domain_point(domain, "long", "llm-atlas", {"ncols": (4096, 8192), "nrows": (512, 1024), "cols": (4096, 8192), "rows": (512, 1024)}, fallback="hi", weight=0.4),
        ]
    )


def _rope_provider(domain: ShapeDomain) -> list[ShapePoint]:
    return _dedupe_points(
        [
            _domain_point(domain, "decode-1", "llm-atlas", {"ncols": (64, 80, 128), "n_dims": (64, 80, 128), "rows": (8, 16, 32), "cols": (1,)}, fallback="lo", weight=1.0),
            _domain_point(domain, "multi-token", "llm-atlas", {"ncols": (64, 80, 128), "n_dims": (64, 80, 128), "rows": (8, 16, 32), "cols": (8, 32)}, fallback="mid", weight=0.7),
            _domain_point(domain, "prompt-128", "llm-atlas", {"ncols": (64, 80, 128), "n_dims": (64, 80, 128), "rows": (8, 16, 32), "cols": (128,)}, fallback="mid", weight=0.6),
        ]
    )


def _moe_provider(domain: ShapeDomain) -> list[ShapePoint]:
    return _dedupe_points(
        [
            _domain_point(domain, "topk-decode", "llm-atlas", {"k": (1024, 2048), "rows": (768, 2048), "cols": (8,), "nrows": (1, 2)}, fallback="lo", weight=1.0),
            _domain_point(domain, "topk-prompt", "llm-atlas", {"k": (1024, 2048), "rows": (768, 2048), "cols": (8,), "nrows": (8, 32)}, fallback="mid", weight=0.7),
            _domain_point(domain, "topk-tail", "llm-atlas", {"k": (1024, 2048), "rows": (768, 2048), "cols": (7, 9), "nrows": (33,)}, fallback="mid", weight=0.4),
        ]
    )


def _smoke_preferences() -> dict[str, tuple[int, ...]]:
    return {
        "k": (256, 512, 1024),
        "rows": (1, 8, 16, 64, 128),
        "cols": (1, 8, 16, 64),
        "ncols": (64, 128, 256),
        "nrows": (1, 8, 16),
        "n_dims": (64, 80, 128),
    }


GENERIC_SCHEDULE = FamilyScheduleSpec(("__generic__",), _generic_provider, max_points=6)
MATMUL_SCHEDULE = FamilyScheduleSpec(
    (
        "mul_mat_q4_k_f32",
        "mul_mat_q5_k_f32",
        "mul_mat_q6_k_f32",
        "mul_mat_q8_0_f32",
        "mul_mat_f16_f32_batched",
        "mul_mat_f16_f32_batched_cont",
        "mul_mat_f16_f32_batched_kq_split_experiment",
    ),
    _matmul_provider,
    max_points=6,
)
MOE_SCHEDULE = FamilyScheduleSpec(
    ("mul_mat_id_q4_k_f32", "mul_mat_id_q5_k_f32", "mul_mat_id_q6_k_f32", "get_rows_moe_weights_f32"),
    _moe_provider,
    max_points=4,
)
ROW_MAJOR_SCHEDULE = FamilyScheduleSpec(
    (
        "add_f32",
        "argsort_f32_i32",
        "clamp_f32",
        "cont_f32",
        "cont_set_rows_f32",
        "copy_f32_f16",
        "div_f32",
        "get_rows_f32",
        "get_rows_q4_k_f32",
        "get_rows_q5_k_f32",
        "get_rows_q6_k_f32",
        "get_rows_q8_0_f32",
        "mul_f32",
        "quantize_q8_1_f32",
        "rms_norm_f32",
        "rms_norm_mul_f32",
        "add_rms_norm_mul_f32",
        "scale_f32",
        "set_rows_f32",
        "soft_max_f32",
        "sum_rows_f32",
        "swiglu_f32",
    ),
    _row_major_provider,
    max_points=5,
)
ROPE_SCHEDULE = FamilyScheduleSpec(
    ("rope_f32", "rope_neox_f32", "rope_scale_f32", "rope_set_rows_f32"),
    _rope_provider,
    max_points=4,
)

SCHEDULE_SPECS = (MATMUL_SCHEDULE, MOE_SCHEDULE, ROW_MAJOR_SCHEDULE, ROPE_SCHEDULE)
SCHEDULE_SPECS_BY_FAMILY: dict[str, FamilyScheduleSpec] = {
    family_id: spec for spec in SCHEDULE_SPECS for family_id in spec.family_ids
}


def _route_priority_hint(family_id: str, route: Mapping[str, Any]) -> int:
    route_id = str(route.get("id") or "")
    supports = route.get("supports") if isinstance(route.get("supports"), Mapping) else {}
    hints = {
        "mul_mat_q4_k_f32": ("_direct_",),
        "mul_mat_q8_0_f32": ("_packed_loop_",),
        "mul_mat_q4_k_swiglu_f32": ("mul_mat_q4_k_swiglu_f32_direct_",),
        "mul_mat_f16_f32_batched": ("_attention_wg256",),
        "mul_mat_f16_f32_batched_cont": ("decode",),
        "rope_f32": ("freq", "_t1_"),
        "rope_neox_f32": ("freq", "_t1_"),
        "rope_scale_f32": ("freq", "_t1_"),
    }
    required = hints.get(family_id)
    if required and all(text in route_id for text in required):
        return -10
    preferred_layouts = {
        "add_f32": {"contiguous", "contiguous_or_rhs_row_broadcast"},
        "mul_f32": {"contiguous"},
        "swiglu_f32": {"packed_contiguous", "contiguous_split_swiglu"},
    }.get(family_id)
    if preferred_layouts and supports.get("layout") in preferred_layouts:
        return -5
    return 0


def _domain_cost(domain: Mapping[str, Any]) -> int:
    cost = 0
    for key, value in domain.items():
        if not str(key).endswith("_min"):
            continue
        stem = str(key)[:-4]
        max_value = domain.get(f"{stem}_max", value)
        if isinstance(value, int) and isinstance(max_value, int):
            cost += max(value, 1) * max(max_value, 1)
    return cost
