from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

from .import_models import ImportedCase


POINTWISE_ROUTE_SOURCES = frozenset(
    {
        "shape.ncols",
        "shape.nrows",
        "shape.pointwise.src0_row_stride",
        "shape.pointwise.src1_row_stride",
        "shape.pointwise.src1_ncols",
    }
)

COPY_ROUTE_SOURCES = frozenset({"shape.copy.n"})


def _params(case: ImportedCase) -> Mapping[str, Any]:
    return case.normalized_params


def _int_list(case: ImportedCase, key: str) -> list[int]:
    raw = _params(case).get(key)
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"expected a non-empty list for {key}")
    values: list[int] = []
    for index, value in enumerate(raw):
        if not isinstance(value, int):
            raise ValueError(f"{key}[{index}] must be an integer")
        values.append(value)
    return values


def _int_value(case: ImportedCase, key: str, *, default: int = 0) -> int:
    raw = _params(case).get(key, default)
    if not isinstance(raw, int):
        raise ValueError(f"{key} must be an integer")
    return raw


def _all_equal(values: Iterable[int], expected: int) -> bool:
    return all(value == expected for value in values)


def _binding_sources(route: Mapping[str, Any]) -> frozenset[str]:
    bindings = (route.get("specialization") or {}).get("bindings") or ()
    sources = {
        str(binding["source"])
        for binding in bindings
        if isinstance(binding, dict) and "source" in binding
    }
    return frozenset(sources)


def _route_layout(route: Mapping[str, Any]) -> str:
    supports = route.get("supports") or {}
    if not isinstance(supports, dict):
        return ""
    return str(supports.get("layout") or "")


def _identity_permutation(case: ImportedCase, key: str) -> bool:
    return _all_equal(_int_list(case, key), 0)


def _pointwise_contract(route: Mapping[str, Any]) -> bool:
    sources = _binding_sources(route)
    return bool(sources) and sources.issubset(POINTWISE_ROUTE_SOURCES)


def _copy_contract(route: Mapping[str, Any]) -> bool:
    return _binding_sources(route) == COPY_ROUTE_SOURCES


@dataclass(frozen=True)
class PointwiseCaseFacts:
    ne: tuple[int, int, int, int]
    nr: tuple[int, int, int, int]
    nf: int
    perm1: int

    @property
    def ncols(self) -> int:
        return self.ne[0]

    @property
    def nrows(self) -> int:
        return math.prod(self.ne[1:])


def _pointwise_facts(case: ImportedCase) -> PointwiseCaseFacts:
    ne = _int_list(case, "ne")
    nr = _int_list(case, "nr")
    nf = _int_value(case, "nf", default=0)
    perm1 = _int_value(case, "perm1", default=0)
    if len(ne) != 4 or len(nr) != 4:
        raise ValueError("pointwise lowering requires 4-D extents")
    return PointwiseCaseFacts(
        ne=(ne[0], ne[1], ne[2], ne[3]),
        nr=(nr[0], nr[1], nr[2], nr[3]),
        nf=nf,
        perm1=perm1,
    )


def _pointwise_contiguous_shape(facts: PointwiseCaseFacts) -> dict[str, int]:
    if facts.nf != 1:
        raise ValueError("pointwise contiguous layout requires nf=1")
    if facts.perm1 != 0:
        raise ValueError("pointwise contiguous layout requires perm1=0")
    if not _all_equal(facts.nr, 1):
        raise ValueError("pointwise contiguous layout requires same-shape inputs")
    return {"ncols": facts.ncols, "nrows": facts.nrows}


def _pointwise_rhs_column_broadcast_shape(facts: PointwiseCaseFacts) -> dict[str, int]:
    if facts.nf != 1:
        raise ValueError("pointwise rhs column broadcast requires nf=1")
    if facts.perm1 != 0:
        raise ValueError("pointwise rhs column broadcast requires perm1=0")
    if facts.ne[0] != 1 or facts.nr[0] <= 1:
        raise ValueError(
            "pointwise rhs column broadcast requires a single-column rhs source"
        )
    if not _all_equal(facts.nr[1:], 1):
        raise ValueError(
            "pointwise rhs column broadcast requires column-only repetition"
        )
    return {
        "ncols": facts.nr[0],
        "nrows": facts.nrows,
        "src1_row_stride": 1,
        "src1_ncols": 1,
    }


POINTWISE_LAYOUT_LOWERERS: dict[
    str, Callable[[PointwiseCaseFacts], dict[str, int]]
] = {
    "contiguous": _pointwise_contiguous_shape,
    "contiguous_or_rhs_row_broadcast": _pointwise_contiguous_shape,
    "contiguous_src0_rhs_column_broadcast": _pointwise_rhs_column_broadcast_shape,
}


def _lower_pointwise_case(case: ImportedCase, route: Mapping[str, Any]) -> dict[str, int]:
    facts = _pointwise_facts(case)
    layout = _route_layout(route)
    lower = POINTWISE_LAYOUT_LOWERERS.get(layout)
    if lower is None:
        raise ValueError(f"no pointwise lowering is implemented for layout {layout!r}")
    return lower(facts)


def _lower_copy_case(case: ImportedCase, route: Mapping[str, Any]) -> dict[str, int]:
    layout = _route_layout(route)
    if layout != "contiguous_src_to_contiguous_dst":
        raise ValueError(f"no copy lowering is implemented for layout {layout!r}")
    ne = _int_list(case, "ne")
    src_transpose = _int_value(case, "_src_transpose", default=0)
    if len(ne) != 4:
        raise ValueError("copy lowering requires 4-D extents")
    if src_transpose != 0:
        raise ValueError("copy lowering requires non-transposed inputs")
    if not _identity_permutation(case, "permute_src"):
        raise ValueError("copy lowering requires identity source permutation")
    if not _identity_permutation(case, "permute_dst"):
        raise ValueError("copy lowering requires identity destination permutation")
    return {"nrows": 1, "ncols": math.prod(ne)}


ROUTE_CONTRACT_LOWERERS: dict[
    str, Callable[[ImportedCase, Mapping[str, Any]], dict[str, int]]
] = {
    "pointwise": _lower_pointwise_case,
    "copy": _lower_copy_case,
}


def route_contract(route: Mapping[str, Any]) -> str | None:
    if _pointwise_contract(route):
        return "pointwise"
    if _copy_contract(route):
        return "copy"
    return None


def lower_case_for_route(case: ImportedCase, route: Mapping[str, Any]) -> dict[str, int]:
    contract = route_contract(route)
    if contract is None:
        raise ValueError(
            "no raw-case lowering is implemented for this route specialization"
        )
    lower = ROUTE_CONTRACT_LOWERERS.get(contract)
    if lower is None:
        raise ValueError(f"no raw-case lowering is implemented for route contract {contract!r}")
    return lower(case, route)
