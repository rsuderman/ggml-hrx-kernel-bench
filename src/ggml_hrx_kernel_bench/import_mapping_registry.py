from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

from .import_models import ImportedCase

POINTWISE_BASE_PERMUTATION = (0, 1, 2, 3)

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


def _pointwise_permutation(case: ImportedCase, key: str = "perm1") -> tuple[int, int, int, int]:
    raw = _params(case).get(key, list(POINTWISE_BASE_PERMUTATION))
    if not isinstance(raw, list) or len(raw) != 4:
        raise ValueError(f"{key} must be a 4-D permutation list")
    values = tuple(_int_list(case, key))
    if tuple(sorted(values)) != POINTWISE_BASE_PERMUTATION:
        raise ValueError(f"{key} must be a permutation of [0, 1, 2, 3]")
    return values


def _is_base_pointwise_permutation(permutation: tuple[int, int, int, int]) -> bool:
    return permutation == POINTWISE_BASE_PERMUTATION


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
    perm1: tuple[int, int, int, int]

    @property
    def src1_ncols(self) -> int:
        return self.ne[0]

    @property
    def src1_nrows(self) -> int:
        return math.prod(self.ne[1:])

    @property
    def src0_shape(self) -> tuple[int, int, int, int]:
        return tuple(self.ne[index] * self.nr[index] for index in range(4))

    @property
    def ncols(self) -> int:
        return self.src0_shape[0]

    @property
    def nrows(self) -> int:
        return math.prod(self.src0_shape[1:])

    @property
    def src1_elements(self) -> int:
        return math.prod(self.ne)


def _pointwise_facts(case: ImportedCase) -> PointwiseCaseFacts:
    ne = _int_list(case, "ne")
    nr = _int_list(case, "nr")
    nf = _int_value(case, "nf", default=0)
    perm1 = _pointwise_permutation(case)
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
    if not _is_base_pointwise_permutation(facts.perm1):
        raise ValueError("pointwise contiguous layout requires perm1=[0, 1, 2, 3]")
    if not _all_equal(facts.nr, 1):
        raise ValueError("pointwise contiguous layout requires same-shape inputs")
    return {"ncols": facts.ncols, "nrows": facts.nrows}


def _pointwise_rhs_row_broadcast_shape(facts: PointwiseCaseFacts) -> dict[str, int]:
    if facts.nf != 1:
        raise ValueError("pointwise rhs row broadcast requires nf=1")
    if not _is_base_pointwise_permutation(facts.perm1):
        raise ValueError("pointwise rhs row broadcast requires perm1=[0, 1, 2, 3]")
    if facts.src1_nrows != 1:
        raise ValueError("pointwise rhs row broadcast requires a single rhs row")
    if facts.src1_ncols != facts.ncols:
        raise ValueError("pointwise rhs row broadcast requires rhs ncols to match dst")
    return {
        "ncols": facts.ncols,
        "nrows": facts.nrows,
        "src1_row_stride": 0,
        "src1_ncols": facts.ncols,
    }


def _pointwise_scalar_broadcast_shape(facts: PointwiseCaseFacts) -> dict[str, int]:
    if facts.nf != 1:
        raise ValueError("pointwise scalar broadcast requires nf=1")
    if facts.src1_elements != 1:
        raise ValueError("pointwise scalar broadcast requires a scalar rhs source")
    return {
        "ncols": facts.ncols,
        "nrows": facts.nrows,
        "src1_row_stride": 0,
        "src1_ncols": 1,
    }

def _pointwise_rhs_column_broadcast_shape(facts: PointwiseCaseFacts) -> dict[str, int]:
    if facts.nf != 1:
        raise ValueError("pointwise rhs column broadcast requires nf=1")
    if not _is_base_pointwise_permutation(facts.perm1):
        raise ValueError("pointwise rhs column broadcast requires perm1=[0, 1, 2, 3]")
    if facts.ne[0] != 1 or facts.ne[1] != 1:
        raise ValueError(
            "pointwise rhs column broadcast requires singleton leading rhs dims"
        )
    if facts.nr[0] * facts.nr[1] <= 1:
        raise ValueError(
            "pointwise rhs column broadcast requires repetition across leading dst dims"
        )
    if not _all_equal(facts.nr[2:], 1):
        raise ValueError(
            "pointwise rhs column broadcast requires repetition only across leading dst dims"
        )
    return {
        "ncols": facts.nr[0] * facts.nr[1],
        "nrows": facts.ne[2] * facts.ne[3],
        "src1_row_stride": 1,
        "src1_ncols": 1,
    }


def _pointwise_rhs_intra_row_repeat_shape(
    facts: PointwiseCaseFacts,
) -> dict[str, int]:
    if facts.nf != 1:
        raise ValueError("pointwise rhs intra-row repeat requires nf=1")
    if not _is_base_pointwise_permutation(facts.perm1):
        raise ValueError("pointwise rhs intra-row repeat requires perm1=[0, 1, 2, 3]")
    if facts.ne[0] <= 1:
        raise ValueError(
            "pointwise rhs intra-row repeat requires more than one rhs column"
        )
    if facts.nr[0] <= 1:
        raise ValueError(
            "pointwise rhs intra-row repeat requires repetition across dst columns"
        )
    if not _all_equal(facts.nr[1:], 1):
        raise ValueError(
            "pointwise rhs intra-row repeat requires repetition only across dst columns"
        )
    return {
        "ncols": facts.ne[0] * facts.nr[0],
        "nrows": facts.src1_nrows,
        "src1_row_stride": facts.ne[0],
        "src1_ncols": facts.ne[0],
    }


POINTWISE_LAYOUT_LOWERERS: dict[
    str, tuple[Callable[[PointwiseCaseFacts], dict[str, int]], ...]
] = {
    "contiguous": (_pointwise_contiguous_shape,),
    "contiguous_or_rhs_row_broadcast": (
        _pointwise_contiguous_shape,
        _pointwise_rhs_row_broadcast_shape,
        _pointwise_scalar_broadcast_shape,
        _pointwise_rhs_column_broadcast_shape,
    ),
    "contiguous_src0_rhs_row_broadcast": (_pointwise_rhs_row_broadcast_shape,),
    "contiguous_or_row_strided_rhs": (
        _pointwise_contiguous_shape,
        _pointwise_rhs_row_broadcast_shape,
        _pointwise_scalar_broadcast_shape,
        _pointwise_rhs_column_broadcast_shape,
        _pointwise_rhs_intra_row_repeat_shape,
    ),
    "row_strided_sources_contiguous_dst": (
        _pointwise_contiguous_shape,
        _pointwise_rhs_row_broadcast_shape,
        _pointwise_scalar_broadcast_shape,
        _pointwise_rhs_column_broadcast_shape,
    ),
    "contiguous_src0_rhs_column_broadcast": (_pointwise_rhs_column_broadcast_shape,),
}


def _lower_pointwise_case(case: ImportedCase, route: Mapping[str, Any]) -> dict[str, int]:
    facts = _pointwise_facts(case)
    layout = _route_layout(route)
    lowers = POINTWISE_LAYOUT_LOWERERS.get(layout)
    if lowers is None:
        raise ValueError(f"no pointwise lowering is implemented for layout {layout!r}")
    errors: list[str] = []
    for lower in lowers:
        try:
            return lower(facts)
        except ValueError as exc:
            errors.append(str(exc))
    raise ValueError(errors[0] if errors else f"no pointwise lowering is implemented for layout {layout!r}")


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
