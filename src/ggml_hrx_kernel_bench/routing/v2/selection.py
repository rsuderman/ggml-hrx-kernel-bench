from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Protocol, Sequence

import _ggml_hrx_v2_selector_native as _native_selector

from .matching import route_accepts_tensors
from .models import ConcreteTensor, V2Route
from .query import RouteCatalog, routes_for_op


class _NativeSelectionStatus(str, Enum):
    MATCH = "match"
    NO_MATCH = "no_match"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class RouteMatchQuery:
    tensors: Mapping[str, ConcreteTensor]
    allowed_route_ids: tuple[str, ...] | None = None


@dataclass(frozen=True)
class RouteMatch:
    """A validated route returned by ``RouteSelector.select``."""

    route_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.route_id, str) or not self.route_id:
            raise ValueError("route_id must be a non-empty string")


@dataclass(frozen=True)
class _NativeSelection:
    status: _NativeSelectionStatus
    match: RouteMatch | None = None
    detail: str | None = None


class RouteSelector(Protocol):
    """Select one configured route for a normalized v2 query."""

    def select(
        self,
        op: str,
        query: RouteMatchQuery,
    ) -> RouteMatch | None:
        """Return a configured route that accepts the query, or no match.

        Backends may raise ``RuntimeError`` when they cannot evaluate the query.
        """
        ...


class PythonRouteSelector:
    """Selects routes with the descriptor interpreter in matching.py."""

    def __init__(self, routes: RouteCatalog | Sequence[V2Route]):
        self._catalog = routes if isinstance(routes, RouteCatalog) else None
        self._routes = None if isinstance(routes, RouteCatalog) else tuple(routes)

    def _routes_for_op(self, op: str) -> tuple[V2Route, ...]:
        normalized_op = str(op).strip().upper()
        if self._catalog is not None:
            return routes_for_op(self._catalog, normalized_op)
        assert self._routes is not None
        return tuple(route for route in self._routes if route.op == normalized_op)

    def select(
        self,
        op: str,
        query: RouteMatchQuery,
    ) -> RouteMatch | None:
        for route in self._routes_for_op(op):
            if query.allowed_route_ids is not None and route.id not in query.allowed_route_ids:
                continue
            # Ensure cardinality and semantics of each tensor match.
            # {src0, dst} must not match {src0, src1, dst} or {otherThing, dst}
            if set(route.tensors.keys()) != set(query.tensors.keys()):
                continue
            if route_accepts_tensors(route, query.tensors):
                return RouteMatch(route_id=route.id)
        return None


_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1


class _NativeRouteSelector:
    """Thin adapter for the native v2 selector extension."""

    def __init__(self, catalog: RouteCatalog):
        self._catalog = catalog

    @staticmethod
    def _int64(value: int) -> int | None:
        normalized = int(value)
        if normalized < _INT64_MIN or normalized > _INT64_MAX:
            return None
        return normalized

    def _marshal_query(
        self,
        query: RouteMatchQuery,
    ) -> tuple[dict[str, object] | None, str | None]:
        tensors: dict[str, object] = {}
        for role, tensor in query.tensors.items():
            dimensions: list[int] = []
            strides: list[int] = []
            for dimension in tensor.dimensions:
                size = self._int64(dimension.size)
                stride = self._int64(dimension.stride)
                if size is None or stride is None:
                    return (
                        None,
                        f"tensor {role!r} contains a value outside int64",
                    )
                dimensions.append(size)
                strides.append(stride)
            tensors[str(role)] = {
                "dtype": str(tensor.dtype).strip().upper(),
                "dimensions": dimensions,
                "strides": strides,
            }
        return (
            {
                "tensors": tensors,
                "allowed_route_ids": (
                    None
                    if query.allowed_route_ids is None
                    else [str(route_id) for route_id in query.allowed_route_ids]
                ),
            },
            None,
        )

    def select(
        self,
        op: str,
        query: RouteMatchQuery,
    ) -> _NativeSelection:
        normalized_op = str(op).strip().upper()
        supported_route_ids = tuple(
            str(route_id) for route_id in _native_selector.supported_route_ids(normalized_op)
        )
        if not supported_route_ids:
            return _NativeSelection(
                status=_NativeSelectionStatus.UNSUPPORTED,
                detail=f"native selector does not support op {normalized_op!r}",
            )
        active_route_ids = tuple(route.id for route in routes_for_op(self._catalog, normalized_op))
        if active_route_ids != supported_route_ids:
            return _NativeSelection(
                status=_NativeSelectionStatus.UNSUPPORTED,
                detail=(
                    f"native route table for {normalized_op!r} is stale: "
                    f"native={supported_route_ids!r}, active={active_route_ids!r}"
                ),
            )

        payload, detail = self._marshal_query(query)
        if payload is None:
            return _NativeSelection(
                status=_NativeSelectionStatus.UNSUPPORTED,
                detail=detail,
            )
        raw_result = _native_selector.select(normalized_op, payload)
        if not isinstance(raw_result, (tuple, list)) or len(raw_result) != 2:
            raise RuntimeError(f"native selector returned an invalid result: {raw_result!r}")
        raw_status, raw_route_id = raw_result
        try:
            status = _NativeSelectionStatus(str(raw_status).strip().lower())
        except ValueError as exc:
            raise RuntimeError(
                f"native selector returned an invalid status: {raw_status!r}"
            ) from exc
        if status == _NativeSelectionStatus.MATCH:
            if raw_route_id is None:
                raise RuntimeError(f"native selector returned an incomplete match: {raw_result!r}")
            if not isinstance(raw_route_id, str):
                raise RuntimeError(f"native selector returned an invalid match: {raw_result!r}")
            if raw_route_id not in supported_route_ids:
                raise RuntimeError(f"native selector returned an invalid match: {raw_result!r}")
            return _NativeSelection(
                status=status,
                match=RouteMatch(route_id=raw_route_id),
            )
        if raw_route_id is not None:
            raise RuntimeError(f"native selector returned invalid non-match data: {raw_result!r}")
        return _NativeSelection(
            status=status,
            detail=(
                f"native selector cannot evaluate query for op {normalized_op!r}"
                if status == _NativeSelectionStatus.UNSUPPORTED
                else None
            ),
        )


class _ShadowRouteSelector:
    def __init__(self, python: RouteSelector, native: _NativeRouteSelector):
        self._python = python
        self._native = native

    def select(
        self,
        op: str,
        query: RouteMatchQuery,
    ) -> RouteMatch | None:
        python_result = self._python.select(op, query)
        native_result = self._native.select(op, query)
        if native_result.status == _NativeSelectionStatus.UNSUPPORTED:
            return python_result
        if python_result != native_result.match:
            raise RuntimeError(
                "v2 route selector disagreement for "
                f"op={str(op).strip().upper()!r}, query={query!r}: "
                f"python={python_result!r}, native={native_result!r}"
            )
        return python_result


class _NativeOnlyRouteSelector:
    def __init__(self, native: _NativeRouteSelector):
        self._native = native

    def select(
        self,
        op: str,
        query: RouteMatchQuery,
    ) -> RouteMatch | None:
        result = self._native.select(op, query)
        if result.status == _NativeSelectionStatus.UNSUPPORTED:
            detail = result.detail or "native selector cannot evaluate the query"
            raise RuntimeError(detail)
        return result.match


def create_route_selector(
    catalog: RouteCatalog,
    *,
    mode: str | None = None,
) -> RouteSelector:
    selected_mode = mode if mode is not None else os.environ.get("GGML_HRX_V2_SELECTOR", "python")
    selected_mode = str(selected_mode).strip().lower()
    if selected_mode == "python":
        return PythonRouteSelector(catalog)
    native = _NativeRouteSelector(catalog)
    if selected_mode == "shadow":
        return _ShadowRouteSelector(PythonRouteSelector(catalog), native)
    if selected_mode == "native":
        return _NativeOnlyRouteSelector(native)
    raise ValueError(
        f"unsupported v2 selector mode {selected_mode!r}; expected 'python', 'shadow', or 'native'"
    )


__all__ = [
    "PythonRouteSelector",
    "RouteMatch",
    "RouteMatchQuery",
    "RouteSelector",
    "create_route_selector",
]
