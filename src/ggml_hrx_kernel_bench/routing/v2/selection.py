from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

from .matching import route_accepts_tensors
from .models import ConcreteTensor, V2Route
from .query import RouteCatalog, routes_for_op


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


class RouteSelector(Protocol):
    """Total selector whose backend-specific unsupported cases fall back internally."""

    def select(
        self,
        op: str,
        query: RouteMatchQuery,
    ) -> RouteMatch | None:
        """Return a configured route that accepts the query, or no match."""
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


def create_route_selector(catalog: RouteCatalog) -> RouteSelector:
    return PythonRouteSelector(catalog)


__all__ = [
    "PythonRouteSelector",
    "RouteMatch",
    "RouteMatchQuery",
    "RouteSelector",
    "create_route_selector",
]
