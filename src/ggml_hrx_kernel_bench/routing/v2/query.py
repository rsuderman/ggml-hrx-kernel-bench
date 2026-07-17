from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from ..models import CandidateQuery
from .catalog import load_route_file, load_route_index
from .models import V2Route


@dataclass(frozen=True)
class RouteCatalog:
    routes: tuple[V2Route, ...]
    routes_by_op: Mapping[str, tuple[V2Route, ...]]
    routes_by_family: Mapping[str, tuple[V2Route, ...]]
    routes_by_id: Mapping[str, V2Route]

    def __post_init__(self) -> None:
        object.__setattr__(self, "routes_by_op", MappingProxyType(dict(self.routes_by_op)))
        object.__setattr__(self, "routes_by_family", MappingProxyType(dict(self.routes_by_family)))
        object.__setattr__(self, "routes_by_id", MappingProxyType(dict(self.routes_by_id)))


def require_route_catalog(
    *,
    routing_dir: Path | None = None,
    catalog: RouteCatalog | None = None,
) -> RouteCatalog:
    if catalog is not None:
        return catalog
    if routing_dir is None:
        raise ValueError("routing_dir or catalog is required")
    return load_route_catalog(routing_dir)


def load_route_catalog(routing_dir: Path) -> RouteCatalog:
    route_index = load_route_index(routing_dir)
    routes_by_op: dict[str, tuple[V2Route, ...]] = {}
    all_routes: list[V2Route] = []
    routes_by_family_lists: dict[str, list[V2Route]] = {}
    routes_by_id: dict[str, V2Route] = {}
    for op, route_files in route_index.items():
        op_routes = tuple(load_route_file(routing_dir, op=op, route_file_name=route_file_name) for route_file_name in route_files)
        routes_by_op[op] = op_routes
        all_routes.extend(op_routes)
        for route in op_routes:
            routes_by_family_lists.setdefault(route.family, []).append(route)
            if route.id in routes_by_id:
                raise RuntimeError(f"duplicate v2 route id {route.id!r} in {routing_dir}")
            routes_by_id[route.id] = route
    return RouteCatalog(
        routes=tuple(all_routes),
        routes_by_op=routes_by_op,
        routes_by_family={family: tuple(routes) for family, routes in routes_by_family_lists.items()},
        routes_by_id=routes_by_id,
    )


def routes_for_op(catalog: RouteCatalog, op: str) -> tuple[V2Route, ...]:
    return catalog.routes_by_op.get(str(op).strip().upper(), ())


def routes_for_family(catalog: RouteCatalog, family: str) -> tuple[V2Route, ...]:
    return catalog.routes_by_family.get(str(family), ())


def normalize_architecture(architecture: str | None) -> str | None:
    if architecture is None:
        return None
    normalized = str(architecture).strip().lower()
    return normalized or None


def route_supports_architecture(route: V2Route, architecture: str | None) -> bool:
    if not route.architectures:
        return True
    normalized = normalize_architecture(architecture)
    return normalized is not None and normalized in route.architectures


def routes_for_architecture(routes: tuple[V2Route, ...], architecture: str | None) -> tuple[V2Route, ...]:
    return tuple(route for route in routes if route_supports_architecture(route, architecture))


def select_route(
    catalog: RouteCatalog,
    *,
    family: str,
    route_id: str | None = None,
    architecture: str | None = None,
) -> V2Route:
    matches = list(routes_for_architecture(routes_for_family(catalog, family), architecture))
    if not matches:
        suffix = "" if normalize_architecture(architecture) is None else f" architecture={architecture}"
        raise RuntimeError(f"no v2 route found for family={family}{suffix}")
    if route_id is not None:
        route = catalog.routes_by_id.get(str(route_id))
        if route is None or route.family != family or not route_supports_architecture(route, architecture):
            suffix = "" if normalize_architecture(architecture) is None else f" architecture={architecture}"
            raise RuntimeError(f"no v2 route found for family={family} route_id={route_id}{suffix}")
        return route
    if len(matches) != 1:
        raise RuntimeError(f"minimal v2 config requires exactly one route for {family}, found {len(matches)}")
    return matches[0]


def candidate_routes(catalog: RouteCatalog, query: CandidateQuery) -> tuple[V2Route, ...]:
    routes = routes_for_architecture(catalog.routes, query.target)
    if not query.families:
        return routes
    filtered = [
        route
        for route in routes
        if route.family in query.families
        or route.source_id in query.families
        or route.id in query.families
    ]
    return tuple(filtered)
