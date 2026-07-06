from __future__ import annotations

from pathlib import Path

from ..models import Candidate, CandidateQuery
from .matching import (
    default_shape_for_route,
    materialize_route_tensors,
    route_dispatch,
    route_values,
    shape_overrides_from_tensors,
    shape_from_tensors,
    value_from_route_source,
    value_from_tensor_source,
)
from .models import ConcreteTensor, V2Route, stable_id
from .query import RouteCatalog, candidate_routes
from .serialization import route_json, route_supports, source_path_for_route
from .shape import resolve_shape_source


def build_config(
    route: V2Route,
    shape: dict[str, int],
    tensors: dict[str, ConcreteTensor],
    resolved_values: dict[str, int | tuple[int, ...]],
) -> tuple[dict[str, str], dict[str, int | str], list[str]]:
    config: dict[str, str] = {}
    values: dict[str, int | str] = dict(shape)
    missing: list[str] = []
    for binding in route.bindings:
        key = str(binding["key"])
        if "source" in binding:
            source = str(binding["source"])
            value = value_from_route_source(source, resolved_values)
            if value is None:
                value = value_from_tensor_source(source, tensors)
            if value is None:
                value = resolve_shape_source(source, shape)
            if value is None:
                missing.append(source)
                continue
            config[key] = str(value)
            values[source] = value
        else:
            config[key] = str(binding["value"])
    return config, values, missing


def candidate_from_shape(
    *,
    kernel_dir: Path,
    route: V2Route,
    shape: dict[str, int],
    status: str = "planned",
    message: str | None = None,
) -> Candidate:
    tensors = materialize_route_tensors(route, shape)
    shape = {**shape_from_tensors(tensors), **shape_overrides_from_tensors(tensors), **shape}
    resolved_values = route_values(route, tensors)
    if resolved_values is None:
        raise RuntimeError(f"v2 route {route.id!r} failed to resolve route values for shape {shape!r}")
    config, values, missing = build_config(route, shape, tensors, resolved_values)
    if missing:
        status = "missing_config"
        message = "missing shape/config values: " + ", ".join(missing)
    return Candidate(
        id=f"{route.id}_{stable_id(route.id, shape, config, length=8)}",
        family=route.family,
        op=route.op,
        source_id=route.source_id,
        source_path=source_path_for_route(kernel_dir, route),
        root_symbol=route.root_symbol,
        export_name=route.export_name,
        route_id=route.id,
        route=route_json(route),
        shape=shape,
        values=values,
        config=config,
        dispatch=route_dispatch(route, shape, values=resolved_values),
        supports=route_supports(route),
        schedule=None,
        coverage="route_backed",
        status=status,
        message=message,
    )


def list_candidates(*, kernel_dir: Path, catalog: RouteCatalog, query: CandidateQuery) -> list[Candidate]:
    candidates: list[Candidate] = []
    for route in candidate_routes(catalog, query):
        source_path = source_path_for_route(kernel_dir, route)
        status = "planned" if source_path.exists() else "missing_source"
        message = None
        if status != "planned":
            message = f"kernel source is not available for source_id={route.source_id}"
        candidates.append(
            candidate_from_shape(
                kernel_dir=kernel_dir,
                route=route,
                shape=default_shape_for_route(route),
                status=status,
                message=message,
            )
        )
        if query.limit and len(candidates) >= query.limit:
            break
    return candidates
