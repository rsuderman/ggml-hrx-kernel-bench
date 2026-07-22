from __future__ import annotations

import json

from .routing.v2.selection import RouteQuery


def parse_route_query_json(text: str) -> RouteQuery:
    """Parse one RouteQuery from JSON text."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("malformed JSON") from exc
    return RouteQuery.from_json(payload)
