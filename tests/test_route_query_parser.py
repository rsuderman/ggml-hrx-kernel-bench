from __future__ import annotations

import json
from io import StringIO

import pytest

from ggml_hrx_kernel_bench.route_query_parser import parse_route_query_json
from ggml_hrx_kernel_bench.routing.v2.selection import RouteQuery


def _query_payload() -> dict[str, object]:
    return {
        "op": "ABS",
        "tensors": {
            "src0": {
                "dtype": "F32",
                "dimensions": [5, 7],
                "strides": [1, 5],
            }
        },
        "attributes": {"enabled": True},
    }


def test_parse_route_query_json_parses_text() -> None:
    query = parse_route_query_json(json.dumps(_query_payload()))

    assert query.operation == "ABS"
    assert query.tensors["src0"].dtype == "F32"
    assert query.attributes == {"enabled": True}


def test_parse_route_query_json_rejects_malformed_json() -> None:
    with pytest.raises(ValueError, match="^malformed JSON$"):
        parse_route_query_json('{"op": "ABS"')


def test_parse_route_query_json_delegates_schema_validation() -> None:
    with pytest.raises(ValueError, match="missing required field 'tensors'"):
        parse_route_query_json('{"op":"ABS"}')


def test_parse_route_query_json_matches_existing_payload_construction() -> None:
    stream = StringIO(json.dumps(_query_payload()))
    text = stream.read()

    assert parse_route_query_json(text) == RouteQuery.from_json(json.loads(text))
