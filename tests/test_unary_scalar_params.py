from __future__ import annotations

import numpy as np
import pytest

from ggml_hrx_kernel_bench.materialized_assets import materialize_asset_root
from ggml_hrx_kernel_bench.oracles import _unary_apply
from ggml_hrx_kernel_bench.route_query_config import _execution_abi_for_route
from ggml_hrx_kernel_bench.routing.v2.query import load_route_catalog


@pytest.fixture(scope="module")
def catalog(tmp_path_factory: pytest.TempPathFactory):
    asset_root = materialize_asset_root(tmp_path_factory.mktemp("unary-params") / "assets", force=True)
    return load_route_catalog(asset_root / "catalog" / "v2")


def _scalar_entries(abi: dict) -> list[dict]:
    return [entry for entry in abi["entries"] if entry["kind"] == "scalar"]


@pytest.mark.parametrize(
    ("route_id", "attribute", "supplied", "default"),
    (
        ("leaky_relu_f32_contiguous_4d", "negative_slope", 0.2, 0.1),
        ("leaky_relu_f16_contiguous_4d", "negative_slope", 0.2, 0.1),
        ("softcap_f32_contiguous_4d", "softcap", 30.0, 50.0),
    ),
)
def test_execution_abi_threads_query_scalar_with_default(
    catalog, route_id: str, attribute: str, supplied: float, default: float
) -> None:
    route = catalog.routes_by_id[route_id]

    supplied_abi = _scalar_entries(_execution_abi_for_route(route, attributes={attribute: supplied}))
    assert supplied_abi == [
        {"position": 0, "role": attribute, "kind": "scalar", "dtype": "f32", "value": supplied}
    ]

    default_abi = _scalar_entries(_execution_abi_for_route(route, attributes={}))
    assert default_abi == [
        {"position": 0, "role": attribute, "kind": "scalar", "dtype": "f32", "value": default}
    ]


def test_unary_apply_honors_scalar_values() -> None:
    values = np.array([-2.0, 3.0], dtype=np.float32)

    leaky = _unary_apply(np, "leaky_relu_f32", values, {"negative_slope": 0.2})
    assert leaky.tolist() == pytest.approx([-0.4, 3.0])
    # Absent value falls back to the ggml default.
    leaky_default = _unary_apply(np, "leaky_relu_f32", values, {})
    assert leaky_default.tolist() == pytest.approx([-0.2, 3.0])

    softcap = _unary_apply(np, "softcap_f32", values, {"softcap": 30.0})
    expected = np.float32(30.0) * np.tanh(values / np.float32(30.0))
    assert softcap.tolist() == pytest.approx(expected.tolist())
