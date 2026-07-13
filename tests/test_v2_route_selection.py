from __future__ import annotations

import pytest

import ggml_hrx_kernel_bench.routing.v2.selection as selection_module
from ggml_hrx_kernel_bench.routing.v2.models import (
    ConcreteTensor,
    ConcreteTensorDimension,
    RouteConstraints,
    V2Route,
)
from ggml_hrx_kernel_bench.routing.v2.query import RouteCatalog
from ggml_hrx_kernel_bench.routing.v2.selection import (
    RouteMatch,
    RouteMatchQuery,
    create_route_selector,
)


class _StubNativeModule:
    def __init__(
        self,
        route_ids: tuple[str, ...],
        result: tuple[str, str | None] = ("no_match", None),
    ) -> None:
        self._route_ids = route_ids
        self._result = result
        self.select_calls: list[tuple[str, dict[str, object]]] = []

    def supported_route_ids(self, op: str) -> tuple[str, ...]:
        return self._route_ids if op == "ABS" else ()

    def select(self, op: str, payload: dict[str, object]) -> tuple[str, str | None]:
        self.select_calls.append((op, payload))
        return self._result


def _route() -> V2Route:
    return V2Route(
        id="abs_test",
        family="abs_test",
        op="ABS",
        source_id="abs_test",
        kernel_path="abs_test.loom",
        root_symbol="abs_test",
        export_name=None,
        tensors={},
        values=(),
        constraints=RouteConstraints(),
        launch={},
        bindings=(),
    )


def _catalog() -> RouteCatalog:
    route = _route()
    return RouteCatalog(
        routes=(route,),
        routes_by_op={"ABS": (route,)},
        routes_by_family={route.family: (route,)},
        routes_by_id={route.id: route},
    )


def _query() -> RouteMatchQuery:
    return RouteMatchQuery(tensors={})


def _install_native_module(
    monkeypatch: pytest.MonkeyPatch,
    module: _StubNativeModule,
) -> None:
    monkeypatch.setattr(selection_module, "_native_selector", module)


def test_native_mode_raises_for_unsupported_op(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _StubNativeModule(())
    _install_native_module(monkeypatch, module)
    selector = create_route_selector(_catalog(), mode="native")

    with pytest.raises(RuntimeError, match="native selector does not support op 'FLASH_ATTN_EXT'"):
        selector.select("FLASH_ATTN_EXT", _query())

    assert module.select_calls == []


def test_native_mode_raises_for_stale_route_table(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _StubNativeModule(("native_abs",))
    _install_native_module(monkeypatch, module)
    selector = create_route_selector(_catalog(), mode="native")

    with pytest.raises(RuntimeError) as exc_info:
        selector.select("ABS", _query())

    assert str(exc_info.value) == (
        "native route table for 'ABS' is stale: "
        "native=('native_abs',), active=('abs_test',)"
    )
    assert module.select_calls == []


def test_native_mode_raises_for_query_outside_int64(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _StubNativeModule(("abs_test",))
    _install_native_module(monkeypatch, module)
    selector = create_route_selector(_catalog(), mode="native")
    query = RouteMatchQuery(
        tensors={
            "src0": ConcreteTensor(
                dtype="F32",
                dimensions=(
                    ConcreteTensorDimension(name="d0", size=1 << 63, stride=1),
                ),
            )
        }
    )

    with pytest.raises(
        RuntimeError,
        match="tensor 'src0' contains a value outside int64",
    ):
        selector.select("ABS", query)

    assert module.select_calls == []


@pytest.mark.parametrize(
    ("native_result", "expected"),
    [
        (("match", "abs_test"), RouteMatch(route_id="abs_test")),
        (("no_match", None), None),
    ],
)
def test_native_mode_returns_native_match_or_no_match(
    monkeypatch: pytest.MonkeyPatch,
    native_result: tuple[str, str | None],
    expected: RouteMatch | None,
) -> None:
    module = _StubNativeModule(("abs_test",), native_result)
    _install_native_module(monkeypatch, module)
    selector = create_route_selector(_catalog(), mode="native")

    assert selector.select("ABS", _query()) == expected
    assert len(module.select_calls) == 1


def test_native_mode_raises_when_native_cannot_evaluate_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _StubNativeModule(("abs_test",), ("unsupported", None))
    _install_native_module(monkeypatch, module)
    selector = create_route_selector(_catalog(), mode="native")

    with pytest.raises(
        RuntimeError,
        match="native selector cannot evaluate query for op 'ABS'",
    ):
        selector.select("ABS", _query())


def test_shadow_mode_still_falls_back_when_native_is_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _StubNativeModule(())
    _install_native_module(monkeypatch, module)
    selector = create_route_selector(_catalog(), mode="shadow")

    assert selector.select("ABS", _query()) == RouteMatch(route_id="abs_test")
    assert module.select_calls == []
