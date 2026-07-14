from typing import Literal, TypeAlias

_SelectionResult: TypeAlias = (
    tuple[Literal["match"], str]
    | tuple[Literal["no_match", "unsupported"], None]
)


def select(op: str, query: dict[str, object]) -> _SelectionResult: ...


def supported_route_ids(op: str) -> list[str]: ...
