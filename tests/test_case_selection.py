from __future__ import annotations

import pytest

from ggml_hrx_kernel_bench.routing.case_selection import (
    list_indexed_cases,
    select_case,
    select_cases,
    select_indexed_case,
    select_indexed_cases,
)


def _config() -> dict[str, object]:
    return {
        "kernel": "example",
        "params": ["d0"],
        "cases": [[4], [4], [8]],
    }


def test_indexed_case_selection_distinguishes_duplicate_case_ids() -> None:
    config = _config()

    assert list_indexed_cases(config) == [
        (0, "d04", [4]),
        (1, "d04", [4]),
        (2, "d08", [8]),
    ]
    assert select_indexed_case(config, 1) == (1, "d04", [4])
    assert select_case(config, 2) == ("d08", [8])
    assert select_indexed_cases(config, [1, 0, 1]) == [
        (1, "d04", [4]),
        (0, "d04", [4]),
        (1, "d04", [4]),
    ]
    assert select_cases(config, [1, 0]) == [("d04", [4]), ("d04", [4])]


@pytest.mark.parametrize("index", ["d04", "1", 1.0, True, None])
def test_case_selection_rejects_non_integer_indices(index: object) -> None:
    with pytest.raises(RuntimeError, match="case index must be an integer"):
        select_indexed_case(_config(), index)  # type: ignore[arg-type]


@pytest.mark.parametrize("index", [-1, 3])
def test_case_selection_rejects_out_of_range_index(index: int) -> None:
    with pytest.raises(RuntimeError, match=rf"case index out of range: {index}"):
        select_indexed_case(_config(), index)
