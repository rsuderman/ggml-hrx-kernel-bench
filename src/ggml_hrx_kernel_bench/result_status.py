from __future__ import annotations

from typing import Any


def correctness_ok(value: Any) -> bool:
    if value in (None, "", "passed", "ok", True):
        return True
    if isinstance(value, dict):
        failed = value.get("failed_sample_count")
        if failed is not None:
            try:
                return int(failed) == 0
            except (TypeError, ValueError):
                return False
        state = value.get("state")
        if state is not None:
            return state in ("passed", "ok")
    return False
