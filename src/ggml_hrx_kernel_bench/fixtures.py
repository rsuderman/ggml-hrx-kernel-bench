from __future__ import annotations

from typing import Any


def require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required for fixture generation; install numpy in the venv") from exc
    return np
