from __future__ import annotations

from pathlib import Path
import shutil


REQUIRED_TOOL_NAMES = (
    "loom-link",
    "loom-compile",
    "iree-benchmark-loom",
)


def resolve_tool(tool_name: str, *, tool_dir: str | None = None) -> str | None:
    if tool_dir:
        candidate = Path(tool_dir) / tool_name
        if candidate.is_file():
            return str(candidate.resolve())
    return shutil.which(tool_name)


def require_tool(tool_name: str, *, tool_dir: str | None = None) -> str:
    path = resolve_tool(tool_name, tool_dir=tool_dir)
    if path is None:
        location = f"{tool_dir} or PATH" if tool_dir else "PATH"
        raise RuntimeError(f"required tool is not available in {location}: {tool_name}")
    return path
