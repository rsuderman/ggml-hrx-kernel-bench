from __future__ import annotations

import os
import shutil
from pathlib import Path


REQUIRED_TOOL_NAMES = (
    "loom-link",
    "loom-compile",
    "iree-test-loom",
    "iree-benchmark-loom",
)
TOOL_DIR_ENV_VAR = "GGML_HRX_TOOL_DIR"


def configured_tool_dir(tool_dir: str | None = None) -> str | None:
    if tool_dir:
        return tool_dir
    env_tool_dir = os.environ.get(TOOL_DIR_ENV_VAR)
    return env_tool_dir or None


def configured_tool_dirs(tool_dir: str | None = None) -> tuple[Path, ...]:
    configured = configured_tool_dir(tool_dir)
    if not configured:
        return ()
    return tuple(
        Path(entry)
        for entry in configured.split(os.pathsep)
        if entry
    )


def resolve_tool(tool_name: str, *, tool_dir: str | None = None) -> str | None:
    effective_tool_dirs = configured_tool_dirs(tool_dir)
    if effective_tool_dirs:
        for directory in effective_tool_dirs:
            candidate = directory / tool_name
            if candidate.is_file():
                return str(candidate.resolve())
        return None
    return shutil.which(tool_name)


def require_tool(tool_name: str, *, tool_dir: str | None = None) -> str:
    path = resolve_tool(tool_name, tool_dir=tool_dir)
    effective_tool_dir = configured_tool_dir(tool_dir)
    if path is None:
        location = effective_tool_dir if effective_tool_dir else "PATH"
        raise RuntimeError(f"required tool is not available in {location}: {tool_name}")
    return path
