from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


REQUIRED_TOOL_NAMES = (
    "loom-link",
    "loom-compile",
    "iree-run-loom",
    "iree-test-loom",
    "iree-benchmark-loom",
)
TOOL_DIR_ENV_VAR = "GGML_HRX_TOOL_DIR"
IREE_RUN_LOOM_EXPECTED_BUFFER_TOLERANCE_FLAG = "--expected-kernel-buffer-tolerance"


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


def iree_run_loom_supports_expected_buffer_tolerance(
    tool_path: str | Path,
    *,
    timeout: float = 10.0,
) -> bool:
    result = subprocess.run(
        [str(tool_path), "--help"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    return IREE_RUN_LOOM_EXPECTED_BUFFER_TOLERANCE_FLAG in result.stdout


def require_iree_run_loom_expected_buffer_tolerance(
    *,
    tool_dir: str | None = None,
    tool_path: str | Path | None = None,
) -> str:
    path = str(tool_path) if tool_path is not None else require_tool("iree-run-loom", tool_dir=tool_dir)
    try:
        supports_flag = iree_run_loom_supports_expected_buffer_tolerance(path)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"failed to query iree-run-loom capabilities at {path}: {exc}") from exc
    if not supports_flag:
        raise RuntimeError(
            "iree-run-loom does not support "
            f"{IREE_RUN_LOOM_EXPECTED_BUFFER_TOLERANCE_FLAG}: {path}"
        )
    return path
