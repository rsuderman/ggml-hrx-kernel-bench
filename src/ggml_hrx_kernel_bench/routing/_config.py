from __future__ import annotations

from pathlib import Path

from .models import RoutingContext


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROUTING_VERSION = "v1"
DEFAULT_KERNEL_DIR = PROJECT_ROOT / "kernels" / "hrx2"
DEFAULT_ROUTING_DIRS: dict[str, Path] = {
    "v1": PROJECT_ROOT / "catalog" / "hrx2",
    "v2": PROJECT_ROOT / "catalog" / "v2",
}


def supported_routing_versions() -> tuple[str, ...]:
    return tuple(DEFAULT_ROUTING_DIRS)


def resolve_routing_dir(version: str, routing_dir: Path | None) -> Path:
    if routing_dir is not None:
        return routing_dir
    try:
        return DEFAULT_ROUTING_DIRS[version]
    except KeyError as exc:
        raise ValueError(f"unsupported routing version: {version}") from exc


def build_routing_context(
    *,
    version: str,
    kernel_dir: Path | None,
    routing_dir: Path | None,
    observed_shapes,
) -> RoutingContext:
    return RoutingContext(
        kernel_dir=kernel_dir or DEFAULT_KERNEL_DIR,
        routing_dir=resolve_routing_dir(version, routing_dir),
        observed_shapes=observed_shapes,
    )
