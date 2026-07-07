from __future__ import annotations

from pathlib import Path

from ..materialized_assets import (
    ASSET_DIR_NAMES,
    SUPPORTED_VERSIONS,
    default_kernel_dir,
    default_routing_dir,
)
from .models import RoutingContext


DEFAULT_ROUTING_VERSION = "v1"


def supported_routing_versions() -> tuple[str, ...]:
    return SUPPORTED_VERSIONS


def _derive_peer_dir(kind: str, version: str, peer_dir: Path | None) -> Path | None:
    if peer_dir is None:
        return None
    asset_dir_name = ASSET_DIR_NAMES[version]
    resolved_peer = peer_dir.resolve()
    if resolved_peer.name != asset_dir_name:
        return None
    if kind == "routing" and resolved_peer.parent.name == "kernels":
        return resolved_peer.parent.parent / "catalog" / asset_dir_name
    if kind == "kernel" and resolved_peer.parent.name == "catalog":
        return resolved_peer.parent.parent / "kernels" / asset_dir_name
    return None


def resolve_routing_dir(
    version: str, routing_dir: Path | None, *, kernel_dir: Path | None = None
) -> Path:
    if routing_dir is not None:
        return routing_dir
    derived_routing_dir = _derive_peer_dir("routing", version, kernel_dir)
    if derived_routing_dir is not None:
        return derived_routing_dir
    try:
        return default_routing_dir(version)
    except KeyError as exc:
        raise ValueError(f"unsupported routing version: {version}") from exc


def resolve_kernel_dir(
    version: str, kernel_dir: Path | None, *, routing_dir: Path | None = None
) -> Path:
    if kernel_dir is not None:
        return kernel_dir
    derived_kernel_dir = _derive_peer_dir("kernel", version, routing_dir)
    if derived_kernel_dir is not None:
        return derived_kernel_dir
    try:
        return default_kernel_dir(version)
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
        kernel_dir=resolve_kernel_dir(version, kernel_dir, routing_dir=routing_dir),
        routing_dir=resolve_routing_dir(version, routing_dir, kernel_dir=kernel_dir),
        observed_shapes=observed_shapes,
    )
